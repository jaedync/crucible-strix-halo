"""Run a temporary server (loopback by default). The caller must hold the shared GPU lock."""
import argparse
import ipaddress
import os
import socket
import socketserver
from http.server import ThreadingHTTPServer as StdlibServer
from pathlib import Path
import signal
import sys
from . import metrics, update_check
from .api import MAX_COUNT, MAX_MTP_WINDOW, MTP_WINDOW, MAX_ALL, ApiError, ServerDefaults
from .backend import EngineBackend, LOOKUP_DEPTH, MAX_CONVERSATION_CACHE_MIB
from .batch_scheduler import INTERLEAVE_MIN_ROWS
from .hosts import ALLOWED_HOSTS_ENV, HostError, resolve_allowed_hosts
from .http import Service, handler_for
from .think_budget import MIN_THINKING_DEFAULT
from .tokenization import ModelTokenizer
from .version import server_commit, server_version

# serve --draft-blocks: the engine's file limit (draft_policy.h kDraftBlocksFileMax).
DRAFT_BLOCKS_FILE_MAX = 64 * 1024


class ThreadingHTTPServer(StdlibServer):
    """The stdlib server without socket.getfqdn(): HTTPServer.server_bind asks the resolver for a name that nothing
    here uses, and a slow or absent DNS would delay the bind. Tests patch this name, so main() keeps using it."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]


class Ipv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self):
        # Linux defaults to 0 but other hosts differ: set it, so "::" accepts IPv4 clients on every host.
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def listen_address(text):
    """--host: an IPv4 or IPv6 literal. A host name is refused, so the address never comes from a lookup."""
    try:
        if "%" in text:
            raise ValueError("scope id")
        return ipaddress.ip_address(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an IPv4 or IPv6 address literal (host names and scope ids are not accepted)") from None


def bind_server(host, port, handler):
    """The HTTP server bound to host:port. The IPv6 class sets address_family, which the IPv4 class does not."""
    address = ipaddress.ip_address(host)
    if address.version == 6:
        return Ipv6Server((str(address), port), handler)
    return ThreadingHTTPServer((str(address), port), handler)


def listen_url(host, port):
    return f"http://{f'[{host}]' if host.version == 6 else host}:{port}/v1"


def main():
    parser = argparse.ArgumentParser(description="Temporary local Crucible server (default 127.0.0.1:8790)")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--version", action="version", version=f"crucible {server_version()}")
    parser.add_argument("--no-update-check", action="store_true",
                        help="Do not ask GitHub for a newer release at start. CRUCIBLE_NO_UPDATE_CHECK=1 does the same. "
                             "The check is a notice only: it never updates anything.")
    parser.add_argument("--engine", type=Path, default=Path(__file__).resolve().parents[1] / "build/engine/crucible")
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--model-id", default="crucible")
    parser.add_argument("--host", type=listen_address, default="127.0.0.1", metavar="ADDRESS",
                        help="Bind address: an IPv4 or IPv6 literal (default 127.0.0.1). The server has no "
                             "authentication: an address that is not loopback (0.0.0.0, ::, a LAN address) lets every "
                             "machine that can reach it use the server. Use --allowed-host and a firewall then.")
    parser.add_argument("--port", type=int, default=8790, help="Port (default 8790).")
    parser.add_argument("--streams", type=int, choices=range(1, 9), default=None,
                        help="Batch slots (default 1). The batched stack serves every request, including speculation.")
    parser.add_argument("--serial", action="store_true",
                        help="Serve with the serial worker: one request at a time, in a FIFO queue. It accepts draft_head "
                             "full, draft_adapt 1 and draft_q 0. Not with --streams.")
    parser.add_argument("--conversation-cache-mib", type=int, default=0)
    parser.add_argument("--prefill", type=int, default=None)
    parser.add_argument("--host-reserve-mib", type=int, default=None, metavar="1024..131072",
                        help="Free host memory (MemAvailable) that each conversation cache save keeps (default 8192, "
                             "the engine's). On a unified-memory host the GPU uses the same memory.")
    parser.add_argument("--short-streams", type=int, default=0, help="Batch slots that hold only --short-ctx tokens.")
    parser.add_argument("--short-ctx", type=int, default=0)
    parser.add_argument("--packed-op", choices=("verify_step", "step"), default=None,
                        help="Batch decode with several sessions: packed verification (default verify_step) or the "
                             "G6c packed step.")
    # Default 10: 3 decoders at 32K keep a longest token gap of 0.66 s (3.0 s off) while
    # a new 32K prompt prefills; that prompt's TTFT grows about 11 percent.
    parser.add_argument("--prefill-interleave", type=int, default=None, metavar="LAYERS",
                        help="Batch mode (default 10): decoders run one pass after every LAYERS layers of another "
                             "request's prompt chunk (0 = off). Lower values cut the longest token gap and slow the prompt.")
    parser.add_argument("--interleave-min-rows", type=int, default=None, metavar="ROWS",
                        help="Batch mode: a prompt chunk with fewer rows runs without decode passes; it ends within the "
                             f"decode gap (default {INTERLEAVE_MIN_ROWS}; 0 = passes in every bulk chunk).")
    parser.add_argument("--session-replay", type=int, choices=(0, 1), default=None,
                        help="Session replay (default 1 with a conversation cache): a later turn resumes from the "
                             "previous turn's decode-time state when its prompt repeats the generated tokens. "
                             "Use 0 for the fresh-request path (exact reproduction of a request alone).")
    parser.add_argument("--slot-resume", type=int, choices=(0, 1), default=None,
                        help="Batch serve with session replay (default 1): a turn that repeats the state still on "
                             "its slot resumes there with no host restore. 0 always restores from the host cache; "
                             "outputs are the same.")
    parser.add_argument("--slot-checkpoints", type=int, choices=(0, 1), default=None,
                        help="Batch serve with session replay (default 1): keep resume points on each slot at a "
                             "turn's start, last chunk boundary and prompt end, so a turn that leaves the previous "
                             "output prefills only the rest. 0 turns them off.")
    parser.add_argument("--cache-probation", type=int, choices=(0, 1), default=None,
                        help="Batch serve with a conversation cache (default 1): a state that no later request used "
                             "(for example a one-time summary request) leaves the cache first and never takes the "
                             "place of a live session's state. Outputs are the same; 0 turns it off.")
    parser.add_argument("--cache-idle-minutes", type=int, default=None, metavar="1..10080",
                        help="Batch serve with a conversation cache (default 30): a session state that no request "
                             "used for this many minutes may leave the host cache (to --state-dir when set).")
    parser.add_argument("--state-dir", type=Path, default=None, metavar="DIR",
                        help="Batch serve with session replay: write conversation states that the host cache evicts, "
                             "and the states at a stop, to this directory inside your home directory (mode 0700), "
                             "so a later turn or a restarted server continues without a full prefill. Off by default. "
                             "DIR must be a folder of its own: a file of anything else in it stops the engine start. "
                             "Example: ~/.cache/crucible/states")
    parser.add_argument("--state-disk-gib", type=int, default=None, metavar="1..65536",
                        help="Largest size of the files in --state-dir (default 64 GiB).")
    parser.add_argument("--state-disk-write-gib-per-day", type=int, default=None, metavar="0..65536",
                        help="SSD wear guard: GiB that --state-dir may write in one local day, across restarts "
                             "(default 300). Above it, spills and stop flushes are refused; 0 only reads.")
    parser.add_argument("--state-disk-write-delay-ms", type=int, default=None, metavar="0..60000",
                        help="Tests only: each state write waits this long first (the same-second restore proof). "
                             "Its states go to their own folder. Default 0.")
    parser.add_argument("--allowed-host", action="append", default=[], metavar="HOST[:PORT]",
                        help=f"Strict mode: refuse (HTTP 403) every request whose Host header is not listed. Repeat the "
                             f"flag or use commas; HOST without a port matches any port; the match ignores case. It "
                             f"replaces {ALLOWED_HOSTS_ENV} (same format). Default: any Host is accepted, which a "
                             "reverse proxy needs; strict mode is the defence against DNS rebinding, so use it where a "
                             "browser can reach the server, for example --allowed-host 127.0.0.1,localhost,[::1].")
    parser.add_argument("--clamp-max-tokens", action="store_true",
                        help="Lower max_tokens to the context room instead of rejecting the request.")
    parser.add_argument("--temperature", type=float, default=None, metavar="0..2",
                        help="Server default temperature for requests that send none (default 0, greedy). A request "
                             "with logprobs and no temperature stays at 0. A request field always wins.")
    parser.add_argument("--top-k", type=int, default=None, metavar="0..248320",
                        help="Server default top_k for requests that send none (default 0, off).")
    parser.add_argument("--top-p", type=float, default=None, metavar="0..1",
                        help="Server default top_p for requests that send none (default 1, off).")
    parser.add_argument("--min-p", type=float, default=None, metavar="0..1",
                        help="Server default min_p for requests that send none (default 0, off).")
    parser.add_argument("--default-max-tokens", default="128", metavar=f"1..{MAX_COUNT}|max",
                        help="Output tokens for a request that sends no max_tokens or max_completion_tokens (default "
                             "128). max gives it all the context room after its prompt. A request count is not changed.")
    parser.add_argument("--max-body-mib", type=int, default=None, metavar="N",
                        help="Largest request body: 1..64 MiB (default 1), or 1..1024 MiB with --vision (default 256). "
                             "A 256K-token conversation of code needs more than 1 MiB.")
    parser.add_argument("--vision", action="store_true",
                        help="Load the vision tower and accept image_url parts (base64 data: URLs).")
    parser.add_argument("--vision-max-pixels", type=int, default=None,
                        help="Downscale larger images to this pixel count (default 2560*1440, as Halogen).")
    parser.add_argument("--vision-max-images", type=int, default=None, metavar="N",
                        help="Most images in one request (default and limit 256).")
    parser.add_argument("--vision-max-image-mib", type=int, default=None, metavar="N",
                        help="Largest encoded image in MiB (default 16).")
    parser.add_argument("--thinking-answer-room", default="auto", metavar="auto|N",
                        help="Tokens of max_tokens kept for the answer of a thinking request. auto is Halogen's "
                             "max(1024, 15%% of max_tokens); 0 lets thinking use all of max_tokens.")
    parser.add_argument("--min-thinking-tokens", type=int, default=MIN_THINKING_DEFAULT, metavar="N",
                        help="A thinking request with no explicit thinking budget is served with thinking off when "
                             "the answer room leaves fewer than N tokens before the close (default "
                             f"{MIN_THINKING_DEFAULT}; 0 = never). Below it the forced close sentence is a large part "
                             "of the thinking.")
    parser.add_argument("--max-thinking-tokens", type=int, default=None, metavar="N",
                        help="Server default thinking budget. A request's own budget wins.")
    parser.add_argument("--stream-tool-calls", choices=("on", "off"), default="on",
                        help="Stream tool-call ids, names and string arguments as they are written (default on).")
    parser.add_argument("--max-requests-per-client", type=int, default=0, choices=range(0, 65), metavar="0..64",
                        help="Open completion requests per client address (0, the default: no limit). "
                             "All loopback clients share one address.")
    parser.add_argument("--log-engine-errors", action="store_true",
                        help="Also log the raw text of engine errors that are not known engine messages. Off by default: "
                             "such a text could quote request content, so use it only to debug with your own data.")
    # RA-8: content is client data. A tool result that quotes a chat template must not forge a turn or a tool call.
    parser.add_argument("--content-special-tokens", choices=("text", "parse"), default="text",
                        help="Special-token strings such as <|im_end|> or <tool_call> in message content, tool results, "
                             "tool arguments and tool definitions: text (default) encodes them as plain text; parse "
                             "makes them control tokens, as before. Only the chat template writes control tokens with "
                             "text. Prompts with no such string get the same IDs in both modes.")
    parser.add_argument("--lookup", type=int, default=LOOKUP_DEPTH, choices=range(0, LOOKUP_DEPTH + 1),
                        metavar=f"0..{LOOKUP_DEPTH}",
                        help=f"Prompt lookup depth for speculative requests (default {LOOKUP_DEPTH}; 0 turns it off). "
                             "It copies earlier prompt or output tokens as drafts; outputs do not change.")
    parser.add_argument("--mtp-window", type=int, default=MTP_WINDOW, metavar=f"0..{MAX_MTP_WINDOW}",
                        help=f"MTP warm-up rows (mtp_prefix) for requests that set none (default {MTP_WINDOW}; 0 "
                             "warms the whole prompt). It changes only the drafts at an equal restore point.")
    parser.add_argument("--draft-capture", type=Path, default=None, metavar="DIR",
                        help="Batch mode, default off: write draft vocabulary data of finished turns to DIR "
                             "(absolute, made 0700). Never text, request IDs or times finer than the day.")
    parser.add_argument("--draft-capture-mode", choices=("counts", "tokens"), default="counts",
                        help="counts (default): block counts of generated tokens per day and turn class, numbers "
                             "only; it never writes a token ID. tokens: a token ID spool for the machine-only "
                             "training pipeline; only this exact value turns it on, and it needs the user's "
                             "explicit approval.")
    parser.add_argument("--draft-capture-max-gib", type=int, default=None, metavar="1..1024",
                        help="tokens mode: the spool size limit over all days (default 8).")
    parser.add_argument("--draft-blocks", type=Path, default=None, metavar="FILE",
                        help="Batch mode: the draft head's 128-token vocabulary blocks (sorted unique IDs, comma-"
                             "separated; default the built-in 64K head). It changes only the drafts, never an "
                             "output.")
    parser.add_argument("--draft-stats", action="store_true",
                        help="Batch mode, default off (needs --draft-capture in counts mode): also write "
                             "counts of draft tokens by position, turn class and rejected target token to "
                             "DIR/draft-stats-DAY.json. Counts of single token IDs only; the engine merges each "
                             "request before it reports.")
    parser.add_argument("--default-thinking", choices=("on", "off"), default="off",
                        help="Thinking for chat requests that set no thinking field. The checkpoint template, "
                             "other engines, and Halogen use on; off keeps older clients and benchmarks unchanged.")
    args = parser.parse_args()
    if args.serial and args.streams is not None:
        parser.error("--serial runs one request at a time; do not use it with --streams")
    if not args.serial and args.streams is None:
        args.streams = 1  # the batched stack is the default; --streams 1 equals the old serial output.
    # These options exist only in the batched stack: None means "not given", so that --serial can refuse them.
    batch_only = {"--slot-resume": args.slot_resume, "--slot-checkpoints": args.slot_checkpoints,
                  "--packed-op": args.packed_op, "--prefill-interleave": args.prefill_interleave,
                  "--interleave-min-rows": args.interleave_min_rows,
                  "--draft-capture": args.draft_capture, "--draft-blocks": args.draft_blocks}
    for flag, value in batch_only.items():
        if args.serial and value is not None:
            parser.error(f"{flag} needs the batched stack; do not use it with --serial")
    if args.packed_op is None: args.packed_op = "verify_step"
    if args.prefill_interleave is None: args.prefill_interleave = 10
    if args.interleave_min_rows is None: args.interleave_min_rows = INTERLEAVE_MIN_ROWS
    room = args.thinking_answer_room
    if room != "auto" and not (room.isdecimal() and int(room) <= 262144):
        parser.error("--thinking-answer-room must be auto or a token count in [0,262144]")
    try:
        sampling = {name: getattr(args, name) for name in ("temperature", "top_k", "top_p", "min_p")
                    if getattr(args, name) is not None}
        count = args.default_max_tokens
        if count != MAX_ALL and not count.isdecimal():
            parser.error(f"--default-max-tokens must be a count in [1,{MAX_COUNT}] or {MAX_ALL}")
        defaults = ServerDefaults.checked(**sampling, max_tokens=count if count == MAX_ALL else int(count))
    except ApiError as error:
        parser.error(f"{error} (server default)")
    if not 0 <= args.mtp_window <= MAX_MTP_WINDOW:
        parser.error(f"--mtp-window must be in [0,{MAX_MTP_WINDOW}]")
    if args.draft_capture is not None and not args.draft_capture.is_absolute():
        parser.error("--draft-capture must be an absolute directory path")
    if args.draft_stats and (args.draft_capture is None or args.draft_capture_mode != "counts" or args.serial):
        parser.error("--draft-stats needs --draft-capture in counts mode and the batch server")
    if args.draft_capture is None and (args.draft_capture_mode != "counts" or args.draft_capture_max_gib is not None):
        parser.error("--draft-capture-mode and --draft-capture-max-gib need --draft-capture")
    if args.draft_capture_max_gib is not None and (args.draft_capture_mode != "tokens" or
                                                   not 1 <= args.draft_capture_max_gib <= 1024):
        parser.error("--draft-capture-max-gib needs --draft-capture-mode tokens and a value in [1,1024]")
    if args.draft_blocks is not None:
        if not args.draft_blocks.is_absolute() or not args.draft_blocks.is_file():
            parser.error("--draft-blocks must be an absolute path of an existing file")
        # The engine reads at most 64 KiB (kDraftBlocksFileMax); the full 1940-block selection is about 10 KB.
        if not 0 < args.draft_blocks.stat().st_size <= DRAFT_BLOCKS_FILE_MAX:
            parser.error(f"--draft-blocks must be a file of 1 to {DRAFT_BLOCKS_FILE_MAX} bytes")
    if not 0 <= args.min_thinking_tokens <= 262144:
        parser.error("--min-thinking-tokens must be in [0,262144]")
    if args.max_thinking_tokens is not None and not 1 <= args.max_thinking_tokens <= 262144:
        parser.error("--max-thinking-tokens must be in [1,262144]")
    if not 1 <= args.ctx <= 262144:
        parser.error("--ctx must be in [1,262144]")
    if (not 0 <= args.conversation_cache_mib <= MAX_CONVERSATION_CACHE_MIB or
            (args.prefill is not None and not 0 <= args.prefill <= 4096)):
        parser.error(f"The cache must fit 0..{MAX_CONVERSATION_CACHE_MIB} MiB. Prefill rows must fit 0..4096.")
    if (args.short_streams or args.short_ctx) and (args.serial or not 0 < args.short_streams < args.streams or
                                              not 0 < args.short_ctx < args.ctx):
        parser.error("--short-streams needs 1..streams-1 slots (not with --serial) and --short-ctx below --ctx")
    if args.host_reserve_mib is not None and (not args.conversation_cache_mib or
                                              not 1024 <= args.host_reserve_mib <= MAX_CONVERSATION_CACHE_MIB):
        parser.error(f"--host-reserve-mib needs --conversation-cache-mib and a value in [1024,{MAX_CONVERSATION_CACHE_MIB}]")
    if args.session_replay and not args.conversation_cache_mib:
        parser.error("--session-replay needs --conversation-cache-mib")
    # Design decision (Oct 4): replay is the default whenever a conversation cache exists.
    if args.session_replay is None:
        args.session_replay = 1 if args.conversation_cache_mib else 0
    if args.state_dir is not None:
        if args.serial or not args.session_replay:
            parser.error("--state-dir needs session replay (a conversation cache) and not --serial")
        args.state_dir = args.state_dir.expanduser().absolute()
    if args.cache_idle_minutes is not None and (args.serial or not args.conversation_cache_mib or
                                                not 1 <= args.cache_idle_minutes <= 10080):
        parser.error("--cache-idle-minutes needs --conversation-cache-mib, a value in [1,10080], and not --serial")
    if args.cache_probation is not None and (args.serial or not args.conversation_cache_mib):
        parser.error("--cache-probation needs --conversation-cache-mib and not --serial")
    if args.state_disk_write_gib_per_day is not None and (args.state_dir is None or
                                                          not 0 <= args.state_disk_write_gib_per_day <= 65536):
        parser.error("--state-disk-write-gib-per-day needs --state-dir and a value in [0,65536]")
    if args.state_disk_write_delay_ms is not None and (args.state_dir is None or
                                                       not 0 <= args.state_disk_write_delay_ms <= 60000):
        parser.error("--state-disk-write-delay-ms needs --state-dir and a value in [0,60000]")
    if args.state_disk_gib is not None and (args.state_dir is None or not 1 <= args.state_disk_gib <= 65536):
        parser.error("--state-disk-gib needs --state-dir and a value in [1,65536]")
    if args.vision_max_pixels is not None and (not args.vision or not 65536 <= args.vision_max_pixels <= 4194304):
        parser.error("--vision-max-pixels needs --vision and a value in [65536,4194304]")
    for flag, value, high in (("--vision-max-images", args.vision_max_images, 256),
                              ("--vision-max-image-mib", args.vision_max_image_mib, 64)):
        if value is not None and (not args.vision or not 1 <= value <= high):
            parser.error(f"{flag} needs --vision and a value in [1,{high}]")
    vision = None
    if args.vision:
        from . import vision as vision_module
        from .vision import MAX_PIXELS, MAX_IMAGES, MAX_IMAGE_BYTES
        # The image tokens of one request must fit the context (review F2: no aggregate pixel cap otherwise).
        vision = vision_module.VisionSettings(
            max_image_tokens=args.ctx, max_pixels=args.vision_max_pixels or MAX_PIXELS,
            max_images=args.vision_max_images or MAX_IMAGES,
            max_image_bytes=args.vision_max_image_mib << 20 if args.vision_max_image_mib else MAX_IMAGE_BYTES)
    from .vision import VISION_BODY_MIB, VISION_BODY_MIB_MAX
    body_high = VISION_BODY_MIB_MAX if args.vision else 64
    if args.max_body_mib is not None and not 1 <= args.max_body_mib <= body_high:
        parser.error(f"--max-body-mib must be in [1,{body_high}]" + ("" if args.vision else " without --vision"))
    if args.max_body_mib is None:
        args.max_body_mib = VISION_BODY_MIB if args.vision else 1
    engine_options = ("--vision", "1") if args.vision else ()
    # Sent only when set, so the default engine command stays as before; the engine prints its reserve at startup.
    if args.host_reserve_mib is not None:
        engine_options += ("--host-reserve-mib", str(args.host_reserve_mib))
    if args.draft_blocks is not None:
        engine_options += ("--draft-blocks", str(args.draft_blocks))
    try:
        allowed_hosts = resolve_allowed_hosts(args.allowed_host, os.environ)
    except HostError as error:
        parser.error(str(error))
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be in [1024,65535]")
    if not args.model_id or len(args.model_id) > 200:
        parser.error("--model-id must contain 1 to 200 characters")
    metrics.LOG_ENGINE_TEXT = args.log_engine_errors
    def stop(_signal, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    backend = None
    server = None
    capture = None
    if args.draft_capture is not None:
        from .draft_capture import DraftCapture
        try:
            capture = DraftCapture(args.draft_capture, mode=args.draft_capture_mode,
                                   max_bytes=(args.draft_capture_max_gib or 8) << 30,
                                   draft_stats=args.draft_stats)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"draft capture: {error}", file=sys.stderr, flush=True)
            return 2
        print(f"draft capture: {capture.mode} mode" + (", per-token draft stats on" if args.draft_stats else ""),
              file=sys.stderr, flush=True)
    try:
        tokenizer = ModelTokenizer(args.checkpoint, content_special_tokens=args.content_special_tokens)
        if args.serial:
            backend = EngineBackend(args.engine, args.checkpoint, args.ctx,
                                    args.conversation_cache_mib, args.prefill,
                                    engine_options=(("--session-replay", "1") if args.session_replay else ()) +
                                                   engine_options, lookup=args.lookup)
        else:
            from .batch_backend import BatchedEngineBackend
            backend = BatchedEngineBackend(args.engine, args.checkpoint, args.ctx, streams=args.streams, prefill=args.prefill,
                                           conversation_cache_mib=args.conversation_cache_mib,
                                           short_streams=args.short_streams, short_ctx=args.short_ctx,
                                           packed_op=args.packed_op, session_replay=bool(args.session_replay),
                                           slot_resume=args.slot_resume, slot_checkpoints=args.slot_checkpoints,
                                           prefill_interleave=args.prefill_interleave,
                                           interleave_min_rows=args.interleave_min_rows,
                                           state_dir=None if args.state_dir is None else str(args.state_dir),
                                           state_disk_gib=args.state_disk_gib, cache_probation=args.cache_probation,
                                           state_write_gib_per_day=args.state_disk_write_gib_per_day,
                                           state_write_delay_ms=args.state_disk_write_delay_ms,
                                           cache_idle_minutes=args.cache_idle_minutes,
                                           engine_options=engine_options,
                                           lookup=args.lookup, mtp_window=args.mtp_window)
            backend.draft_detail = args.draft_stats
        server = bind_server(args.host, args.port, handler_for(
            Service(backend, tokenizer, args.model_id, clamp_max_tokens=args.clamp_max_tokens,
                    default_thinking=args.default_thinking == "on", max_body_mib=args.max_body_mib,
                    thinking_answer_room=None if room == "auto" else int(room),
                    max_thinking_tokens=args.max_thinking_tokens, min_thinking_tokens=args.min_thinking_tokens,
                    stream_tool_calls=args.stream_tool_calls == "on",
                    max_requests_per_client=args.max_requests_per_client, vision=vision,
                    default_mtp_prefix=args.mtp_window, capture=capture, allowed_hosts=allowed_hosts,
                    defaults=defaults)))
        server.daemon_threads = True
        commit = server_commit()
        print(f"Crucible v{server_version()}" + (f" ({commit})" if commit else ""), file=sys.stderr, flush=True)
        # The switch changes prompt IDs, so the startup line records it with the address.
        print(f"Crucible is ready at {listen_url(args.host, args.port)} (content special tokens: "
              f"{args.content_special_tokens})", file=sys.stderr, flush=True)
        if not args.host.is_loopback:
            print(f"WARNING: listening on {args.host}, which is not loopback. The server has no authentication: "
                  "every machine that can reach this address can use it. Use --allowed-host (strict Host mode) "
                  "and a firewall.", file=sys.stderr, flush=True)
        if allowed_hosts:
            print("Host check: only " + ", ".join(host + (":" + port if port else "") for host, port in allowed_hosts),
                  file=sys.stderr, flush=True)
        else:
            print("Host check: off (any Host)", file=sys.stderr, flush=True)
        if defaults.describe():
            print("Request defaults: " + defaults.describe(), file=sys.stderr, flush=True)
        update_check.start(server_version(), disabled=args.no_update_check)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"server error: {error}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.server_close()
        # The capture writes its last sums before the engine stops, and the engine stops even if that fails.
        try:
            if capture is not None:
                capture.close()
        finally:
            if backend is not None:
                backend.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
