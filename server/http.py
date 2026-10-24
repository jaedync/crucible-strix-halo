"""Local-only HTTP endpoints. Client disconnects never leave unread engine output."""
import json
import math
import queue
import select
import socket
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler
from os.path import basename
from traceback import extract_tb
from . import anthropic, metrics
from .api import DEFAULTS, LEGACY_MODEL, MTP_WINDOW, ApiError, validate_request
from .hosts import REFUSED, host_allowed
from .response import CompletionResponse
from .think_budget import SKIPPED_BY, skips_thinking, think_cap
from .version import server_commit, server_version

# While no event comes, the handler polls the client this often, so a disconnect cancels the job between chunks.
POLL_SECONDS = 0.25
# An SSE comment this often keeps a started stream alive while it waits or prefills (undici's body timer is 300 s).
KEEPALIVE_SECONDS = 10
# A job's first event can wait behind several long cold prefills; a connected client keeps waiting.
FIRST_EVENT_SECONDS = 3600
# Between the events of a running job; the engine watchdogs stop a stalled command sooner.
NEXT_EVENT_SECONDS = 360
# Anthropic Messages token count (Claude-style clients and gateways ask it before each request; some add ?beta=true).
COUNT_TOKENS_PATH = "/v1/messages/count_tokens"
# RA-7: open connections at one time (one thread each), and the time to send the request head and body. The 30 s
# socket timeout is per operation, so a client that sends one byte every 25 s held a thread for ever.
MAX_CONNECTIONS = 256
READ_DEADLINE_SECONDS = 60
WATCH_SECONDS = 1.0
# SH-2: a refused body up to this size is read and discarded, so the close does not reset the connection before the
# client reads the error. A larger one is not worth the bandwidth; the read deadline bounds the time either way.
DRAIN_LIMIT_BYTES = 256 * 1024 * 1024
# SH-3: the access log names only these methods, versions and paths; any other request line is client text.
LOGGED_METHODS = ("GET", "POST")
LOGGED_VERSIONS = ("HTTP/1.0", "HTTP/1.1")
LOGGED_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/crucible/trace", "/v1/crucible/version",
                COUNT_TOKENS_PATH, "/tokenize", "/v1/tokenize", "/apply-template", "/health")
# Fixed client texts for the protocol errors that BaseHTTPRequestHandler raises before a method handler runs.
PROTOCOL_ERRORS = {400: "The request line or headers are malformed.", 414: "The request line is too long.",
                   431: "The request headers are too large or too many.",
                   501: "This method is not supported. Use GET or POST.", 505: "Use HTTP/1.1."}


class ReadDeadlines:
    """One thread closes connections that did not finish their request head and body in time."""
    def __init__(self):
        self.lock, self.deadlines, self.thread = threading.Lock(), {}, None

    def watch(self, connection, seconds):
        with self.lock:
            self.deadlines[connection] = time.monotonic() + seconds
            if self.thread is None:
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()

    def done(self, connection):
        with self.lock:
            self.deadlines.pop(connection, None)

    def run(self):
        while True:
            time.sleep(WATCH_SECONDS)
            now = time.monotonic()
            with self.lock:
                late = [connection for connection, deadline in self.deadlines.items() if deadline <= now]
                for connection in late:
                    del self.deadlines[connection]
            for connection in late:
                try:
                    connection.shutdown(socket.SHUT_RDWR)  # the blocked read returns, and the handler closes
                except OSError:
                    pass

class Service:
    def __init__(self, backend, tokenizer, model="crucible", clamp_max_tokens=False, default_thinking=False,
                 max_body_mib=1, thinking_answer_room=None, max_thinking_tokens=None, stream_tool_calls=True,
                 max_requests_per_client=0, vision=None, default_mtp_prefix=MTP_WINDOW, capture=None, allowed_hosts=None,
                 defaults=DEFAULTS, min_thinking_tokens=0):
        self.backend, self.tokenizer, self.model = backend, tokenizer, model
        self.allowed_hosts = allowed_hosts  # hosts.parse_allowed_hosts(): None accepts any Host header
        self.capture = capture  # serve --draft-capture: None unless the operator turns it on
        self.default_mtp_prefix = default_mtp_prefix  # serve --mtp-window
        # RA-10: open completion requests per client address (0: no limit). Loopback clients share one address.
        self.max_requests_per_client = max_requests_per_client
        self.client_lock, self.client_requests = threading.Lock(), {}
        self.stream_tool_calls = stream_tool_calls
        self.thinking_answer_room, self.max_thinking_tokens = thinking_answer_room, max_thinking_tokens
        self.min_thinking_tokens = min_thinking_tokens  # serve --min-thinking-tokens; 0 is the rule off
        self.vision = vision
        self.max_body_mib = max_body_mib
        self.clamp_max_tokens = clamp_max_tokens
        self.default_thinking = default_thinking
        self.defaults = defaults  # serve sampling flags and --default-max-tokens (api.ServerDefaults)

    def capture_turn(self, request, prompt_ids, output_ids):
        """Give a finished turn to the draft capture, if any. Only the class and the IDs leave here."""
        if self.capture is None:
            return
        try:
            self.capture.record(tools=bool(request.tools), thinking=bool(request.enable_thinking),
                                prompt_ids=prompt_ids, output_ids=output_ids)
        except Exception as error:  # noqa: BLE001
            # The reply is already complete: a capture fault never fails a request. The text names only the type.
            print(f"draft capture: turn not recorded ({type(error).__name__})", file=sys.stderr, flush=True)

    def capture_draft(self, request, turns):
        """Give the per-token draft counters of a finished turn to the day file, if --draft-stats is on."""
        if self.capture is None or getattr(self.capture, "stats", None) is None:
            return
        try:
            events = [turn["draft_event"] for turn in turns if isinstance(turn, dict) and "draft_event" in turn]
            if events:
                self.capture.record_draft(tools=bool(request.tools), thinking=bool(request.enable_thinking), events=events)
        except Exception as error:  # noqa: BLE001 (a counter fault never reaches the handler)
            print(f"draft stats: turn not recorded ({type(error).__name__})", file=sys.stderr, flush=True)

    def admit_client(self, address):
        """Count one open request for this address, or raise 429 when it has its limit."""
        if not self.max_requests_per_client:
            return False
        with self.client_lock:
            if self.client_requests.get(address, 0) >= self.max_requests_per_client:
                raise ApiError("This client has too many open requests. Try again later.", status=429)
            self.client_requests[address] = self.client_requests.get(address, 0) + 1
        return True

    def release_client(self, address):
        with self.client_lock:
            left = self.client_requests[address] - 1
            if left: self.client_requests[address] = left
            else: del self.client_requests[address]

    def prompt_ids(self, request):
        """(prompt IDs, image offsets) as the engine prefills them. Chat completions and count_tokens both use this
        one function, so a change to the prompt (template, encoding) applies to both."""
        if request.images:
            return self.tokenizer.encode_images(request)
        return self.tokenizer.encode(request), []

    def prompt_text(self, request):
        """The rendered chat prompt that prompt_ids() encodes. A tokenizer that marks content special-token strings
        with sentinels (RA-8) returns them; the text gets the strings back, as the client wrote them."""
        render = getattr(self.tokenizer, "render_prompt", None)
        if render is None:
            return self.tokenizer.render_chat(request)
        text, restore = render(request)
        return "".join(restore.get(character, character) for character in text) if restore else text

    def fit(self, request, prompt_count):
        """--default-max-tokens max: a request with no count gets the room left in the context. Opt-in
        --clamp-max-tokens: lower max_tokens to that room. A prompt that alone overflows still fails."""
        if not (request.max_tokens_all or self.clamp_max_tokens):
            return request
        room = self.backend.context - prompt_count + 1
        if request.max_tokens_all:
            # Room below 1 means the prompt overflows: 1 token lets the backend refuse it with the usual message.
            return replace(request, max_tokens=max(room, 1), max_tokens_all=False)
        return replace(request, max_tokens=room) if 1 <= room < request.max_tokens else request

    def plan(self, request):
        """(request, prompt IDs, image offsets) as the engine gets them: the prompt is counted, max_tokens is fitted,
        and a thinking request whose answer-room close limit is below --min-thinking-tokens is served with thinking off.

        The order matters. The decision needs the final max_tokens, and with --default-max-tokens max (or the clamp)
        Service.fit sets max_tokens from the context room after the prompt is counted. So the thinking-on prompt is
        fitted first and the decision uses that count. The thinking-off prompt has another length (usually fewer
        tokens: the template adds a reasoning-effort system line only when thinking is on, but at effort medium the
        empty think block makes it longer), so a skip renders and fits again on the off prompt.
        The off request has no cap, so the decision is made once.

        A skip happens only when both plans fit the context. A prompt that the thinking-on plan refuses gets the same
        refusal as before this rule, and one that only the off plan refuses stays on thinking.
        """
        ids, offsets = self.prompt_ids(request)
        fitted = self.fit(request, len(ids))
        if self.skips_thinking(fitted) and self.fits(ids, fitted):
            off = replace(request, enable_thinking=False, thinking_skipped_by=SKIPPED_BY)
            off_ids, off_offsets = self.prompt_ids(off)
            off_fitted = self.fit(off, len(off_ids))
            if self.fits(off_ids, off_fitted):
                return off_fitted, off_ids, off_offsets
        return fitted, ids, offsets

    def fits(self, ids, request):
        """The backend's own context check (backend.submit), so a plan never turns a served request into a refusal."""
        context = getattr(self.backend, "context", None)
        return context is None or (bool(ids) and len(ids) + request.max_tokens - 1 <= context)

    def skips_thinking(self, request):
        if request.kind != "chat" or not request.enable_thinking or not hasattr(self.tokenizer, "think_close_ids"):
            return False
        budget = request.thinking_budget or self.max_thinking_tokens
        return skips_thinking(request.max_tokens, budget, self.thinking_answer_room, self.min_thinking_tokens)

    def think_cap(self, request):
        """Where the server closes the think block of this request (after the clamp), or None."""
        if request.kind != "chat" or not request.enable_thinking or not hasattr(self.tokenizer, "think_close_ids"):
            return None
        budget = request.thinking_budget or self.max_thinking_tokens
        return think_cap(request.max_tokens, budget, self.thinking_answer_room)

    @staticmethod
    def next_event(job, timeout=NEXT_EVENT_SECONDS, idle=None):
        """The job's next event. idle() runs between polls while none comes; it may raise to end the wait."""
        try:
            if idle is None:
                event = job.events.get(timeout=timeout)
            else:
                deadline = time.monotonic() + timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    try:
                        event = job.events.get(timeout=min(POLL_SECONDS, remaining))
                        break
                    except queue.Empty:
                        idle()
        except queue.Empty:
            raise ApiError("The queued request timed out. Try again later.", status=503) from None
        if event.get("event") == "error":
            raise ApiError(event.get("error", "The engine request failed."), status=503)
        return event

    @property
    def vocab(self):
        """The engine's vocabulary size from its ready event; a backend without model fields keeps the legacy one."""
        return getattr(self.backend, "model", LEGACY_MODEL).vocab

    def validate_event(self, event, ids, prompt_count, request, first_id=0):
        """Check one engine event. A continuation job starts at output position first_id."""
        if event.get("event") == "token":
            token = event.get("id")
            if type(token) is not int or not 0 <= token < self.vocab or len(ids) >= request.max_tokens:
                raise ApiError("The engine returned an invalid token event.", status=500)
            return token
        if (event.get("event") != "done" or event.get("prompt_tokens") != prompt_count or
                event.get("completion_tokens") != len(ids) - first_id or
                event.get("finish_reason") not in ("stop", "length")):
            raise ApiError("The engine returned an invalid completion event.", status=500)
        return None

def trace_since(query):
    from urllib.parse import parse_qs
    values = parse_qs(query, strict_parsing=False).get("since", ["0"])
    try:
        since = float(values[-1])
    except ValueError:
        raise ApiError("since must be a number of server monotonic seconds.", "since") from None
    if not math.isfinite(since):
        raise ApiError("since must be finite.", "since")
    return since

def error_body(error):
    return dict(error=dict(message=str(error), type="invalid_request_error" if error.status < 500 else "server_error",
                           param=error.param, code=None))

def declared_length(headers):
    """The one valid Content-Length of a body without Transfer-Encoding, or None."""
    lengths = headers.get_all("Content-Length", []) if headers is not None else []
    if (headers is None or headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isascii() or
            not lengths[0].isdecimal() or len(lengths[0]) > 10):
        return None
    return int(lengths[0])

def json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result

BUSY_BODY = b'{"error":{"message":"Too many connections. Try again later.","type":"server_error","param":null,"code":null}}'
BUSY = (b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\n"
        b"Content-Length: %d\r\n\r\n" % len(BUSY_BODY) + BUSY_BODY)


def handler_for(service):
    slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
    deadlines = ReadDeadlines()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "Crucible/0.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)
            self.admitted = slots.acquire(blocking=False)
            if self.admitted:
                deadlines.watch(self.connection, READ_DEADLINE_SECONDS)

        def handle(self):
            if not self.admitted:
                try:
                    self.wfile.write(BUSY)
                except OSError:
                    pass
                return
            super().handle()

        def finish(self):
            deadlines.done(self.connection)
            if self.admitted:
                slots.release()
            super().finish()

        def request_read(self):
            """The request head and body are in: the read deadline no longer applies."""
            connection = getattr(self, "connection", None)
            if connection is not None:
                deadlines.done(connection)

        def version_string(self):
            return self.server_version  # the default also names the Python version

        def log_request(self, code="-", size="-"):
            # SH-3: the default line quotes the raw request line, which is client text (privacy rule).
            command, version = getattr(self, "command", None), getattr(self, "request_version", None)
            path = (getattr(self, "path", None) or "").partition("?")[0]
            self.log_message('"%s %s %s" %s %s', command if command in LOGGED_METHODS else "OTHER",
                             path if path in LOGGED_PATHS else "<other>",
                             version if version in LOGGED_VERSIONS else "OTHER", int(code) if isinstance(code, int)
                             else code, size)

        def send_error(self, code, message=None, explain=None):
            """SH-3: a protocol error before a method handler runs. The default page is HTML, and its reason phrase
            and log line quote the request line; this reply is JSON with a fixed text and a crucible_error line."""
            error = ApiError(PROTOCOL_ERRORS.get(code, "The request could not be processed."), status=code)
            metrics.emit(metrics.client_error_line(None, code, error))
            try:
                self.send_json(code, error_body(error))
            except OSError:
                self.close_connection = True

        def send_response(self, code, message=None):
            self.response_started = True
            super().send_response(code, message)

        def handle_one_request(self):
            """Last resort: socketserver would drop the connection and print a traceback, whose text can hold content."""
            # A keep-alive connection reuses the handler: the SSE ending below must see this request's output only.
            self.response_started, self.output = False, None
            try:
                super().handle_one_request()
            except Exception as error:
                self.close_connection = True
                # The type and the last code locations only: the exception text and frame locals can quote the request.
                try:
                    frames = " > ".join(f"{basename(frame.filename)}:{frame.lineno}:{frame.name}"
                                         for frame in extract_tb(error.__traceback__)[-3:])
                    turn = getattr(self.jobs[0], "turn", None) if getattr(self, "jobs", None) else None
                    request_id = turn.get("request_id") if isinstance(turn, dict) else None
                    message = f"unexpected {type(error).__name__} at {frames}"
                except Exception:
                    # A raise here would reach socketserver, which prints the chained exception text.
                    request_id, message = None, f"unexpected {type(error).__name__}"
                metrics.emit(metrics.error_line(request_id, 500, message, self.response_started))
                body = error_body(ApiError("The server could not process the request.", status=500))
                output = getattr(self, "output", None)
                try:
                    if not self.response_started:
                        self.send_json(500, body)
                    elif output is not None and output.headers_sent:
                        # Stream headers go out at admission, so a fault after them still ends the stream as SSE.
                        self.sse(body)
                        self.sse("[DONE]")
                except Exception:
                    pass  # the line above records the failure; the client connection is gone or unusable
            finally:
                # the response holds this handler, so the pair is a cycle that keeps the request and its decoded
                # images (about 10.5 MiB per 2560x1440) until a full cyclic collection, which seldom runs.
                self.output, self.jobs = None, []

        def local_host(self):
            if service.allowed_hosts is not None and len(self.headers.get_all("Host", [])) > 1:
                raise ApiError("Send one Host header.", status=400)  # two headers could differ for proxy and server
            if not host_allowed(self.headers.get("Host"), service.allowed_hosts):
                raise ApiError(REFUSED, status=403)

        def send_json(self, status, body):
            encoded = json.dumps(body, ensure_ascii=True, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
            self.close_connection = True

        def send_api_error(self, error, started=False):
            # A server-side failure, or any failure after HTTP 200 was sent, is otherwise invisible in the log.
            # A 4xx gets a fixed reason code and counts (its text can name request values).
            turn = getattr(self.jobs[0], "turn", None) if getattr(self, "jobs", None) else None
            request_id = turn.get("request_id") if isinstance(turn, dict) else None
            if error.status < 500:
                metrics.emit(metrics.client_error_line(request_id, error.status, error, started))
            else:
                metrics.emit(metrics.error_line(request_id, error.status, error, started))
            try:
                if started:
                    self.sse(error_body(error))
                    self.sse("[DONE]")
                else:
                    self.send_json(error.status, error_body(error))
                    self.discard_body()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass  # Error output has the same disconnect semantics as successful output.
            self.close_connection = True

        def do_GET(self):
            self.request_read()
            try:
                self.local_host()
                path, _, query = self.path.partition("?")
                if path == "/v1/crucible/trace":
                    self.send_json(200, dict(object="crucible.trace", data=service.backend.trace.since(trace_since(query))))
                    return
                if path == "/v1/crucible/version":
                    self.send_json(200, dict(object="crucible.version", version=server_version(),
                                             commit=server_commit()))
                    return
                if path == "/health":
                    # The server binds only after the engine is ready, so an answer means ready. llama-swap polls this.
                    if getattr(service.backend, "fatal", None):
                        raise ApiError("The engine stopped.", status=503)
                    self.send_json(200, dict(status="ok"))
                    return
                if self.path != "/v1/models":
                    raise ApiError("This endpoint is not supported.", status=404)
                self.send_json(200, dict(object="list", crucible_version=server_version(),
                                         data=[dict(id=service.model, object="model", created=0, owned_by="crucible",
                                                    crucible_version=server_version())]))
            except ApiError as error:
                self.send_api_error(error)

        def discard_body(self):
            """SH-2: read the rest of a refused body before the close. With unread bytes the kernel resets the
            connection, and the client's upload fails (Node fetch: EPIPE) before it reads the error."""
            if getattr(self, "body_read", True):
                return
            self.body_read = True
            length = declared_length(getattr(self, "headers", None))
            if length is None or length > DRAIN_LIMIT_BYTES:
                return
            try:
                while length > 0:
                    chunk = self.rfile.read(min(length, 1 << 16))
                    if not chunk:
                        return
                    length -= len(chunk)
            except OSError:
                pass  # the client stopped sending; the close follows

        def read_body(self):
            self.local_host()
            if self.headers.get("Transfer-Encoding"):
                raise ApiError("Use Content-Length. Chunked request bodies are not supported.")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
                raise ApiError("A valid ASCII Content-Length is required.", status=411)
            # Bound conversion before int(), which rejects extremely long digit strings.
            limit = f"The request body must fit the {service.max_body_mib} MiB limit."
            if len(lengths[0]) > 10:
                raise ApiError(limit, status=413)
            length = int(lengths[0])
            if length == 0:
                raise ApiError("The request body is empty. Send a JSON object.")
            if length > service.max_body_mib * 1024 * 1024:
                raise ApiError(limit, status=413)
            if self.headers.get_content_type() != "application/json":
                raise ApiError("Use Content-Type: application/json.", status=415)
            raw = self.rfile.read(length)
            self.body_read = True
            self.request_read()
            if len(raw) != length:
                raise ApiError("The request body is incomplete.")
            try:
                return json.loads(raw, object_pairs_hook=json_object,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            except (ValueError, UnicodeError, RecursionError):
                raise ApiError("The request body must contain valid UTF-8 JSON.") from None

        def sse(self, body):
            encoded = body if isinstance(body, str) else json.dumps(body, ensure_ascii=True, allow_nan=False)
            self.wfile.write(("data: " + encoded + "\n\n").encode("utf-8"))
            self.wfile.flush()
            self.last_write = time.monotonic()

        def client_gone(self):
            """True when the client closed its connection. A client sends nothing after its request body, so a
            readable socket with no bytes is a close (an error is a reset)."""
            connection = getattr(self, "connection", None)
            if not isinstance(connection, socket.socket):
                return False
            try:
                readable, _, _ = select.select([connection], [], [], 0)
                return bool(readable) and connection.recv(1, socket.MSG_PEEK) == b""
            except (OSError, ValueError):
                return True

        def idle(self):
            if self.client_gone():
                raise ConnectionResetError("The client closed the connection.")
            output = getattr(self, "output", None)
            if output is not None and output.headers_sent and time.monotonic() - self.last_write >= KEEPALIVE_SECONDS:
                self.wfile.write(b": keepalive\n\n")  # an SSE comment: clients ignore it
                self.wfile.flush()
                self.last_write = time.monotonic()

        def wait_event(self, job, first=False):
            return service.next_event(job, FIRST_EVENT_SECONDS if first else NEXT_EVENT_SECONDS, self.idle)

        def complete(self, request, prompt_ids, first, options, cap):
            output = self.output
            output.start()
            job, job_prompt, first_id, event = self.jobs[-1], prompt_ids, 0, first
            while True:
                token = service.validate_event(event, output.ids, len(job_prompt), request, first_id)
                if token is None:
                    resumed = self.resume_bounded(request, prompt_ids, output, job, first_id, event, options)
                    if resumed is None:
                        break
                    job, job_prompt, first_id = self.jobs[-1], resumed, len(output.ids)
                    event = self.wait_event(job, first=True)
                    continue
                output.token(token, event.get("crucible_t"))
                if getattr(request, "top_logprobs", 0) and "logprobs" in event:
                    output.scores = event["logprobs"]
                if cap and not first_id and not output.think.thinking and getattr(job, "think_cap", None) is not None:
                    # The model closed its own think block: no continuation comes, so the batch worker must not
                    # hold new chunks for one at the cap token.
                    job.think_cap = None
                if output.stop_filter.stopped:
                    # The resident worker drains canceled jobs before its next request.
                    self.log_turn(cap, prompt_ids, output, dict(finish_reason="stop"))
                    output.finish(self.opened_count(dict(finish_reason="stop"), prompt_ids))
                    return
                if cap and output.think.thinking and len(output.ids) == cap.tokens:
                    job_prompt = self.close_thinking(request, prompt_ids, output, cap, options)
                    if job_prompt is None:
                        self.log_turn(cap, prompt_ids, output, dict(finish_reason="length"))
                        output.finish(self.opened_count(dict(finish_reason="length"), prompt_ids))
                        return
                    job, first_id = self.jobs[-1], len(output.ids)
                # A continuation job prefills the reasoning again before its first event.
                event = self.wait_event(job, first=bool(first_id) and len(output.ids) == first_id)
            self.log_turn(cap, prompt_ids, output, event)
            if first_id:
                # The continuation restores the snapshot that the first job wrote, so its count says nothing about
                # the client's prompt.
                event = self.opened_count(event, prompt_ids)
            output.finish(event)

        def resume_bounded(self, request, prompt_ids, output, job, first_id, event, options):
            """The serial worker stops a first job at the think cap (backend.Job.bounded_at). If the model closed its own
            think block before the cap, the answer goes on in a new job from the prompt and the output so far.
            Returns the new job's prompt, or None when the reply ends here."""
            room = request.max_tokens - len(output.ids)
            if (first_id or getattr(job, "bounded_at", None) != len(output.ids) or room < 1 or output.think.thinking or
                    event.get("finish_reason") != "length"):
                return None
            continuation = prompt_ids + output.ids
            self.jobs.append(service.backend.submit(continuation, replace(request, max_tokens=room), **options))
            return continuation

        def opened_count(self, event, prompt_ids):
            """event with the first job's restore as cached_tokens, for a reply with no done event of that job (a
            continuation, no room after the cap, a stop string). The batch engine names the restore in the opened
            event. Without that count the reply omits it."""
            event = {key: value for key, value in event.items() if key != "cached_tokens"}
            opened = (getattr(self.jobs[0], "turn", None) or {}).get("opened_cached")
            if type(opened) is int and 0 <= opened <= len(prompt_ids):  # == len: an End resume
                event["cached_tokens"] = opened
            return event

        def log_turn(self, cap, prompt_ids, output, event):
            """One crucible_turn line for a request with a think cap, with the counts that the client sees.

            The backend logs every other request. A request with a cap may use two engine jobs, and a log collector joins
            a line to its own request record by prompt and completion, so the server writes the line for both jobs.
            """
            if cap is None:
                return
            first = getattr(self.jobs[0], "turn", None) or metrics.new_turn()
            counts = dict(prompt_tokens=len(prompt_ids), completion_tokens=len(output.ids),
                          finish_reason=event.get("finish_reason"))
            if len(self.jobs) == 1:
                done = event if event.get("event") == "done" else counts
                metrics.emit_turn(metrics.request_line(first, done), first)
            elif output.closed is None:
                # resume_bounded: the model closed its own block, so the line names no close, only the client's counts.
                metrics.emit_turn(metrics.request_line(first, counts), first)
            else:
                cont = getattr(self.jobs[-1], "turn", None) or metrics.new_turn()
                metrics.emit_turn(metrics.request_line(first, counts, output.closed or cap, cont, event), first)

        def close_thinking(self, request, prompt_ids, output, cap, options):
            """Force the close text after cap.tokens generated tokens, then continue on a new engine job.

            The engine has no forced-token command, so the continuation prefills the generated reasoning again
            after it restores the prompt's snapshot. Tokens that the first job made past the cap are discarded.
            Returns the continuation's prompt, or None when no room is left.
            """
            first = self.jobs[-1]
            close = service.tokenizer.think_close_ids()
            written = close[:request.max_tokens - len(output.ids)]
            for token in written:
                output.token(token)
            if len(written) == len(close):
                output.closed = cap  # the usage names a close only when the whole close text was written
            room = request.max_tokens - len(output.ids)
            turn = getattr(first, "turn", None)
            if isinstance(turn, dict):
                # The client's count when the continuation starts: its crucible_decode lines go on from here.
                turn["closed_count"] = len(output.ids)
            if room < 1:
                first.cancelled.set()
                return None
            continuation = prompt_ids + output.ids
            canonical = getattr(service.tokenizer, "closed_think_ids", None)
            if output.closed and canonical:
                # Prefill the reasoning as the client's next prompt renders it, so the replay tail matches it. A longer
                # form could pass the context that the clamp left, so it keeps the generated IDs.
                rendered = prompt_ids + list(canonical(output.think.reasoning, output.ids))
                continuation = rendered if len(rendered) <= len(continuation) else continuation
            # The continuation must resume from the first job's own end state, whatever peers do meanwhile: the batch
            # worker keeps the first job's slot for it. A backend without that rule cancels and submits.
            follow = replace(request, max_tokens=room)
            continue_job = getattr(service.backend, "continue_job", None)
            if continue_job is not None:
                self.jobs.append(continue_job(first, continuation, follow, **options))
            else:
                first.cancelled.set()
                self.jobs.append(service.backend.submit(continuation, follow, **options))
            return continuation

        def count_tokens(self):
            """Anthropic-style input token count. CPU only: no engine command, queue entry, slot or cache change."""
            prompt_ids, _ = service.prompt_ids(self.chat_request(anthropic.chat_body(self.read_body())))
            self.send_json(200, dict(input_tokens=len(prompt_ids)))

        def chat_request(self, body):
            """A chat request from a utility body, with the chat path's validation and server flags."""
            return validate_request(body, "chat", service.model, default_thinking=service.default_thinking,
                                    vision=service.vision, vocab=service.vocab,
                                    default_mtp_prefix=service.default_mtp_prefix, defaults=service.defaults)

        def tokenize(self):
            """OpenAI-compatible /tokenize: {"content": text or IDs} -> the completion prompt's IDs; {"messages": ...} (any
            chat field) -> the IDs that /v1/chat/completions prefills. CPU only, as count_tokens."""
            body = self.read_body()
            if not isinstance(body, dict):
                raise ApiError("The request must be a JSON object.")
            if body.get("with_pieces"):
                raise ApiError("with_pieces is not supported.", "with_pieces")
            for name in ("add_special", "parse_special", "with_pieces"):
                if name in body and type(body[name]) is not bool:
                    raise ApiError(f"{name} must be true or false.", name)
            # Qwen adds no BOS, and the prompt path already reads special-token text: these switches change nothing.
            body = {key: value for key, value in body.items() if key not in ("add_special", "parse_special", "with_pieces")}
            if "messages" in body and "content" in body:
                raise ApiError("Send content or messages, not both.", "content")
            if "messages" in body:
                _, ids, _ = service.plan(self.chat_request(body))  # the IDs that chat prefills, thinking skip included
            elif "content" in body and set(body) - {"content", "model"}:
                field = sorted(set(body) - {"content", "model"})[0]
                raise ApiError(f"{field} is not supported with content.", field)
            elif body.get("content") == "":
                ids = []
            elif "content" in body:
                if not isinstance(body["content"], (str, list)):
                    raise ApiError("content must be text or a list of token IDs.", "content")
                model = {"model": body["model"]} if "model" in body else {}
                request = validate_request(dict(prompt=body["content"]) | model, "completion", service.model,
                                           vocab=service.vocab)
                ids, _ = service.prompt_ids(request)
            else:
                raise ApiError("Send content (text or token IDs) or messages.", "content")
            self.send_json(200, dict(tokens=list(ids)))

        def apply_template(self):
            """OpenAI-compatible /apply-template: {"messages": ...} -> {"prompt": the rendered chat prompt}. CPU only."""
            body = self.read_body()
            if isinstance(body, dict) and "messages" not in body:
                raise ApiError("messages must be a nonempty list.", "messages")
            request, _, _ = service.plan(self.chat_request(body))  # the prompt that chat renders, skip included
            self.send_json(200, dict(prompt=service.prompt_text(request)))

        def do_POST(self):
            self.jobs, self.output, self.body_read = [], None, False
            self.last_write = time.monotonic()
            started = admitted = finished = False
            address = self.client_address[0] if isinstance(getattr(self, "client_address", None), tuple) else None
            try:
                utility = {COUNT_TOKENS_PATH: self.count_tokens, "/tokenize": self.tokenize, "/v1/tokenize": self.tokenize,
                           "/apply-template": self.apply_template}.get(self.path.partition("?")[0])
                if utility is not None:
                    utility()
                    return
                kinds = {"/v1/chat/completions": "chat", "/v1/completions": "completion"}
                if self.path not in kinds:
                    raise ApiError("This endpoint is not supported.", status=404)
                request = validate_request(self.read_body(), kinds[self.path], service.model,
                                           default_thinking=service.default_thinking, vision=service.vision,
                                           vocab=service.vocab, default_mtp_prefix=service.default_mtp_prefix,
                                           defaults=service.defaults)
                admitted = service.admit_client(address)
                request, prompt_ids, offsets = service.plan(request)
                options = {}
                cap = service.think_cap(request)
                if cap:
                    options["log_turn"] = False  # log_turn() writes one line for the whole request
                if offsets:
                    options["image_offsets"] = offsets  # a thinking-cap continuation reuses the same images
                if hasattr(service.tokenizer, "cache_prefix_tokens"):
                    prefix = service.tokenizer.cache_prefix_tokens(request, prompt_ids)
                    shared = getattr(service.tokenizer, "shared_prefix_tokens", lambda request, ids: 0)(request, prompt_ids)
                    options |= dict(cache_prefix_tokens=prefix) | (dict(shared_prefix_tokens=shared) if shared else {})
                # The batch worker holds new chunks for the continuation once the cap token is out.
                self.jobs.append(service.backend.submit(prompt_ids, request, **options,
                                                        **(dict(think_cap=cap.tokens) if cap else {})))
                self.output = CompletionResponse(self, service, request, prompt_ids)
                # A stream gets its headers at admission. Keepalives then cover the queue wait and the prefill,
                # and a later failure ends the stream with an SSE error.
                self.output.headers()
                started, self.last_write = request.stream, time.monotonic()
                first = self.wait_event(self.jobs[-1], first=True)
                self.complete(request, prompt_ids, first, options, cap)
                finished = True
            except ApiError as error:
                self.send_api_error(error, started)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
                if self.jobs:
                    turn = getattr(self.jobs[0], "turn", None)
                    output = self.output
                    metrics.emit(metrics.cancel_line(turn.get("request_id") if isinstance(turn, dict) else None,
                                                     len(output.ids) if output else 0, started))
            finally:
                for job in self.jobs:
                    job.cancelled.set()
                if admitted:
                    service.release_client(address)
                # The draft capture runs last, after the job cancel and the client release.
                if finished:
                    service.capture_turn(request, prompt_ids, self.output.ids)
                    service.capture_draft(request, [getattr(job, "turn", None) for job in self.jobs])
    return Handler
