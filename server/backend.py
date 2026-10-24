"""One resident engine, one FIFO worker, and bounded client event queues."""
from dataclasses import dataclass, field
import json
import os
import queue
import subprocess
import sys
import threading
import time
from . import metrics
from .engine_messages import ENGINE_REJECTED, ProtocolError
from .api import LEGACY_MODEL, ApiError, ModelInfo, STOP_IDS
from .hosts import ALLOWED_HOSTS_ENV
from .pixel_files import PixelStore

@dataclass
class Job:
    payload: dict
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=32))
    cancelled: threading.Event = field(default_factory=threading.Event)
    turn: dict = field(default_factory=metrics.new_turn)
    # A think-cap continuation (http.py close_thinking) names the job whose end state it resumes; that job names it.
    after: object = None
    successor: object = None
    pixels: object = None  # PixelFiles of an image request; removed when the engine has finished with the job
    # The generated-token count at which the server closes this job's think block (think_budget.think_cap), or None.
    think_cap: object = None
    # The serial worker sends think_cap as the engine's max_tokens (it cannot cancel a job); http.py then resumes the
    # turn on a second job when the model closed its own think block first. None: the engine has the request's max.
    bounded_at: object = None

# A turn that waited longer than this for the device worker counts as queued.
QUEUED_SECONDS = 0.01
# The host's memory (the test machine: 128 GiB). The engine's startup admission checks the real fit against free memory.
MAX_CONVERSATION_CACHE_MIB = 131072
# engine/serve/serve.h kMaxLine. A longer command line is an error before the engine knows its op, so the batch engine
# cannot recover it (RA-2): the server refuses such a request before it reaches the engine.
ENGINE_LINE_BYTES = 2 * 1024 * 1024
# RA-6: the serial worker waits this long for a full client queue to drain, then stops that client's output, so a
# client that stops reading cannot hold the engine for the others (the batch backend has output_timeout).
OUTPUT_TIMEOUT_SECONDS = 30
# Room for the batch wrapper {"op":"open","request_id":<20 digits>,"request":...} and the newline.
LINE_WRAPPER_BYTES = 64
# Prompt lookup (engine draft_policy.h): copy up to LOOKUP_DEPTH history tokens after a suffix match of at least
# LOOKUP_MIN tokens. The serve default since train 16: +90% decode on verbatim copy, about 1% elsewhere
# Verification commits only target tokens, so outputs do not change.
LOOKUP_DEPTH, LOOKUP_MIN = 7, 8

# RA-15: the only CRUCIBLE_* engine switches that serve passes on. Each one changes how the engine computes (a kernel
# or layout A/B, a profile's PLE table or ablation file) or prints numbers only (timings, draft counts, kill-check
# counts). An allow-list fails closed: a capture, dump, analysis or file switch, and any switch added later, stays off under serve until someone classifies it
# (tests/test_switch_lists.py checks this set).
ENGINE_KEEP_SWITCHES = frozenset({
    "CRUCIBLE_ABLATE", "CRUCIBLE_COMPACT_STATE",
    "CRUCIBLE_DECODE_RING", "CRUCIBLE_EXACT",
    "CRUCIBLE_FUSE", "CRUCIBLE_HC_WMMA",
    "CRUCIBLE_LOOKAHEAD", "CRUCIBLE_LOOKAHEAD_MIB",
    "CRUCIBLE_MTP_SIDECAR", "CRUCIBLE_PACKED_ROWS",
    "CRUCIBLE_PERSISTENT_GDN",
    "CRUCIBLE_PF_EXACT_SPLIT", "CRUCIBLE_PF_FX_LEAD", "CRUCIBLE_PF_GDN_EAGER",
    "CRUCIBLE_PF_GDN_MICRO", "CRUCIBLE_PF_HC_FUSED", "CRUCIBLE_PF_LAUNCH_ROWS",
    "CRUCIBLE_PF_MOE_RM",
    "CRUCIBLE_PF_PAD_H", "CRUCIBLE_PF_PRUNE", "CRUCIBLE_PF_QSA_ROWS",
    "CRUCIBLE_PF_STAGE_SHARE", "CRUCIBLE_PF_TILES",
    "CRUCIBLE_PLE_BF16", "CRUCIBLE_PLE_HOST", "CRUCIBLE_PLE_TRACE", "CRUCIBLE_PROFILE",
    "CRUCIBLE_SERVE_PROFILE", "CRUCIBLE_SPEC_STATS", "CRUCIBLE_STATE_CACHE",
    "CRUCIBLE_STATIC_CACHE", "CRUCIBLE_TUNE_ROWS",
    "CRUCIBLE_VERIFY_QSA_CHECK", "CRUCIBLE_WARMUP",
})

# Read by the server itself, never by the engine (server/update_check.py). Not a switch, so no removal note, and the
# engine environment drops it (the state-disk identity hashes every CRUCIBLE_* name the engine sees).
SERVER_ONLY_VARIABLES = frozenset({"CRUCIBLE_NO_UPDATE_CHECK"})


def engine_environment(environ):
    """The engine's environment with only the allowed CRUCIBLE_* switches (RA-15), and the removed names to report.

    The environment always loses CRUCIBLE_ALLOWED_HOSTS and the SERVER_ONLY_VARIABLES, but the list does not name
    them: a server setting is not an ignored engine switch. The caller must pass the environment to the engine even when the list is empty, because
    the state-disk identity hashes every CRUCIBLE_* name the engine sees."""
    removed = sorted(name for name in environ if name.startswith("CRUCIBLE_") and name not in ENGINE_KEEP_SWITCHES)
    # A server setting, not an engine switch: the engine never sees it, and the startup line does not report it.
    removed_here = [name for name in removed if name != ALLOWED_HOSTS_ENV and name not in SERVER_ONLY_VARIABLES]
    return {name: value for name, value in environ.items() if name not in removed}, removed_here


class EngineBackend:
    # Client event queue size per job; a batch worker that decodes inside prefill chunks raises it.
    event_capacity = 32
    # Batch mode with serve --draft-stats: speculative requests ask the engine for per-token draft counters.
    draft_detail = False
    # Seconds that close() waits after the end of input before it stops the engine with a signal. The batch worker
    # sets it when the engine writes its states to disk at the end of input (engine/runtime/state_disk.h).
    flush_seconds = 0
    # The engine's model (vocabulary, EOS, size) from its ready event.
    model = LEGACY_MODEL
    # Prompt lookup depth for speculative requests; __init__ sets it (the server's --lookup). The library default
    # equals the serve default, so bench, soak and gate tools that build a backend measure what production runs.
    lookup = LOOKUP_DEPTH
    # The longest engine command line; None reads ENGINE_LINE_BYTES at each submit. Image pixels go in private files
    # so one limit serves every engine.
    engine_line_bytes = None
    engine_id = None  # set in __init__; None in tests that build the worker without it
    # The batch worker times the prefill chunks that hold a decoder back (crucible_turn stall_s); the serial worker does not.
    counts_stall = False
    # The serial protocol has no cancel: a first job that ran past the think cap decoded up to max_tokens for nothing
    # while its continuation waited (Discord, "thinking ignores max tokens"). The batch worker pauses a job at its cap.
    bounds_think_cap = True

    def __init__(self, executable, checkpoint, context=32768, conversation_cache_mib=0, prefill=None, *,
                 engine_options=(), lookup=LOOKUP_DEPTH):
        self.context = context
        self.engine_id = metrics.file_id(executable)  # a number-like name for crucible_turn, read once
        self.lookup = self.check_lookup(lookup)
        self.trace = metrics.Trace()
        self.closed = threading.Event()
        self.ownership_lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.owned_jobs = {}
        self.jobs = queue.Queue(maxsize=64)
        self.engine_events = queue.Queue(maxsize=128)
        self.fatal = None
        self.pixels = PixelStore()
        arguments = [str(executable), "serve", str(checkpoint), "--ctx", str(context)]
        if conversation_cache_mib:
            arguments += ["--conversation-cache-mib", str(conversation_cache_mib)]
        if prefill is not None:
            arguments += ["--prefill", str(prefill)]
        arguments += list(engine_options)
        environment, removed = engine_environment(os.environ)
        if removed:
            # The names only: a value can be a path that names a user.
            print("crucible: serve ignores " + ", ".join(removed) + " (not on the serve allow-list of engine switches)",
                  file=sys.stderr, flush=True)
        self.process = subprocess.Popen(arguments,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1,
                                        env=environment)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            ready = self._next()
            self._validate_ready(ready)
        except BaseException:
            self.close()
            raise
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    @staticmethod
    def check_lookup(depth):
        if type(depth) is not int or not 0 <= depth <= LOOKUP_DEPTH:
            raise ValueError(f"Prompt lookup depth must be an integer in [0,{LOOKUP_DEPTH}].")
        return depth

    def _validate_ready(self, ready):
        if not isinstance(ready, dict) or ready.get("event") != "ready" or ready.get("context") != self.context:
            raise ProtocolError("The engine did not return a valid ready event.")
        # Engines before the replay-tail guard reject the field, so it is sent only when the engine names it.
        self.cache_tail = ready.get("cache_tail") is True
        if "model" in ready:
            self.model = self._ready_model(ready["model"])

    @staticmethod
    def _ready_model(fields):
        names = ("arch", "vocab", "eos", "layers", "hidden")
        if not isinstance(fields, dict) or set(fields) != set(names):
            raise ProtocolError("The engine did not return valid model fields.")
        model = ModelInfo(**fields)
        sizes = (model.vocab, model.layers, model.hidden)
        # The server stops on STOP_IDS, so the engine's EOS must be one of them.
        if (type(model.arch) is not str or any(type(size) is not int or size < 1 for size in sizes) or
                type(model.eos) is not int or not 0 <= model.eos < model.vocab or model.eos not in STOP_IDS):
            raise ProtocolError("The engine did not return valid model fields.")
        return model

    def _read(self):
        try:
            for line in self.process.stdout:
                if len(line) > 65536:
                    raise ProtocolError("The engine returned an oversized protocol event.")
                event = json.loads(line)
                while not self.closed.is_set():
                    try:
                        self.engine_events.put(event, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception:
            self.fatal = "The engine returned an invalid protocol event. Check its stderr log."
        finally:
            if not self.closed.is_set():
                self.fatal = self.fatal or "The engine stopped. Check its stderr log."

    def _next(self):
        for _ in range(3000):
            try:
                return self.engine_events.get(timeout=0.1)
            except queue.Empty:
                if self.fatal or self.closed.is_set():
                    raise ProtocolError(self.fatal or "The engine was stopped.")
        raise ProtocolError("The engine did not respond within 300 seconds.")

    def continue_job(self, job, ids, request, **options):
        """Cancel `job` and continue its request on a new job that resumes from `job`'s own end state.

        The new job is queued and linked before the cancel, so the worker never sees the cancel without it. The batch
        worker keeps the slot for it; the serial worker runs jobs in order.
        """
        follow = self.submit(ids, request, after=job, **options)
        job.cancelled.set()
        return follow

    def submit(self, ids, request, cache_prefix_tokens=0, shared_prefix_tokens=0, log_turn=True, image_offsets=(),
               after=None, think_cap=None):
        """log_turn=False leaves the crucible_turn line to the caller (one line for a request that uses two jobs)."""
        counts = dict(prompt_tokens=len(ids), max_tokens=request.max_tokens, context=self.context)
        if not ids:
            raise ApiError("The prompt and requested output must fit the configured context.", "max_tokens",
                           code="context_length", counts=counts)
        if len(ids) + request.max_tokens - 1 > self.context:
            # Clients such as Pi detect "exceeds the context window" and compact their history.
            raise ApiError(f"The prompt ({len(ids)} tokens) plus max_tokens ({request.max_tokens}) exceeds the context "
                           f"window of {self.context} tokens.", "max_tokens", code="context_length", counts=counts)
        policy = dict(draft_q=request.draft_q, draft_head=request.draft_head, mtp_prefix=request.mtp_prefix)
        if request.draft_adapt is not None:
            policy = policy | dict(draft_adapt=request.draft_adapt)
        if self.lookup and request.spec and request.max_tokens > 1:
            # Sent only to speculative requests, so plain payloads and the off switch keep the old bytes.
            policy = policy | dict(lookup=self.lookup, lookup_min=LOOKUP_MIN)
        if shared_prefix_tokens:
            # Sent only when known, so engines without the G7 field see the same payload as before.
            policy = policy | dict(shared_prefix_tokens=shared_prefix_tokens)
        if self.draft_detail and request.spec and request.max_tokens > 1:
            policy = policy | dict(draft_detail=1)  # sent only with the flag, so the default payload stays as before
        if getattr(request, "enable_thinking", False) and getattr(self, "cache_tail", False):
            # Clients often send back the answer without the reasoning, and a closed think block continues on a
            # second job. The tail of such a turn must not replace the session's resume point
            # It changes only what the cache keeps, never an output.
            policy = policy | dict(cache_tail="guard")
        pixels = None
        if request.images:
            # Sent only with images, so text-only payloads stay byte-identical.
            from .vision import engine_images
            pixels = self.pixels.write(request.images)
            policy = policy | dict(images=engine_images(request.images, image_offsets, pixels.paths))
        scoring = dict(top_logprobs=request.top_logprobs) if getattr(request, "top_logprobs", 0) else {}
        bounded = self.bounds_think_cap and think_cap is not None and 0 < think_cap < request.max_tokens
        try:
            job = Job(dict(ids=list(ids), max_tokens=think_cap if bounded else request.max_tokens, spec=request.spec,
                           stop_ids=list(STOP_IDS),
                           temperature=request.temperature, top_k=request.top_k, top_p=request.top_p,
                           min_p=request.min_p, seed=request.seed, cache_control=request.cache_control,
                           cache_prefix_tokens=cache_prefix_tokens) | policy | scoring,
                      events=queue.Queue(maxsize=self.event_capacity), pixels=pixels)
            if len(json.dumps(job.payload, separators=(",", ":"))) + LINE_WRAPPER_BYTES > (self.engine_line_bytes or ENGINE_LINE_BYTES):
                raise ApiError("The request is too large for the engine. Send a shorter prompt.", status=413,
                               code="line_too_large")
            job.turn["enabled"] = request.crucible_metrics
            job.turn["log"] = log_turn
            job.turn["engine_id"] = self.engine_id
            # crucible_decode lines: only a request that speculates writes them (metrics.decode_line).
            job.turn["speculative"] = bool(request.spec) and request.max_tokens > 1
            if self.counts_stall: job.turn["stall_s"] = 0.0
            # Token classes (crucible_turn): a continuation resumes after the closed think block.
            job.turn["think_open"] = after is None and getattr(request, "kind", None) == "chat" and \
                getattr(request, "enable_thinking", False) is True
            job.think_cap = think_cap
            job.bounded_at = think_cap if bounded else None
            if after is not None:
                job.after, after.successor = after, job
            with self.ownership_lock:
                if self.closed.is_set() or self.fatal:
                    raise ApiError("The engine is not available. Check its stderr log.", status=503)
                try:
                    self.jobs.put_nowait(job)
                except queue.Full:
                    raise ApiError("The request queue is full. Try again later.", status=503) from None
                self.owned_jobs[id(job)] = job
        except BaseException:
            # A request that never reaches the worker removes its pixel files here.
            if pixels is not None: pixels.remove()
            raise
        return job

    def _forget_job(self, job):
        """The engine has finished with this job (or never saw it): its pixel files go now."""
        with self.ownership_lock:
            self.owned_jobs.pop(id(job), None)
        if job.pixels is not None: job.pixels.remove()

    def _try_deliver(self, job, event):
        # Publication and shutdown share one lock, so no token can follow a shutdown error.
        with self.ownership_lock:
            if not self.closed.is_set() and not job.cancelled.is_set():
                try:
                    job.events.put_nowait(event)
                except queue.Full:
                    return False
            if event.get("event") in ("done", "error"):
                self.owned_jobs.pop(id(job), None)
            return True

    def _deliver(self, job, event):
        deadline = time.monotonic() + OUTPUT_TIMEOUT_SECONDS
        while not self._try_deliver(job, event):
            if time.monotonic() >= deadline:
                self._stop_output(job)
                return
            self.closed.wait(0.1)

    def _stop_output(self, job):
        """End a stalled client's output with a terminal error; the worker then drains the engine without it."""
        with self.ownership_lock:
            job.cancelled.set()
            # Drop the unread output too: the client gets the error, never a reply with a gap.
            while True:
                try: job.events.get_nowait()
                except queue.Empty: break
            job.events.put_nowait(dict(event="error", error="The output consumer timed out. Retry the request."))
            self.owned_jobs.pop(id(job), None)

    def _work(self):
        while not self.closed.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job.cancelled.is_set():
                self._forget_job(job)
                continue
            try:
                if self.fatal:
                    raise ProtocolError(self.fatal)
                job.turn["t_admit"] = job.turn["t_first_gpu"] = metrics.now()
                job.turn["queued"] = job.turn["t_admit"] - job.turn["t_submit"] > QUEUED_SECONDS
                self.process.stdin.write(json.dumps(job.payload, separators=(",", ":")) + "\n")
                self.process.stdin.flush()
                while True:
                    event = self._next()
                    if not isinstance(event, dict) or event.get("event") not in ("token", "done", "error"):
                        raise ProtocolError("The engine returned an unexpected event.")
                    if event["event"] == "error":
                        # RA-13: the engine text names internal limits; it goes to the server log only.
                        metrics.emit(metrics.engine_error_line(job.turn.get("request_id"), "generate",
                                                               event.get("error"), True))
                        event = dict(event="error", error=ENGINE_REJECTED)
                    self._deliver(job, self._stamp(job, event))
                    if event["event"] != "token":
                        break
            except Exception as error:
                metrics.emit(metrics.worker_error_line(error))
                self.fatal = "The engine request failed. Restart the test server and check its stderr log."
                self._deliver(job, dict(event="error", error=self.fatal))
            finally:
                self._forget_job(job)

    @staticmethod
    def _stamp(job, event):
        """Track server timestamps for every turn and log one line when it ends.
        Responses carry the timestamps only when the request opts in (crucible_metrics)."""
        t = metrics.now()
        enabled = job.turn["enabled"]
        if event.get("event") == "token":
            if job.turn["t_first_token"] is None: job.turn["t_first_token"] = t
            if type(event.get("id")) is int: metrics.count_token(job.turn, event["id"])
            metrics.note_token(job.turn, event)
            return dict(event, crucible_t=t) if enabled else event
        if event.get("event") == "done":
            job.turn["t_done"] = t
            if "draft_tally" in event: job.turn["draft_event"] = event  # for the per-token day file (draft_stats.py)
            summary = metrics.finalize(job.turn, event)
            if job.turn.get("log", True):
                metrics.emit_turn(metrics.turn_line(summary, event, metrics.draft_fields((job.turn, event)),
                                                    metrics.load_fields((job.turn,))), job.turn)
            return dict(event, crucible_metrics=summary) if enabled else event
        return event

    def close(self):
        with self.ownership_lock:
            self.closed.set()
            for job in self.owned_jobs.values():
                error = dict(event="error", error=self.fatal or "The engine was stopped. Retry the request.")
                while True:
                    try:
                        job.events.put_nowait(error)
                        break
                    except queue.Full:
                        # A full consumer queue must not prevent its terminal shutdown event.
                        try: job.events.get_nowait()
                        except queue.Empty: pass
                job.cancelled.set()
            self.owned_jobs.clear()
            while True:
                try: self.jobs.get_nowait()
                except queue.Empty: break
        with self.close_lock:
            if self.process.poll() is None and self.flush_seconds:
                try:
                    self.process.stdin.close()
                    self.process.wait(timeout=self.flush_seconds)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            if self.process.poll() is None and self.flush_seconds:
                # A tier engine keeps running after SIGTERM (it would flush again): it overran its flush time.
                self.process.kill()
                try: self.process.wait(timeout=10)
                except subprocess.TimeoutExpired: pass
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.stdout:
                self.process.stdout.close()
            # The engine has exited, so no request reads a pixel file any more.
            if getattr(self, "pixels", None) is not None:
                self.pixels.close()
        for thread in (getattr(self, "worker", None), getattr(self, "reader", None)):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=10)
