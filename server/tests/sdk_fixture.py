"""CPU-only recorded token scripts for real HTTP client compatibility tests, and the live switch of the SDK tests.

Live switch: CRUCIBLE_LIVE_BASE_URL (for example http://127.0.0.1:18793, a trailing /v1 is
accepted) points the SDK test classes at a running server with the model id `crucible`. A test that needs the fake backend
(recorded token scripts, fault injection, the fake's request log) then skips with NEEDS_FAKE; the rest go to that server,
and server/tests/test_sdk_live.py runs its live cases. With the variable unset, every class serves the real HTTP layer on
the CPU fake backend in this process, as before."""
from collections import deque
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
import os
import queue
import threading
from types import SimpleNamespace
import unittest
from server.api import ApiError
from server.http import Service, handler_for

LIVE_ENV = "CRUCIBLE_LIVE_BASE_URL"
NEEDS_FAKE = f"live mode ({LIVE_ENV} is set): the test needs the CPU fake backend (recorded token scripts)"
FAKE_TIMEOUT_S, LIVE_TIMEOUT_S = 5, 120   # a client timeout: the fake answers at once, the engine decodes up to 256 tokens


def live_root(environ=os.environ):
    """The root URL of the live server (no /v1, no trailing /), or "" when the switch is unset."""
    url = environ.get(LIVE_ENV, "").strip().rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


LIVE = live_root()
fake_only = unittest.skipIf(LIVE, NEEDS_FAKE)


class HttpTarget:
    """Mixin of the SDK test classes: the server under test. Fake mode: the real HTTP layer on `backend` (a free port: two
    sessions may run the tests at once). Live mode: the server of LIVE_ENV; `backend` is None."""

    @classmethod
    def start_target(cls, backend=None):
        """(root URL, client timeout in seconds)."""
        cls.server = None
        if LIVE:
            cls.tokenizer = cls.backend = None
            return LIVE, LIVE_TIMEOUT_S
        cls.tokenizer = RecordedTokenizer()
        cls.backend = (backend or RecordedBackend)(cls.tokenizer)
        handler = handler_for(Service(cls.backend, cls.tokenizer))
        handler.log_message = lambda *args: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        return f"http://127.0.0.1:{cls.server.server_address[1]}", FAKE_TIMEOUT_S

    @classmethod
    def stop_target(cls):
        if cls.server is not None:
            cls.server.shutdown()
            cls.thread.join(timeout=5)
            cls.server.server_close()

    def reset(self, *scripts):
        if self.backend is not None:
            self.backend.reset(*scripts)

    def assertNotSubmitted(self):
        """The server refused the requests before the engine queue (checked on the fake backend only)."""
        if self.backend is not None:
            self.assertEqual(self.backend.requests, [])

@dataclass(frozen=True)
class Script:
    pieces: tuple = ("Hello", " snow 雪.")
    reason: str = "stop"
    error: object = None
    cached_tokens: int | None = None

class RecordedTokenizer:
    def __init__(self):
        self.pieces = {}
        self.next_id = 100

    def encode(self, request):
        return [10, 11, 12]

    def tokens(self, pieces):
        ids = tuple(range(self.next_id, self.next_id + len(pieces)))
        self.next_id += len(pieces)
        self.pieces.update(zip(ids, pieces))
        return ids

    def stream(self):
        return None

    def step(self, stream, token):
        return self.pieces.get(token, "")

    def decode(self, ids):
        return "".join(self.step(None, token) for token in ids)

class RecordedBackend:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.scripts = deque()
        self.requests = []
        self.jobs = []

    def reset(self, *scripts):
        self.scripts = deque(scripts)
        self.requests = []
        self.jobs = []

    def submit(self, ids, request):
        script = self.scripts.popleft() if self.scripts else Script()
        if isinstance(script.error, ApiError):
            raise script.error
        self.requests.append(request)
        events = queue.Queue()
        if script.error:
            events.put(dict(event="error", error=script.error))
        else:
            tokens = self.tokenizer.tokens(script.pieces)
            limited = len(tokens) > request.max_tokens
            tokens = tokens[:request.max_tokens]
            for token in tokens:
                events.put(dict(event="token", id=token))
            cached = {} if script.cached_tokens is None else dict(cached_tokens=script.cached_tokens)
            events.put(dict(event="done", prompt_tokens=len(ids), completion_tokens=len(tokens),
                            finish_reason="length" if limited else script.reason) | cached)
        job = SimpleNamespace(events=events, cancelled=threading.Event())
        self.jobs.append(job)
        return job

TOOLS = [dict(type="function", function=dict(name="add", parameters=dict(type="object",
    properties=dict(a=dict(type="integer"), b=dict(type="integer")), required=["a", "b"],
    additionalProperties=False)))]

def tool_block(a=17, b=25):
    return (f"<tool_call>\n<function=add>\n<parameter=a>\n{a}\n</parameter>\n"
            f"<parameter=b>\n{b}\n</parameter>\n</function>\n</tool_call>")
