"""A streamed request that waits for a tool-tail extension keeps the HTTP status of a refusal.

server/README.md: a full queue returns HTTP 503, a draft_* refusal of the batch worker is HTTP 400, and a stream gets
its HTTP 200 headers when the server accepts it. The wait for a running extension (server/tool_tail.py) comes before
the submit, so the stream headers wait too: they go out only when the submit accepts the request.

The refusals come from the real BatchedEngineBackend.submit (server/batch_backend.py:275-293); the refusal happens
before the engine process is used, so the backend here has no process. Every text is synthetic.
"""
import queue
import threading
import time

import pytest

from server import http, tool_tail
from server.batch_backend import BatchedEngineBackend
from server.tests.serve_fixtures import invoke
from server.tests.test_think_budget import SequenceBackend
from server.tests.test_tool_replay import REPLY, USER, TimedWriter, chat_body, root, submitted  # noqa: F401
from server.tokenization import ModelTokenizer

# A client header timer that the silent wait must stay under: undici (Node fetch) headersTimeout, 300 s.
CLIENT_HEADERS_TIMEOUT = 300.0


class RefusingBackend(SequenceBackend):
    """Each submit runs the real batch refusal checks on a backend whose queue is full."""

    def __init__(self):
        super().__init__()
        self.batch = BatchedEngineBackend.__new__(BatchedEngineBackend)
        self.batch.admission_lock, self.batch.jobs = threading.Lock(), queue.Queue(maxsize=1)
        self.batch.jobs.put_nowait(None)
        self.batch.pending_token_limit, self.batch.pending_tokens = 1 << 30, 0

    def submit(self, ids, request, **options):
        self.submissions.append((list(ids), request, options))
        self.submitted_at = time.monotonic()
        return BatchedEngineBackend.submit(self.batch, ids, request, **options)


QUEUE_FULL = (dict(), 503)
DRAFT_REFUSED = (dict(spec=7, draft_head="full"), 400)


def extension_of(monkeypatch, adapter, body, finish_after):
    """A running extension of the next prompt (as test_tool_replay.running_extension) that ends after finish_after s.
    Returns the list that gets the monotonic time of the end, just before the extension ends."""
    pending, ended = tool_tail.Pending(), []
    entry = pending.add(submitted(adapter, body)[:-8])
    monkeypatch.setattr(tool_tail, "Pending", lambda: pending)
    monkeypatch.setattr(http, "KEEPALIVE_SECONDS", 0.05)

    def finish():
        ended.append(time.monotonic())
        pending.finish(entry)
    timer = threading.Timer(finish_after, finish)
    timer.start()
    return ended, timer


def served(backend, adapter, body, writer=None, extend=True):
    response = invoke(backend, adapter, body, writer=writer, service_options=dict(extend_tool_tails=extend))
    return response.statuses, response.headers.get("Content-Type"), response.body


@pytest.mark.parametrize("extra,status", [QUEUE_FULL, DRAFT_REFUSED], ids=["queue_full", "draft_refused"])
def test_a_stream_that_waited_keeps_the_status_of_a_refusal(root, monkeypatch, extra, status):
    adapter = ModelTokenizer(root)
    body = chat_body(USER, stream=True, **extra)
    direct = served(RefusingBackend(), adapter, body)  # no extension runs: no wait
    assert direct[:2] == ([status], "application/json")
    ended, timer = extension_of(monkeypatch, adapter, body, 0.3)
    backend = RefusingBackend()
    try:
        waited = served(backend, adapter, body)
    finally:
        timer.join()
    assert len(backend.submissions) == 1 and backend.submitted_at >= ended[0]  # it waited, then the submit refused it
    assert waited == direct  # the same status, type and bytes as without the wait
    assert b"data:" not in waited[2] and b"keepalive" not in waited[2]


def test_a_stream_that_waits_gets_its_headers_when_the_submit_accepts(root, monkeypatch):
    """The headers and the keepalives start after the wait, as on trunk after the submit."""
    adapter = ModelTokenizer(root)
    body = chat_body(USER, stream=True)
    writer = TimedWriter()  # its clock starts before the extension's timer
    ended, timer = extension_of(monkeypatch, adapter, body, 0.4)
    reply = adapter.tokenizer.encode(REPLY, add_special_tokens=False).ids
    backend = SequenceBackend((reply, "stop"))
    try:
        statuses, kind, data = served(backend, adapter, body, writer)
    finally:
        timer.join()
    end = ended[0] - writer.start
    assert statuses == [200] and kind == "text/event-stream" and len(backend.submissions) == 1
    assert writer.flushes and all(at >= end for at, _ in writer.flushes)  # no byte before the wait ends
    assert b"data:" in data and data.endswith(b"data: [DONE]\n\n")


def test_the_silent_wait_stays_under_client_header_timers():
    """A waiting stream gets no byte for at most WAIT_SECONDS; then it goes on as on trunk."""
    assert tool_tail.WAIT_SECONDS < CLIENT_HEADERS_TIMEOUT


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "stream"])
@pytest.mark.parametrize("extra,status", [QUEUE_FULL, DRAFT_REFUSED, (None, 200)],
                         ids=["queue_full", "draft_refused", "accepted"])
def test_with_no_wait_the_switch_changes_no_byte(root, extra, status, stream):
    """Differential: with no running extension, the switch on and off give the same status, type and bytes."""
    adapter = ModelTokenizer(root)
    body = chat_body(USER, stream=stream, **(extra or {}))
    reply = adapter.tokenizer.encode(REPLY, add_special_tokens=False).ids
    results = []
    for extend in (False, True):
        backend = RefusingBackend() if extra is not None else SequenceBackend((reply, "stop"))
        results.append(served(backend, adapter, body, extend=extend))
    assert results[0][0] == [status]
    normal = [(statuses, kind, normalized(data)) for statuses, kind, data in results]
    assert normal[0] == normal[1]


def normalized(data):
    """The reply bytes without the fields that change per request (id, created, seed)."""
    import re
    return re.sub(rb'"(id|created|seed)": ?("[^"]*"|\d+)', b'"\\1": 0', data)


def test_the_extension_job_keeps_the_rope_mode_but_not_the_clamp(root):
    """The extension job keeps the request's RoPE mode (the engine policy of the tail), but not its
    clamp: else rope.done_lines writes a clamp line and a cut line (2 tokens = rope_cap) for it (backend.py:353-358)."""
    import json
    from dataclasses import replace
    from types import SimpleNamespace
    from server.api import validate_request
    from server.tests.test_tool_replay import FORMS, edit_call, generated, output_text, serve
    adapter = ModelTokenizer(root)
    ids = generated(adapter, output_text([edit_call()], FORMS["compact"]))
    _, response = serve(adapter, chat_body(USER), ids, extend=False)
    message = json.loads(response.body)["choices"][0]["message"]
    request = replace(validate_request(chat_body(USER), "chat", "crucible"), rope_clamped_from=9000, rope_mode="plain")
    backend = SequenceBackend(([1, 2], "length"))
    service = SimpleNamespace(tokenizer=adapter, backend=backend, tail_extensions=tool_tail.Pending())
    job = tool_tail.extend(service, request, adapter.encode(request), ids, message)
    assert job is not None and job.cancelled.wait(5)
    job_request = backend.submissions[0][1]
    assert job_request.rope_clamped_from == 0 and job_request.rope_mode == "plain"
