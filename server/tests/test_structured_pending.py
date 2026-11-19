"""The grammar compile never runs on the serve loop: the engine refuses the open of
a spec that still compiles with constraint_pending, and the server opens it again later. Other requests keep
streaming meanwhile. A compile that the engine refuses is HTTP 400 for that request only, with a message that names
the limit."""
import time

from server import structured
from server.tests.test_backend_paths import collect, request, start_backend  # noqa: F401 (fixture)
from server.tests.test_serve_regressions import engine_lines


def test_a_pending_open_is_retried_and_peers_keep_streaming(start_backend, capsys):
    backend = start_backend(True, streams=2, env={"FAKE_PENDING_TOKEN": "555", "FAKE_PENDING_OPENS": "4",
                                                  "FAKE_DELAY": ".002"})
    peer = backend.submit([1], request(60))
    first = peer.events.get(timeout=3)
    assert first["event"] == "token"
    start = time.monotonic()
    pending = backend.submit([555], request(4))
    ids, done = collect(pending)
    assert ids == [556, 557, 558, 559] and done["event"] == "done"
    assert time.monotonic() - start < 3
    assert collect(peer, prefix=[first["id"]])[0] == list(range(2, 62))
    assert not backend.fatal
    # A pending open is no engine error: no log line for it.
    assert not [line for line in engine_lines(capsys) if "constraint_pending" in str(line)]


def test_a_pending_open_holds_no_slot(start_backend):
    # One slot: while the grammar compiles, the other request takes the slot and finishes.
    backend = start_backend(True, streams=1, env={"FAKE_SLOT_CONTEXTS": "128", "FAKE_PENDING_TOKEN": "555",
                                                  "FAKE_PENDING_OPENS": "50", "FAKE_DELAY": ".002"})
    pending = backend.submit([555], request(2))
    time.sleep(.05)
    assert collect(backend.submit([7], request(3)))[0] == [8, 9, 10]
    assert collect(pending)[0] == [556, 557]


def test_a_cancelled_pending_open_is_dropped(start_backend):
    backend = start_backend(True, streams=2, env={"FAKE_PENDING_TOKEN": "555", "FAKE_PENDING_OPENS": "100000",
                                                  "FAKE_DELAY": ".002"})
    pending = backend.submit([555], request(2))
    time.sleep(.05)
    pending.cancelled.set()
    time.sleep(.1)
    assert collect(backend.submit([7], request(3)))[0] == [8, 9, 10]
    assert not backend.fatal


def test_a_compile_refused_by_a_limit_is_a_400_that_names_the_limit(start_backend):
    for text, words in (("constraint_limit: the grammar compile exceeds kHelperDeadlineMs = 5000", "5 s"),
                        ("constraint_limit: the grammar compile exceeds kHelperMemoryBytes = 2 GiB", "2 GiB"),
                        ("constraint_compile: the grammar compiler stopped on this schema", "compiler stopped")):
        backend = start_backend(True, streams=2, env={"FAKE_REJECT_TOKEN": "777", "FAKE_REJECT_TEXT": text,
                                                      "FAKE_DELAY": ".002"})
        peer = backend.submit([1], request(40))
        assert peer.events.get(timeout=3)["event"] == "token"
        event = backend.submit([777], request()).events.get(timeout=3)
        assert event["event"] == "error" and event.get("status") == 400, (text, event)
        assert words in event["error"], (text, event)
        assert collect(peer, prefix=[2])[0][-1] == 41
        assert not backend.fatal


def test_a_full_compile_queue_is_a_503_not_a_client_error(start_backend):
    # The schema is valid: the engine is busy. A 400 tells a client not to retry; a 503 tells it to retry later.
    text = "constraint_busy: the grammar compile queue is full (kCompileQueue = 8)"
    backend = start_backend(True, streams=2, env={"FAKE_REJECT_TOKEN": "777", "FAKE_REJECT_TEXT": text,
                                                  "FAKE_DELAY": ".002"})
    event = backend.submit([777], request()).events.get(timeout=3)
    assert event["event"] == "error" and event.get("status") == 503, event
    assert "Retry" in event["error"], event
    assert collect(backend.submit([7], request(3)))[0] == [8, 9, 10]
    assert not backend.fatal


def test_the_compile_wait_is_bounded():
    assert 0 < structured.COMPILE_POLL_SECONDS <= 0.05
    assert structured.COMPILE_WAIT_SECONDS >= 60  # the helper start (60 s) is the longest wait
