"""X-Client-Request-Id: the gateway's request id reaches the log lines of the request as client_request_id, so the gateway pairs
a request with its lines when requests overlap. Absent or invalid: the lines keep today's keys. Never in a reply."""
import json
import socket
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

import pytest

from server import metrics
from server.api import ApiError
from server.http import Service, handler_for
from server.tests.serve_fixtures import DisconnectWriter, PiecesTokenizer, ScriptBackend, chat, invoke, token_events
from server.tests.test_backend_paths import collect, request, start_backend  # noqa: F401 (fixture)
from server.tests.test_live_drafts import LIVE, parsed
from server.tests.test_think_budget import ANSWER, IM_END, REASON, BudgetTokenizer, LinkedBackend, chat as think_chat

ID = "0123456789abcdef0123456789abcdef"
INVALID = ["", "a" * 65, "bad id", "bad/id", "id\u00e9", "id;drop", "id,2", " "]


def kinds(lines):
    return {line.split(" ", 1)[0] for line in lines}


def ids_of(lines, *wanted):
    return [(line.split(" ", 1)[0], json.loads(line.split(" ", 1)[1]).get("client_request_id"))
            for line in lines if line.split(" ", 1)[0] in wanted]


def test_the_id_must_match_the_pattern_in_full():
    assert metrics.client_request_id(ID) == ID
    assert metrics.client_request_id("a" * 64) and metrics.client_request_id("A_b.c:d-1")
    for value in INVALID + ["ok\n", None, 7]:
        assert metrics.client_request_id(value) is None, value


def test_every_line_builder_writes_the_field_only_for_a_valid_id():
    turn = dict(request_id=9, t_first_gpu=1.0, client_request_id=ID)
    plain = dict(turn, client_request_id=None)
    error = ApiError("x")
    built = lambda cid, t: [  # noqa: E731
        metrics.prefill_line(t, 1, 2, 2.0),
        metrics.error_line(1, 500, "x", False, cid), metrics.client_error_line(1, 400, error, False, cid),
        metrics.cancel_line(1, 2, False, cid), metrics.fallback_line(1, 2, False, None, cid),
        metrics.turn_line(dict(request_id=1), dict(), None, metrics.load_fields((t,)))]
    for line, bare in zip(built(ID, turn), built(None, plain)):
        fields, base = json.loads(line.split(" ", 1)[1]), json.loads(bare.split(" ", 1)[1])
        assert fields.pop("client_request_id") == ID and fields == base
        assert "client_request_id" not in bare
    for bad in INVALID:
        assert built(bad, dict(plain, client_request_id=bad)) == built(None, plain)


def test_a_bad_value_never_reaches_a_line_through_a_turn():
    lines = [metrics.turn_line(dict(request_id=1), dict(), None, metrics.load_fields((dict(client_request_id="x y"),))),
             metrics.decode_line(SimpleNamespace(turn=dict(request_id=3, speculative=True, t_first_token=1.0,
                                                           spec_rounds=1, round_tokens=2, decode_tokens=2,
                                                           client_request_id="x y",
                                                           draft_live=dict(draft_proposed=2, draft_accepted=1,
                                                                           spec_steps=1)), after=None), 5.0)]
    assert lines[1].startswith("crucible_decode ") and all("client_request_id" not in line for line in lines)


@pytest.mark.parametrize("batch", [False, True])
def test_a_job_writes_the_id_in_its_turn_line(start_backend, batch):  # noqa: F811
    backend = start_backend(batch, streams=2) if batch else start_backend(False)
    lines = []
    with mock.patch.object(metrics, "emit", lines.append):
        collect(backend.submit([1, 2, 3], request(6), client_request_id=ID))
        collect(backend.submit([1, 2, 3], request(6), client_request_id="not valid"))
        collect(backend.submit([1, 2, 3], request(6)))
    turns = parsed(lines, "crucible_turn")
    assert [turn.get("client_request_id") for turn in turns] == [ID, None, None]
    assert all(set(turns[0]) - {"client_request_id"} == set(turn) for turn in turns[1:] if "request_id" in turn)


def test_the_batch_worker_writes_the_id_in_prefill_and_decode_lines(start_backend):  # noqa: F811
    backend = start_backend(True, streams=2, env=LIVE)
    lines = []
    with mock.patch.object(metrics, "DECODE_LINE_SECONDS", 0.0), mock.patch.object(metrics, "emit", lines.append):
        collect(backend.submit([1, 2, 3, 4], request(10, spec=7), client_request_id=ID))
        collect(backend.submit([1, 2, 3, 4], request(10, spec=7)))
    kinds_seen = ids_of(lines, "crucible_prefill", "crucible_decode", "crucible_turn")
    first, second = kinds_seen[:len(kinds_seen) // 2], kinds_seen[len(kinds_seen) // 2:]
    assert {kind for kind, _ in first} == {"crucible_prefill", "crucible_decode", "crucible_turn"}
    assert all(value == ID for _, value in first) and all(value is None for _, value in second)


def test_a_closed_think_continuation_carries_the_id_on_both_jobs(start_backend):  # noqa: F811
    backend = start_backend(True, streams=2, env=dict(LIVE, FAKE_CHECKPOINTS="1", FAKE_DELAY=".002"))
    lines, cap = [], 5
    with mock.patch.object(metrics, "DECODE_LINE_SECONDS", 0.0), mock.patch.object(metrics, "emit", lines.append):
        first = backend.submit([1, 2, 3, 4], request(40, spec=7), log_turn=False, think_cap=cap, client_request_id=ID)
        seen = 0
        while seen < cap:
            seen += first.events.get(timeout=3)["event"] == "token"
        first.turn["closed_count"] = cap + 3
        cont = backend.continue_job(first, [1, 2, 3, 4, 5, 6], request(6, spec=7), log_turn=False, client_request_id=ID)
        _, cont_event = collect(cont)
        metrics.emit_turn(metrics.request_line(first.turn, dict(prompt_tokens=4, completion_tokens=20, finish_reason="stop"),
                                               SimpleNamespace(tokens=cap, by="answer_room"), cont.turn, cont_event),
                          first.turn)
    written = ids_of(lines, "crucible_prefill", "crucible_decode", "crucible_turn")
    assert {kind for kind, _ in written} == {"crucible_prefill", "crucible_decode", "crucible_turn"}
    assert all(value == ID for _, value in written), written
    prefill = parsed(lines, "crucible_prefill")
    assert any("turn_request_id" in line for line in prefill)  # the continuation's own lines carry both ids


def served(backend, headers=None, **fields):
    extra = [("Host", "127.0.0.1:8790"), ("Content-Type", "application/json")]
    body = json.dumps(chat(**fields)).encode()
    return invoke(backend, PiecesTokenizer(["hi"]), raw=body, headers=extra + [("Content-Length", str(len(body)))] +
                  (headers or []))


@pytest.mark.parametrize("name", ["X-Client-Request-Id", "x-client-request-id", "X-CLIENT-REQUEST-ID"])
def test_a_valid_header_reaches_the_submit_options_and_no_reply(name):
    backend = ScriptBackend(token_events(["hi"]))
    for stream in (False, True):
        response = served(backend, [(name, ID)], stream=stream)
        assert response.statuses == [200]
        assert ID.encode() not in response.body
        assert ID not in "".join(f"{key}{value}" for key, value in response.headers.items())
    assert [options["client_request_id"] for _, _, options in backend.submissions] == [ID, ID]


@pytest.mark.parametrize("value", INVALID)
def test_an_invalid_header_changes_nothing(value):
    backend = ScriptBackend(token_events(["hi"]))
    served(backend, [("X-Client-Request-Id", value)])
    served(backend)
    assert all("client_request_id" not in options for _, _, options in backend.submissions) and backend.submissions


def test_error_and_cancel_lines_carry_the_id_and_the_reply_does_not():
    lines = []
    with mock.patch.object(metrics, "emit", lines.append):
        bad = served(ScriptBackend([]), [("X-Client-Request-Id", ID)], max_tokens=-1)
        none = served(ScriptBackend([]), max_tokens=-1)
        # The client goes away after the stream headers: the handler writes the crucible_cancel line.
        body = json.dumps(chat(stream=True))
        base = [("Host", "127.0.0.1:8790"), ("Content-Length", str(len(body))), ("Content-Type", "application/json")]
        for extra in ([("X-Client-Request-Id", ID)], []):
            invoke(ScriptBackend(token_events(["hi"])), PiecesTokenizer(["hi"]), raw=body.encode(),
                   headers=base + extra, writer=DisconnectWriter(1, BrokenPipeError))
    assert bad.statuses == none.statuses == [400] and bad.body == none.body and ID.encode() not in bad.body
    errors = ids_of(lines, "crucible_error")
    assert errors == [("crucible_error", ID), ("crucible_error", None)]
    assert ids_of(lines, "crucible_cancel") == [("crucible_cancel", ID), ("crucible_cancel", None)]


@pytest.fixture
def live_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Service(ScriptBackend(token_events(["hi"])),
                                                                       PiecesTokenizer(["hi"]))))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def wire(port, head):
    client = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        client.sendall(head)
        reply = b""
        while chunk := client.recv(65536):
            reply += chunk
        return reply
    finally:
        client.close()


def test_a_wire_refusal_writes_the_id_and_does_not_echo_it(live_server, capsys):
    head = lambda extra: (b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"  # noqa: E731
                          b"Content-Length: 2\r\n" + extra + b"\r\n{}")
    for extra in (b"x-client-request-id: " + ID.encode() + b"\r\n", b"X-Client-Request-Id: bad id\r\n", b""):
        reply = wire(live_server, head(extra))
        assert reply.startswith(b"HTTP/1.1 400 ") and ID.encode() not in reply
    # A protocol error after the headers parsed (501) also names the request.
    reply = wire(live_server, b"PUT /v1/models HTTP/1.1\r\nHost: x\r\nX-Client-Request-Id: " + ID.encode() + b"\r\n\r\n")
    assert reply.startswith(b"HTTP/1.1 501 ") and ID.encode() not in reply
    err = capsys.readouterr().err
    errors = [json.loads(line.split(" ", 1)[1]) for line in err.splitlines() if line.startswith("crucible_error ")]
    assert [line.get("client_request_id") for line in errors] == [ID, None, None, ID]


def test_the_continuation_job_of_a_closed_think_turn_gets_the_id():
    backend = LinkedBackend(([REASON] * 1500, "length", 0), ([ANSWER, IM_END], "stop", 9))
    body = json.dumps(think_chat(max_tokens=2048)).encode()
    headers = [("Host", "127.0.0.1:8790"), ("Content-Type", "application/json"), ("Content-Length", str(len(body))),
               ("X-Client-Request-Id", ID)]
    response = invoke(backend, BudgetTokenizer(), raw=body, headers=headers)
    assert len(backend.submissions) == 2 and response.statuses == [200]
    assert [options["client_request_id"] for _, _, options in backend.submissions] == [ID, ID]
    assert ID.encode() not in response.body
