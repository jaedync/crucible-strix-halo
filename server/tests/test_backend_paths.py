"""Resident fake-engine ownership, cancellation, admission, and fault isolation."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from unittest.mock import Mock, patch

import pytest

from server.api import ApiError, validate_request
from server.backend import EngineBackend, Job
from server.batch_backend import BatchedEngineBackend
from server.tests.serve_fixtures import DisconnectWriter, chat, chunks, invoke


@pytest.fixture
def start_backend():
    resources = []

    def start(batch=False, *, env=None, context=None, **options):
        fixture = Path(__file__).parent / ("fake_batch_engine.py" if batch else "fake_engine.py")
        popen, processes, commands = subprocess.Popen, [], []

        def launch(arguments, **kwargs):
            commands.append(list(arguments))
            process = popen([sys.executable, str(fixture)], **kwargs)
            processes.append(process)
            return process

        with patch.dict(os.environ, env or {}), patch("server.backend.subprocess.Popen", side_effect=launch):
            backend = (BatchedEngineBackend if batch else EngineBackend)("unused", "unused", context or (128 if batch else 32), **options)
        jobs, submit = [], backend.submit

        def tracked_submit(*args, **kwargs):
            job = submit(*args, **kwargs)
            jobs.append(job)
            return job

        backend.submit = tracked_submit
        backend.test_commands = commands
        resources.append((backend, jobs))
        return backend

    yield start
    for backend, jobs in reversed(resources):
        for job in jobs: job.cancelled.set()
        backend.close()
        backend.worker.join(timeout=2)
        backend.reader.join(timeout=2)
        assert backend.process.poll() is not None
        assert not backend.worker.is_alive()
        assert not backend.reader.is_alive()


def request(count=8, **options):
    return validate_request({"prompt": "x", "max_tokens": count, "spec": 0, **options}, "completion", "crucible")


def collect(job, *, prefix=()):
    ids = list(prefix)
    while True:
        event = job.events.get(timeout=3)
        assert event["event"] != "error", event
        if event["event"] == "done":
            assert event["completion_tokens"] == len(ids)
            return ids, event
        ids.append(event["id"])


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "fixture did not reach the required state"
        time.sleep(.002)


@pytest.mark.parametrize("batch", [False, True])
def test_cancelled_waiter_never_reaches_engine_and_releases_admission(start_backend, tmp_path, batch):
    log = tmp_path / "commands.jsonl"
    env = {"FAKE_DELAY": ".002", "FAKE_REQUEST_LOG": str(log), "FAKE_COMMAND_LOG": str(log)}
    if batch: env["FAKE_SLOT_CONTEXTS"] = "128"
    else: env["FAKE_CONTEXT"] = "128"
    backend = start_backend(batch, env=env, context=128, **({"streams": 1} if batch else {}))
    first = backend.submit([1], request(80))
    wait_for(lambda: first.events.qsize() == (30 if batch else 32))
    cancelled = backend.submit([50], request())
    cancelled.cancelled.set()
    peer = backend.submit([9], request())
    first.cancelled.set()
    assert collect(peer)[0] == list(range(10, 18))
    records = [json.loads(line) for line in log.read_text().splitlines()]
    payloads = [item["request"] for item in records if item.get("op") == "open"] if batch else records
    assert [payload["ids"] for payload in payloads] == [[1], [9]]
    assert cancelled.events.empty()
    if batch:
        wait_for(lambda: backend.pending_tokens == 0)
        assert backend.free_contexts == [128]
    assert not backend.fatal


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError, TimeoutError])
@pytest.mark.parametrize("batch", [False, True])
def test_midstream_disconnect_keeps_next_session_exact(start_backend, batch, error):
    backend = start_backend(batch, env={"FAKE_DELAY": ".002"}, **({"streams": 2} if batch else {}))
    tokenizer = NumericTokenizer()
    response = invoke(backend, tokenizer, chat(stream=True, max_tokens=8), writer=DisconnectWriter(2, error))
    assert response.statuses == [200]
    assert response.handler.close_connection
    peer = backend.submit([90], request())
    assert collect(peer)[0] == list(range(91, 99))
    assert not backend.fatal


class NumericTokenizer:
    def encode(self, request):
        return [1, 2, 10 + int(request.messages[-1]["content"])] if request.messages[-1]["content"].isdigit() else [10]

    def cache_prefix_tokens(self, request, ids):
        return 2 if len(ids) == 3 else 0

    def shared_prefix_tokens(self, request, ids):
        return self.cache_prefix_tokens(request, ids)

    def stream(self): return None
    def step(self, stream, token): return f"{token},"
    def decode(self, ids): return "".join(f"{token}," for token in ids)


def test_six_concurrent_sessions_cache_metadata_and_slot_reuse(start_backend):
    backend = start_backend(True, streams=4, conversation_cache_mib=1,
                            env={"FAKE_SLOT_CONTEXTS": "128,128,128,128", "FAKE_DELAY": ".001", "FAKE_CACHE": "1"})
    tokenizer = NumericTokenizer()

    def turn(index, control):
        payload = chat(messages=[{"role": "user", "content": str(index)}], stream=bool(index % 2),
                       stream_options={"include_usage": True}, max_tokens=8, spec=0, crucible_metrics=True,
                       return_token_ids=True, cache_control=control)
        response = invoke(backend, tokenizer, payload)
        assert response.statuses == [200]
        if payload["stream"]:
            events = chunks(response)
            ids = [choice["token_id"] for event in events for choice in event["choices"] if "token_id" in choice]
            content = "".join(choice["delta"].get("content", "") for event in events for choice in event["choices"])
            usage = events[-1]["usage"]
            assert len([event for event in events if "crucible_t" in event]) == 8
        else:
            event = json.loads(response.body)
            ids, content, usage = event["choices"][0]["token_ids"], event["choices"][0]["message"]["content"], event["usage"]
        assert ids == list(range(11 + index, 19 + index))
        assert content == "".join(f"{token}," for token in ids)
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (3, 8, 11)
        return ids, usage

    with ThreadPoolExecutor(max_workers=6) as pool:
        groups = []
        for control in ("auto", "auto", "bypass", "refresh"):
            futures = [pool.submit(turn, index, control) for index in range(6)]
            groups.append([future.result(timeout=5) for future in futures])
    assert [item[0] for item in groups[0]] == [item[0] for item in groups[1]] == [item[0] for item in groups[2]]
    for group, cached in ((groups[1], 2), (groups[2], 0), (groups[3], 0)):
        for _, usage in group:
            assert usage["prompt_tokens_details"] == {"cached_tokens": cached}
            assert usage["crucible_metrics"]["restored_tokens"] == cached
            assert usage["crucible_metrics"]["session_label"] == ("hit_shared" if cached else "miss")
    wait_for(lambda: not backend.active)
    assert sorted(backend.free_contexts) == [128] * 4
    assert backend.pending_tokens == 0
    assert not backend.fatal


def test_malformed_model_call_is_isolated_from_concurrent_tool_sessions(start_backend):
    from server.tests.test_tools import TOOLS, ToolOutputTests
    backend = start_backend(True, streams=2, env={"FAKE_DELAY": ".002"})
    block = ToolOutputTests().block()

    class ToolTokenizer(NumericTokenizer):
        def encode(self, request):
            return [19 if request.messages[-1]["content"] == "bad" else 29]
        def step(self, stream, token):
            return {20: "<tool_call>broken</tool_call>", 30: block}[token]
        def decode(self, ids):
            return "".join(self.step(None, token) for token in ids)

    def turn(text):
        response = invoke(backend, ToolTokenizer(), chat(messages=[{"role": "user", "content": text}],
                          tools=TOOLS, spec=0, max_tokens=1))
        body = json.loads(response.body)
        if text == "bad":
            assert response.statuses == [200]
            assert body["choices"][0]["message"] == {"role": "assistant", "content": "<tool_call>broken</tool_call>"}
        else:
            assert response.statuses == [200]
            assert body["choices"][0]["finish_reason"] == "tool_calls"
            assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(turn, text) for text in ("good", "bad", "good", "good")]
        for future in futures: future.result(timeout=5)
    turn("good")
    assert not backend.fatal


def test_decoder_gets_passes_between_long_prompt_chunks(start_backend):
    backend = start_backend(True, streams=2, context=1024, prefill=1,
                            env={"FAKE_SLOT_CONTEXTS": "1024,1024", "FAKE_PREFILL_ROWS": "1", "FAKE_DELAY": ".003"})
    short = backend.submit([10], request())
    long = backend.submit([1] * 200, request(2))
    assert collect(short)[0] == list(range(11, 19))
    assert collect(long)[0] == [2, 3]
    entries = backend.trace.since(0)
    long_id = long.turn["request_id"]
    prefill = [index for index, entry in enumerate(entries) if entry["op"] == "prefill" and entry["request_ids"] == [long_id]]
    assert len(prefill) == 200
    assert any(prefill[0] < index < prefill[-1] for index, entry in enumerate(entries) if entry["op"] == "step")
    assert not backend.fatal


def test_cancel_during_prefill_closes_lease_before_slot_reuse(start_backend):
    backend = start_backend(True, streams=1, prefill=1, env={"FAKE_SLOT_CONTEXTS": "128", "FAKE_PREFILL_ROWS": "1", "FAKE_DELAY": ".002"})
    cancelled = backend.submit([1] * 100, request())
    wait_for(lambda: any(entry["op"] == "prefill" for entry in backend.trace.since(0)))
    cancelled.cancelled.set()
    peer = backend.submit([9], request())
    assert collect(peer)[0] == list(range(10, 18))
    entries = backend.trace.since(0)
    close_index = next(index for index, entry in enumerate(entries) if entry["op"] == "close")
    assert entries[close_index]["request_ids"] == [cancelled.turn["request_id"]]
    assert all(cancelled.turn["request_id"] not in entry["request_ids"] for entry in entries[close_index + 1:])
    assert backend.pending_tokens == 0
    assert not backend.fatal


def test_batch_output_timeout_frees_slow_slot_and_preserves_peer(start_backend):
    backend = start_backend(True, streams=2, output_timeout=.03)
    slow = backend.submit([1], request(80))
    wait_for(lambda: slow.cancelled.is_set())
    events = []
    while not slow.events.empty(): events.append(slow.events.get_nowait())
    assert events[-1]["event"] == "error"
    assert "consumer timed out" in events[-1]["error"]
    assert collect(backend.submit([9], request()))[0] == list(range(10, 18))
    wait_for(lambda: not backend.active)
    assert sorted(backend.free_contexts) == [128, 128]
    assert not backend.fatal


def test_serial_request_rejection_does_not_poison_next_turn(start_backend):
    backend = start_backend(env={"FAKE_REJECT_TOKEN": "777"})
    bad = backend.submit([777], request())
    # RA-13: the client gets a fixed text; the engine text goes to the server log only.
    assert bad.events.get(timeout=3) == {"event": "error", "error": "The engine rejected this request. "
                                                     "The crucible_engine_error line in the server log has the cause."}
    assert collect(backend.submit([9], request()))[0] == list(range(10, 18))
    assert not backend.fatal


def test_serial_metrics_stamp_every_token_and_usage(start_backend):
    backend = start_backend(conversation_cache_mib=8, prefill=16)
    assert backend.test_commands[0][-4:] == ["--conversation-cache-mib", "8", "--prefill", "16"]
    job = backend.submit([1], request(2, crucible_metrics=True))
    events = [job.events.get(timeout=3) for _ in range(3)]
    assert [event["event"] for event in events] == ["token", "token", "done"]
    assert events[0]["crucible_t"] <= events[1]["crucible_t"]
    measured = events[-1]["crucible_metrics"]
    assert measured["t_submit"] <= measured["t_admit"] <= measured["t_first_token"] <= measured["t_done"]
    assert measured["completion_tokens"] == 2


@pytest.mark.parametrize("mode", ["wrong-context", "invalid-json", "oversized", "exit"])
def test_bad_serial_handshake_stops_the_child(mode):
    popen, children = subprocess.Popen, []

    def launch(*args, **kwargs):
        child = popen([sys.executable, str(Path(__file__).parent / "fake_engine.py")], **kwargs)
        children.append(child)
        return child
    with patch.dict(os.environ, FAKE_ENGINE_MODE=mode), patch("server.backend.subprocess.Popen", side_effect=launch):
        with pytest.raises(RuntimeError): EngineBackend("unused", "unused", 32)
    assert len(children) == 1
    assert children[0].poll() is not None


def test_unexpected_serial_protocol_event_is_fatal_and_rejects_later_work(start_backend):
    backend = start_backend(env={"FAKE_ENGINE_MODE": "unexpected-event"})
    event = backend.submit([1], request()).events.get(timeout=3)
    assert event["event"] == "error"
    assert "Restart" in event["error"]
    with pytest.raises(ApiError) as error: backend.submit([9], request())
    assert error.value.status == 503


def test_serial_queue_limits_and_shutdown_reject_without_enqueue():
    backend = EngineBackend.__new__(EngineBackend)
    backend.context, backend.closed, backend.fatal, backend.jobs = 32, threading.Event(), None, queue.Queue(maxsize=1)
    backend.ownership_lock, backend.owned_jobs = threading.Lock(), {}
    backend.submit([1], request())
    with pytest.raises(ApiError, match="queue is full") as error: backend.submit([9], request())
    assert error.value.status == 503
    assert backend.jobs.qsize() == 1
    backend.closed.set()
    with pytest.raises(ApiError, match="not available"): backend.submit([9], request())
    with pytest.raises(ApiError, match="fit"): backend.submit([], request())


def test_next_event_stop_and_timeout_are_bounded():
    backend = EngineBackend.__new__(EngineBackend)
    backend.closed, backend.fatal, backend.engine_events = threading.Event(), None, Mock()
    backend.engine_events.get.side_effect = queue.Empty
    backend.closed.set()
    with pytest.raises(RuntimeError, match="stopped"): backend._next()
    backend.closed.clear()
    backend.engine_events.get.reset_mock()
    with pytest.raises(RuntimeError, match="300 seconds"): backend._next()
    assert backend.engine_events.get.call_count == 3000


def test_close_kills_a_child_that_ignores_termination():
    backend = EngineBackend.__new__(EngineBackend)
    backend.closed, backend.process = threading.Event(), Mock()
    backend.ownership_lock, backend.close_lock, backend.owned_jobs = threading.Lock(), threading.Lock(), {}
    backend.jobs = queue.Queue()
    backend.process.poll.return_value = None
    backend.process.wait.side_effect = [subprocess.TimeoutExpired("fake", 10), 0]
    backend.close()
    backend.process.terminate.assert_called_once()
    backend.process.kill.assert_called_once()
    assert backend.process.wait.call_count == 2
    backend.process.stdin.close.assert_called_once()
    backend.process.stdout.close.assert_called_once()


def test_thinking_turns_guard_their_replay_tail(start_backend):
    """Clients often drop the reasoning, so a thinking turn's tail must not replace the resume point (Oct 6)."""
    backend = start_backend()
    chat_request = lambda thinking: validate_request(
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 2, "enable_thinking": thinking}, "chat",
        "crucible")
    guarded, plain = backend.submit([1], chat_request(True)), backend.submit([2], chat_request(False))
    assert guarded.payload["cache_tail"] == "guard"
    assert "cache_tail" not in plain.payload  # the payload of older requests is unchanged
    for job in (guarded, plain): collect(job)


def test_cache_tail_guard_needs_engine_support(start_backend):
    """Engines before the guard reject unknown fields, so it is sent only when the ready event names it."""
    backend = start_backend(env={"FAKE_NO_CACHE_TAIL": "1"})
    request = validate_request(
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 2, "enable_thinking": True}, "chat", "crucible")
    job = backend.submit([1], request)
    assert "cache_tail" not in job.payload
    collect(job)


def test_slot_resume_option_needs_replay(start_backend):
    with pytest.raises(ValueError):
        BatchedEngineBackend("unused", "unused", 128, slot_resume=0)


def test_slot_checkpoints_option_needs_replay(start_backend):
    with pytest.raises(ValueError):
        BatchedEngineBackend("unused", "unused", 128, slot_checkpoints=0)
    with pytest.raises(ValueError):
        BatchedEngineBackend("unused", "unused", 128, session_replay=True, slot_checkpoints=2)


def test_think_continuation_resumes_from_its_first_job_slot(start_backend):
    """A closed think block continues on a second job (http.py close_thinking). That job must open on the first
    job's slot before any waiting peer, so it resumes from the first job's end state whatever the timing: a peer on
    that slot would replace the state, and another restore point gives other bits."""
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128", "FAKE_CHECKPOINTS": "1",
                                                  "FAKE_DELAY": ".002"})
    first = backend.submit([1, 2, 3, 4], request(40))
    assert first.events.get(timeout=3)["event"] == "token"
    peer = backend.submit([9, 9, 9], request(40))
    wait_for(lambda: peer.turn.get("request_id") is not None)
    waiting = backend.submit([7, 7, 7], request(4))  # no free slot: it waits, ahead of the continuation
    cont = backend.continue_job(first, [1, 2, 3, 4, 5, 6], request(4))
    collect(cont)
    assert cont.turn["opened_cached"] == 4
    collect(waiting)


def test_identical_resend_reports_the_whole_prompt_restored(start_backend):
    """Re-send fix A: an End resume opens with cached_tokens equal to the prompt length. The server keeps
    that count (SRPT ranks no rows left), and the prefill command with no rows publishes the first token."""
    backend = start_backend(True, streams=1, env={"FAKE_SLOT_CONTEXTS": "128", "FAKE_CHECKPOINTS": "1",
                                                  "FAKE_END_RESUME": "1", "FAKE_DELAY": ".002"})
    cold, _ = collect(backend.submit([1, 2, 3, 4], request(4)))
    again = backend.submit([1, 2, 3, 4], request(4))
    ids, _ = collect(again)
    assert ids == cold and again.turn["opened_cached"] == 4
    entry_rows = [entry["rows"] for entry in backend.trace.since(0) if entry["op"] == "prefill" and
                  entry["request_ids"] == [again.turn["request_id"]]]
    assert len(entry_rows) == 1


def test_think_continuation_waits_for_its_first_job_to_close(start_backend):
    """continue_job queues the continuation before it cancels the first job. A free slot must not take the
    continuation in between: it would open without the first job's end state."""
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128", "FAKE_CHECKPOINTS": "1",
                                                  "FAKE_DELAY": ".002"})
    first = backend.submit([1, 2, 3, 4], request(40))
    assert first.events.get(timeout=3)["event"] == "token"
    cont = backend.submit([1, 2, 3, 4, 5, 6], request(4), after=first)
    time.sleep(.1)
    assert cont.turn.get("request_id") is None
    first.cancelled.set()
    collect(cont)
    assert cont.turn["opened_cached"] == 4


def test_kept_slot_does_not_stall_peers_on_other_slots(start_backend):
    """A first job whose continuation never queues keeps only its own slot: a peer opens on a free slot at once."""
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128", "FAKE_CHECKPOINTS": "1",
                                                  "FAKE_DELAY": ".002"})
    first = backend.submit([1, 2, 3, 4], request(40))
    assert first.events.get(timeout=3)["event"] == "token"
    first.successor = Job(payload=dict(ids=[1, 2, 3, 4, 5]))  # linked, never submitted
    first.cancelled.set()
    peer = backend.submit([9, 9, 9], request(4))
    collect(peer)
    assert peer.turn["t_admit"] - peer.turn["t_submit"] < 1.0


def test_think_continuation_releases_its_slot_when_the_client_leaves(start_backend):
    backend = start_backend(True, streams=1, env={"FAKE_SLOT_CONTEXTS": "128", "FAKE_CHECKPOINTS": "1",
                                                  "FAKE_DELAY": ".002"})
    first = backend.submit([1, 2, 3, 4], request(40))
    assert first.events.get(timeout=3)["event"] == "token"
    cont = backend.continue_job(first, [1, 2, 3, 4, 5, 6], request(4))
    cont.cancelled.set()
    later = backend.submit([7, 7, 7], request(4))
    collect(later)


def test_think_cap_token_holds_peer_chunks_for_the_continuation(start_backend):
    """A first job that reached its think cap gets its continuation from the HTTP thread a moment later. A peer's
    chunk must not start in between: in a 4-agent test run that chunk delayed 40 of 105 continuations by about 3 s
    (decode gaps)."""
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128", "FAKE_CHECKPOINTS": "1",
                                                  "FAKE_DELAY": ".002", "FAKE_PREFILL_ROWS": "4"})
    peer = backend.submit([9] * 100, request(4))
    wait_for(lambda: peer.turn.get("request_id") is not None)
    first = backend.submit([1, 2, 3], request(40), think_cap=1)
    assert first.events.get(timeout=3)["event"] == "token"
    token_at = time.monotonic()
    time.sleep(.05)  # the HTTP thread writes the close text before it queues the continuation
    cont = backend.continue_job(first, [1, 2, 3, 4, 5], request(4))
    collect(cont)
    peer_chunks = [entry for entry in backend.trace.since(token_at) if entry["op"] == "prefill" and
                   entry["request_ids"] == [peer.turn["request_id"]] and entry["t_start"] < cont.turn["t_admit"]]
    assert peer_chunks == []
    collect(peer)


def test_think_cap_hold_ends_when_no_continuation_comes(start_backend):
    backend = start_backend(True, streams=2, env={"FAKE_SLOT_CONTEXTS": "128,128", "FAKE_CHECKPOINTS": "1",
                                                  "FAKE_DELAY": ".002", "FAKE_PREFILL_ROWS": "4"})
    first = backend.submit([1, 2, 3], request(40), think_cap=1)
    assert first.events.get(timeout=3)["event"] == "token"
    peer = backend.submit([9] * 20, request(4))
    start = time.monotonic()
    collect(peer)
    assert time.monotonic() - start < 2.0


def test_think_cap_hold_still_steps_ready_decoders(start_backend):
    """While a new turn's chunk waits for a continuation, ready decoders step at once instead of waiting for the
    decode bound."""
    backend = start_backend(True, streams=3, env={"FAKE_SLOT_CONTEXTS": "128,128,128", "FAKE_CHECKPOINTS": "1",
                                                  "FAKE_DELAY": ".002", "FAKE_PREFILL_ROWS": "4"})
    decoder = backend.submit([5, 6, 7], request(100))
    assert decoder.events.get(timeout=3)["event"] == "token"
    stop = threading.Event()
    def consume():
        while not stop.is_set():
            try: decoder.events.get(timeout=.05)
            except queue.Empty: pass
    drain = threading.Thread(target=consume)
    drain.start()
    peer = backend.submit([9] * 100, request(4))
    wait_for(lambda: peer.turn.get("request_id") is not None)
    first = backend.submit([1, 2, 3], request(40), think_cap=1)
    assert first.events.get(timeout=3)["event"] == "token"
    token_at = time.monotonic()
    time.sleep(.1)
    cont = backend.continue_job(first, [1, 2, 3, 4, 5], request(4))
    collect(cont)
    steps = [entry for entry in backend.trace.since(token_at) if decoder.turn["request_id"] in entry["request_ids"]
             and entry["op"] in ("step", "verify_step") and entry["t_start"] < token_at + .1]
    stop.set(); drain.join(timeout=5); decoder.cancelled.set()
    assert len(steps) >= 5, len(steps)
    collect(peer)
