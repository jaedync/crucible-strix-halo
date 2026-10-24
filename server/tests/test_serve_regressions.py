"""Serve bug regressions and the fail-closed native error boundary."""
from concurrent.futures import ThreadPoolExecutor
import json
import queue
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

import server.__main__ as startup
from server.tests.test_backend_paths import collect, request, start_backend, wait_for
from server.tests.serve_fixtures import DisconnectWriter, PiecesTokenizer, ScriptBackend, chat, invoke


def test_batch_request_error_does_not_kill_live_peer_or_worker(start_backend):
    backend = start_backend(True, streams=2, env={"FAKE_REJECT_TOKEN": "777", "FAKE_DELAY": ".002"})
    peer = backend.submit([1], request(80))
    first = peer.events.get(timeout=3)
    assert first["event"] == "token"
    bad = backend.submit([777], request())
    assert bad.events.get(timeout=3)["event"] == "error"
    assert collect(peer, prefix=[first["id"]])[0] == list(range(2, 82))
    assert not backend.fatal
    assert collect(backend.submit([9], request()))[0] == list(range(10, 18))


@pytest.mark.parametrize("mode", ["unsafe", "missing-field", "integer-field", "opened-first", "duplicate-error", "missing-boundary", "wrong-boundary"])
def test_batch_open_error_without_a_safe_boundary_stops_all_peers(start_backend, mode):
    from server.api import ApiError
    backend = start_backend(True, streams=2, env={"FAKE_REJECT_TOKEN": "777", "FAKE_REJECTION_MODE": mode, "FAKE_DELAY": ".002"})
    peer = backend.submit([1], request(80))
    assert peer.events.get(timeout=3)["event"] == "token"
    bad = backend.submit([777], request())
    assert bad.events.get(timeout=3)["event"] == "error"
    events = []
    while True:
        event = peer.events.get(timeout=3)
        events.append(event)
        if event["event"] == "error": break
    wait_for(lambda: backend.closed.is_set())
    assert backend.fatal
    assert all(event["event"] != "done" for event in events)
    with pytest.raises(ApiError) as error: backend.submit([9], request())
    assert error.value.status == 503


# With G6d, two decoding sessions share one packed verify_step, so the plain step and solo spec_step
# ops do not run while the peer decodes.
@pytest.mark.parametrize("op", ["prefill", "verify_step"])
def test_non_open_error_stays_fatal_even_if_marked_recoverable(start_backend, op):
    backend = start_backend(True, streams=2, env={"FAKE_REJECT_TOKEN": "777", "FAKE_ERROR_OP": op, "FAKE_DELAY": ".002"})
    peer = backend.submit([1], request(80))
    assert peer.events.get(timeout=3)["event"] == "token"
    bad = backend.submit([777], request(spec=7 if op == "spec_step" else 0))
    while bad.events.get(timeout=3)["event"] != "error": pass
    wait_for(lambda: backend.closed.is_set())
    assert backend.fatal
    wait_for(lambda: backend.process.poll() is not None)


def test_a_failed_prompt_end_after_the_first_token_ends_the_request_with_an_error(start_backend):
    # Lever 3: the engine publishes the first token, then the prompt end fails; its error event follows the
    # boundary and the engine stops. The client gets the token, then an error, never a done event.
    backend = start_backend(True, streams=2, env={"FAKE_SETTLE_ERROR": "1"})
    job = backend.submit([5], request(8))
    events = [job.events.get(timeout=3)]
    while events[-1]["event"] not in ("error", "done"): events.append(job.events.get(timeout=3))
    assert events[0]["event"] == "token" and events[-1]["event"] == "error"
    wait_for(lambda: backend.closed.is_set())
    assert backend.fatal


def test_serial_cli_explicit_zero_prefill_reaches_engine(start_backend):
    listener, arguments = MagicMock(), []
    listener.serve_forever.side_effect = KeyboardInterrupt

    def backend(executable, checkpoint, context, cache, prefill, **options):
        engine = start_backend(context=context, conversation_cache_mib=cache, prefill=prefill)
        arguments.extend(engine.test_commands[0])
        return engine

    with patch.object(sys, "argv", ["crucible", "checkpoint", "--serial", "--ctx", "32", "--prefill", "0"]), \
            patch.object(startup.signal, "signal"), patch.object(startup, "ModelTokenizer"), \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
            patch.object(startup, "EngineBackend", side_effect=backend):
        assert startup.main() == 0
    assert "--prefill" in arguments
    assert arguments[arguments.index("--prefill") + 1] == "0"


@pytest.mark.parametrize("batch", [False, True])
def test_shutdown_delivers_terminal_error_to_waiting_turn(start_backend, batch):
    env = {"FAKE_CONTEXT": "128", "FAKE_SLOT_CONTEXTS": "128", "FAKE_DELAY": ".003"}
    backend = start_backend(batch, context=128, env=env, **({"streams": 1} if batch else {}))
    live = backend.submit([1], request(80))
    wait_for(lambda: live.events.qsize() == (30 if batch else 32))
    waiting = backend.submit([9], request())
    backend.close()
    event = waiting.events.get(timeout=.1)
    assert event["event"] == "error"
    assert not backend.worker.is_alive()


@pytest.mark.parametrize("batch", [False, True])
def test_shutdown_owns_full_active_waiting_and_queued_jobs(start_backend, batch):
    slots = 2 if batch else 1
    backend = start_backend(batch, context=128, env={"FAKE_CONTEXT": "128", "FAKE_SLOT_CONTEXTS": "128,128"},
                            **({"streams": slots} if batch else {}))
    live = [backend.submit([1 + index], request(80)) for index in range(slots)]
    wait_for(lambda: all(job.events.qsize() == (30 if batch else 32) for job in live))
    waiting = [backend.submit([9], request()) for _ in range(64 if batch else 8)]
    if batch: wait_for(lambda: len(backend.waiting) == 64)
    queued = [backend.submit([20], request()) for _ in range(3)] if batch else []
    backend.close()
    for job in live + waiting + queued:
        events = []
        while not job.events.empty(): events.append(job.events.get_nowait())
        assert events[-1]["event"] == "error"
        assert sum(event["event"] == "error" for event in events) == 1
    assert not backend.worker.is_alive()
    if batch:
        assert backend.pending_tokens == 0
        assert not backend.active and not backend.waiting and backend.jobs.empty()
    backend.close()
    assert all(job.events.empty() for job in live + waiting + queued)


@pytest.mark.parametrize("batch", [False, True])
def test_shutdown_does_not_replace_a_completed_turn(start_backend, batch):
    backend = start_backend(batch, **({"streams": 2} if batch else {}))
    job = backend.submit([1], request(1))
    wait_for(lambda: job.events.qsize() == 2)
    backend.close()
    assert [job.events.get_nowait()["event"] for _ in range(2)] == ["token", "done"]
    assert job.events.empty()


@pytest.mark.parametrize("batch", [False, True])
def test_submit_racing_shutdown_never_accepts_an_orphan_job(start_backend, batch):
    from server.api import ApiError
    from server.backend import Job
    backend = start_backend(batch, **({"streams": 2} if batch else {}))
    entered, resume = threading.Event(), threading.Event()

    def construct(*args, **kwargs):
        entered.set()
        assert resume.wait(timeout=3)
        return Job(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool, patch("server.backend.Job", side_effect=construct):
        submitting = pool.submit(backend.submit, [1], request())
        assert entered.wait(timeout=3)
        closing = pool.submit(backend.close)
        try:
            assert backend.closed.wait(timeout=3)
        finally:
            resume.set()
        with pytest.raises(ApiError) as error: submitting.result(timeout=3)
        assert error.value.status == 503
        closing.result(timeout=3)
    assert not backend.owned_jobs


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError, TimeoutError])
def test_disconnect_during_get_error_output_is_handled(error):
    backend = ScriptBackend([])
    response = invoke(backend, PiecesTokenizer([]), path="/missing", method="GET", writer=DisconnectWriter(0, error))
    assert response.handler.close_connection
    assert not backend.jobs


@pytest.mark.parametrize("position", ["before", "during"])
@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError, TimeoutError])
def test_disconnect_while_reporting_engine_error_is_handled(position, error):
    rejection = {"event": "error", "error": "fixture request rejected"}
    source = [rejection] if position == "before" else [{"event": "token", "id": 20}, rejection]
    backend = ScriptBackend(source)
    writer = DisconnectWriter(0 if position == "before" else 2, error)
    response = invoke(backend, PiecesTokenizer(["hello"]), chat(stream=True), writer=writer)
    assert response.handler.close_connection
    assert backend.jobs[0].cancelled.is_set()


def engine_lines(capsys):
    import json
    return [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().err.splitlines()
            if line.startswith("crucible_engine_error ")]


@pytest.mark.parametrize("mode,recoverable", [("safe", True), ("unsafe", False)])
def test_batch_engine_error_text_is_logged_with_the_request_id(start_backend, capsys, mode, recoverable):
    # a 503 storm gave no cause. The engine's own error text goes to the server log, never to the client.
    text = "prompt token IDs must be in [0,248320)"
    backend = start_backend(True, streams=2, env={"FAKE_REJECT_TOKEN": "777", "FAKE_REJECTION_MODE": mode,
                                                  "FAKE_REJECT_TEXT": text})
    bad = backend.submit([777], request())
    event = bad.events.get(timeout=3)
    assert event["event"] == "error" and text not in event["error"]
    if not recoverable:
        wait_for(lambda: backend.closed.is_set())
    lines = engine_lines(capsys)
    assert lines and lines[0] == dict(request_id=1, op="open", error=text, recoverable=recoverable)


CANARY = "CANARY-serve-api-5519"


@pytest.mark.parametrize("batch", [False, True])
def test_engine_error_logs_never_carry_unknown_text(start_backend, capsys, batch):
    # RA-13 and RA-14: an engine text that is not one of the engine's own fixed messages is logged as "other",
    # and the client always gets a fixed text.
    backend = start_backend(batch, env={"FAKE_REJECT_TOKEN": "777", "FAKE_REJECT_TEXT": CANARY + " near a quote"},
                            **(dict(streams=2) if batch else {}))
    event = backend.submit([777], request()).events.get(timeout=3)
    assert event["event"] == "error" and CANARY not in event["error"]
    err = capsys.readouterr().err
    assert CANARY not in err
    lines = [json.loads(line.split(" ", 1)[1]) for line in err.splitlines() if line.startswith("crucible_engine_error ")]
    assert len(lines) == 1 and lines[0]["error"] == "other" and lines[0]["op"] == ("open" if batch else "generate")
    # A developer can match repeats and check a candidate text without the text in the log.
    import hashlib
    raw = (CANARY + " near a quote").encode()
    assert lines[0]["error_bytes"] == len(raw) and lines[0]["error_sha256"] == hashlib.sha256(raw).hexdigest()[:16]
    assert event["error"] == ("The engine rejected this request. "
                              "The crucible_engine_error line in the server log has the cause.")


@pytest.mark.parametrize("batch", [False, True])
def test_log_engine_errors_logs_the_raw_text_when_asked(start_backend, capsys, batch):
    from server import metrics
    with patch.object(metrics, "LOG_ENGINE_TEXT", True):
        backend = start_backend(batch, env={"FAKE_REJECT_TOKEN": "777", "FAKE_REJECT_TEXT": CANARY},
                                **(dict(streams=2) if batch else {}))
        backend.submit([777], request()).events.get(timeout=3)
        lines = engine_lines(capsys)
    assert len(lines) == 1 and lines[0]["error"] == "other" and lines[0]["text"] == CANARY


def test_log_engine_errors_flag_is_off_by_default_and_names_the_risk():
    import io, contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out), patch.object(sys, "argv", ["server", "--help"]), pytest.raises(SystemExit):
        startup.main()
    text = " ".join(out.getvalue().split())
    from server import metrics
    assert metrics.LOG_ENGINE_TEXT is False
    assert "--log-engine-errors" in text and "request content" in text


def test_worker_error_line_logs_only_server_messages():
    from server import metrics
    from server.engine_messages import ProtocolError
    assert CANARY not in metrics.worker_error_line(KeyError(CANARY))
    assert json.loads(metrics.worker_error_line(KeyError(CANARY)).split(" ", 1)[1]) == dict(type="KeyError", error="other",
                                                                                           at="")
    def deep():
        {}[CANARY]
    try:
        deep()
    except KeyError as error:
        caught = error
    fields = json.loads(metrics.worker_error_line(caught).split(" ", 1)[1])
    assert CANARY not in json.dumps(fields) and fields["error"] == "other"
    assert fields["at"].endswith("test_serve_regressions.py:%d:deep" % (deep.__code__.co_firstlineno + 1))
    line = metrics.worker_error_line(ProtocolError("The engine returned the wrong batch boundary."))
    assert "wrong batch boundary" in line


def test_engine_error_text_keeps_known_messages_and_short_keys():
    from server.engine_messages import loggable
    assert loggable("prompt token IDs must be in [0,248320)") == "prompt token IDs must be in [0,248320)"
    assert loggable("duplicate request field: max_tokens") == "duplicate request field: max_tokens"
    assert loggable("duplicate request field: " + CANARY + " x") == "other"
    # A suffix that is a number could be a token ID.
    assert loggable("duplicate request field: 248044") == "other"
    assert loggable(None) == "other" and loggable("x" * 100000) == "other"


def test_engine_messages_cover_the_serve_path():
    # Every fixed message that the serve engines can send is known; a new one fails here until it is added.
    import re
    from pathlib import Path
    from server.engine_messages import KNOWN, PREFIXES
    # Repo paths, so that a layout move rewrites them with the files.
    root = Path(__file__).parents[2]
    engine = root / "engine"
    pattern = re.compile(r'(?:runtime_error|invalid_argument|logic_error|out_of_range|overflow_error|length_error|'
                         r'RecoverableOpenError|ServeTenantError)\("([^"\\]*)"')
    serve = ("engine/serve/serve.h", "engine/serve/serve_protocol.h", "engine/serve/batched_serve.h", "engine/serve/batched_serve_protocol.h",
             "engine/serve/batch_session.h", "engine/runtime/conversation_admission.h", "engine/model/flashnext/multimodal.h", "engine/model/flashnext/vision.h",
             "engine/serve/serve_settings.h", "engine/serve/batched_serve_settings.h", "engine/engine.hip",
             "engine/runtime/draft_policy.h")
    # throw std::runtime_error(flag ? "text" : "text"), as the line limit with and without --vision.
    choice = re.compile(r'runtime_error\([^"();]*\?\s*"([^"\\]*)"\s*:\s*"([^"\\]*)"')
    def messages(text):
        return set(pattern.findall(text)) | {message for pair in choice.findall(text) for message in pair}
    found = {message for name in serve for message in messages((root / name).read_text())}
    # Validators return a message that a caller throws (serve_protocol.h throws sampling.h's Error()).
    returned = re.compile(r'return "([^"\\]* [^"\\]*)";')
    validators = ("engine/runtime/sampling.h", "engine/runtime/batch_contract.h", "engine/runtime/batch_admission.h",
                  "engine/runtime/batch_spec_contract.h", "engine/runtime/retained_prefix.h", "engine/serve/batched_serve.h")
    found |= {message for name in validators for message in returned.findall((root / name).read_text())}
    assert "temperature must be finite and nonnegative" in found
    # The suffix of this text is a request key (any text), so it stays "other" and is neither known nor a prefix.
    unlogged = {"duplicate image field: "}
    assert found - KNOWN - set(PREFIXES) == unlogged
    everything = {message for path in [*engine.rglob("*.h"), engine / "engine.hip"]
                  for message in messages(path.read_text()) | set(returned.findall(path.read_text()))}
    assert KNOWN - everything <= {"std::bad_alloc"}


def test_backend_failures_raise_fixed_protocol_errors():
    # RA-14: the worker log shows only ProtocolError texts; a plain RuntimeError is logged as "other". Train 13 merged
    # a decode-interleave raise that predates ProtocolError; this keeps later merges from bringing one back.
    from pathlib import Path
    server = Path(__file__).parents[1]
    for name in ("backend.py", "batch_backend.py"):
        assert "raise RuntimeError(" not in (server / name).read_text(), name


def test_batch_worker_failure_logs_its_cause(start_backend, capsys):
    backend = start_backend(True, streams=2, env={"FAKE_REJECT_TOKEN": "777", "FAKE_REJECTION_MODE": "wrong-boundary"})
    backend.submit([777], request()).events.get(timeout=3)
    wait_for(lambda: backend.closed.is_set())
    worker = [line for line in capsys.readouterr().err.splitlines() if line.startswith("crucible_worker_error ")]
    assert len(worker) == 1 and "wrong batch boundary" in worker[0]


@pytest.mark.parametrize("batch", [False, True])
def test_a_request_above_the_engine_line_limit_fails_alone(start_backend, batch):
    # RA-2: the engine rejects an oversized line before it knows the op, so batch serve stopped for every client.
    from server import backend as backend_module
    from server.api import ApiError
    with patch.object(backend_module, "ENGINE_LINE_BYTES", 2000):
        env = {"FAKE_LINE_LIMIT": "2000", "FAKE_CONTEXT": "4096", "FAKE_SLOT_CONTEXTS": "4096,4096"}
        backend = start_backend(batch, env=env, context=4096, **(dict(streams=2) if batch else {}))
        peer = backend.submit([1], request(40))
        assert peer.events.get(timeout=3)["event"] == "token"
        with pytest.raises(ApiError) as error:
            backend.submit(list(range(1, 700)), request())
        assert error.value.status == 413 and "too large" in str(error.value)
        assert len(str(error.value)) < 200
        assert not backend.fatal
        assert len(collect(peer, prefix=[2])[0]) == 40
        assert collect(backend.submit([9], request()))[0] == list(range(10, 18))


def test_the_engine_line_limit_matches_the_engine_source():
    # One limit for text and image lines: image pixels go in private files, not in the line.
    from pathlib import Path
    from server.backend import ENGINE_LINE_BYTES
    source = (Path(__file__).parents[2] / "engine/serve/serve.h").read_text()
    assert "constexpr size_t kMaxLine = 2 * 1024 * 1024;" in source and ENGINE_LINE_BYTES == 2 * 1024 * 1024
    assert "kServeVisionLine" not in source


@pytest.mark.parametrize("batch", [False, True])
def test_a_full_size_image_passes_the_line_check_and_large_text_still_gets_413(start_backend, batch):
    # Train 13 integration: one 2560x1440 image reaches a --vision 1 engine (the image-file change: as a file, so the
    # line stays small); a text request over 2 MiB still gets 413 before it reaches the engine.
    from dataclasses import replace
    from server.api import ApiError
    from server.vision import VisionImage, image_digest
    rgb = bytes(2560 * 1440 * 3)
    # The fake engines check the digest, as the engine does.
    image = VisionImage(height=1440, width=2560, rgb=rgb, digest=image_digest(1440, 2560, rgb))
    ids = [1, 2] + [248056] * image.tokens + [3]
    context = 8192
    env = {"FAKE_LINE_LIMIT": str(2 << 20), "FAKE_CONTEXT": str(context),
           "FAKE_SLOT_CONTEXTS": f"{context},{context}"}
    backend = start_backend(batch, env=env, context=context, engine_options=("--vision", "1"),
                            **(dict(streams=2) if batch else {}))
    tokens, done = collect(backend.submit(ids, replace(request(), images=(image,)), image_offsets=[2]))
    assert len(tokens) == 8 and not backend.fatal
    if batch:
        return
    text = start_backend(False, env={"FAKE_CONTEXT": "400000"}, context=400000)
    with pytest.raises(ApiError) as error:
        text.submit([248000] * 300_000, request())  # about 2.1 MB of IDs
    assert error.value.status == 413 and not text.fatal


@pytest.mark.parametrize("batch", [False, True])
def test_a_vision_engine_has_the_text_line_limit(start_backend, batch):
    # serve --vision 1 reads the same 2 MiB lines as text serve, so a long prompt gets 413 with vision on too.
    from server import backend as backend_module
    from server.api import ApiError
    with patch.object(backend_module, "ENGINE_LINE_BYTES", 4000):
        env = {"FAKE_LINE_LIMIT": "4000", "FAKE_CONTEXT": "4096", "FAKE_SLOT_CONTEXTS": "4096,4096"}
        backend = start_backend(batch, env=env, context=4096, engine_options=("--vision", "1"),
                                **(dict(streams=2) if batch else {}))
        ids, done = collect(backend.submit(list(range(1, 300)), request()))
        assert len(ids) == 8 and not backend.fatal
        with pytest.raises(ApiError) as error:
            backend.submit(list(range(1, 1200)), request())
        assert error.value.status == 413 and not backend.fatal


def test_serve_never_passes_capture_switches_to_the_engine(capsys):
    # RA-15 (design decision): content-derived captures are off in serve, whatever the environment says.
    from server.backend import EngineBackend
    seen = {}

    class Stop(Exception):
        pass

    def launch(arguments, **kwargs):
        seen.update(kwargs.get("env") or {})
        raise Stop
    env = {"CRUCIBLE_QSA_CAPTURE_DIR": "/tmp/x", "CRUCIBLE_QSA_CAPTURE_AT": "3", "CRUCIBLE_PROFILE": "1"}
    with patch.dict("os.environ", env), patch("server.backend.subprocess.Popen", side_effect=launch):
        with pytest.raises(Stop):
            EngineBackend("unused", "unused", 32)
    assert "CRUCIBLE_QSA_CAPTURE_DIR" not in seen and "CRUCIBLE_QSA_CAPTURE_AT" not in seen
    assert seen.get("CRUCIBLE_PROFILE") == "1"
    warnings = [line for line in capsys.readouterr().err.splitlines() if "CRUCIBLE_QSA_CAPTURE" in line]
    assert len(warnings) == 1 and "/tmp/x" not in warnings[0]


def test_serial_slow_reader_cannot_stall_other_clients(start_backend):
    # RA-6: a stream that stops reading filled its queue and held the serial worker. Its output now stops instead.
    from server import backend as backend_module
    import time
    with patch.object(backend_module, "OUTPUT_TIMEOUT_SECONDS", 0.3):
        backend = start_backend(env={"FAKE_DELAY": "0.001", "FAKE_CONTEXT": "128"}, context=128)
        slow = backend.submit([1], request(32 + 40))
        start = time.monotonic()
        ids, _ = collect(backend.submit([9], request()))
        assert ids == list(range(10, 18)) and time.monotonic() - start < 3
        events = []
        while not events or events[-1]["event"] not in ("done", "error"):
            events.append(slow.events.get(timeout=3))
        assert events[-1] == {"event": "error", "error": "The output consumer timed out. Retry the request."}
        assert slow.cancelled.is_set()
