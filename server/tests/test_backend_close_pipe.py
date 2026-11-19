"""close() after the engine exited first: a write that met the closed pipe leaves its line in the stdin buffer.

The flake (test-server gates of Nov 16-17): teardown sets closed, the engine exits on SIGTERM, the batch worker
writes one more command (BrokenPipeError, the line stays in the BufferedWriter), and the last stdin.close() in
close() flushes that line again and raised BrokenPipeError. The fake engine is the real batch fixture
(fake_batch_engine.py, FAKE_EXIT_AFTER_READY=0: exit code 1 right after its ready event)."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

from server.batch_backend import BatchedEngineBackend

FAKE = Path(__file__).parent / "fake_batch_engine.py"


@pytest.fixture
def exited_backend():
    """A batch backend whose engine has exited (code 1) after its ready event; the worker is idle."""
    popen, made = subprocess.Popen, []
    environment = dict(FAKE_EXIT_AFTER_READY="0", FAKE_PING="1", FAKE_SLOT_CONTEXTS="128,128")
    with patch.dict(os.environ, environment), \
            patch("server.backend.subprocess.Popen", side_effect=lambda *a, **k: popen([sys.executable, str(FAKE)], **k)):
        backend = BatchedEngineBackend("unused", "unused", context=128, streams=2)
    made.append(backend)
    assert backend.process.wait(timeout=10) == 1
    yield backend
    for backend in made:
        backend.close()
        backend.worker.join(timeout=5)
        backend.reader.join(timeout=5)
        assert not backend.worker.is_alive() and not backend.reader.is_alive()


def exit_lines(text):
    return [json.loads(line.split(" ", 1)[1]) for line in text.splitlines() if line.startswith("crucible_engine_exit ")]


def test_close_after_a_write_met_the_closed_pipe_does_not_raise(exited_backend, capsys):
    backend = exited_backend
    with pytest.raises(BrokenPipeError):
        backend._ping()  # the worker's own write path: in normal use the error stays an error
    backend.close()
    backend.close()  # the fixture teardown closes again: no second line
    assert backend.process.stdin.closed and backend.process.stdout.closed
    assert exit_lines(capsys.readouterr().err) == [dict(code=1, unsent="stdin")]


def test_worker_write_after_engine_exit_fails_the_backend_and_closes_cleanly(exited_backend, capsys):
    backend = exited_backend
    failures = []
    hook, threading.excepthook = threading.excepthook, failures.append
    try:
        backend.ping_idle_seconds = 0  # the idle worker pings now, after the engine exited
        backend.worker.join(timeout=10)
    finally:
        threading.excepthook = hook
    assert not backend.worker.is_alive()
    assert failures == []  # before the fix, the worker's own close() raised BrokenPipeError out of the thread
    assert backend.fatal and backend.closed.is_set()
    err = capsys.readouterr().err
    worker = [json.loads(line.split(" ", 1)[1]) for line in err.splitlines() if line.startswith("crucible_worker_error ")]
    assert [line["type"] for line in worker] == ["BrokenPipeError"]
    assert exit_lines(err) == [dict(code=1, unsent="stdin")]
    assert backend.process.stdout.closed


def test_another_stdin_close_error_still_raises(exited_backend, capsys):
    backend = exited_backend
    real = backend.process.stdin

    class FailingStdin:
        closed = False
        def close(self):
            real.close()
            raise OSError(errno.EIO, "Input/output error")

    backend.process.stdin = FailingStdin()
    try:
        with pytest.raises(OSError) as raised:
            backend.close()
        assert raised.value.errno == errno.EIO
        assert exit_lines(capsys.readouterr().err) == []
    finally:
        backend.process.stdin = real
