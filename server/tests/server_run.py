"""A real server process with the fake batch engine, for the stop and engine-death tests. No GPU."""
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAKE = ROOT / "server/tests/fake_batch_engine.py"
sys.path.insert(0, str(ROOT))

from server.tests.test_content_tokens import synthetic_checkpoint  # noqa: E402


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def health(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1) as sock:
            sock.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            return sock.recv(64).split(b" ")[1].decode()
    except (OSError, IndexError):
        return ""


class ServerRun:
    def __init__(self, test, env=None, args=(), streams=2, ctx=512):
        self.tmp = Path(tempfile.mkdtemp(prefix="engine-exit-"))
        test.addCleanup(self.cleanup)
        ckpt = self.tmp / "ckpt"
        ckpt.mkdir()
        synthetic_checkpoint(ckpt)
        engine = self.tmp / "engine.sh"
        engine.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
        engine.chmod(0o755)
        self.port = free_port()
        self.log = self.tmp / "server.log"
        self.pid_file = self.tmp / "engine.pid"
        argv = [sys.executable, "-m", "server", str(ckpt), "--engine", str(engine), "--ctx", str(ctx), "--streams",
                str(streams), "--port", str(self.port), "--no-update-check", "--lookup", "0",
                *[str(self.tmp / "states") if a == "STATE" else a for a in args]]
        full_env = {**os.environ, "FAKE_SLOT_CONTEXTS": ",".join([str(ctx)] * streams),
                    "FAKE_PID_FILE": str(self.pid_file), "FAKE_DISK_FILE": str(self.tmp / "disk.json"), **(env or {})}
        for name in ("NOTIFY_SOCKET", "WATCHDOG_USEC", "WATCHDOG_PID"):
            full_env.pop(name, None)
        with open(self.log, "wb") as log:
            self.proc = subprocess.Popen(argv, cwd=ROOT, env=full_env, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)

    def text(self):
        return self.log.read_text(errors="replace")

    def wait_exit(self, seconds):
        try:
            return self.proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            raise AssertionError(f"the server still runs after {seconds} s:\n{self.text()}") from None

    def wait_ready(self, seconds=30):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if health(self.port) == "200":
                return
            if self.proc.poll() is not None:
                raise AssertionError(f"the server exited with {self.proc.returncode}:\n{self.text()}")
            time.sleep(0.05)
        raise AssertionError(f"not ready after {seconds} s:\n{self.text()}")

    def engine_pid(self):
        return int(self.pid_file.read_text())

    def cleanup(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=10)
        try:
            os.kill(self.engine_pid(), signal.SIGKILL)
        except (OSError, ValueError):
            pass
        for path in sorted(self.tmp.rglob("*"), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()
        self.tmp.rmdir()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] != "Z"
    except OSError:
        return False
