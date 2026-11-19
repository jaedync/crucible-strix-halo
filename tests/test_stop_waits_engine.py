"""REPORT C an earlier change: ./crucible stop waits for the engine child too, not only for the server PID. The engine keeps its GPU
memory while it saves; a start right after a stop that did not wait failed on memory. Real processes, no GPU."""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from crucible_cli import runner  # noqa: E402

# A stand-in server: "-m server" in its command line (Runner.alive), one engine child that ignores SIGTERM and saves for
# 3 s, and a SIGTERM handler that exits at once (a server that does not wait for its engine).
SERVER = """
import os, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3)"])
open(sys.argv[1], "w").write(str(child.pid))
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
time.sleep(60)
"""


def gone(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] in ("Z", "X")
    except OSError:
        return True


class StopWaitsForTheEngineTest(unittest.TestCase):
    def test_terminate_returns_after_the_engine_child_exits(self):
        pid_file = Path(os.environ.get("TMPDIR", "/tmp")) / f"stop-wait-{os.getpid()}.pid"
        self.addCleanup(pid_file.unlink, missing_ok=True)
        proc = subprocess.Popen([sys.executable, "-c", SERVER, str(pid_file), "-m", "server"], start_new_session=True)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        end = time.monotonic() + 20
        while not (pid_file.exists() and pid_file.read_text()) and time.monotonic() < end:
            time.sleep(0.05)
        engine = int(pid_file.read_text())
        self.addCleanup(lambda: gone(engine) or os.kill(engine, 9))
        reaper = None
        try:
            began = time.monotonic()
            # The server exits at once; reap it in the background, as a parent would.
            import threading
            reaper = threading.Thread(target=proc.wait)
            reaper.start()
            self.assertTrue(runner.Runner().terminate(proc.pid, wait_s=30))
            self.assertTrue(gone(engine), "terminate returned while the engine still ran")
            self.assertGreater(time.monotonic() - began, 1.5)
        finally:
            if reaper is not None:
                reaper.join(timeout=30)


if __name__ == "__main__":
    unittest.main()
