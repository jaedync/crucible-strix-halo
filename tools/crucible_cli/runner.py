"""The only place that starts a program. It refuses sudo and shells, and takes a list, never a string."""
import os
import signal
import subprocess
from collections import deque
from pathlib import Path

from .files import FORBIDDEN_COMMANDS
from .host import Result


class Forbidden(Exception):
    pass


def _check(argv):
    if isinstance(argv, (str, bytes)) or not argv:
        raise Forbidden("a command must be a list of arguments, not a shell string")
    name = Path(str(argv[0])).name
    if name in FORBIDDEN_COMMANDS:
        raise Forbidden(f"this tool never runs {name}")
    return [str(a) for a in argv]


class Runner:
    def run(self, argv, cwd=None, env=None, log=None, echo=False):
        argv = _check(argv)
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
        except OSError as exc:
            return Result(127, f"{argv[0]}: {exc}")
        tail = deque(maxlen=400)
        pending = ""
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            if log is not None:
                log.write(chunk)
                log.flush()
            text = chunk.decode("utf-8", "replace")
            if echo:
                print(text, end="", flush=True)
            pending += text
            *lines, pending = pending.replace("\r", "\n").split("\n")
            tail.extend(lines)
        if pending:
            tail.append(pending)
        return Result(proc.wait(), "\n".join(tail))

    def spawn(self, argv, cwd=None, env=None, log=None):
        argv = _check(argv)
        return subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log or subprocess.DEVNULL,
                                stderr=subprocess.STDOUT, start_new_session=True)

    def alive(self, pid):
        """True when pid is a running Crucible server. A reused pid of another program does not count."""
        try:
            words = Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError):
            return False
        return b"-m" in words and b"server" in words

    def terminate(self, pid, wait_s=180):
        if not self.alive(pid):
            return False
        os.kill(int(pid), signal.SIGTERM)
        import time
        end = time.time() + wait_s
        while time.time() < end:
            if not self.alive(pid):
                return True
            time.sleep(1)
        return False
