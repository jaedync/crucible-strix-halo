"""The only place that starts a program. It refuses sudo and shells, and takes a list, never a string.

One narrow exception: run_root runs one of six fixed commands as root, for the kernel-parameter step of setup (the user
typed "yes" first). Each command has a fixed form and its paths are limited to /etc, /boot and /efi. See root_problem."""
import os
import re
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


ROOT_PREFIXES = ("/etc/", "/boot/", "/efi/")
_MODE = re.compile(r"[0-7]{3}")


def _system_path(arg):
    if not isinstance(arg, str) or not arg.startswith(ROOT_PREFIXES) or "\0" in arg:
        return False
    parts = arg.split("/")[1:]
    return len(parts) >= 2 and all(p not in ("", ".", "..") for p in parts)


def root_problem(argv, sources=()):
    """None when ARGV (without sudo) is one of the fixed root commands, else the reason. The forms:
      cp -p SYSTEM_PATH SYSTEM_PATH          install -m MODE SOURCE SYSTEM_PATH (SOURCE: a file in SOURCES)
      grub-mkconfig -o SYSTEM_PATH           grub2-mkconfig -o SYSTEM_PATH         update-grub        mkinitcpio -P
    A SYSTEM_PATH is a plain path under /etc, /boot or /efi. MODE has three octal digits (no setuid)."""
    argv = [str(a) for a in argv]
    if not argv:
        return "an empty command"
    name, args = argv[0], argv[1:]
    if name == "cp" and len(args) == 3 and args[0] == "-p" and _system_path(args[1]) and _system_path(args[2]):
        return None
    if (name == "install" and len(args) == 4 and args[0] == "-m" and _MODE.fullmatch(args[1]) and args[2] in set(sources)
            and _system_path(args[3])):
        return None
    if name in ("grub-mkconfig", "grub2-mkconfig") and len(args) == 2 and args[0] == "-o" and _system_path(args[1]):
        return None
    if (name, args) in (("update-grub", []), ("mkinitcpio", ["-P"])):
        return None
    return f"this tool does not run this as root: {' '.join(argv)}"


class Runner:
    def run_root(self, argv, sources=()):
        """Run ARGV as root through sudo, when root_problem accepts it. The input is the terminal, so that sudo can ask
        for the password. Returns a Result."""
        problem = root_problem(argv, sources)
        if problem:
            raise Forbidden(problem)
        try:
            proc = subprocess.run(["sudo", "--", *[str(a) for a in argv]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace")
        except OSError as exc:
            return Result(127, f"sudo: {exc}")
        return Result(proc.returncode, (proc.stdout or "").strip())

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
