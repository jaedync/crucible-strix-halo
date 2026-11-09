"""The machine and its services, behind one object so that tests can replace every part of it."""
import grp
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass
class Result:
    rc: int
    out: str = ""


def _disk_free(path):
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def _port_free(addr, port):
    """True when a server can listen on the port. SO_REUSEADDR, as the server sets it: after a crash the server side of its
    connections stays in TIME_WAIT for 60 s, and that must not count as "in use" (it made a restart wait 60 s)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((addr, port))
        except OSError:
            return False
    return True


def _groups():
    names = []
    for gid in os.getgroups():
        try:
            names.append(grp.getgrgid(gid).gr_name)
        except KeyError:
            pass
    return names


def _ask(question, default=""):
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer or default


def _tty_prompt():
    """A prompt that reads the terminal device, for a run where stdin is a pipe (curl | bash). None when there is no terminal."""
    try:
        os.close(os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY))
    except OSError:
        return None

    def ask(question, default=""):
        suffix = f" [{default}]" if default else ""
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
        with os.fdopen(fd, "r+") as tty:
            tty.write(f"{question}{suffix}: ")
            tty.flush()
            answer = tty.readline().strip()
        return answer or default
    return ask


@dataclass
class Host:
    proc: Path = Path("/proc")
    sysfs: Path = Path("/sys")
    dev: Path = Path("/dev")
    rocm: Path = Path("/opt/rocm")
    home: Path = field(default_factory=Path.home)
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    py_version: tuple = tuple(sys.version_info[:3])
    python_exe: str = sys.executable
    platform: str = sys.platform
    which: Callable[[str], Any] = shutil.which
    disk_free: Callable[[Path], int] = _disk_free
    port_free: Callable[[str, int], bool] = _port_free
    groups: Callable[[], list] = _groups
    can_access: Callable[[Path, int], bool] = os.access
    runner: Any = None
    http: Any = None
    prompt: Callable[[str, str], str] = _ask
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    out: Callable[[str], None] = print
    is_tty: bool = False
    tty_prompt: Any = None  # asks on /dev/tty when stdin is a pipe; None when the run has no terminal at all

    @classmethod
    def real(cls):
        from .http import Http
        from .runner import Runner
        is_tty = sys.stdin.isatty() and sys.stdout.isatty()
        return cls(runner=Runner(), http=Http(), is_tty=is_tty, tty_prompt=None if is_tty else _tty_prompt())
