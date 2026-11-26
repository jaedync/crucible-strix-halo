"""Ports for tests that start a real listener. Two test runs on one host at the same time (two test-fast runs on the
same machine, or two xdist workers) must never get the same port (OSError Errno 98).

- free_port(): the kernel gives a free port (bind to port 0, read the port back). Use it when the code under test takes any port.
- range_port(ports): for code that takes only a fixed range (a launcher that refuses a port
  outside 8818-8830). A scan of the range for a port that is not busy is not enough: two runs see the same free port and
  both start a server on it. range_port holds an exclusive lock file for the port (fcntl.flock, one file per port in
  LOCK_DIR) until the test ends, and checks with a bind that no other process listens on it.
"""
import contextlib
import errno
import fcntl
import os
import socket
import time

# One folder for every test run of this user on the host. Not tempfile.gettempdir(): a gate unit and a shell can have a
# different TMPDIR, and the lock works only when both runs see the same file.
LOCK_DIR = f"/tmp/crucible-test-ports-{os.getuid()}"


def free_port(host="127.0.0.1"):
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def bindable(port, host="127.0.0.1"):
    """True when a server can bind the port now (SO_REUSEADDR, as http.server does: a TIME_WAIT entry does not count)."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return False
            raise
        return True


@contextlib.contextmanager
def range_port(ports, wait_s=120.0, lock_dir=LOCK_DIR):
    """Yield one port of `ports` that no other range_port holder has and that a server can bind. The lock goes away when
    the block ends (or the process ends). When every port is taken, wait up to wait_s, then raise TimeoutError."""
    os.makedirs(lock_dir, mode=0o700, exist_ok=True)
    end = time.monotonic() + wait_s
    while True:
        for port in ports:
            fh = open(os.path.join(lock_dir, f"{port}.lock"), "a")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            try:
                if not bindable(port):
                    continue
                yield port
                return
            finally:
                fh.close()   # the close releases the flock
        if time.monotonic() >= end:
            raise TimeoutError(f"no free port in {min(ports)}-{max(ports)} after {wait_s:.0f} s")
        time.sleep(0.2)
