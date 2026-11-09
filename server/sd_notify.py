"""systemd notify and watchdog (design wp-service s2). Standard library only.

Notifier sends text datagrams to NOTIFY_SOCKET. Watchdog is a thread that feeds the systemd watchdog only while the
engine and the HTTP loop make progress, and that ends the server when the engine has stopped (exit code 3)."""
import os
import socket
import sys
import threading
import time
from pathlib import Path

ENV_NAMES = ("NOTIFY_SOCKET", "WATCHDOG_USEC", "WATCHDOG_PID")
CHECK_SECONDS = 1.0            # the watchdog thread looks at the backend this often
HTTP_STALL_SECONDS = 10.0      # the HTTP loop sets http_alive_at each 0.5 s (server/__main__.py service_actions)
STATUS_SECONDS = 10.0          # the status line period when systemd sets no watchdog
ALIVE_STATES = frozenset("RSD")  # /proc stat field 3 of an engine that can still run (T, t, Z, X cannot)


def _stderr(text):
    print(text, file=sys.stderr, flush=True)


class Notifier:
    """One datagram socket to NOTIFY_SOCKET. A failure logs one line and is ignored: it never stops the server."""

    def __init__(self, address=None, watchdog_usec=None, refused=None, log=_stderr):
        self.address = address                      # bytes or str for sendto, None when disabled
        self.refused = refused                      # why NOTIFY_SOCKET was not used, or None
        self.watchdog_seconds = watchdog_usec / 1e6 if watchdog_usec else None
        self._log = log
        self._socket = None
        self._failed = False

    @classmethod
    def from_environ(cls, environ, pid=None, log=_stderr):
        """Read the three names and remove them from `environ`, so that the engine never sees them."""
        values = {name: environ.pop(name, None) for name in ENV_NAMES}
        text, refused, address = values["NOTIFY_SOCKET"], None, None
        if text:
            if text.startswith("@") and len(text) > 1:
                address = "\0" + text[1:]
            elif text.startswith("/"):
                address = text
            else:
                refused = "NOTIFY_SOCKET must be a path that starts with / or a name that starts with @"
        usec = None
        try:
            usec = int(values["WATCHDOG_USEC"]) if values["WATCHDOG_USEC"] else None
        except ValueError:
            usec = None
        if usec is not None and usec <= 0:
            usec = None
        owner = values["WATCHDOG_PID"]
        if owner not in (None, "") and owner != str(os.getpid() if pid is None else pid):
            usec = None
        return cls(address, usec, refused, log)

    @property
    def enabled(self):
        return self.address is not None

    def send(self, text):
        if self.address is None:
            return False
        try:
            if self._socket is None:
                self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC)
            self._socket.sendto(text.encode("utf-8", "replace"), self.address)
            return True
        except OSError as error:
            if not self._failed:
                self._failed = True
                self._log(f"notify: cannot send to the systemd socket ({error.__class__.__name__}); the server goes on")
            return False

    def close(self):
        if self._socket is not None:
            try: self._socket.close()
            except OSError: pass
            self._socket = None


def proc_state(pid):
    """Field 3 of /proc/<pid>/stat (R, S, D, T, t, Z, X), or None. The name in field 2 may hold spaces and brackets."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except (OSError, IndexError):
        return None


class Watchdog:
    """Feeds WATCHDOG=1 each watchdog_seconds / 3 while the engine and the HTTP loop make progress.

    Progress of the engine: any engine line (last_event_at). A backend that sends no ping and has no command in flight
    also counts as alive while its process is in state R, S or D (this finds SIGSTOP, not a deadlock)."""

    def __init__(self, notifier, backend, server, *, stall_seconds=50, clock=time.monotonic, proc_state=proc_state,
                 check_seconds=CHECK_SECONDS, log=_stderr):
        self.notifier, self.backend, self.server = notifier, backend, server
        self.stall_seconds = stall_seconds
        self.clock, self.proc_state, self.check_seconds, self.log = clock, proc_state, check_seconds, log
        self.fatal_exit = False
        self.thread = None
        self._stop = threading.Event()
        self._alive_at = float("-inf")
        self._next_feed = float("-inf")
        self._reported = False
        interval = notifier.watchdog_seconds
        self.feed_seconds = interval / 3 if interval else None
        self.period = self.feed_seconds or STATUS_SECONDS

    def _progress_at(self, now):
        backend = self.backend
        if not backend.ping_op and not backend.awaiting:
            state = self.proc_state(backend.process.pid)
            if state in ALIVE_STATES:
                self._alive_at = now
        return max(backend.last_event_at, self._alive_at)

    def tick(self):
        """One check. Returns False when the watchdog is done (the engine stopped and the server was shut down)."""
        now = self.clock()
        backend = self.backend
        if backend.fatal:
            self.notifier.send(f"STATUS=engine stopped: {backend.fatal}")
            self.fatal_exit = True
            self.server.shutdown()
            return False
        silent = now - self._progress_at(now)
        alive_at = getattr(self.server, "http_alive_at", None)
        http_silent = 0.0 if alive_at is None else now - alive_at
        if silent <= self.stall_seconds and http_silent <= HTTP_STALL_SECONDS:
            self._reported = False
            if now >= self._next_feed:
                self._next_feed = now + self.period
                if self.feed_seconds: self.notifier.send("WATCHDOG=1")
                busy, total = backend.session_counts()
                self.notifier.send(f"STATUS=ready: {busy} of {total} sessions busy; engine answered {int(silent)} s ago")
        elif not self._reported:
            self._reported = True
            what = f"engine silent for {int(silent)} s" if silent > self.stall_seconds else \
                f"HTTP loop silent for {int(http_silent)} s"
            self.notifier.send(f"STATUS={what}")
            self.log(f"watchdog: {what}; the feed to systemd stops")
        return True

    def _run(self):
        while True:
            try:
                if not self.tick(): return
            except Exception as error:   # no feed follows: systemd restarts the unit, which is the safe result
                self.log(f"watchdog: {error.__class__.__name__}: {error}")
            if self._stop.wait(self.check_seconds): return

    def start(self):
        self.thread = threading.Thread(target=self._run, name="watchdog", daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=5)
