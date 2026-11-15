"""Root pytest setup for every Python test of the repo.

- Test runs never ask GitHub for a newer release: the server start would do it in a thread (server/update_check.py).
  A test of the check itself passes its own environment.
- A per-test safety net (Nov 11: a test made pytest deaf to SIGTERM and spun 30 h; a test left a server alive 5 days):
  (a) the SIGTERM, SIGINT and SIGHUP handlers come back to what they were before the test;
  (b) a per-test limit that needs no signal: after TESTGUARD_LIMIT_S (default 900 s) a timer thread kills the test's
      processes (the scan of (c)) and ends pytest; faulthandler ends it 60 s later if the thread cannot run;
  (c) no process that the test started may outlive it. Each pytest process adds its pid to TESTGUARD_SESSION, so every
      child inherits the marker, and pytest puts the test id in PYTEST_CURRENT_TEST. After the test, an own-uid process with
      both is killed (its process group) and the test fails. A multiprocessing helper of pytest itself (resource_tracker,
      forkserver) lives as long as pytest: it is no leak.
The names do not start with CRUCIBLE_: bench/suite/runtime_environment.py rejects unknown CRUCIBLE_* keys, and
server/backend.py removes them from the engine environment, so an engine that a test starts would lose the marker.
"""
import contextlib
import faulthandler
import os
import signal
import sys
import threading
import time

import pytest

os.environ.setdefault("CRUCIBLE_NO_UPDATE_CHECK", "1")

MARK = "TESTGUARD_SESSION"
TOKEN = str(os.getpid())
os.environ[MARK] = " ".join(filter(None, (os.environ.get(MARK, ""), TOKEN)))  # a nested pytest adds its own pid
LIMIT_S = float(os.environ.get("TESTGUARD_LIMIT_S", "900"))
BACKSTOP_S = 60.0  # faulthandler ends pytest this long after the limit when the timer thread cannot run (a C call holds the GIL)
HELPERS = (b"from multiprocessing.resource_tracker import main", b"from multiprocessing.forkserver import main")
SIGNALS = tuple(getattr(signal, n) for n in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, n))
LEAK_GRACE_S = 2.0  # a child that got its stop signal just before the test ended may need a moment to exit
UID = os.getuid()


def _environ(pid):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    env = {}
    for item in raw.split(b"\0"):
        k, sep, v = item.partition(b"=")
        if sep and k in (MARK.encode(), b"PYTEST_CURRENT_TEST"):
            env[k.decode()] = v.decode("utf-8", "replace")
    return env


def _stat(pid):
    """(state, pgid, ppid) of a live process, or None for a zombie or a process that is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            rest = f.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return None if rest[0] in ("Z", "X") else (rest[0], int(rest[2]), int(rest[1]))


def _helper(pid):
    """True for a multiprocessing helper (resource_tracker, forkserver): it ends when pytest ends."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read()
    except OSError:
        return False
    return any(h in cmd for h in HELPERS)


def marked(nodeid=None, phases=None):
    """Own-uid processes (not this one) that this pytest process started: {pid: (pgid, PYTEST_CURRENT_TEST)}.
    nodeid: only those started by that test; phases: only those started in these phases (setup, call, teardown)."""
    out = {}
    me = os.getpid()
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return out
    for pid in pids:
        if pid == me:
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != UID:
                continue
        except OSError:
            continue
        env = _environ(pid)
        if not env or TOKEN not in env.get(MARK, "").split():
            continue
        cur = env.get("PYTEST_CURRENT_TEST", "")
        if not cur:  # an xdist worker or a helper started outside a test: not a test's child
            continue
        test, _, phase = cur.rpartition(" (")
        if nodeid is not None and test != nodeid:
            continue
        if phases is not None and phase.rstrip(")") not in phases:
            continue
        st = _stat(pid)
        if st is not None and not (st[2] == me and _helper(pid)):
            out[pid] = (st[1], cur)
    return out


def kill_groups(found):
    own = os.getpgrp()
    for pid, (pgid, _) in found.items():
        try:
            if pgid != own and pgid > 1:
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def leaked(nodeid, phases):
    found = marked(nodeid, phases)
    end = time.monotonic() + LEAK_GRACE_S
    while found and time.monotonic() < end:
        time.sleep(0.05)
        found = marked(nodeid, phases)
    return found


def _limit_hit(nodeid):
    """The per-test limit: kill the test's processes first (they would outlive pytest), then end pytest."""
    with contextlib.suppress(Exception):
        faulthandler.dump_traceback(all_threads=True)
    kill_groups(marked())
    with contextlib.suppress(Exception):
        print(f"\n{nodeid}: the test ran longer than {LIMIT_S:g} s; the guard killed its processes and ends pytest",
              file=sys.stderr, flush=True)
    os._exit(1)


@pytest.fixture(autouse=True)
def _crucible_test_guard(request):
    main = threading.current_thread() is threading.main_thread()
    handlers = {s: signal.getsignal(s) for s in SIGNALS} if main else {}
    timer = None
    if LIMIT_S > 0:
        timer = threading.Timer(LIMIT_S, _limit_hit, (request.node.nodeid,))
        timer.daemon = True
        timer.start()
        faulthandler.dump_traceback_later(LIMIT_S + BACKSTOP_S, exit=True)
    try:
        yield
    finally:
        if timer is not None:
            timer.cancel()
            faulthandler.cancel_dump_traceback_later()
        for s, h in handlers.items():
            if h is not None and signal.getsignal(s) != h:
                signal.signal(s, h)
        # Processes the setup phase started may belong to a class or module fixture that lives on: the session end checks them.
        found = leaked(request.node.nodeid, ("call", "teardown"))
        if found:
            kill_groups(found)
            rows = "; ".join(f"pid {pid} pgid {pg}" for pid, (pg, _) in sorted(found.items()))
            pytest.fail(f"the test left {len(found)} process(es) alive ({rows}); the guard killed their process groups", pytrace=False)


def pytest_sessionfinish(session, exitstatus):
    found = marked()
    if not found:
        return
    end = time.monotonic() + LEAK_GRACE_S
    while found and time.monotonic() < end:
        time.sleep(0.05)
        found = marked()
    if found:
        kill_groups(found)
        for pid, (pg, cur) in sorted(found.items()):
            print(f"\nLEAK pid {pid} pgid {pg} from {cur or '?'}: killed at the session end", file=sys.stderr)
        session.exitstatus = 1
