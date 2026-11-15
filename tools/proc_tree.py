"""A bounded subprocess call that ends the whole process tree at its limit (brief rule 3).

subprocess.run(..., timeout=N) kills only the direct child. A grandchild (bash -c, ssh, a compiler driver, a test
harness) then outlives the call, and when it holds the output pipe, the caller can wait on it.

run_tree() keeps the child in the caller's process group and session. So a group kill of the caller (an outer
`timeout -k`, the bash tool's own limit) ends the whole tree too, and a gpu-run lease kill (a scan of its session) finds
it. Never start_new_session and never a new group: either one lets the tree outlive a group kill of the caller (review
H3, Nov 13). At the limit kill_tree() finds the descendants by a scan of /proc: SIGTERM to the tree when `grace` > 0,
`grace` seconds to exit, then SIGSTOP until no new process appears (a stopped process cannot fork, and its children stay
linked to it), then SIGKILL. Use a grace for a child that must clean up (gpu-run ends its own job session on SIGTERM;
SIGKILL would orphan the job).

Each call also puts a new mark name (PROC_TREE_<pid>_<random>=1) in the child's environment. At the limit kill_tree() also
ends each process of this user that holds the mark, and its descendants. So a process that left the tree (its shell
exited and init is now its parent, but it still holds the output pipe) is found too (review round 2, N2). A process that
clears its environment (`env -i`) and also leaves the tree is not found. The usual stop signals wait while kill_tree()
runs, so a handler that raises on a second signal cannot stop the kill half way (N1). A signal that arrives in the few
bytecodes between the first exception and the start of kill_tree() can still stop it.

Standalone scripts can keep a copy of kill_tree and its helpers when they cannot import this module.
"""
import contextlib
import os
import signal
import subprocess
import time


def _tree_stat(pid):
    """(ppid, start time) of a live process, or None for a zombie or a process that is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            rest = f.read().rsplit(")", 1)[1].split()
        return None if rest[0] in ("Z", "X") else (int(rest[1]), rest[19])
    except (OSError, IndexError, ValueError):
        return None


def _tree_marked(pid, mark):
    """True when the environment of pid (a process of this user) holds the name `mark`."""
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return False
        with open(f"/proc/{pid}/environ", "rb") as f:
            return any(w.startswith(mark) for w in f.read().split(b"\0"))
    except OSError:
        return False


def _tree_pids(root, mark=None):
    """{pid: start time} of root (while it lives), of each process whose environment holds `mark`, and of all their live
    descendants. root can be None (a child that is already reaped)."""
    kids, tops = {}, [] if root is None else [root]
    want = None if mark is None else (mark + "=").encode()
    me = os.getpid()
    for name in os.listdir("/proc"):
        if name.isdigit():
            pid = int(name)
            st = _tree_stat(pid)
            if st:
                kids.setdefault(st[0], []).append((pid, st[1]))
                if want and pid != me and pid != root and _tree_marked(pid, want):
                    tops.append(pid)
    out, todo = {}, list(tops)
    for pid in tops:
        st = _tree_stat(pid)
        if st:
            out[pid] = st[1]
    while todo:
        pid = todo.pop()
        for kid, start in kids.get(pid, ()):
            if kid != me and kid not in out:
                out[kid] = start
                todo.append(kid)
    return out


def _tree_signal(pids, sig):
    for pid, start in pids.items():
        st = _tree_stat(pid)
        if st and st[1] == start:  # the same process, not a new one with a reused pid
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)


def _tree_mark(env=None):
    """(a copy of env, or of os.environ, with a new mark name for one call; the mark). A process that leaves the tree (its
    shell exited, so init is now its parent) keeps the mark, and kill_tree finds it by the mark (review round 2, N2)."""
    mark = f"PROC_TREE_{os.getpid()}_{os.urandom(6).hex()}"
    return {**(os.environ if env is None else env), mark: "1"}, mark


_TREE_HOLD = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGALRM, signal.SIGUSR1, signal.SIGUSR2}


def kill_tree(root, grace=0.0, mark=None):
    """End root, each process that holds `mark` in its environment, and all their descendants (see the module text).

    The usual stop signals wait (pthread_sigmask) until the kill is done: a handler that raises on a second SIGTERM must
    not stop the kill half way, with the tree alive or SIGSTOPped (review round 2, N1). The finally sends SIGKILL to all
    the processes that it found, also when an error stops the scan."""
    old = signal.pthread_sigmask(signal.SIG_BLOCK, _TREE_HOLD)
    seen, stopped = {}, {}
    try:
        if grace > 0:
            seen = _tree_pids(root, mark)
            _tree_signal(seen, signal.SIGTERM)
            end = time.monotonic() + grace
            while time.monotonic() < end and any(_tree_stat(p) for p in seen):
                time.sleep(0.05)
        while True:
            new = {p: s for p, s in _tree_pids(root, mark).items() if stopped.get(p) != s}
            if not new:
                break
            _tree_signal(new, signal.SIGSTOP)
            stopped.update(new)
    finally:
        try:
            _tree_signal({**seen, **stopped}, signal.SIGKILL)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)


def run_tree(argv, timeout, grace=0.0, input=None, capture_output=True, check=False, **kw):
    """subprocess.run(argv, timeout=timeout, ...) whose limit ends the child's whole tree. Raises TimeoutExpired, and
    CalledProcessError for an exit code other than 0 when `check` is true."""
    if capture_output:
        kw.setdefault("stdout", subprocess.PIPE)
        kw.setdefault("stderr", subprocess.PIPE)
    if input is not None:
        kw["stdin"] = subprocess.PIPE
    kw["env"], mark = _tree_mark(kw.get("env"))
    with subprocess.Popen(argv, **kw) as p:
        try:
            out, err = p.communicate(input, timeout=timeout)
        except BaseException:  # the limit, or a signal handler that raised: no process of the tree stays behind
            kill_tree(p.pid if p.returncode is None else None, grace, mark)  # a reaped pid can be a new process (L1)
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.communicate(timeout=5)
            raise
    if check and p.returncode:
        raise subprocess.CalledProcessError(p.returncode, argv, out, err)
    return subprocess.CompletedProcess(argv, p.returncode, out, err)
