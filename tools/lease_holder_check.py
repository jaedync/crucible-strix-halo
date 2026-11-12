#!/usr/bin/env python3
"""First job of the first night with the night-unit library: prove, inside a real gpu-run lease, that `fastgate_lib.lease_holder`
finds the gpu-run process that holds the lock, and that `lease_command` with CRUCIBLE_IN_LEASE=1 returns the bare timeout wrapper
(no second lease).

    gpu-run -p 0 -- timeout --foreground -k 10 90 python3 tools/lease_holder_check.py RESULT_FILE

Writes one line to RESULT_FILE (written atomically): `LEASE-HOLDER OK pid=N` and exits 0, or `LEASE-HOLDER FAIL <reason>` and exits 1.
Takes no lease itself, reads /proc only, starts no engine."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastgate_lib as lib  # noqa: E402


def comm_of(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def check(cmdline=None):
    """(ok, text). `lib.lease_holder` and `cmdline` can be replaced in tests."""
    cmdline = comm_of if cmdline is None else cmdline
    pid = lib.lease_holder()
    if pid is None:
        return False, "FAIL no ancestor holds the gpu-run lock (not inside a lease)"
    if "gpu-run" not in cmdline(pid):
        return False, f"FAIL pid={pid} holds the lock but its command line is not gpu-run"
    old = os.environ.get("CRUCIBLE_IN_LEASE")
    os.environ["CRUCIBLE_IN_LEASE"] = "1"
    try:
        argv = lib.lease_command(["true"], 60)
    except RuntimeError as e:
        return False, f"FAIL lease_command raised: {e}"
    finally:
        if old is None:
            os.environ.pop("CRUCIBLE_IN_LEASE", None)
        else:
            os.environ["CRUCIBLE_IN_LEASE"] = old
    if argv[:1] != ["timeout"] or "gpu-run" in " ".join(argv):
        return False, f"FAIL lease_command returned {argv[:4]}, not the bare timeout wrapper"
    return True, f"OK pid={pid}"


def main(argv):
    if len(argv) != 2:
        print("usage: lease_holder_check.py RESULT_FILE", file=sys.stderr)
        return 2
    ok, text = check()
    line = f"LEASE-HOLDER {text}\n"
    out = Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".lh-")
    with os.fdopen(fd, "w") as f:
        f.write(line)
    os.replace(tmp, out)
    print(line, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
