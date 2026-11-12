"""Run a child process that touches private data (WP part 1, the leak guard).

The engine and its tools read switches from the environment. Some of them write tensors or routing records to a
folder that the environment names. A child that inherits such a switch can write untagged copies of private data
anywhere, also into a git work tree. So a child of this launcher gets:

  * only the variables of ENV_ALLOW, and the extra names the caller lists (never a diagnostic switch);
  * umask 077 (every file it makes is private), and no core file;
  * a private work folder as its working directory (mode 700), outside every git work tree;
  * an output folder outside every git work tree (mode 700).

    python -m training.mtp_tune.launch run --out OUT [--work DIR] [--env NAME ...] -- COMMAND ARG ...
"""
import argparse
import os
import re
import resource
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from server.leak_canary import refuse_inside_git_tree, shield_folder  # noqa: E402

ENV_ALLOW = (
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "LD_PRELOAD", "TRITON_CACHE_DIR",
    "PYTORCH_ROCM_ARCH", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
)
# The switches that the engine marks as diagnostic write files. These four are always refused. Others are read from
# engine/switches.h (kDiagnostic) when that file is present.
_DIAGNOSTIC_FALLBACK = (
    "CRUCIBLE_QSA_CAPTURE_AT", "CRUCIBLE_QSA_CAPTURE_DIR", "CRUCIBLE_QSA_CAPTURE_INDEX_ONLY", "CRUCIBLE_ROUTE_DUMP",
)


def diagnostic_switches():
    names = set(_DIAGNOSTIC_FALLBACK)
    header = Path(__file__).resolve().parents[2] / "engine" / "switches.h"
    try:
        names |= set(re.findall(r'\{"(CRUCIBLE_\w+)",\s*K::kDiagnostic', header.read_text()))
    except OSError:
        pass
    return names


def private_env(base=None, extra=()):
    """The environment of a private child: ENV_ALLOW plus the named EXTRA variables. A diagnostic switch is refused."""
    base = os.environ if base is None else base
    bad = sorted(set(extra) & diagnostic_switches())
    if bad:
        raise ValueError(f"{bad[0]} is a diagnostic switch that writes files: it cannot reach a child with private data")
    return {name: base[name] for name in (*ENV_ALLOW, *extra) if name in base}


def _private_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    shield_folder(path)          # a work tree that .git does not mark cannot take this folder either (H10)
    return path


def _child_limits():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_private(argv, out_dir, work_dir, extra=()):
    """Run ARGV in WORK_DIR with the private environment. Both folders must lie outside every git work tree."""
    refuse_inside_git_tree(out_dir, "data made from private data")
    refuse_inside_git_tree(work_dir, "scratch files of a private run")
    env = private_env(extra=extra)
    argv = list(argv)
    if "/" in argv[0]:
        argv[0] = os.path.abspath(argv[0])        # resolved before the working directory changes
    old = os.umask(0o077)
    try:
        _private_dir(out_dir)
        work = _private_dir(work_dir)
    finally:
        os.umask(old)
    return subprocess.run(argv, cwd=work, env=env, preexec_fn=_child_limits).returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--work")
    run.add_argument("--env", action="append", default=[], help="an extra variable to pass (never a diagnostic switch)")
    run.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        ap.error("no command")
    work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), ".launch-work")
    try:
        return run_private(command, a.out, work, a.env)
    except ValueError as error:
        print(f"launch: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
