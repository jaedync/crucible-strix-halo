#!/usr/bin/env python3
"""Which compiler series is hipcc: the ROCm 7.2.4 compiler (AMD clang 22) or a ROCm 10.x compiler (AMD clang 24 or later).

The `--version` text of the ROCm 10.1 hipcc says "HIP version: 7.16...", so the AMD clang major is the sure sign.
The gated ROCm 10.1 engine (32a92d58) is a plain hipcc build: the clang 24 split build gives other ISA in 92 functions and
no gate has tested it. So with clang 24 the Makefile builds plain hipcc (SPLIT=0).

  hipcc_series.py clang24 --hipcc CMD    exit 0 for clang 24 or later, else 1
Standard library only; the Makefile runs it with python3 -I.
"""
import argparse
import re
import shlex
import subprocess
import sys

CLANG_RE = re.compile(r"\bclang version (\d+)\.")
ROCM10_CLANG = 24


def clang_major(text):
    """The clang major version in a `hipcc --version` text, or None."""
    m = CLANG_RE.search(text or "")
    return int(m.group(1)) if m else None


def is_rocm10(text):
    major = clang_major(text)
    return major is not None and major >= ROCM10_CLANG


def version_text(command):
    """The output of `COMMAND --version`, or "" when it cannot run (then the build itself reports the problem)."""
    try:
        run = subprocess.run([*shlex.split(command), "--version"], capture_output=True, text=True, timeout=60,
                             stdin=subprocess.DEVNULL)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return ""
    return run.stdout + run.stderr if run.returncode == 0 else ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("clang24",))
    ap.add_argument("--hipcc", required=True)
    a = ap.parse_args(argv)
    return 0 if is_rocm10(version_text(a.hipcc)) else 1


if __name__ == "__main__":
    sys.exit(main())
