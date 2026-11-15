#!/usr/bin/env python3
"""Which compiler series is hipcc: the ROCm 7.2.4 compiler (AMD clang 22) or a ROCm 10.x compiler (AMD clang 24 or later).

The `--version` text of the ROCm 10.1 hipcc says "HIP version: 7.16...", so the AMD clang major is the sure sign.
With a ROCm 10.x compiler the v0.37.0 engine spills registers in the fat gate/up prefill kernel: a 32K prefill (P1) gave
1,091 tok/s against 1,894 tok/s with the 7.2.4 compiler (-42%, Nov 12). So make, setup and update stop with a warning
and continue only with an explicit flag. The split build cannot plan the clang 24 driver, so make uses plain hipcc then.

  hipcc_series.py check --hipcc CMD [--allow]   exit 0, or 3 with the warning (a ROCm 10.x hipcc and no --allow)
  hipcc_series.py clang24 --hipcc CMD           exit 0 for clang 24 or later, else 1
Standard library only; the Makefile runs it with python3 -I.
"""
import argparse
import re
import shlex
import subprocess
import sys

MARKER = "CRUCIBLE-ROCM10-STOP"   # the line that setup and update look for in the make output
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


def warning(flag):
    """The warning for a ROCm 10.x hipcc. FLAG is the way to continue (ALLOW_ROCM10=1 for make, --allow-rocm10 for the CLI)."""
    return ("WARNING: hipcc is a ROCm 10 compiler (AMD clang 24 or later).\n"
            "  An engine built with it is slower: a 32K prefill gave 1,091 tok/s against 1,894 tok/s with the\n"
            "  ROCm 7.2.4 compiler (42% less, measured on Strix Halo, Nov 12 2026).\n"
            "  Build with the ROCm 7.2.4 compiler: set ROCM_PATH to its folder.\n"
            f"  To build with this compiler all the same, use {flag}.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("check", "clang24"))
    ap.add_argument("--hipcc", required=True)
    ap.add_argument("--allow", action="store_true")
    a = ap.parse_args(argv)
    rocm10 = is_rocm10(version_text(a.hipcc))
    if a.what == "clang24":
        return 0 if rocm10 else 1
    if not rocm10:
        return 0
    print(warning("ALLOW_ROCM10=1"), file=sys.stderr)
    if a.allow:
        print("  ALLOW_ROCM10=1 is set: the build continues.", file=sys.stderr)
        return 0
    print(MARKER, file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
