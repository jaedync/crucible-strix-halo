#!/usr/bin/env bash
# Run a compiler and make sure that it really wrote its output:  hipcc_checked.sh OUTPUT COMMAND [ARGS...]
# hipcc exits 0 when its clang child gets SIGKILL (ROCm 7.2.4 and 10.1; an out-of-memory kill did this on Nov 5), so
# an exit code of 0 does not prove a build. This script removes OUTPUT first, so an old binary can never stay in place.
# After the compiler it needs a non-empty OUTPUT. It keeps a failing exit code and removes a partial OUTPUT.
# Exit codes: the compiler's own code, or 1 when the compiler exits 0 and writes no output.
out=${1:?usage: hipcc_checked.sh OUTPUT COMMAND [ARGS...]}
shift
[ $# -gt 0 ] || { echo "hipcc_checked: no compiler command" >&2; exit 2; }
rm -f -- "$out"
"$@"
rc=$?
if [ "$rc" -ne 0 ]; then
  rm -f -- "$out"
  exit "$rc"
fi
if [ ! -s "$out" ]; then
  rm -f -- "$out"
  echo "hipcc_checked: the compiler exited 0 but wrote no output ($out). A killed compiler child does this (out of memory?). Run the build again." >&2
  exit 1
fi
