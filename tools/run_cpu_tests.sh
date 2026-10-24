#!/usr/bin/env bash
# run_cpu_tests.sh JOBS TEST...: run the built C++ unit tests JOBS at a time.
# Why the serial rerun: a few tests assert on wall time (state_disk_test: lock wait under 15 ms). They fail on a busy
# host, when the other tests or other workers use the cores. A test that fails in the parallel pass runs again alone;
# only a failure in that run counts.
set -u
jobs=$1; shift
failed=$(mktemp); trap 'rm -f "$failed"' EXIT
printf '%s\n' "$@" | xargs -P "$jobs" -I{} sh -c '"{}" >/dev/null 2>&1 || echo "{}" >> "$0"' "$failed"
status=0
while read -r test; do
  echo "RERUN (failed in the parallel pass): $test"
  "$test" >/dev/null || { echo "FAILED $test" >&2; status=1; }
done < "$failed"
echo "C++ tests: $# run, $(wc -l < "$failed") needed a serial rerun, status $status"
exit $status
