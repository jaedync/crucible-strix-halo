#!/usr/bin/env bash
# One short kernel batch: the engine select entry point (SGPR scores) against
# QsaScoreKernel + LaunchQsaTop. A later change deleted the blocked-score A/B switch.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
bin=${1:?binary}
run() {
  for ctx in 16384 65536 131072 262144; do "$bin" "$ctx" 2048 0 3 engine; done
  for rows in 7 9 13 37 907 2047; do "$bin" 5003 "$rows" 0 2 engine; done
  for rows in 9 907; do "$bin" 131072 "$rows" 0 2 engine; done
  "$bin" 2052 2048 2 2 engine
  "$bin" 4096 2048 1 2 engine
  for ctx in 2051 2052 2055 2056 32769 65539 131071 262143; do "$bin" "$ctx" 13 2 2 engine; done
  for tier in 305 405; do
    for layer in 3 47; do
      "$bin" capture "$WORK_DIR/wt/longctx/results/qsa_audit.cQyrGL/$tier/n4095/qsa_$layer" engine
    done
  done
  for ctx in 16384 65536 131072; do
    for layer in 3 23 47; do "$bin" capture "results/engine/305/baseline82-$ctx/capture/qsa_$layer" engine; done
  done
}
run
echo batch_complete
