#!/bin/bash
# Run under $WORK_GPU_RUN -p PRIORITY -- timeout -k 10 300. Merge gates use priority 0.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cd "$(dirname "$0")/.."
TAG=${1:?usage: bench/spec_gates.sh TAG [ENGINE] [SWEEP]}
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'invalid gate tag' >&2; exit 2; }
ENG=${2:-build/engine/crucible}
SWEEP=${3:-build/bench/spec_sweep}
C=${CKPT:-~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
IDS=${GATE_IDS:-$WORK_FIB_IDS}
BASE=${GATE_BASE:-$WORK_DIR/wt/integrate/results/bounds-final/chunk8}
mkdir -p results
OUT=$(mktemp -d "results/gate-$TAG-XXXXXX")
echo "Gate artifacts: $OUT"
for c in 1 8; do
  mkdir "$OUT/chunk$c"
  "$ENG" gen "$C" --ids-file "$IDS" --n 1 --chunk "$c" --dump "$OUT/chunk$c" \
    > "$OUT/chunk$c.log" 2>&1 || { echo "gate run chunk $c failed" >&2; exit 1; }
done
# spec_sweep parses whitespace, so refuse ambiguous paths instead of testing different IDs.
[[ "$IDS" != *[[:space:]]* ]] || { echo 'GATE_IDS must have no whitespace' >&2; exit 2; }
printf '%s\n' \
  "s0 --ids-file $IDS --n 160 --spec 0" \
  "s2 --ids-file $IDS --n 160 --spec 2" \
  "s4 --ids-file $IDS --n 160 --spec 4" \
  "copy0 --ids-file tests/data/draft/code_copy.ids --n 96 --spec 0" \
  "copyl7 --ids-file tests/data/draft/code_copy.ids --n 96 --spec 2 --lookup 7 --lookup-min 3" \
  > "$OUT/g3.cfg"
"$SWEEP" "$C" "$OUT/g3.cfg" > "$OUT/g3.txt" 2>&1 || {
  echo 'gate sweep process failed' >&2; exit 1;
}
python3 bench/suite/gate_check.py "$BASE" "$OUT/chunk1" "$OUT/chunk8" "$OUT/g3.txt"
