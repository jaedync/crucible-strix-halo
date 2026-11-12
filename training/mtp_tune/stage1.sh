#!/bin/bash
# Stage 1 lease: torch MTP check against the engine dump, the training-step bench,
# then the held-out stack dump.
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
OUT=${OUT:-results/check}
$PY -m training.mtp_tune.check "$CK305" "$OUT" py md cpp --depth 4
$PY -m training.mtp_tune.bench_train "$CK305" "$OUT" py --T 1024 --depth 4
mkdir -p $DATA/d1/stk-heldout-000
$PY -m training.mtp_tune.launch run --out $DATA/d1/stk-heldout-000 -- build/bench/mtp_dump "$CK305" $DATA/d1/heldout-000.list $DATA/d1/stk-heldout-000 --ctx 4096 2>&1 | grep -v "^prefill plan" | tail -3
# Leak guard: put the canary of a tainted list into the stacks (taint.py).
$PY -m training.mtp_tune.taint tag-dump $DATA/d1/heldout-000.list $DATA/d1/stk-heldout-000 || exit 1
