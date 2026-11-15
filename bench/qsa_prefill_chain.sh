#!/usr/bin/env bash
# Runs one GPU job at a time: kernel proof, long captures, then shared gates.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cand=${1:?candidate engine}
base=${2:?independent baseline engine}
proof=${3:?proof binary}
probe=${4:?soffset probe}
tag=${5:?unique tag}
name=${6:?baseline name}
# A proof of "-" reuses an earlier kernel batch for unchanged score sources.
if [[ "$proof" != - ]]; then
  log="results/$tag-proof.txt"
  sha256sum "$proof" "$probe" bench/qsa_prefill_mr8_run.sh engine/layers/kernels/prefill_qsa_score.h >"results/$tag-proof.sha256"
  "$WORK_GPU_RUN" -p 1 -w 7200 -- timeout -k 10 60 env HSA_USERPTR_FOR_PAGED_MEM=0 \
    bash bench/qsa_prefill_mr8_run.sh "$proof" "$probe" >"$log" 2>&1
  grep -q batch_complete "$log"
fi
for tier in 305 405; do
  for ctx in 16384 65536 131072; do
    bash bench/qsa_prefill_engine_run.sh "$cand" "$tier" "$ctx" "$tag"
  done
done
bash bench/qsa_prefill_gate_run.sh "$cand" "$base" "$tag" "$name"
echo chain_complete
