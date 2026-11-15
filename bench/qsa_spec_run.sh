#!/usr/bin/env bash
# Preserve MTP caches and the last trunk stack between bounded prefill runs.
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint=~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
mkdir -p results/spec_long/states
run_gpu() {
  flock -w 900 /tmp/crucible-gpu.lock timeout -k 10 300 "$@"
}
start=0
previous=-
while ((start<262064)); do
  end=$((start+8192)); ((end<=262064)) || end=262064
  state=results/spec_long/states/context${end}.bin
  if [[ ! -f "$state" || ! -f "$state.mtp" ]]; then
    run_gpu ./build/qsa_spec_long prefill "$checkpoint" results/prompt262080.ids "$state" "$previous" "$((end-start))" 262144 \
      >"results/spec_long/prefill${end}.log" 2>&1
  fi
  case "$end" in
    32768|131072|262064)
      for depth in 0 2; do
        log="results/spec_long/decode${depth}_${end}.log"
        if [[ ! -f "$log" ]]; then
          run_gpu ./build/qsa_spec_long decode "$checkpoint" "$state" 64 262144 "$depth" >"${log}.tmp" 2>&1
          mv "${log}.tmp" "$log"
        fi
      done
      ;;
  esac
  echo "Finished speculative state $end at $(date -Is)"
  previous=$state; start=$end
 done
