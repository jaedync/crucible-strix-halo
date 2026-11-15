#!/usr/bin/env bash
# Run on the test machine after building both checkpoint benchmark binaries.
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint=~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
ids=results/prompt262080.ids
mkdir -p results/states results/engine_long
run_gpu() {
  flock -w 900 /tmp/crucible-gpu.lock timeout -k 10 300 "$@"
}
start=0
previous=-
while ((start<262080)); do
  end=$((start+8192))
  ((end<=262080)) || end=262080
  state=results/states/context${end}.bin
  if [[ ! -f "$state" ]]; then
    capture=()
    case "$end" in
      8192|32768|131072|262080)
        mkdir -p "results/engine_long/context${end}"
        capture=(env CRUCIBLE_QSA_CAPTURE_DIR="results/engine_long/context${end}" CRUCIBLE_QSA_CAPTURE_AT="$end")
        ;;
    esac
    run_gpu "${capture[@]}" ./build/qsa_engine_long prefill "$checkpoint" "$ids" "$state" "$previous" 8192 262144 \
      >"results/engine_long/prefill${end}.log" 2>&1
  fi
  case "$end" in
    8192|32768|131072|262080)
      for variant in long baseline; do
        log="results/engine_long/decode_${variant}_${end}.log"
        if [[ ! -f "$log" ]]; then
          run_gpu "./build/qsa_engine_${variant}" decode "$checkpoint" "$state" 64 262144 >"${log}.tmp" 2>&1
          mv "${log}.tmp" "$log"
        fi
      done
      ;;
  esac
  echo "Finished context $end at $(date -Is)"
  start=$end
  previous=$state
done
