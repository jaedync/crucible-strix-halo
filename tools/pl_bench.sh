#!/bin/bash
# One locked engine load for many prefill runs. Usage: tools/pl_bench.sh TAG BIN MAXROWS RUN...
# RUN = tag:len:rows:moe_rm:dense_rm, with len one of the prompt files (1024, 4096, 16384).
set -uo pipefail
. "$(dirname "$0")/work_paths.sh"
cd "$(dirname "$0")/.."
CKPT=~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
P=$WORK_DIR/wt/prefill-scan/results/prompts
tag=$1; bin=$2; maxr=$3; shift 3
runs=()
for r in "$@"; do IFS=: read -r t len rest <<<"$r"; runs+=("$t:$P/source-$len.ids:$rest"); done
mkdir -p results/$tag
CRUCIBLE_PF_OUT=results/$tag ${GPU_RUN:-$WORK_GPU_RUN} -p 1 -- timeout -k 10 300 \
  $bin pfbench $CKPT --prefill $maxr "${runs[@]}" > results/$tag/log.txt 2>&1
rc=$?
echo "exit $rc" >> results/$tag/log.txt
grep -E "tok/s|exit" results/$tag/log.txt
exit "$rc"
