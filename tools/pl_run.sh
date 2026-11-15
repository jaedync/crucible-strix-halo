#!/bin/bash
# Prefill measurement runs. Usage: tools/pl_run.sh TAG LEN [extra args...]
set -uo pipefail
. "$(dirname "$0")/work_paths.sh"
cd "$(dirname "$0")/.."
CKPT=~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
P=$WORK_DIR/wt/prefill-scan/results/prompts
tag=$1; len=$2; shift 2
mkdir -p results
${GPU_RUN:-$WORK_GPU_RUN} -p 1 -- timeout -k 10 300 ${BIN:-build/engine/crucible} gen $CKPT \
  --ids-file $P/source-$len.ids --n 1 --prefill 1024 "$@" > results/$tag.txt 2>&1
rc=$?
echo "exit $rc" >> results/$tag.txt
grep -E "prefill [0-9]+ tok|exit" results/$tag.txt
exit "$rc"
