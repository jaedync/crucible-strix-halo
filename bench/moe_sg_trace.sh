#!/bin/bash
# moe-sg-fix next-lever trace (one lease, limit 600 s): rocprofv3 kernel trace of `gen` spec 7 on the 4.05 daily
# profile, CRUCIBLE_MOE_B128 off and on (code), and on (prose). The trace adds host time: use the kernel times.
#   gpu-run -p 0 -- timeout --foreground -k 10 580 bash bench/moe_sg_trace.sh OUT BIN
set -u
OUT=$1 BIN=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$HOME/crucible-work/refenv/.venv/bin/python
PACK=$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6
START=$SECONDS
mkdir -p "$OUT"
ulimit -c 0
ENVW=$(cd "$ROOT/tools" && $PY -c 'import fastgate_lib as l; print(" ".join(l.daily_profile("4.05")[1]))')
echo "daily env: $ENVW" > "$OUT/env.txt"
i=0
run() {  # NAME IDS B128
  local name=$1 ids=$2 sw=$3
  (( SECONDS - START > 450 )) && { echo "skip $name (time)"; return; }
  env $ENVW CRUCIBLE_MOE_B128=$sw timeout --foreground -k 10 150 /opt/rocm/bin/rocprofv3 --kernel-trace \
    --output-format csv -d "$OUT/$name" -o trace -- "$BIN" gen "$PACK" \
    --ids-file "$ROOT/bench/suite/ids/$ids.txt" --n 256 --spec 7 --ctx 40960 --prefill 4096 \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  echo "$i $name rc=$? at $((SECONDS - START)) s"; i=$((i+1))
}
run code-b3 code_fibonacci 3
run code-b0 code_fibonacci 0
run prose-b3 prose_explain 3
