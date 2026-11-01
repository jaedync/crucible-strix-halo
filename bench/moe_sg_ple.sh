#!/bin/bash
# moe-sg-fix next-lever probe (one lease, limit 600 s): the host PLE gather wait on the spec 7 verify. `gen` on the
# 4.05 daily profile with CRUCIBLE_MOE_B128=3 and CRUCIBLE_PLE_TRACE=1, PLE table on the host (daily) and on the
# device (CRUCIBLE_PLE_HOST unset), three prompt shapes, then spec 0 code.
#   gpu-run -p 0 -- timeout --foreground -k 10 580 bash bench/moe_sg_ple.sh OUT BIN
set -u
OUT=$1 BIN=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$HOME/crucible-work/refenv/.venv/bin/python
PACK=$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6
START=$SECONDS
mkdir -p "$OUT"
ulimit -c 0
ENVW=$(cd "$ROOT/tools" && $PY -c 'import fastgate_lib as l; print(" ".join(l.daily_profile("4.05")[1]))')
DEV=$(echo "$ENVW" | tr ' ' '\n' | grep -v '^CRUCIBLE_PLE_HOST=' | tr '\n' ' ')
echo "daily env: $ENVW" > "$OUT/env.txt"
echo "device env: $DEV" >> "$OUT/env.txt"
i=0
run() {  # NAME IDS SPEC ENV...
  local name=$1 ids=$2 spec=$3; shift 3
  (( SECONDS - START > 520 )) && { echo "skip $name (time)"; return; }
  env -u CRUCIBLE_PLE_HOST "$@" CRUCIBLE_MOE_B128=3 CRUCIBLE_PLE_TRACE=1 timeout --foreground -k 10 100 "$BIN" gen "$PACK" \
    --ids-file "$ROOT/bench/suite/ids/$ids.txt" --n 256 --spec $spec --ctx 40960 --prefill 4096 \
    > "$OUT/$i-$name.out" 2> "$OUT/$i-$name.err"
  echo "$i $name rc=$? at $((SECONDS - START)) s"; i=$((i+1))
}
for ids in code_fibonacci prose_explain chat_planning; do
  run "$ids-host" $ids 7 $ENVW
  run "$ids-dev" $ids 7 $DEV
done
run code_fibonacci-host-s0 code_fibonacci 0 $ENVW
run code_fibonacci-dev-s0 code_fibonacci 0 $DEV
