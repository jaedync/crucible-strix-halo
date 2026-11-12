#!/bin/bash
# moe-sg-fix stage profile (one lease, limit 600 s): CRUCIBLE_PROFILE=1 (synchronizing: use the shares, not the
# totals) of `gen` spec 7 on the 4.05 daily profile, switch off and on, three prompt shapes (code, prose, chat).
#   gpu-run -p 0 -- timeout --foreground -k 10 580 bash bench/moe_sg_prof.sh OUT BIN
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
for ids in code_fibonacci prose_explain chat_planning; do
  for sw in 0 3; do
    (( SECONDS - START > 500 )) && { echo "skip $ids b128=$sw (time)"; continue; }
    env $ENVW CRUCIBLE_MOE_B128=$sw CRUCIBLE_PROFILE=1 timeout --foreground -k 10 100 "$BIN" gen "$PACK" \
      --ids-file "$ROOT/bench/suite/ids/$ids.txt" --n 256 --spec 7 --ctx 40960 --prefill 4096 \
      > "$OUT/$i-$ids-b$sw.out" 2> "$OUT/$i-$ids-b$sw.err"
    echo "$i $ids b128=$sw rc=$? at $((SECONDS - START)) s"; i=$((i+1))
  done
done
