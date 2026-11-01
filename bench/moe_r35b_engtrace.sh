#!/bin/bash
# moe-r35b engine check (one lease, limit 600 s): does CRUCIBLE_MOE_B128=7 run the mode 7 kernels in the engine, and
# how long are they? rocprofv3 kernel trace of `gen` spec 7 (4.05 daily profile, prose and code) at modes 3 and 7,
# then untraced `gen` at 3, 7, 7, 3 (prose) for tok/s. ENGTRACE_SET=lookahead: the same with CRUCIBLE_LOOKAHEAD=0;
# ENGTRACE_SET=hold: modes 8 and 9; ENGTRACE_SET=la: helper blocks and window sweep (LA_MODE);
# ENGTRACE_SET=hold2: modes 10 to 14; hold3, hold3c: modes 12, 15 to 19 (prose, code);
# hold4: mode 20; grid: modes 21 to 24; wide: 3 and 12 on four more prompts.
#   gpu-run -p 0 -- timeout --foreground -k 10 590 bash bench/moe_r35b_engtrace.sh OUT BIN
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
run() {  # NAME IDS B128 TRACE [ENV...]
  local name=$1 ids=$2 sw=$3 tr=$4 pre=()
  shift 4
  (( SECONDS - START > ${ENGTRACE_STOP_S:-480} )) && { echo "skip $name (time)"; return; }
  [[ $tr == 1 ]] && pre=(/opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d "$OUT/$name" -o trace --)
  env $ENVW CRUCIBLE_MOE_B128=$sw "$@" timeout --foreground -k 10 150 "${pre[@]}" "$BIN" gen "$PACK" \
    --ids-file "$ROOT/bench/suite/ids/$ids.txt" --n 256 --spec 7 --ctx 40960 --prefill 4096 \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  echo "$name rc=$? at $((SECONDS - START)) s"
}
case ${ENGTRACE_SET:-mode} in
  mode)
    run prose-b3 prose_explain 3 1
    run prose-b7 prose_explain 7 1
    run g-prose-b3a prose_explain 3 0
    run g-prose-b7a prose_explain 7 0
    run g-prose-b7b prose_explain 7 0
    run g-prose-b3b prose_explain 3 0
    run code-b7 code_fibonacci 7 1
    run code-b3 code_fibonacci 3 1 ;;
  lookahead)  # the lookahead helper (4 blocks, exl3/lookahead.h) runs beside every MoeSgKernel; turn it off
    run prose-b3-la0 prose_explain 3 1 CRUCIBLE_LOOKAHEAD=0
    run prose-b7-la0 prose_explain 7 1 CRUCIBLE_LOOKAHEAD=0
    run g-prose-b3-la0 prose_explain 3 0 CRUCIBLE_LOOKAHEAD=0
    run g-prose-b7-la0 prose_explain 7 0 CRUCIBLE_LOOKAHEAD=0
    run g-prose-b7a prose_explain 7 0
    run g-prose-b3a prose_explain 3 0
    run g-prose-b3b prose_explain 3 0
    run g-prose-b7b prose_explain 7 0 ;;
  hold)  # modes 8 and 9: MoeSgKernel holds the lookahead helper while it runs
    run prose-b8 prose_explain 8 1
    run prose-b9 prose_explain 9 1
    for m in 3 8 9 7 7 9 8 3; do run g-prose-b$m-$SECONDS prose_explain $m 0; done
    for m in 3 8 9 9 8 3; do run g-code-b$m-$SECONDS code_fibonacci $m 0; done
    run g-prose-b8m32a prose_explain 8 0 CRUCIBLE_LOOKAHEAD_MIB=32
    run g-prose-b8m32b prose_explain 8 0 CRUCIBLE_LOOKAHEAD_MIB=32 ;;
  la)  # lookahead helper blocks and window (row_policy.h tuned 4 blocks, 24 MiB) at mode ${LA_MODE:-3}
    m=${LA_MODE:-3}
    for ids in prose_explain code_fibonacci; do
      for c in 4:24 8:24 4:32 8:32 6:24 2:24 4:16 6:32 4:24; do
        run g-$ids-b$m-la${c%:*}-m${c#*:}-$SECONDS $ids $m 0 CRUCIBLE_LOOKAHEAD=${c%:*} CRUCIBLE_LOOKAHEAD_MIB=${c#*:}
      done
    done ;;
  hold2)  # hold variants 10 to 14 (moe_b128_switch.h)
    for m in 3 10 11 13 12 12 13 11 10 3; do run g-prose-b$m-$SECONDS prose_explain $m 0; done
    for m in 3 10 13 14 8 3; do run g-code-b$m-$SECONDS code_fibonacci $m 0; done ;;
  hold3)  # the hold window around mode 12 (modes 15 to 19), prose
    for m in 3 12 15 16 17 18 19 19 18 17 16 15 12 3; do run g-prose-b$m-$SECONDS prose_explain $m 0; done ;;
  hold3c)  # the same, code
    for m in 3 12 15 16 17 18 19 19 18 17 16 15 12 3; do run g-code-b$m-$SECONDS code_fibonacci $m 0; done ;;
  hold4)  # mode 20 (12 and the GdnKernel hold) against 12 and 3
    for m in 3 12 20 20 12 3 12 20; do run g-prose-b$m-$SECONDS prose_explain $m 0; done
    for m in 3 12 20 20 12 3; do run g-code-b$m-$SECONDS code_fibonacci $m 0; done ;;
  grid)  # mode 12 grid caps (21: 40, 22: 60, 23: 80, 24: none)
    for m in 3 12 21 22 23 24 24 23 22 21 12 3; do run g-prose-b$m-$SECONDS prose_explain $m 0; done
    for m in 12 21 22 22 21 12; do run g-code-b$m-$SECONDS code_fibonacci $m 0; done ;;
  wide)  # modes 3 and 12 (ABBA) on four more prompts
    for ids in chat_explain code_json copy_python chat_spanish; do
      for m in 3 12 12 3; do run g-$ids-b$m-$SECONDS $ids $m 0; done
    done ;;
esac
