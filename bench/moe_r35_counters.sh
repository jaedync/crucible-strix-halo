#!/bin/bash
# moe-r35 counter lease body (diagnostic, gates nothing; run under gpu-run at p0, at most 10 min): rocprofv3 SQ
# counters of MoeSgKernel at R = 3, 4, 5 for v170 (default), 180, 181 and 182 (bench/moe_sg_engine.hip). The
# question: does the one-call-site kernel wait less for instruction issue (SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES)?
# gfx1151 lists no SQC_ICACHE counter; a counter set that rocprofv3 cannot collect fails alone (its rc is logged).
# Usage: moe_r35_counters.sh OUT BENCH      summary: python3 -I bench/moe_r35_counters.py OUT
set -uo pipefail
ulimit -c 0
out=$1 bench=$2
routes=$HOME/crucible-work/wt/verify-moe/res/l1/routes/routes.bin
mkdir -p "$out"
echo "start $(date +%T)"
sets=("SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAIT_INST_ANY" "SQ_WAIT_ANY SQ_WAVES SQ_INSTS_VALU" "SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_SMEM")
n=0
for s in "${sets[@]}"; do
  n=$((n + 1))
  rc=0
  timeout --foreground -k 5 150 /opt/rocm/bin/rocprofv3 --pmc $s --kernel-include-regex MoeSgKernel -f csv \
    -d "$out/set$n" -o run -- "$bench" "$routes" 3,4,5 0,170,180,181,182 engine 1 24 8 > "$out/set$n.txt" 2>&1 || rc=$?
  echo "set $n ($s) rc=$rc $(date +%T) $(grep '^SUMMARY' "$out/set$n.txt")"
done
cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | head -4 || true
echo "end $(date +%T)"
