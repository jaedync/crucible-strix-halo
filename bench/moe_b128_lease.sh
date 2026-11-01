#!/bin/bash
# One moe-b128 bench lease body (run it under gpu-run): verify_moe on the decode route dump, 8 layers, 192 calls,
# R = 1, 2, 4, 8. Two passes (A then B order of the same list) so each variant has 2 samples in the lease.
# Usage: moe_b128_lease.sh OUT_DIR BIN [VARIANTS] [RLIST]
set -uo pipefail
out=$1 bin=$2
vars=${3:-0,76,110,111,112,113,114,118,119,115,116,117,120,121,122,123,124,125,126,0}
rl=${4:-1,2,4,8}
routes=$HOME/crucible-work/wt/verify-moe/res/l1/routes/routes.bin
mkdir -p "$out"
echo "start $(date +%T)"
for pass in 1 2; do
  rc=0
  timeout --foreground -k 5 200 "$bin" "$routes" 8 192 "$rl" 8 "$vars" > "$out/bench$pass.txt" 2>&1 || rc=$?
  echo "pass $pass rc=$rc $(date +%T)"
  grep -m1 -i "sclk\|clock" /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | head -1 || true
done
cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | head -4 || true
echo "end $(date +%T)"
