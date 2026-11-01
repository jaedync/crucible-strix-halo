#!/bin/bash
# ABBA timing lease: PAIRS x (stock, tuned) spec_sweep runs of CFG, with the GPU clock, the socket power and the GPU busy
# percent sampled every 0.2 s during each run. The second arm of a pair runs about 0.7% faster (Oct 25), so the pairs run in
# ABBA blocks (stock tuned, tuned stock, tuned stock, stock tuned, ...) with equal numbers of each order; an odd PAIRS is
# rounded up to the next even number. A = stock, B = tuned. Logs: OUT.<i>.<arm>.log and OUT.<i>.<arm>.samples (lines
# "epoch sclk_hz power_uW busy_pct"), OUT.order ("<i> AB|BA" per pair) and OUT.env (platform_profile).
# Report: python3 training/mtp_tune/abab_report.py OUT
#   abab.sh CKPT CFG OUT SIDECAR PAIRS
# Test hooks: ABAB_SWEEP (the spec_sweep binary), ABAB_SYSFS, ABAB_PROFILE, ABAB_PERIOD.
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
ck=$1 cfg=$2 out=$3 side=$4 pairs=$5
sweep=${ABAB_SWEEP:-build/bench/spec_sweep} period=${ABAB_PERIOD:-0.2}
order_py=(${PY:-python3} -I tools/ab_order.py)
freq=$(ls "${ABAB_SYSFS:-/sys}"/class/drm/card*/device/hwmon/hwmon*/freq1_input | head -1)
hw=$(dirname "$freq")
echo "platform_profile $(cat "${ABAB_PROFILE:-/sys/firmware/acpi/platform_profile}")" > "$out.env"
want=$pairs; pairs=$("${order_py[@]}" pairs "$pairs")
[ "$pairs" != "$want" ] && echo "abab.sh: PAIRS $want is odd; running $pairs pairs so that the stock-first and tuned-first pairs are equal in number"
: > "$out.order"
for i in $(seq 1 "$pairs"); do
  order=$("${order_py[@]}" order $((i - 1)))
  echo "$i $order" >> "$out.order"
  arms="stock tuned"; [ "$order" = BA ] && arms="tuned stock"
  for arm in $arms; do
    ( while :; do echo "$(date +%s.%N) $(cat "$hw/freq1_input") $(cat "$hw/power1_average") $(cat "$hw/../../gpu_busy_percent")"; sleep "$period"; done ) > "$out.$i.$arm.samples" &
    sampler=$!
    if [ $arm = tuned ]; then
      CRUCIBLE_MTP_SIDECAR="$side" "$sweep" "$ck" "$cfg" > "$out.$i.$arm.log" 2>&1
    else
      env -u CRUCIBLE_MTP_SIDECAR "$sweep" "$ck" "$cfg" > "$out.$i.$arm.log" 2>&1
    fi
    kill "$sampler"
  done
done
