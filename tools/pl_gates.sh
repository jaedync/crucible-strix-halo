#!/bin/bash
# Engine gates for a prefill build in one locked block. Usage: tools/pl_gates.sh TAG BIN
# Writes results/TAG/: prefill24 dumps (Gate 1(b) input), chunk1/chunk8 dumps
# (Gate 2), spec0/spec2 ids after a 1024-row prefill (Gate 3).
set -uo pipefail
. "$(dirname "$0")/work_paths.sh"
cd "$(dirname "$0")/.."
# PL_CKPT selects another checkpoint (4.05: pass its baseline to pl_gate1.py).
CKPT=${PL_CKPT:-~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
FIB=$WORK_FIB_IDS
SRC=$WORK_DIR/wt/prefill-scan/results/prompts/source-1024.ids
tag=$1; bin=$2; d=results/$tag
if [[ ! $tag =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || -e $d ]]; then
  echo "use a new gate tag with only letters, digits, dots, underscores, or hyphens" >&2
  exit 2
fi
mkdir -p results || exit 1
# Reserve the tag atomically before another process can use the same captures.
mkdir "$d" || exit 2
mkdir -p "$d/p24" "$d/c1" "$d/c8" || exit 1
"${GPU_RUN:-$WORK_GPU_RUN}" -p 1 -- timeout -k 10 290 bash -e -c '
  bin=$1; ckpt=$2; fib=$3; src=$4; d=$5
  "$bin" gen "$ckpt" --ids-file "$fib" --n 1 --prefill 24 --dump "$d/p24" > "$d/p24.txt" 2>&1
  "$bin" gen "$ckpt" --ids-file "$fib" --n 1 --chunk 1 --dump "$d/c1" > "$d/c1.txt" 2>&1
  "$bin" gen "$ckpt" --ids-file "$fib" --n 1 --chunk 8 --dump "$d/c8" > "$d/c8.txt" 2>&1
  "$bin" gen "$ckpt" --ids-file "$src" --n 160 --prefill 1024 --spec 0 > "$d/spec0.txt" 2>&1
  "$bin" gen "$ckpt" --ids-file "$src" --n 160 --prefill 1024 --spec 2 > "$d/spec2.txt" 2>&1
' _ "$bin" "$CKPT" "$FIB" "$SRC" "$d"
rc=$?
echo "exit $rc" > $d/exit.txt
(( rc == 0 )) || exit "$rc"
python3 - "$d" "$FIB" <<'PY' || exit 1
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
ids = [int(x) for x in re.split(r"[,\s]+", Path(sys.argv[2]).read_text().strip()) if x]
# Every residual contains all prompt rows, with 10240 fp32 values per row.
size = len(ids) * 10240 * 4
expected = [f"h_{i:02d}.f32" for i in range(49)]
if not size:
    raise SystemExit("gate2: the prompt must contain token IDs")
for name in ("c1", "c8"):
    files = sorted((root / name).glob("h_*.f32"))
    if [f.name for f in files] != expected or any(f.stat().st_size != size for f in files):
        raise SystemExit(f"gate2: {name} needs 49 complete residual captures")
for name in expected:
    if (root / "c1" / name).read_bytes() != (root / "c8" / name).read_bytes():
        raise SystemExit(f"gate2: residual mismatch in {name}")
PY
echo "gate2: 49 of 49 chunk-1/chunk-8 dumps bitwise equal" | tee $d/gate2.txt
a=$(grep '^ids:' $d/spec0.txt); b=$(grep '^ids:' $d/spec2.txt)
if [[ -z $a || $a != "$b" || $(echo "$a" | wc -w) != 161 ]]; then
  echo "gate3: FAIL" | tee $d/gate3.txt
  exit 1
fi
echo "gate3: PASS 161 ids" | tee $d/gate3.txt
grep -h "tok/s" $d/spec0.txt $d/spec2.txt
