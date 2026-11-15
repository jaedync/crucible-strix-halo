#!/bin/bash
# Gate 1(b) oracle over a dump dir, CPU only (no GPU lock). Usage: tools/pl_oracle.sh DUMPDIR
set -uo pipefail
. "$(dirname "$0")/work_paths.sh"
cd "$(dirname "$0")/.."
# PL_CKPT selects another checkpoint (4.05: pass its baseline to pl_gate1.py).
CKPT=${PL_CKPT:-~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
FIB=$WORK_FIB_IDS
PY=$WORK_DIR/refenv/.venv/bin/python
d=$1
# One oracle process at a time, nice 5, 8 threads.
for l in $(seq 0 47); do
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 nice -n 5 timeout -k 10 600 $PY ref/oracle.py $CKPT $d $l --pos 0,14,23 --ids $FIB > $d.oracle.$l.txt 2>&1
done
cat $(for l in $(seq 0 47); do echo $d.oracle.$l.txt; done) > $d.oracle.txt
$PY tools/pl_gate1.py $d.oracle.txt ${PL_BASE:-}
