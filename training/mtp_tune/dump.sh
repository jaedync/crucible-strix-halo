#!/bin/bash
# One dump lease: mtp_dump of one LIST shard into DATA/<set>/stk-<shard>/.
#   dump.sh SET SHARD
set -u
cd "$(dirname "$0")/../.."
. training/mtp_tune/env.sh
mkdir -p "$DATA/$1/stk-$2"
# Leak guard: the launcher gives the child an allow-listed environment, umask 077, no core file, a private work folder.
"$PY" -m training.mtp_tune.launch run --out "$DATA/$1/stk-$2" -- build/bench/mtp_dump "$CK305" "$DATA/$1/$2.list" "$DATA/$1/stk-$2" --ctx 4096 2>&1 | grep -v "^prefill plan"
# Leak guard: mtp_dump drops the canary of a tainted list. Put it back into every .stk and .ids (taint.py).
"$PY" -m training.mtp_tune.taint tag-dump "$DATA/$1/$2.list" "$DATA/$1/stk-$2" || exit 1
