#!/usr/bin/env bash
# Engine proofs for the default no-lo union flash: pfbench last-position logits
# at 1K/4K/16K against the prefill-nx binary's saved dumps, the LO split path,
# a 4K profile pair, and one 64K gen pair. One queued job per step.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
cand=${1:?candidate engine}
base=${2:?baseline engine (prefill-nx 6c5b9fe)}
saved=${3:?baseline dump root with nx, nxs, nx4}
out=${4:?output root}
C3=$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
C4=$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6
P=$WORK_DIR/wt/prefill-scan/results/prompts
runs="n1k:$P/source-1024.ids:4096:0:0:0:0:0:0 n4k:$P/source-4096.ids:4096:0:0:0:0:0:0 n16k:$P/source-16384.ids:4096:0:0:0:0:0:0"
mkdir -p "$out"/{d305,s305,d405,b305p,g64}
sha256sum "$cand" "$base" "$P"/source-{1024,4096,16384}.ids "$saved"/{nx,nxs,nx4}/*.logits.f32 >"$out/inputs.sha256"
q() { "$WORK_GPU_RUN" -p 2 -w 7200 -- timeout -k 10 300 "$@"; }
cat >"$out/job305.sh" <<JOB
set -u
export HSA_USERPTR_FOR_PAGED_MEM=0
CRUCIBLE_PROFILE=1 '$base' pfbench '$C3' --prefill 4096 n4k:$P/source-4096.ids:4096:0:0:0:0:0:0 >'$out/b305p/log.txt' 2>&1; echo base-profile rc \$?
CRUCIBLE_PROFILE=1 CRUCIBLE_PF_OUT='$out/d305' '$cand' pfbench '$C3' --prefill 4096 $runs >'$out/d305/log.txt' 2>&1; echo default rc \$?
CRUCIBLE_PF_EXACT_SPLIT=1 CRUCIBLE_PF_OUT='$out/s305' '$cand' pfbench '$C3' --prefill 4096 $runs >'$out/s305/log.txt' 2>&1; echo split rc \$?
JOB
q bash "$out/job305.sh" >"$out/job305.log" 2>&1
cat >"$out/job405.sh" <<JOB
set -u
export HSA_USERPTR_FOR_PAGED_MEM=0
CRUCIBLE_PF_OUT='$out/d405' '$cand' pfbench '$C4' --prefill 4096 $runs >'$out/d405/log.txt' 2>&1; echo default405 rc \$?
JOB
q bash "$out/job405.sh" >"$out/job405.log" 2>&1
ids=$WORK_DIR/wt/qsa-prefill/results/prompts/long_65536.txt
sha256sum "$ids" >>"$out/inputs.sha256"
cat >"$out/job64.sh" <<JOB
set -u
export HSA_USERPTR_FOR_PAGED_MEM=0
CRUCIBLE_PROFILE=1 '$base' gen '$C3' --ids-file '$ids' --n 1 --prefill 4096 --spec 0 --ctx 65540 --dump-logits '$out/g64/base.f32' >'$out/g64/base.log' 2>&1; echo base64 rc \$?
CRUCIBLE_PROFILE=1 '$cand' gen '$C3' --ids-file '$ids' --n 1 --prefill 4096 --spec 0 --ctx 65540 --dump-logits '$out/g64/cand.f32' >'$out/g64/cand.log' 2>&1; echo cand64 rc \$?
JOB
q bash "$out/job64.sh" >"$out/job64.log" 2>&1
sha256sum --check "$out/inputs.sha256" >"$out/inputs-checked.log"
for t in n1k n4k n16k; do
  cmp -s "$out/d305/$t.logits.f32" "$saved/nx/$t.logits.f32" && echo "$t 3.05 default EQUAL" || echo "$t 3.05 default DIFFERS"
  cmp -s "$out/s305/$t.logits.f32" "$saved/nxs/$t.logits.f32" && echo "$t 3.05 split EQUAL" || echo "$t 3.05 split DIFFERS"
  cmp -s "$out/d405/$t.logits.f32" "$saved/nx4/$t.logits.f32" && echo "$t 4.05 default EQUAL" || echo "$t 4.05 default DIFFERS"
done
cmp -s "$out/g64/base.f32" "$out/g64/cand.f32" && echo "64K 3.05 EQUAL" || echo "64K 3.05 DIFFERS"
echo proof_complete
