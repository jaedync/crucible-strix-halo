#!/usr/bin/env bash
# Profiles each engine in turn inside one queued job, so both see the same box state.
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
tier=${1:?305 or 405}
ctx=${2:?context}
tag=${3:?unique tag}
shift 3
(( $# >= 1 )) || { echo 'need at least one engine' >&2; exit 2; }
if [[ ! "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ || ${#tag} -gt 80 ]]; then
  echo 'invalid tag: use 1 to 80 letters, digits, periods, underscores, or hyphens' >&2
  exit 2
fi
case "$tier" in
  305) ckpt="$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5" ;;
  405) ckpt="$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6" ;;
  *) echo 'invalid tier: use 305 or 405' >&2; exit 2 ;;
esac
case "$ctx" in
  16384|65536|131072) ;;
  *) echo 'invalid context: use 16384, 65536, or 131072' >&2; exit 2 ;;
esac
ids="results/prompts/long_$ctx.txt"
out="results/profile/$tier/$tag-$ctx"
mkdir -p "$(dirname "$out")"
mkdir "$out"
sha256sum "$@" "$ids" >"$out/input.sha256"
script="$out/job.sh"
{
  echo 'set -u'
  i=0
  for bin in "$@"; do
    echo "env HSA_USERPTR_FOR_PAGED_MEM=0 CRUCIBLE_PROFILE=1 '$bin' gen '$ckpt' --ids-file '$ids' --n 1 --prefill 4096 --spec 0 --ctx $((ctx+4)) >'$out/run$i.log' 2>&1; echo \"run$i rc \$?\""
    i=$((i+1))
  done
} >"$script"
"$WORK_GPU_RUN" -p 2 -w 7200 -- timeout -k 10 300 bash "$script" >"$out/job.log" 2>&1
sha256sum --check "$out/input.sha256" >"$out/inputs-checked.log"
