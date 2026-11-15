#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/../tools/work_paths.sh"
binary=${1:?binary}
tier=${2:?305 or 405}
ctx=${3:?context}
tag=${4:?unique tag}
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
  16384|65536|131072|262144) ;;
  *) echo 'invalid context: use 16384, 65536, 131072, or 262144' >&2; exit 2 ;;
esac
ids="results/prompts/long_$ctx.txt"
out="results/engine/$tier/$tag-$ctx"
mkdir -p "$(dirname "$out")"
mkdir "$out"
mkdir "$out/capture"
priority=2
deadline=300
capacity=$((ctx+4))
# The first output does not consume a cache slot. Keep the supported 256K bound.
if [[ "$ctx" == 262144 ]]; then priority=3; deadline=600; capacity=$ctx; fi
sha256sum "$binary" "$ids" >"$out/input.sha256"
"$WORK_GPU_RUN" -p "$priority" -w 7200 -- timeout -k 10 "$deadline" \
  env HSA_USERPTR_FOR_PAGED_MEM=0 CRUCIBLE_QSA_CAPTURE_DIR="$out/capture" \
  CRUCIBLE_QSA_CAPTURE_AT="$ctx" CRUCIBLE_QSA_CAPTURE_INDEX_ONLY=1 \
  "$binary" gen "$ckpt" --ids-file "$ids" --n 1 --prefill 4096 --spec 0 \
  --ctx "$capacity" --dump-logits "$out/logits.f32" >"$out/run.log" 2>&1
sha256sum --check "$out/input.sha256" >"$out/inputs-checked.log"
sha256sum "$out/logits.f32" "$out"/capture/*.i32 >"$out/output.sha256"
