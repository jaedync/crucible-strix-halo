#!/bin/bash
# Night check of the container image of public an earlier change (Crucible v0.32.0 for gfx1151) on the test machine.
#
# The image is built on the test machine from the public release tag v0.32.0 and shipped as a zstd docker-archive in this
# folder. This script loads it, runs the server from the image, waits for /health, sends one greedy chat on a
# synthetic prompt, then sends the same request to the native trunk binary and compares the token ids.
# Same pack, same sampler, same server options (--ctx 32768).
#
# If the ids differ, the script also runs the tree and the engine binary that it takes out of the image itself,
# natively and without a container. That answers the question a FAIL raises: is the difference the container, or
# only the two versions of the engine (v0.32.0 image against trunk)?
#
# The script always exits 0 and prints exactly one verdict line, so the night unit reads the line and not a return
# code: "PR5 DOCKER PASS <detail>" or "PR5 DOCKER FAIL <reason>".
#
# Ports 8937 (container) and 8938 (native server) are not the gateway ports (7000, 8000, 8001, 8731, 8790).
set -uo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
IMG_TAR=$HERE/crucible-v0.32.0.tar.zst
TREE=$HERE/tree
OUT=$HERE/unit-out/pr5
PACK=${PR5_PACK:-$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6}
NATIVE=${PR5_NATIVE:-$HOME/crucible-work/wt/t33/bin/crucible-t33-03f1a109}
PY=$HOME/crucible-work/refenv/.venv/bin/python
CPORT=8937
PPORT=8938
NAME=pr5-docker-$$
EXTRACT=$HERE/unit-out/image
PROMPT=${PR5_PROMPT:-"A workshop has 7 shelves and puts 12 grey boxes on each shelf. It moves 15 boxes away. Write one sentence with the number of boxes that stay."}

NPID=""

say() { printf '%s pr5: %s\n' "$(date -u +%H:%M:%S)" "$*"; }

stop_all() {
  podman stop -t 30 "$NAME" >/dev/null 2>&1
  podman rm -f "$NAME" >/dev/null 2>&1
  if [ -n "$NPID" ]; then kill -TERM "$NPID" >/dev/null 2>&1; wait "$NPID" 2>/dev/null; fi
}

fail() {
  say "PR5 DOCKER FAIL: $*"
  stop_all
  exit 0
}

trap 'stop_all' EXIT

mkdir -p "$OUT" || fail "cannot create $OUT"

# ---- phase 0: the inputs are there, and the GPU nodes are readable ----
[ -r "$IMG_TAR" ] || fail "no image archive at $IMG_TAR"
[ -d "$PACK" ] || fail "no pack at $PACK"
[ -x "$NATIVE" ] || fail "no native binary at $NATIVE"
[ -f "$TREE/VERSION" ] || fail "no server tree at $TREE"
[ -r /dev/kfd ] || fail "/dev/kfd is not readable by $(id -un)"
command -v podman >/dev/null || fail "no podman on this host"
command -v zstd >/dev/null || fail "no zstd on this host"
for p in $CPORT $PPORT; do
  if ss -ltn 2>/dev/null | grep -q ":$p "; then fail "port $p is already in use"; fi
done
# A container of an earlier run that a hard kill left behind holds the pack and the GPU: remove it by name prefix.
for c in $(podman ps -a --format '{{.Names}}' 2>/dev/null | grep '^pr5-docker-'); do
  say "remove a leftover container $c"
  podman rm -f "$c" >/dev/null 2>&1
done
say "inputs ok: archive $(stat -c %s "$IMG_TAR") B, pack $(basename "$PACK"), native $(basename "$NATIVE") $(sha256sum "$NATIVE" | cut -c1-16)"

# ---- phase 1: load the image ----
LOAD=$(zstd -dc "$IMG_TAR" | podman load 2>&1) || fail "podman load: $(printf '%s' "$LOAD" | tail -2 | tr '\n' ' ')"
REF=$(printf '%s\n' "$LOAD" | sed -n 's/^Loaded image: //p' | tail -1)
[ -n "$REF" ] || REF=crucible:v0.32.0
podman image exists "$REF" || fail "the archive holds no image tag ($(printf '%s' "$LOAD" | tail -1))"
IMG=$(podman image inspect --format '{{.Id}}' "$REF") || fail "cannot inspect the loaded image"
say "loaded $REF id $IMG size $(podman image inspect --format '{{.Size}}' "$REF") B"

# ---- phase 2: run the container ----
# <gpu-host> runs podman rootless, so the run line differs from the docker line in docs/DOCKER.md in three flags:
# `--group-add keep-groups` in place of the numeric video and render gids, `--ipc=host` and `--ulimit memlock=-1:-1`
# (the HIP runtime needs the host IPC namespace and locked memory). The image itself is not changed.
VGID=$(getent group video | cut -d: -f3)
RGID=$(getent group render | cut -d: -f3)
say "container start on port $CPORT (rootless podman: keep-groups; host gids video $VGID, render $RGID)"
CID=$(podman run -d --name "$NAME" \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups \
  --ipc=host --ulimit memlock=-1:-1 \
  --security-opt seccomp=unconfined \
  -v "$PACK":/models/pack:ro \
  -p 127.0.0.1:$CPORT:8937 \
  "$REF" /models/pack --host 0.0.0.0 --port 8937 --ctx 32768 2>&1) \
  || fail "podman run: $(printf '%s' "$CID" | tail -2 | tr '\n' ' ')"
say "container id $CID"

ready=no
for _ in $(seq 1 120); do
  if curl -sf -m 5 "http://127.0.0.1:$CPORT/health" >/dev/null 2>&1; then ready=yes; break; fi
  if ! podman container exists "$CID" >/dev/null 2>&1; then break; fi
  sleep 5
done
if [ "$ready" != yes ]; then
  podman logs --tail 60 "$CID" > "$OUT/container.log" 2>&1
  say "container log tail:"; tail -12 "$OUT/container.log"
  fail "the container did not answer /health on port $CPORT"
fi
say "container healthy: version $(podman exec "$CID" head -1 /opt/crucible/VERSION 2>/dev/null), api $(curl -sS -m 5 "http://127.0.0.1:$CPORT/v1/crucible/version" 2>/dev/null | head -c 160)"

# ---- phase 3: one greedy chat from the container ----
REQ=$(printf '{"model":"crucible","messages":[{"role":"user","content":%s}],"temperature":0,"max_tokens":32,"return_token_ids":true}' \
  "$(printf '%s' "$PROMPT" | "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")
curl -sS -m 900 "http://127.0.0.1:$CPORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" \
  -o "$OUT/chat-container.json" || fail "the container chat request failed"
"$PY" - "$OUT/chat-container.json" > "$OUT/ids-container.txt" 2>&1 <<'PY' || fail "the container answer has no token_ids: $(tail -2 "$OUT/ids-container.txt")"
import json, sys
d = json.load(open(sys.argv[1]))
ids = d["choices"][0].get("token_ids")
assert isinstance(ids, list) and ids, d.get("error") or list(d)[:4]
print(len(ids), ",".join(map(str, ids)))
PY
say "container answer: $(cut -d' ' -f1 "$OUT/ids-container.txt") token ids: $(cut -d' ' -f2- "$OUT/ids-container.txt" | cut -c1-60)"

# ---- phase 4: stop the container by its own ID ----
podman stop -t 30 "$CID" >/dev/null 2>&1 || say "podman stop returned nonzero"
podman rm -f "$CID" >/dev/null 2>&1 || say "podman rm returned nonzero"
podman container exists "$CID" >/dev/null 2>&1 && fail "the container $CID is still there"
say "container stopped and removed"

# ---- phase 5: the same chat on the native trunk binary ----
chat_native() {                       # $1 = tree, $2 = engine binary, $3 = output file
  local tree=$1 engine=$2 out=$3 rc=0
  ( cd "$tree" && exec "$PY" -m server "$PACK" --engine "$engine" --host 127.0.0.1 --port "$PPORT" --ctx 32768 ) \
    > "$OUT/native.log" 2>&1 &
  NPID=$!

  for _ in $(seq 1 120); do
    if curl -sf -m 5 "http://127.0.0.1:$PPORT/health" >/dev/null 2>&1; then break; fi
    kill -0 "$NPID" 2>/dev/null || break
    sleep 5
  done
  if ! curl -sf -m 5 "http://127.0.0.1:$PPORT/health" >/dev/null 2>&1; then
    tail -15 "$OUT/native.log"
    kill -TERM "$NPID" >/dev/null 2>&1; wait "$NPID" 2>/dev/null; NPID=""
    return 1
  fi
  curl -sS -m 900 "http://127.0.0.1:$PPORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" -o "$out" || rc=1
  kill -TERM "$NPID" >/dev/null 2>&1
  wait "$NPID" 2>/dev/null
  NPID=""
  return $rc
}

chat_native "$TREE" "$NATIVE" "$OUT/chat-native.json" || fail "the native server (trunk binary) did not answer on port $PPORT"
say "native trunk answer: $("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["choices"][0].get("token_ids") or []))' "$OUT/chat-native.json" 2>/dev/null) token ids"

# ---- phase 6: compare ----
compare() {                           # prints PASS/FAIL and the texts; rc 0 on equal ids
  "$PY" - "$OUT/chat-container.json" "$1" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
ia = a["choices"][0].get("token_ids") or []
ib = b["choices"][0].get("token_ids") or []
ta = a["choices"][0].get("message", {}).get("content", "")
tb = b["choices"][0].get("message", {}).get("content", "")
if not ia or not ib:
    print("FAIL a side has no token_ids: container %d, other %d" % (len(ia), len(ib)))
    sys.exit(1)
if ia == ib:
    print("PASS %d token ids equal: %s" % (len(ia), ",".join(map(str, ia[:16]))))
    sys.exit(0)
n = min(len(ia), len(ib))
k = next((i for i in range(n) if ia[i] != ib[i]), n)
print("FAIL ids differ at %d of %d: container %s | other %s" % (k, n, ia[max(0, k - 2):k + 3], ib[max(0, k - 2):k + 3]))
print("text equal: %s | container: %s | other: %s" % (ta == tb, ta[:100].replace("\n", " "), tb[:100].replace("\n", " ")))
sys.exit(1)
PY
}
VERDICT=$(compare "$OUT/chat-native.json"); CRC=$?
say "vs native trunk: ${VERDICT%%$'\n'*}"

# ---- phase 7: on a mismatch, take the tree and the binary out of the image and run them natively ----
if [ $CRC -ne 0 ]; then
  say "mismatch: run the image's own tree and binary without the container (attribution)"
  rm -rf "$EXTRACT" "$EXTRACT-tree"
  if podman create --name "$NAME-x" "$REF" >/dev/null 2>&1 \
     && podman cp "$NAME-x:/opt/crucible" "$EXTRACT-tree" >/dev/null 2>&1; then
    podman rm -f "$NAME-x" >/dev/null 2>&1
    if [ -x "$EXTRACT-tree/build/engine/crucible" ] \
       && chat_native "$EXTRACT-tree" "$EXTRACT-tree/build/engine/crucible" "$OUT/chat-image-native.json"; then
      IMGV=$(compare "$OUT/chat-image-native.json")
      say "image tree, native run: ${IMGV%%$'\n'*}"
      case "$IMGV" in
        PASS*) VERDICT="$VERDICT | the image's own binary agrees with the container, so the difference is the version (v0.32.0 vs trunk)";;
        *)     VERDICT="$VERDICT | the image's own binary ALSO differs from the container: the container changes the tokens";;
      esac
    else
      VERDICT="$VERDICT | the image tree did not run natively (see unit-out/pr5/native.log)"
    fi
  else
    VERDICT="$VERDICT | could not copy the tree out of the image"
  fi
  podman rm -f "$NAME-x" >/dev/null 2>&1
  rm -rf "$EXTRACT-tree"
fi

case "$VERDICT" in
  PASS*) say "PR5 DOCKER PASS ${VERDICT%%$'\n'*}" ;;
  *)     say "PR5 DOCKER FAIL $(printf '%s' "$VERDICT" | tr '\n' ' ' | cut -c1-700)" ;;
esac
