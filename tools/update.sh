#!/usr/bin/env bash
# Update this checkout to the newest Crucible release, or undo the last update.
#   tools/update.sh              fetch the release tags, check out the newest, rebuild, self-test
#   tools/update.sh --rollback   go back to the previous tag and the previous binary, and hold the release you left
# After a rollback, the update skips that release (build/update-hold) until a newer release exists. Without this hold,
# the daily timer would install the same release again. Delete build/update-hold to install it anyway.
# `make update` and `make rollback` call it. The update never discards local work: it stops at once when a tracked
# file has changes, or when HEAD has commits that the newest release does not have.
# Environment: CKPT=<model folder> adds the GPU self-test (a few greedy tokens against the release's expected IDs);
# CRUCIBLE_UPDATE_REMOTE (origin), CRUCIBLE_MAKE (make) and PYTHON (python3) name the tools; CRUCIBLE_RESTART_UNIT
# names a systemd user unit to restart after a good update (the timer template in deploy/user-systemd sets it).
#
# An update can be killed half way (the unit is stopped, the machine loses power). It first writes build/update-running
# (the ref to go back to, and whether a binary existed) and removes it only after the update passed or the restore is
# done. The next `make update` or `make rollback` sees the file and finishes the restore first, with no build.
#
# The body is one function and the script ends with `exit`: the checkout replaces this file while it runs, and bash
# reads a script as it goes. The function is read in full before it runs.

main() {
  set -u
  local root remote make_cmd python prev_file hold_file running_file engine prev_engine
  root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P) || return 1  # git reports the physical path
  cd "$root" || return 1
  remote=${CRUCIBLE_UPDATE_REMOTE:-origin}
  make_cmd=${CRUCIBLE_MAKE:-make}
  python=${PYTHON:-python3}
  prev_file=build/update-prev-tag
  hold_file=build/update-hold
  running_file=build/update-running
  engine=build/engine/crucible
  prev_engine=build/engine/crucible.prev

  fail() { echo "update: $*" >&2; return 1; }

  check_checkout() {
    [ "$(git rev-parse --show-toplevel 2>/dev/null)" = "$root" ] ||
      fail "This folder is not a git checkout of Crucible. Clone https://github.com/jaedync/crucible-strix-halo to update." || return 1
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
      fail "You have local changes. Commit or stash them, then run this again. Nothing was changed." || return 1
    fi
  }

  newest_release() {
    git tag --list 'v[0-9]*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1
  }

  # The tag that HEAD sits on, else its commit. A rollback goes back to exactly this.
  current_ref() {
    local tag
    tag=$(git tag --points-at HEAD --list 'v[0-9]*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1)
    if [ -n "$tag" ]; then echo "$tag"; else git rev-parse HEAD; fi
  }

  # Best effort: ask the disk to write a file out. A failure of this call must not stop the update.
  flush() { sync -- "$@" 2>/dev/null || true; }

  # Write a file in one step: a kill leaves the old file or the whole new one, never half of it.
  write_whole() {
    local file=$1; shift
    printf '%s\n' "$@" > "$file.tmp" || return 1
    flush "$file.tmp"
    mv -f "$file.tmp" "$file"
  }

  # Go back to $1 (a tag or a commit) and to the previous binary. $2 is 1 when a binary existed before the update.
  # Safe to run again after a kill at any point, and the marker goes last:
  # - the previous binary is moved over the engine in one step (mv). When it is gone and $2 is 1, that move is done
  #   (or the copy never started, and the engine is still the previous one), so the engine stays.
  # - $2 is 0: there was no binary before the update, so none should remain, and a half-built one goes with it.
  restore() {
    local ref=$1 had=${2:-1}
    echo "update: restoring $ref and the previous binary."
    git checkout --quiet --detach "$ref" || return 1
    if [ -f "$prev_engine" ]; then
      mv -f "$prev_engine" "$engine" || return 1
    elif [ "$had" = 0 ]; then
      rm -f "$engine"
    fi
    # The linker writes build/engine/crucible.tmpXXXXXXX and renames it, so a kill leaves the old binary or none, never
    # half of one. Only the leftover temporary files go here.
    rm -f "$prev_file" "$engine".tmp* "$prev_engine.tmp" || return 1
    rm -f "$running_file"
  }

  # Finish an update or a rollback that a kill stopped, before anything else. It restores and never builds: a build
  # can be stopped again by the same cause (a memory limit, a timer that stops the unit), and a restore cannot. Each
  # run then ends in a known release, and the restore leaves the checkout and the binary of one release.
  heal() {
    [ -f "$running_file" ] || return 0
    local ref had
    ref=$(sed -n 1p "$running_file")
    had=$(sed -n 2p "$running_file")
    if ! { [[ $ref =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || [[ $ref =~ ^[0-9a-f]{40}$ ]]; } || ! [[ $had =~ ^[01]$ ]]; then
      echo "update: $running_file is not valid. Removing it." >&2
      rm -f "$running_file"
      return 0
    fi
    echo "update: an earlier update was stopped before it finished. Going back to $ref first."
    restore "$ref" "$had" || { fail "The restore failed. Run git status to see the state of the checkout. Nothing else was changed."; return 1; }
  }

  self_test() {
    local tag=$1
    "$python" -I tools/update_selftest/selftest.py cpu "${tag#v}" || return 1
    if [ -n "${CKPT:-}" ]; then
      "$python" -I tools/update_selftest/selftest.py gpu "$CKPT"
      case $? in 0|3) ;; *) return 1 ;; esac  # 3: this release ships no expected IDs for the pack. The tool said so.
    else
      echo "update: GPU self-test skipped. Stop the server and run: make update CKPT=<model folder> to include it."
    fi
  }

  rollback() {
    local left target had
    check_checkout || return 1
    heal || return 1
    [ -f "$prev_file" ] || { fail "There is no earlier update to undo."; return 1; }
    target=$(cat "$prev_file")
    left=$(current_ref)
    had=0; [ -f "$prev_engine" ] && had=1
    # Hold the release that the user left first, so a rollback that a kill stops half way is still held.
    if [[ $left =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then write_whole "$hold_file" "$left" || return 1; fi
    write_whole "$running_file" "$target" "$had" || return 1
    if ! restore "$target" "$had"; then
      fail "The rollback failed. Run git status to see the state of the checkout."
      return 1
    fi
    echo "update: rolled back to $(current_ref)."
    if [[ $left =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "update: $left is on hold. make update waits for a newer release. To install $left anyway, delete $hold_file."
    fi
  }

  # 0 (true) when the newest release is the one that a rollback left, or an older one. A newer release lifts the hold.
  on_hold() {
    local newest held
    [ -f "$hold_file" ] || return 1
    held=$(cat "$hold_file")
    if ! [[ $held =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then rm -f "$hold_file"; return 1; fi
    newest=$(printf '%s\n%s\n' "$held" "$1" | sort -V | tail -n 1)
    if [ "$newest" = "$1" ] && [ "$held" != "$1" ]; then rm -f "$hold_file"; return 1; fi
    echo "Release $1 is on hold because you rolled back from $held. Nothing was changed. To install it anyway, delete $PWD/$hold_file and run this again."
    return 0
  }

  restart_unit() {
    local unit=${CRUCIBLE_RESTART_UNIT:-}
    [ -n "$unit" ] || return 0
    if ! [[ $unit =~ ^[A-Za-z0-9@_.:-]+\.service$ ]]; then
      echo "update: CRUCIBLE_RESTART_UNIT must name a .service unit. No restart." >&2
      return 0
    fi
    systemctl --user try-restart -- "$unit" || echo "update: could not restart $unit. Restart it yourself." >&2
  }

  update() {
    local target before_commit target_commit
    check_checkout || return 1
    heal || return 1
    git fetch --tags --quiet "$remote" || { fail "Could not fetch the release tags from '$remote'. Check your network and try again. Nothing was changed."; return 1; }
    target=$(newest_release)
    [ -n "$target" ] || { fail "No release tag found. Nothing was changed."; return 1; }
    on_hold "$target" && return 0
    before_commit=$(git rev-parse HEAD)
    target_commit=$(git rev-parse "$target^{commit}")
    if [ "$before_commit" = "$target_commit" ]; then
      echo "Crucible is up to date ($target)."
      return 0
    fi
    if git merge-base --is-ancestor "$target_commit" "$before_commit"; then
      echo "This checkout is ahead of the newest release ($target). Nothing to update."
      return 0
    fi
    if ! git merge-base --is-ancestor "$before_commit" "$target_commit"; then
      fail "HEAD is not on the release line: it has commits that $target does not have. Nothing was changed. To follow the releases, run: git checkout <release tag>"
      return 1
    fi

    local previous had=0
    previous=$(current_ref)
    mkdir -p build/engine || return 1
    [ -f "$engine" ] && had=1
    # The marker comes first. From here on, a kill at any point leaves what heal needs. The old copy goes before the
    # new one lands: an old copy (of an earlier release) must never be moved over the engine of this release.
    write_whole "$running_file" "$previous" "$had" || return 1
    rm -f "$prev_engine" || return 1
    if [ "$had" = 1 ]; then
      cp -p "$engine" "$prev_engine.tmp" && flush "$prev_engine.tmp" && mv -f "$prev_engine.tmp" "$prev_engine" ||
        { rm -f "$prev_engine.tmp" "$running_file"; return 1; }
    fi
    write_whole "$prev_file" "$previous" || { rm -f "$running_file"; return 1; }

    echo "update: $previous -> $target"
    git checkout --quiet --detach "$target" || { fail "The checkout of $target failed. Nothing was changed."; rm -f "$prev_file" "$running_file"; return 1; }
    echo "update: building (this takes a while)."
    if ! "$make_cmd"; then
      echo "update: the build of $target failed." >&2
      restore "$previous" "$had" || fail "The restore failed. Run git status to see the state of the checkout."
      return 1
    fi
    if ! self_test "$target"; then
      echo "update: $target did not pass the self-test." >&2
      restore "$previous" "$had" || fail "The restore failed. Run git status to see the state of the checkout."
      return 1
    fi
    rm -f "$running_file" || return 1  # the update passed: the checkout and the binary are of one release
    echo "update: now on $target. The previous binary is $prev_engine. To go back: make rollback"
    restart_unit
  }

  mkdir -p build || return 1
  if command -v flock >/dev/null 2>&1; then
    exec 9>build/.update.lock
    flock -n 9 || { fail "Another update is running."; return 1; }
  fi
  case "${1:-}" in
    "") update ;;
    --rollback) rollback ;;
    *) echo "usage: tools/update.sh [--rollback]" >&2; return 2 ;;
  esac
}

main "$@"
exit $?
