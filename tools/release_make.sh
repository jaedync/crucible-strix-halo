#!/usr/bin/env bash
# The build step of `crucible update` (set CRUCIBLE_MAKE=tools/release_make.sh).
# An install that uses the release engine gets the release engine of the new tag: it is downloaded and checked against
# SHA256SUMS by `install.sh engine`. Any other case runs `make`: a source build install, a commit that is not a release,
# or a release engine that cannot be fetched or cannot run on this machine.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P) || exit 1
cd "$root" || exit 1
if [ -f build/engine/release.json ] && [ -f install.sh ]; then
  if bash ./install.sh engine --dir "$root"; then
    exit 0
  fi
  echo "release_make: no release engine for this version, so make builds the engine from source (about 16 minutes)." >&2
fi
exec make
