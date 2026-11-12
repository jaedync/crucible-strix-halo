"""The capture marker (canary). One fixed value, shared by the server, the tools and the export scan.

Every file that the draft capture writes carries this marker. So does every file that a tool makes from
such a file (a tokenized set, a trained draft head). The export scan and the guard test of each repository
fail when any file holds the marker, in any common encoding.

The value is computed, so no source file holds it as a literal. A file that holds it is a leak, with one
exception: none. This file is the same byte for byte in the server package, in the tools
folder of the private repository and in the export tool. A test checks the SHA-256 of the value.
"""

import hashlib
import os
from pathlib import Path

_SEED = b"crucible draft-capture canary v1: a file with this marker holds recordings or data made from them"
CANARY = "CRUCIBLE-CAPTURE-CANARY-v1-" + hashlib.sha256(_SEED).hexdigest()[:40]
CANARY_BYTES = CANARY.encode("ascii")
# A test pins this digest, so a change of the seed or of the prefix cannot pass unseen.
CANARY_SHA256 = hashlib.sha256(CANARY_BYTES).hexdigest()


def _inside(path, root):
    for candidate in (Path(os.path.abspath(path)), Path(os.path.realpath(path))):
        if candidate == root or root in candidate.parents:
            return True
    return False


def git_root_of(path):
    """The git work tree that holds PATH, or None.

    1. GIT_WORK_TREE in the environment names the work tree. GIT_DIR alone makes the current folder the work tree (this
       is how git itself reads them). A bare dotfiles repository with core.worktree has no .git above its work tree, so
       the walk below cannot see it; shield_folder() covers that case (outside check H10).
    2. Else the nearest folder above PATH (or PATH itself) that holds .git, as a directory or a file.

    The path as given and the path with symlinks resolved are both checked, so a link into a work tree does not help.
    """
    named = os.environ.get("GIT_WORK_TREE") or (os.getcwd() if os.environ.get("GIT_DIR") else None)
    if named:
        root = Path(os.path.abspath(named))
        if _inside(path, root) or _inside(path, Path(os.path.realpath(named))):
            return root
    for start in (Path(os.path.abspath(path)), Path(os.path.realpath(path))):
        for folder in (start, *start.parents):
            if os.path.lexists(folder / ".git"):
                return folder
    return None


SHIELD = ("*\n# " + CANARY + "\n").encode()


def shield_folder(path):
    """Make PATH a folder that no `git add` takes: a .gitignore with `*`, and the canary on a comment line (so a forced
    `git add -f` is still caught by the repository guard). It covers a work tree that the .git walk cannot see: a bare
    dotfiles repository, core.worktree, a worktree set only in the environment (outside check H10). Cheap and idempotent:
    call it before every write into the folder."""
    target = os.path.join(path, ".gitignore")
    try:
        with open(target, "rb") as handle:
            if handle.read(len(SHIELD) + 1) == SHIELD:
                return
    except OSError:
        pass
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(SHIELD)


def refuse_inside_git_tree(path, what="capture data"):
    """Raise ValueError when PATH lies inside a git work tree. Every tool that writes capture data or data made from it
    calls this for its output folder before it creates the folder."""
    root = git_root_of(path)
    if root is not None:
        raise ValueError(
            f"the folder {path} is inside the git work tree {root}. This folder holds {what}, and a commit must never "
            "carry it. Choose a folder outside every git work tree, for example ~/.local/share/crucible/draft-capture")
