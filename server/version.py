"""The Crucible version: the VERSION file at the repository root, and the git commit when the root is a checkout."""
import re
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = "unknown"
# A release is 0.<train>.<fix>. The file holds no leading v.
VERSION_PATTERN = re.compile(r"\d{1,6}\.\d{1,6}\.\d{1,6}", re.ASCII)
GIT_TIMEOUT_SECONDS = 2


def read_version(root=ROOT):
    try:
        text = (Path(root) / "VERSION").read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        return UNKNOWN
    return text if VERSION_PATTERN.fullmatch(text) else UNKNOWN


def git_commit(root=ROOT):
    """The short commit of HEAD, or None. Only a checkout of its own counts: a tarball unpacked inside another
    repository must not report that repository's commit."""
    if not (Path(root) / ".git").exists():
        return None
    try:
        done = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=9", "HEAD"], capture_output=True,
                              text=True, timeout=GIT_TIMEOUT_SECONDS, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = done.stdout.strip()
    return commit if re.fullmatch(r"[0-9a-f]{7,40}", commit) else None


def at_release_tag(root, version):
    """True when HEAD is the commit of the tag v<version> (plain or annotated)."""
    try:
        done = subprocess.run(["git", "-C", str(root), "tag", "--points-at", "HEAD", "--list", f"v{version}"],
                              capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.stdout.split() == [f"v{version}"]


def build_label(root=ROOT):
    """What this build is, for the owner: "v0.37.0" for a release (a tree without git, or a checkout of the tag
    v<VERSION>), else "development build <commit>, based on v0.37.0". VERSION names the last release it is based on."""
    version, commit = read_version(root), git_commit(root)
    if commit is None or (version != UNKNOWN and at_release_tag(root, version)):
        return f"v{version}"
    return f"development build {commit}, based on v{version}"


def startup_line(root=ROOT):
    label, commit = build_label(root), git_commit(root)
    return f"Crucible {label}" + (f" ({commit})" if commit and label.startswith("v") else "")


@lru_cache(maxsize=1)
def server_label():
    return build_label()


@lru_cache(maxsize=1)
def server_version():
    return read_version()


@lru_cache(maxsize=1)
def server_commit():
    return git_commit()
