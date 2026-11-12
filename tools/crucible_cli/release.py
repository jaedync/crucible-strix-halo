"""The release engine record that install.sh writes, and the check that the engine on the disk is still that file."""
import hashlib
import json
import re
from pathlib import Path

RECORD = "build/engine/release.json"
ENGINE = "build/engine/crucible"
KEYS = ("format", "tag", "commit", "engine_sha256", "arch", "rocm", "asset")
TAG = re.compile(r"^v\d+\.\d+\.\d+$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def engine_release(repo):
    """The record of the release engine in REPO, or None.

    None when there is no record, the record is not valid, or the engine file is not the file that the record names
    (for example a build from source replaced it). A symbolic link is never accepted."""
    repo = Path(repo)
    engine = repo / ENGINE
    try:
        record = json.loads((repo / RECORD).read_text())
        if engine.is_symlink() or not engine.is_file():
            return None
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("format") != 1 or any(k not in record for k in KEYS):
        return None
    if not (isinstance(record["tag"], str) and TAG.fullmatch(record["tag"])):
        return None
    if not (isinstance(record["engine_sha256"], str) and HEX64.fullmatch(record["engine_sha256"])):
        return None
    try:
        if _sha256(engine) != record["engine_sha256"]:
            return None
    except OSError:
        return None
    return record
