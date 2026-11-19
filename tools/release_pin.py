#!/usr/bin/env python3
"""Trust root Part A: set PIN_TAG and PIN_SUMS_SHA256 in install.sh to a release and the sha256 of its SHA256SUMS.

Use (after `gh release create`, in the public checkout of main):
    python3 -I tools/release_pin.py install.sh vX.Y.Z dist/release-vX.Y.Z/SHA256SUMS
then `bash -n install.sh`, the install test, and one commit with the printed subject on main.

Scope: the pin makes a replaced release file visible as a commit on main. It does not protect against a person who can
push to main. Standard library only. Exit 1 on an error, with no change to the file.
"""
import hashlib
import re
import sys
from pathlib import Path

TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
STEM = "crucible-engine-{tag}-gfx1151-rocm7.2-linux-x86_64"
PIN_TAG_RE = re.compile(r'^PIN_TAG="(v\d+\.\d+\.\d+)"$', re.M)
PIN_SUMS_RE = re.compile(r'^PIN_SUMS_SHA256="([0-9a-f]{64})?"$', re.M)


class PinError(Exception):
    pass


def _version(tag):
    m = TAG_RE.match(tag)
    if not m or tag != tag.strip():
        raise PinError(f"the tag must look like v0.37.2, not {tag!r}")
    return tuple(int(x) for x in m.groups())


def pin(install_sh, tag, sums_file):
    """Write the pin into INSTALL_SH. Returns the commit subject. Raises PinError and changes nothing on an error."""
    want = _version(tag)
    path = Path(install_sh)
    text = path.read_text()
    old_tag, old_sums = PIN_TAG_RE.search(text), PIN_SUMS_RE.search(text)
    if not old_tag or not old_sums or len(PIN_TAG_RE.findall(text)) != 1 or len(PIN_SUMS_RE.findall(text)) != 1:
        raise PinError(f"{path} must have one PIN_TAG line and one PIN_SUMS_SHA256 line")
    if want < _version(old_tag.group(1)):
        raise PinError(f"{tag} is lower than the pinned {old_tag.group(1)}")
    data = Path(sums_file).read_bytes()
    names = {ln.split()[-1].lstrip("*") for ln in data.decode(errors="replace").splitlines() if len(ln.split()) == 2}
    stem = STEM.format(tag=tag)
    missing = [n for n in (stem + ".gz", stem + ".needs") if n not in names]
    if missing:
        raise PinError(f"{sums_file} does not list {', '.join(missing)}: it is not the SHA256SUMS of {tag}")
    sha = hashlib.sha256(data).hexdigest()
    text = PIN_TAG_RE.sub(f'PIN_TAG="{tag}"', text, count=1)
    text = PIN_SUMS_RE.sub(f'PIN_SUMS_SHA256="{sha}"', text, count=1)
    path.write_text(text)
    return f"release: pin SHA256SUMS of {tag} in install.sh"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    try:
        subject = pin(argv[0], argv[1], argv[2])
    except (PinError, OSError) as exc:
        print(f"release_pin.py: {exc}", file=sys.stderr)
        return 1
    print(subject)
    return 0


if __name__ == "__main__":
    sys.exit(main())
