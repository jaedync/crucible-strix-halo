"""Which HIP runtime does this machine have, and is it one that Crucible supports?

Crucible recommends the latest ROCm (10.1 is the one tested today). Version 7.2.4 stays supported. A user on the latest
ROCm sees no warning and no message that asks for an older version. The code only reads files.
"""
import re
from pathlib import Path
from typing import NamedTuple, Optional

RECOMMENDED = "10.1"          # tested, and what the guide says to install
TESTED = ("10.1", "7.2.4")    # the two versions that the release engine was run on
CURRENT_MAJOR = 10            # this major version and every later one count as current
SONAME_MIN = 7                # the release engine links libamdhip64.so.7
PRIVATE_DIR = "rocm-runtime"  # inside the work folder (.crucible)

# libamdhip64 file version -> ROCm version. Only builds that were looked at are here; any other build is "unknown".
HIP_TO_ROCM = {"7.16.26385": "10.1", "7.2.53211": "7.2.4"}

_HIP = re.compile(r"^libamdhip64\.so\.(\d+)((?:\.\d+)*)$")


class Runtime(NamedTuple):
    root: Optional[Path]     # the ROCm folder (system), or the private runtime folder; None for an env folder
    lib: Optional[Path]      # the folder that holds libamdhip64
    version: Optional[str]   # the ROCm version, when known
    state: str               # current | tested | other | unknown | none
    source: str              # system | env | private | none


def _parse(text):
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", (text or "").strip())
    return tuple(int(x) if x is not None else 0 for x in m.groups()) if m else None


def classify(version):
    """current: major 10 or later. tested: 7.2.4. other: every other version, and text that is not a version."""
    v = _parse(version)
    if v is None:
        return "other"
    if v[0] >= CURRENT_MAJOR:
        return "current"
    if v == (7, 2, 4):
        return "tested"
    return "other"


def version_from_lib(name):
    """The ROCm version for a library file name (libamdhip64.so.7.16.26385), or None when the build is not known."""
    m = _HIP.match(Path(name).name)
    if not m or not m.group(2):
        return None
    return HIP_TO_ROCM.get(m.group(1) + m.group(2))


def _hip_files(folder):
    try:
        names = sorted(p.name for p in Path(folder).iterdir() if _HIP.match(p.name))
    except OSError:
        return []
    return names


def _usable(names):
    return [n for n in names if int(_HIP.match(n).group(1)) >= SONAME_MIN]


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _from_lib_dir(folder, root, source):
    """A Runtime for a folder with HIP libraries, or None when it holds no usable one."""
    names = _usable(_hip_files(folder))
    if not names:
        return None
    version = next((v for v in (version_from_lib(n) for n in names) if v), None)
    return Runtime(root, Path(folder), version, classify(version) if version else "unknown", source)


def detect(host, work=None):
    """Look for a HIP runtime in this order: the system ROCm folder, the folders of LD_LIBRARY_PATH, the private runtime
    that setup made in WORK (the .crucible folder)."""
    root = Path(host.rocm)
    version = _read(root / ".info/version")
    if version:
        lib = root / "lib"
        return Runtime(root, lib, version, classify(version), "system")
    found = _from_lib_dir(root / "lib", root, "system")
    if found:
        return found
    for entry in (host.env.get("LD_LIBRARY_PATH") or "").split(":"):
        if entry:
            found = _from_lib_dir(entry, None, "env")
            if found:
                return found
    if work is not None:
        found = private(work)
        if found:
            return found
    return Runtime(None, None, None, "none", "none")


def private(work):
    """The private runtime in WORK/rocm-runtime, or None."""
    import json
    base = Path(work) / PRIVATE_DIR
    lib = base / "lib"
    if not _usable(_hip_files(lib)):
        return None
    try:
        doc = json.loads((base / "runtime.json").read_text())
        version = doc.get("rocm") if isinstance(doc, dict) else None
    except (OSError, ValueError):
        version = None
    if not isinstance(version, str):
        version = next((v for v in (version_from_lib(n) for n in _hip_files(lib)) if v), None)
    return Runtime(base, lib, version, classify(version) if version else "unknown", "private")


def kfd_gfx(host):
    """The GPU names that the kernel reports (from /sys/class/kfd), for a machine without rocminfo. 110501 is gfx1151."""
    out = []
    nodes = Path(host.sysfs) / "class/kfd/kfd/topology/nodes"
    try:
        props = sorted(nodes.glob("*/properties"))
    except OSError:
        return out
    for p in props:
        for line in (_read(p) or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "gfx_target_version" and parts[1].isdigit() and int(parts[1]) > 0:
                v = int(parts[1])
                out.append(f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}")
    return sorted(set(out))
