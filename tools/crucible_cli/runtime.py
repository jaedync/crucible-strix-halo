"""The private ROCm runtime: a fallback for a machine that has no HIP runtime.

Crucible recommends the latest ROCm from the system (10.1 is the one tested). When a machine has none, setup can download
the HIP runtime into the install folder (.crucible/rocm-runtime). It needs no root and changes no system package.
`./crucible uninstall` removes it with the install folder. A system ROCm that is installed later is used instead.

The source is AMD's own package index: the pin (runtime-pin.json, made by tools/make_runtime_pin.py) names the wheel, and
for each file its sha256 and its byte range inside the wheel. Setup reads only those ranges and checks every file against
its sha256 before it uses it. Crucible does not host or change the files.
"""
import hashlib
import json
import zlib
from pathlib import Path
from urllib.parse import urlparse

from .files import OutsideRoots, Sandbox
from .rocm import PRIVATE_DIR

PIN_FILE = Path(__file__).with_name("runtime-pin.json")
URL_ENV = "CRUCIBLE_RUNTIME_URL"   # a mirror of the wheel: the files are checked against the pin, so any address is safe
_CHUNK = 1 << 20
_HEX = set("0123456789abcdef")


class RuntimeFailed(Exception):
    pass


def _mib(n):
    return f"{n / 2**20:.0f}"


def _safe_to(to, top):
    parts = to.split("/") if isinstance(to, str) else [""]
    return (parts[0] == top and len(parts) > 1 and all(p not in ("", ".", "..") for p in parts) and "\\" not in to
            and "\0" not in to)


def _check_entry(m, top):
    ok = (isinstance(m, dict) and _safe_to(m.get("to"), top) and m.get("method") in (0, 8)
          and isinstance(m.get("sha256"), str) and len(m["sha256"]) == 64 and set(m["sha256"]) <= _HEX
          and all(isinstance(m.get(k), int) and not isinstance(m.get(k), bool) and m[k] >= 0 for k in ("offset", "csize", "size"))
          and m["size"] > 0)
    if not ok:
        raise RuntimeFailed(f"the runtime pin is not valid: bad entry {m.get('to') if isinstance(m, dict) else m!r}")


def load_pin(path=None):
    """The pin, checked. Every place is a plain relative name under lib/ or licenses/: nothing leaves the runtime folder."""
    path = Path(path or PIN_FILE)
    try:
        pin = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeFailed(f"the runtime pin cannot be read: {exc}") from None
    check_pin(pin)
    return pin


def check_pin(pin):
    if not isinstance(pin, dict) or pin.get("format") != 1 or not isinstance(pin.get("members"), list) or not pin["members"]:
        raise RuntimeFailed("the runtime pin is not valid: format")
    if not str(pin.get("url", "")).startswith("https://"):
        raise RuntimeFailed("the runtime pin is not valid: the address must be https")
    for m in pin["members"]:
        _check_entry(m, "lib")
    for m in pin.get("licenses", []):
        _check_entry(m, "licenses")


def pin_id(pin):
    rows = [[m["to"], m["sha256"], m["size"]] for m in pin["members"] + pin.get("licenses", [])]
    return hashlib.sha256(json.dumps([pin["rocm"], pin["hip"], rows]).encode()).hexdigest()


def offer_text(pin, work):
    host = urlparse(pin["url"]).hostname
    return (f"No HIP runtime (ROCm) was found on this machine. Setup can download a private copy of the ROCm {pin['rocm']} "
            f"runtime for Crucible: {_mib(pin['download_bytes'])} MB from {host}, into {Path(work) / PRIVATE_DIR}. "
            "It needs no root and changes no system package. ./crucible uninstall removes it. "
            "A system ROCm that you install later is used instead.")


def installed_ok(base, pin):
    """True when BASE holds the runtime of this pin: the record matches and every file has its size."""
    try:
        doc = json.loads((Path(base) / "runtime.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(doc, dict) or doc.get("pin_sha256") != pin_id(pin):
        return False
    for m in pin["members"] + pin.get("licenses", []):
        p = Path(base) / m["to"]
        try:
            if p.is_symlink() or not p.is_file() or p.stat().st_size != m["size"]:
                return False
        except OSError:
            return False
    return True


def _fetch(host, box, url, stage, m):
    part = stage / ".part"
    dest = stage / m["to"]
    box.mkdir(stage)
    with box.open_out(part) as f:
        code, got = host.http.download(url, f, start=m["offset"], end=m["offset"] + m["csize"] - 1)
    if code != 206:
        why = "the server did not answer" if code == 0 else f"HTTP {code}" + (" (the server ignores byte ranges)" if code == 200 else "")
        raise RuntimeFailed(f"download of {m['to']} failed: {why}")
    if got != m["csize"]:
        raise RuntimeFailed(f"download of {m['to']} stopped: {got} bytes of {m['csize']} bytes")
    box.mkdir(dest.parent)
    digest, total = hashlib.sha256(), 0
    inflate = zlib.decompressobj(-15) if m["method"] == 8 else None
    with open(box.check(part), "rb") as src, box.open_out(dest) as out:
        while True:
            chunk = src.read(_CHUNK)
            if not chunk:
                break
            data = inflate.decompress(chunk, m["size"] - total + 1) if inflate else chunk
            total += len(data)
            if total > m["size"]:
                raise RuntimeFailed(f"{m['to']} is larger than its pinned size ({m['size']} bytes)")
            digest.update(data)
            out.write(data)
        if inflate:
            data = inflate.flush()
            total += len(data)
            digest.update(data)
            out.write(data)
    box.unlink(part, missing_ok=True)
    if total != m["size"]:
        raise RuntimeFailed(f"{m['to']} has {total} bytes, the pin says {m['size']} bytes")
    if digest.hexdigest() != m["sha256"]:
        raise RuntimeFailed(f"{m['to']}: the sha256 does not match the pin. The file was not used.")
    box.chmod(dest, 0o755 if ".so" in dest.name else 0o644)


def install(host, s, pin=None, say=None):
    """Download the runtime into WORK/rocm-runtime. Returns a note. Raises RuntimeFailed; a failed run leaves the folder
    that was there before (or none). It writes only inside the work folder."""
    pin = pin if pin is not None else load_pin()
    check_pin(pin)
    say = say or (lambda text: None)
    work = Path(s.work)
    try:
        box = Sandbox([work])
    except ValueError as exc:
        raise RuntimeFailed(str(exc)) from None
    base, stage, old = work / PRIVATE_DIR, work / (PRIVATE_DIR + ".new"), work / (PRIVATE_DIR + ".old")
    if installed_ok(base, pin):
        return f"already installed (ROCm {pin['rocm']})"
    url = host.env.get(URL_ENV) or pin["url"]
    if not url.startswith("https://"):
        raise RuntimeFailed(f"the address of the runtime must be https ({URL_ENV} or the pin)")
    need = pin["disk_bytes"] + (64 << 20)
    if host.disk_free(work) < need:
        raise RuntimeFailed(f"not enough disk space for the runtime: {_mib(need)} MB needed in {work}")
    members = pin["members"] + pin.get("licenses", [])
    try:
        box.rmtree(stage)
        box.mkdir(stage)
        for i, m in enumerate(members, 1):
            say(f"  [{i}/{len(members)}] {m['to']} ({_mib(m['csize'])} MB)")
            _fetch(host, box, url, stage, m)
        box.write_json(stage / "runtime.json", {"rocm": pin["rocm"], "hip": pin["hip"], "url": pin["url"],
                                                "pin_sha256": pin_id(pin), "files": len(members),
                                                "made_by": "crucible setup"})
        box.rmtree(old)
        if base.exists() or base.is_symlink():
            box.rename(base, old)
        box.rename(stage, base)
        box.rmtree(old)
    except OutsideRoots as exc:
        raise RuntimeFailed(str(exc)) from None
    except (OSError, zlib.error) as exc:
        raise RuntimeFailed(f"the runtime could not be written: {exc}") from None
    finally:
        try:
            box.rmtree(stage)
        except OSError:
            pass
    return f"ROCm {pin['rocm']} runtime installed in {base}"
