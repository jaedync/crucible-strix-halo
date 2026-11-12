#!/usr/bin/env python3
"""Make tools/crucible_cli/runtime-pin.json: the pin of the private ROCm runtime that `./crucible setup` can download.

  make_runtime_pin.py --wheel URL_OR_FILE --rocm 10.1 --hip 7.16.26385 [--out PATH]

The source is the rocm-sdk-core wheel (a zip file) on AMD's package index. Crucible does not host the runtime. The pin
holds, for each file that the engine needs: the place in the install folder, the size, the sha256 of the file, and the
byte range of its data in the wheel. Setup then reads only those ranges (about 95 MB of the 420 MB wheel) and checks
each file against its sha256. Run this tool when the runtime changes, read the result, and commit it. It needs the
network when the wheel is a URL (it reads the files once to hash them).
"""
import argparse
import hashlib
import json
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

PREFIX = "_rocm_sdk_core/"
# The files that the engine loads with the 10.1 runtime (the set that ran on the test machine) and the license files
# that the wheel holds for them.
DEFAULT_WANT = [PREFIX + n for n in (
    "lib/libamdhip64.so.7", "lib/libamd_comgr.so.3", "lib/libhsa-runtime64.so.1", "lib/librocm_kpack.so.0",
    "lib/librocprofiler-register.so.0", "lib/llvm/lib/libLLVM.so.24.0git", "lib/llvm/lib/libclang-cpp.so.24.0git",
    "lib/rocm_sysdeps/lib/librocm_sysdeps_bz2.so", "lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1",
    "lib/rocm_sysdeps/lib/librocm_sysdeps_liblzma.so.5", "lib/rocm_sysdeps/lib/librocm_sysdeps_drm_amdgpu.so.1",
    "lib/rocm_sysdeps/lib/librocm_sysdeps_elf.so.1", "lib/rocm_sysdeps/lib/librocm_sysdeps_numa.so.1",
    "lib/rocm_sysdeps/lib/librocm_sysdeps_z.so.1", "lib/rocm_sysdeps/lib/librocm_sysdeps_drm.so.2",
    "lib/rocm_sysdeps/share/libdrm/amdgpu.ids")]
DEFAULT_LICENSES = [
    (PREFIX + "share/doc/rocr/LICENSE.md", "licenses/rocr-LICENSE.md"),
    (PREFIX + "share/doc/amd_comgr/LICENSE.txt", "licenses/amd_comgr-LICENSE.txt"),
    (PREFIX + "share/doc/rocprofiler-register/LICENSE.md", "licenses/rocprofiler-register-LICENSE.md")]


class PinError(Exception):
    pass


class BytesSource:
    def __init__(self, data):
        self.data, self.size = data, len(data)

    def read(self, start, end):
        return self.data[start:end + 1]


class FileSource:
    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size

    def read(self, start, end):
        with open(self.path, "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)


class UrlSource:
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size, self.url = int(r.headers["Content-Length"]), r.geturl()   # the address after redirects

    def read(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            if r.status != 206:
                raise PinError(f"{self.url} ignores byte ranges")
            return r.read()


def central_directory(src):
    """{name: (local header offset, method, compressed size, size)} of a zip file."""
    tail_start = max(0, src.size - 70000)
    tail = src.read(tail_start, src.size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise PinError("not a zip file")
    count, cd_size, cd_off = struct.unpack("<HII", tail[i + 10:i + 20])
    if cd_off == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x06")
        count, cd_size, cd_off = struct.unpack("<QQQ", tail[j + 32:j + 56])
    cd = src.read(cd_off, cd_off + cd_size - 1)
    out, p = {}, 0
    while p < len(cd):
        if struct.unpack("<I", cd[p:p + 4])[0] != 0x02014B50:
            raise PinError("bad zip directory")
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, size, nlen, elen, clen = struct.unpack("<IIHHH", cd[p + 20:p + 34])
        lho, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode()
        out[name] = (lho, method, csize, size)
        p += 46 + nlen + elen + clen
    return out


def _entry(src, directory, name, to):
    if name not in directory:
        raise PinError(f"{name} is not in the wheel")
    lho, method, csize, size = directory[name]
    head = src.read(lho, lho + 29)
    if struct.unpack("<I", head[:4])[0] != 0x04034B50:
        raise PinError(f"{name}: bad local header")
    nlen, elen = struct.unpack("<HH", head[26:30])
    offset = lho + 30 + nlen + elen
    raw = src.read(offset, offset + csize - 1)
    data = zlib.decompress(raw, -15) if method == 8 else raw
    if method not in (0, 8) or len(data) != size:
        raise PinError(f"{name}: method {method} or size does not match")
    return {"name": name, "to": to, "method": method, "offset": offset, "csize": csize, "size": size,
            "sha256": hashlib.sha256(data).hexdigest()}


def build_pin(src, url, rocm, hip, want, licenses):
    """The pin (a dict). WANT and LICENSES: lists of (name in the wheel, place in the runtime folder)."""
    directory = central_directory(src)
    members = [_entry(src, directory, n, t) for n, t in want]
    lic = [_entry(src, directory, n, t) for n, t in licenses]
    return {"format": 1, "rocm": rocm, "hip": hip, "arch": "x86_64", "url": url, "wheel_size": src.size,
            "download_bytes": sum(m["csize"] for m in members + lic), "disk_bytes": sum(m["size"] for m in members + lic),
            "members": members, "licenses": lic}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wheel", required=True, help="the URL of the wheel, or a file")
    ap.add_argument("--rocm", required=True, help="the ROCm version, for example 10.1")
    ap.add_argument("--hip", required=True, help="the libamdhip64 file version, for example 7.16.26385")
    ap.add_argument("--url", help="the address to write in the pin (default: the --wheel value)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "crucible_cli/runtime-pin.json")
    args = ap.parse_args(argv)
    src = UrlSource(args.wheel) if args.wheel.startswith("https://") else FileSource(args.wheel)
    url = args.url or (src.url if isinstance(src, UrlSource) else args.wheel)
    pin = build_pin(src, url, args.rocm, args.hip, [(n, n[len(PREFIX):]) for n in DEFAULT_WANT], DEFAULT_LICENSES)
    args.out.write_text(json.dumps(pin, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(pin['members'])} files, download {pin['download_bytes'] / 2**20:.1f} MiB, "
          f"disk {pin['disk_bytes'] / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
