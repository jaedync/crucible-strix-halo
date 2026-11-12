#!/usr/bin/env python3
"""The release engine: list the libraries it needs, and make the three files of a release.

  release_engine.py needs   --binary ENGINE [--rocm-path DIR] [--libdir DIR]... [--allow-incomplete] [--json]
  release_engine.py release --binary ENGINE --tag vX.Y.Z --commit SHA --out NEWDIR [--rocm-path DIR] [--libdir DIR]...
  release_engine.py stamp begin|end --binary ENGINE [--repo DIR]

`needs` walks the library list of the engine (readelf, no program is run). It prints two lists of libraries that are
not part of ROCm and not part of the base system (libc, libm, libstdc++, libgcc_s):
  external       the libraries that the engine itself names (its own NEEDED list). Install.sh requires them on the target.
  rocm_external  the libraries that ROCm libraries of the BUILD host need, for example fmt 12 and glog 2 from the
                 packaging of one distribution. Information only: the ROCm of the target brings its own, and the real
                 ldd of the engine on the target decides. Install.sh never requires them.
It also prints the lowest glibc and libstdc++ version that the engine needs.

`release` writes three files to a new folder; upload them with the tag:
  crucible-engine-TAG-gfx1151-rocm7.2-linux-x86_64.gz     the engine, gzip, no name or time inside (reproducible)
  crucible-engine-TAG-gfx1151-rocm7.2-linux-x86_64.needs  the manifest: tag, commit, sha256 of the engine, version floors,
                                                          `need` lines (required) and `rocm_external` lines (information)
  SHA256SUMS                                              sha256 of the two files above
`stamp` is the build-info of an engine build (the Makefile runs it before and after the compile). `begin` removes the
old ENGINE.build-info and notes the commit and whether the tree has local changes. `end` writes ENGINE.build-info: the
commit, dirty 0 or 1 (1 also when the commit or the tree changed during the compile), and the sha256 of ENGINE.
`release` refuses an engine with no build-info, with another hash, with another commit than --commit, or built from a
dirty tree. So an old binary that a killed compiler left in place cannot be shipped.
It refuses to run when TAG is not the version in VERSION and the PIN_TAG in install.sh, so a release cannot ship
an installer that points at another tag.

Exit codes: 0 done, 2 a refusal (the text says why). Standard library only; needs readelf from binutils.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = "gfx1151"
ROCM_SERIES = "7.2"
FORMAT = 1
BASE = frozenset({"libc.so.6", "libm.so.6", "libstdc++.so.6", "libgcc_s.so.1", "libdl.so.2", "libpthread.so.0", "librt.so.1",
                  "libutil.so.1", "ld-linux-x86-64.so.2", "linux-vdso.so.1"})
ROCM_NAME = re.compile(r"^lib(amdhip|amd_comgr|hsa-runtime|hsakmt|hip|roc|rccl|miopen|migraphx)")
SYSTEM_DIRS = ("/usr/lib", "/usr/lib64", "/lib64", "/lib", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu")
TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")
# The compile settings of the release engine: the defaults of the Makefile. `make HIPFLAGS=... build/engine/crucible` writes
# the settings that it used to the build-info, and `release` refuses an engine that was built with other ones.
DEFAULT_FLAGS = {"gpu_arch": "gfx1151", "hipflags": "-O3", "cppflags": "", "ldflags": "", "ldlibs": "-pthread", "rocm_path": "/opt/rocm",
                 "hipcc_compile_flags_append": "", "hipcc_link_flags_append": "",
                 "hip_clang_path": "", "hip_device_lib_path": ""}
FLAG_NAMES = {"gpu_arch": "GPU_ARCH", "hipflags": "HIPFLAGS", "cppflags": "CPPFLAGS", "ldflags": "LDFLAGS", "ldlibs": "LDLIBS", "rocm_path": "ROCM_PATH",
              "hipcc_compile_flags_append": "HIPCC_COMPILE_FLAGS_APPEND", "hipcc_link_flags_append": "HIPCC_LINK_FLAGS_APPEND",
              "hip_clang_path": "HIP_CLANG_PATH", "hip_device_lib_path": "HIP_DEVICE_LIB_PATH"}
# The settings that the Makefile passes to `stamp end`. The other ones are environment variables that hipcc reads (they change
# the code or the compiler without a word on the make line), so `stamp end` reads them from its own environment, which is the one
# of the compile step.
MAKE_FLAGS = ("gpu_arch", "hipflags", "cppflags", "ldflags", "ldlibs", "rocm_path")


class Refusal(Exception):
    pass


def _readelf(*args):
    exe = shutil.which("readelf")
    if exe is None:
        raise Refusal("readelf is not installed (package binutils).")
    return subprocess.run([exe, *args], capture_output=True, text=True)


def _vkey(text):
    return tuple(int(x) for x in text.split("."))


def elf_info(path):
    """(needed names, rpath folders) of one ELF file. Refuses a file that is not a 64-bit x86-64 ELF."""
    head = _readelf("-h", str(path))
    if head.returncode != 0 or "ELF64" not in head.stdout or "X86-64" not in head.stdout:
        raise Refusal(f"{path} is not a 64-bit x86-64 ELF file.")
    dyn = _readelf("-d", "-W", str(path)).stdout
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dyn)
    rpath = []
    for m in re.finditer(r"\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[([^\]]*)\]", dyn):
        rpath += [p.replace("$ORIGIN", str(Path(path).resolve().parent)) for p in m.group(1).split(":") if p]
    return needed, rpath


def version_floors(path):
    text = _readelf("-V", "-W", str(path)).stdout
    out = {}
    for key, pat in (("glibc_min", r"\bGLIBC_(\d+(?:\.\d+)+)"), ("glibcxx_min", r"\bGLIBCXX_(\d+(?:\.\d+)+)")):
        found = re.findall(pat, text)
        out[key] = ".".join(str(x) for x in max(_vkey(v) for v in found)) if found else ""
    return out


def _ldconfig_paths():
    exe = shutil.which("ldconfig") or next((p for p in ("/sbin/ldconfig", "/usr/sbin/ldconfig") if os.access(p, os.X_OK)), None)
    found = {}
    if exe:
        res = subprocess.run([exe, "-p"], capture_output=True, text=True)
        for m in re.finditer(r"^\s*(\S+) \(.*\) => (\S+)$", res.stdout, re.M):
            found.setdefault(m.group(1), m.group(2))
    return found


def needs(binary, rocm_path, libdirs=(), allow_incomplete=False):
    """Walk the libraries. Returns a dict with external (name -> "the engine": the engine names it itself), rocm_external
    (name -> the ROCm libraries of the build host that need it), rocm, unresolved, floors."""
    binary = Path(binary)
    rocm = Path(rocm_path).resolve() if rocm_path else None
    first = [Path(p) for p in libdirs] + ([rocm / "lib", rocm / "lib64"] if rocm else [])
    system = _ldconfig_paths()

    def find(name, rpath):
        for d in [*rpath, *first, *(Path(x) for x in SYSTEM_DIRS)]:
            cand = Path(d) / name
            if cand.is_file():
                return cand
        hit = system.get(name)
        return Path(hit) if hit and Path(hit).is_file() else None

    names, rpath = elf_info(binary)
    external, rocm_libs, unresolved = {}, [], []
    seen = set()
    queue = [(n, rpath, "the engine") for n in names]
    while queue:
        name, rp, parent = queue.pop(0)
        if name in seen:
            if name in external and parent not in external[name]:
                external[name].append(parent)
            continue
        seen.add(name)
        if name in BASE:
            continue
        path = find(name, rp)
        in_rocm = bool(ROCM_NAME.match(name)) or (path is not None and rocm is not None and rocm in path.resolve().parents)
        if in_rocm:
            rocm_libs.append(name)
            if path is None:
                unresolved.append(name)
                continue
            sub_names, sub_rpath = elf_info(path)
            queue += [(n, sub_rpath + rp, name) for n in sub_names]
        else:
            external[name] = [parent]
    if unresolved and not allow_incomplete:
        raise Refusal("The list is incomplete: these ROCm libraries were not found: " + ", ".join(sorted(unresolved))
                      + ". Their own libraries (for example fmt) cannot be listed. Point --rocm-path at a complete ROCm, "
                      "or add the folder that holds them with --libdir DIR, or use --allow-incomplete for a check only.")
    # A library that the engine names itself is required on the target. A library that only a ROCm library of the build host
    # needs is not: the ROCm of the target has its own list (AMD packages for Ubuntu need no fmt and no glog).
    direct = {k: sorted(set(v)) for k, v in sorted(external.items()) if "the engine" in v}
    via_rocm = {k: sorted(set(v)) for k, v in sorted(external.items()) if "the engine" not in v}
    return {"external": direct, "rocm_external": via_rocm, "rocm": sorted(rocm_libs),
            "unresolved": sorted(unresolved), **version_floors(binary)}


def cmd_needs(args):
    data = needs(args.binary, args.rocm_path, args.libdir, args.allow_incomplete)
    if data["unresolved"]:
        print("warning: the list is incomplete: not found: " + ", ".join(data["unresolved"]), file=sys.stderr)
    if args.json:
        print(json.dumps({**data, "external": list(data["external"]), "rocm_external": list(data["rocm_external"])}, indent=2))
        return 0
    print(f"Libraries that the engine itself needs, outside ROCm and the base system ({len(data['external'])}); install.sh requires them:")
    for name, parents in data["external"].items():
        print(f"  {name:<24} needed by {', '.join(parents)}")
    print(f"Libraries that ROCm of this build host needs ({len(data['rocm_external'])}); information only, install.sh does not require them:")
    for name, parents in data["rocm_external"].items():
        print(f"  {name:<24} needed by {', '.join(parents)}")
    print(f"ROCm libraries seen: {', '.join(data['rocm']) or 'none'}")
    print(f"Lowest glibc: {data['glibc_min'] or 'none'}    Lowest libstdc++ (GLIBCXX): {data['glibcxx_min'] or 'none'}")
    return 0


def _pin_tag(install_sh):
    m = re.search(r'^PIN_TAG="([^"]*)"', Path(install_sh).read_text(), re.M)
    return m.group(1) if m else ""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_state(repo):
    """(HEAD commit or 'none', True when tracked files have local changes or git cannot tell)."""
    git = shutil.which("git")
    if git is None:
        return "none", True
    head = subprocess.run([git, "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run([git, "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
    commit = head.stdout.strip()
    if head.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit) or status.returncode != 0:
        return "none", True
    return commit, bool(status.stdout.strip())


def _info_path(binary, suffix=""):
    return Path(str(binary) + ".build-info" + suffix)


def _words(text):
    return " ".join((text or "").split())


def hipcc_version(command):
    """The first line of `hipcc --version`, or "unknown" when it cannot run. COMMAND is the HIPCC word of the Makefile."""
    try:
        run = subprocess.run([*shlex.split(command), "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return "unknown"
    lines = [line.strip() for line in (run.stdout + run.stderr).splitlines() if line.strip()]
    return _words(lines[0]) if run.returncode == 0 and lines else "unknown"


def cmd_stamp(args):
    pending = _info_path(args.binary, ".pending")
    _info_path(args.binary).unlink(missing_ok=True)
    if args.phase == "begin":
        commit, dirty = _git_state(args.repo)
        pending.write_text(f"commit {commit}\ndirty {int(dirty)}\n")
        return 0
    if not pending.is_file():
        raise Refusal(f"no {pending.name} next to the binary: `stamp begin` did not run before the build.")
    if not Path(args.binary).is_file() or Path(args.binary).stat().st_size == 0:
        raise Refusal(f"no binary at {args.binary}: the compiler wrote none.")
    before = dict(line.split(" ", 1) for line in pending.read_text().splitlines())
    commit, dirty = _git_state(args.repo)
    changed = commit != before.get("commit") or dirty or before.get("dirty") != "0"
    text = f"commit {before.get('commit', 'none')}\ndirty {int(changed)}\nbinary_sha256 {_sha256(args.binary)}\n"
    if args.hipcc is not None:  # the Makefile passes the compile settings and the compiler
        value = lambda name: getattr(args, name) if name in MAKE_FLAGS else os.environ.get(FLAG_NAMES[name], "")  # noqa: E731
        text += "".join(f"{name} {_words(value(name))}\n" for name in DEFAULT_FLAGS)
        text += f"hipcc_version {hipcc_version(args.hipcc)}\n"
    _info_path(args.binary).write_text(text)
    pending.unlink()
    return 0


def check_flags(info, name_of_file):
    """Refuse a build-info (as a dict) that lacks a compile setting, shows a non-default one or an unknown compiler."""
    for name, default in DEFAULT_FLAGS.items():
        if name not in info:
            raise Refusal(f"The build-info {name_of_file} has no value for {FLAG_NAMES[name]}, so it is not proven that the engine was built "
                          "with the default settings. Delete the engine and run `make release-engine` again.")
        if _words(info[name]) != default:
            raise Refusal(f"The engine was built with {FLAG_NAMES[name]}={info[name]!r}, and the release engine uses {default!r}. "
                          "Delete the engine and build it again without that setting.")
    version = info.get("hipcc_version", "").strip()
    if not version or version == "unknown":
        raise Refusal(f"The build-info {name_of_file} does not name the compiler version (the compiler could not run `--version`). "
                      "Fix the compiler, delete the engine and build it again.")


def check_build_info(binary, info_file, commit):
    """Refuse a binary that is not the clean build of `commit` (the file that `stamp end` wrote)."""
    info_file = Path(info_file)
    if not info_file.is_file():
        raise Refusal(f"There is no build-info file {info_file.name} next to the engine, so it is not proven that this engine was built "
                      "from this commit. Delete the engine and run `make release-engine` again to build it.")
    info = {line.partition(" ")[0]: line.partition(" ")[2] for line in info_file.read_text().splitlines()}
    if info.get("binary_sha256") != _sha256(binary):
        raise Refusal(f"The build-info {info_file.name} does not match the binary (the binary changed after the build, "
                      "or an old binary stayed in place). Delete the engine and run `make release-engine` again.")
    if info.get("dirty") != "0":
        raise Refusal("The engine was built from a tree with local changes. Commit them, delete the engine and build it again.")
    check_flags(info, info_file.name)
    if info.get("commit") != commit:
        raise Refusal(f"The engine was built from another commit ({info.get('commit', 'none')[:12]}), not {commit[:12]}. "
                      "Delete the engine and run `make release-engine` again.")


def cmd_release(args):
    if not TAG_RE.fullmatch(args.tag):
        raise Refusal(f"The tag must look like v0.34.0, not {args.tag!r}.")
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise Refusal("The commit must be 40 lowercase hex characters (git rev-parse HEAD).")
    version = Path(args.version_file).read_text().strip()
    if version != args.tag[1:]:
        raise Refusal(f"The file VERSION says {version}, and the tag is {args.tag}. Change VERSION first.")
    pin = _pin_tag(args.install_sh)
    if pin != args.tag:
        raise Refusal(f"PIN_TAG in install.sh is {pin!r}, and the tag is {args.tag}. Change PIN_TAG first, so the installer points at this release.")
    rocm_info = Path(args.rocm_path) / ".info/version"
    if rocm_info.is_file() and not rocm_info.read_text().strip().startswith(ROCM_SERIES):
        raise Refusal(f"ROCm in {args.rocm_path} is {rocm_info.read_text().strip()}. The release engine is built for ROCm {ROCM_SERIES}.")
    out = Path(args.out)
    if out.exists():
        raise Refusal(f"{out} exists. Give a new folder.")
    check_build_info(args.binary, args.build_info or _info_path(args.binary), args.commit)
    data = needs(args.binary, args.rocm_path, args.libdir, args.allow_incomplete)
    stem = f"crucible-engine-{args.tag}-{ARCH}-rocm{ROCM_SERIES}-linux-x86_64"
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".release-", dir=out.parent))
    try:
        gz = work / f"{stem}.gz"
        with open(args.binary, "rb") as src, open(gz, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        lines = ["format %d" % FORMAT, f"tag {args.tag}", f"commit {args.commit}", f"arch {ARCH}", f"rocm {ROCM_SERIES}",
                 f"binary_sha256 {_sha256(args.binary)}", f"glibc_min {data['glibc_min']}", f"glibcxx_min {data['glibcxx_min']}"]
        lines += [f"need {n}" for n in data["external"]]
        lines += [f"rocm_external {n}" for n in data["rocm_external"]]
        (work / f"{stem}.needs").write_text("\n".join(lines) + "\n")
        sums = [f"{_sha256(work / n)}  {n}" for n in sorted((gz.name, f"{stem}.needs"))]
        (work / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        for p in work.iterdir():
            p.chmod(0o644)
        work.chmod(0o755)
        work.rename(out)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print(f"Wrote {out}: {stem}.gz, {stem}.needs, SHA256SUMS")
    print(f"Libraries the engine needs besides ROCm and the base system: {', '.join(data['external']) or 'none'}")
    print(f"Libraries that ROCm of this build host needs (information only): {', '.join(data['rocm_external']) or 'none'}")
    print(f"Lowest glibc {data['glibc_min']}, lowest GLIBCXX {data['glibcxx_min']}")
    print("Next: upload the three files to the release of the tag. Users can pin this hash with CRUCIBLE_SUMS_SHA256:")
    print(f"  sha256sum {out}/SHA256SUMS")
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--binary", required=True, type=Path)
        p.add_argument("--rocm-path", default=os.environ.get("ROCM_PATH", "/opt/rocm"))
        p.add_argument("--libdir", action="append", default=[], help="a folder with more libraries (searched before ROCm)")
        p.add_argument("--allow-incomplete", action="store_true", help="go on when a ROCm library is not found (a check only)")

    n = sub.add_parser("needs", help="list the libraries of the engine")
    common(n)
    n.add_argument("--json", action="store_true")
    r = sub.add_parser("release", help="write the release files")
    common(r)
    r.add_argument("--tag", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--version-file", type=Path, default=ROOT / "VERSION")
    r.add_argument("--install-sh", type=Path, default=ROOT / "install.sh")
    r.add_argument("--build-info", type=Path, help="default: ENGINE.build-info")
    st = sub.add_parser("stamp", help="write the build-info of an engine build (the Makefile calls this)")
    st.add_argument("phase", choices=("begin", "end"))
    st.add_argument("--binary", required=True, type=Path)
    st.add_argument("--repo", type=Path, default=ROOT)
    st.add_argument("--hipcc", help="phase end: the compiler command; with it the build-info also names the settings below")
    for name in MAKE_FLAGS:
        st.add_argument("--" + name.replace("_", "-"), default="", help=f"phase end: the value of {FLAG_NAMES[name]} (write --{name.replace('_', '-')}=VALUE)")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return {"needs": cmd_needs, "release": cmd_release, "stamp": cmd_stamp}[args.cmd](args)
    except Refusal as exc:
        print(f"release_engine: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"release_engine: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
