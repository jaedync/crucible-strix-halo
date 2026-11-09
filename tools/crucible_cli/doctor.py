"""./crucible doctor: every check with one PASS/FAIL/WARN/SKIP line and, for a failure, the fix as text.

The doctor only reads. It never changes the machine. A fix is a text for the user to apply.
"""
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import keyfile, packs, service
from .files import Sandbox

GTT_PARAMS = "amdgpu.gttsize=126976 ttm.pages_limit=32505856"
GTT_MIN_BYTES = 130_000_000_000      # the tested machine shows about 133 GB
GPU_BUSY_BYTES = 8 * 1024 ** 3        # more than this in use: another program holds the GPU memory
MEM_MIN_KB = 120 * 1024 * 1024        # 120 GiB: the guide expects "total 120 or more" in free -g
ROCM_VERSION = "7.2.4"
PY_MIN = (3, 12)
STEPS_ALL = ("venv", "download", "build", "start")
GB = 1_000_000_000



def release_engine(repo):
    """The record of a verified release engine in REPO, or None (also when release.py is not there)."""
    try:
        from .release import engine_release
    except ImportError:
        return None
    return engine_release(repo)


NOT_NEEDED = "not needed: the release engine is installed"

ROCM_FIX = (
    f"Install ROCm {ROCM_VERSION}: the HIP compiler (hipcc) and rocminfo.\n"
    "Arch Linux: sudo pacman -Syu --needed base-devel git python hip-runtime-amd rocm-llvm rocminfo\n"
    "Ubuntu or Debian: follow AMD's install page for your release (https://rocm.docs.amd.com), then: sudo apt install build-essential git\n"
    "Fedora: follow AMD's install page (https://rocm.docs.amd.com), then: sudo dnf install make gcc-c++ git\n"
    "This tool does not install system packages. You run these lines yourself."
)

KERNEL_FIX = (
    f"Add these two parameters to the kernel command line, then restart: {GTT_PARAMS}\n"
    "Keep every parameter that is already there (see: cat /proc/cmdline) and do not add one twice. "
    "Where the line is depends on how your system starts:\n"
    "  - systemd-boot with entry files: the options line of your entry in /boot/loader/entries/\n"
    "  - GRUB: GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub, then: sudo grub-mkconfig -o /boot/grub/grub.cfg\n"
    "  - a unified kernel image: usually /etc/kernel/cmdline, then build the image again (for example: sudo mkinitcpio -P)\n"
    "  - another boot loader: see its documentation\n"
    "Keep a working boot entry until the new one starts. A wrong command line can stop the machine from starting.\n"
    "This tool never changes the boot loader or the kernel command line. You apply the line yourself."
)


@dataclass
class CheckResult:
    id: str
    title: str
    status: str
    detail: str = ""
    fix: str = ""
    blocks: tuple = ()

    def as_dict(self):
        return {"id": self.id, "title": self.title, "status": self.status, "detail": self.detail,
                "fix": self.fix, "blocks": list(self.blocks)}


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _int(path):
    value = _read(path)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _meminfo(host, key):
    for line in (_read(host.proc / "meminfo") or "").splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return None


def _cards(host):
    base = Path(host.sysfs) / "class/drm"
    return sorted(base.glob("card*/device")) if base.is_dir() else []


def _gb(n):
    return f"{n / GB:.1f} GB"


def run_checks(host, s):
    out = []
    rel = release_engine(s.repo)  # a verified release engine needs no compiler and no make

    def add(id_, title, status, detail="", fix="", blocks=()):
        out.append(CheckResult(id_, title, status, detail, fix, tuple(blocks)))

    # System and Python
    if host.platform.startswith("linux"):
        add("os", "Linux", "pass", host.platform)
    else:
        add("os", "Linux", "fail", f"this system is {host.platform}", "Use a Linux system on a Strix Halo machine.", STEPS_ALL)
    ver = ".".join(str(x) for x in host.py_version)
    if tuple(host.py_version[:2]) >= PY_MIN:
        add("python", "Python 3.12 or newer", "pass", ver)
    else:
        add("python", "Python 3.12 or newer", "fail", f"this Python is {ver}",
            "Install Python 3.12 or newer, then run the command with it: python3.12 ./crucible setup", STEPS_ALL)

    # ROCm
    hipcc = Path(host.rocm) / "bin/hipcc"
    version = _read(Path(host.rocm) / ".info/version")
    if version is None:
        if hipcc.exists():
            add("rocm-version", "ROCm version", "warn", f"no version file; the tested version is {ROCM_VERSION}")
        else:
            add("rocm-version", f"ROCm {ROCM_VERSION}", "fail", f"no ROCm found under {host.rocm}", ROCM_FIX,
                ("start",) if rel else ("build", "start"))
    elif version.startswith(ROCM_VERSION):
        add("rocm-version", f"ROCm {ROCM_VERSION}", "pass", version)
    else:
        add("rocm-version", f"ROCm {ROCM_VERSION}", "warn", f"found {version}; only {ROCM_VERSION} is tested")
    if hipcc.exists():
        add("hipcc", "HIP compiler (hipcc)", "pass", str(hipcc))
    elif rel:
        add("hipcc", "HIP compiler (hipcc)", "skip", NOT_NEEDED)
    else:
        add("hipcc", "HIP compiler (hipcc)", "fail", f"{hipcc} not found", ROCM_FIX, ("build", "start"))
    rocminfo = Path(host.rocm) / "bin/rocminfo"
    if not rocminfo.exists():
        add("gpu-arch", "GPU is gfx1151", "fail", f"{rocminfo} not found", ROCM_FIX, ("start",))
    else:
        res = host.runner.run([str(rocminfo)])
        names = sorted(set(re.findall(r"\bgfx[0-9a-f]+\b", res.out)))
        if "gfx1151" in names:
            add("gpu-arch", "GPU is gfx1151", "pass", "gfx1151")
        elif res.rc != 0:
            add("gpu-arch", "GPU is gfx1151", "fail", f"rocminfo failed: {res.out.strip()[:200]}",
                "Check that your user can open the GPU (see gpu-access) and that ROCm is complete.", ("start",))
        else:
            add("gpu-arch", "GPU is gfx1151", "fail", f"found {', '.join(names) or 'no GPU'}; the engine runs on gfx1151 only",
                "Use a Ryzen AI Max+ 395 machine (Radeon 8060S). Other machines are not supported.", ("start",))

    # GPU access
    kfd = Path(host.dev) / "kfd"
    renders = sorted((Path(host.dev) / "dri").glob("renderD*")) if (Path(host.dev) / "dri").is_dir() else []
    rw = os.R_OK | os.W_OK
    if not kfd.exists():
        add("gpu-kfd", "GPU device /dev/kfd exists", "fail", "/dev/kfd is missing",
            "The amdgpu driver or ROCm is not complete, or the machine has just started. Check: lsmod | grep amdgpu", ("start",))
    elif host.can_access(kfd, rw) and any(host.can_access(r, rw) for r in renders):
        add("gpu-kfd", "GPU device /dev/kfd exists", "pass", "/dev/kfd")
        add("gpu-access", "User can open the GPU", "pass", "/dev/kfd and a render node")
    else:
        add("gpu-kfd", "GPU device /dev/kfd exists", "pass", "/dev/kfd")
        groups = host.groups()
        add("gpu-access", "User can open the GPU", "fail",
            f"cannot open /dev/kfd or a render node; your groups: {', '.join(dict.fromkeys(groups)) or 'none'}",
            'Add your user to the groups render and video, then log out and in again: sudo usermod -aG render,video "$USER" '
            "(run it yourself; this tool does not use sudo). If /dev/kfd is missing, the amdgpu driver or ROCm is not complete.",
            ("start",))

    # Build tools
    for id_, names, level in (("tool-make", ("make",), "fail"), ("tool-cxx", ("g++", "c++", "clang++"), "warn"),
                              ("tool-git", ("git",), "warn")):
        found = next((n for n in names if host.which(n)), None)
        title = f"{'/'.join(names)} installed"
        if found:
            add(id_, title, "pass", str(host.which(found)))
        elif rel and id_ != "tool-git":
            add(id_, title, "skip", NOT_NEEDED)
        else:
            add(id_, title, level, f"{names[0]} not found",
                "Install the build tools: Arch: sudo pacman -S --needed base-devel git. "
                "Ubuntu or Debian: sudo apt install build-essential git. Fedora: sudo dnf install make gcc-c++ git.",
                ("build", "start") if id_ == "tool-make" else ())

    # Memory
    total = _meminfo(host, "MemTotal")
    if total is not None and total >= MEM_MIN_KB:
        add("memory", "System memory 120 GiB or more", "pass", f"{total / 1024 / 1024:.0f} GiB")
    else:
        got = "unknown" if total is None else f"{total / 1024 / 1024:.0f} GiB"
        add("memory", "System memory 120 GiB or more", "fail", f"this machine has {got}",
            "The 4.05 pack with 4 long sessions needs a 128 GB machine. Other sizes are not supported.", ("start",))

    # GPU memory limit (GTT)
    totals = [t for t in (_int(c / "mem_info_gtt_total") for c in _cards(host)) if t]
    cmdline = _read(Path(host.proc) / "cmdline") or ""
    has_params = "amdgpu.gttsize" in cmdline and "ttm.pages_limit" in cmdline
    if not totals:
        add("gtt", "GPU memory limit (GTT)", "fail", "no amdgpu memory value found in /sys/class/drm",
            "Check that the amdgpu driver is loaded (lsmod | grep amdgpu) and that this is a Strix Halo machine.\n"
            + KERNEL_FIX, ("start",))
    elif max(totals) >= GTT_MIN_BYTES:
        add("gtt", "GPU memory limit (GTT)", "pass", _gb(max(totals)))
    else:
        why = ("The GPU of Strix Halo uses the system memory, and Linux lets it use only part of it by default. "
               "The 4.05 pack with 4 long sessions needs about 133 GB.")
        note = " The two parameters are on the command line already: restart if you changed them just now." if has_params else ""
        add("gtt", "GPU memory limit (GTT)", "fail", f"the limit is {_gb(max(totals))}. {why}{note}", KERNEL_FIX, ("start",))

    # GPU memory in use
    live = _server_alive(host, s)
    used = [(_int(c / "mem_info_gtt_used") or 0) for c in _cards(host)]
    if live:
        add("gpu-free", "GPU memory is free", "pass", "in use by the Crucible server (started by this tool, or the service)")
    elif used and max(used) > GPU_BUSY_BYTES:
        add("gpu-free", "GPU memory is free", "fail", f"{_gb(max(used))} of GPU memory is in use",
            "The GPU memory is held by another program (for example another inference server). Stop it first. "
            "Crucible does not start while it runs.", ("start",))
    else:
        add("gpu-free", "GPU memory is free", "pass", f"{_gb(max(used or [0]))} in use")

    # Disk
    state, detail = packs.verify(s.ckpt)
    need = s.spec.disk_gb * GB
    free = host.disk_free(s.models_dir)
    if state == "ok":
        add("disk-models", "Disk space for the pack", "pass", "the pack is on the disk already")
    elif free >= need:
        add("disk-models", "Disk space for the pack", "pass", f"{_gb(free)} free; the {s.pack} pack needs about {s.spec.disk_gb} GB")
    else:
        add("disk-models", "Disk space for the pack", "fail", f"{_gb(free)} free; the {s.pack} pack needs about {s.spec.disk_gb} GB",
            f"Free about {s.spec.disk_gb} GB, or choose another folder: ./crucible setup --models-dir DIR", ("download",))
    if s.state_dir is not None:
        free_state = host.disk_free(s.state_dir)
        want = s.state_disk_gib * 1024 ** 3
        if free_state >= want:
            add("disk-state", "Disk space for the conversation cache", "pass", f"{_gb(free_state)} free; limit {s.state_disk_gib} GiB")
        else:
            add("disk-state", "Disk space for the conversation cache", "warn",
                f"{_gb(free_state)} free; the limit is {s.state_disk_gib} GiB",
                "Lower the limit: ./crucible setup --state-disk-gib N, or use --no-state-dir.")
        home = Path(os.path.abspath(host.home))
        sd = Path(os.path.abspath(s.state_dir))
        if sd == home or home not in sd.parents:
            add("state-dir", "State folder is inside your home folder", "fail", f"{sd} is not inside your home folder {home}",
                "Use a new folder inside your home folder: ./crucible setup --state-dir ~/.cache/crucible/states", ("start",))
        elif sd.is_dir() and any(sd.iterdir()) and not (s.work / "server.pid").exists():
            add("state-dir", "State folder is inside your home folder", "warn", f"{sd} holds files; the server stops at start if "
                "one of them is not a Crucible state file", "Use a new, empty folder: ./crucible setup --state-dir DIR")
        else:
            add("state-dir", "State folder is inside your home folder", "pass", str(sd))
    else:
        add("state-dir", "State folder", "skip", "the conversation cache stays in memory (--no-state-dir)")

    # Port
    if live:
        add("port", f"Port {s.port} is free", "pass", "used by the Crucible server (started by this tool, or the service)")
    elif host.port_free(s.bind, s.port):
        add("port", f"Port {s.port} is free", "pass", f"{s.bind}:{s.port}")
    else:
        add("port", f"Port {s.port} is free", "fail", f"{s.bind}:{s.port} is in use",
            "Stop the program that uses the port, or choose another one: ./crucible setup --port 8791", ("start",))

    # What setup made
    if s.venv_python.exists():
        add("venv", "Python environment", "pass", str(s.venv))
    else:
        add("venv", "Python environment", "skip", "not made yet: ./crucible setup makes it")
    if s.engine.exists():
        add("engine", "Engine is built", "pass", f"release engine {rel['tag']}" if rel else str(s.engine))
    else:
        add("engine", "Engine is built", "skip", "not built yet: ./crucible setup builds it")
    if state == "ok":
        add("pack", f"Pack {s.pack} is complete", "pass", str(s.ckpt))
    elif state == "missing":
        add("pack", f"Pack {s.pack} is complete", "skip", "not downloaded yet: ./crucible setup downloads it")
    else:
        add("pack", f"Pack {s.pack} is complete", "fail", detail, "Run ./crucible setup again: a stopped download continues.")
    _key_file_check(s, add)
    return out


def key_file_results(s):
    """The key-file check alone (setup asks it again after the config step has made the key)."""
    out = []
    _key_file_check(s, lambda id_, title, status, detail="", fix="", blocks=(): out.append(CheckResult(id_, title, status, detail, fix, tuple(blocks))))
    return out


def _key_file_check(s, add):
    """The API key file (service design s3). Never shows the key. A bad file stops the server with exit 78."""
    title = "API key file"
    if s.api_key_file is None:
        add("key-file", title, "skip", "no key: the server listens on loopback only (./crucible setup --bind ADDRESS adds one)")
        return
    problem = keyfile.check_key_file(s.api_key_file)
    if problem is None:
        add("key-file", title, "pass", str(s.api_key_file))
    elif not os.path.lexists(s.api_key_file) and not (s.work / "config.json").exists():
        add("key-file", title, "skip", f"{s.api_key_file} is not made yet: ./crucible setup makes it")
    elif not os.path.lexists(s.api_key_file):
        add("key-file", title, "fail", f"{s.api_key_file} does not exist",
            f"Make the key: ./crucible setup --bind {s.bind}" + (" --api-key" if keyfile.is_loopback(s.bind) else ""),
            ("start",))
    else:
        fix = f"chmod 600 {s.api_key_file}" if "chmod 600" in problem else \
            f"Remove the file and make a new key: rm {s.api_key_file} && ./crucible setup --bind {s.bind}"
        add("key-file", title, "fail", problem.split(" Run:")[0], fix, ("start",))


def _server_alive(host, s):
    """True when the server that this tool started runs, or the service is active (the service writes no server.pid)."""
    if service.active(host):
        return True
    try:
        pid = int((s.work / "server.pid").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return False
    return bool(host.runner and host.runner.alive(pid))


def format_text(results):
    lines = []
    seen = {}
    for r in results:
        lines.append(f"{r.status.upper():<4}  {r.id:<13} {r.title}" + (f": {r.detail}" if r.detail else ""))
        if r.status == "fail" and r.fix:
            if r.fix in seen:
                lines.append(f"      fix: the same as for {seen[r.fix]}")
                continue
            seen[r.fix] = r.id
            fix = r.fix.split("\n")
            lines.append("      fix: " + fix[0])
            lines.extend("           " + x for x in fix[1:])
    counts = {k: sum(1 for r in results if r.status == k) for k in ("pass", "fail", "warn", "skip")}
    lines.append(f"{counts['pass']} passed, {counts['fail']} failed, {counts['warn']} warnings, {counts['skip']} skipped")
    return "\n".join(lines)


def main(host, s, as_json=False):
    results = run_checks(host, s)
    if as_json:
        host.out(json.dumps({"checks": [r.as_dict() for r in results],
                             "ok": not any(r.status == "fail" for r in results)}, indent=2))
    else:
        host.out(format_text(results))
    return 1 if any(r.status == "fail" for r in results) else 0


# ./crucible doctor --for start -------------------------------------------------------------------------------------
EX_TEMPFAIL = 75   # a condition that may pass soon (GPU memory in use, port busy, /dev/kfd not ready at boot)
EX_CONFIG = 78     # a bad setting: waiting does not help
TRANSIENT = frozenset({"gpu-free", "port", "gpu-kfd"})
START_WAIT_SECONDS = 120.0
START_WAIT_INTERVAL = 5.0
CHECK_FILE = "last-start-check.json"


def start_failures(host, s):
    """The failed checks that stop a start. Build-only checks (compiler, make) do not count. The files that a start reads count."""
    results = [r for r in run_checks(host, s) if r.id not in ("venv", "engine", "pack")]
    fails = [r for r in results if r.status == "fail" and "start" in r.blocks and "build" not in r.blocks]
    fix = "Run ./crucible setup again: it repairs what is missing."
    if not s.venv_python.exists():
        fails.append(CheckResult("venv", "Python environment", "fail", f"the Python environment {s.venv} is missing", fix, ("start",)))
    if not s.engine.exists():
        fails.append(CheckResult("engine", "Engine is built", "fail", f"the engine file {s.engine} is missing", fix, ("start",)))
    state, detail = packs.verify(s.ckpt)
    if state != "ok":
        what = "is missing" if state == "missing" else "is incomplete (" + detail + ")"
        fails.append(CheckResult("pack", f"Pack {s.pack}", "fail", f"the pack folder {s.ckpt} {what}", fix, ("start",)))
    return fails


def for_start(host, s, quiet=False, as_json=False, check_file=None):
    """Exit 0: ready to start. 75: try again later. 78: a bad setting. Writes the result in last-start-check.json."""
    t0 = host.clock()
    while True:
        fails = start_failures(host, s)
        hard = [f for f in fails if f.id not in TRANSIENT]
        if not fails or hard or host.clock() - t0 >= START_WAIT_SECONDS:
            break
        host.sleep(START_WAIT_INTERVAL)
    code = 0 if not fails else (EX_CONFIG if hard else EX_TEMPFAIL)
    fails = hard + [f for f in fails if f.id in TRANSIENT]  # a bad setting first: status shows the first one, and a busy port may pass soon
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(host.clock())), "ok": not fails, "code": code,
              "failed": [{"id": f.id, "detail": f.detail, "fix": f.fix} for f in fails]}
    path = Path(check_file) if check_file else s.work / CHECK_FILE
    try:
        Sandbox([path.parent]).write_json(path, record)
    except (OSError, ValueError):
        pass  # a check file that cannot be written must not hide the result
    if as_json:
        host.out(json.dumps({"checks": [f.as_dict() for f in fails], "ok": not fails, "code": code}, indent=2))
    elif quiet:
        for f in fails:
            host.out(f"{f.id}: {f.detail}")
    elif fails:
        host.out(format_text(fails))
    else:
        host.out("Ready to start.")
    return code
