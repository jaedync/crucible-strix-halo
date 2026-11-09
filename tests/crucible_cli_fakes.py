"""Fakes for the tests of ./crucible setup and doctor: a fake machine (/proc, /sys, ROCm), a fake runner,
a fake HTTP client and a fake prompt. No GPU, no network, no real disk beyond a temporary folder."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from crucible_cli import host as hostmod  # noqa: E402

GTT_OK = 133_000_000_000


class FakeProc:
    def __init__(self, pid):
        self.pid = pid
        self.dead = False

    def poll(self):
        return 1 if self.dead else None


class FakeRunner:
    """Records every command. Knows enough of venv, pip, hf, make, rocminfo and the server to run setup."""

    def __init__(self, fail=()):
        self.calls = []
        self.spawned = []
        self.terminated = []
        self.fail = set(fail)  # names: "download", "make", "pip", "venv"
        self.gfx = "gfx1151"
        self.make_output = ""
        self.live = set()
        self.server_dies = False

    def run(self, argv, cwd=None, env=None, log=None, echo=False):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        name = Path(argv[0]).name
        if name == "rocminfo":
            return hostmod.Result(0, f"  Name:                    {self.gfx}\n")
        if len(argv) > 2 and argv[1:3] == ["-m", "venv"]:
            if "venv" in self.fail:
                return hostmod.Result(1, "ensurepip is not available")
            py = Path(argv[3]) / "bin" / "python"
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("#!/bin/sh\n")
            py.chmod(0o755)
            (py.parent / "hf").write_text("#!/bin/sh\n")
            return hostmod.Result(0, "")
        if len(argv) > 2 and argv[1:3] == ["-m", "pip"]:
            return hostmod.Result(1 if "pip" in self.fail else 0, "")
        if len(argv) > 1 and argv[1] == "-c":  # the import check in the venv
            return hostmod.Result(0 if (Path(argv[0]).exists() and "pip" not in self.fail) else 1, "")
        if name == "hf":
            if "download" in self.fail:
                self.fail.discard("download")  # one failure, then success (a stopped download)
                return hostmod.Result(1, "network error")
            ckpt = Path(argv[argv.index("--local-dir") + 1])
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "config.json").write_text("{}")
            (ckpt / "tokenizer.json").write_text("{}")
            (ckpt / "ngram_embedding.safetensors").write_text("x")
            (ckpt / "model-00001-of-00001.safetensors").write_text("x")
            (ckpt / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}))
            return hostmod.Result(0, "")
        if name == "make":
            if "make" in self.fail:
                return hostmod.Result(2, self.make_output or "error")
            if self.make_output and "SPLIT=0" not in argv:
                return hostmod.Result(2, self.make_output)
            exe = Path(cwd) / "build" / "engine" / "crucible"
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.write_text("#!/bin/sh\n")
            exe.chmod(0o755)
            return hostmod.Result(0, "")
        return hostmod.Result(0, "")

    def spawn(self, argv, cwd=None, env=None, log=None):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        self.spawned.append({"argv": argv, "env": dict(env or {}), "cwd": str(cwd)})
        proc = FakeProc(40000 + len(self.spawned))
        proc.dead = self.server_dies
        if not self.server_dies:
            self.live.add(proc.pid)
        return proc

    def alive(self, pid):
        return pid in self.live

    def terminate(self, pid):
        self.terminated.append(pid)
        self.live.discard(pid)
        return True


class FakeHttp:
    def __init__(self, ready_after=0):
        self.gets = []
        self.posts = []
        self.key_files = []  # the key_file argument of each get and chat_stream call
        self.ready_after = ready_after

    def get(self, url, timeout=5, key_file=None):
        self.gets.append(url)
        self.key_files.append(key_file)
        if len(self.gets) > self.ready_after:
            return 200, '{"data":[{"id":"crucible"}]}'
        return 0, ""

    def chat_stream(self, url, body, timeout=600, key_file=None):
        self.posts.append((url, body))
        self.key_files.append(key_file)
        return {"ok": True, "first_token_s": 0.42, "total_s": 1.5, "text": "Hello!", "tokens": 3}


class FakePrompt:
    def __init__(self, answers=()):
        self.answers = list(answers)
        self.asked = []

    def __call__(self, question, default=""):
        self.asked.append(question)
        if self.answers:
            return self.answers.pop(0)
        return default


class NoPrompt:
    def __call__(self, question, default=""):
        raise AssertionError(f"prompted with --yes: {question}")


def make_machine(tmp, *, gtt=GTT_OK, gtt_used=2_000_000, vram_used=600_000_000, version="7.2.4", rocm=True,
                 hipcc=True, mem_total_kb=130_000_000, cmdline="root=/dev/nvme0n1p2 rw", py=(3, 14, 0),
                 free=2_000_000_000_000, busy_ports=(), deny=(), tools=("make", "g++", "git"),
                 groups=("video", "render"), platform="linux"):
    """Build a healthy fake machine under tmp/machine. Arguments break one thing at a time."""
    m = Path(tmp) / "machine"
    (m / "proc").mkdir(parents=True, exist_ok=True)
    (m / "proc/meminfo").write_text(f"MemTotal:       {mem_total_kb} kB\nMemAvailable:   {mem_total_kb - 5_000_000} kB\n")
    (m / "proc/cmdline").write_text(cmdline + "\n")
    dev = m / "sys/class/drm/card0/device"
    dev.mkdir(parents=True, exist_ok=True)
    (dev / "mem_info_gtt_total").write_text(f"{gtt}\n")
    (dev / "mem_info_gtt_used").write_text(f"{gtt_used}\n")
    (dev / "mem_info_vram_total").write_text("536870912\n")
    (dev / "mem_info_vram_used").write_text(f"{vram_used}\n")
    (m / "dev/dri").mkdir(parents=True, exist_ok=True)
    (m / "dev/kfd").write_text("")
    (m / "dev/dri/renderD128").write_text("")
    r = m / "rocm"
    (r / ".info").mkdir(parents=True, exist_ok=True)
    (r / "bin").mkdir(parents=True, exist_ok=True)
    if rocm:
        (r / ".info/version").write_text(version + "\n")
        (r / "bin/rocminfo").write_text("")
    if hipcc:
        (r / "bin/hipcc").write_text("")
    home = Path(tmp) / "home"
    home.mkdir(exist_ok=True)
    bins = {t: f"/usr/bin/{t}" for t in tools}
    denied = {str(m / d) for d in deny}
    return hostmod.Host(
        proc=m / "proc", sysfs=m / "sys", dev=m / "dev", rocm=r, home=home, env={"HOME": str(home)},
        py_version=py, platform=platform, which=lambda n: bins.get(n),
        disk_free=lambda p: free, port_free=lambda h, p: p not in busy_ports,
        groups=lambda: list(groups), can_access=lambda p, mode: str(p) not in denied,
        runner=None, http=None, prompt=NoPrompt(), clock=lambda: 1000.0, sleep=lambda s: None, out=lambda s: None,
        is_tty=False)


def wire(tmp, runner=None, http=None, prompt=None, **machine):
    host = make_machine(tmp, **machine)
    host.runner = runner or FakeRunner()
    host.http = http or FakeHttp()
    if prompt is not None:
        host.prompt = prompt
    host.lines = []
    host.out = host.lines.append
    return host


def make_repo(tmp):
    """A fake checkout: the files that setup reads."""
    repo = Path(tmp) / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "requirements-cpu.txt").write_text("numpy\n")
    (repo / "VERSION").write_text("0.31.0\n")
    return repo
