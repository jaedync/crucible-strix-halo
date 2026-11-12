"""A fake world for the tests of `./crucible update` and `./crucible uninstall`.

The world has: a local git repo with two release tags, a release folder for the new tag (the three release files), an
installed tree (a real git clone at the old tag with the work folder, the venv, the models and the release engine), a
fake systemd, a fake server health check, and a runner that runs git and the stub `./crucible` of each tree for real.
No network, no GPU, no real systemd, no sudo.

Markers: a file with one of these names in the tree that is at DIR makes the fake machine fail at that point.
  BAD_START    the service does not start        BAD_HEALTH  the service starts and /health does not answer
  BAD_SMOKE    the test request gets no token    BAD_INSTALL the stub `service install` exits 1
"""
import gzip
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from crucible_cli import host as hostmod  # noqa: E402
from crucible_cli.packs import PACKS  # noqa: E402
from crucible_cli.runner import Runner  # noqa: E402
from crucible_cli_fakes import FakePrompt, make_machine  # noqa: E402

OLD, NEW = "v0.1.0", "v0.2.0"
STEM = "crucible-engine-{tag}-gfx1151-rocm7.2-linux-x86_64"
CANARY = "ENGINE-WAS-RUN"
XDG_HOMES = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME")
GIT = shutil.which("git")
EMAIL = "t" + chr(64) + "example.invalid"   # built, so that the export scan sees no address
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": EMAIL, "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": EMAIL}

STUB = '''#!/usr/bin/env python3
"""Test stand-in for ./crucible. It logs each call to CRUCIBLE_TEST_LOG and acts on `service install` and `start`."""
import os
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent
version = (root / "VERSION").read_text().strip()
args = sys.argv[1:]
with open(os.environ["CRUCIBLE_TEST_LOG"], "a") as log:
    log.write("crucible %s %s\\n" % (version, " ".join(args)))
if args[:2] == ["service", "install"]:
    if (root / "BAD_INSTALL").exists():
        sys.exit(1)
    (root / ".crucible").mkdir(exist_ok=True)
    (root / ".crucible" / "serve.sh").write_text("serve %s\\n" % version)
    unit = pathlib.Path(os.environ["CRUCIBLE_TEST_UNIT"])
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("unit %s\\n" % version)
    sys.exit(0)
if args[:1] == ["start"]:
    if (root / "BAD_START").exists():
        sys.exit(1)
    (root / ".crucible").mkdir(exist_ok=True)
    (root / ".crucible" / "server.pid").write_text("40001 0\\n")
    sys.exit(0)
sys.exit(0)
'''


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def git(repo, *args, check=True):
    res = subprocess.run([GIT, "-C", str(repo), *args], capture_output=True, text=True, env={**os.environ, **GIT_ENV})
    if check and res.returncode != 0:
        raise AssertionError(f"git {args}: {res.stderr}")
    return res.stdout.strip()


def snapshot(tree):
    """A stable description of every file, link and folder of a tree (names, modes, content hashes), without .git."""
    out = {}
    tree = Path(tree)
    for dirpath, dirnames, filenames in os.walk(tree, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in sorted(filenames + [d for d in dirnames if (Path(dirpath) / d).is_symlink()]):
            p = Path(dirpath) / name
            rel = str(p.relative_to(tree))
            if p.is_symlink():
                out[rel] = ("link", os.readlink(p))
            else:
                out[rel] = (stat.S_IMODE(p.stat().st_mode), sha256_file(p))
    return out


class Machine:
    """The fake systemd and the fake server. `starts` is a queue of results for the next starts (default: all good)."""

    def __init__(self, world):
        self.world = world
        self.active = False
        self.unit_present = False
        self.calls = []
        self.starts = []
        self.live = set()
        self.terminated = []
        self.make_calls = []
        self.fail_make = False
        self.stop_fails = False
        self.fail_pip = False
        self.freeze = "numpy==1.0.0\nrequests==2.0.0\n"
        self.put_back = []
        self.installed_after = None   # what `pip freeze` lists after the install of new requirements (default: the same)
        self.pip_installed = False
        self.fail_put_back = False
        self.old_unhealthy = False
        self.events = []

    @property
    def verbs(self):
        return [c[0] for c in self.calls]

    def marker(self, name):
        return (self.world.dir / name).exists()


class World:
    def __init__(self, test, release_engine=True, bad_start=False, bad_health=False, bad_smoke=False, bad_install=False,
                 new_requirements=None, tracked_dirs=()):
        self.tracked_dirs = tuple(tracked_dirs)   # folders that the repository tracks, besides docs (a file keep.txt in each)
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self.test = test
        self.t = Path(self.tmp.name)
        self.home = self.t / "home"
        self.home.mkdir()
        self.parent = self.t / "apps"
        self.parent.mkdir()
        self.dir = self.parent / "crucible"
        self.release = self.t / "release"
        self.release.mkdir()
        self.log = self.t / "stub.log"
        self.log.write_text("")
        self.unit = self.home / ".config/systemd/user/crucible-server.service"
        self.machine = Machine(self)
        self.make_source(bad_start, bad_health, bad_smoke, bad_install, new_requirements)
        self.make_release()
        self.make_install(release_engine)
        self.old_commit = git(self.dir, "rev-parse", "HEAD")

    # source repo and release folder ---------------------------------------------------------------------------------
    def make_source(self, bad_start, bad_health, bad_smoke, bad_install, new_requirements):
        self.src = self.t / "src"
        self.src.mkdir()
        git(self.src, "init", "-q", "-b", "main")
        (self.src / "crucible").write_text(STUB)
        (self.src / "crucible").chmod(0o755)
        (self.src / "VERSION").write_text("0.1.0\n")
        (self.src / "requirements-cpu.txt").write_text("numpy\n")
        (self.src / "docs").mkdir()
        (self.src / "docs" / "old.md").write_text("old\n")
        for d in self.tracked_dirs:
            (self.src / d).mkdir(parents=True, exist_ok=True)
            (self.src / d / "keep.txt").write_text("tracked\n")
        git(self.src, "add", "-A")
        git(self.src, "commit", "-q", "-m", "old release")
        git(self.src, "tag", OLD)
        self.old_src_commit = git(self.src, "rev-parse", "HEAD")
        (self.src / "VERSION").write_text("0.2.0\n")
        (self.src / "docs" / "new.md").write_text("new\n")
        if new_requirements is not None:
            (self.src / "requirements-cpu.txt").write_text(new_requirements)
        for flag, name in ((bad_start, "BAD_START"), (bad_health, "BAD_HEALTH"), (bad_smoke, "BAD_SMOKE"),
                           (bad_install, "BAD_INSTALL")):
            if flag:
                (self.src / name).write_text("x\n")
        git(self.src, "add", "-A")
        git(self.src, "commit", "-q", "-m", "new release")
        git(self.src, "tag", NEW)
        self.new_commit = git(self.src, "rev-parse", "HEAD")
        self.repo_url = f"file://{self.src}"

    def make_release(self, engine=None, tag=NEW, commit=None, arch="gfx1151", rocm="7.2", needs=(), glibc_min="2.1",
                     glibcxx_min="3.4", binary_sha=None):
        for p in self.release.iterdir():
            p.unlink()
        stem = STEM.format(tag=tag)
        raw = engine if engine is not None else f"#!/bin/sh\necho {CANARY}\n".encode()
        self.new_engine = raw
        self.new_engine_sha = sha256_bytes(raw)
        with gzip.GzipFile(self.release / f"{stem}.gz", "wb", mtime=0) as f:
            f.write(raw)
        lines = ["format 1", f"tag {tag}", f"commit {commit or self.new_commit}", f"arch {arch}", f"rocm {rocm}",
                 f"binary_sha256 {binary_sha or self.new_engine_sha}", f"glibc_min {glibc_min}", f"glibcxx_min {glibcxx_min}"]
        lines += [f"need {n}" for n in needs]
        (self.release / f"{stem}.needs").write_text("\n".join(lines) + "\n")
        self.write_sums(tag)

    def write_sums(self, tag=NEW, skip=(), extra=()):
        stem = STEM.format(tag=tag)
        names = [n for n in (f"{stem}.gz", f"{stem}.needs") if n not in skip]
        lines = [f"{sha256_file(self.release / n)}  {n}" for n in names] + list(extra)
        (self.release / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    def asset(self, kind, tag=NEW):
        return self.release / f"{STEM.format(tag=tag)}.{kind}"

    # the installed tree ---------------------------------------------------------------------------------------------
    def make_install(self, release_engine):
        res = subprocess.run([GIT, "-c", "advice.detachedHead=false", "clone", "-q", "--branch", OLD, str(self.src),
                              str(self.dir)], capture_output=True, text=True, env={**os.environ, **GIT_ENV})
        assert res.returncode == 0, res.stderr
        (self.dir / ".git/crucible-install").write_text(f"installed by install.sh, tag {OLD}\n")
        work = self.dir / ".crucible"
        work.mkdir()
        self.models = self.dir / "models"
        self.pack = self.models / PACKS["4.05"].folder
        self.pack.mkdir(parents=True)
        (self.pack / "weights.safetensors").write_text("WEIGHTS\n")
        self.states = self.home / ".cache/crucible/states"
        self.states.mkdir(parents=True)
        (self.states / "conv.state").write_text("STATE\n")
        (work / "state-dir.json").write_text(json.dumps({"state_dir": os.path.realpath(self.states), "made_by": "crucible setup"}) + "\n")
        (work / "config.json").write_text(json.dumps({"pack": "4.05", "models_dir": str(self.models), "state_dir": str(self.states),
                                                     "port": 8790, "streams": 4, "ctx": 262144, "state_disk_gib": 200,
                                                     "jobs": 1, "bind": "127.0.0.1", "api_key_file": None}))
        (work / "serve.sh").write_text("serve 0.1.0\n")
        (work / "setup-state.json").write_text(json.dumps({"version": 1, "steps": {}, "load_seconds": 60}))
        (self.dir / ".venv/bin").mkdir(parents=True)
        (self.dir / ".venv/bin/python").write_text("#!/bin/sh\n")
        (self.dir / ".venv/bin/python").chmod(0o755)
        (self.dir / "build/engine").mkdir(parents=True)
        engine = b"#!/bin/sh\necho old engine\n"
        (self.dir / "build/engine/crucible").write_bytes(engine)
        (self.dir / "build/engine/crucible").chmod(0o755)
        self.old_engine_sha = sha256_bytes(engine)
        if release_engine:
            rec = {"format": 1, "tag": OLD, "commit": self.old_src_commit, "engine_sha256": self.old_engine_sha,
                   "arch": "gfx1151", "rocm": "7.2", "asset": STEM.format(tag=OLD) + ".gz", "glibc_min": "2.1",
                   "glibcxx_min": "3.4"}
            (self.dir / "build/engine/release.json").write_text(json.dumps(rec) + "\n")
        self.unit.parent.mkdir(parents=True)
        self.unit.write_text("unit 0.1.0\n")
        self.machine.unit_present = True
        self.machine.active = True

    # the host -------------------------------------------------------------------------------------------------------
    def host(self, answers=None, tty=True, env=None, **machine):
        h = make_machine(self.t, **machine)
        h.home = self.home
        # The XDG folders of the outside home would move the unit and the key out of the fake home (the GitHub image sets
        # XDG_CONFIG_HOME): the fake home starts with none.
        outside = {k: v for k, v in os.environ.items() if k not in XDG_HOMES}
        h.env = {**outside, **GIT_ENV, "HOME": str(self.home), "USER": "tester", "CRUCIBLE_REPO_URL": self.repo_url,
                 "CRUCIBLE_RELEASE_BASE": f"file://{self.release}", "CRUCIBLE_TEST_LOG": str(self.log),
                 "CRUCIBLE_TEST_UNIT": str(self.unit), **(env or {})}
        h.runner = UpdateRunner(self)
        h.http = UpdateHttp(self)
        h.is_tty = tty
        h.prompt = FakePrompt(answers or [])
        h.tty_prompt = None
        h.lines = []
        h.out = h.lines.append
        h.python_exe = sys.executable
        h.sleep = lambda s: None
        self.last_host = h
        return h

    def stub_calls(self):
        return [ln for ln in self.log.read_text().splitlines() if ln.strip()]

    def leftovers(self):
        """Names in the parent folder other than the install folder."""
        return sorted(p.name for p in self.parent.iterdir() if p.name != "crucible")


class UpdateRunner(Runner):
    """Runs git and the stub for real. Answers systemctl, loginctl, journalctl, make and the server checks."""

    def __init__(self, world):
        self.world = world
        self.m = world.machine

    def run(self, argv, cwd=None, env=None, log=None, echo=False):
        argv = [str(a) for a in argv]
        name = Path(argv[0]).name
        m, w = self.m, self.world
        if name == "systemctl":
            args = argv[2:] if argv[1] == "--user" else argv[1:]
            m.calls.append(args)
            verb = args[0]
            m.events.append(("ctl", verb))
            if verb == "is-system-running":
                return hostmod.Result(0, "running")
            if verb in ("start", "restart"):
                ok = m.starts.pop(0) if m.starts else True
                if ok and m.marker("BAD_START"):
                    ok = False
                if not ok:
                    m.active = False
                    return hostmod.Result(1, "Job for crucible-server.service failed.")
                m.active = True
                return hostmod.Result(0, "")
            if verb == "stop":
                if m.stop_fails:
                    return hostmod.Result(1, "stop failed")
                m.active = False
                return hostmod.Result(0, "")
            if verb == "is-active":
                return hostmod.Result(0 if m.active else 3, "active" if m.active else "inactive")
            if verb == "show":
                names = [args[i + 1] for i, a in enumerate(args) if a == "-p"]
                props = {"LoadState": "loaded", "ActiveState": "active" if m.active else "inactive",
                         "SubState": "running" if m.active else "dead", "Result": "success", "NRestarts": "0",
                         "ExecMainStatus": "0", "StatusText": "", "UnitFileState": "enabled", "ActiveEnterTimestamp": ""}
                return hostmod.Result(0, "\n".join(f"{n}={props.get(n, '')}" for n in names))
            return hostmod.Result(0, "")
        if len(argv) > 3 and argv[1:3] == ["-m", "pip"]:
            m.events.append(("pip", " ".join(argv[3:])))
            if argv[3] == "freeze":
                text = m.installed_after if (m.pip_installed and m.installed_after is not None) else m.freeze
                if log is not None:
                    log.write(text.encode())
                return hostmod.Result(0, text)
            if argv[3] == "install" and "--no-deps" in argv:
                m.put_back.append(Path(argv[-1]).read_text())
                return hostmod.Result(1 if m.fail_put_back else 0, "put back failed" if m.fail_put_back else "")
            if argv[3] == "install":
                m.pip_installed = True
            return hostmod.Result(1 if m.fail_pip else 0, "pip error" if m.fail_pip else "")
        if name == "loginctl":
            return hostmod.Result(0, "Linger=yes")
        if name == "journalctl":
            return hostmod.Result(0, "journal line")
        if name == "systemd-run":
            return hostmod.Result(127, "systemd-run: not found")
        if name == "make":
            m.make_calls.append((argv, str(cwd)))
            if m.fail_make:
                return hostmod.Result(2, "error: build failed")
            exe = Path(cwd) / "build/engine/crucible"
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.write_text("#!/bin/sh\necho built\n")
            exe.chmod(0o755)
            return hostmod.Result(0, "")
        if name in ("ldconfig",):
            return hostmod.Result(0, "")
        stub = len(argv) > 2 and Path(argv[1]).name == "crucible"
        if stub:
            m.events.append(("stub", " ".join(argv[2:])))
        res = super().run(argv, cwd=cwd, env=env if env is not None else w.last_host.env, log=log, echo=echo)
        if stub and argv[2:3] == ["start"] and res.rc == 0:
            m.live.add(40001)
        return res

    def alive(self, pid):
        return int(pid) in self.m.live

    def terminate(self, pid, wait_s=180):
        self.m.terminated.append(int(pid))
        self.m.live.discard(int(pid))
        return True


class UpdateHttp:
    def __init__(self, world):
        self.world = world
        self.gets = []
        self.posts = []

    def get(self, url, timeout=5, key_file=None):
        self.gets.append(url)
        m = self.world.machine
        server_up = m.active or bool(m.live)
        old = (self.world.dir / "VERSION").read_text() == "0.1.0\n"
        if server_up and not m.marker("BAD_HEALTH") and not (old and m.old_unhealthy):
            return 200, "ok"
        return 0, ""

    def chat_stream(self, url, body, timeout=900, key_file=None):
        self.posts.append((url, body))
        if self.world.machine.marker("BAD_SMOKE"):
            return {"ok": False, "error": "no token"}
        return {"ok": True, "first_token_s": 0.4, "total_s": 1.0, "text": "Hello", "tokens": 3}
