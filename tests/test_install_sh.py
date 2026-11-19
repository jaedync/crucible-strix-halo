"""Tests of install.sh against a local fake release folder and a local git repo. No network, no real sudo, no GPU.

The release folder has the three files of a real release: the engine (gzip), the manifest (.needs) and SHA256SUMS.
The source repo is a local git repo with one tag. PATH shims log every call of sudo, git and the other tools.
Every run uses its own session, so no run can read a real terminal: the answers come from CRUCIBLE_TTY (a file).
"""
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "install.sh"
TAG = "v9.9.9"
STEM = f"crucible-engine-{TAG}-gfx1151-rocm7.2-linux-x86_64"
PACK_FOLDER = "Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6"   # a pack folder that setup makes (tests/test_sec_install.py compares the list in install.sh)
CANARY = "ENGINE-WAS-RUN"
REAL_GIT = shutil.which("git")
REAL_LDCONFIG = shutil.which("ldconfig") or "/sbin/ldconfig"
LOGGED = ("sudo", "curl", "git", "gzip", "tar", "python3", "pacman", "apt-get", "dnf", "su", "systemctl")

STUB_CRUCIBLE = """#!/bin/sh
# Test stand-in for ./crucible: records how it was called.
printf '%s\\n' "$*" > "$PWD/handover.args"
cat > "$PWD/handover.stdin" 2>/dev/null
exit "${FAKE_SETUP_RC:-0}"
"""

HOST_TOOLS = ["bash", "sh", "env", "sed", "awk", "grep", "sort", "head", "tail", "tr", "cut", "uname", "getconf", "id", "mkdir", "mv",
              "cp", "rm", "cat", "ls", "du", "df", "date", "dirname", "basename", "mktemp", "chmod", "ln", "readlink", "realpath", "wc",
              "tee", "stat", "find", "touch", "printf", "true", "false", "test", "sleep", "xargs", "cmp", "uniq", "od", "hostname"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


class Env:
    """One test world: a home folder, a fake release folder, a local source repo and PATH shims."""

    def __init__(self, test):
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.home = self.t / "home"
        self.home.mkdir()
        self.release = self.t / "release"
        self.release.mkdir()
        self.shims = self.t / "shims"
        self.shims.mkdir()
        self.log = self.t / "shim.log"
        self.log.write_text("")
        self.tty = self.t / "tty"
        self.tty.write_text("")
        self.dir = self.home / "crucible"
        self.ld_hide = self.t / "ld.hide"
        self.ld_hide.write_text("")
        self.ld_add = self.t / "ld.add"
        self.ld_add.write_text("")
        self.os_release = self.t / "os-release"
        self.os_release.write_text('ID=ubuntu\nID_LIKE=debian\nVERSION_ID="24.04"\n')
        self.df_avail = self.t / "df.avail"   # free KiB that the df shim reports (item 3: the disk checks)
        self.df_avail.write_text(str(2 * 1024 ** 3))
        self.make_machine()
        self.make_shims()
        self.make_repo()
        self.make_release()

    # world ------------------------------------------------------------------------------------------------------
    def make_shims(self):
        for name in LOGGED:
            real = REAL_GIT if name == "git" else None
            body = f'#!/bin/sh\nprintf "%s %s\\n" {name} "$*" >> "{self.log}"\n'
            if name == "git":
                body += f'exec "{real}" "$@"\n'
            elif name == "sudo":
                # "installs" what the add file lists: the next ldconfig call shows it
                body += f'[ -n "$SHIM_INSTALL" ] && printf "\\t%s (libc6,x86-64) => /usr/lib/%s\\n" "$SHIM_INSTALL" "$SHIM_INSTALL" >> "{self.ld_add}"\nexit 0\n'
            elif name in ("curl", "gzip", "tar"):
                body += f'exec "$(PATH="$SHIM_REAL_PATH" command -v {name})" "$@"\n'
            elif name == "python3":
                body += 'exec "$(PATH="$SHIM_REAL_PATH" command -v python3)" "$@"\n'
            else:
                body += "exit 99\n"
            p = self.shims / name
            p.write_text(body)
            p.chmod(0o755)
        df = self.shims / "df"
        df.write_text(f'#!/bin/sh\nprintf "Filesystem 1024-blocks Used Available Capacity Mounted on\\n"\n'
                      f'printf "fake 9999999999 0 %s 1%% /\\n" "$(cat "{self.df_avail}")"\n')
        df.chmod(0o755)
        ld = self.shims / "ldconfig"
        ld.write_text(f'#!/bin/sh\n"{REAL_LDCONFIG}" -p | grep -v -F -f "{self.ld_hide}"\ncat "{self.ld_add}"\n')
        ld.chmod(0o755)

    def make_machine(self, mem_kb=130_000_000, gfx_version="110501", gtt=133_000_000_000, kfd=True):
        """A Strix Halo machine for the read-only checks of install.sh (CRUCIBLE_PROC, CRUCIBLE_SYS)."""
        self.proc = self.t / "machine/proc"
        self.sys = self.t / "machine/sys"
        shutil.rmtree(self.t / "machine", ignore_errors=True)
        self.proc.mkdir(parents=True)
        (self.proc / "meminfo").write_text(f"MemTotal:       {mem_kb} kB\nMemAvailable:   {mem_kb // 2} kB\n")
        card = self.sys / "class/drm/card0/device"
        card.mkdir(parents=True)
        (card / "mem_info_gtt_total").write_text(f"{gtt}\n")
        if kfd:
            node = self.sys / "class/kfd/kfd/topology/nodes/1"
            node.mkdir(parents=True)
            (node / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {gfx_version}\n")

    def make_repo(self):
        self.src = self.t / "src"
        self.src.mkdir()
        g = lambda *a: run([REAL_GIT, "-C", str(self.src), *a], env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"})
        g("init", "-q", "-b", "main")
        g("config", "user.email", "test-user")
        g("config", "user.name", "t")
        (self.src / "crucible").write_text(STUB_CRUCIBLE)
        (self.src / "crucible").chmod(0o755)
        (self.src / "VERSION").write_text("9.9.9\n")
        (self.src / "tools").mkdir()
        (self.src / "tools" / "update.sh").write_text('#!/bin/sh\nprintf "update.sh %s make=%s\\n" "$*" "$CRUCIBLE_MAKE" > "$PWD/update.args"\n')
        (self.src / "tools" / "update.sh").chmod(0o755)
        g("add", "-A")
        g("commit", "-q", "-m", "release")
        g("tag", TAG)
        self.commit = g("rev-parse", "HEAD").stdout.strip()
        self.repo_url = f"file://{self.src}"

    def make_release(self, engine=None, glibc_min="2.1", glibcxx_min="3.4", needs=(), commit=None):
        for p in self.release.iterdir():
            p.unlink()
        engine = engine if engine is not None else f"#!/bin/sh\necho {CANARY} > \"$HOME/engine.ran\"\n".encode()
        raw = self.t / "engine.raw"
        raw.write_bytes(engine if isinstance(engine, bytes) else engine.encode())
        gz = self.release / f"{STEM}.gz"
        with gzip.GzipFile(gz, "wb", mtime=0) as f:
            f.write(raw.read_bytes())
        self.engine_sha = sha256(raw)
        lines = ["format 1", f"tag {TAG}", f"commit {commit or self.commit}", "arch gfx1151", "rocm 7.2",
                 f"binary_sha256 {self.engine_sha}", f"glibc_min {glibc_min}", f"glibcxx_min {glibcxx_min}"]
        lines += [f"need {n}" for n in needs]
        (self.release / f"{STEM}.needs").write_text("\n".join(lines) + "\n")
        self.write_sums()

    def write_sums(self, skip=()):
        out = [f"{sha256(self.release / n)}  {n}" for n in (f"{STEM}.gz", f"{STEM}.needs") if n not in skip]
        (self.release / "SHA256SUMS").write_text("\n".join(out) + "\n")

    # run --------------------------------------------------------------------------------------------------------
    def env(self, **extra):
        e = {
            "HOME": str(self.home), "PATH": f"{self.shims}:{os.environ['PATH']}", "SHIM_REAL_PATH": os.environ["PATH"],
            "CRUCIBLE_RELEASE_BASE": f"file://{self.release}", "CRUCIBLE_REPO_URL": self.repo_url, "CRUCIBLE_TAG": TAG,
            "CRUCIBLE_DIR": str(self.dir), "CRUCIBLE_TTY": str(self.tty), "CRUCIBLE_OS_RELEASE": str(self.os_release),
            "CRUCIBLE_PROC": str(self.proc), "CRUCIBLE_SYS": str(self.sys),
            "XDG_STATE_HOME": str(self.home / ".state"), "TERM": "dumb", "LC_ALL": "C",
            "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
        }
        e.update(extra)
        return e

    def install(self, *args, answers="", via_pipe=False, script=INSTALL, **extra):
        self.tty.write_text(answers)
        if via_pipe:
            return subprocess.run(["bash", "-s", "--", *args], input=script.read_text(), capture_output=True, text=True,
                                  env=self.env(**extra), start_new_session=True, cwd=self.t, timeout=120)
        return subprocess.run(["bash", str(script), *args], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              env=self.env(**extra), start_new_session=True, cwd=self.t, timeout=120)

    def calls(self, name=None):
        lines = self.log.read_text().splitlines()
        return [ln for ln in lines if name is None or ln.split(" ", 1)[0] == name]


class InstallTestCase(unittest.TestCase):
    def setUp(self):
        self.w = Env(self)

    def out(self, r):
        return r.stdout + r.stderr


class GoodDownloadTest(InstallTestCase):
    def test_tag_flag_overrides_environment_and_is_not_passed_to_setup(self):
        for option in ("--tag", "--version", "--tag=", "--version="):
            args = [option + TAG] if option.endswith("=") else [option, TAG]
            with self.subTest(option=option):
                r = self.w.install("--yes", *args, CRUCIBLE_TAG="v8.8.8")
                self.assertEqual(r.returncode, 0, self.out(r))
                self.assertEqual((self.w.dir / "handover.args").read_text().strip(), "setup --yes")
                self.assertEqual(json.loads((self.w.dir / "build/engine/release.json").read_text())["tag"], TAG)

    def test_tag_flag_needs_a_release_name(self):
        for args in (("--tag",), ("--version",), ("--tag=",), ("--tag", "latest"), ("--tag", ""), ("--version", "")):
            with self.subTest(args=args):
                r = self.w.install(*args)
                self.assertEqual(r.returncode, 2, self.out(r))
                self.assertFalse(self.w.dir.exists())
                if args[0] in ("--tag", "--version") and args[1:] in ((), ("",)):
                    self.assertIn(f"{args[0]} needs a release name, such as v0.37.2.", self.out(r))

    def test_update_keeps_the_tag_flag_for_the_launcher(self):
        self.assertEqual(self.w.install("--yes").returncode, 0)
        cli = self.w.dir / "tools/crucible_cli"
        cli.mkdir(parents=True)
        (cli / "update.py").write_text("# use the launcher updater\n")
        r = self.w.install("update", "--tag", TAG)
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual((self.w.dir / "handover.args").read_text().strip(), f"update --tag {TAG}")

    def test_good_release_installs_and_hands_over(self):
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        d = self.w.dir
        engine = d / "build/engine/crucible"
        self.assertTrue(engine.is_file() and os.access(engine, os.X_OK))
        self.assertEqual(sha256(engine), self.w.engine_sha)
        info = json.loads((d / "build/engine/release.json").read_text())
        self.assertEqual(info["tag"], TAG)
        self.assertEqual(info["commit"], self.w.commit)
        self.assertEqual(info["engine_sha256"], self.w.engine_sha)
        self.assertEqual(info["format"], 1)
        self.assertEqual((d / "handover.args").read_text().strip(), "setup --yes")
        self.assertEqual(run([REAL_GIT, "-C", str(d), "rev-parse", "HEAD"]).stdout.strip(), self.w.commit)
        self.assertEqual(self.w.calls("sudo"), [])
        self.assertFalse((self.w.home / "engine.ran").exists(), "the installer must never run the engine")

    def test_steps_are_numbered_and_the_log_path_is_shown(self):
        r = self.w.install("--yes")
        text = self.out(r)
        for n in range(1, 8):
            self.assertIn(f"[{n}/7]", text)
        log = self.w.home / ".state/crucible/install.log"
        self.assertTrue(log.is_file())
        self.assertIn(str(log), text)
        self.assertIn(f"Run {self.w.dir}/crucible setup", text)
        self.assertNotIn("./crucible", text)

    def test_pipe_install_gives_setup_the_terminal(self):
        r = self.w.install("--yes", via_pipe=True, answers="from-the-terminal\n")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual((self.w.dir / "handover.stdin").read_text(), "from-the-terminal\n")

    def test_setup_failure_code_is_returned(self):
        r = self.w.install("--yes", FAKE_SETUP_RC="7")
        self.assertEqual(r.returncode, 7, self.out(r))

    def test_own_flags_are_not_passed_to_setup(self):
        r = self.w.install("--dir", str(self.w.home / "elsewhere"), "--pack", "3.05", "--state-dir", "x")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual((self.w.home / "elsewhere/handover.args").read_text().strip(), "setup --pack 3.05 --state-dir x")

    def test_rerun_finishes_offline_when_the_install_is_complete(self):
        self.assertEqual(self.w.install("--yes").returncode, 0)
        shutil.rmtree(self.w.release)
        self.w.log.write_text("")
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual(self.w.calls("curl"), [])
        self.assertIn("already installed", self.out(r))

    def test_rerun_downloads_again_when_the_engine_was_changed(self):
        self.assertEqual(self.w.install("--yes").returncode, 0)
        (self.w.dir / "build/engine/crucible").write_text("#!/bin/sh\n# rebuilt\n")
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual(sha256(self.w.dir / "build/engine/crucible"), self.w.engine_sha)


class SumsPinTest(InstallTestCase):
    """The SHA256SUMS pin: an empty pin gives a WARN line and the install goes on; a pin that is present
    and does not match stops the install before any file runs. The built-in pin is for PIN_TAG only."""

    def pinned(self, pin_tag, pin):
        text = INSTALL.read_text()
        text = re.sub(r'^PIN_TAG="[^"]*"', f'PIN_TAG="{pin_tag}"', text, count=1, flags=re.M)
        text = re.sub(r'^PIN_SUMS_SHA256="[^"]*"', f'PIN_SUMS_SHA256="{pin}"', text, count=1, flags=re.M)
        path = self.w.t / "pinned-install.sh"
        path.write_text(text)
        return path

    def run_script(self, path, *args, **extra):
        self.w.tty.write_text("")
        return subprocess.run(["bash", str(path), *args], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              env=self.w.env(**extra), start_new_session=True, cwd=self.w.t, timeout=120)

    def nothing_ran(self, r):
        self.assertNotEqual(r.returncode, 0, self.out(r))
        self.assertFalse(self.w.dir.exists(), "no install folder after a pin mismatch")
        self.assertFalse((self.w.home / "engine.ran").exists())

    def warn_lines(self, r):
        return [ln for ln in self.out(r).splitlines() if ln.startswith("WARN") and "SHA256SUMS" in ln]

    def test_an_empty_pin_warns_once_and_installs(self):
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        warn = self.warn_lines(r)
        self.assertEqual(len(warn), 1, self.out(r))
        self.assertIn("replaced", warn[0])

    def test_a_built_in_pin_that_matches_installs_with_no_warning(self):
        r = self.run_script(self.pinned(TAG, sha256(self.w.release / "SHA256SUMS")), "--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual(self.warn_lines(r), [])

    def test_a_built_in_pin_that_does_not_match_stops_before_any_file_runs(self):
        r = self.run_script(self.pinned(TAG, "0" * 64), "--yes")
        self.nothing_ran(r)
        self.assertEqual(r.returncode, 4, self.out(r))
        self.assertIn("pinned", self.out(r))

    def test_the_built_in_pin_is_only_for_its_own_tag(self):
        r = self.run_script(self.pinned("v1.2.3", "0" * 64), "--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual(len(self.warn_lines(r)), 1, self.out(r))

    def test_the_pin_of_the_environment_wins(self):
        r = self.run_script(self.pinned(TAG, "0" * 64), "--yes", CRUCIBLE_SUMS_SHA256=sha256(self.w.release / "SHA256SUMS"))
        self.assertEqual(r.returncode, 0, self.out(r))

    def test_the_dry_run_names_the_pin_state(self):
        r = self.w.install("--dry-run")
        self.assertIn("SHA256SUMS pin: none", self.out(r))
        r = self.run_script(self.pinned(TAG, "a" * 64), "--dry-run")
        self.assertIn("SHA256SUMS pin: " + "a" * 16, self.out(r))

    def test_the_scope_line_is_honest(self):
        text = INSTALL.read_text()
        self.assertIn("makes a replaced release file visible as a commit on main", text)
        self.assertIn("does not protect against a person who can push to main", text)


class BadDownloadTest(InstallTestCase):
    def nothing_ran(self, r):
        self.assertNotEqual(r.returncode, 0, self.out(r))
        self.assertFalse(self.w.dir.exists(), "no install folder after a bad download")
        self.assertFalse((self.w.home / "engine.ran").exists())
        self.assertEqual(self.w.calls("sudo"), [])
        self.assertEqual([c for c in self.w.calls("git") if " clone" in c], [], "no clone before the checks pass")
        self.assertEqual(list(self.w.home.glob("crucible*")), [])

    def test_bad_sha_runs_nothing(self):
        gz = self.w.release / f"{STEM}.gz"
        data = bytearray(gz.read_bytes())
        data[len(data) // 2] ^= 0xFF
        gz.write_bytes(bytes(data))
        r = self.w.install("--yes")
        self.nothing_ran(r)
        self.assertIn("sha256", self.out(r))
        self.assertIn(f"{STEM}.gz", self.out(r))

    def test_partial_file_runs_nothing(self):
        gz = self.w.release / f"{STEM}.gz"
        gz.write_bytes(gz.read_bytes()[:20])
        self.nothing_ran(self.w.install("--yes"))

    def test_missing_asset_runs_nothing(self):
        (self.w.release / f"{STEM}.needs").unlink()
        self.nothing_ran(self.w.install("--yes"))

    def test_missing_sums_runs_nothing(self):
        (self.w.release / "SHA256SUMS").unlink()
        self.nothing_ran(self.w.install("--yes"))

    def test_asset_not_listed_in_sums_runs_nothing(self):
        self.w.write_sums(skip=(f"{STEM}.gz",))
        r = self.w.install("--yes")
        self.nothing_ran(r)
        self.assertIn("not listed", self.out(r))

    def test_manifest_changed_after_sums_runs_nothing(self):
        needs = self.w.release / f"{STEM}.needs"
        needs.write_text(needs.read_text() + "need libevil.so.1\n")
        self.nothing_ran(self.w.install("--yes"))

    def test_sums_pin_mismatch_runs_nothing(self):
        r = self.w.install("--yes", CRUCIBLE_SUMS_SHA256="0" * 64)
        self.nothing_ran(r)
        self.assertIn("SHA256SUMS", self.out(r))

    def test_sums_pin_match_passes(self):
        r = self.w.install("--yes", CRUCIBLE_SUMS_SHA256=sha256(self.w.release / "SHA256SUMS"))
        self.assertEqual(r.returncode, 0, self.out(r))

    def test_manifest_for_another_tag_runs_nothing(self):
        needs = self.w.release / f"{STEM}.needs"
        needs.write_text(needs.read_text().replace(f"tag {TAG}", "tag v0.0.1"))
        self.w.write_sums()
        self.nothing_ran(self.w.install("--yes"))

    def test_engine_hash_in_manifest_must_match_the_unpacked_file(self):
        needs = self.w.release / f"{STEM}.needs"
        needs.write_text(re.sub(r"binary_sha256 \w+", "binary_sha256 " + "1" * 64, needs.read_text()))
        self.w.write_sums()
        self.nothing_ran(self.w.install("--yes"))

    def test_source_commit_must_match_the_manifest(self):
        self.w.make_release(commit="2" * 40)
        r = self.w.install("--yes")
        self.assertNotEqual(r.returncode, 0, self.out(r))
        self.assertIn("commit", self.out(r))
        self.assertFalse(self.w.dir.exists())
        self.assertFalse((self.w.home / "engine.ran").exists())

    def test_clone_failure_leaves_no_folder(self):
        r = self.w.install("--yes", CRUCIBLE_REPO_URL=f"file://{self.w.t}/nope")
        self.assertNotEqual(r.returncode, 0, self.out(r))
        self.assertFalse(self.w.dir.exists())

    def test_release_url_must_be_https_or_file(self):
        r = self.w.install("--yes", CRUCIBLE_RELEASE_BASE="http://example.invalid/x")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.w.calls("curl"), [])


class TruncatedScriptTest(InstallTestCase):
    """A download that stops anywhere must run nothing: the whole script is functions and one last call."""

    def test_every_cut_runs_nothing(self):
        text = INSTALL.read_text()
        cuts = sorted(set(list(range(0, len(text), max(1, len(text) // 150))) + [len(text) - 2, len(text) - 3, len(text) - 12]))
        shim_dir = self.w.t / "cutshims"
        shim_dir.mkdir()
        for name in ("curl", "git", "sudo", "rm", "mkdir", "mv", "cp", "tar", "gzip", "python3", "touch", "tee", "ln", "chmod", "sha256sum"):
            p = shim_dir / name
            p.write_text(f'#!/bin/sh\nprintf "%s\\n" "{name} $*" >> "{self.w.log}"\nexit 99\n')
            p.chmod(0o755)
        env = self.w.env()
        env["PATH"] = f"{shim_dir}:{os.environ['PATH']}"
        before = sorted(p.name for p in self.w.home.rglob("*"))
        for n in cuts:
            r = subprocess.run(["bash", "-s", "--", "--yes"], input=text[:n], capture_output=True, text=True, env=env,
                               start_new_session=True, cwd=self.w.t, timeout=60)
            self.assertNotIn("[1/", r.stdout + r.stderr, f"cut at {n}")
            self.assertEqual(self.w.calls(), [], f"cut at {n}: a tool ran")
        self.assertEqual(before, sorted(p.name for p in self.w.home.rglob("*")))

    def test_all_but_the_last_line_runs_nothing(self):
        lines = INSTALL.read_text().splitlines()
        self.assertTrue(lines[-1].startswith("if ") and lines[-1].rstrip().endswith("fi") and 'main "$@"' in lines[-1])
        body = self.w.t / "body.sh"
        body.write_text("\n".join(lines[:-1]) + "\n")
        r = run(["bash", str(body), "--yes"], env=self.w.env(), cwd=self.w.t, stdin=subprocess.DEVNULL)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "", ""))
        self.assertEqual(self.w.calls(), [])


class PackageTest(InstallTestCase):
    def need_missing(self, lib="libnuma.so.1", os_id="ubuntu"):
        self.w.make_release(needs=[lib])
        self.w.ld_hide.write_text(lib + "\n")
        self.w.os_release.write_text(f"ID={os_id}\n")

    def test_missing_library_shows_the_exact_command_and_asks_once(self):
        self.need_missing()
        r = self.w.install("--yes", answers="y\n", SHIM_INSTALL="libnuma.so.1")
        self.assertEqual(r.returncode, 0, self.out(r))
        text = self.out(r)
        self.assertIn("sudo apt-get install -y libnuma1", text)
        self.assertEqual(text.count("(y/N)"), 1)
        self.assertEqual(self.w.calls("sudo"), ["sudo apt-get update", "sudo apt-get install -y libnuma1"])
        self.assertTrue((self.w.dir / "build/engine/release.json").is_file())

    def test_command_text_per_system(self):
        want = {"arch": "sudo pacman -S --needed --noconfirm numactl", "fedora": "sudo dnf install -y numactl-libs", "debian": "sudo apt-get install -y libnuma1"}
        for os_id, cmd in want.items():
            with self.subTest(os_id):
                w = Env(self)
                w.make_release(needs=["libnuma.so.1"])
                w.ld_hide.write_text("libnuma.so.1\n")
                w.os_release.write_text(f"ID={os_id}\n")
                r = w.install("--yes", answers="n\n")
                self.assertIn(cmd, self.out(r))
                self.assertEqual(w.calls("sudo"), [])

    def test_shown_command_is_the_run_command_and_never_asks_again(self):
        # stdin of the run is closed, so a second question inside the package tool would have no defined answer
        flag = {"arch": "--noconfirm", "fedora": "-y", "debian": "-y"}
        for os_id, noninteractive in flag.items():
            with self.subTest(os_id):
                w = Env(self)
                w.make_release(needs=["libnuma.so.1"])
                w.ld_hide.write_text("libnuma.so.1\n")
                w.os_release.write_text(f"ID={os_id}\n")
                r = w.install("--yes", answers="y\n", SHIM_INSTALL="libnuma.so.1")
                self.assertEqual(r.returncode, 0, self.out(r))
                shown = [ln.strip() for ln in self.out(r).splitlines() if ln.startswith("  sudo ")]
                self.assertEqual(shown, w.calls("sudo"))
                self.assertIn(noninteractive, w.calls("sudo")[-1].split())

    def test_answer_no_stops_before_sudo_and_changes_nothing(self):
        self.need_missing()
        r = self.w.install("--yes", answers="n\n")
        self.assertEqual(r.returncode, 3, self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])
        self.assertFalse(self.w.dir.exists())

    def test_empty_answer_is_no(self):
        self.need_missing()
        r = self.w.install("--yes", answers="\n")
        self.assertEqual(r.returncode, 3, self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])

    def test_no_terminal_never_calls_sudo(self):
        self.need_missing()
        r = self.w.install("--yes", CRUCIBLE_TTY=str(self.w.t / "no-such-tty"))
        self.assertEqual(r.returncode, 3, self.out(r))
        self.assertIn("sudo apt-get install -y libnuma1", self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])

    def test_yes_flag_does_not_answer_the_sudo_question(self):
        self.need_missing()
        r = self.w.install("--yes", CRUCIBLE_TTY=str(self.w.t / "no-such-tty"))
        self.assertEqual(self.w.calls("sudo"), [])
        self.assertEqual(r.returncode, 3)

    def test_unknown_library_has_no_package_and_stops(self):
        self.need_missing(lib="libfmt.so.99")
        r = self.w.install("--yes", answers="y\n")
        self.assertEqual(r.returncode, 3, self.out(r))
        self.assertIn("libfmt.so.99", self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])

    def test_library_still_missing_after_the_install_stops(self):
        self.need_missing()
        r = self.w.install("--yes", answers="y\n")  # the sudo shim does not add the library
        self.assertEqual(r.returncode, 3, self.out(r))
        self.assertEqual(len(self.w.calls("sudo")), 2)

    def test_libraries_present_means_no_question(self):
        self.w.make_release(needs=["libm.so.6"])
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertNotIn("(y/N)", self.out(r))

    def test_only_one_sudo_call_site_and_the_printed_command_is_the_run_command(self):
        text = INSTALL.read_text()
        code = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
        runs = [ln for ln in code if "${sudo_cmd[@]}" in ln and "say " not in ln]
        self.assertEqual(len(runs), 2, runs)  # the package list update and the package install, in one function
        self.assertEqual([ln.strip() for ln in code if "sudo_cmd=(sudo)" in ln], ["sudo_cmd=(sudo)"])


class TargetRocmTest(InstallTestCase):
    """The ROCm of the target decides what its libraries need. The real ldd of the engine, after the unpack, is the check.

    Found on a real Ubuntu 24.04 with the ROCm of AMD (Nov 6): the manifest named the libraries that the ROCm of the BUILD host
    needed (libfmt.so.12, libglog.so.2), and install.sh stopped with exit 3 on a system whose ROCm needs neither."""

    def setUp(self):
        super().setUp()
        self.rocm = self.w.t / "rocm"
        (self.rocm / "lib").mkdir(parents=True)
        self.cc_dir = self.w.t / "cc"
        self.cc_dir.mkdir()

    def elf(self, soname=None):
        """A real engine file. With a soname, it needs that library, which exists nowhere (ldd says: not found)."""
        if not shutil.which("cc"):
            self.skipTest("needs a C compiler")
        main = self.cc_dir / "main.c"
        out = self.cc_dir / "engine"
        if soname is None:
            main.write_text("int main(void){return 0;}\n")
            self.assertEqual(run(["cc", str(main), "-o", str(out)]).returncode, 0)
        else:
            lib = self.cc_dir / "libx.c"
            lib.write_text("int xf(void){return 1;}\n")
            so = self.cc_dir / soname
            self.assertEqual(run(["cc", "-shared", "-fPIC", f"-Wl,-soname,{soname}", str(lib), "-o", str(so)]).returncode, 0)
            main.write_text("int xf(void); int main(void){return xf()>9;}\n")
            self.assertEqual(run(["cc", str(main), "-L", str(self.cc_dir), f"-l:{soname}", "-Wl,--no-as-needed", "-o", str(out)]).returncode, 0)
            so.unlink()
        return out.read_bytes()

    def release(self, engine, needs=(), extra=()):
        self.w.make_release(engine=engine, needs=needs)
        if extra:
            f = self.w.release / f"{STEM}.needs"
            f.write_text(f.read_text() + "".join(f"{ln}\n" for ln in extra))
            self.w.write_sums()

    def test_a_library_that_only_the_build_host_rocm_needed_does_not_stop_the_install(self):
        self.release(self.elf(), extra=["rocm_external libfmt.so.12", "rocm_external libglog.so.2"])
        self.w.ld_hide.write_text("libfmt.so.12\nlibglog.so.2\n")
        self.w.os_release.write_text("ID=ubuntu\n")
        r = self.w.install("--yes", ROCM_PATH=str(self.rocm))
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])
        self.assertNotIn("(y/N)", self.out(r))

    def test_the_real_ldd_decides_after_the_unpack_not_the_manifest(self):
        # an older manifest that lists libfmt.so.12 as "need": the engine on this machine does not need it
        self.release(self.elf(), needs=["libfmt.so.12"])
        self.w.ld_hide.write_text("libfmt.so.12\n")
        self.w.os_release.write_text("ID=ubuntu\n")
        r = self.w.install("--yes", ROCM_PATH=str(self.rocm))
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])

    def test_without_rocm_on_the_machine_the_manifest_list_is_still_checked(self):
        self.release(self.elf(), needs=["libfmt.so.12"])
        self.w.ld_hide.write_text("libfmt.so.12\n")
        self.w.os_release.write_text("ID=ubuntu\n")
        r = self.w.install("--yes", ROCM_PATH=str(self.w.t / "no-rocm"), answers="n\n")
        self.assertEqual(r.returncode, 3, self.out(r))
        self.assertIn("libfmt.so.12", self.out(r))

    def test_a_library_that_ldd_misses_and_that_has_a_package_gets_the_package_message(self):
        if "libfmt.so.12" in run(["ldconfig", "-p"]).stdout:
            self.skipTest("libfmt.so.12 is installed here")
        self.release(self.elf("libfmt.so.12"))
        self.w.os_release.write_text("ID=arch\n")
        r = self.w.install("--yes", ROCM_PATH=str(self.rocm), answers="n\n")
        self.assertEqual(r.returncode, 3, self.out(r))
        self.assertIn("Missing: libfmt.so.12", self.out(r))
        self.assertIn("sudo pacman -S --needed --noconfirm fmt", self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])

    def test_a_library_that_ldd_misses_and_that_has_no_package_stops_with_the_name(self):
        self.release(self.elf("libfmt.so.99"))
        r = self.w.install("--yes", ROCM_PATH=str(self.rocm))
        self.assertEqual(r.returncode, 3, self.out(r))
        self.assertIn("No package is known on this system for: libfmt.so.99", self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])

    def test_a_rocm_library_that_ldd_misses_is_not_a_package(self):
        self.release(self.elf("libamdhip64.so.7"))
        r = self.w.install("--yes", ROCM_PATH=str(self.rocm))
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])

    # The HIP runtime note: the latest ROCm and 7.2.4 give no warning and no request for an older version.
    def note(self, version=None, lib=None, rocm=True):
        self.release(self.elf())
        if version is not None:
            (self.rocm / ".info").mkdir()
            (self.rocm / ".info/version").write_text(version + "\n")
        if lib is not None:
            (self.rocm / "lib" / lib).write_text("")
        path = self.rocm if rocm else self.w.t / "no-rocm"
        r = self.w.install("--yes", ROCM_PATH=str(path))
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual(self.w.calls("sudo"), [])
        return self.out(r)

    def test_rocm_10_1_is_current_and_gives_no_warning(self):
        text = self.note("10.1.0")
        self.assertIn("HIP runtime: ROCm 10.1.0 (current)", text)
        self.assertNotIn("only 7.2", text.lower())
        self.assertNotIn("older version", text.lower())

    def test_a_later_major_version_is_current(self):
        self.assertIn("HIP runtime: ROCm 11.0.2 (current)", self.note("11.0.2"))

    def test_rocm_7_2_4_is_tested_and_gives_no_warning(self):
        text = self.note("7.2.4")
        self.assertIn("HIP runtime: ROCm 7.2.4 (tested)", text)
        # the one WARN line of an empty SHA256SUMS pin (trust root Part A) is not about ROCm
        self.assertNotIn("warn", "\n".join(ln for ln in text.lower().splitlines() if "sha256sums" not in ln))

    def test_the_library_name_names_the_version_when_there_is_no_version_file(self):
        self.assertIn("HIP runtime: ROCm 10.1 (current)", self.note(lib="libamdhip64.so.7.16.26385"))

    def test_another_version_is_named_with_the_tested_ones(self):
        text = self.note("7.1.0")
        self.assertIn("HIP runtime: ROCm 7.1.0", text)
        self.assertIn("Tested: 10.1 and 7.2.4", text)

    def test_no_runtime_says_what_setup_can_do_and_does_not_stop(self):
        text = self.note(rocm=False)
        self.assertIn("HIP runtime: none found", text)
        self.assertIn("current ROCm (10.1 tested)", text)

    def test_glog_package_names_match_the_soname_of_the_distribution(self):
        text = INSTALL.read_text()
        func = re.search(r"^pkg_for\(\) \{\n.*?^\}\n", text, re.S | re.M).group(0)

        def pkg(family, name):
            return run(["bash", "-c", func + '\npkg_for "$1" "$2"', "x", family, name]).stdout.strip()
        # Ubuntu 24.04 (glog 0.6) ships libglog.so.1 in libgoogle-glog0v6t64 and has no package for libglog.so.2
        self.assertEqual(pkg("debian", "libglog.so.1"), "libgoogle-glog0v6t64")
        self.assertEqual(pkg("debian", "libglog.so.2"), "")
        self.assertEqual(pkg("arch", "libglog.so.2"), "google-glog")


class FallbackTest(InstallTestCase):
    def test_old_glibc_falls_back_to_the_source_build(self):
        self.w.make_release(glibc_min="99.0")
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertIn("build the engine from source", self.out(r))
        self.assertFalse((self.w.dir / "build/engine/release.json").exists())
        self.assertFalse((self.w.dir / "build/engine/crucible").exists())
        self.assertEqual((self.w.dir / "handover.args").read_text().strip(), "setup --yes")

    def test_old_libstdcxx_falls_back(self):
        self.w.make_release(glibcxx_min="3.4.999")
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertFalse((self.w.dir / "build/engine/release.json").exists())

    def test_source_build_flag_skips_the_engine_download(self):
        r = self.w.install("--yes", "--source-build")
        self.assertEqual(r.returncode, 0, self.out(r))
        # the manifest and the list are fetched: they pin the source to the commit of the release
        self.assertEqual(len(self.w.calls("curl")), 2, self.w.calls("curl"))
        self.assertFalse([c for c in self.w.calls("curl") if c.endswith(".gz")])
        self.assertFalse((self.w.dir / "build/engine").exists())
        self.assertTrue((self.w.dir / "handover.args").exists())

    def test_download_failure_does_not_fall_back_by_itself_and_names_the_flag(self):
        for p in self.w.release.iterdir():
            p.unlink()
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 4, self.out(r))
        self.assertIn("--source-build", self.out(r))
        self.assertFalse(self.w.dir.exists())


class MachineCheckTest(InstallTestCase):
    """Item 3: the dry run and step 1 run the read-only checks, and a bad setup option stops before any download."""

    def no_download(self, r):
        self.assertEqual(self.w.calls("curl"), [], self.out(r))
        self.assertEqual([c for c in self.w.calls("git") if " clone" in c], [])
        self.assertFalse(self.w.dir.exists())

    def test_dry_run_on_a_good_machine_says_it_can_run(self):
        r = self.w.install("--dry-run")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertIn("This machine can run Crucible", self.out(r))
        for word in ("gfx1151", "memory", "disk"):
            self.assertIn(word, self.out(r))
        self.no_download(r)

    def test_dry_run_on_a_small_machine_says_plainly_that_it_cannot_run(self):
        self.w.make_machine(mem_kb=40_000_000)
        r = self.w.install("--dry-run")
        self.assertEqual(r.returncode, 6, self.out(r))
        self.assertIn("cannot run Crucible", self.out(r))
        self.assertIn("memory", self.out(r))
        self.no_download(r)

    def test_dry_run_without_the_gpu_or_with_another_gpu_cannot_run(self):
        for machine in (dict(kfd=False), dict(gfx_version="110000")):
            with self.subTest(machine=machine):
                self.w.make_machine(**machine)
                r = self.w.install("--dry-run")
                self.assertEqual(r.returncode, 6, self.out(r))
                self.assertIn("gfx1151", self.out(r))

    def test_a_96_gb_machine_can_run_the_3_05_pack(self):
        self.w.make_machine(mem_kb=97_602_320, gtt=94208 * 2 ** 20)
        r = self.w.install("--dry-run")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertIn("a 96 GB machine: the 3.05 pack with 2 sessions", self.out(r))
        self.assertIn("This machine can run Crucible", self.out(r))
        self.no_download(r)

    def test_the_96_gb_line_names_the_start_time_fit(self):
        # Review L2: the line does not promise 2 sessions at the bottom of the class.
        self.w.make_machine(mem_kb=92_400_000, gtt=94208 * 2 ** 20)
        r = self.w.install("--dry-run")
        self.assertIn("a smaller conversation cache or fewer sessions", self.out(r))
        self.assertNotIn("UMA frame buffer", self.out(r))

    def test_the_96_gb_line_gives_the_uma_advice_when_the_vram_is_large(self):
        # Review L1: a 128 GB machine with a 16 GiB UMA frame buffer looks like a 96 GB machine.
        self.w.make_machine(mem_kb=114_000_000, gtt=94208 * 2 ** 20)
        (self.w.sys / "class/drm/card0/device/mem_info_vram_total").write_text(f"{16 * 2 ** 30}\n")
        r = self.w.install("--dry-run")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertIn("16 GiB of dedicated memory (UMA frame buffer)", self.out(r))
        self.assertIn("512 MB", self.out(r))

    def test_a_96_gb_machine_refuses_the_4_05_pack(self):
        self.w.make_machine(mem_kb=97_602_320, gtt=94208 * 2 ** 20)
        r = self.w.install("--dry-run", "--pack", "4.05")
        self.assertEqual(r.returncode, 6, self.out(r))
        self.assertIn("--pack 3.05", self.out(r))

    def tree_setup_script(self):
        """install.sh with the option table of this tree's setup, as the release record writes it for a pin that has
        the 4.18 pack. A pin before v0.38.0 has no 4.18 choice, so its option check stops a 4.18 install first."""
        sys.path.insert(0, str(ROOT / "tools"))
        import setup_options
        script = self.w.t / "install-tree-options.sh"
        table = setup_options.text(setup_options.table())
        script.write_text(setup_options.BLOCK_RE.sub(lambda m: f'SETUP_OPTIONS="\n{table}"\n', INSTALL.read_text(),
                                                     count=1))
        self.assertIn("4.18", setup_options.install_table(script.read_text())["--pack"])
        return script

    def test_a_96_gb_machine_refuses_the_4_18_pack(self):
        self.w.make_machine(mem_kb=97_602_320, gtt=94208 * 2 ** 20)
        r = self.w.install("--dry-run", "--pack", "4.18", script=self.tree_setup_script())
        self.assertEqual(r.returncode, 6, self.out(r))
        self.assertIn("the 4.18 pack needs a 128 GB machine", self.out(r))
        self.assertIn("--pack 3.05", self.out(r))

    def test_a_96_gb_machine_names_its_own_gtt_value(self):
        self.w.make_machine(mem_kb=97_602_320, gtt=48 * 2 ** 30)
        r = self.w.install("--dry-run")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertIn("94208 MiB on this 96 GB machine", self.out(r))
        self.assertIn("kernel parameters", self.out(r))

    def test_a_64_gb_machine_cannot_run(self):
        self.w.make_machine(mem_kb=64_047_888, gtt=60 * 2 ** 30)
        r = self.w.install("--dry-run")
        self.assertEqual(r.returncode, 6, self.out(r))
        self.assertIn("96 GB or 128 GB machine", self.out(r))
        self.no_download(r)

    def test_a_fixable_problem_is_a_fix_line_not_a_stop(self):
        self.w.make_machine(gtt=60_000_000_000)
        r = self.w.install("--dry-run")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertIn("after the fixes", self.out(r))
        self.assertIn("kernel parameters", self.out(r))

    def test_small_disk_for_the_pack_is_named_with_the_size(self):
        self.w.df_avail.write_text(str(50 * 1024 ** 2))
        r = self.w.install("--dry-run", "--pack", "4.05")
        self.assertIn("110 GB", self.out(r))
        self.assertIn("--models-dir", self.out(r))

    def test_the_418_pack_needs_its_own_disk_size(self):
        # 115 GB free: enough for 4.05 (110 GB), not for 4.18 (125 GB).
        self.w.df_avail.write_text(str(115_000_000_000 // 1024))
        r = self.w.install("--dry-run", "--pack", "4.18", script=self.tree_setup_script())
        self.assertIn("the 4.18 pack needs about 125 GB", self.out(r))

    def test_the_real_install_stops_before_any_download_when_the_machine_cannot_run(self):
        self.w.make_machine(mem_kb=40_000_000)
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 6, self.out(r))
        self.no_download(r)

    def test_bad_setup_options_stop_before_any_download(self):
        for args, word in ((("--pack", "9.99"), "--pack"), (("--streams", "9"), "--streams"), (("--port", "x"), "--port"),
                           (("--no-such-flag",), "--no-such-flag"), (("--service",), "--service"),
                           (("--service=maybe",), "--service"), (("--smoke=1",), "--smoke")):
            for mode in ((), ("--dry-run",)):
                with self.subTest(args=args, mode=mode):
                    r = self.w.install(*mode, *args)
                    self.assertEqual(r.returncode, 2, self.out(r))
                    self.assertIn(word, self.out(r))
                    self.no_download(r)

    def test_good_setup_options_pass_in_both_forms(self):
        r = self.w.install("--dry-run", "--pack=3.05", "--streams", "2", "--service", "no", "--state-dir", "x", "--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertIn("setup --pack=3.05 --streams 2 --service no --state-dir x --yes", self.out(r))

    def test_the_sizes_are_the_ones_of_setup(self):
        sys.path.insert(0, str(ROOT / "tools"))
        from crucible_cli import doctor, packs
        text = INSTALL.read_text()
        self.assertIn(f"PACK_DISK_GB_405={packs.PACKS['4.05'].disk_gb}\n", text)
        self.assertIn(f"PACK_DISK_GB_305={packs.PACKS['3.05'].disk_gb}\n", text)
        self.assertIn(f"PACK_DISK_GB_418={packs.PACKS['4.18'].disk_gb}\n", text)
        self.assertIn(f"MEM_MIN_KB={doctor.MEM_MIN_KB}\n", text)
        self.assertIn(f"GTT_MIN_BYTES={doctor.GTT_MIN_BYTES}\n", text)
        from crucible_cli import machine
        self.assertIn(f"MEM_MID_KB={machine.MID_MIN_KB}\n", text)

    def test_the_option_table_is_the_setup_parser_of_the_pinned_release(self):
        # install.sh installs PIN_TAG, so it checks the options of that setup, not the parser of this tree (an option
        # that the pinned setup lacks would pass the check and stop the install after the downloads).
        sys.path.insert(0, str(ROOT / "tools"))
        import setup_options
        text = INSTALL.read_text()
        record = ROOT / "tests/data/setup_options" / f"{setup_options.pin_tag(text)}.txt"
        self.assertTrue(record.is_file(), "no option table of the pinned release: run tools/setup_options.py record")
        self.assertEqual(setup_options.install_table(text), setup_options.parse(record.read_text()))

    def test_an_option_that_the_pinned_setup_lacks_stops_before_any_download(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import setup_options
        # A fixed older pin: install.sh with the option table of v0.37.3. After a release record, the table of the
        # real pin can equal this tree, and the loop would check nothing.
        old = self.w.t / "install-v0.37.3.sh"
        older = (ROOT / "tests/data/setup_options/v0.37.3.txt").read_text()
        old.write_text(setup_options.BLOCK_RE.sub(lambda m: f'SETUP_OPTIONS="\n{older}"\n', INSTALL.read_text(), count=1))
        pinned = setup_options.install_table(old.read_text())
        options = setup_options.table()
        newer = [name for name in options if name not in pinned]
        self.assertGreaterEqual(len(newer), 2, "the v0.37.3 table must lack options of this setup")
        for name in newer[:2]:
            kind = options[name]
            args = [name] if kind == "flag" else [name, kind.split(":", 1)[1].split(",")[0].split("-")[0]] \
                if ":" in kind else [name, "1"]
            with self.subTest(option=name):
                r = self.w.install("--dry-run", "--yes", *args, script=old)
                self.assertEqual(r.returncode, 2, self.out(r))
                self.assertIn(f"{name} is not an option of ", self.out(r))
                self.assertIn("Nothing was downloaded.", self.out(r))
                self.assertEqual([x for x in self.w.calls("curl")], [])

class InCloneTest(InstallTestCase):
    """Item 7c: install.sh run from inside a clone uses that clone. It never clones again."""

    def clone(self, later=False):
        c = self.w.home / "src/crucible-strix-halo"
        c.parent.mkdir(parents=True)
        run([REAL_GIT, "-c", "advice.detachedHead=false", "clone", "-q", "--branch", TAG, self.w.repo_url, str(c)],
            env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"})
        if later:
            g = [REAL_GIT, "-C", str(c), "-c", "user.email=test-user", "-c", "user.name=t"]
            run([*g, "commit", "-q", "--allow-empty", "-m", "later"])
        shutil.copy(INSTALL, c / "install.sh")
        return c

    def run_in(self, c, *args):
        return subprocess.run(["bash", str(c / "install.sh"), *args], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, env=self.w.env(CRUCIBLE_DIR=""), start_new_session=True, cwd=c, timeout=120)

    def test_a_clone_at_the_tag_is_used_in_place(self):
        c = self.clone()
        r = self.run_in(c, "--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual([x for x in self.w.calls("git") if " clone" in x], [])
        self.assertFalse((self.w.home / "crucible").exists())
        self.assertEqual((c / "handover.args").read_text().strip(), "setup --yes")
        self.assertEqual(sha256(c / "build/engine/crucible"), self.w.engine_sha)
        self.assertFalse((c / ".git/crucible-install").exists(), "uninstall must never remove a clone of the user")
        self.assertIn(str(c), self.out(r))

    def test_a_clone_with_source_build_downloads_no_engine(self):
        c = self.clone()
        r = self.run_in(c, "--yes", "--source-build")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertFalse([x for x in self.w.calls("curl") if x.endswith(".gz")])
        self.assertFalse((c / "build/engine/crucible").exists())
        self.assertTrue((c / "handover.args").exists())

    def test_a_clone_that_is_not_at_the_tag_builds_from_its_source_with_no_download(self):
        c = self.clone(later=True)
        r = self.run_in(c, "--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual(self.w.calls("curl"), [])
        self.assertEqual([x for x in self.w.calls("git") if " clone" in x], [])
        self.assertFalse((c / "build/engine/crucible").exists())
        self.assertTrue((c / "handover.args").exists())
        self.assertIn("from this source", self.out(r))

    def test_the_dry_run_in_a_clone_names_the_clone(self):
        c = self.clone()
        r = self.run_in(c, "--dry-run")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertIn(f"source:         this clone ({c})", self.out(r))

    def test_dir_still_wins_over_the_clone(self):
        c = self.clone()
        other = self.w.home / "other"
        r = subprocess.run(["bash", str(c / "install.sh"), "--yes", "--dir", str(other)], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, env=self.w.env(CRUCIBLE_DIR=""), start_new_session=True,
                           cwd=c, timeout=120)
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertTrue((other / "handover.args").exists())

    def test_a_pipe_run_inside_a_clone_folder_installs_as_before(self):
        c = self.clone()
        self.w.tty.write_text("")
        r = subprocess.run(["bash", "-s", "--", "--yes"], input=INSTALL.read_text(), capture_output=True, text=True,
                           env=self.w.env(CRUCIBLE_DIR=""), start_new_session=True, cwd=c, timeout=120)
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertTrue((self.w.home / "crucible/handover.args").exists())
        self.assertFalse((c / "handover.args").exists())


class FolderTest(InstallTestCase):
    def test_folder_with_other_files_is_refused(self):
        self.w.dir.mkdir()
        (self.w.dir / "mine.txt").write_text("keep")
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 2, self.out(r))
        self.assertEqual((self.w.dir / "mine.txt").read_text(), "keep")
        self.assertEqual(self.w.calls("curl"), [])

    def test_empty_folder_is_used(self):
        self.w.dir.mkdir()
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 0, self.out(r))

    def test_other_tag_in_the_folder_points_to_update(self):
        self.assertEqual(self.w.install("--yes").returncode, 0)
        git = [REAL_GIT, "-C", str(self.w.dir), "-c", "user.email=test-user", "-c", "user.name=t"]
        run([*git, "-c", "advice.detachedHead=false", "commit", "-q", "--allow-empty", "-m", "later"])
        run([*git, "tag", "v8.0.0"])
        r = self.w.install("--yes")
        self.assertEqual(r.returncode, 2, self.out(r))
        self.assertIn("update", self.out(r))

    def test_dir_must_not_be_home_or_root(self):
        for d in (str(self.w.home), "/", "/usr"):
            r = self.w.install("--yes", "--dir", d)
            self.assertEqual(r.returncode, 2, d + self.out(r))


class EntryPointsTest(InstallTestCase):
    def installed(self):
        self.assertEqual(self.w.install("--yes").returncode, 0)

    def test_dry_run_changes_nothing_and_uses_no_network(self):
        r = self.w.install("--dry-run")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertFalse(self.w.dir.exists())
        self.assertEqual(self.w.calls("curl"), [])
        self.assertIn(TAG, self.out(r))
        self.assertIn(str(self.w.dir), self.out(r))

    def test_help(self):
        r = self.w.install("--help")
        self.assertEqual(r.returncode, 0)
        for word in ("status", "update", "uninstall", "--dir", "--source-build", "--dry-run"):
            self.assertIn(word, r.stdout)
        self.assertNotIn("\u2014", r.stdout)

    def test_status_before_and_after(self):
        r = self.w.install("status")
        self.assertEqual(r.returncode, 1, self.out(r))
        self.assertIn("not installed", self.out(r))
        self.installed()
        r = self.w.install("status")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertIn(TAG, r.stdout)
        self.assertIn("release engine", r.stdout)
        self.assertIn("install.log", r.stdout)

    def test_status_notices_a_rebuilt_engine(self):
        self.installed()
        (self.w.dir / "build/engine/crucible").write_text("#!/bin/sh\n# own build\n")
        r = self.w.install("status")
        self.assertIn("source build", r.stdout)

    def test_update_uses_update_sh_with_the_release_make(self):
        self.installed()
        r = self.w.install("update")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual((self.w.dir / "update.args").read_text().strip(), "update.sh  make=tools/release_make.sh")

    def test_update_prefers_crucible_update_when_it_exists(self):
        self.installed()
        (self.w.dir / "tools/crucible_cli").mkdir(parents=True)
        (self.w.dir / "tools/crucible_cli/update.py").write_text("")
        r = self.w.install("update", "--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertEqual((self.w.dir / "handover.args").read_text().strip(), "update --yes")

    def test_update_passes_rollback_on_to_crucible_update_when_it_exists(self):
        # ./crucible update answers --rollback itself (exit 2, one message); install.sh must not swallow it
        self.installed()
        (self.w.dir / "tools/crucible_cli").mkdir(parents=True)
        (self.w.dir / "tools/crucible_cli/update.py").write_text("")
        self.w.install("update", "--rollback")
        self.assertEqual((self.w.dir / "handover.args").read_text().strip(), "update --rollback")

    def test_update_with_a_tag_on_an_old_install_refuses_with_a_message(self):
        """r373 secrev L3: an install with no update.py hands over to tools/update.sh, which has no --tag. The tag must not
        be dropped with no message (the update would go to the newest release, not the asked one)."""
        self.installed()
        for argv in (("update", "--tag", "v0.37.2"), ("update", "--version=v0.37.2")):
            with self.subTest(argv=argv):
                args = self.w.dir / "update.args"
                if args.exists():
                    args.unlink()
                r = self.w.install(*argv)
                self.assertEqual(r.returncode, 2, self.out(r))
                self.assertIn("v0.37.2", self.out(r))
                self.assertIn("--tag", self.out(r))
                self.assertFalse(args.exists(), "tools/update.sh ran and dropped the tag")

    def test_update_without_install_says_so(self):
        r = self.w.install("update")
        self.assertEqual(r.returncode, 1)
        self.assertIn("not installed", self.out(r))

    def test_uninstall_removes_the_folder_and_keeps_the_pack(self):
        self.installed()
        models = self.w.home / "models-keep"
        (models / "pack").mkdir(parents=True)
        (models / "pack/weights.bin").write_text("x")
        cfg = self.w.dir / ".crucible"
        cfg.mkdir()
        (cfg / "config.json").write_text(json.dumps({"models_dir": str(models), "state_dir": None}))
        key = self.w.home / ".config/crucible/api-key"
        key.parent.mkdir(parents=True)
        key.write_text("crucible-secret\n")
        r = self.w.install("uninstall", "--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertFalse(self.w.dir.exists())
        self.assertTrue((models / "pack/weights.bin").exists())
        self.assertTrue(key.exists())
        self.assertIn(str(models), self.out(r))

    def test_uninstall_can_remove_the_pack(self):
        # --remove-models removes the pack folders that setup makes, not the whole models folder (finding 1 of the review)
        self.installed()
        models = self.w.home / "models-del"
        (models / PACK_FOLDER).mkdir(parents=True)
        (models / "other.gguf").write_text("mine")
        (self.w.dir / ".crucible").mkdir()
        (self.w.dir / ".crucible/config.json").write_text(json.dumps({"models_dir": str(models)}))
        r = self.w.install("uninstall", "--yes", "--remove-models")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertFalse((models / PACK_FOLDER).exists())
        self.assertTrue((models / "other.gguf").exists())

    def test_uninstall_asks_once_and_no_keeps_everything(self):
        self.installed()
        r = self.w.install("uninstall", answers="n\n")
        self.assertEqual(r.returncode, 0, self.out(r))
        self.assertTrue(self.w.dir.exists())
        self.assertEqual(self.out(r).count("(y/N)"), 1)

    def test_uninstall_refuses_a_folder_that_is_not_an_install(self):
        d = self.w.home / "docs"
        d.mkdir()
        (d / "a.txt").write_text("keep")
        r = self.w.install("uninstall", "--yes", "--dir", str(d))
        self.assertNotEqual(r.returncode, 0, self.out(r))
        self.assertTrue((d / "a.txt").exists())

    def test_uninstall_refuses_a_models_dir_outside_home(self):
        self.installed()
        (self.w.dir / ".crucible").mkdir()
        (self.w.dir / ".crucible/config.json").write_text(json.dumps({"models_dir": "/usr/share"}))
        r = self.w.install("uninstall", "--yes", "--remove-models")
        self.assertTrue(Path("/usr/share").exists())
        self.assertIn("/usr/share", self.out(r))

    def test_uninstall_stops_the_service_through_crucible_when_it_can(self):
        self.installed()
        (self.w.dir / "crucible").write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$HOME/crucible.calls"\n')
        run(["chmod", "+x", str(self.w.dir / "crucible")])
        r = self.w.install("uninstall", "--yes")
        self.assertEqual(r.returncode, 0, self.out(r))
        calls = (self.w.home / "crucible.calls").read_text().split("\n")
        self.assertIn("service uninstall", calls)
        self.assertIn("stop", calls)


class StaticTest(unittest.TestCase):
    text = INSTALL.read_text() if INSTALL.exists() else ""

    def test_script_exists_and_is_executable(self):
        self.assertTrue(INSTALL.is_file())
        self.assertTrue(INSTALL.stat().st_mode & stat.S_IXUSR)
        self.assertTrue(self.text.startswith("#!/usr/bin/env bash\n"))

    def test_never_touches_boot_settings(self):
        banned = r"efibootmgr|grub-mkconfig|update-grub|bootctl|mkinitcpio|dracut|sbctl|kernel-install|/boot/|/etc/default/grub|cmdline|amdgpu\.gttsize|loader\.conf|tpm2"
        code = [ln for ln in self.text.splitlines() if not ln.strip().startswith("#")]
        self.assertEqual([ln for ln in code if re.search(banned, ln)], [])

    def test_no_system_path_is_written(self):
        code = "\n".join(ln for ln in self.text.splitlines() if not ln.strip().startswith("#"))
        self.assertIsNone(re.search(r">\s*/(etc|usr|boot|opt|var|lib)/", code))

    def test_pinned_tag_is_a_release_tag(self):
        m = re.search(r'^PIN_TAG="([^"]*)"', self.text, re.M)
        self.assertIsNotNone(m)
        self.assertRegex(m.group(1), r"^v\d+\.\d+\.\d+$")

    def test_text_has_no_em_dash(self):
        self.assertNotIn("\u2014", self.text)
        self.assertNotIn("\u2013", self.text)

    def test_no_eval_and_no_curl_pipe_inside(self):
        code = [ln for ln in self.text.splitlines() if not ln.strip().startswith("#")]
        self.assertEqual([ln for ln in code if re.search(r"\beval\b|\|\s*(ba)?sh\b", ln)], [])

    @unittest.skipUnless(shutil.which("shellcheck"), "shellcheck is not installed (uvx shellcheck-py)")
    def test_shellcheck_is_clean(self):
        self._assert_shellcheck_clean(["shellcheck"])

    def test_shellcheck_is_clean_under_the_ci_version(self):
        # The public CI (ubuntu-24.04 image, apt shellcheck 0.9.0-1) flags notes that 0.11.0 does not
        # (SC2015 on install.sh, v0.37.3). Run that exact version from the uv cache, offline, when it is there.
        cmd = ci_shellcheck()
        if cmd is None:
            self.skipTest(f"shellcheck {CI_SHELLCHECK} is not runnable (set CRUCIBLE_SHELLCHECK_CI, see Makefile)")
        try:
            r = run([*cmd, "--version"], timeout=60)
        except OSError as e:
            self.fail(f"CRUCIBLE_SHELLCHECK_CI does not run: {e}")
        self.assertRegex(r.stdout, rf"(?m)^version: {re.escape(CI_SHELLCHECK)}$", r.stdout + r.stderr)
        self._assert_shellcheck_clean(cmd)

    def test_ci_shellcheck_env_var_is_used_and_never_skipped(self):
        # make test-fast sets CRUCIBLE_SHELLCHECK_CI when the pinned binary exists: a broken path must fail, not skip.
        with unittest.mock.patch.dict(os.environ, {"CRUCIBLE_SHELLCHECK_CI": "/nonexistent/shellcheck"}):
            self.assertEqual(ci_shellcheck(), ["/nonexistent/shellcheck"])

    def _assert_shellcheck_clean(self, cmd):
        for path in (INSTALL, ROOT / "tools/release_make.sh"):
            if path.exists():
                with self.subTest(path=path.name):
                    r = run([*cmd, "-x", "-S", "style", str(path)], timeout=120)
                    self.assertEqual(r.returncode, 0, r.stdout)


CI_SHELLCHECK = "0.9.0"
CI_SHELLCHECK_PY = "shellcheck-py==0.9.0.6"


def ci_shellcheck():
    """The command that runs the CI's shellcheck version, or None. A local binary on PATH that has
    the version wins; else uvx from its cache (no network, so the test stays fast and offline).
    CRUCIBLE_SHELLCHECK_CI (set by make test-fast when the pinned binary exists) wins over both and is never skipped."""
    forced = os.environ.get("CRUCIBLE_SHELLCHECK_CI")
    if forced:
        return [forced]
    candidates = []
    if shutil.which("shellcheck"):
        candidates.append(["shellcheck"])
    uvx = shutil.which("uvx")
    if uvx:
        candidates.append([uvx, "--offline", "--from", CI_SHELLCHECK_PY, "shellcheck"])
    for cmd in candidates:
        try:
            r = run([*cmd, "--version"], timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and re.search(rf"^version: {re.escape(CI_SHELLCHECK)}$", r.stdout, re.M):
            return cmd
    return None


if __name__ == "__main__":
    unittest.main()
