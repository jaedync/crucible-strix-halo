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
        ld = self.shims / "ldconfig"
        ld.write_text(f'#!/bin/sh\n"{REAL_LDCONFIG}" -p | grep -v -F -f "{self.ld_hide}"\ncat "{self.ld_add}"\n')
        ld.chmod(0o755)

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
            "XDG_STATE_HOME": str(self.home / ".state"), "TERM": "dumb", "LC_ALL": "C",
            "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
        }
        e.update(extra)
        return e

    def install(self, *args, answers="", via_pipe=False, **extra):
        self.tty.write_text(answers)
        if via_pipe:
            return subprocess.run(["bash", "-s", "--", *args], input=INSTALL.read_text(), capture_output=True, text=True,
                                  env=self.env(**extra), start_new_session=True, cwd=self.t, timeout=120)
        return subprocess.run(["bash", str(INSTALL), *args], stdin=subprocess.DEVNULL, capture_output=True, text=True,
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
        for path in (INSTALL, ROOT / "tools/release_make.sh"):
            if path.exists():
                r = run(["shellcheck", "-x", "-S", "style", str(path)])
                self.assertEqual(r.returncode, 0, r.stdout)


if __name__ == "__main__":
    unittest.main()
