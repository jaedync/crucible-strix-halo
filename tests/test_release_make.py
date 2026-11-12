"""Tests of tools/release_make.sh: the build step that `update` runs after it checks out a new tag."""
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/release_make.sh"


class ReleaseMakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.repo = self.t / "repo"
        (self.repo / "tools").mkdir(parents=True)
        (self.repo / "build/engine").mkdir(parents=True)
        shutil.copy2(SCRIPT, self.repo / "tools/release_make.sh")
        self.log = self.t / "calls.log"
        self.bin = self.t / "bin"
        self.bin.mkdir()
        self.shim("make", 0)

    def shim(self, name, code, path=None):
        p = Path(path) if path else self.bin / name
        p.write_text(f'#!/bin/sh\necho "{name} $*" >> "{self.log}"\nexit {code}\n')
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    def installer(self, code):
        self.shim("install.sh", code, self.repo / "install.sh")

    def run_it(self):
        env = {"PATH": f"{self.bin}:{os.environ['PATH']}", "HOME": str(self.t)}
        return subprocess.run(["bash", "tools/release_make.sh"], cwd=self.repo, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def mark_release(self):
        (self.repo / "build/engine/release.json").write_text("{}")

    def test_source_build_install_runs_make(self):
        self.installer(0)
        r = self.run_it()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.calls(), ["make "])

    def test_release_install_fetches_the_release_engine_and_skips_make(self):
        self.mark_release()
        self.installer(0)
        r = self.run_it()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.calls(), [f"install.sh engine --dir {self.repo.resolve()}"])

    def test_no_release_engine_for_this_commit_falls_back_to_make_and_says_so(self):
        self.mark_release()
        self.installer(5)
        r = self.run_it()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.calls()[-1], "make ")
        self.assertIn("builds the engine from source", r.stdout + r.stderr)

    def test_make_failure_is_passed_on(self):
        self.shim("make", 2)
        self.installer(4)
        self.mark_release()
        self.assertEqual(self.run_it().returncode, 2)

    def test_no_installer_in_the_checkout_runs_make(self):
        self.mark_release()
        r = self.run_it()
        self.assertEqual((r.returncode, self.calls()), (0, ["make "]))


class MakeTargetTest(unittest.TestCase):
    def dry(self, *args):
        return subprocess.run(["make", "-n", "release-engine", *args], cwd=ROOT, capture_output=True, text=True)

    def test_target_builds_the_engine_then_writes_the_release_files(self):
        r = self.dry("ROCM_PATH=/opt/rocm-x")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("tools/release_engine.py release", r.stdout)
        self.assertIn("--binary build/engine/crucible", r.stdout)
        self.assertIn("--rocm-path /opt/rocm-x", r.stdout)
        self.assertIn("--tag v" + (ROOT / "VERSION").read_text().strip(), r.stdout)
        self.assertIn("--out dist/", r.stdout)

    def test_target_is_in_help_and_phony(self):
        text = (ROOT / "Makefile").read_text()
        self.assertRegex(text, r"\.PHONY:[^\n]*\brelease-engine\b")
        self.assertIn("make release-engine", text)


if __name__ == "__main__":
    unittest.main()
