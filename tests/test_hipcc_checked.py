"""hipcc exits 0 when its clang child dies by SIGKILL (ROCm 7.2.4 and 10.1; real OOM kills on Nov 5). These tests run a
fake compiler that does the same through tools/hipcc_checked.sh and through every hipcc recipe of the Makefile, with an
OLD output file in place. The build must fail and the old output must be gone. No real compile runs."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools/hipcc_checked.sh"

# FAKE_MODE: ok (writes the output), kill (a child gets SIGKILL, the wrapper exits 0, no output),
# none (exits 0, no output), empty (exits 0, empty output), rc3 (writes a partial output, exits 3)
FAKE = r"""#!/bin/bash
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift; fi
  shift
done
case "$FAKE_MODE" in
  ok) printf 'NEW' > "$out" ;;
  kill) sleep 30 & child=$!; kill -9 "$child"; wait "$child" 2>/dev/null; exit 0 ;;
  none) exit 0 ;;
  empty) : > "$out" ;;
  rc3) printf 'PARTIAL' > "$out"; exit 3 ;;
esac
"""


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.fake = self.t / "fake_hipcc.sh"
        self.fake.write_text(FAKE)
        self.fake.chmod(self.fake.stat().st_mode | stat.S_IXUSR)


class HelperTest(Base):
    def run_helper(self, mode, out):
        env = dict(os.environ, FAKE_MODE=mode)
        return subprocess.run(["bash", str(HELPER), str(out), "bash", str(self.fake), "-o", str(out)],
                              capture_output=True, text=True, env=env)

    def old(self):
        out = self.t / "bin"
        out.write_text("OLD")
        return out

    def test_killed_compiler_with_exit_0_fails_and_removes_the_old_output(self):
        out = self.old()
        r = self.run_helper("kill", out)
        self.assertNotEqual(r.returncode, 0, r.stderr)
        self.assertFalse(out.exists())
        self.assertIn("no output", r.stderr)

    def test_exit_0_without_output_fails(self):
        out = self.old()
        self.assertNotEqual(self.run_helper("none", out).returncode, 0)
        self.assertFalse(out.exists())

    def test_empty_output_fails(self):
        out = self.old()
        self.assertNotEqual(self.run_helper("empty", out).returncode, 0)
        self.assertFalse(out.exists())

    def test_compiler_failure_keeps_its_exit_code_and_removes_the_partial_output(self):
        out = self.old()
        r = self.run_helper("rc3", out)
        self.assertEqual(r.returncode, 3)
        self.assertFalse(out.exists())

    def test_a_good_compile_replaces_the_old_output(self):
        out = self.old()
        r = self.run_helper("ok", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(out.read_text(), "NEW")


class MakeTest(Base):
    """A copy of the Makefile in a small tree, run with HIPCC=<fake>."""

    def setUp(self):
        super().setUp()
        self.tree = self.t / "tree"
        for rel in ("Makefile", "VERSION", "tools/hipcc_checked.sh", "tools/release_engine.py", "tools/devsplit.py",
                    "tools/devsplit_ir.py", "tools/devsplit_groups.tsv", "tools/devsplit_facts.tsv"):
            if (ROOT / rel).exists():
                (self.tree / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / rel, self.tree / rel)
        (self.tree / "engine").mkdir(exist_ok=True)
        (self.tree / "engine/engine.hip").write_text("// engine\n")
        (self.tree / "tests").mkdir(exist_ok=True)
        (self.tree / "tests/probe.hip").write_text("// probe\n")
        git = ["git", "-C", str(self.tree), "-c", "user.name=t", "-c", "user.email=test-user"]
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-q", "-m", "t"], check=True)
        self.git = git

    def make(self, target, mode, *more):
        env = dict(os.environ, FAKE_MODE=mode)
        env.pop("MAKEFLAGS", None)
        env.pop("MFLAGS", None)
        return subprocess.run(["make", target, f"HIPCC=bash {self.fake}", "PYTHON=" + sys.executable, *more],
                              cwd=self.tree, capture_output=True, text=True, env=env)

    def old_output(self, target):
        out = self.tree / target
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("OLD")
        os.utime(out, (1, 1))  # older than every prerequisite, so make runs the recipe
        return out

    def check_fails(self, target, *more):
        for mode in ("kill", "none"):
            out = self.old_output(target)
            r = self.make(target, mode, *more)
            self.assertNotEqual(r.returncode, 0, (mode, r.stdout, r.stderr))
            self.assertFalse(out.exists(), (mode, "the old output stayed"))

    def test_generic_hip_recipe(self):
        self.check_fails("build/tests/probe")

    def test_mono_engine_recipe(self):
        self.check_fails("build/engine/crucible.mono")

    def test_engine_recipe_with_plain_hipcc(self):
        self.check_fails("build/engine/crucible", "SPLIT=0")

    def test_engine_recipe_with_the_split_driver_falling_back_to_hipcc(self):
        # the fake compiler gives no driver plan, so devsplit exits 2 and the recipe runs plain hipcc
        self.check_fails("build/engine/crucible", "SPLIT=1")

    def test_a_good_build_still_works_and_writes_the_build_info(self):
        r = self.make("build/engine/crucible", "ok", "SPLIT=0")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual((self.tree / "build/engine/crucible").read_text(), "NEW")
        info = dict(line.split(" ", 1) for line in (self.tree / "build/engine/crucible.build-info").read_text().splitlines())
        head = subprocess.run([*self.git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.assertEqual(info["commit"], head)
        self.assertEqual(info["dirty"], "0")

    def test_a_failed_build_leaves_no_old_build_info(self):
        self.make("build/engine/crucible", "ok", "SPLIT=0")
        os.utime(self.tree / "engine/engine.hip", None)
        self.assertNotEqual(self.make("build/engine/crucible", "kill", "SPLIT=0").returncode, 0)
        self.assertFalse((self.tree / "build/engine/crucible.build-info").exists())

    def test_a_build_of_a_dirty_tree_is_marked_dirty(self):
        (self.tree / "engine/engine.hip").write_text("// changed\n")
        self.assertEqual(self.make("build/engine/crucible", "ok", "SPLIT=0").returncode, 0)
        text = (self.tree / "build/engine/crucible.build-info").read_text()
        self.assertIn("dirty 1", text.splitlines())


if __name__ == "__main__":
    unittest.main()
