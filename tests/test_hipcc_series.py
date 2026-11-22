"""ROCm 10.x builds after v0.37.2: no stop, and clang 24 builds with plain hipcc (tools/hipcc_series.py).

v0.37.2 stopped make, setup and update for a ROCm 10.x hipcc (AMD clang 24), because the v0.37.0 engine spilled in the fat
gate/up prefill kernel (P1 -42%). Trunk has the source fix of that spill (cb97f418; the t38 gate measured 10.1 P1 -1.39%
against 7.2.4), so the stop is gone, and the words of the v0.37.2 notes (`ALLOW_ROCM10=1`, `--allow-rocm10`) are accepted
and change nothing. The gated 10.1 binary (32a92d58) is the plain hipcc build; the clang 24 split build gives other ISA in
92 functions and has no gate. So with clang 24, make still builds plain hipcc (SPLIT=0) (the decision of Nov 15).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import hipcc_series as hs  # noqa: E402

V724 = ("HIP version: 7.2.53211-9999\nAMD clang version 22.0.0git (/srcdest/rocm-llvm f58b06dc)\n"
        "Target: x86_64-pc-linux-gnu\n")
V101 = ("HIP version: 7.16.26385-0000000\nAMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git d6f6cb69)\n"
        "Target: x86_64-unknown-linux-gnu\n")
FAKE = '#!/bin/bash\nif [ "$1" = "--version" ]; then printf "%s" "$VERSION_TEXT"; exit 0; fi\n' \
       'out=""\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && { out="$2"; shift; }; shift; done\nprintf NEW > "$out"\n'


class ParseTest(unittest.TestCase):
    def test_the_clang_major_of_each_compiler(self):
        self.assertEqual(hs.clang_major(V724), 22)
        self.assertEqual(hs.clang_major(V101), 24)
        self.assertIsNone(hs.clang_major("HIP version: 7.2.0\n"))
        self.assertIsNone(hs.clang_major(""))

    def test_only_clang_24_or_later_is_rocm10(self):
        self.assertFalse(hs.is_rocm10(V724))
        self.assertTrue(hs.is_rocm10(V101))
        self.assertTrue(hs.is_rocm10(V101.replace("version 24", "version 25")))
        self.assertFalse(hs.is_rocm10("unknown"))


class Tree:
    """A copy of the build files with a fake hipcc, as tests/test_sec_server.py ReleaseTest makes it."""

    FILES = ("Makefile", "VERSION", "tools/hipcc_checked.sh", "tools/release_engine.py", "tools/devsplit.py",
             "tools/devsplit_ir.py", "tools/devsplit_groups.tsv", "tools/devsplit_facts.tsv", "tools/hipcc_series.py")

    def __init__(self, test):
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)
        self.tree = self.t / "tree"
        for rel in self.FILES:
            (self.tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, self.tree / rel)
        (self.tree / "engine").mkdir()
        (self.tree / "engine/engine.hip").write_text("// engine\n")
        self.fake = self.t / "hipcc"
        self.fake.write_text(FAKE)
        self.fake.chmod(0o755)
        git = ["git", "-C", str(self.tree), "-c", "user.name=t", "-c", "user.email=test-user"]
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "t"]):
            subprocess.run([*git, *args], check=True, capture_output=True)

    def make(self, version, *args, env_extra=None):
        env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "ALLOW_ROCM10")}
        env["VERSION_TEXT"] = version
        env.update(env_extra or {})
        return subprocess.run(["timeout", "-k", "10", "120", "make", "build/engine/crucible", f"HIPCC=bash {self.fake}",
                               "PYTHON=" + sys.executable, "STATE_EPOCH=0", *args],
                              cwd=self.tree, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)


DEVSPLIT_STUB = "import sys\nprint('DEVSPLIT RAN')\nsys.exit(2)\n"   # exit 2: no plan, the recipe runs plain hipcc


class MakeTest(unittest.TestCase):
    def test_make_with_the_724_compiler_builds_and_says_nothing(self):
        tree = Tree(self)
        r = tree.make(V724, "SPLIT=0")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("ROCm 10", r.stdout + r.stderr)
        self.assertEqual((tree.tree / "build/engine/crucible").read_bytes(), b"NEW")

    def test_make_with_a_rocm10_compiler_builds_with_no_stop(self):
        tree = Tree(self)
        r = tree.make(V101, "SPLIT=0")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertNotIn("42%", out)
        self.assertNotIn("CRUCIBLE-ROCM10-STOP", out)
        self.assertEqual((tree.tree / "build/engine/crucible").read_bytes(), b"NEW")

    def test_allow_rocm10_is_accepted_and_changes_nothing(self):
        """The v0.37.2 CHANGELOG tells users to run `ALLOW_ROCM10=1 ./crucible update`: the make word and the environment
        still build, with the same output and no warning."""
        tree = Tree(self)
        for how in (("ALLOW_ROCM10=1",), ()):
            env = None if how else {"ALLOW_ROCM10": "1"}
            for version in (V101, V724):
                r = tree.make(version, "SPLIT=0", *how, env_extra=env)
                out = r.stdout + r.stderr
                self.assertEqual(r.returncode, 0, out)
                self.assertNotIn("42%", out)
                self.assertEqual((tree.tree / "build/engine/crucible").read_bytes(), b"NEW")
                (tree.tree / "build/engine/crucible").unlink()

    def test_split_build_with_clang_24_uses_plain_hipcc(self):
        """SPLIT=1 (the default) with clang 24: plain hipcc at once, devsplit is not asked (the gated 10.1 recipe)."""
        tree = Tree(self)
        (tree.tree / "tools/devsplit.py").write_text(DEVSPLIT_STUB)
        r = tree.make(V101)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertNotIn("DEVSPLIT RAN", out)
        self.assertIn("SPLIT=0", out)
        self.assertEqual((tree.tree / "build/engine/crucible").read_bytes(), b"NEW")

    def test_split_build_with_clang_22_still_runs_devsplit(self):
        tree = Tree(self)
        (tree.tree / "tools/devsplit.py").write_text(DEVSPLIT_STUB)
        r = tree.make(V724)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("DEVSPLIT RAN", out)


class CliTest(unittest.TestCase):
    """setup and update accept --allow-rocm10 (the v0.37.2 flag) with no error; it changes nothing."""

    def test_setup_and_update_accept_the_flag(self):
        from crucible_cli import cli
        p = cli.parser() if hasattr(cli, "parser") else cli.build_parser()
        for cmd in ("setup", "update"):
            p.parse_args([cmd, "--allow-rocm10"])
            p.parse_args([cmd])

    def test_the_flag_no_longer_reaches_make(self):
        self.assertFalse(hasattr(hs, "MARKER"))
        from crucible_cli import steps, update
        self.assertFalse(hasattr(steps, "rocm10_stop"))
        for mod in (steps, update):
            self.assertNotIn('"ALLOW_ROCM10"', Path(mod.__file__).read_text())


if __name__ == "__main__":
    unittest.main()
