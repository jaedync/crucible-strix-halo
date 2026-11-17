"""Tests of tools/hipcc_series.py and its use in the Makefile, setup and update (v0.37.2).

A ROCm 10.x hipcc (AMD clang 24; its `--version` says "HIP version: 7.16...", so the clang major is the only sure sign)
builds the v0.37.0 engine with a spill in the fat gate/up prefill kernel: P1 1,091 against 1,894 tok/s (-42%, log Nov 12
19:20). So make, setup and update warn, give the 7.2.4 advice and continue only with an explicit flag. The split build
cannot plan the clang 24 driver, so make uses plain hipcc (SPLIT=0) for it.
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

    def test_the_warning_names_the_loss_the_advice_and_the_flag(self):
        text = hs.warning("ALLOW_ROCM10=1")
        for word in ("ROCm 10", "42%", "1,091", "1,894", "ROCm 7.2.4", "ALLOW_ROCM10=1"):
            self.assertIn(word, text)
        self.assertIn("--allow-rocm10", hs.warning("--allow-rocm10"))


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


class MakeTest(unittest.TestCase):
    def test_make_with_the_724_compiler_builds_and_says_nothing(self):
        tree = Tree(self)
        r = tree.make(V724, "SPLIT=0")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("ROCm 10", r.stdout + r.stderr)
        self.assertEqual((tree.tree / "build/engine/crucible").read_bytes(), b"NEW")

    def test_make_with_a_rocm10_compiler_stops_with_the_warning(self):
        tree = Tree(self)
        r = tree.make(V101, "SPLIT=0")
        self.assertNotEqual(r.returncode, 0)
        out = r.stdout + r.stderr
        self.assertIn("42%", out)
        self.assertIn("ALLOW_ROCM10=1", out)
        self.assertFalse((tree.tree / "build/engine/crucible").exists())

    def test_make_with_a_rocm10_compiler_and_the_flag_warns_and_builds(self):
        tree = Tree(self)
        for how in (("ALLOW_ROCM10=1",), ()):
            env = None if how else {"ALLOW_ROCM10": "1"}   # the make word and the environment (setup and update use it)
            r = tree.make(V101, "SPLIT=0", *how, env_extra=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("42%", r.stdout + r.stderr)
            self.assertEqual((tree.tree / "build/engine/crucible").read_bytes(), b"NEW")
            (tree.tree / "build/engine/crucible").unlink()

    def test_split_build_with_clang_24_uses_plain_hipcc(self):
        """SPLIT=1 (the default) with clang 24: plain hipcc at once, devsplit is not asked (SPLIT=0 guard)."""
        tree = Tree(self)
        (tree.tree / "tools/devsplit.py").write_text("import sys\nprint('DEVSPLIT RAN')\nsys.exit(1)\n")
        r = tree.make(V101, "ALLOW_ROCM10=1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertNotIn("DEVSPLIT RAN", out)
        self.assertIn("SPLIT=0", out)
        self.assertEqual((tree.tree / "build/engine/crucible").read_bytes(), b"NEW")

    def test_split_build_with_clang_22_still_runs_devsplit(self):
        tree = Tree(self)
        (tree.tree / "tools/devsplit.py").write_text("import sys\nprint('DEVSPLIT RAN')\nsys.exit(1)\n")
        r = tree.make(V724)
        self.assertIn("DEVSPLIT RAN", r.stdout + r.stderr)


class CliTest(unittest.TestCase):
    """setup and update: an --allow-rocm10 flag that sets ALLOW_ROCM10=1 for make, and a clear stop without it."""

    def test_setup_and_update_have_the_flag(self):
        from crucible_cli import cli
        p = cli.parser() if hasattr(cli, "parser") else cli.build_parser()
        self.assertTrue(p.parse_args(["setup", "--allow-rocm10"]).allow_rocm10)
        self.assertFalse(p.parse_args(["setup"]).allow_rocm10)
        self.assertTrue(p.parse_args(["update", "--allow-rocm10"]).allow_rocm10)

    def test_the_stop_message_leaves_the_flag_to_the_caller(self):
        # setup and update each add their own "Run ... again with --allow-rocm10" sentence (review L2: once only).
        from crucible_cli import steps
        out = "make: ... " + hs.MARKER + " ..."
        msg = steps.rocm10_stop(out)
        self.assertNotIn("--allow-rocm10", msg)
        self.assertIn("ROCM_PATH", msg)
        self.assertIn("42%", msg)
        self.assertIsNone(steps.rocm10_stop("make: other error"))


if __name__ == "__main__":
    unittest.main()
