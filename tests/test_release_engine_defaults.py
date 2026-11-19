"""The release defaults (tools/release_engine.py DEFAULT_FLAGS) against the real engine recipe of the Makefile.

`make -n -B build/engine/crucible` prints the `stamp end` line of the engine recipe with the settings that make computes for
that target (target-specific LDLIBS included). The tests run that line against a stand-in binary and compiler, then run
check_build_info. So a Makefile change to the engine settings and a stale DEFAULT_FLAGS cannot drift apart again (v0.38.0
cut, Nov 18: the structured-output merge added the constraint library to LDLIBS, and every trunk build was refused).
"""
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import release_engine  # noqa: E402

MAKE = shutil.which("make")
# The make and hipcc variables that change the recipe line or the stamp. The test clears them, so the defaults of the Makefile apply.
CLEARED = ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEFILES", "GNUMAKEFLAGS", "HIPCC", "CXX", *release_engine.FLAG_NAMES.values())
# The library settings of the Makefile (Makefile: XGR_DEFINES, XGR_INCLUDES; CXX is the make default g++).
MAKEFILE_LIB = {"cxx": "g++", "xgr_defines": "-DXGRAMMAR_ENABLE_CPPTRACE=0 -DXGRAMMAR_LOG_CUSTOMIZE=1",
                "xgr_includes": "-Ithird_party/xgrammar/include -isystem third_party/xgrammar/3rdparty/picojson "
                                "-isystem third_party/xgrammar/3rdparty/dlpack/include"}


def clean_env(**more):
    env = {k: v for k, v in os.environ.items() if k not in CLEARED}
    env.update(more)
    return env


def recipe_stamp_args(*make_args, env=None):
    """The words after `stamp end` in the engine recipe of the Makefile of this tree."""
    run = subprocess.run([MAKE, "-n", "-B", "--no-print-directory", "-C", str(ROOT), *make_args, "build/engine/crucible"],
                         capture_output=True, text=True, env=env or clean_env(), timeout=120)
    lines = [line for line in run.stdout.splitlines() if "tools/release_engine.py stamp end" in line]
    if run.returncode != 0 or len(lines) != 1:
        raise AssertionError(f"make -n gave rc {run.returncode} and {len(lines)} stamp end lines: {run.stderr[-400:]}")
    words = shlex.split(lines[0])
    return words[words.index("end") + 1:]


def replace(args, option, value):
    out = [w for w in args if not w.startswith(option + "=")]
    if option in out:
        i = out.index(option)
        del out[i:i + 2]
    return [*out, f"{option}={value}"]


@unittest.skipUnless(MAKE and shutil.which("git"), "needs make and git")
class RecipeDefaultsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)
        self.repo = self.t / "repo"
        self.repo.mkdir()
        git = ["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=test-user"]
        for cmd in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "t"]):
            subprocess.run([*git, *cmd], check=True, capture_output=True)
        self.head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        self.hipcc = self.t / "hipcc"
        self.hipcc.write_text('#!/bin/sh\necho "HIP version: 7.2.53211-fake"\n')
        self.hipcc.chmod(0o755)
        self.binary = self.t / "crucible"
        # A stand-in g++ first on PATH: the build-info names the compiler that `g++` runs, and its version line.
        self.bin = self.t / "bin"
        self.bin.mkdir()
        (self.bin / "g++").write_text('#!/bin/sh\necho "g++ (GCC) 15.2.1-fake 20260101"\necho more\n')
        (self.bin / "g++").chmod(0o755)

    def build_info(self, recipe_args):
        """Run `stamp begin` and the recipe's `stamp end` for a stand-in binary; return the build-info path."""
        tool = [sys.executable, "-I", str(ROOT / "tools/release_engine.py"), "stamp"]
        env = clean_env(PATH=f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}")
        begin = subprocess.run([*tool, "begin", "--binary", str(self.binary), "--repo", str(self.repo)], capture_output=True, text=True, env=env)
        self.assertEqual(begin.returncode, 0, begin.stderr)
        self.binary.write_bytes(b"engine " + hashlib.sha256(repr(recipe_args).encode()).digest())
        args = replace(replace(recipe_args, "--binary", str(self.binary)), "--hipcc", str(self.hipcc))
        end = subprocess.run([*tool, "end", *args, "--repo", str(self.repo)], capture_output=True, text=True, env=env)
        self.assertEqual(end.returncode, 0, end.stderr)
        return Path(str(self.binary) + ".build-info")

    def test_a_default_build_of_the_engine_recipe_passes_check_build_info(self):
        info = self.build_info(recipe_stamp_args())
        release_engine.check_build_info(self.binary, info, self.head)

    def test_the_recipe_records_each_make_setting_with_its_default(self):
        info = dict(line.partition(" ")[::2] for line in self.build_info(recipe_stamp_args()).read_text().splitlines())
        for name in release_engine.MAKE_FLAGS:
            with self.subTest(name=name):
                self.assertEqual(" ".join(info[name].split()), release_engine.DEFAULT_FLAGS[name])

    def test_the_recipe_records_the_library_compiler_and_settings(self):
        """M1 of the release-ldlibs review: CXX, XGR_DEFINES and XGR_INCLUDES build the static library in the engine."""
        info = dict(line.partition(" ")[::2] for line in self.build_info(recipe_stamp_args()).read_text().splitlines())
        for name, value in MAKEFILE_LIB.items():
            with self.subTest(name=name):
                self.assertEqual(" ".join(info[name].split()), value)
                self.assertEqual(release_engine.DEFAULT_FLAGS[name], value)
        self.assertEqual(info["cxx_path"], str(self.bin / "g++"))
        self.assertEqual(info["cxx_version"], "g++ (GCC) 15.2.1-fake 20260101")

    def test_a_non_default_build_of_the_engine_recipe_is_refused(self):
        cases = (("HIPFLAGS", (["HIPFLAGS=-O0"], None)),
                 ("LDLIBS", ([], clean_env(LDLIBS="-pthread -lfoo"))),
                 ("CPPFLAGS", (["CPPFLAGS=-DLOCAL_EXPERIMENT"], None)),
                 ("GPU_ARCH", (["GPU_ARCH=gfx1100"], None)),
                 ("CXX", (["CXX=clang++"], None)),
                 ("CXX", ([], clean_env(CXX="clang++"))),
                 ("XGR_DEFINES", (["XGR_DEFINES=-DXGRAMMAR_ENABLE_CPPTRACE=1 -DXGRAMMAR_LOG_CUSTOMIZE=1"], None)),
                 ("XGR_INCLUDES", (["XGR_INCLUDES=-I/tmp/other/xgrammar/include"], None)))
        for shown, (make_args, env) in cases:
            with self.subTest(shown=shown, make_args=make_args):
                info = self.build_info(recipe_stamp_args(*make_args, env=env))
                with self.assertRaises(release_engine.Refusal) as caught:
                    release_engine.check_build_info(self.binary, info, self.head)
                self.assertIn(shown, str(caught.exception))


class DefaultsCommandTest(unittest.TestCase):
    def test_defaults_prints_each_default_setting_as_a_build_info_line(self):
        run = subprocess.run([sys.executable, "-I", str(ROOT / "tools/release_engine.py"), "defaults"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(dict(line.partition(" ")[::2] for line in run.stdout.splitlines()), release_engine.DEFAULT_FLAGS)


if __name__ == "__main__":
    unittest.main()
