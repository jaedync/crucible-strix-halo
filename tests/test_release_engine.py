"""Tests of tools/release_engine.py (the release engine and its library list) and crucible_cli/release.py.

The library tests compile tiny ELF files with the system C compiler and skip when there is none.
"""
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from crucible_cli import release  # noqa: E402

TOOL = ROOT / "tools/release_engine.py"
CC = shutil.which("cc") or shutil.which("gcc")
READELF = shutil.which("readelf")
TAG = "v9.9.9"
COMMIT = "a" * 40
STEM = f"crucible-engine-{TAG}-gfx1151-rocm7.2-linux-x86_64"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        (self.t / "VERSION").write_text("9.9.9\n")
        (self.t / "install.sh").write_text('#!/usr/bin/env bash\nPIN_TAG="v9.9.9"\n')

    def tool(self, *args, **kw):
        return sh([sys.executable, "-I", str(TOOL), *args], **kw)

    def stamp(self, binary, commit=COMMIT, dirty=0, sha=None, flags=None, drop=()):
        """The build-info file that the engine recipe writes next to the binary (tools/release_engine.py stamp)."""
        binary = Path(binary)
        if binary.exists():
            sha = sha or sha256_file(binary)
            values = {"gpu_arch": "gfx1151", "hipflags": "-O3", "cppflags": "", "ldflags": "", "ldlibs": "-pthread",
                      "rocm_path": "/opt/rocm", "hipcc_compile_flags_append": "", "hipcc_link_flags_append": "",
                      "hip_clang_path": "", "hip_device_lib_path": "", "hipcc_version": "HIP version: 7.2.0", **(flags or {})}
            lines = [f"commit {commit}", f"dirty {dirty}", f"binary_sha256 {sha}"] + [f"{k} {v}" for k, v in values.items() if k not in drop]
            Path(str(binary) + ".build-info").write_text("\n".join(lines) + "\n")

    def release_args(self, binary, out, *more):
        self.stamp(binary)
        return ["release", "--binary", str(binary), "--tag", TAG, "--commit", COMMIT, "--out", str(out),
                "--version-file", str(self.t / "VERSION"), "--install-sh", str(self.t / "install.sh"), *more]


@unittest.skipUnless(CC and READELF, "needs a C compiler and readelf")
class LibraryListTest(Fixture):
    # Link flags that the system compiler adds by itself. Ubuntu's gcc adds --as-needed (the GitHub ubuntu-24.04 runner);
    # Arch's gcc adds nothing. The stand-ins must link and list the same libraries under both.
    compiler_default = ()

    def cc(self, out, src, *flags):
        c = self.t / (Path(out).name + ".c")
        c.write_text(src)
        # The source goes before the libraries, and --no-as-needed keeps each -l as a NEEDED entry whatever the default.
        r = sh([CC, *self.compiler_default, str(c), "-Wl,--no-as-needed", *flags, "-o", str(out)])
        self.assertEqual(r.returncode, 0, r.stderr)

    def world(self, hip_needs="-lz", register_needs=None, with_register=True, engine_needs=()):
        rocm = self.t / "rocm"
        (rocm / "lib").mkdir(parents=True)
        extra = self.t / "extra"
        extra.mkdir()
        # a stand-in for libamdhip64: needs a system library, and (optionally) another ROCm library
        flags = ["-shared", "-fPIC", "-Wl,-soname,libamdhip64.so.7", hip_needs]
        if with_register:
            reg = rocm / "lib/librocprofiler-register.so.0"
            self.cc(reg, "int reg_f(void){return 1;}", "-shared", "-fPIC", "-Wl,-soname,librocprofiler-register.so.0",
                    *(register_needs or ["-lzstd"]))
            flags += [f"-L{rocm / 'lib'}", f"-l:{reg.name}", f"-Wl,-rpath,{rocm / 'lib'}"]
        hip = rocm / "lib/libamdhip64.so.7"
        self.cc(hip, "int hip_f(void){return 2;}", *flags)
        binary = self.t / "engine"
        self.cc(binary, "int hip_f(void); int main(void){return hip_f()>9;}", f"-L{rocm / 'lib'}", f"-l:{hip.name}",
                "-lm", f"-Wl,-rpath,{rocm / 'lib'}", *(["-Wl,--no-as-needed", *engine_needs] if engine_needs else []))
        return binary, rocm

    def needs(self, binary, rocm, *more):
        r = self.tool("needs", "--binary", str(binary), "--rocm-path", str(rocm), "--json", *more)
        return r, (json.loads(r.stdout) if r.stdout.strip().startswith("{") else None)

    def test_the_stand_ins_link_the_same_when_the_compiler_adds_as_needed(self):
        # Ubuntu: "undefined reference to `hip_f'" (a library before the object is dropped), and a stand-in that does not
        # call libz would lose its NEEDED libz entry (CPU workflow, Nov 12).
        self.compiler_default = ("-Wl,--as-needed",)
        binary, rocm = self.world()
        r, data = self.needs(binary, rocm)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(data["external"], [])
        self.assertEqual(sorted(data["rocm_external"]), ["libz.so.1", "libzstd.so.1"])

    def test_the_list_holds_only_what_the_engine_itself_needs(self):
        # The engine needs ROCm and the base runtime. Its ROCm libraries need libz and libzstd on the BUILD host, but
        # the ROCm of the target brings its own: those two are information, never a requirement.
        binary, rocm = self.world()
        r, data = self.needs(binary, rocm)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(data["external"], [])
        self.assertEqual(sorted(data["rocm_external"]), ["libz.so.1", "libzstd.so.1"])
        self.assertIn("libamdhip64.so.7", data["rocm"])
        self.assertIn("librocprofiler-register.so.0", data["rocm"])
        for base in ("libc.so.6", "libm.so.6", "libstdc++.so.6", "libgcc_s.so.1"):
            self.assertNotIn(base, data["external"])
            self.assertNotIn(base, data["rocm_external"])

    def test_a_library_that_the_engine_needs_itself_is_listed_even_when_rocm_needs_it_too(self):
        binary, rocm = self.world(engine_needs=["-lz"])  # the hip stand-in needs libz as well
        _, data = self.needs(binary, rocm)
        self.assertEqual(data["external"], ["libz.so.1"])
        self.assertEqual(sorted(data["rocm_external"]), ["libzstd.so.1"])

    def test_a_library_of_a_rocm_library_goes_to_the_information_list(self):
        binary, rocm = self.world(hip_needs="-lm", register_needs=["-lzstd"])
        r, data = self.needs(binary, rocm)
        self.assertEqual(data["external"], [])
        self.assertEqual(sorted(data["rocm_external"]), ["libzstd.so.1"])
        text = self.tool("needs", "--binary", str(binary), "--rocm-path", str(rocm)).stdout
        self.assertRegex(text, r"libzstd\.so\.1\s+needed by librocprofiler-register\.so\.0")

    def test_needs_text_says_which_list_is_required(self):
        binary, rocm = self.world(engine_needs=["-lz"])
        r = self.tool("needs", "--binary", str(binary), "--rocm-path", str(rocm))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("libz.so.1", r.stdout)
        self.assertIn("the engine itself needs", r.stdout)
        self.assertIn("libzstd.so.1", r.stdout)
        self.assertIn("information only", r.stdout)

    def test_missing_rocm_library_stops_the_release_list(self):
        binary, rocm = self.world()
        (rocm / "lib/librocprofiler-register.so.0").unlink()
        r, _ = self.needs(binary, rocm)
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("librocprofiler-register.so.0", r.stderr)
        self.assertIn("--libdir", r.stderr)

    def test_extra_library_folder_completes_the_list(self):
        binary, rocm = self.world()
        keep = self.t / "keep"
        keep.mkdir()
        shutil.copy2(rocm / "lib/librocprofiler-register.so.0", keep)
        (rocm / "lib/librocprofiler-register.so.0").unlink()
        r, data = self.needs(binary, rocm, "--libdir", str(keep))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("libzstd.so.1", data["rocm_external"])

    def test_allow_incomplete_warns_and_goes_on(self):
        binary, rocm = self.world()
        (rocm / "lib/librocprofiler-register.so.0").unlink()
        r, data = self.needs(binary, rocm, "--allow-incomplete")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("incomplete", r.stderr)
        self.assertEqual(data["unresolved"], ["librocprofiler-register.so.0"])

    def test_version_floors_come_from_the_symbols(self):
        binary, rocm = self.world()
        _, data = self.needs(binary, rocm)
        text = sh([READELF, "-V", str(binary)]).stdout
        found = [tuple(int(x) for x in v.split(".")) for v in re.findall(r"GLIBC_(\d+(?:\.\d+)+)", text)]
        self.assertEqual(tuple(int(x) for x in data["glibc_min"].split(".")), max(found))

    def test_not_an_elf_file_is_refused(self):
        bad = self.t / "script"
        bad.write_text("#!/bin/sh\n")
        r = self.tool("needs", "--binary", str(bad), "--rocm-path", str(self.t))
        self.assertEqual(r.returncode, 2)
        self.assertIn("ELF", r.stderr)

    def test_release_writes_three_files_that_agree(self):
        binary, rocm = self.world()
        out = self.t / "dist"
        r = self.tool(*self.release_args(binary, out, "--rocm-path", str(rocm)))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(["SHA256SUMS", f"{STEM}.gz", f"{STEM}.needs"]))
        sums = dict(reversed(line.split("  ", 1)) for line in (out / "SHA256SUMS").read_text().splitlines())
        self.assertEqual(sums[f"{STEM}.gz"], sha256_file(out / f"{STEM}.gz"))
        self.assertEqual(sums[f"{STEM}.needs"], sha256_file(out / f"{STEM}.needs"))
        self.assertEqual(len(sums), 2)
        manifest = dict(line.split(" ", 1) for line in (out / f"{STEM}.needs").read_text().splitlines() if not line.startswith(("need ", "rocm_external ")))
        self.assertEqual(manifest["tag"], TAG)
        self.assertEqual(manifest["commit"], COMMIT)
        self.assertEqual(manifest["arch"], "gfx1151")
        self.assertEqual(manifest["format"], "1")
        self.assertEqual(manifest["binary_sha256"], sha256_file(binary))
        self.assertEqual(hashlib.sha256(gzip.decompress((out / f"{STEM}.gz").read_bytes())).hexdigest(), manifest["binary_sha256"])
        lines = (out / f"{STEM}.needs").read_text().splitlines()
        needs = [line.split(" ", 1)[1] for line in lines if line.startswith("need ")]
        self.assertEqual(needs, [])  # install.sh requires only these
        info = sorted(line.split(" ", 1)[1] for line in lines if line.startswith("rocm_external "))
        self.assertEqual(info, ["libz.so.1", "libzstd.so.1"])  # information: the build host ROCm needs them, the target ROCm may not
        self.assertTrue(sh(["sha256sum", "-c", "SHA256SUMS"], cwd=out).returncode == 0)

    def test_release_is_reproducible(self):
        binary, rocm = self.world()
        a, b = self.t / "a", self.t / "b"
        for d in (a, b):
            self.assertEqual(self.tool(*self.release_args(binary, d, "--rocm-path", str(rocm))).returncode, 0)
        self.assertEqual((a / "SHA256SUMS").read_text(), (b / "SHA256SUMS").read_text())

    def test_the_release_files_install_with_install_sh(self):
        binary, rocm = self.world()
        out = self.t / "dist"
        self.assertEqual(self.tool(*self.release_args(binary, out, "--rocm-path", str(rocm))).returncode, 0)
        src = self.t / "src"
        src.mkdir()
        env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
        git = shutil.which("git")
        for args in (["init", "-q", "-b", "main"], ["config", "user.email", "test-user"], ["config", "user.name", "t"]):
            sh([git, "-C", str(src), *args], env=env)
        (src / "crucible").write_text("#!/bin/sh\nexit 0\n")
        (src / "crucible").chmod(0o755)
        sh([git, "-C", str(src), "add", "-A"], env=env)
        sh([git, "-C", str(src), "commit", "-q", "-m", "r"], env=env)
        sh([git, "-C", str(src), "tag", TAG], env=env)
        commit = sh([git, "-C", str(src), "rev-parse", "HEAD"], env=env).stdout.strip()
        # the manifest names the real commit of the test repo
        out2 = self.t / "dist2"
        args = self.release_args(binary, out2, "--rocm-path", str(rocm))
        args[args.index("--commit") + 1] = commit
        self.stamp(binary, commit=commit)
        self.assertEqual(self.tool(*args).returncode, 0)
        home = self.t / "home"
        home.mkdir()
        tty = self.t / "tty"
        tty.write_text("")
        r = sh(["bash", str(ROOT / "install.sh"), "--yes"], stdin=subprocess.DEVNULL, start_new_session=True, env={
            "HOME": str(home), "PATH": os.environ["PATH"], "CRUCIBLE_RELEASE_BASE": f"file://{out2}",
            "CRUCIBLE_REPO_URL": f"file://{src}", "CRUCIBLE_TAG": TAG, "CRUCIBLE_DIR": str(home / "c"),
            "CRUCIBLE_TTY": str(tty), "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(sha256_file(home / "c/build/engine/crucible"), sha256_file(binary))
        self.assertIn("Libraries: all", r.stdout)


REAL_ENGINE = os.environ.get("CRUCIBLE_TEST_ENGINE")
REAL_ROCM = os.environ.get("CRUCIBLE_TEST_ROCM")
REAL_LIBDIR = os.environ.get("CRUCIBLE_TEST_LIBDIR")


@unittest.skipUnless(REAL_ENGINE and REAL_ROCM and Path(REAL_ENGINE).is_file() and READELF,
                     "set CRUCIBLE_TEST_ENGINE and CRUCIBLE_TEST_ROCM (and CRUCIBLE_TEST_LIBDIR) to check a real engine build")
class RealEngineTest(Fixture):
    """The soname check on a real engine build: fmt 12 must be in the list, so install.sh checks it on the target."""

    def test_real_engine_lists_the_libraries_that_are_not_rocm(self):
        args = ["needs", "--binary", REAL_ENGINE, "--rocm-path", REAL_ROCM, "--json"]
        if REAL_LIBDIR:
            args += ["--libdir", REAL_LIBDIR]
        r = self.tool(*args)
        if not REAL_LIBDIR and "librocprofiler-register" in r.stderr:
            self.assertEqual(r.returncode, 2)  # this ROCm tree is incomplete: the check says so instead of a short list
            return
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertIn("libfmt.so.12", data["rocm_external"])  # ROCm of the build host needs it; install.sh does not require it
        self.assertIn("libnuma.so.1", data["rocm_external"])
        self.assertNotIn("libfmt.so.12", data["external"])
        for rocm_lib in ("libamdhip64.so.7", "libhsa-runtime64.so.1"):
            self.assertIn(rocm_lib, data["rocm"])
            self.assertNotIn(rocm_lib, data["external"])
        self.assertRegex(data["glibc_min"], r"^2\.\d+")


class BuildInfoTest(Fixture):
    """The release refuses a binary that is not the build of the commit it claims, or that was built from a dirty tree."""

    def setUp(self):
        super().setUp()
        self.binary = self.t / "engine"
        self.binary.write_bytes(b"\x7fELF-not-real")

    def refused(self, why):
        r = self.tool(*self.release_args(self.binary, self.t / "dist"))
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn(why, r.stderr)
        self.assertFalse((self.t / "dist").exists())

    def args_without_stamp(self):
        args = self.release_args(self.binary, self.t / "dist")
        Path(str(self.binary) + ".build-info").unlink(missing_ok=True)
        return args

    def test_no_build_info_is_refused(self):
        r = self.tool(*self.args_without_stamp())
        self.assertEqual(r.returncode, 2)
        self.assertIn("build-info", r.stderr)

    def test_a_binary_that_changed_after_the_build_is_refused(self):
        args = self.release_args(self.binary, self.t / "dist")
        self.binary.write_bytes(b"\x7fELF-other")
        r = self.tool(*args)
        self.assertEqual(r.returncode, 2)
        self.assertIn("does not match the binary", r.stderr)

    def test_a_build_of_another_commit_is_refused(self):
        args = self.release_args(self.binary, self.t / "dist")
        self.stamp(self.binary, commit="b" * 40)
        r = self.tool(*args)
        self.assertEqual(r.returncode, 2)
        self.assertIn("another commit", r.stderr)

    def test_a_build_of_a_dirty_tree_is_refused(self):
        args = self.release_args(self.binary, self.t / "dist")
        self.stamp(self.binary, dirty=1)
        r = self.tool(*args)
        self.assertEqual(r.returncode, 2)
        self.assertIn("local changes", r.stderr)

    def test_an_engine_built_with_other_flags_is_refused(self):
        for name, value, shown in (("hipflags", "-O0 -DLOCAL_EXPERIMENT", "HIPFLAGS"), ("cppflags", "-DX=1", "CPPFLAGS"),
                                   ("ldflags", "-L/tmp/x", "LDFLAGS"), ("ldlibs", "-pthread -lfoo", "LDLIBS"),
                                   ("gpu_arch", "gfx1100", "GPU_ARCH")):
            with self.subTest(name=name):
                args = self.release_args(self.binary, self.t / "dist")
                self.stamp(self.binary, flags={name: value})
                r = self.tool(*args)
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertIn(shown, r.stderr)
                self.assertFalse((self.t / "dist").exists())

    def test_an_engine_with_no_compile_settings_in_the_build_info_is_refused(self):
        for drop in (("hipflags",), ("cppflags",), ("hipcc_version",)):
            with self.subTest(drop=drop):
                args = self.release_args(self.binary, self.t / "dist")
                self.stamp(self.binary, drop=drop)
                r = self.tool(*args)
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertRegex(r.stderr, "no value for|compiler version")

    def test_the_default_settings_are_accepted_and_spaces_do_not_matter(self):
        args = self.release_args(self.binary, self.t / "dist")
        self.stamp(self.binary, flags={"hipflags": "  -O3  "})
        self.assertNotRegex(self.tool(*args).stderr, "built with|no value for|compiler version")  # it goes on to the library check


class StampTest(Fixture):
    """stamp begin / stamp end: the build info of one engine build."""

    def setUp(self):
        super().setUp()
        self.repo = self.t / "repo"
        self.repo.mkdir()
        self.git = ["git", "-C", str(self.repo), "-c", "user.name=t", "-c", "user.email=test-user"]
        sh([*self.git, "init", "-q"], check=True)
        (self.repo / "a.txt").write_text("a\n")
        sh([*self.git, "add", "-A"], check=True)
        sh([*self.git, "commit", "-q", "-m", "t"], check=True)
        self.head = sh([*self.git, "rev-parse", "HEAD"]).stdout.strip()
        self.binary = self.t / "engine"

    def stamp(self, phase):
        return self.tool("stamp", phase, "--binary", str(self.binary), "--repo", str(self.repo))

    def info(self):
        return dict(line.split(" ", 1) for line in Path(str(self.binary) + ".build-info").read_text().splitlines())

    def test_end_writes_commit_clean_and_the_hash_of_the_binary(self):
        self.assertEqual(self.stamp("begin").returncode, 0)
        self.binary.write_bytes(b"engine")
        r = self.stamp("end")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.info(), {"commit": self.head, "dirty": "0", "binary_sha256": sha256_file(self.binary)})

    def test_end_with_the_compiler_writes_the_settings_and_the_compiler_version(self):
        fake = self.t / "hipcc"
        fake.write_text('#!/bin/sh\n[ "$1" = "--version" ] && echo "HIP version: 7.2.4-fake"; echo "more"\n')
        fake.chmod(0o755)
        self.stamp("begin")
        self.binary.write_bytes(b"engine")
        r = self.tool("stamp", "end", "--binary", str(self.binary), "--repo", str(self.repo), "--hipcc", str(fake),
                      "--gpu-arch=gfx1151", "--hipflags=-O0  -DX='y'", "--cppflags=", "--ldflags=", "--ldlibs=-pthread")
        self.assertEqual(r.returncode, 0, r.stderr)
        text = Path(str(self.binary) + ".build-info").read_text()
        self.assertIn("hipflags -O0 -DX='y'\n", text)
        self.assertIn("cppflags \n", text)
        self.assertIn("hipcc_version HIP version: 7.2.4-fake\n", text)

    def test_end_records_the_hip_clang_and_device_library_variables_of_the_environment(self):
        fake = self.t / "hipcc"
        fake.write_text('#!/bin/sh\necho "HIP version: 7.2.4-fake"\n')
        fake.chmod(0o755)
        self.stamp("begin")
        self.binary.write_bytes(b"engine")
        r = self.tool("stamp", "end", "--binary", str(self.binary), "--repo", str(self.repo), "--hipcc", str(fake),
                      env={**os.environ, "HIP_CLANG_PATH": "/tmp/other/clang", "HIP_DEVICE_LIB_PATH": "/tmp/other/lib"})
        self.assertEqual(r.returncode, 0, r.stderr)
        text = Path(str(self.binary) + ".build-info").read_text()
        self.assertIn("hip_clang_path /tmp/other/clang\n", text)
        self.assertIn("hip_device_lib_path /tmp/other/lib\n", text)

    def test_a_compiler_that_cannot_run_gives_unknown(self):
        self.stamp("begin")
        self.binary.write_bytes(b"engine")
        self.tool("stamp", "end", "--binary", str(self.binary), "--repo", str(self.repo), "--hipcc", "/no/such/hipcc")
        self.assertIn("hipcc_version unknown\n", Path(str(self.binary) + ".build-info").read_text())

    def test_a_tree_with_local_changes_is_dirty(self):
        (self.repo / "a.txt").write_text("changed\n")
        self.stamp("begin")
        self.binary.write_bytes(b"engine")
        self.stamp("end")
        self.assertEqual(self.info()["dirty"], "1")

    def test_a_change_during_the_build_makes_it_dirty(self):
        self.stamp("begin")
        self.binary.write_bytes(b"engine")
        (self.repo / "a.txt").write_text("edited while the compiler ran\n")
        self.stamp("end")
        self.assertEqual(self.info()["dirty"], "1")

    def test_a_new_commit_during_the_build_makes_it_dirty(self):
        self.stamp("begin")
        self.binary.write_bytes(b"engine")
        sh([*self.git, "commit", "-q", "--allow-empty", "-m", "more"], check=True)
        self.stamp("end")
        self.assertEqual(self.info()["dirty"], "1")

    def test_begin_removes_an_old_build_info(self):
        Path(str(self.binary) + ".build-info").write_text("commit x\n")
        self.stamp("begin")
        self.assertFalse(Path(str(self.binary) + ".build-info").exists())

    def test_end_without_a_binary_or_without_begin_refuses(self):
        self.stamp("begin")
        r = self.stamp("end")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no binary", r.stderr)
        self.binary.write_bytes(b"engine")
        Path(str(self.binary) + ".build-info.pending").unlink()
        self.assertEqual(self.stamp("end").returncode, 2)

    def test_a_folder_that_is_not_a_git_repo_gives_no_usable_info(self):
        r = self.tool("stamp", "begin", "--binary", str(self.binary), "--repo", str(self.t))
        self.binary.write_bytes(b"engine")
        self.tool("stamp", "end", "--binary", str(self.binary), "--repo", str(self.t))
        self.assertEqual(self.info()["commit"], "none")


class PinTest(Fixture):
    def make_elf(self):
        # not a real ELF: the pin checks run before the binary is read
        p = self.t / "engine"
        p.write_bytes(b"\x7fELF")
        return p

    def test_tag_must_match_install_sh(self):
        (self.t / "install.sh").write_text('PIN_TAG="v1.0.0"\n')
        r = self.tool(*self.release_args(self.make_elf(), self.t / "dist"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("PIN_TAG", r.stderr)
        self.assertFalse((self.t / "dist").exists())

    def test_tag_must_match_version_file(self):
        (self.t / "VERSION").write_text("1.0.0\n")
        r = self.tool(*self.release_args(self.make_elf(), self.t / "dist"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("VERSION", r.stderr)

    def test_tag_and_commit_format(self):
        for bad_tag, bad_commit in (("0.34.0", COMMIT), ("v0.34", COMMIT), (TAG, "abc")):
            args = self.release_args(self.make_elf(), self.t / "dist")
            args[args.index("--tag") + 1] = bad_tag
            args[args.index("--commit") + 1] = bad_commit
            self.assertEqual(self.tool(*args).returncode, 2, (bad_tag, bad_commit))

    def test_out_folder_must_be_new(self):
        out = self.t / "dist"
        out.mkdir()
        (out / "keep").write_text("x")
        r = self.tool(*self.release_args(self.make_elf(), out))
        self.assertEqual(r.returncode, 2)
        self.assertTrue((out / "keep").exists())


class EngineReleaseRecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        (self.repo / "build/engine").mkdir(parents=True)
        self.engine = self.repo / "build/engine/crucible"
        self.engine.write_bytes(b"engine bytes")
        self.record = {"format": 1, "tag": "v0.34.0", "commit": "b" * 40, "engine_sha256": hashlib.sha256(b"engine bytes").hexdigest(),
                       "arch": "gfx1151", "rocm": "7.2", "asset": "x.gz", "glibc_min": "2.38", "glibcxx_min": "3.4.32"}
        self.write()

    def write(self, record=None):
        (self.repo / "build/engine/release.json").write_text(json.dumps(self.record if record is None else record))

    def test_valid_record(self):
        info = release.engine_release(self.repo)
        self.assertEqual(info["tag"], "v0.34.0")

    def test_changed_engine_is_not_a_release_engine(self):
        self.engine.write_bytes(b"rebuilt from source")
        self.assertIsNone(release.engine_release(self.repo))

    def test_no_record_or_no_engine(self):
        (self.repo / "build/engine/release.json").unlink()
        self.assertIsNone(release.engine_release(self.repo))
        self.write()
        self.engine.unlink()
        self.assertIsNone(release.engine_release(self.repo))

    def test_bad_records(self):
        for bad in ("not json", "[]", json.dumps({"format": 2, **{k: v for k, v in self.record.items() if k != "format"}}),
                    json.dumps({k: v for k, v in self.record.items() if k != "tag"}),
                    json.dumps({**self.record, "engine_sha256": "xyz"}), json.dumps({**self.record, "tag": "main"})):
            (self.repo / "build/engine/release.json").write_text(bad)
            self.assertIsNone(release.engine_release(self.repo), bad)

    def test_symlinked_engine_is_refused(self):
        real = self.repo / "elsewhere"
        real.write_bytes(b"engine bytes")
        self.engine.unlink()
        self.engine.symlink_to(real)
        self.assertIsNone(release.engine_release(self.repo))


if __name__ == "__main__":
    unittest.main()
