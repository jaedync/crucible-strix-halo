"""Delta review of the install and service fixes (sec-delta-wps): N5, N6, N8. Each test states the fixed behaviour."""
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("CRUCIBLE_NO_UPDATE_CHECK", "1")
import test_install_sh as ish  # noqa: E402
from crucible_update_fakes import World  # noqa: E402
from crucible_cli import cli, packs, settings  # noqa: E402

PACK = packs.PACKS["4.05"].folder


class UninstallTest(unittest.TestCase):
    def cli(self, w, *args):
        host = w.host()
        code = cli.main(list(args), host=host, repo=w.dir)
        self.out = "\n".join(host.lines)
        return code

    def config(self, w, **values):
        path = w.dir / ".crucible/config.json"
        cfg = json.loads(path.read_text())
        cfg.update(values)
        path.write_text(json.dumps(cfg))

    def test_ok_remove_models_keeps_other_files_of_the_models_folder_inside_the_install(self):
        w = World(self)
        (w.models / "my-other-model.gguf").write_text("OTHER")
        self.assertEqual(self.cli(w, "uninstall", "--yes", "--remove-models"), 0, self.out)
        self.assertFalse(w.pack.exists())
        self.assertTrue((w.models / "my-other-model.gguf").exists(), self.out)

    def test_n5_remove_models_checks_the_content_of_an_owned_state_folder(self):
        """setup accepts a folder that does not exist yet (for example ~/.cache on a new account) and records it. Files that
        come later must stay: --remove-models looks inside the folder before it removes it."""
        w = World(self)
        cache = w.home / "newcache"
        self.assertIsNone(settings.state_folder_problem(cache))  # setup accepts it
        (w.dir / ".crucible/state-dir.json").write_text(settings.state_marker_text(cache))
        (cache / "pip").mkdir(parents=True)
        (cache / "pip/keep").write_text("x")
        self.config(w, state_dir=str(cache))
        self.assertEqual(self.cli(w, "uninstall", "--yes", "--remove-models"), 0, self.out)
        self.assertTrue((cache / "pip/keep").exists(), self.out)

    def test_n5_a_state_folder_with_only_engine_files_is_still_removed(self):
        w = World(self)
        state = w.home / "states"
        (state / "write-ledger").mkdir(parents=True)
        (w.dir / ".crucible/state-dir.json").write_text(settings.state_marker_text(state))
        self.config(w, state_dir=str(state))
        self.assertEqual(self.cli(w, "uninstall", "--yes", "--remove-models"), 0, self.out)
        self.assertFalse(state.exists(), self.out)


class UpdatePipTest(unittest.TestCase):
    def test_n6_the_recorded_pins_hold_pip_setuptools_and_wheel(self):
        """`pip freeze` without --all leaves out pip, setuptools and wheel, so a rollback cannot put them back."""
        w = World(self, bad_start=True, new_requirements="numpy\nrequests\n")
        cli.main(["update", "--yes"], host=w.host(), repo=w.dir)
        freezes = [e[1] for e in w.machine.events if e[0] == "pip" and e[1].startswith("freeze")]
        self.assertTrue(freezes)
        self.assertTrue(all("--all" in f.split() for f in freezes), freezes)

    def test_n6_a_rollback_removes_the_packages_that_the_new_requirements_added(self):
        w = World(self, bad_start=True, new_requirements="numpy\nrequests\n")
        w.machine.freeze = "numpy==1.0.0\nrequests==2.0.0\n"
        w.machine.installed_after = "numpy==1.0.0\nrequests==2.0.0\nnewpkg==3.0.0\n"
        cli.main(["update", "--yes"], host=w.host(), repo=w.dir)
        pips = [e[1] for e in w.machine.events if e[0] == "pip"]
        uninstalls = [p for p in pips if p.startswith("uninstall")]
        self.assertEqual(len(uninstalls), 1, pips)
        self.assertIn("newpkg", uninstalls[0])
        self.assertNotIn("numpy", uninstalls[0])


class StaticTest(unittest.TestCase):
    def test_n8_a_source_build_install_is_pinned_to_the_release_manifest(self):
        text = (ROOT / "install.sh").read_text()
        body = text[text.index("cmd_install() {"):text.index("cmd_engine() {")]
        self.assertNotIn('MAN_COMMIT=""', body)


class SourceBuildPinTest(unittest.TestCase):
    def setUp(self):
        self.w = ish.Env(self)

    def test_n8_a_moved_tag_is_refused_with_the_source_build(self):
        self.w.make_release(commit="2" * 40)
        r = self.w.install("--yes", "--source-build")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 4, out)
        self.assertIn("manifest", out)
        self.assertFalse(self.w.dir.exists())

    def test_n8_the_source_build_fetches_the_manifest_but_not_the_engine(self):
        r = self.w.install("--yes", "--source-build")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        urls = " ".join(self.w.calls("curl"))
        self.assertIn(".needs", urls)
        self.assertIn("SHA256SUMS", urls)
        self.assertNotIn(".gz", urls)
        self.assertFalse((self.w.dir / "build/engine").exists())

    def test_n8_a_bad_hash_of_the_manifest_is_refused_with_the_source_build(self):
        needs = self.w.release / f"{ish.STEM}.needs"
        needs.write_text(needs.read_text() + "need libevil.so.1\n")
        r = self.w.install("--yes", "--source-build")
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertFalse(self.w.dir.exists())


class VenvPackageTest(unittest.TestCase):
    """A stock Ubuntu has python3 but not python3-venv. The installer must find that before setup does."""

    def world(self, family="ubuntu", venv_works=False):
        w = ish.Env(self)
        w.os_release.write_text(f"ID={family}\n")
        flag = w.t / "venv.ok"
        if venv_works:
            flag.write_text("")
        w.venv_flag = flag
        real = '"$(PATH="$SHIM_REAL_PATH" command -v python3)"'
        (w.shims / "python3").write_text(
            f'#!/bin/sh\nprintf "python3 %s\\n" "$*" >> "{w.log}"\n'
            f'case "$*" in "-m venv"*) [ -f "{flag}" ] || {{ echo "ensurepip is not available" >&2; exit 1; }} ;; esac\n'
            f'exec {real} "$@"\n')
        sudo = (w.shims / "sudo").read_text()
        (w.shims / "sudo").write_text(sudo.replace("exit 0\n", f'case "$*" in *python3-venv*) : > "{flag}" ;; esac\nexit 0\n'))
        return w

    def test_a_python3_that_cannot_make_a_venv_is_asked_once_on_debian(self):
        w = self.world()
        r = w.install("--yes", answers="y\n")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertEqual(out.count("(y/N)"), 1, out)
        self.assertEqual(w.calls("sudo"), ["sudo apt-get update", "sudo apt-get install -y python3-venv"])

    def test_a_python3_that_can_make_a_venv_changes_nothing(self):
        w = self.world(venv_works=True)
        r = w.install("--yes")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(w.calls("sudo"), [])

    def test_the_venv_package_joins_the_question_for_the_libraries(self):
        w = self.world()
        w.make_release(needs=["libnuma.so.1"])
        w.ld_hide.write_text("libnuma.so.1")
        r = w.install("--yes", answers="y\n", SHIM_INSTALL="libnuma.so.1")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertEqual(out.count("(y/N)"), 1, out)
        self.assertEqual([c for c in w.calls("sudo") if "install" in c], ["sudo apt-get install -y python3-venv", "sudo apt-get install -y libnuma1"])   # one question, one command per group

    def test_arch_and_fedora_ship_venv_with_python_so_no_question(self):
        for family in ("arch", "fedora"):
            with self.subTest(family):
                w = self.world(family)
                r = w.install("--yes")
                self.assertEqual(w.calls("sudo"), [], r.stdout + r.stderr)
                self.assertNotIn("(y/N)", r.stdout + r.stderr)

    def test_the_package_table_has_the_venv_key_for_debian_only(self):
        func = re.search(r"^pkg_for\(\) \{\n.*?^\}\n", ish.INSTALL.read_text(), re.S | re.M).group(0)
        def pkg(family):
            return subprocess.run(["bash", "-c", func + '\npkg_for "$1" tool-python3-venv', "x", family],
                                  capture_output=True, text=True).stdout.strip()
        self.assertEqual(pkg("debian"), "python3-venv")
        self.assertEqual((pkg("arch"), pkg("fedora")), ("", ""))


if __name__ == "__main__":
    unittest.main()
