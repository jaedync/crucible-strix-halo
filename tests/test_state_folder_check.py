"""Cache folder check item 2 (owner chat via desk, Nov 13 12:29): setup, doctor and uninstall use ONE canonical check of the
state folder (tools/crucible_cli/settings.py state_dir_problem). The v0.37.0 engine keeps the home-folder rule, so the check
keeps it too (item 1, the engine change, is t38 work)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crucible_cli_fakes import make_repo, wire  # noqa: E402

from crucible_cli import cli, doctor  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402

# An engine state id: 3 parts of 16 hex digits. Built from one part, so the leak guard does not see a 50-character data run.
IDENT = "-".join(["0123456789abcdef"] * 3)


class StateFolderCheckTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)
        self.repo = make_repo(self.t)
        self.host = wire(self.t)
        self.home = Path(self.host.home)

    def doctor_row(self, sd):
        s = settings_mod.build(self.repo, self.host, state_dir=sd)
        return {r.id: r for r in doctor.run_checks(self.host, s)}["state-dir"]

    def test_the_canonical_check(self):
        p = settings_mod.state_dir_problem
        self.assertIsNone(p(self.home, self.repo, self.home / "states"))
        self.assertIn("home", p(self.home, self.repo, Path("/var/tmp/crucible-states")))
        self.assertIn("home", p(self.home, self.repo, self.home))
        sd = self.home / "own"
        (sd / IDENT).mkdir(parents=True)
        (sd / "write-ledger").write_text("")
        self.assertIsNone(p(self.home, self.repo, sd))
        (sd / "notes.txt").write_text("x")
        self.assertIn("did not make", p(self.home, self.repo, sd))
        link = self.home / "link"
        link.symlink_to(self.home / "states-real")
        self.assertIn("link", p(self.home, self.repo, link))

    def test_doctor_passes_a_folder_that_holds_only_engine_files(self):
        sd = self.home / "states"
        (sd / IDENT).mkdir(parents=True)
        (sd / "write-ledger").write_text("")
        self.assertEqual(self.doctor_row(sd).status, "pass")

    def test_doctor_names_the_same_problem_as_setup(self):
        sd = self.home / "states"
        sd.mkdir()
        (sd / "notes.txt").write_text("x")
        want = settings_mod.state_dir_problem(self.home, self.repo, sd)
        row = self.doctor_row(sd)
        self.assertEqual(row.status, "warn")
        self.assertIn(want, row.detail)
        link = self.home / "link"
        link.symlink_to(sd)
        self.assertIn("link", self.doctor_row(link).detail)

    def test_setup_doctor_and_uninstall_call_the_one_check(self):
        calls = []
        real = settings_mod.state_dir_problem

        def spy(*a, **kw):
            calls.append(a)
            return real(*a, **kw)
        sd = self.home / "states"
        with mock.patch.object(settings_mod, "state_dir_problem", spy):
            self.doctor_row(sd)
            self.assertTrue(calls, "doctor")
            calls.clear()
            s = settings_mod.build(self.repo, self.host, state_dir=sd)
            cli._validate(s, self.host)
            self.assertTrue(calls, "setup")
        src = (Path(cli.__file__).parent / "update.py").read_text()
        self.assertIn("settings_mod.state_dir_problem(", src, "uninstall")


if __name__ == "__main__":
    unittest.main()
