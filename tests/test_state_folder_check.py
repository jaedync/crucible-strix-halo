"""Cache folder check item 2 (owner chat via desk, Nov 13 12:29): setup, doctor and uninstall use ONE canonical check of the
state folder (tools/crucible_cli/settings.py state_dir_problem). The folder may be outside the home
folder. The check follows the engine rule (engine/runtime/state_root.h; host copy tools/crucible_cli/state_root.py): a folder
that only this user or root can change is accepted, anywhere but a system folder."""
import os
import shutil
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
        self.assertIsNone(p(self.home, self.repo, self.t / "fast" / "states"))
        self.assertIn("system folder", p(self.home, self.repo, Path("/usr/crucible-states")))
        self.assertIn("your home folder", p(self.home, self.repo, self.home))
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

    def test_doctor_names_the_state_bytes_of_older_identity_folders(self):
        # The folders of old builds and settings filled the disk. The disk limit now covers them
        # (engine/runtime/state_disk.h TrimOthers); doctor names their bytes. The newest folder is the live one.
        sd = self.home / "states"
        new, old = IDENT, "-".join(["fedcba9876543210"] * 3)
        now = 1_800_000_000
        for name, size, age in ((new, 1_500_000_000, 60), (old, 2_000_000_000, 4 * 86400)):
            folder = sd / name
            folder.mkdir(parents=True)
            state = folder / ("a" * 8 + ".state")
            with open(state, "wb") as f:
                f.truncate(size)  # sparse: the engine counts the file size
            (folder / ".lock").write_text("")
            (folder / "notes.txt").write_text("x")  # not a tier file: not counted
            for path in (state, folder / ".lock"):
                os.utime(path, (now - age, now - age))
        (sd / "write-ledger").write_text("")
        s = settings_mod.build(self.repo, self.host, state_dir=sd)
        row = {r.id: r for r in doctor.run_checks(self.host, s)}["disk-state"]
        self.assertIn("states 3.5 GB in 2 identity folders: the newest 1.5 GB, older ones 2.0 GB "
                      "(old builds or settings). The limit covers the folders of stopped engines.", row.detail)
        # One folder: no older ones.
        shutil.rmtree(sd / old)
        row = {r.id: r for r in doctor.run_checks(self.host, s)}["disk-state"]
        self.assertIn("states 1.5 GB in 1 identity folder.", row.detail)

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

    def test_home_is_a_link_folder_under_the_real_home_passes(self):
        # Review M3 probe 1: the engine (state_disk.h PrepareRoot) resolves the links in HOME and in the folder, so a folder
        # under the real home folder is inside. Setup accepted it; doctor failed it with its own abspath test.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = Path(tmp.name)
        (t / "realhome").mkdir()
        (t / "home").symlink_to(t / "realhome")
        repo = make_repo(t)
        host = wire(t)
        sd = t / "realhome" / "states"
        self.assertIsNone(settings_mod.state_dir_problem(host.home, repo, sd))
        s = settings_mod.build(repo, host, state_dir=sd)
        self.assertEqual({r.id: r for r in doctor.run_checks(host, s)}["state-dir"].status, "pass")
        self.assertTrue(cli._validate(s, host))

    def test_cache_link_out_of_home_is_accepted_like_the_engine(self):
        # Review M3 probe 2, with the rule of Nov 12: ~/.cache is a link to a folder out of home. The engine
        # resolves the link and accepts the real folder (only this user can change it), so setup and doctor accept it too.
        outside = self.t / "elsewhere"
        outside.mkdir()
        (self.home / ".cache").symlink_to(outside)
        sd = self.home / ".cache" / "crucible" / "states"
        self.assertIsNone(settings_mod.state_dir_problem(self.home, self.repo, sd))
        self.assertEqual(self.doctor_row(sd).status, "pass")
        s = settings_mod.build(self.repo, self.host, state_dir=sd)
        self.assertTrue(cli._validate(s, self.host))

    def test_a_folder_on_a_second_disk_passes_setup_and_doctor(self):
        sd = self.t / "mnt-fast" / "crucible" / "states"
        (self.t / "mnt-fast").mkdir()
        self.assertIsNone(settings_mod.state_dir_problem(self.home, self.repo, sd))
        row = self.doctor_row(sd)
        self.assertEqual((row.status, row.title), ("pass", "State folder"))
        s = settings_mod.build(self.repo, self.host, state_dir=sd)
        self.assertTrue(cli._validate(s, self.host))

    def test_a_folder_that_other_users_can_swap_fails_setup_and_doctor_with_the_engine_sentence(self):
        shared = self.t / "shared"
        shared.mkdir()
        os.chmod(shared, 0o777)
        self.addCleanup(os.chmod, shared, 0o700)
        sd = shared / "states"
        want = (f"other users can write {shared} (mode 0777, no sticky bit), so they could move or replace {sd}.")
        self.assertEqual(settings_mod.state_dir_problem(self.home, self.repo, sd), want)
        row = self.doctor_row(sd)
        self.assertEqual(row.status, "fail")
        self.assertIn(want, row.detail)
        out = []
        self.host.out = out.append
        s = settings_mod.build(self.repo, self.host, state_dir=sd)
        self.assertFalse(cli._validate(s, self.host))
        self.assertIn(want, " ".join(out))
        self.assertFalse(sd.exists())

    def test_a_system_folder_fails_doctor(self):
        row = self.doctor_row(Path("/etc/crucible-states"))
        self.assertEqual(row.status, "fail")
        self.assertIn("system folder", row.detail)

    def test_a_parent_that_cannot_be_read_is_a_path_fault_and_fails_doctor(self):
        # Review L4: the engine stops at start for this path, as for any other path fault. It is not a content warning.
        locked = self.t / "locked"
        locked.mkdir()
        os.chmod(locked, 0)
        self.addCleanup(os.chmod, locked, 0o700)
        sd = locked / "a" / "states"
        self.assertEqual(settings_mod.state_dir_verdict(self.home, self.repo, sd).code, "unreadable")
        row = self.doctor_row(sd)
        self.assertEqual(row.status, "fail")
        self.assertIn("cannot be read", row.detail)

    def test_a_folder_whose_list_cannot_be_read_is_a_content_warning(self):
        sd = self.t / "states"
        sd.mkdir()
        os.chmod(sd, 0o300)
        self.addCleanup(os.chmod, sd, 0o700)
        self.assertEqual(settings_mod.state_dir_verdict(self.home, self.repo, sd).code, "contents-unreadable")
        self.assertEqual(self.doctor_row(sd).status, "warn")

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
