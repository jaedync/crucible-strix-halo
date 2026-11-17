"""`update --check` is read-only, also as root (r373 security review M2).

cli.py lets root run `update --check` because it changes nothing. Before this fix, Update._locked ran recover() (the restore
of a cut update: it stops servers, swaps folders and runs pip) and clean_stale() before the --check test. As root, that left
root-owned files in the install. Update.run also refuses root for every changing run, as the other changing commands do."""
import unittest
from types import SimpleNamespace
from unittest import mock

from crucible_update_fakes import World

from crucible_cli import cli, update


class Crash(BaseException):
    pass


def _crash_at(name):
    def point(at):
        if at == name:
            raise Crash(at)
    return point


class UpdateCheckIsReadOnly(unittest.TestCase):
    def cut_update(self, w):
        """Leave a journal, as a kill during an update does."""
        with mock.patch.object(update, "crash_point", _crash_at("swapped")), self.assertRaises(Crash):
            cli.main(["update", "--yes"], host=w.host(), repo=w.dir)
        self.journal = w.parent / ".crucible.update.json"
        self.assertTrue(self.journal.exists())

    def test_root_update_check_does_not_restore_a_cut_update(self):
        w = World(self)
        self.cut_update(w)
        host = w.host()
        host.uid = 0
        with mock.patch.object(update.Update, "restore", side_effect=AssertionError("restore ran")), \
                mock.patch.object(update.Update, "clean_stale", side_effect=AssertionError("clean_stale ran")):
            code = cli.main(["update", "--check"], host=host, repo=w.dir)
        out = "\n".join(host.lines)
        self.assertNotEqual(code, 0, out)
        self.assertTrue(self.journal.exists(), "the check changed the journal")
        self.assertIn("An earlier update was stopped", out)
        self.assertIn("without sudo", out)

    def test_update_check_does_not_restore_or_clean_for_a_normal_user(self):
        w = World(self)
        self.cut_update(w)
        host = w.host()
        with mock.patch.object(update.Update, "restore", side_effect=AssertionError("restore ran")), \
                mock.patch.object(update.Update, "clean_stale", side_effect=AssertionError("clean_stale ran")):
            code = cli.main(["update", "--check"], host=host, repo=w.dir)
        out = "\n".join(host.lines)
        self.assertNotEqual(code, 0, out)
        self.assertTrue(self.journal.exists())
        self.assertIn("./crucible update", out.replace(str(w.dir) + "/crucible", "./crucible"))
        # A normal run (no --check) still restores.
        self.assertEqual(cli.main(["update", "--yes"], host=w.host(), repo=w.dir), 4)
        self.assertFalse(self.journal.exists())

    def test_update_check_with_no_journal_skips_clean_stale(self):
        w = World(self)
        stale = w.parent / ".crucible-update-old"
        stale.mkdir()
        host = w.host()
        code = cli.main(["update", "--check"], host=host, repo=w.dir)
        self.assertEqual(code, 0, "\n".join(host.lines))
        self.assertTrue(stale.exists(), "--check removed a work folder")

    def test_update_run_refuses_root_for_a_changing_run(self):
        w = World(self)
        host = w.host()
        host.uid = 0
        args = SimpleNamespace(yes=True, check=False, tag=None, source_build=False, rollback=False)
        with mock.patch.object(update.Update, "_locked", side_effect=AssertionError("ran as root")):
            code = update.run_update(host, w.dir, args, None)
        self.assertEqual(code, update.EXIT_USAGE)
        self.assertIn("without sudo", "\n".join(host.lines))


if __name__ == "__main__":
    unittest.main()
