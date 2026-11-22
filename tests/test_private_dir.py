"""Setup makes the state folder one name at a time with no link at any depth (review cache-folder-r4 L-A, host copy).

The engine copy is engine/runtime/state_private_dir.h (tests/state_private_dir_test.cpp). Before this fix,
Sandbox.mkdir_private used Path.mkdir(parents=True) and os.chmod by path: another user who made a missing parent a link
in a shared sticky folder, after the check and before the mkdir, got the mode change onto another folder of this user.
"""
import errno
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from crucible_cli import files  # noqa: E402


def mode(path):
    return stat.S_IMODE(os.lstat(path).st_mode)


class PrivateDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(os.path.realpath(tmp.name))
        os.chmod(self.t, 0o700)
        self.shared = self.t / "shared"  # a shared sticky folder, as /tmp
        self.shared.mkdir()
        os.chmod(self.shared, 0o1777)
        self.elsewhere = self.t / "elsewhere"  # another folder of this user, with the same last name inside
        (self.elsewhere / "states").mkdir(parents=True)
        os.chmod(self.elsewhere / "states", 0o755)
        (self.elsewhere / "states" / "keep.txt").write_text("x")

    def raced(self, path, plant):
        """A Sandbox whose check is followed at once by PLANT (the other user's swap after the check)."""
        box = files.Sandbox([path])
        check = box.check

        def check_then_swap(p):
            real = check(p)
            plant()
            return real

        box.check = check_then_swap
        return box

    def assert_elsewhere_kept(self):
        self.assertEqual(mode(self.elsewhere / "states"), 0o755)
        self.assertEqual((self.elsewhere / "states" / "keep.txt").read_text(), "x")

    def test_a_missing_parent_swapped_for_a_link_stops_setup(self):
        path = self.shared / "x" / "states"
        box = self.raced(path, lambda: (self.shared / "x").symlink_to(self.elsewhere))
        with self.assertRaisesRegex(Exception, r"/shared/x is a link, and Crucible does not follow a link"):
            box.mkdir_private(path)
        self.assert_elsewhere_kept()

    def test_a_deeper_link_stops_setup(self):
        path = self.shared / "x" / "y" / "states"

        def plant():
            (self.shared / "x").mkdir(mode=0o700)
            (self.shared / "x" / "y").symlink_to(self.elsewhere)

        box = self.raced(path, plant)
        with self.assertRaisesRegex(Exception, r"/shared/x/y is a link"):
            box.mkdir_private(path)
        self.assert_elsewhere_kept()

    def test_the_folder_swapped_for_a_link_stops_setup(self):
        path = self.shared / "states"
        box = self.raced(path, lambda: path.symlink_to(self.elsewhere / "states"))
        with self.assertRaisesRegex(Exception, r"/shared/states is a link"):
            box.mkdir_private(path)
        self.assert_elsewhere_kept()

    def test_a_dangling_link_target_is_not_made(self):
        path = self.shared / "states"
        box = self.raced(path, lambda: path.symlink_to(self.t / "target"))
        with self.assertRaisesRegex(Exception, r"is a link"):
            box.mkdir_private(path)
        self.assertFalse(os.path.lexists(self.t / "target"))

    def test_an_open_parent_without_the_sticky_bit_stops_setup(self):
        opened = self.t / "open"
        opened.mkdir()
        os.chmod(opened, 0o777)
        path = opened / "states"
        with self.assertRaisesRegex(Exception, r"other users can write .*/open \(mode 0777, no sticky bit\)"):
            files.Sandbox([path]).mkdir_private(path)
        self.assertFalse(path.exists())

    def test_a_file_in_the_path_stops_setup(self):
        (self.t / "file").write_text("x")
        path = self.t / "file" / "states"
        with self.assertRaisesRegex(Exception, r"/file is not a folder\."):
            files.Sandbox([path]).mkdir_private(path)

    def test_each_new_folder_gets_0700_whatever_the_umask(self):
        # 0277 leaves 0500 (no write bit); 0377 leaves 0400 (no search bit: the change goes through /proc/self/fd).
        for mask in (0o277, 0o377):
            path = self.shared / f"new{mask:o}" / "deeper" / "states"
            old = os.umask(mask)
            try:
                files.Sandbox([path]).mkdir_private(path)
            finally:
                os.umask(old)
            for folder in (path, path.parent, path.parent.parent):
                self.assertEqual(mode(folder), 0o700, folder)

    def test_an_own_folder_gets_0700_and_keeps_its_files(self):
        files.Sandbox([self.elsewhere / "states"]).mkdir_private(self.elsewhere / "states")
        self.assertEqual(mode(self.elsewhere / "states"), 0o700)
        self.assertEqual((self.elsewhere / "states" / "keep.txt").read_text(), "x")

    def test_a_parent_without_the_read_bit_does_not_stop_setup(self):
        # Review state-dir-race M1: a parent that this user can search but not read (as a root-owned /home at 0711).
        # Trunk needed only the search bit of a parent, so the walk must not need more.
        xonly = self.t / "xonly"
        (xonly / "old").mkdir(parents=True)
        os.chmod(xonly / "old", 0o300)
        os.chmod(xonly, 0o300)
        self.addCleanup(os.chmod, xonly, 0o700)
        path = xonly / "new" / "states"
        files.Sandbox([path]).mkdir_private(path)
        self.assertEqual(mode(xonly / "new"), 0o700)
        self.assertEqual(mode(path), 0o700)
        self.assertEqual(mode(xonly), 0o300)
        # The last folder of this user without the read bit gets 0700, as on trunk.
        files.Sandbox([xonly / "old"]).mkdir_private(xonly / "old")
        self.assertEqual(mode(xonly / "old"), 0o700)


class PrivateDirRuleTest(unittest.TestCase):
    """The walk with a fake user: a folder of another user is never changed; a parent of another user stops the walk."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(os.path.realpath(tmp.name))
        os.chmod(self.t, 0o700)
        from crucible_cli import private_dir
        self.walk = private_dir

    def test_a_folder_of_another_user_keeps_its_mode(self):
        path = self.t / "theirs"
        path.mkdir()
        os.chmod(path, 0o755)
        me, other = os.geteuid(), os.geteuid() + 1
        owners = self.walk.state_root.trusted_owners(me) | {other}
        with self.assertRaisesRegex(self.walk.Refused, r"belongs to another user \(uid %d\)" % me):
            self.walk.make_private_dir(path, uid=other, owners=owners)
        self.assertEqual(mode(path), 0o755)

    def test_a_parent_of_an_untrusted_owner_stops_the_walk(self):
        path = self.t / "a" / "states"
        with self.assertRaisesRegex(self.walk.Refused, r"belongs to another user, who could move or replace"):
            self.walk.make_private_dir(path, owners={0})
        self.assertFalse((self.t / "a").exists())

    def test_a_made_folder_of_another_user_stops_the_walk(self):
        # Review state-dir-race L1: another user made the name between the mkdir and the open (a fake uid stands in).
        path = self.t / "came" / "states"
        with self.assertRaisesRegex(self.walk.Refused, r"/came belongs to another user \(it came while Crucible made it\)"):
            self.walk.make_private_dir(path, uid=os.geteuid() + 1, owners=self.walk.state_root.trusted_owners(os.geteuid()))
        self.assertTrue((self.t / "came").exists())
        self.assertFalse(path.exists())

    def test_an_os_error_of_the_walk_is_refused(self):
        # Review state-dir-race L3: setup (steps.py) catches only Refused, so an OSError of the walk must become one.
        failure = OSError(errno.EIO, os.strerror(errno.EIO))
        with mock.patch.object(self.walk.os, "fstat", side_effect=failure):
            with self.assertRaisesRegex(self.walk.Refused, r"^/ cannot be read \(Input/output error\)\.$"):
                self.walk.make_private_dir(self.t / "states")
        real_open = os.open

        def open_but_not_root(name, *args, **kwargs):
            if name == "/":
                raise failure
            return real_open(name, *args, **kwargs)

        with mock.patch.object(self.walk.os, "open", side_effect=open_but_not_root):
            with self.assertRaisesRegex(self.walk.Refused, r"^/ cannot be opened \(Input/output error\)\.$"):
                self.walk.make_private_dir(self.t / "states")
        self.assertFalse((self.t / "states").exists())

    def test_without_make_a_missing_name_is_reported(self):
        with self.assertRaises(self.walk.Refused) as caught:
            self.walk.make_private_dir(self.t / "missing", make=False)
        self.assertTrue(caught.exception.missing)
        self.assertFalse((self.t / "missing").exists())

    def test_a_relative_path_and_a_dot_name_are_refused(self):
        with self.assertRaisesRegex(self.walk.Refused, "is not an absolute path"):
            self.walk.make_private_dir("states")
        with self.assertRaisesRegex(self.walk.Refused, r"has a '\.\.' name"):
            self.walk.make_private_dir(str(self.t) + "/../states")


if __name__ == "__main__":
    unittest.main()
