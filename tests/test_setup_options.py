"""tools/setup_options.py: the option table of the setup parser, and the record of a pinned release."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import setup_options  # noqa: E402

TOOL = ROOT / "tools/setup_options.py"


class TableTest(unittest.TestCase):
    def test_the_table_names_each_option_of_setup_with_its_kind(self):
        options = setup_options.table()
        for name, kind in (("--pack", "choice:3.05,4.05,4.18"), ("--streams", "int:1-8"), ("--yes", "flag"), ("-y", "flag"),
                           ("--port", "int"), ("--bind", "value"), ("--state-write-gib-day", "int:0-65536")):
            self.assertEqual(options[name], kind, name)

    def test_the_cli_prints_the_table_of_the_tree(self):
        r = subprocess.run([sys.executable, "-I", str(TOOL), "table"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(setup_options.parse(r.stdout), setup_options.table())


class RecordTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)
        self.install = self.t / "install.sh"
        self.install.write_text('#!/usr/bin/env bash\nPIN_TAG="v0.1.0"\nSETUP_OPTIONS="\n--old flag\n"\necho done\n')

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", str(TOOL), "record", *args, "--install", str(self.install),
                               "--out", str(self.t / "rec")], capture_output=True, text=True, timeout=60)

    def test_record_writes_the_file_and_the_install_block_and_keeps_the_pin(self):
        r = self.run_tool("v9.9.9")
        self.assertEqual(r.returncode, 0, r.stderr)
        text = self.install.read_text()
        self.assertEqual(setup_options.install_table(text), setup_options.table())
        self.assertEqual(setup_options.parse((self.t / "rec/v9.9.9.txt").read_text()), setup_options.table())
        self.assertEqual(setup_options.pin_tag(text), "v0.1.0")
        self.assertTrue(text.startswith('#!/usr/bin/env bash\nPIN_TAG="v0.1.0"\nSETUP_OPTIONS="\n'))
        self.assertTrue(text.endswith('"\necho done\n'))

    def test_record_refuses_a_bad_tag_and_changes_nothing(self):
        before = self.install.read_text()
        r = self.run_tool("0.38")
        self.assertEqual(r.returncode, 1)
        self.assertIn("the tag must look like", r.stderr)
        self.assertEqual(self.install.read_text(), before)
        self.assertFalse((self.t / "rec").exists())

    def test_record_refuses_an_install_file_without_the_block(self):
        self.install.write_text('PIN_TAG="v0.1.0"\n')
        r = self.run_tool("v9.9.9")
        self.assertEqual(r.returncode, 1)
        self.assertIn("SETUP_OPTIONS", r.stderr)
        self.assertFalse((self.t / "rec").exists())

    def test_a_failed_install_write_keeps_the_record_file_as_it_was(self):
        # Both files or none: a record file of the new table next to the old install block fails the pin test.
        (self.t / "rec").mkdir()
        (self.t / "rec/v9.9.9.txt").write_text("--old flag\n")
        before = self.install.read_text()
        self.install.chmod(0o444)
        r = self.run_tool("v9.9.9")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("install.sh", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual((self.t / "rec/v9.9.9.txt").read_text(), "--old flag\n")
        self.assertEqual(self.install.read_text(), before)

    def test_a_failed_install_write_leaves_no_new_record_file(self):
        self.install.chmod(0o444)
        r = self.run_tool("v9.9.9")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertFalse((self.t / "rec").exists())


class PinnedRecordTest(unittest.TestCase):
    def test_the_v0373_record_has_no_option_that_came_after_that_release(self):
        options = setup_options.parse((ROOT / "tests/data/setup_options/v0.37.3.txt").read_text())
        for name in ("--ngram16", "--abliterated", "--state-write-gib-day", "--download-anyway", "--no-disk-check"):
            self.assertNotIn(name, options)
        for name in ("--pack", "--bind", "--api-key", "--no-api-key", "--allowed-host", "--allow-rocm10"):
            self.assertIn(name, options)


if __name__ == "__main__":
    unittest.main()
