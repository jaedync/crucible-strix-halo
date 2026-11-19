"""Tests of tools/release_pin.py: trust root Part A. After a release, one commit on main sets PIN_TAG and PIN_SUMS_SHA256 in
install.sh to the new release and the sha256 of its SHA256SUMS. CPU only; it changes only the copy in a temp folder."""
import hashlib
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import release_pin  # noqa: E402

# The tag that install.sh pins now: the pin step after a release pins that tag again or a newer one.
TAG = re.search(r'^PIN_TAG="(v[0-9.]+)"$', (ROOT / "install.sh").read_text(), re.M).group(1)
STEM = f"crucible-engine-{TAG}-gfx1151-rocm7.2-linux-x86_64"


class ReleasePinTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.t = Path(tmp.name)
        self.install = self.t / "install.sh"
        self.install.write_text((ROOT / "install.sh").read_text())
        self.sums = self.t / "SHA256SUMS"
        self.sums.write_text(f"{'1' * 64}  {STEM}.gz\n{'2' * 64}  {STEM}.needs\n")
        self.sha = hashlib.sha256(self.sums.read_bytes()).hexdigest()

    def lines(self, path):
        return path.read_text().splitlines()

    def test_it_sets_the_tag_and_the_pin_and_changes_no_other_line(self):
        before = self.lines(self.install)
        subject = release_pin.pin(self.install, TAG, self.sums)
        after = self.lines(self.install)
        self.assertIn(f'PIN_TAG="{TAG}"', after)
        self.assertIn(f'PIN_SUMS_SHA256="{self.sha}"', after)
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        self.assertEqual(len(before), len(after))
        self.assertLessEqual(len(changed), 2, changed)
        self.assertEqual(subject, f"release: pin SHA256SUMS of {TAG} in install.sh")
        self.assertEqual(subprocess.run(["bash", "-n", str(self.install)]).returncode, 0)

    def test_the_sums_file_must_list_the_assets_of_that_tag(self):
        self.sums.write_text(f"{'1' * 64}  crucible-engine-v0.37.0-gfx1151-rocm7.2-linux-x86_64.gz\n")
        before = self.install.read_text()
        with self.assertRaises(release_pin.PinError):
            release_pin.pin(self.install, TAG, self.sums)
        self.assertEqual(self.install.read_text(), before)

    def test_a_tag_lower_than_the_pinned_tag_is_refused(self):
        with self.assertRaises(release_pin.PinError):
            release_pin.pin(self.install, "v0.1.0", self.sums)

    def test_a_bad_tag_name_is_refused(self):
        for bad in ("0.37.2", "v0.37", "v0.37.2-rc1", "v0.37.2\nX=1"):
            with self.assertRaises(release_pin.PinError, msg=bad):
                release_pin.pin(self.install, bad, self.sums)

    def test_an_install_sh_without_the_two_lines_is_refused(self):
        self.install.write_text("#!/bin/bash\necho hi\n")
        with self.assertRaises(release_pin.PinError):
            release_pin.pin(self.install, TAG, self.sums)

    def test_the_command_line_prints_the_commit_subject(self):
        r = subprocess.run([sys.executable, "-I", str(ROOT / "tools/release_pin.py"), str(self.install), TAG, str(self.sums)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"release: pin SHA256SUMS of {TAG} in install.sh", r.stdout)
        r = subprocess.run([sys.executable, "-I", str(ROOT / "tools/release_pin.py"), str(self.install), "v0.0.1",
                            str(self.sums)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()
