"""Review L3 (rel-delta-review, Nov 12): the program strings that give the source build time name the compiler.

The 16 minutes is the split build with the ROCm 7.2.4 compiler; a plain build with ROCm 10.1 takes about 30 minutes.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("install.sh", "tools/crucible_cli/steps.py", "tools/crucible_cli/update.py", "tools/release_make.sh")


class BuildMinutesTextTest(unittest.TestCase):
    def test_each_16_minutes_names_the_724_compiler(self):
        seen = 0
        for rel in FILES:
            for n, line in enumerate((ROOT / rel).read_text().splitlines(), 1):
                if "16 minutes" in line:
                    seen += 1
                    self.assertRegex(line, r"16 minutes with the ROCm 7\.2\.4 compiler", f"{rel}:{n}")
        self.assertEqual(seen, 5, "the five strings of review L3")


if __name__ == "__main__":
    unittest.main()
