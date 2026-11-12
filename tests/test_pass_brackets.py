"""Every decode pass that runs the trunk must bracket it with the lookahead.

ForwardBatch (the server's serial and batched decode) did not call lookahead_.Start for months: serial decode
had neither the weight prefetch nor the pacer. The scan reads names: in each decode
file, the text before each RunTrunk( call (back to the previous RunTrunk( call) must hold each Start call.
Prefill runs its own trunk (runtime/prefill_engine.inc) and is not scanned.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("engine/engine.hip", "engine/runtime/batch_forward.h", "engine/runtime/batch_spec_forward.h")
STARTS = ("lookahead_.Start(",)


class PassBracketTests(unittest.TestCase):
    def test_each_trunk_pass_starts_the_lookahead(self):
        passes = 0
        for name in FILES:
            text = (ROOT / name).read_text()
            previous = 0
            for match in re.finditer(r"\bRunTrunk\(", text):
                segment = text[previous:match.start()]
                for start in STARTS:
                    self.assertTrue(start in segment, f"{name}: a RunTrunk pass without {start}")
                previous = match.end()
                passes += 1
        self.assertEqual(passes, 3, "the decode trunk passes are Forward, ForwardBatch and ForwardVerifySpans")


if __name__ == "__main__":
    unittest.main()
