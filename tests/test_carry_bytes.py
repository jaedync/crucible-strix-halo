"""An earlier change: the carry tail of a slot checkpoint (kCarryBytes, engine/serve/batched_serve.h) must hold
exactly one MTP carry (BatchMtpSlots::Carry). batched_serve.h compiles only on the HIP host, so this test reads the
source: one constant sizes both, and a static_assert ties them where kCarryBytes is defined."""
from pathlib import Path
import re
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "engine/serve/batched_serve.h").read_text(encoding="utf-8")


class CarryBytesTest(unittest.TestCase):
    def test_one_constant_sizes_the_carry_and_its_tail(self):
        slots = SOURCE[SOURCE.index("struct BatchMtpSlots {"):SOURCE.index("};", SOURCE.index("struct BatchMtpSlots {"))]
        self.assertIn("static constexpr size_t kCarryFloats = size_t(kWide);", slots)
        self.assertRegex(SOURCE, r"carry_\[slot\] = Zeros<float>\(kCarryFloats\);")
        self.assertIsNone(re.search(r"carry_\[slot\] = Zeros<float>\(kWide\)", SOURCE))
        self.assertIn("static constexpr size_t kCarryBytes = BatchMtpSlots::kCarryFloats * sizeof(float);", SOURCE)
        self.assertIn("static_assert(kCarryBytes == BatchMtpSlots::kCarryFloats * sizeof(float)", SOURCE)


if __name__ == "__main__":
    unittest.main()
