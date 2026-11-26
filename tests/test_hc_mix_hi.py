"""CRUCIBLE_HC_MIX_HI reaches both HC arithmetic classes of the engine.

The engine has two HC arithmetic classes: bulk prefill (R > 8 rows, prefill_hc.h) and the WMMA read of hc_wmma2.h
(decode, verify, chunked prefill and the MTP last row; one arithmetic for every R). The cache never restores decode
rows into a bulk request, and verify keeps the decode arithmetic, so each class must change in
the same way when the switch is on. The VALU read is refused with the switch (hc_mix_hi_test.cpp). The device bit
rules are in tests/hc_mix_hi_gpu_test.hip.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gates  # noqa: E402

NAME = "CRUCIBLE_HC_MIX_HI"


def engine_sources():
    for path in sorted((ROOT / "engine").rglob("*")):
        if path.suffix in (".h", ".hip", ".cpp") and path.is_file():
            yield path.relative_to(ROOT).as_posix(), path.read_text()


class HcMixHiTests(unittest.TestCase):
    def test_registry_row_is_a_default_on_numerics_switch_that_serve_passes(self):
        # A landed option-B default (the CRUCIBLE_HC_WMMA precedent): registry default 1, so the state epoch
        # (tools/state_epoch.py registry part) changes and states of the old default do not restore.
        text = (ROOT / "engine/switches.h").read_text()
        self.assertRegex(text, r'\{"%s", K::kNumerics, "1", "layers", \d+, 0, SR, ' % NAME)
        self.assertIn(NAME, gates.NUMERICS_SWITCHES)
        self.assertIn(NAME, gates.RUNTIME_KNOBS)

    def test_engine_reads_it_once_and_refuses_the_valu_read(self):
        engine = (ROOT / "engine/engine.hip").read_text()
        # The default follows the HC read: on with WMMA, off with VALU; read once, inside the start error handler.
        self.assertEqual(len(re.findall(r"\bHcMixHiFromEnv\(", engine)), 1)
        self.assertRegex(engine, r"try \{ hc_mix_hi_ = HcMixHiFromEnv\(hc_wmma_\); \}")
        mix = (ROOT / "engine/layers/hc_mix_hi.h").read_text()
        self.assertRegex(mix, r'NumericsDefaultInt\("%s", hc_wmma \? 1 : 0, 0, 0, 1\)' % NAME)
        self.assertNotRegex(mix, r'NumericsOn\("%s"' % NAME)
        self.assertIn("HcMixHiConflict(hc_mix_hi_, hc_wmma_)", engine)
        reads = [rel for rel, text in engine_sources() if '"%s"' % NAME in text and rel != "engine/switches.h"]
        self.assertEqual(reads, ["engine/layers/hc_mix_hi.h"])

    def test_bulk_mix_launches_go_through_the_one_launcher(self):
        for rel, text in engine_sources():
            if rel == "engine/layers/prefill_hc_hi.h":
                continue
            self.assertNotRegex(text, r"PrefillHcMixWideKernel<[^>]*>\s*<<<", rel)
        engine = (ROOT / "engine/engine.hip").read_text()
        calls = re.findall(r"LaunchPrefillHcMix\(([^,]+),", engine)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {"hc_mix_hi_"})

    def test_bulk_mid_pack_goes_through_the_one_launcher(self):
        # With the switch on, the mid pack writes no low half of t (the hi mix does not read it).
        for rel, text in engine_sources():
            if rel == "engine/layers/prefill_hc_hi.h":
                continue
            self.assertNotRegex(text, r"PrefillHcMidPackKernel(<[^>]*>)?\s*<<<", rel)
        engine = (ROOT / "engine/engine.hip").read_text()
        calls = re.findall(r"LaunchPrefillHcMidPack\(([^,]+),", engine)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {"hc_mix_hi_"})
        hi = (ROOT / "engine/layers/prefill_hc_hi.h").read_text()
        self.assertRegex(hi, r"PrefillHcMidPackKernel<false><<<[^;]*nullptr")

    def test_every_wmma_read_takes_the_engine_scratch_that_carries_the_flag(self):
        sites = 0
        for rel, text in engine_sources():
            if rel.startswith("engine/layers/hc_wmma"):
                continue
            for call in re.findall(r"HcW2Launch\(([^;]*)\);", text):
                sites += 1
                self.assertIn("hcw_", call, rel)
        self.assertGreaterEqual(sites, 3)
        engine = (ROOT / "engine/engine.hip").read_text()
        self.assertRegex(engine, r"hcw_ = HcW2Alloc\(\);\s*\n\s*hcw_\.hi_only = hc_mix_hi_;")
        for header in ("engine/layers/hc_wmma.h", "engine/layers/hc_wmma2.h"):
            self.assertIn("sc.hi_only", (ROOT / header).read_text(), header)

    def test_the_constructor_sets_the_flag_before_any_use(self):
        # hc_mix_hi_ is a mutable member (default false). Bulk prefill reads the member, the WMMA read reads the copy in
        # hcw_. If the constructor copies the member into hcw_ before it reads the switch, bulk prefill gets the
        # high-half mix and the WMMA read gets both halves: a mixed state that the deterministic gates do not see.
        engine = (ROOT / "engine/engine.hip").read_text()
        code = re.sub(r"//[^\n]*", "", engine)  # comments do not count as uses
        start = code.index("  Engine(const std::string& dir,")
        uses = [m.start() for m in re.finditer(r"\bhc_mix_hi_\b", code)]
        decl = [m.start(1) for m in re.finditer(r"\bbool (hc_mix_hi_) = false;", code)]
        self.assertEqual(len(decl), 1)
        setter = [m.start() for m in re.finditer(r"\bhc_mix_hi_ = HcMixHiFromEnv\(", code)]
        self.assertEqual(len(setter), 1)
        # No other write to the member anywhere in the engine.
        writes = [m.start() for m in re.finditer(r"\bhc_mix_hi_\s*=[^=]", code)]
        self.assertEqual(sorted(w for w in writes if w not in decl), sorted(setter))
        self.assertTrue(all(u > start for u in uses if u not in decl))
        # From the constructor on, the first use is the set; the copy into hcw_ comes after it.
        first = min(u for u in uses if u > start)
        self.assertEqual(first, setter[0], "the first use of hc_mix_hi_ in the constructor is not the switch read")
        copy = code.index("hcw_.hi_only = hc_mix_hi_;")
        self.assertLess(setter[0], copy)
        self.assertNotRegex(code, r"hcw_\.hi_only\s*=(?! hc_mix_hi_;)")

    def test_start_log_line_keeps_the_night_unit_string(self):
        # The binary_has check of a night unit compares whole `strings` lines: the first part is its own literal.
        engine = (ROOT / "engine/engine.hip").read_text()
        self.assertIn('"hc mix: fp16 high half of t only (CRUCIBLE_HC_MIX_HI default on, every HC read; =0 turns it off)")',
                      engine)
        self.assertIn('"%s; the VALU final HC of a bulk prefill chunk keeps both halves\\n"', engine)


if __name__ == "__main__":
    unittest.main()
