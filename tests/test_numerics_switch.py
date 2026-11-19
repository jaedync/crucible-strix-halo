"""CPU tests for the master numerics switch CRUCIBLE_EXACT."""
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import gates as g
import batch_gate_report as bgr

READERS = r'(?:std::getenv|getenv|SwitchText|RowPolicyInt|EnvOn|EnvFlag)'

def numerics_env_reads(text):
    # NumericsEnv returns null when the switch is unset or CRUCIBLE_EXACT=1, and the raw text otherwise. A truthiness
    # test turns =0 into "on", and a parser (std::string, atoi, strtol) reads a null pointer. No code outside
    # numerics.h may call it: use NumericsOn, NumericsInt or NumericsDefaultOn.
    return [text[max(0, m.start() - 12):m.end() + 30].replace('\n', ' ') for m in re.finditer(r'NumericsEnv\(', text)]


def engine_sources():
    # bench/ builds engine binaries too (gen, pfbench and the batch tools include engine headers).
    for directory in ('engine', 'kernels', 'bench'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.h', '.hip', '.inc', '.cpp'):
                yield path, path.read_text(errors='replace')


class EngineReadsTests(unittest.TestCase):
    def test_numerics_switches_are_read_only_through_the_numerics_mode(self):
        for name in g.NUMERICS_SWITCHES:
            direct = [str(path.relative_to(ROOT)) for path, text in engine_sources()
                      if re.search(READERS + r'\(\s*"' + name + '"', text)]
            self.assertEqual(direct, [], f'{name} must be read through Numerics*()')
            guarded = [path for path, text in engine_sources()
                       if re.search(r'Numerics(?:Env|Int|On|DefaultOn|DefaultInt)\(\s*"' + name + '"', text)]
            self.assertTrue(guarded, f'{name} has no Numerics*() read')

    def test_opt_in_switches_read_zero_as_off(self):
        # NumericsEnv returns the raw text, so any truthiness test on it (null check, `if (p)`, `p && ...`, a local
        # pointer) turns CRUCIBLE_X=0 into "on", and a parser reads null. An opt-in switch reads through NumericsOn
        # (only "1" is on); nothing outside numerics.h calls NumericsEnv.
        users = [f'{path.relative_to(ROOT)}: {m}' for path, text in engine_sources() if path.name != 'numerics.h'
                 for m in numerics_env_reads(text)]
        self.assertEqual(users, [], 'read opt-in switches with NumericsOn()')

    def test_the_scan_finds_every_kind_of_truthiness_read(self):
        for bad in ('if (NumericsEnv("CRUCIBLE_X")) f();', 'if (!NumericsEnv("CRUCIBLE_X")) f();',
                    'if (NumericsEnv("CRUCIBLE_X") != nullptr) f();', 'bool on = NumericsEnv("CRUCIBLE_X");',
                    'const char* v = NumericsEnv("CRUCIBLE_X"); if (v) f();',
                    'return NumericsEnv("CRUCIBLE_X") && g();', 'auto on = NumericsEnv("CRUCIBLE_X") ? 1 : 0;'):
            self.assertTrue(numerics_env_reads(bad), bad)
        for bad in ('int n = std::atoi(NumericsEnv("CRUCIBLE_X"));', 'std::string(NumericsEnv("CRUCIBLE_X")) == "1"'):
            self.assertTrue(numerics_env_reads(bad), bad)
        self.assertEqual(numerics_env_reads('bool on = NumericsOn("CRUCIBLE_X");'), [])

    def test_the_startup_check_covers_every_opt_in_switch(self):
        # ValidateNumericsSwitches walks the registry rows of kind kNumerics with default "off". Those rows must be
        # exactly the switches that engine code reads through NumericsOn, and the entry point must call it first.
        registry = set(re.findall(r'\{"(CRUCIBLE_[A-Z0-9_]+)", K::kNumerics, "off"', (ROOT / 'engine/switches.h').read_text()))
        read = set()
        for _, text in engine_sources():
            read |= set(re.findall(r'NumericsOn\(\s*"(CRUCIBLE_[A-Z0-9_]+)"', text))
        self.assertEqual(sorted(registry), sorted(read))
        main = (ROOT / 'engine/cli/gen_cli.h').read_text()
        self.assertLess(main.index('ValidateNumericsSwitches()'), main.index('TryBatchedServeCommand(argc, argv)'))

    def test_every_numerics_read_is_classified(self):
        names = set()
        for _, text in engine_sources():
            names |= set(re.findall(r'Numerics(?:Env|Int|On|DefaultOn|DefaultInt)\(\s*"(CRUCIBLE_[A-Z0-9_]+)"', text))
        self.assertEqual(sorted(names), sorted(g.NUMERICS_SWITCHES))

    def test_the_master_switch_is_not_a_numerics_switch(self):
        self.assertNotIn('CRUCIBLE_EXACT', g.NUMERICS_SWITCHES)


class GateEnvironmentTests(unittest.TestCase):
    def test_exact_reference_runs_the_stock_baseline(self):
        on = {name: '1' for name in g.NUMERICS_SWITCHES}
        with patch.dict(g.os.environ, {**on, 'CRUCIBLE_EXACT': '1', 'CRUCIBLE_PLE_HOST': '1', 'CRUCIBLE_FUSE': '2'}):
            reference = g.capture_environment(reference=True)
            candidate = g.capture_environment()
        self.assertFalse(set(on) & set(reference))
        self.assertNotIn('CRUCIBLE_EXACT', reference)
        # Exact switches still reach both sides; the host PLE gather keeps its own rule (resident-table reference).
        self.assertEqual(reference['CRUCIBLE_FUSE'], '2')
        self.assertEqual(reference['CRUCIBLE_PLE_HOST'], '0')
        self.assertEqual(candidate['CRUCIBLE_PLE_HOST'], '1')
        self.assertEqual({k: candidate[k] for k in on}, on)
        self.assertEqual(candidate['CRUCIBLE_EXACT'], '1')

    def test_without_the_master_switch_both_sides_keep_the_switches(self):
        with patch.dict(g.os.environ, {'CRUCIBLE_PF_EXACT_SPLIT': '1', 'CRUCIBLE_EXACT': '0'}):
            reference = g.capture_environment(reference=True)
        self.assertEqual(reference['CRUCIBLE_PF_EXACT_SPLIT'], '1')
        self.assertEqual(reference['CRUCIBLE_EXACT'], '0')

    def test_reports_record_the_mode(self):
        for knobs in (g.RUNTIME_KNOBS, bgr.RUNTIME_KNOBS):
            self.assertIn('CRUCIBLE_EXACT', knobs)
            self.assertTrue(set(g.NUMERICS_SWITCHES) <= set(knobs))


if __name__ == '__main__':
    unittest.main()
