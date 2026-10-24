"""CPU checks of the switch registry engine/switches.h against the engine's reads."""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import switch_registry as reg

REGISTRY = 'engine/switches.h'
# Every function that takes a SwitchName. A literal name passed to one of them is a read of that switch.
READERS = r'\b(?:SwitchText|RowPolicyInt|EnvOn|Numerics(?:Env|Int|On|DefaultOn))\(\s*"([A-Z0-9_]+)"'


def sources():
    for directory in ('engine', 'kernels'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.relative_to(ROOT).as_posix() != REGISTRY:
                yield path.relative_to(ROOT).as_posix(), path.read_text(errors='replace')


def reads():
    return {name for _, text in sources() for name in re.findall(READERS, text)}


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.switches = reg.load()
        self.by_name = {s.name: s for s in self.switches}

    def test_every_table_line_parses(self):
        text = (ROOT / REGISTRY).read_text()
        self.assertEqual(len(self.switches), text.count('{"CRUCIBLE_'))
        self.assertEqual(len(self.by_name), len(self.switches))
        self.assertEqual([s.name for s in self.switches], sorted(self.by_name))

    def test_every_engine_name_is_registered(self):
        names = {n for _, text in sources() for n in re.findall(r'CRUCIBLE_[A-Z0-9_]+', text)}
        self.assertEqual(sorted(names - set(self.by_name)), [], f'add the switch to {REGISTRY}')
        self.assertEqual(sorted(set(self.by_name) - names), [], 'delete the switch that the engine no longer names')

    def test_every_read_is_a_live_switch_or_an_external_variable(self):
        allowed = {s.name for s in self.switches if s.kind != 'build'} | set(reg.external())
        self.assertEqual(sorted(reads() - allowed), [])

    def test_every_live_switch_is_read(self):
        live = {s.name for s in self.switches if s.kind != 'build'}
        self.assertEqual(sorted(live - reads()), [], 'a live switch with no read: delete it or mark it build')

    def test_build_macros_are_not_read(self):
        build = {s.name for s in self.switches if s.kind == 'build'}
        self.assertEqual(sorted(build & reads()), [])

    def test_unchecked_names_stay_in_tests(self):
        users = [path for path, text in sources() if 'Unchecked(' in text]
        self.assertEqual(users, [], 'SwitchName::Unchecked is for C++ test fixtures only')

    def test_columns(self):
        for s in self.switches:
            with self.subTest(name=s.name):
                self.assertTrue(s.default and s.module and s.note)
                self.assertGreaterEqual(s.issue, 0)
                if s.kind in ('exact-speed', 'layout'):
                    # At decide_by the switch becomes the code default or goes (at most two trains after its train:
                    # train 18 for the old rows, train 23 for the later re-send fixes).
                    self.assertTrue(18 <= s.decide_by <= (25 if s.issue == 276 else 20), s.decide_by)
                if s.kind == 'build':
                    self.assertFalse(s.serve or s.record)

    def test_defaults_match_the_code_where_the_code_states_them(self):
        # Cheap partial check: integer fallbacks of RowPolicyInt/NumericsInt, and opt-ins read by NumericsOn/EnvOn.
        text = '\n'.join(body for _, body in sources())
        checked = 0
        for name, value in re.findall(r'\b(?:RowPolicyInt|NumericsInt)\(\s*"(CRUCIBLE_[A-Z0-9_]+)",\s*(-?\d+)\s*,', text):
            if self.by_name[name].default.lstrip('-').isdigit():
                self.assertEqual(self.by_name[name].default, value, name)
                checked += 1
        for name in re.findall(r'\b(?:NumericsOn|EnvOn)\(\s*"(CRUCIBLE_[A-Z0-9_]+)"', text):
            self.assertEqual(self.by_name[name].default, 'off', name)
            checked += 1
        self.assertGreater(checked, 5)

    def test_launch_rows_default_is_the_code_default(self):
        # 8192-row launches are the default; CRUCIBLE_PF_LAUNCH_ROWS=4096 turns them off.
        code = (ROOT / 'engine/model/flashnext/prefill_launch.h').read_text()
        value = re.search(r'kDefaultPrefillLaunchRows\s*=\s*(\d+)', code)
        self.assertIsNotNone(value)
        row = self.by_name['CRUCIBLE_PF_LAUNCH_ROWS']
        join = re.search(r'kDefaultPrefillJoinRows\s*=\s*(\d+)', code)
        self.assertIsNotNone(join)
        # X4 (x4-tail): the default is the daily value "8192+128".
        self.assertEqual((row.kind, row.default, row.issue, row.decide_by), ('exact-speed', f'{value[1]}+{join[1]}', 24, 20))

    def test_counts(self):
        result = reg.counts(self.switches)
        self.assertEqual(result['names'], len(self.switches))
        self.assertEqual(result['live'] + result['build'], result['names'])


if __name__ == '__main__':
    unittest.main()
