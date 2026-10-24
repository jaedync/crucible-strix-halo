import importlib.util
from pathlib import Path
import unittest
import tempfile
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('split2048_probe', ROOT / 'tools/split2048_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class SplitProbeTest(unittest.TestCase):
    def test_core_cuts(self):
        self.assertEqual(probe.schedule(4096, 0, 2048), [2048, 2048])
        self.assertEqual(probe.schedule(20480, 0, 18432), [4096] * 4 + [2048, 2048])
        self.assertEqual(probe.schedule(10000, 5000, 2048), [2048, 2048, 904, 4096, 904])

    def test_reject_short_piece_and_non_core_cut(self):
        for end in range(2049, 2057):
            with self.assertRaises(ValueError):
                probe.schedule(end, 0, 2048)
        for cut in (21, 64, 4096, 5000, 6144):
            with self.assertRaises(ValueError):
                probe.schedule(7000, 5000, cut)
        self.assertEqual(probe.schedule(2057, 0, 2048), [2048, 9])

    def test_preserve_template_core_origin(self):
        self.assertEqual(probe.schedule(13000, 5000, 7048), [4096, 904, 2048, 2048, 3904])
        with self.assertRaises(ValueError):
            probe.schedule(13000, 5000, 6144)

    def test_capture_inventory_and_bits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('whole', 'split'):
                (root / name).mkdir()
                for filename, data in [('state.bin', b'\x01' * 16),
                                       ('logits.f32', b'\x00\x00\x80\x3f' * 4),
                                       ('ids.i32', (1).to_bytes(4, 'little') * 8),
                                       ('state.json', b'{"length":4096,"mpos":255,"have_carry":true,"next_id":1,"bytes":16}')]:
                    (root / name / filename).write_bytes(data)
            self.assertTrue(probe.compare_case(root, vocabulary=4)['equal'])
            shutil.copytree(root / 'whole', root / 'restored')
            with self.assertRaises(ValueError):
                probe.compare_case(root, vocabulary=4, modes=('whole', 'split', 'restored'), expected_cut=2048)
            (root / 'restored/restore.json').write_text('{"length":2048,"bytes":16}')
            self.assertTrue(probe.compare_case(root, vocabulary=4,
                            modes=('whole', 'split', 'restored'), expected_cut=2048)['equal'])
            with self.assertRaises(ValueError):
                probe.compare_case(root, vocabulary=4, modes=('whole', 'split', 'restored'), expected_cut=6144)
            (root / 'split/logits.f32').write_bytes(b'\x00\x00\x80\xbf' * 4)
            self.assertFalse(probe.compare_case(root, vocabulary=4)['equal'])
            (root / 'split/extra').touch()
            with self.assertRaises(ValueError):
                probe.compare_case(root, vocabulary=4)

    def test_cpp_cut_validation(self):
        source = (ROOT / 'bench/split2048.hip').read_text()
        body = source[source.index('std::vector<int> Schedule('):source.index('// The batch-serve schedules')]
        program = ('#include <vector>\n#include <algorithm>\n#include <stdexcept>\n'
                   'constexpr int kPrefillCoreRows=2048, kMaxRows=8;\n' + body + '''
int main() {
  if (Schedule(4096,2048) != std::vector<int>{2048,2048}) return 1;
  for (int cut : {21,64,4096,6144}) {
    try { Schedule(4096,cut); return 2; } catch (const std::runtime_error&) {}
  }
  for (int length=2049; length<=2056; ++length) {
    try { Schedule(length,2048); return 3; } catch (const std::runtime_error&) {}
  }
}
''')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'check.cpp'
            path.write_text(program)
            binary = Path(temp) / 'check'
            subprocess.run(['g++', '-std=c++20', str(path), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_pre_mode_schedules(self):
        # An earlier change: the Pre-cut mode takes its schedules from bench/split2048_schedule.h and refuses a
        # prompt without a Pre point or a schedule pair that differs by more than the cut.
        source = (ROOT / 'bench/split2048.hip').read_text()
        body = source[source.index('struct PreCase {'):source.index('void Capture(')]
        program = ('#include "%s"\n#include <stdexcept>\n'
                   'constexpr int kPrefillCoreRows=2048, kMaxRows=8;\n'
                   'namespace crucible { inline constexpr size_t kPreEndMinTokens = 32768; }\n'
                   % (ROOT / 'bench/split2048_schedule.h') + body + '''
int main() {
  const PreCase c = PreSchedules(39964, 4096, 8192);
  if (c.pre != 38912 || c.whole != std::vector<int>{8192,8192,8192,8192,7196}) return 1;
  if (c.split != std::vector<int>{8192,8192,8192,8192,4096,2048,1052}) return 2;
  if (PreSchedules(34825, 4096, 8192).split.back() != 9) return 3;
  for (int length : {34824, 32000}) {
    try { PreSchedules(length, 4096, 8192); return 4; } catch (const std::runtime_error&) {}
  }
}
''')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'check.cpp'
            path.write_text(program)
            binary = Path(temp) / 'check'
            subprocess.run(['g++', '-std=c++20', str(path), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_merged_launch_rows_are_allowed(self):
        # The Pre-cut mode runs the serve schedule, whose launches merge two grid chunks (8,192 rows).
        adapted = probe.adapt_prompt((ROOT / 'engine/runtime/prompt.h').read_text())
        self.assertIn('R > EngineLaunchRows(eng)', adapted)
        self.assertNotIn('R > step', adapted)

    def test_fail_closed_source_transform(self):
        source = (ROOT / 'engine/runtime/prompt.h').read_text()
        adapted = probe.adapt_prompt(source)
        self.assertIn('RunSplitPrompt', adapted)
        self.assertIn('schedule.at(call++)', adapted)
        with self.assertRaises(ValueError):
            probe.adapt_prompt(source.replace('int R = ConversationRows', 'int N = ConversationRows'))


if __name__ == '__main__':
    unittest.main()
