"""CPU checks of tools/state_epoch.py: the state epoch that the build stamps into the engine binary."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import state_epoch as se

PREFIX = b'STATE_EPOCH_SLOT_V1:'
HEX64 = 'ab' * 32


def slot_file(path, extra=b'', count=1, digits=b'0' * 64):
    path.write_bytes(b'\x7fELF-ish head\0' + (PREFIX + digits + b'\0' + extra) * count + b'tail')
    return path


class StripTests(unittest.TestCase):
    def test_comments_and_blank_space_do_not_count(self):
        a = 'int f() {\n  return 1;  // one\n}\n'
        b = '/* a block\n comment */\nint   f()  {\n\n\n    return 1;\n}\n\n'
        self.assertEqual(se.strip_source(a), se.strip_source(b))

    def test_code_counts(self):
        self.assertNotEqual(se.strip_source('int f() { return 1; }'), se.strip_source('int f() { return 2; }'))

    def test_comment_marks_inside_strings_stay(self):
        text = 'const char* a = "// not a comment"; /* gone */ int b;'
        stripped = se.strip_source(text)
        self.assertIn('"// not a comment"', stripped)
        self.assertNotIn('gone', stripped)
        self.assertIn("'\"'", se.strip_source("char q = '\"'; // x\nint z;"))
        self.assertIn('int z;', se.strip_source("char q = '\"'; // x\nint z;"))

    def test_escaped_quote_in_string(self):
        self.assertIn('"a\\" // b"', se.strip_source('x = "a\\" // b"; // c'))

    def test_open_block_comment_is_an_error(self):
        with self.assertRaises(ValueError):
            se.strip_source('int a; /* never closed')


class SourcePartTests(unittest.TestCase):
    def make_root(self, tmp):
        root = Path(tmp)
        (root / 'engine/runtime').mkdir(parents=True)
        (root / 'engine/runtime/a.h').write_text('int a;\n')
        (root / 'engine/runtime/b.h').write_text('int b;\n')
        sources = root / 'sources.txt'
        sources.write_text('# note\nengine/runtime/a.h\n\nengine/runtime/b.h\n')
        return root, sources

    def test_source_hash_follows_code_not_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, sources = self.make_root(tmp)
            base = se.source_hash(root, sources)
            (root / 'engine/runtime/a.h').write_text('// new note\nint  a;\n')
            self.assertEqual(se.source_hash(root, sources), base)
            (root / 'engine/runtime/a.h').write_text('int a2;\n')
            self.assertNotEqual(se.source_hash(root, sources), base)

    def test_a_missing_listed_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, sources = self.make_root(tmp)
            sources.write_text(sources.read_text() + 'engine/runtime/gone.h\n')
            with self.assertRaises(FileNotFoundError):
                se.source_hash(root, sources)

    def test_an_empty_list_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, sources = self.make_root(tmp)
            sources.write_text('# nothing\n')
            with self.assertRaises(ValueError):
                se.source_hash(root, sources)

    def test_the_real_list_names_existing_files(self):
        listed = se.source_list(ROOT / 'tools/state_epoch_sources.txt')
        self.assertGreaterEqual(len(listed), 5)
        for name in listed:
            self.assertTrue((ROOT / name).is_file(), name)
        self.assertIn('engine/runtime/state_format.h', listed)
        self.assertIn('engine/runtime/state_disk.h', listed)
        # SequenceState::PendBytes takes its pending GDN rows from spec_span.h (SpecPendRows).
        self.assertIn('engine/runtime/spec_span.h', listed)


class RegistryPartTests(unittest.TestCase):
    ROWS = ('  {{"CRUCIBLE_A", K::kNumerics, "{a}", "layers", 0, 0, SR, "{note}"}},\n'
            '  {{"CRUCIBLE_B", K::kExactSpeed, "{b}", "layers", 0, 0, SR, "speed"}},\n'
            '  {{"CRUCIBLE_C", K::kLayout, "x", "layers", 0, 0, N, "layout"}},\n')

    def header(self, tmp, a='off', b='0', note='n', extra=''):
        path = Path(tmp) / 'switches.h'
        path.write_text('inline constexpr SwitchInfo kSwitches[] = {\n' + self.ROWS.format(a=a, b=b, note=note) + extra +
                        '};\ninline constexpr const char* kExternalVariables[] = {\n};\n')
        return path

    def test_state_kinds_count_and_speed_kinds_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = se.registry_hash(self.header(tmp))
            self.assertEqual(se.registry_hash(self.header(tmp, b='9')), base)
            self.assertEqual(se.registry_hash(self.header(tmp, note='other words')), base)
            self.assertNotEqual(se.registry_hash(self.header(tmp, a='on')), base)
            extra = '  {"CRUCIBLE_D", K::kNumerics, "off", "layers", 0, 0, SR, "new"},\n'
            self.assertNotEqual(se.registry_hash(self.header(tmp, extra=extra)), base)

    def test_the_real_registry_gives_a_hash(self):
        self.assertRegex(se.registry_hash(ROOT / 'engine/switches.h'), '^[0-9a-f]{64}$')


class EpochTests(unittest.TestCase):
    def test_every_input_counts(self):
        base = se.combine('s' * 64, 'r' * 64, 'i' * 64)
        self.assertRegex(base, '^[0-9a-f]{64}$')
        self.assertEqual(base, se.combine('s' * 64, 'r' * 64, 'i' * 64))
        self.assertNotEqual(base, se.combine('t' * 64, 'r' * 64, 'i' * 64))
        self.assertNotEqual(base, se.combine('s' * 64, 't' * 64, 'i' * 64))
        self.assertNotEqual(base, se.combine('s' * 64, 'r' * 64, 't' * 64))
        # The parts do not slide into each other.
        self.assertNotEqual(se.combine('ab', 'c', 'd'), se.combine('a', 'bc', 'd'))

    def test_a_pin_replaces_the_epoch_and_nothing_else_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            pins = Path(tmp) / 'pins.tsv'
            computed, adopted = 'c' * 64, 'd' * 64
            pins.write_text(f'# note\n{computed}\t{adopted}\tgate run night 2026-11-03\n')
            self.assertEqual(se.apply_pins(computed, pins), (adopted, True))
            self.assertEqual(se.apply_pins('e' * 64, pins), ('e' * 64, False))

    def test_a_bad_pin_row_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            pins = Path(tmp) / 'pins.tsv'
            for row in ('short\t' + 'd' * 64 + '\tevidence', 'c' * 64 + '\t' + 'd' * 64,
                        'c' * 64 + '\t' + 'd' * 64 + '\t', 'c' * 64 + '\t' + '0' * 64 + '\tevidence'):
                pins.write_text(row + '\n')
                with self.assertRaises(ValueError, msg=row):
                    se.apply_pins('e' * 64, pins)

    def test_the_real_pin_file_parses(self):
        self.assertEqual(se.apply_pins('e' * 64, ROOT / 'tools/state_epoch_pins.tsv'), ('e' * 64, False))


class StampTests(unittest.TestCase):
    def test_stamp_patches_the_slot_and_keeps_the_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = slot_file(Path(tmp) / 'bin')
            size = binary.stat().st_size
            self.assertEqual(se.read_slot(binary), None)
            se.write_slot(binary, HEX64)
            self.assertEqual(binary.stat().st_size, size)
            self.assertEqual(se.read_slot(binary), HEX64)
            data = binary.read_bytes()
            self.assertTrue(data.startswith(b'\x7fELF-ish head\0' + PREFIX + HEX64.encode() + b'\0'))
            self.assertTrue(data.endswith(b'tail'))

    def test_stamping_again_replaces_the_digits(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = slot_file(Path(tmp) / 'bin')
            se.write_slot(binary, HEX64)
            se.write_slot(binary, 'cd' * 32)
            self.assertEqual(se.read_slot(binary), 'cd' * 32)

    def test_no_slot_or_two_slots_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            none = Path(tmp) / 'none'
            none.write_bytes(b'nothing here')
            with self.assertRaises(ValueError):
                se.write_slot(none, HEX64)
            two = slot_file(Path(tmp) / 'two', count=2)
            with self.assertRaises(ValueError):
                se.write_slot(two, HEX64)

    def test_bad_digits_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = slot_file(Path(tmp) / 'bin')
            for bad in ('0' * 64, 'AB' * 32, 'ab' * 31, 'zz' * 32, ''):
                with self.assertRaises(ValueError, msg=bad):
                    se.write_slot(binary, bad)

    def test_the_prefix_and_size_match_the_header(self):
        header = (ROOT / 'engine/runtime/state_epoch.h').read_text()
        self.assertIn(f'"{PREFIX.decode()}' + '0' * 64 + '"', header)
        self.assertEqual(se.PREFIX, PREFIX)
        self.assertIn('kStateEpochPrefix = 20, kStateEpochDigits = 64', header)

    def test_the_cli_stamps_with_given_machine_code_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = slot_file(Path(tmp) / 'bin')
            args = [sys.executable, '-I', str(ROOT / 'tools/state_epoch.py'), 'stamp', str(binary), '--isa-hash', 'a' * 64]
            first = subprocess.run(args, capture_output=True, text=True, check=True).stdout
            self.assertRegex(first, r'epoch=[0-9a-f]{64}')
            epoch = first.split('epoch=')[1].split()[0]
            self.assertEqual(se.read_slot(binary), epoch)
            again = subprocess.run(args, capture_output=True, text=True, check=True).stdout
            self.assertEqual(again, first)
            other = subprocess.run(args[:-1] + ['b' * 64], capture_output=True, text=True, check=True).stdout
            self.assertNotEqual(other.split('epoch=')[1].split()[0], epoch)

    def test_the_engine_reads_what_the_tool_wrote(self):
        cxx = os.environ.get('CXX') or shutil.which('g++')
        if not cxx:
            self.skipTest('no C++ compiler')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'probe.cpp'
            source.write_text(textwrap.dedent(f'''
                #include "{ROOT}/engine/runtime/state_epoch.h"
                #include <cstdio>
                int main() {{
                  bool stamped = true;
                  const std::string epoch = crucible::StateEpoch(std::string(64, 'b'), &stamped);
                  std::printf("%s %d\\n", epoch.c_str(), int(stamped));
                }}
            '''))
            binary = Path(tmp) / 'probe'
            subprocess.run([cxx, '-O2', '-std=c++20', str(source), '-o', str(binary)], check=True)
            before = subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout.split()
            self.assertEqual(before, ['b' * 64, '0'])
            se.write_slot(binary, HEX64)
            after = subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout.split()
            self.assertEqual(after, [HEX64, '1'])
            self.assertEqual(se.read_slot(binary), HEX64)


if __name__ == '__main__':
    unittest.main()
