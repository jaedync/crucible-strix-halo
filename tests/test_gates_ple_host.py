"""CPU tests for the CRUCIBLE_PLE_HOST handling of the shared gate tool."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import batch_gate_report as batch
import gates as g


class PleHostEnvironmentTests(unittest.TestCase):
    def test_reference_capture_uses_the_resident_table(self):
        # The same binary as baseline then compares the host gather with the resident table.
        with patch.dict(g.os.environ, {'CRUCIBLE_PLE_HOST': '1'}):
            self.assertEqual(g.capture_environment(reference=True)['CRUCIBLE_PLE_HOST'], '0')
            self.assertEqual(g.capture_environment()['CRUCIBLE_PLE_HOST'], '1')

    def test_every_value_except_zero_gets_a_resident_reference(self):
        # The engine parses with strtol: '01', '+1' and ' 1' also turn the host gather on.
        for value in ('01', '+1', ' 1', '1 ', 'x'):
            with self.subTest(value=value), patch.dict(g.os.environ, {'CRUCIBLE_PLE_HOST': value}):
                self.assertEqual(g.capture_environment(reference=True)['CRUCIBLE_PLE_HOST'], '0')
                self.assertEqual(g.capture_environment()['CRUCIBLE_PLE_HOST'], value)
        with patch.dict(g.os.environ, {'CRUCIBLE_PLE_HOST': '0'}):
            self.assertEqual(g.capture_environment(reference=True)['CRUCIBLE_PLE_HOST'], '0')

    def test_an_unset_flag_stays_unset_in_the_reference(self):
        with patch.dict(g.os.environ):
            g.os.environ.pop('CRUCIBLE_PLE_HOST', None)
            self.assertNotIn('CRUCIBLE_PLE_HOST', g.capture_environment(reference=True))

    def test_report_records_the_raw_value(self):
        self.assertEqual(g.runtime_environment({'CRUCIBLE_PLE_HOST': '1'})['CRUCIBLE_PLE_HOST'], '1')
        self.assertEqual(g.runtime_environment({'CRUCIBLE_PLE_HOST': '+1'})['CRUCIBLE_PLE_HOST'], '+1')

    def test_reference_capture_cannot_take_the_bf16_table(self):
        # The capture sources stay pinned (CAPTURE_VERSION): a reference capture keeps CRUCIBLE_PLE_BF16 but
        # gets CRUCIBLE_PLE_HOST=0, and the engine then stops at load (PleBf16Path, tests/ple_bf16_test.cpp).
        # So a BF16 gate fails loudly; it never compares against a BF16 reference.
        env = {'CRUCIBLE_PLE_HOST': '1', 'CRUCIBLE_PLE_BF16': '/x/ngram_embedding_bf16.safetensors'}
        with patch.dict(g.os.environ, env):
            ref = g.capture_environment(reference=True)
            self.assertEqual((ref['CRUCIBLE_PLE_HOST'], ref['CRUCIBLE_PLE_BF16']), ('0', env['CRUCIBLE_PLE_BF16']))

    def test_reports_record_the_bf16_table(self):
        # A gate with the BF16 table must not pass as a default-path gate.
        self.assertIn('CRUCIBLE_PLE_BF16', g.RUNTIME_KNOBS)
        self.assertIn('CRUCIBLE_PLE_BF16', batch.RUNTIME_KNOBS)

    def test_reports_publish_the_bf16_name_and_size_not_the_path(self):
        # The path names the user's home, as the MTP sidecar path does.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ngram_embedding_bf16.safetensors'
            path.write_bytes(b'0123456789')
            for module in (g, batch):
                with self.subTest(module=module.__name__):
                    published = module.runtime_environment({'CRUCIBLE_PLE_BF16': str(path)})
                    self.assertEqual(published['CRUCIBLE_PLE_BF16'], 'ngram_embedding_bf16.safetensors (10 bytes)')
                    self.assertNotIn(directory, repr(published))
                    self.assertEqual(module.runtime_environment({'CRUCIBLE_PLE_BF16': directory})['CRUCIBLE_PLE_BF16'], 'set')
                    missing = module.runtime_environment({'CRUCIBLE_PLE_BF16': directory + '/gone'})
                    self.assertEqual(missing['CRUCIBLE_PLE_BF16'], 'set')
                    self.assertIsNone(module.runtime_environment({})['CRUCIBLE_PLE_BF16'])
                    self.assertEqual(module.runtime_environment({'CRUCIBLE_PLE_BF16': ''})['CRUCIBLE_PLE_BF16'], '')

    def test_cache_key_does_not_publish_the_bf16_path(self):
        import baseline_cache as bc
        self.assertNotIn('CRUCIBLE_PLE_BF16', g.CACHE_PUBLIC_KNOBS)
        path = '~/models/ngram_embedding_bf16.safetensors'
        key_env = bc.output_environment(g.cache_environment({'CRUCIBLE_PLE_BF16': path}), g.CACHE_PUBLIC_KNOBS)
        self.assertNotIn('someone', repr(key_env))
        self.assertTrue(key_env['CRUCIBLE_PLE_BF16'].startswith('sha256:'))

    def test_reports_record_the_trace_switch(self):
        self.assertIn('CRUCIBLE_PLE_TRACE', g.RUNTIME_KNOBS)
        self.assertIn('CRUCIBLE_PLE_TRACE', batch.RUNTIME_KNOBS)


class EngineDrainTests(unittest.TestCase):
    def test_engine_drains_the_host_gather_before_its_rows_and_buffer_go(self):
        # m_ owns the gather thread and is destroyed after the rows and the staging buffer,
        # so a guard declared after both must drain a started gather first.
        source = (ROOT / 'engine/engine.hip').read_text()
        guard = source.index('PleDrainGuard ple_drain_{&m_.ple.host};')
        for member in ('std::vector<int64_t> ple_host_rows_;', '> ple_stage_;', '  Model m_;'):
            self.assertLess(source.index(member), guard, member)


if __name__ == '__main__':
    unittest.main()
