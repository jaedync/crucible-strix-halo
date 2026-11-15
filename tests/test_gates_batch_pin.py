"""CPU regression: gates.py pins the candidate, then the batch supplement checks the dual-build record.
The record names the built path in its -o field; the pinned copy has the same sha but another path."""
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import gates as g
from tools import batch_gate_report as b
import tempfile
import tests.test_batch_gate_report as report_tests


class Stop(Exception):
    """Raised at the probe's GPU ticket: the build checks before it are what this test covers."""


class BatchPinTest(unittest.TestCase):
    # The batch report tests' fixture: frozen source, dual-build records, probe and prompts.
    prepare_run, populate = report_tests.BatchReportTest.prepare_run, report_tests.BatchReportTest.populate

    def setUp(self):
        report_tests.BatchReportTest.setUp(self)
        preflight = patch.object(g, 'disk_preflight', return_value=0)
        preflight.start()
        self.addCleanup(preflight.stop)
        self.payload = b'\x00\x00\x80\x3f' * g.WIDTH

    def gate_arguments(self, kwargs):
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir()
        for file in ('config.json', 'quantization_config.json', 'model.safetensors.index.json'):
            (checkpoint / file).write_text('{}')
        baseline = self.root / 'baseline-engine'
        baseline.write_bytes(b'baseline-binary')
        return SimpleNamespace(output=self.root / 'gate', engine=kwargs['engine'], checkpoint=checkpoint,
                               ids=kwargs['primary_ids'], baseline_name='named-main', baseline_engine=baseline,
                               oracle_baseline=None, tokens=3, prefill_rows=0, gpu_run=Path('/queue'), priority=3,
                               engine_build_record=kwargs['build_record'], batch_gates_binary=kwargs['probe'],
                               batch_build_record=kwargs['probe_build_record'], batch_prompt_list=kwargs['prompts'])

    def engine_process(self, command, **kwargs):
        if len(command) == 1:
            return SimpleNamespace(returncode=2, stdout='--draft-head full|64k', stderr='')
        time.sleep(0.003)
        def option(flag): return command[command.index(flag) + 1]
        dump = Path(option('--dump'))
        n, spec = int(option('--n')), int(option('--spec'))
        prompt = len(g.parse_ids(Path(option('--ids-file')).read_text()))
        for name in g.NAMES: (dump / name).write_bytes(self.payload * (prompt + (n - 1 if spec == 0 else 0)))
        kwargs['stdout'].write(('ids:' + ' 7' * n + '\n').encode())
        return SimpleNamespace(returncode=0)

    def run_gate(self, mutate_record=None):
        self.root = Path(tempfile.mkdtemp(dir=self.temp.name))
        kwargs, binding = self.prepare_run()
        for path in (kwargs['engine'], kwargs['probe']):
            path.chmod(0o755)
        if mutate_record:
            record = json.loads(kwargs['build_record'].read_text())
            mutate_record(record)
            kwargs['build_record'].write_text(json.dumps(record))
        args = self.gate_arguments(kwargs)
        probes = []
        # gates.py (subprocess.run) and batch_gate_report (proc_tree.run_tree): one fake serves both.
        def process(command, **options):
            if command[-1] == '--build-info':
                probes.append(command)
                return subprocess.CompletedProcess(command, 0, json.dumps(binding), '')
            if len(command) == 1 or '--dump' in command:
                return self.engine_process(command, **options)
            raise Stop(command)
        with patch.dict(sys.modules, {'batch_gate_report': b}), patch.object(subprocess, 'run', side_effect=process), \
                patch.object(b, 'run_tree', side_effect=process), \
                self.assertRaises(Stop):
            g.run(args)
        return args, probes

    def test_pinned_candidate_passes_the_dual_build_command_check(self):
        args, probes = self.run_gate()
        report = json.loads((args.output / 'report.json').read_text())
        self.assertEqual(report['steps'], {'gate1': True, 'gate2': True, 'gate3': True})
        # The probe ran its --build-info check after validate_builds: the command check passed.
        self.assertEqual(len(probes), 1)
        self.assertEqual(report['pinned_binaries']['candidate']['path'], str((args.output / 'bin/candidate').resolve()))

    def test_command_check_stays_strict_for_the_candidate(self):
        for mutate in (lambda r: r['command'].__setitem__(r['command'].index('-o') + 1, '/elsewhere/crucible'),
                       lambda r: r['command'].append('-ffast-math'),
                       lambda r: r['command'].__setitem__(-2, '/other/engine.hip')):
            with self.subTest(mutate=mutate):
                with self.assertRaisesRegex(ValueError, 'unqualified flags or sources'):
                    self.run_gate(mutate)



if __name__ == '__main__':
    unittest.main()
