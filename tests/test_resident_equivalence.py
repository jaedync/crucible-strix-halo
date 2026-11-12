import array
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import gates as g
import long_gates as lg
import resident_equivalence as eq


class EquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module in (g, lg):
            p = patch.object(module, 'VOCAB', 4); p.start(); self.addCleanup(p.stop)
        self.out = self.root/'nonce'; self.out.mkdir()
        self.start = time.time_ns(); time.sleep(0.05)  # a file's mtime comes from the coarse kernel clock, up to one tick behind time_ns()
        self.plan = {'nonce': 'nonce', 'root': str(self.out), 'tokens': 8,
                     'commands': [['resident'], ['cli0'], ['cli1'], ['cli2']],
                     'prompts': []}
        for i in range(3):
            path = self.root/f'p{i}.ids'; path.write_text(','.join(['1']*1024)+'\n')
            self.plan['prompts'].append({'file': str(path), 'ids': [1]*1024, 'sha256': g.digest(path)})
        self.emit()

    def emit(self):
        for i in range(3):
            for role in ('resident', 'cli'):
                target = self.out/f'{role}-{i}'
                if role == 'resident': target = target/'native'
                target.mkdir(parents=True)
                (target/'stdout.log').write_text('ids: 0 1 2 3 0 1 2 3\n')
                (target/'logits.f32').write_bytes(array.array('f', [2, 1, 0, -1]).tobytes())
        (self.out/'resident.stdout.log').write_text('{"event":"resident","requests":3,"bulk_rows":4096}\n')
        for i in range(4):
            (self.out/f'child-{i}.stderr.log').write_bytes(b'')
        receipts = [{'nonce': 'nonce', 'command': cmd, 'returncode': 0,
                     'started_ns': self.start, 'ended_ns': self.start}
                    for cmd in self.plan['commands']]
        g.save(self.out/'worker.json', {'nonce': 'nonce', 'children': receipts})

    def validate(self):
        return eq.validate(self.plan, self.start, time.time_ns())

    def test_complete_byte_and_id_comparison(self):
        self.assertEqual(len(self.validate()), 3)

    def test_false_pass_mutations(self):
        mutations = ('stale', 'empty', 'nan', 'zero', 'substituted_ids', 'input_ids',
                     'duplicate_ids_line', 'extra', 'missing', 'hardlink', 'symlink',
                     'unequal_bytes', 'aliased_dispatch', 'old_nonce', 'failed_child')
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                original = self.out
                self.out = Path(tmp)/'nonce'; self.out.mkdir()
                plan = self.plan
                self.plan = {**plan, 'root': str(self.out)}
                self.emit()
                logits = self.out/'cli-0/logits.f32'; stdout = self.out/'cli-0/stdout.log'
                if mutation == 'stale': os.utime(logits, ns=(1, 1))
                if mutation == 'empty': logits.write_bytes(b'')
                if mutation in ('nan', 'zero'):
                    for role in ('cli-0', 'resident-0/native'):
                        (self.out/role/'logits.f32').write_bytes(array.array('f', [float('nan')]*4 if mutation == 'nan' else [0]*4).tobytes())
                if mutation == 'substituted_ids': stdout.write_text('ids: 0 1 2 3 0 1 2 2\n')
                if mutation == 'input_ids':
                    for role in ('cli-0', 'resident-0/native'):
                        (self.out/role/'stdout.log').write_text('ids:'+' 1'*8+'\n')
                        (self.out/role/'logits.f32').write_bytes(array.array('f', [0, 2, 0, -1]).tobytes())
                if mutation == 'duplicate_ids_line': stdout.write_text(stdout.read_text()*2)
                if mutation == 'extra': (self.out/'extra').write_text('bad')
                if mutation == 'missing': logits.unlink()
                if mutation in ('hardlink', 'symlink'):
                    logits.unlink(); other = self.out/'resident-0/native/logits.f32'
                    if mutation == 'hardlink': os.link(other, logits)
                    else: logits.symlink_to(other)
                if mutation == 'unequal_bytes': logits.write_bytes(array.array('f', [2, 1, 0, -2]).tobytes())
                if mutation in ('aliased_dispatch', 'old_nonce', 'failed_child'):
                    path = self.out/'worker.json'; receipt = json.loads(path.read_text())
                    if mutation == 'aliased_dispatch': receipt['children'][1]['command'] = ['resident']
                    if mutation == 'old_nonce': receipt['nonce'] = 'earlier-run'
                    if mutation == 'failed_child': receipt['children'][0]['returncode'] = 124
                    g.save(path, receipt)
                with self.assertRaises((ValueError, OSError)): self.validate()
                self.out, self.plan = original, plan

    def test_frozen_manifest_not_reopened(self):
        manifest = self.root/'plan.json'; g.save(manifest, self.plan)
        frozen = eq.encode(self.plan)
        g.save(manifest, {**self.plan, 'tokens': 1})
        self.assertEqual(json.loads(frozen)['tokens'], 8)
        self.assertEqual(len(self.validate()), 3)

    def test_launch_manifest_uses_frozen_hash(self):
        manifest = self.root/'resident.tsv'; manifest.write_text('original')
        plan = {'launch_hashes': {str(manifest): g.digest(manifest)}}
        eq.unchanged_launch(plan)
        manifest.write_text('substituted after launch')
        with self.assertRaises(ValueError): eq.unchanged_launch(plan)

    def test_empty_import_is_hashed_sized_and_rechecked(self):
        import hashlib
        empty = self.root/'__init__.py'; empty.write_bytes(b'')
        module = SimpleNamespace(__file__=str(empty), __cached__=None)
        with patch.dict(sys.modules, {'eq_empty_import': module}):
            imported = eq.imports()
        self.assertIn(str(empty), imported)
        plan = {'trees': {}, 'inputs': [str(empty)], 'imports': imported}
        before = eq.snapshot(plan)
        self.assertEqual(before['files'][str(empty)], hashlib.sha256(b'').hexdigest())
        self.assertEqual(before['file_sizes'][str(empty)], 0)
        frozen = {**plan, 'launch_hashes': before['files']}
        eq.unchanged_launch(frozen)
        empty.write_bytes(b'changed')
        self.assertNotEqual(eq.snapshot(plan), before)
        with self.assertRaisesRegex(ValueError, 'launch input changed'):
            eq.unchanged_launch(frozen)
        empty.unlink()
        with self.assertRaises(OSError): eq.snapshot(plan)
        with self.assertRaises(OSError): eq.unchanged_launch(frozen)

    def test_empty_nonimport_input_remains_invalid(self):
        empty = self.root/'manifest.tsv'; empty.write_bytes(b'')
        with self.assertRaises(ValueError):
            eq.snapshot({'trees': {}, 'inputs': [str(empty)], 'imports': []})
        with self.assertRaises(ValueError):
            eq.unchanged_launch({'launch_hashes': {str(empty): g.digest(empty)}, 'imports': []})

    def test_snapshot_rejects_file_and_tree_mutations(self):
        source = self.root/'source'; source.mkdir(); (source/'a').write_text('original')
        before = eq.hash_tree(source)
        (source/'a').write_text('changed')
        self.assertNotEqual(eq.hash_tree(source), before)
        (source/'new').write_text('extra')
        self.assertNotEqual(eq.hash_tree(source), before)

    def test_prepare_complete_build_bindings_and_mutations(self):
        from build_resident_probe import RESIDENT_FILES, RESIDENT_ORIGIN
        source = self.root/'bound-source'
        for name in ('engine/engine.hip', 'engine/cli/gen_cli.h', 'engine/exl3/safetensors.h', 'kernels/k.h'):
            path = source/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        engine = self.root/'ordinary'; engine.write_bytes(b'ordinary binary')
        probe = self.root/'producer'; probe.write_bytes(b'resident binary')
        queue = self.root/'queue'; queue.write_bytes(b'queue')
        checkpoint = self.root/'checkpoint'; checkpoint.mkdir(); (checkpoint/'weights').write_text('CPU fake')
        engine_record = Path(str(engine)+'.build.json'); probe_record = Path(str(probe)+'.build.json')
        record = {'source_commit': 'c5dfa33', 'source_hashes': eq.builder.source_hashes(source),
                  'binary_sha256': g.digest(engine), 'command': ['/opt/rocm/bin/hipcc', '-O3', '-std=c++20',
                   '--offload-arch=gfx1151', '--rocm-path=/opt/rocm', '-I/opt/rocm/include',
                   '-o', str(engine), str(source/'engine/engine.hip'), '-lpthread']}
        g.save(engine_record, record)
        binding = {'capture_origin': RESIDENT_ORIGIN, 'binary_sha256': g.digest(probe),
                   'bound_engine_binary_sha256': g.digest(engine), 'source_hashes': record['source_hashes'],
                   'engine_build_record_sha256': g.digest(engine_record), 'inherited_compiler_flags': ['-O3'],
                   'bound_engine_source_path': str(source/'engine/engine.hip'),
                   'probe_hashes': {name: g.digest(g.ROOT/name) for name in RESIDENT_FILES}}
        args = SimpleNamespace(source=source, engine=engine, resident=probe, checkpoint=checkpoint, gpu_run=queue)
        for mutation in (None, 'wrong_source_commit', 'engine_hash', 'source_hash', 'probe_hash',
                         'bound_engine_hash', 'engine_record_hash', 'numeric_flags', 'source_path'):
            with self.subTest(mutation=mutation):
                current, bound = dict(record), dict(binding)
                if mutation == 'wrong_source_commit': current['source_commit'] = 'other'
                if mutation == 'engine_hash': current['binary_sha256'] = '0'*64
                if mutation == 'source_hash': current['source_hashes'] = {}
                if mutation == 'probe_hash': bound['binary_sha256'] = '0'*64
                if mutation == 'bound_engine_hash': bound['bound_engine_binary_sha256'] = '0'*64
                if mutation == 'engine_record_hash': bound['engine_build_record_sha256'] = '0'*64
                if mutation == 'numeric_flags': bound['inherited_compiler_flags'] = ['-Ofast']
                if mutation == 'source_path': bound['bound_engine_source_path'] = '/other/source'
                g.save(engine_record, current); g.save(probe_record, bound)
                target = self.root/f'prepared-{mutation}'; target.mkdir()
                with patch.object(g, 'VOCAB', 248320):
                    if mutation is not None:
                        with self.assertRaises((ValueError, KeyError)): eq.prepare(args, target)
                    else:
                        plan = eq.prepare(args, target)
                        self.assertEqual([len(p['ids']) for p in plan['prompts']], [1024, 4096, 16384])
                        self.assertEqual(len(plan['commands']), 4)
                        self.assertTrue(all(command.count('--spec') == 1 for command in plan['commands'][1:]))
                        self.assertEqual(plan['commands'][0][1:4], [str(checkpoint), '32768', '4096'])

    def test_worker_four_sequential_cpu_children_and_frozen_inputs(self):
        script = self.root/'cpu_child.py'
        script.write_text('import pathlib,sys,time\n'
                          'pathlib.Path(sys.argv[1]).write_text(str(time.time_ns()))\n')
        output = self.root/'worker-run'; output.mkdir()
        manifest = output/'resident.tsv'; manifest.write_text('frozen input')
        plan = {'root': str(output), 'nonce': output.name,
                'commands': [[sys.executable, str(script), str(output/f'cpu-{i}')] for i in range(4)],
                'launch_hashes': {str(manifest): g.digest(manifest), str(script): g.digest(script)}}
        done = subprocess.run([sys.executable, eq.__file__, '--worker'], input=eq.encode(plan),
                              capture_output=True, timeout=20)
        self.assertEqual(done.returncode, 0, done.stderr.decode())
        receipts = json.loads((output/'worker.json').read_text())['children']
        self.assertEqual(len(receipts), 4)
        self.assertTrue(all(a['ended_ns'] <= b['started_ns'] for a, b in zip(receipts, receipts[1:])))
        manifest.write_text('changed after launch')
        rejected = subprocess.run([sys.executable, eq.__file__, '--worker'], input=eq.encode(plan),
                                  capture_output=True, timeout=20)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn(b'launch input changed', rejected.stderr)

    def test_worker_cleans_child_process_groups(self):
        script = self.root/'spawn_cpu_child.py'
        script.write_text('import pathlib,subprocess,sys,time\n'
                          'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])\n'
                          'pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n'
                          'if len(sys.argv)>2: time.sleep(60)\n')
        for interruption in (False, True):
            with self.subTest(interruption=interruption):
                output = self.root/f'cleanup-{interruption}'; output.mkdir()
                manifest = output/'resident.tsv'; manifest.write_text('frozen')
                pids = [output/f'pid-{i}' for i in range(4)]
                commands = [[sys.executable, str(script), str(path)] for path in pids]
                if interruption: commands[0].append('wait')
                plan = {'root': str(output), 'nonce': output.name, 'commands': commands,
                        'launch_hashes': {str(manifest): g.digest(manifest), str(script): g.digest(script)}}
                process = subprocess.Popen([sys.executable, eq.__file__, '--worker'], stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    process.stdin.write(eq.encode(plan)); process.stdin.close(); process.stdin = None
                    deadline = time.monotonic()+5
                    while not pids[0].exists() and time.monotonic()<deadline: time.sleep(0.01)
                    self.assertTrue(pids[0].exists())
                    if interruption: process.terminate()
                    _, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 143 if interruption else 0, stderr.decode())
                    for path in pids:
                        if not path.exists(): continue
                        pid = int(path.read_text()); status = Path(f'/proc/{pid}/stat')
                        deadline = time.monotonic()+2
                        while status.exists() and status.read_text().split()[2] != 'Z' and time.monotonic()<deadline:
                            time.sleep(0.01)
                        self.assertTrue(not status.exists() or status.read_text().split()[2] == 'Z')
                finally:
                    if process.poll() is None: process.kill(); process.wait(timeout=10)
                    for path in pids:
                        if path.exists():
                            try: os.kill(int(path.read_text()), 9)
                            except ProcessLookupError: pass

    def test_parent_one_lease_and_mutations_fail_closed(self):
        for mutation in (None, 'source', 'checkpoint', 'manifest', 'plan', 'ids', 'noop', 'failed'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                original_out, original_plan, original_start = self.out, self.plan, self.start
                args = SimpleNamespace(output=Path(temp)/'runs', gpu_run=Path('/cpu-fake-queue'), priority=3)
                source = Path(temp)/'source'; source.mkdir(); (source/'input').write_text('source')
                checkpoint = Path(temp)/'checkpoint'; checkpoint.mkdir(); (checkpoint/'weights').write_text('fake')
                def prepare(_args, root):
                    self.out = root
                    self.plan = {**original_plan, 'root': str(root), 'nonce': root.name,
                                 'trees': {'source': str(source), 'checkpoint': str(checkpoint),
                                           'tools': str(source), 'prompts': str(source)},
                                 'inputs': [str(root/'resident.tsv')]}
                    (root/'resident.tsv').write_text('frozen')
                    return self.plan
                def queued(command, **kwargs):
                    self.assertEqual(command[:10], ['/cpu-fake-queue', '-p', '3', '-w', '7200', '--', 'timeout', '-k', '10', '300'])
                    self.assertEqual(command.count('/cpu-fake-queue'), 1)
                    frozen = json.loads(kwargs['input'])
                    self.assertEqual(frozen['tokens'], 8)
                    if mutation != 'noop':
                        self.start = time.time_ns(); time.sleep(0.05); self.emit()  # see setUp: coarse mtime vs time_ns()
                        receipt_path = self.out/'worker.json'; receipt = json.loads(receipt_path.read_text())
                        receipt['nonce'] = self.out.name
                        for child in receipt['children']: child['nonce'] = self.out.name
                        g.save(receipt_path, receipt)
                        (self.out/'plan.json').read_bytes()
                    if mutation == 'source': (source/'input').write_text('modified')
                    if mutation == 'checkpoint': (checkpoint/'weights').write_text('modified')
                    if mutation == 'manifest': (self.out/'resident.tsv').write_text('modified')
                    if mutation == 'plan': g.save(self.out/'plan.json', {**frozen, 'tokens': 1})
                    if mutation == 'ids': (self.out/'cli-0/stdout.log').write_text('ids: 0 1 2 3 0 1 2 2\n')
                    return SimpleNamespace(returncode=1 if mutation == 'failed' else 0)
                with patch.object(eq, 'prepare', side_effect=prepare), patch.object(eq.subprocess, 'run', side_effect=queued) as execute:
                    code = eq.run(args)
                self.assertEqual(execute.call_count, 1)
                report = json.loads((self.out/'report.json').read_text())
                self.assertEqual(report['status'], 'PASS' if mutation is None else 'FAIL')
                self.assertEqual(code, 0 if mutation is None else 1)
                self.out, self.plan, self.start = original_out, original_plan, original_start

    def test_dispatch_identity_rejects_self_baseline_and_aliases(self):
        a = self.root/'a'; b = self.root/'b'; a.write_bytes(b'A'); b.write_bytes(b'B')
        eq.distinct(a, b)
        for kind in ('same', 'same_bytes', 'hardlink', 'symlink'):
            with self.subTest(kind=kind):
                if b.exists() or b.is_symlink(): b.unlink()
                if kind == 'same_bytes': b.write_bytes(b'A')
                if kind == 'hardlink': os.link(a, b)
                if kind == 'symlink': b.symlink_to(a)
                with self.assertRaises((ValueError, OSError)): eq.distinct(a, a if kind == 'same' else b)


if __name__ == '__main__': unittest.main()
