"""CPU checks of tools/state_equality_gate.py with a fake server (a Python process that mimics `python -m server`).

The real gate runs on the test machine inside a GPU lease (the state-equality night unit). These tests run its whole
flow on the CPU: start servers, turn 1, stop, compare the saved states, restore, compare with a cold turn.
The fake engine file holds `KEY=value` lines that make the fake server misbehave in one way.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tests'))
import state_equality_gate as gate

FAKE_SERVER = r'''
import hashlib, http.server, json, os, signal, sys, threading
sys.path.insert(0, %(tests)r)
from test_state_digest import state_file, block_file

argv = sys.argv[1:]
def arg(name):
    return argv[argv.index(name) + 1] if name in argv else None
engine = open(arg('--engine')).read()
knobs = dict(line.split('=', 1) for line in engine.split() if '=' in line)
port, root = int(arg('--port')), arg('--state-dir')
env = {k: v for k, v in os.environ.items() if k.startswith('CRUCIBLE_')}
open(os.path.join(os.path.dirname(os.path.dirname(root.rstrip('/'))), 'env-' + str(port) + '-' + knobs.get('ID', 'x') + '.json'), 'w').write(json.dumps(sorted(env)))
open(os.path.join(os.path.dirname(os.path.dirname(root.rstrip('/'))), 'pid-' + knobs.get('ID', 'x') + '.txt'), 'w').write(str(os.getpid()))
if knobs.get('FAIL_START'):
    sys.exit(3)
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
epoch = knobs.get('EPOCH')
identity = epoch if epoch else sha(engine)
folder = '-'.join([sha('dir' + identity)[:16]] * 3)
os.makedirs(os.path.join(root, folder), mode=0o700, exist_ok=True)
fdir = os.path.join(root, folder)
served = []
stored = None
for name in os.listdir(fdir):
    if name.endswith('.key'):
        stored = json.load(open(os.path.join(fdir, name)))
print('state_disk epoch source=%%s epoch=%%s' %% ('stamped' if epoch else 'binary', identity[:16]), flush=True)
print('state_disk start entries=%%d' %% (1 if stored else 0), flush=True)

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if knobs.get('HANG'):
            import time; time.sleep(120)
        messages = body['messages']
        cached = 0
        if body.get('cache_control') != 'bypass' and stored and messages[:len(stored)] == stored:
            cached = 1000 if knobs.get('RESTORE', '1') == '1' else 0
        tag = knobs.get('ANSWER', 'a')
        if cached and knobs.get('RESTORE_WRONG'):
            tag += '-restored'
        if any(k.endswith('FUSE') for k in env) and knobs.get('BAD_FUSE'):
            tag += '-fuse'
        answer = 'answer-%%s-%%s' %% (tag, sha(json.dumps(messages))[:12])
        if not cached and body.get('cache_control') != 'bypass':
            served.append(messages)
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        for piece in (answer[:6], answer[6:]):
            self.wfile.write(('data: ' + json.dumps({'choices': [{'delta': {'content': piece}}]}) + '\n\n').encode())
        usage = {'prompt_tokens': 5000, 'prompt_tokens_details': {'cached_tokens': cached}}
        self.wfile.write(('data: ' + json.dumps({'choices': [], 'usage': usage}) + '\n\n').encode())
        self.wfile.write(b'data: [DONE]\n\n')
        if cached:
            print('state_disk read result=ok', flush=True); print('state_disk restore tokens=%%d' %% cached, flush=True)

server = http.server.ThreadingHTTPServer(('127.0.0.1', port), H)
def stop(*_):
    threading.Thread(target=server.shutdown).start()
signal.signal(signal.SIGTERM, stop)
print('Crucible is ready at http://127.0.0.1:%%d' %% port, flush=True)
server.serve_forever()
if served and not knobs.get('NO_WRITE'):
    tag = knobs.get('STATE', 's') + (os.urandom(4).hex() if knobs.get('NOISE') else '')
    if any(k.endswith('FUSE') for k in env) and knobs.get('BAD_FUSE'):
        tag += 'fuse'
    name = sha(identity + 'state')
    open(os.path.join(fdir, name + '.state'), 'wb').write(state_file(identity, rest=tag.encode() * 40))
    open(os.path.join(fdir, name + '.block'), 'wb').write(block_file(identity))
    json.dump(served[0], open(os.path.join(fdir, name + '.key'), 'w'))
    print('state_disk spill result=written', flush=True)
'''


class World(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tree = self.root / 'tree'
        (self.tree / 'server').mkdir(parents=True)
        (self.tree / 'server/__init__.py').write_text('')
        (self.tree / 'server/__main__.py').write_text(FAKE_SERVER % {'tests': str(ROOT / 'tests')})
        self.out = self.root / 'out'
        self.base_port = 18000 + (os.getpid() % 1000) * 5

    def engine(self, name, **knobs):
        path = self.root / name
        path.write_text('ENGINE\n' + ''.join(f'{k}={v}\n' for k, v in knobs.items()))
        return path

    def config(self, ref, cand, **kw):
        values = dict(pack='4.05', ref_bin=ref, cand_bin=cand, ckpt=self.root / 'ckpt', layout='--streams 1 --ctx 4096',
                      env='CRUCIBLE_PLE_HOST=1 CRUCIBLE_EXACT_RESUME=1', neutral_env='CRUCIBLE_FUSE=0 CRUCIBLE_PF_PRUNE=0',
                      tree=self.tree, python=sys.executable, out=self.out, tokens=2000, port=self.base_port,
                      ready_timeout_s=20, stop_timeout_s=20, request_timeout_s=30)
        values.update(kw)
        return gate.Config(**values)

    def run_gate(self, ref_knobs=None, cand_knobs=None, **kw):
        ref = self.engine('ref', ID='ref', **(ref_knobs or {}))
        cand = self.engine('cand', ID='cand', **(cand_knobs or {}))
        return gate.run_pack(self.config(ref, cand, **kw))


class FlowTests(World):
    def test_equal_engines_pass(self):
        result = self.run_gate()
        self.assertEqual((result.verdict, result.reasons), ('PASS', []), result.notes)
        pack = self.out / '4.05'
        for name in ('ref-turn1.json', 'cand-turn1.json', 'neutral-turn1.json', 'aa-turn1.json', 'restore-turn2-disk.json',
                     'restore-turn2-cold.json', 'result.json', 'compare-ref-cand.txt'):
            self.assertTrue((pack / name).exists(), name)
        self.assertEqual(json.loads((pack / 'result.json').read_text())['verdict'], 'PASS')
        self.assertTrue(any('unstamped' in n or 'binary' in n for n in result.notes), result.notes)

    def test_the_states_are_removed_after_a_pass_and_kept_after_a_fail(self):
        self.run_gate()
        self.assertFalse((self.out / '4.05/states').exists())
        self.setUp()
        result = self.run_gate(cand_knobs={'STATE': 'other'})
        self.assertEqual(result.verdict, 'FAIL')
        self.assertTrue((self.out / '4.05/states').exists())

    def test_other_state_bytes_fail(self):
        result = self.run_gate(cand_knobs={'STATE': 'other'})
        self.assertEqual(result.verdict, 'FAIL')
        self.assertTrue(any('state bytes' in r and 'ref' in r for r in result.reasons), result.reasons)

    def test_other_next_tokens_fail(self):
        result = self.run_gate(cand_knobs={'ANSWER': 'b'})
        self.assertEqual(result.verdict, 'FAIL')
        self.assertTrue(any('answer' in r for r in result.reasons), result.reasons)

    def test_no_restore_fails(self):
        result = self.run_gate(cand_knobs={'RESTORE': '0'})
        self.assertEqual(result.verdict, 'FAIL')
        self.assertTrue(any('cached' in r for r in result.reasons), result.reasons)

    def test_a_restore_that_changes_the_answer_fails(self):
        result = self.run_gate(cand_knobs={'RESTORE_WRONG': '1'})
        self.assertEqual(result.verdict, 'FAIL')
        self.assertTrue(any('restored answer' in r for r in result.reasons), result.reasons)

    def test_a_neutral_switch_that_changes_the_state_fails(self):
        result = self.run_gate(cand_knobs={'BAD_FUSE': '1'})
        self.assertEqual(result.verdict, 'FAIL')
        self.assertTrue(any('neutral' in r for r in result.reasons), result.reasons)

    def test_a_state_that_changes_between_two_equal_runs_is_an_error(self):
        result = self.run_gate(cand_knobs={'NOISE': '1'})
        self.assertEqual(result.verdict, 'ERROR')
        self.assertTrue(any('twice' in r or 'reproduc' in r for r in result.reasons), result.reasons)

    def test_a_server_that_does_not_start_is_an_error(self):
        result = self.run_gate(cand_knobs={'FAIL_START': '1'}, ready_timeout_s=5)
        self.assertEqual(result.verdict, 'ERROR')
        self.assertTrue(any('did not start' in r for r in result.reasons), result.reasons)

    def test_no_state_written_is_an_error(self):
        result = self.run_gate(cand_knobs={'NO_WRITE': '1'})
        self.assertEqual(result.verdict, 'ERROR')

    def test_crucible_variables_of_the_parent_do_not_reach_the_server(self):
        os.environ['CRUCIBLE_IN_LEASE'] = '1'
        os.environ['CRUCIBLE_FUSE'] = '3'
        self.addCleanup(os.environ.pop, 'CRUCIBLE_IN_LEASE', None)
        self.addCleanup(os.environ.pop, 'CRUCIBLE_FUSE', None)
        self.run_gate()
        seen = [json.loads(p.read_text()) for p in (self.out / '4.05').glob('states/**/env-*.json')] or \
               [json.loads(p.read_text()) for p in (self.out / '4.05').rglob('env-*.json')]
        self.assertTrue(seen)
        for names in seen:
            self.assertNotIn('CRUCIBLE_IN_LEASE', names)
            self.assertIn('CRUCIBLE_PLE_HOST', names)
        cand = [n for p in (self.out / '4.05').rglob('env-*cand.json') for n in json.loads(p.read_text())]
        self.assertNotIn('CRUCIBLE_FUSE', cand)

    def test_the_judge_reads_the_files_alone(self):
        self.run_gate()
        again = gate.judge(self.out / '4.05')
        self.assertEqual(again.verdict, 'PASS')
        (self.out / '4.05/restore-turn2-disk.json').write_text(json.dumps([{'sha256': 'x', 'cached_tokens': 5}]))
        self.assertEqual(gate.judge(self.out / '4.05').verdict, 'FAIL')


class CrossTests(World):
    def test_the_cross_arm_runs_when_both_binaries_share_a_stamped_epoch(self):
        result = self.run_gate({'EPOCH': 'e' * 64}, {'EPOCH': 'e' * 64})
        self.assertEqual((result.verdict, result.reasons), ('PASS', []), result.notes)
        self.assertTrue((self.out / '4.05/cross-turn2-disk.json').exists())
        self.assertTrue(any('cross' in n for n in result.notes), result.notes)

    def test_the_cross_arm_catches_a_wrong_restore_by_the_other_binary(self):
        result = self.run_gate({'EPOCH': 'e' * 64, 'RESTORE_WRONG': '1'}, {'EPOCH': 'e' * 64})
        self.assertEqual(result.verdict, 'FAIL')
        self.assertTrue(any('cross' in r for r in result.reasons), result.reasons)

    def test_another_epoch_skips_the_cross_arm(self):
        result = self.run_gate({'EPOCH': 'd' * 64}, {'EPOCH': 'e' * 64})
        self.assertEqual(result.verdict, 'PASS', result.reasons)
        self.assertFalse((self.out / '4.05/cross-turn2-disk.json').exists())
        self.assertTrue(any('cross arm skipped' in n for n in result.notes), result.notes)

    def test_a_candidate_without_a_stamp_fails_when_a_stamp_is_required(self):
        result = self.run_gate({'EPOCH': 'd' * 64}, None, require_stamped=True)
        self.assertEqual(result.verdict, 'FAIL')
        self.assertTrue(any('stamp' in r for r in result.reasons), result.reasons)


class CliTests(World):
    def test_the_cli_exit_codes_and_the_last_line(self):
        ref, cand = self.engine('ref', ID='ref'), self.engine('cand', ID='cand')
        args = [sys.executable, '-I', str(ROOT / 'tools/state_equality_gate.py'), 'run', '--pack', '4.05', '--ref-bin', str(ref),
                '--cand-bin', str(cand), '--ckpt', str(self.root / 'ckpt'), '--layout', '--streams 1', '--env', 'CRUCIBLE_PLE_HOST=1',
                '--neutral-env', 'CRUCIBLE_FUSE=0', '--tree', str(self.tree), '--python', sys.executable, '--out', str(self.out),
                '--tokens', '2000', '--port', str(self.base_port + 2), '--ready-timeout', '20', '--stop-timeout', '20']
        done = subprocess.run(args, capture_output=True, text=True, cwd=self.root)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertRegex(done.stdout.strip().splitlines()[-1], r'^STATE-GATE 4\.05 PASS')
        dry = subprocess.run(args + ['--dry'], capture_output=True, text=True, cwd=self.root)
        self.assertEqual(dry.returncode, 0)
        self.assertIn('arm ref', dry.stdout)
        self.assertIn('arm restore', dry.stdout)


class SignalTests(World):
    def test_a_killed_gate_leaves_no_server_behind(self):
        import signal as sg, time
        ref = self.engine('ref', ID='ref', HANG='1')
        cand = self.engine('cand', ID='cand')
        args = [sys.executable, '-I', str(ROOT / 'tools/state_equality_gate.py'), 'run', '--pack', '4.05', '--ref-bin', str(ref),
                '--cand-bin', str(cand), '--ckpt', str(self.root / 'ckpt'), '--layout', '--streams 1', '--env', 'CRUCIBLE_PLE_HOST=1',
                '--neutral-env', 'CRUCIBLE_FUSE=0', '--tree', str(self.tree), '--python', sys.executable, '--out', str(self.out),
                '--tokens', '2000', '--port', str(self.base_port + 3), '--ready-timeout', '20', '--stop-timeout', '20']
        gate_proc = subprocess.Popen(args, cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pidfile = self.out / '4.05/pid-ref.txt'
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text() and 'Crucible is ready' in (self.out / '4.05/ref.server.log').read_text(errors='replace'):
                break
            time.sleep(0.2)
        time.sleep(1)
        pid = int(pidfile.read_text())
        os.kill(pid, 0)
        gate_proc.send_signal(sg.SIGTERM)
        gate_proc.wait(30)
        time.sleep(1)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertNotEqual(gate_proc.returncode, 0)


if __name__ == '__main__':
    unittest.main()
