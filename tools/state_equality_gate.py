#!/usr/bin/env python3
"""State-equality gate for the state epoch.

The state epoch (engine/runtime/state_epoch.h) lets a new engine binary reuse the SSD conversation cache of the old one.
That is safe only when both binaries write the same state bytes. This gate shows it on a GPU, for one pack, in one lease.
Synthetic text only. The arms (each one server start, a stop that saves the states, and the files it left):

  ref      the reference binary, turn 1 of --tokens tokens, stop              -> state files S_ref
  cand     the candidate binary, the same turn 1, a fresh state folder, stop  -> S_cand
  aa       the candidate again (a second cold turn 1)                         -> S_aa (the noise floor)
  neutral  the candidate with --neutral-env words (the switches that the epoch leaves out of the identity,
           set to other values), the same turn 1                              -> S_neutral
  restore  the candidate on S_cand: turn 2 restores from disk (cached tokens above 0), then turn 2 again with
           cache_control bypass (a cold prefill)
  cross    only when both binaries print the same stamped epoch: the reference binary on S_cand, as `restore`

Checks (a state digest drops the identity, tools/state_digest.py):
  S_ref = S_cand = S_neutral, and S_aa = S_cand (else ERROR: the state is not reproducible, the gate proves nothing);
  the turn 1 answers are equal; the restored turn 2 answer equals the cold turn 2 answer; the writers wrote a state.
Verdict: PASS, FAIL (a measured difference) or ERROR (a fault of the run). The last line is `STATE-GATE <pack> <verdict> ...`.
Exit 0 PASS, 1 FAIL, 2 ERROR. The state folders stay after a verdict other than PASS.

  state_equality_gate.py run --pack 4.05 --ref-bin R --cand-bin C --ckpt DIR --layout '...' --env '...' --neutral-env '...'
                             --tree TREE --python PY --out OUT [--tokens N] [--port P] [--dry] [--keep]
  state_equality_gate.py judge OUT/PACK     the verdict from the files of an earlier run
"""
import argparse
import atexit
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import state_digest  # noqa: E402
import state_disk_restart as client  # noqa: E402

EPOCH_LINE = re.compile(r'^state_disk epoch source=(\w+) epoch=([0-9a-f]+)', re.M)
READY = 'Crucible is ready'
ARMS = ('ref', 'cand', 'aa', 'neutral', 'restore', 'cross')


@dataclass
class Config:
    pack: str
    ref_bin: Path
    cand_bin: Path
    ckpt: Path
    layout: str
    env: str
    neutral_env: str
    tree: Path
    python: str
    out: Path
    tokens: int = 48000
    port: int = 8816
    max_tokens: int = 48
    ready_timeout_s: int = 300
    stop_timeout_s: int = 150
    request_timeout_s: int = 1800
    require_stamped: bool = False
    keep: bool = False


@dataclass
class Result:
    verdict: str
    reasons: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class RunError(Exception):
    pass


def clean_env():
    """The parent environment without CRUCIBLE_* names: every such name is part of the state identity, so the server must
    see the words of the arm and nothing else (the lease sets CRUCIBLE_IN_LEASE)."""
    return {k: v for k, v in os.environ.items() if not k.startswith('CRUCIBLE_')}


LIVE = []  # the servers that run now; a signal or an exit kills them (they have their own session, so the lease would not)


def kill_all(*_):
    for server in list(LIVE):
        server.kill()


def die(signum, _frame):
    kill_all()
    sys.exit(128 + signum)


class Server:
    """One `python -m server` in its own process group."""

    def __init__(self, cfg, arm, binary, words, state_dir, log):
        self.cfg, self.arm, self.log = cfg, arm, log
        env = clean_env()
        env.update(w.split('=', 1) for w in shlex.split(words))
        cmd = [cfg.python, '-m', 'server', str(cfg.ckpt), '--engine', str(binary), *shlex.split(cfg.layout),
               '--port', str(cfg.port), '--state-dir', str(state_dir)]
        self.command = cmd
        self.handle = open(log, 'wb')
        self.proc = subprocess.Popen(cmd, cwd=cfg.tree, env=env, stdout=self.handle, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        LIVE.append(self)

    def wait_ready(self):
        deadline = time.monotonic() + self.cfg.ready_timeout_s
        while time.monotonic() < deadline:
            if READY in Path(self.log).read_text(errors='replace'):
                return
            if self.proc.poll() is not None:
                break
            time.sleep(0.2)
        raise RunError(f'{self.arm}: the server did not start (see {Path(self.log).name})')

    def stop(self):
        """SIGTERM to the server (the engine saves its states), then a wait; SIGKILL to the group after the limit."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
        slow = False
        try:
            self.proc.wait(self.cfg.stop_timeout_s)
        except subprocess.TimeoutExpired:
            slow = True
        self.kill()
        self.handle.close()
        if slow:
            raise RunError(f'{self.arm}: the server was still running {self.cfg.stop_timeout_s} s after SIGTERM')

    def kill(self):
        if self in LIVE:
            LIVE.remove(self)
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.proc.wait()


def text(path):
    return Path(path).read_text(errors='replace') if Path(path).exists() else ''


def new_state_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    Path(path).mkdir(parents=True, mode=0o700)
    os.chmod(path, 0o700)
    return Path(path)


def ask(cfg, mode, out, first=None, bypass=False):
    conversations = client.turn_messages(1, cfg.tokens, first)
    results = client.run_all(cfg.port, conversations, cfg.max_tokens, bypass)
    Path(out).write_text(json.dumps(results, indent=1))
    return results


def read_turn(path):
    data = json.loads(text(path) or 'null')
    if not data or not isinstance(data, list) or not data[0].get('sha256'):
        raise RunError(f'{Path(path).name}: no answer')
    return data[0]


class Gate:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = Path(cfg.out) / cfg.pack
        self.states = self.dir / 'states'
        self.notes = []

    def say(self, line):
        print(line, flush=True)

    def arm(self, name, binary, extra_env, turn1=False, turn2_from=None, state_dir=None):
        """Start a server, run the turns, stop it. Returns the state folder."""
        cfg = self.cfg
        state_dir = state_dir or new_state_dir(self.states / name)
        words = (cfg.env + ' ' + extra_env).strip()
        server = Server(cfg, name, binary, words, state_dir, self.dir / f'{name}.server.log')
        t0 = time.monotonic()
        try:
            server.wait_ready()
            if turn1:
                ask(cfg, 'turn1', self.dir / f'{name}-turn1.json')
            if turn2_from:
                first = json.loads(text(turn2_from))
                ask(cfg, 'turn2', self.dir / f'{name}-turn2-disk.json', first)
                ask(cfg, 'turn2', self.dir / f'{name}-turn2-cold.json', first, bypass=True)
        except Exception:
            server.kill()
            server.handle.close()
            raise
        server.stop()
        if turn1:
            self.write_digest(name, state_dir)
        self.say(f'arm {name} done in {time.monotonic() - t0:.0f} s')
        return state_dir

    def write_digest(self, name, state_dir):
        """digest-NAME.txt: the identity-free digest lines of the state folder (hashes and sizes only)."""
        try:
            body = '\n'.join(state_digest.digest(state_dir)) + '\n'
        except (ValueError, OSError) as error:
            body = f'ERROR {error}\n'
        (self.dir / f'digest-{name}.txt').write_text(body)

    def run(self):
        cfg = self.cfg
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in self.dir.glob('*'):
            if old.is_file():
                old.unlink()
        shutil.rmtree(self.states, ignore_errors=True)
        try:
            self.arm('ref', cfg.ref_bin, '', turn1=True)
            cand_dir = self.arm('cand', cfg.cand_bin, '', turn1=True)
            self.arm('aa', cfg.cand_bin, '', turn1=True)
            self.arm('neutral', cfg.cand_bin, cfg.neutral_env, turn1=True)
            self.arm('restore', cfg.cand_bin, '', turn2_from=self.dir / 'cand-turn1.json', state_dir=cand_dir)
            if self.cross_possible():
                self.arm('cross', cfg.ref_bin, '', turn2_from=self.dir / 'cand-turn1.json', state_dir=cand_dir)
        except RunError as error:
            return self.finish(Result('ERROR', [str(error)]))
        except Exception as error:  # a harness fault must not look like a measured difference
            return self.finish(Result('ERROR', [f'{type(error).__name__}: {error}']))
        return self.finish(judge(self.dir, require_stamped=cfg.require_stamped))

    def cross_possible(self):
        ref, cand = epoch_of(self.dir / 'ref.server.log'), epoch_of(self.dir / 'cand.server.log')
        return bool(ref and cand and ref[0] == cand[0] == 'stamped' and ref[1] == cand[1])

    def finish(self, result):
        (self.dir / 'result.json').write_text(json.dumps(result.__dict__, indent=1))
        if result.verdict == 'PASS' and not self.cfg.keep:
            shutil.rmtree(self.states, ignore_errors=True)
        return result


def epoch_of(log):
    found = EPOCH_LINE.findall(text(log))
    return found[-1] if found else None


def digest_of_folder(folder, out):
    try:
        body = '\n'.join(state_digest.digest(folder)) + '\n'
    except (ValueError, OSError) as error:
        body = f'ERROR {error}\n'
    Path(out).write_text(body)
    return body


def judge(directory, require_stamped=False):
    """The verdict from the files in `directory` (OUT/PACK). Reads files only."""
    d = Path(directory)
    states = d / 'states'
    result = Result('PASS')
    fail, error = result.reasons, []
    note = result.notes.append
    # --- the runs themselves
    answers = {}
    for name in ('ref', 'cand', 'aa', 'neutral'):
        try:
            answers[name] = read_turn(d / f'{name}-turn1.json')['sha256']
        except RunError as e:
            error.append(str(e))
    ep = {name: epoch_of(d / f'{name}.server.log') for name in ('ref', 'cand')}
    for name, found in ep.items():
        note(f'{name} epoch source={found[0] if found else "none"} {found[1] if found else ""}'.strip())
    # --- the state bytes (the digest files of the run; the state folders are removed after a PASS)
    lines = {}
    for name in ('ref', 'cand', 'aa', 'neutral'):
        body = text(d / f'digest-{name}.txt')
        if not body:
            body = digest_of_folder(states / name, d / f'digest-{name}.txt') if (states / name).exists() else ''
        if not body:
            error.append(f'{name}: no state digest and no state folder')
        elif body.startswith('ERROR'):
            error.append(f'{name}: {body.strip()}')
        else:
            lines[name] = [l for l in body.splitlines() if l]
    for left, right, label in (('cand', 'aa', 'cand twice'), ('ref', 'cand', 'ref and cand'), ('cand', 'neutral', 'cand and neutral')):
        if left not in lines or right not in lines:
            continue
        diff = state_digest.diff_lines(lines[left], lines[right])
        (d / f'compare-{left}-{right}.txt').write_text('EQUAL\n' if not diff else 'DIFFERENT\n' + '\n'.join(diff) + '\n')
        if diff and label == 'cand twice':
            error.append('cand wrote different state bytes in two equal runs: the state is not reproducible (the gate cannot show equality)')
        elif diff and label == 'cand and neutral':
            fail.append(f'state bytes differ with the neutral switches set (neutral arm): {len(diff)} lines, compare-{left}-{right}.txt')
        elif diff:
            fail.append(f'state bytes differ: {label} ({len(diff)} lines, compare-{left}-{right}.txt)')
        else:
            note(f'state bytes equal: {label}')
    # --- the writers wrote a state and refused none
    for name in ('ref', 'cand', 'aa', 'neutral'):
        log = text(d / f'{name}.server.log')
        if not re.search(r'^state_disk spill .*result=written', log, re.M):
            error.append(f'{name}: no state_disk spill result=written in the server log')
        if re.search(r'^state_disk spill .*reason=io', log, re.M):
            error.append(f'{name}: a state spill was refused (reason=io)')
    # --- the answers of turn 1
    if len(set(answers.values())) > 1 and set(answers) >= {'ref', 'cand'}:
        different = sorted(n for n in answers if answers[n] != answers.get('cand'))
        fail.append(f'turn 1 answer differs: {", ".join(different)} against cand')
    # --- restore against cold
    for arm in ('restore', 'cross'):
        disk, cold = d / f'{arm}-turn2-disk.json', d / f'{arm}-turn2-cold.json'
        if arm == 'cross' and not disk.exists():
            note('cross arm skipped: the two binaries do not share one stamped epoch')
            continue
        try:
            a, b = read_turn(disk), read_turn(cold)
        except RunError as e:
            error.append(f'{arm}: {e}')
            continue
        who = 'cand' if arm == 'restore' else 'ref binary on the state of cand (cross)'
        if (b.get('cached_tokens') or 0) != 0:
            error.append(f'{arm}: the bypass turn had cached tokens; it is not a cold prefill')
        if not (a.get('cached_tokens') or 0) > 0:
            fail.append(f'{arm}: no cached tokens on the turn 2 restore from disk ({who})')
        elif a['sha256'] != b['sha256']:
            fail.append(f'{arm}: restored answer differs from the cold answer ({who})')
        else:
            note(f'{arm}: restored answer equals cold answer, cached_tokens={a.get("cached_tokens")}')
        log = text(d / f'{arm}.server.log')
        if (a.get('cached_tokens') or 0) > 0 and (not re.search(r'^state_disk read .*result=ok', log, re.M) or
                                                  not re.search(r'^state_disk restore ', log, re.M)):
            error.append(f'{arm}: cached tokens without state_disk read result=ok and state_disk restore in the server log')
    # --- the stamp
    if require_stamped and not (ep['cand'] and ep['cand'][0] == 'stamped'):
        fail.append('the candidate has no epoch stamp (state_disk epoch source=binary): build with tools/state_epoch.py stamp')
    result.verdict = 'ERROR' if error else ('FAIL' if fail else 'PASS')
    result.reasons = error + fail
    return result


def run_pack(cfg):
    return Gate(cfg).run()


def plan(cfg):
    out = [f'pack {cfg.pack}: tokens={cfg.tokens} port={cfg.port} out={cfg.out}/{cfg.pack}']
    words = lambda extra: (cfg.env + ' ' + extra).strip()  # noqa: E731
    for name, binary, extra, what in (('ref', cfg.ref_bin, '', 'turn 1, stop'), ('cand', cfg.cand_bin, '', 'turn 1, stop'),
                                      ('aa', cfg.cand_bin, '', 'turn 1, stop'), ('neutral', cfg.cand_bin, cfg.neutral_env, 'turn 1, stop'),
                                      ('restore', cfg.cand_bin, '', 'on the state of cand: turn 2 from disk, turn 2 bypass, stop'),
                                      ('cross', cfg.ref_bin, '', 'only with one stamped epoch: ref binary on the state of cand, as restore')):
        out.append(f'arm {name}: env {words(extra)} | {binary} | {what}')
    out.append(f'server: {cfg.python} -m server {cfg.ckpt} --engine BIN {cfg.layout} --port {cfg.port} --state-dir DIR')
    return out


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    run = sub.add_parser('run')
    for name in ('pack', 'ref-bin', 'cand-bin', 'ckpt', 'layout', 'env', 'neutral-env', 'tree', 'python', 'out'):
        run.add_argument('--' + name, required=True)
    run.add_argument('--tokens', type=int, default=48000)
    run.add_argument('--port', type=int, default=8816)
    run.add_argument('--max-tokens', type=int, default=48)
    run.add_argument('--ready-timeout', type=int, default=300)
    run.add_argument('--stop-timeout', type=int, default=150)
    run.add_argument('--require-stamped', action='store_true')
    run.add_argument('--keep', action='store_true')
    run.add_argument('--dry', action='store_true')
    j = sub.add_parser('judge')
    j.add_argument('directory')
    j.add_argument('--require-stamped', action='store_true')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, die)
    atexit.register(kill_all)
    if args.cmd == 'judge':
        result = judge(args.directory, args.require_stamped)
        pack = Path(args.directory).name
    else:
        cfg = Config(pack=args.pack, ref_bin=Path(args.ref_bin).expanduser(), cand_bin=Path(args.cand_bin).expanduser(),
                     ckpt=Path(args.ckpt).expanduser(), layout=args.layout, env=args.env, neutral_env=args.neutral_env,
                     tree=Path(args.tree).expanduser(), python=args.python, out=Path(args.out).expanduser(), tokens=args.tokens,
                     port=args.port, max_tokens=args.max_tokens, ready_timeout_s=args.ready_timeout,
                     stop_timeout_s=args.stop_timeout, require_stamped=args.require_stamped, keep=args.keep)
        if args.dry:
            print('\n'.join(plan(cfg)))
            return 0
        pack = cfg.pack
        result = run_pack(cfg)
    for n in result.notes:
        print(f'NOTE {n}')
    reasons = '; '.join(result.reasons) if result.reasons else 'all checks equal'
    print(f'STATE-GATE {pack} {result.verdict} {reasons}')
    return {'PASS': 0, 'FAIL': 1}.get(result.verdict, 2)


if __name__ == '__main__':
    sys.exit(main())
