#!/usr/bin/env python3
"""Run all shared gates with new artifacts. Never accept a summary as evidence."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

# Other tools load this file by path, so tools/ may be missing from sys.path.
sys.path.append(str(Path(__file__).resolve().parent))
import work_paths  # noqa: E402
import layout  # noqa: E402

WIDTH = 4 * 2560
VOCAB = 248320
NAMES = tuple(f"h_{layer:02d}.f32" for layer in range(49))
POSITIONS = (0, 14, 23)
POINTS = {(layer, pos) for layer in range(48) for pos in POSITIONS}
ROOT = Path(__file__).resolve().parents[1]
# One 4K bulk gate writes about 50 GB of dumps; a train writes about 250 GB.
MIN_FREE_GB = 150
BASELINE_CACHE_DIR = work_paths.gates_cache()
CAPTURE_TIMEOUT_S = 300
# Default gpu-run wait for the GPU lock; a full queue can need more (--queue-wait).
QUEUE_WAIT_S = 7200
# One GPU lease stays within 300 s; the runner keeps a margin for its own start and results.
# So a capture of 286-300 s passes one-lease-per-capture mode but fails --lease-batch (fail closed).
LEASE_LIMIT_S, LEASE_BUDGET_S = 300, 285
LEASE_MARGIN_S = LEASE_LIMIT_S - LEASE_BUDGET_S
# The shared lease rule: 900 s at most (--lease-seconds).
LEASE_MAX_S = 900
# Memory floor inside each lease: lease_runner.py polls
# MemAvailable >= 126e6 kB every 5 s for at most 300 s before the first engine starts, else it exits 76 with no
# capture run, and the lease runs again until the queue wait has passed. The gpu-run timeout holds the wait.
LEASE_MEM_FLOOR_GB, LEASE_MEM_WAIT_S, LEASE_MEM_POLL_S = 126, 300, 5
LEASE_EXIT_MEMORY = 76
# chunk1 runs the prompt one row at a time: on a 1024-row prompt it takes about twice as long as
# the other captures. The lease runner estimates the next capture from the slowest one so far, so
# chunk1 runs last in a lease (the checks keep the gate order).
LEASE_LAST = ('chunk1',)
# Its estimate in the lease runner is this many times the slowest capture so far (times the runner's
# 1.25 margin), so that it never starts into a remainder that cuts it. bulk305: 76 s against at most
# 45 s for the others, a ratio of 1.7. 45 s x 1.25 x 1.4 + 5 s = 84 s: above 76 s, and below the 88 s
# that remain after five captures with a fresh baseline, so a nominal gate still fits one lease.
LEASE_ESTIMATE_SCALE = {'chunk1': 1.4}
ORACLE_LOCK = Path('/tmp/crucible-oracle.lock')
# Do not publish arbitrary environment values. Future prefixes can contain credentials.
# The CRUCIBLE_* names are the kRecord switches of engine/switches.h (tests/test_switch_lists.py).
RUNTIME_KNOBS = (
    'HSA_USERPTR_FOR_PAGED_MEM','HSA_USE_SVM','GPU_PINNED_MIN_XFER_SIZE',
    'HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES',
    'CRUCIBLE_WARMUP','CRUCIBLE_PROFILE','CRUCIBLE_TUNE_ROWS',
    'CRUCIBLE_FUSE','CRUCIBLE_LOOKAHEAD','CRUCIBLE_LOOKAHEAD_MIB',
    'CRUCIBLE_STATE_CACHE','CRUCIBLE_STATIC_CACHE',
    'CRUCIBLE_DECODE_RING',
    'CRUCIBLE_PERSISTENT_GDN',
    'CRUCIBLE_PF_EXACT_SPLIT','CRUCIBLE_PF_HC_FUSED',
    'CRUCIBLE_PF_GDN_MICRO','CRUCIBLE_PF_GDN_EAGER',
    'CRUCIBLE_PF_QSA_ROWS','CRUCIBLE_PF_MOE_RM',
    'CRUCIBLE_PF_TILES',
    'CRUCIBLE_PF_FX_LEAD','CRUCIBLE_PF_PAD_H',
    'CRUCIBLE_QSA_CAPTURE_AT','CRUCIBLE_QSA_CAPTURE_DIR',
    'CRUCIBLE_PF_OUT','CRUCIBLE_PF_LAUNCH_ROWS','CRUCIBLE_PF_STAGE_SHARE',
    'CRUCIBLE_COMPACT_STATE',
    'CRUCIBLE_QSA_CAPTURE_INDEX_ONLY',
    'CRUCIBLE_HC_WMMA',
    'CRUCIBLE_ABLATE','CRUCIBLE_ABLATE_ANY_CHECKPOINT','CRUCIBLE_PLE_BF16','CRUCIBLE_PLE_HOST','CRUCIBLE_PF_PRUNE','CRUCIBLE_PLE_TRACE',

    'CRUCIBLE_MTP_SIDECAR',
    'CRUCIBLE_EXACT',
    'CRUCIBLE_PACKED_ROWS',

)
# Switches that select a non-bitwise path. The engine reads each one through
# engine/layers/numerics.h, so CRUCIBLE_EXACT=1 (option A) ignores them (tests/test_numerics_switch.py).
NUMERICS_SWITCHES = (
    'CRUCIBLE_PF_EXACT_SPLIT', 'CRUCIBLE_PF_HC_FUSED', 'CRUCIBLE_PF_GDN_EAGER', 'CRUCIBLE_PF_GDN_MICRO',
    'CRUCIBLE_PF_QSA_ROWS', 'CRUCIBLE_HC_WMMA', 'CRUCIBLE_PERSISTENT_GDN',
)

# Cache keys keep the sidecar and BF16 table paths hashed: they name the user's home.
CACHE_PUBLIC_KNOBS = tuple(name for name in RUNTIME_KNOBS if name not in ('CRUCIBLE_MTP_SIDECAR', 'CRUCIBLE_PLE_BF16'))


def ple_bf16_public(value):
    """The BF16 n-gram table as a report shows it: file name and size. The path names the user's home, and
    the 102 GB file is too large to hash for each gate."""
    path = Path(value)
    try:
        info = path.stat()
    except OSError:
        return 'set'
    return f'{path.name} ({info.st_size} bytes)' if stat.S_ISREG(info.st_mode) else 'set'


def runtime_environment(env):
    # The sidecar path names the user's home; mtp_sidecar_record publishes its sha256 and size instead.
    out = {name: env.get(name) for name in RUNTIME_KNOBS}
    if out.get('CRUCIBLE_MTP_SIDECAR'):
        out['CRUCIBLE_MTP_SIDECAR'] = 'set (see mtp_sidecar)'
    if out.get('CRUCIBLE_PLE_BF16'):
        out['CRUCIBLE_PLE_BF16'] = ple_bf16_public(out['CRUCIBLE_PLE_BF16'])
    return out


def capture_environment(reference=False):
    env = {**os.environ, 'HSA_USERPTR_FOR_PAGED_MEM': os.environ.get('HSA_USERPTR_FOR_PAGED_MEM', '0')}
    # A storage qualification must compare the opt-in with the baseline's full layout.
    if reference and env.get('CRUCIBLE_COMPACT_STATE') == '1':
        env = {**env, 'CRUCIBLE_COMPACT_STATE': '0'}
    # The same: the host PLE gather is compared with the resident table. The engine parses the value
    # with strtol ('01', '+1' and ' 1' turn the gather on), so every value except '0' gets '0'.
    if reference and env.get('CRUCIBLE_PLE_HOST') not in (None, '0'):
        env = {**env, 'CRUCIBLE_PLE_HOST': '0'}
    # A named baseline is the stock model; the ablation is the candidate's opt-in.
    if reference:
        env = {k: v for k, v in env.items() if not k.startswith('CRUCIBLE_ABLATE')}
    # Option A ignores the numerics switches, so its reference is the stock baseline without them.
    if reference and env.get('CRUCIBLE_EXACT') == '1':
        env = {k: v for k, v in env.items() if k != 'CRUCIBLE_EXACT' and k not in NUMERICS_SWITCHES}
    return env


def mtp_sidecar_record(env):
    """Size and sha256 of the CRUCIBLE_MTP_SIDECAR file: a path is not provenance, and it names the user."""
    value = env.get('CRUCIBLE_MTP_SIDECAR')
    if not value:
        return None
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ValueError('CRUCIBLE_MTP_SIDECAR must name a regular file')
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


SIDECAR_LINE = re.compile(r'^mtp sidecar: ([1-9][0-9]*) tensors from \S.* \(([0-9]+) bytes\)$', re.M)


def pin_sidecar(directory):
    """Pin the sidecar as pin_binary pins engines: the GPU wait can last hours, and the source can change.
    The captures, the cache key and the batch probe all read os.environ, so the variable then names the
    private read-only copy. run() restores the variable."""
    record = mtp_sidecar_record(os.environ)
    if record is None:
        return None
    source = Path(os.environ['CRUCIBLE_MTP_SIDECAR'])
    directory.mkdir(exist_ok=True)
    target = directory / 'mtp_sidecar.safetensors'
    shutil.copyfile(source, target)
    target.chmod(0o444)
    pinned = mtp_sidecar_record({'CRUCIBLE_MTP_SIDECAR': str(target)})
    if pinned != record or mtp_sidecar_record(os.environ) != record:
        raise ValueError('CRUCIBLE_MTP_SIDECAR changed while it was copied')
    os.environ['CRUCIBLE_MTP_SIDECAR'] = str(target.resolve())
    return pinned


def cache_environment(env):
    # The pinned copy's path changes with each gate output; its bytes identify the sidecar.
    if not env.get('CRUCIBLE_MTP_SIDECAR'):
        return env
    return {**env, 'CRUCIBLE_MTP_SIDECAR': 'sha256:' + mtp_sidecar_record(env)['sha256']}


def check_sidecar_log(job):
    """An engine that ignores CRUCIBLE_MTP_SIDECAR runs the stock head, so a candidate capture must show the
    engine's own load line with the recorded size, and the copy must still hold the recorded bytes."""
    recorded = job['record'].get('mtp_sidecar')
    if recorded is None:
        return
    if mtp_sidecar_record(job['env']) != recorded:
        raise ValueError('the MTP sidecar changed during the capture')
    if job.get('reference'):
        return  # the named baseline is the stock model; an older binary ignores the variable
    lines = SIDECAR_LINE.findall((job['output'] / 'stderr.log').read_text(errors='replace'))
    if len(lines) != 1 or int(lines[0][1]) != recorded['bytes']:
        raise ValueError(f'mtp sidecar: the engine did not log loading {recorded["bytes"]} sidecar bytes once; '
                         f'inspect {job["output"]}/stderr.log')


def gate1_mode(args, env):
    """Selects the Gate 1 reference. 'ablated-self' is for an intentional
    CRUCIBLE_ABLATE numerics change, which has no stock reference: a second
    capture of the same binary and mode must repeat every dump bit for bit.
    The identity sidecar with a named baseline
    proves the ablated plumbing against stock."""
    chosen = [m for m, on in (('baseline', args.baseline_engine), ('oracle', args.oracle_baseline),
                              ('ablated-self', getattr(args, 'ablated_numerics', False))) if on]
    # argparse requires one; without a baseline binary the old flow is the oracle.
    if not chosen:
        return 'oracle'
    if len(chosen) > 1:
        raise ValueError('choose exactly one Gate 1 reference: --baseline-engine, --oracle-baseline or --ablated-numerics')
    if chosen[0] == 'ablated-self':
        if not env.get('CRUCIBLE_ABLATE'):
            raise ValueError('--ablated-numerics needs CRUCIBLE_ABLATE in the environment')
    return chosen[0]

# These exact CLI sources use a full head when no draft selection is supplied.
LEGACY_FULL_HEAD_SOURCES = {
    '0f16b7431237efc8b3c22e590ae71e6cc2832eaa328a96355f60b9edde3e6351': 'origin/main c286bda',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def disk_preflight(output):
    probe = Path(output).absolute()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < MIN_FREE_GB * 10**9:
        raise ValueError(f'disk preflight: {free / 10**9:.1f} GB free on the filesystem of {probe}, '
                         f'gates need at least {MIN_FREE_GB} GB; prune old dumps and run again')
    return free


def prune_dumps(output):
    # Delete dump binaries only after a PASS report records their hashes; check every file first.
    path = Path(output) / 'report.json'
    report = json.loads(path.read_text())
    if report.get('status') != 'PASS':
        raise ValueError(f'dumps are kept: report status is not PASS: {path}')
    targets = []
    for name, value in report.get('captures', {}).items():
        for file, meta in value.get('dumps', {}).items():
            dump = Path(output) / name / 'dump' / file
            if not dump.exists():
                continue
            info = dump.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_size != meta['bytes']
                    or info.st_mtime_ns != meta['mtime_ns'] or len(meta.get('sha256', '')) != 64):
                raise ValueError(f'dumps are kept: {dump} changed after its hash was recorded')
            targets.append((dump, info.st_size))
    for dump, _ in targets:
        dump.unlink()
    report['dumps_pruned'] = {'ns': time.time_ns(), 'files': len(targets),
                              'bytes': sum(size for _, size in targets)}
    save(path, report)
    return report['dumps_pruned']


def regular(path, started_ns):
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f'missing artifact: {path}') from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f'artifact must be a private regular file: {path}')
    if info.st_size == 0 or min(info.st_mtime_ns, info.st_ctime_ns) < started_ns:
        raise ValueError(f'empty or stale artifact: {path}')
    return info


def check_dumps(directory, rows, started_ns):
    if rows < 1 or directory.is_symlink() or not directory.is_dir():
        raise ValueError(f'invalid dump directory: {directory}')
    if {p.name for p in directory.iterdir()} != set(NAMES):
        raise ValueError(f'expected exactly h_00.f32 through h_48.f32: {directory}')
    result = {}
    for name in NAMES:
        path = directory / name
        info = regular(path, started_ns)
        if info.st_size != rows * WIDTH * 4:
            raise ValueError(f'wrong dump size: {path}: {info.st_size}, expected {rows * WIDTH * 4}')
        result[name] = {'bytes': info.st_size, 'sha256': digest(path),
                        'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}
    return result


def compare_dumps(left, right, prefix_rows=None):
    # Compare bytes directly, not just checksums or a process summary.
    for name in NAMES:
        with (left / name).open('rb') as a, (right / name).open('rb') as b:
            remaining = prefix_rows * WIDTH * 4 if prefix_rows is not None else None
            while remaining is None or remaining > 0:
                size = min(1 << 20, remaining) if remaining is not None else 1 << 20
                av, bv = a.read(size), b.read(size)
                if av != bv or (remaining is not None and not av):
                    raise ValueError(f'dump content differs: {name}')
                if not av:
                    break
                if remaining is not None:
                    remaining -= len(av)


def parse_ids(text, count=None):
    if not re.fullmatch(r'\s*\d+(?:\s*,\s*\d+)*\s*', text):
        raise ValueError('missing or invalid comma-separated IDs')
    ids = [int(value) for value in text.split(',')]
    if any(value >= VOCAB for value in ids) or (count is not None and len(ids) != count):
        raise ValueError('ID count or vocabulary range differs')
    return ids


def parse_output(text, count):
    lines = [line for line in text.splitlines() if line.startswith('ids:')]
    if len(lines) != 1 or not re.fullmatch(r'ids:(?: [0-9]+)+', lines[0]):
        raise ValueError('expected exactly one complete engine ID line')
    return parse_ids(','.join(lines[0][4:].split()), count)


def check_ids(path, count, started_ns):
    regular(path, started_ns)
    return parse_ids(path.read_text(), count)


def full_head_policy(engine, build_record=None):
    probe = subprocess.run([str(engine)], capture_output=True, text=True, timeout=30, check=False)
    usage = probe.stdout + probe.stderr
    if probe.returncode != 2:
        raise ValueError('engine usage probe must return exit code two')
    evidence = {'command': [str(engine)], 'returncode': probe.returncode,
                'usage_sha256': hashlib.sha256(usage.encode()).hexdigest(), 'binary_sha256': digest(engine),
                'mtp_prefix_capable': bool(re.search(r'(?:^|\s|\[)--mtp-prefix(?:\s|=|\]|$)', usage)),
                'jobs_capable': bool(re.search(r'(?:^|\s)gen CKPT --jobs FILE(?:\s|$)', usage))}
    record = json.loads(build_record.read_text()) if build_record is not None else None
    # A binary that was rebuilt or replaced after its audit must not pass as the recorded one.
    if record is not None and record.get('binary_sha256') != evidence['binary_sha256']:
        raise ValueError(f'engine {engine} sha256 {evidence["binary_sha256"]} does not match the build record '
                         f'{build_record} ({record.get("binary_sha256")})')
    if '--draft-head full|64k' in usage:
        return {**evidence, 'native': True, 'source': 'explicit --draft-head full'}
    if record is None:
        raise ValueError('legacy engine needs an audited --engine-build-record')
    # Keys follow tools/layout_map.tsv, so records from before the layout move still verify.
    hashes = {layout.source_key(k): v for k, v in record.get('source_hashes', {}).items()}
    source_hash = hashes.get(layout.source_key('engine/cli/gen_cli.h'))
    if (source_hash not in LEGACY_FULL_HEAD_SOURCES or record.get('binary_sha256') != digest(engine)):
        raise ValueError('legacy engine build hashes do not match an audited full-head source')
    return {**evidence, 'native': False, 'source': 'full from audited default',
            'gen_cli_sha256': source_hash, 'audited_source': LEGACY_FULL_HEAD_SOURCES[source_hash],
            'build_record_sha256': digest(build_record)}


def pin_binary(source, directory, label, build_record=None):
    # Run every capture from a private read-only copy: another process can replace the source mid-run.
    before = digest(source)
    directory.mkdir(exist_ok=True)
    target = directory / label
    shutil.copyfile(source, target)
    target.chmod(0o500)
    sha = digest(target)
    if before != sha or digest(source) != sha:
        raise ValueError(f'binary {source} changed while it was copied')
    if build_record is not None and json.loads(build_record.read_text()).get('binary_sha256') != sha:
        raise ValueError(f'engine {source} sha256 {sha} does not match the build record {build_record}')
    return target.resolve(), {'path': str(target.resolve()), 'source': str(source), 'sha256': sha}


def capture_command(engine, checkpoint, ids, dump, prompt_rows, tokens, chunk, spec, prefill,
                    native_full=True, mtp_prefix=None):
    cmd = [str(engine), 'gen', str(checkpoint), '--ids-file', str(ids), '--n', str(tokens),
           '--chunk', str(chunk), '--spec', str(spec), '--dump', str(dump),
           '--ctx', str(max(32768, prompt_rows + tokens)), '--draft-q', '0',
           '--draft-adapt', '0', '--draft-margin', '0', '--lookup', '0',
           '--draft-overlap', '0', '--draft-last-head', '1']
    # A zero must be explicit when the engine's default is GEMM prefill.
    cmd += ['--prefill', str(prefill)]
    if mtp_prefix is not None:
        cmd += ['--mtp-prefix', str(mtp_prefix)]
    if native_full is True:
        cmd += ['--draft-head', 'full', '--draft-head-auto', '0']
    return cmd


def prepare_capture(engine, checkpoint, ids, output, prompt_rows, tokens, chunk, spec, prefill, gpu_run,
                    priority=3, native_full=True, mtp_prefix=None, reference=False, queue_wait=QUEUE_WAIT_S):
    output.mkdir()  # Existing directories cannot contain trusted fresh evidence.
    dump = output / 'dump'
    dump.mkdir()
    cmd = capture_command(engine, checkpoint, ids, dump, prompt_rows, tokens, chunk, spec, prefill,
                          native_full, mtp_prefix)
    queued = [str(gpu_run), '-p', str(priority), '-w', str(queue_wait), '--', 'timeout', '-k', '10', '300', *cmd]
    start = time.time_ns()
    env = capture_environment(reference)
    record = {'command': cmd, 'queued_command': queued, 'started_ns': start,
              'environment': runtime_environment(env), 'mtp_sidecar': mtp_sidecar_record(env),
              'draft_policy': {'head': 'not used' if native_full is None else 'full',
                               'head_source': ('native flag' if native_full is True else
                                               'not used' if native_full is None else 'full from audited legacy default'),
                               'adapt': 0, 'q': 0, 'margin': 0, 'lookup': 0, 'overlap': 0,
                               'mtp_prefix': mtp_prefix}}
    save(output / 'run.json', record)
    # Serial decode appends every output row except the first. Spec dumps prompt rows only.
    return {'output': output, 'record': record, 'env': env, 'tokens': tokens, 'reference': reference,
            'rows': prompt_rows + (tokens - 1 if spec == 0 else 0)}


def finish_capture(job, returncode, ended_ns, extra=None):
    output, start = job['output'], job['record']['started_ns']
    record = {**job['record'], **(extra or {}), 'returncode': returncode, 'ended_ns': ended_ns}
    save(output / 'run.json', record)
    if returncode:
        raise ValueError(f'engine failed ({returncode}); inspect {output}/stderr.log')
    check_sidecar_log(job)
    dumps = check_dumps(output / 'dump', job['rows'], start)
    text = (output / 'stdout.log').read_text()
    generated = parse_output(text, job['tokens'])
    path = output / 'ids.txt'
    path.write_text(','.join(map(str, generated)) + '\n')
    check_ids(path, job['tokens'], start)
    return {**record, 'dump_rows': job['rows'], 'dumps': dumps, 'ids_sha256': digest(path),
            'stdout_sha256': digest(output / 'stdout.log'), 'output_ids': generated}


def capture(engine, checkpoint, ids, output, prompt_rows, tokens, chunk, spec, prefill, gpu_run,
            priority=3, native_full=True, mtp_prefix=None, reference=False, queue_wait=QUEUE_WAIT_S):
    job = prepare_capture(engine, checkpoint, ids, output, prompt_rows, tokens, chunk, spec, prefill, gpu_run,
                          priority, native_full, mtp_prefix, reference, queue_wait)
    with (output / 'stdout.log').open('wb') as stdout, (output / 'stderr.log').open('wb') as stderr:
        completed = subprocess.run(job['record']['queued_command'], stdout=stdout, stderr=stderr,
                                   env=job['env'], check=False)
    return finish_capture(job, completed.returncode, time.time_ns())


def lease_job(name, job):
    env = job['env']
    return {'name': name, 'cmd': job['record']['command'], 'timeout_s': CAPTURE_TIMEOUT_S,
            'stdout': str(job['output'] / 'stdout.log'), 'stderr': str(job['output'] / 'stderr.log'),
            'env_set': {k: v for k, v in env.items() if os.environ.get(k) != v},
            'env_unset': sorted(k for k in os.environ if k not in env),
            'estimate_scale': LEASE_ESTIMATE_SCALE.get(name, 1)}


def lease_order(schedule):
    return [item for item in schedule if item[0] not in LEASE_LAST] + [item for item in schedule if item[0] in LEASE_LAST]


def run_units_in_leases(units, args, report, done=None):
    """Run lease units (one engine process each) in as few GPU leases as fit.
    units: [(name, prepare)] in order; prepare() makes fresh output directories and returns the
    unit's lease_runner job. A unit that the lease budget cut short is prepared again, once, and runs
    in the next lease. Returns {name: (lease_runner result item, lease index)}; done (optional)
    receives each finished unit at once, so a caller can keep it when a later lease fails."""
    directory = args.output / 'leases'
    directory.mkdir()
    lease_s = getattr(args, 'lease_seconds', LEASE_LIMIT_S)
    budget_s = lease_s - LEASE_MARGIN_S
    prepare = dict(units)
    planned = {name: prepare[name]() for name, _ in units}
    pending, retried = [name for name, _ in units], set()
    done = {} if done is None else done
    report['leases'] = []
    queue_wait = getattr(args, 'queue_wait', QUEUE_WAIT_S)
    latest_ns = time.monotonic_ns() + queue_wait * 10 ** 9  # the last time a lease below the floor runs again
    while pending:
        index = len(report['leases'])
        plan, result_path = directory / f'{index}.plan.json', directory / f'{index}.result.json'
        save(plan, {'budget_s': budget_s, 'mem_floor_gb': LEASE_MEM_FLOOR_GB, 'mem_wait_s': LEASE_MEM_WAIT_S,
                    'mem_poll_s': LEASE_MEM_POLL_S, 'jobs': [planned[name] for name in pending]})
        queued = [str(args.gpu_run), '-p', str(args.priority), '-w', str(queue_wait),
                  '--', 'timeout', '-k', '10', str(lease_s + LEASE_MEM_WAIT_S),
                  sys.executable, str(ROOT / 'tools/lease_runner.py'), str(plan), str(result_path)]
        requested = time.time_ns()
        with (directory / f'{index}.log').open('wb') as log:
            completed = subprocess.run(queued, stdout=log, stderr=log, check=False)
        result = json.loads(result_path.read_text()) if result_path.exists() else {'jobs': []}
        report['leases'].append({**result, 'queued_command': queued, 'requested_ns': requested,
                                 'returncode': completed.returncode, 'ended_ns': time.time_ns()})
        save(args.output / 'report.json', report)
        if (completed.returncode == LEASE_EXIT_MEMORY and not result['jobs']
                and (result.get('memory') or {}).get('ok') is False):
            if time.monotonic_ns() < latest_ns:
                continue  # no engine started: queue a new lease at once
            raise ValueError(f'lease {index}: the memory floor of {LEASE_MEM_FLOOR_GB} GB did not come before the '
                             f'queue wait of {queue_wait} s ended; inspect {directory}/{index}.result.json.mem')
        # The runner exits 0 unless it was stopped; then a capture may have been cut off mid-write.
        if completed.returncode:
            raise ValueError(f'lease {index} exited {completed.returncode}; inspect {directory}/{index}.log')
        if not result['jobs']:
            raise ValueError(f'lease {index} ran no capture; inspect {directory}/{index}.log')
        for item in result['jobs']:
            name = item['name']
            pending.remove(name)
            if item['status'] == 'done':
                done[name] = (item, index)
                continue
            # Cut by the lease budget, or left running: run it again, once, from fresh directories.
            if name in retried:
                raise ValueError(f'capture {name} did not finish in a {budget_s} s lease budget twice '
                                 f'(last status {item["status"]})')
            retried.add(name)
            planned[name] = prepare[name]()
            pending.insert(0, name)
    return done


def capture_unit(name, kwargs, jobs):
    """A lease unit for one capture in its own process. jobs receives the prepared capture."""
    def prepare():
        if name in jobs:
            shutil.rmtree(jobs[name]['output'])
        jobs[name] = prepare_capture(**kwargs)
        return lease_job(name, jobs[name])
    return name, prepare


def run_in_leases(schedule, args, report, finished=None):
    """Run prepared captures, one process each, in as few GPU leases as fit.
    schedule: [(name, prepare_capture kwargs)] in gate order. Returns {name: finished-capture inputs}.
    finished (optional) receives (job, item, lease) of each finished capture, also when a later lease fails."""
    jobs, done = {}, {}
    try:
        run_units_in_leases([capture_unit(name, value, jobs) for name, value in lease_order(schedule)],
                            args, report, done)
    finally:
        if finished is not None:
            finished.update({name: (jobs[name], item, index) for name, (item, index) in done.items()})
    return {name: (jobs[name], item, index, {}) for name, (item, index) in done.items()}


# The canary repeats one capture first and last in its engine process. The first one is candidate:
# Gate 1 compares it with the named baseline, so it must run on a fresh load and never after another
# job on the same prompt (a kernel that reads scratch before it writes it could reuse that job's
# values). A process without candidate (bulk --prefill 0) starts with chunk8, which is cheap.
CANARY_SOURCES = ('candidate', 'chunk8')


def load_key(command):
    """The options that fix one engine load. gen --jobs refuses a job with other values."""
    def value(flag):
        return command[command.index(flag) + 1] if flag in command else None
    return value('--ctx'), value('--prefill')


def job_groups(schedule):
    """Groups captures by load. Each group is one engine process: the canary source first, the
    other captures in gate order, then the canary. Returns [(load, [names], canary source)]."""
    groups = {}
    for name, value in schedule:
        options = {k: value[k] for k in ('prompt_rows', 'tokens', 'chunk', 'spec', 'prefill', 'native_full', 'mtp_prefix')}
        command = capture_command(value['engine'], value['checkpoint'], value['ids'], Path('/dump'), **options)
        groups.setdefault(load_key(command), []).append(name)
    plans = []
    for key, names in groups.items():
        source = next((n for n in CANARY_SOURCES if n in names), names[0])
        plans.append((key, [source] + [n for n in names if n != source], source))
    return plans


def job_line(job):
    command = job['record']['command']
    return {'args': command[3:], 'stdout': str(job['output'] / 'stdout.log'),
            'stderr': str(job['output'] / 'stderr.log')}


def jobs_unit(index, order, kwargs, jobs, args, engine, checkpoint, env):
    """A lease unit that runs several captures in one engine process (gen CKPT --jobs FILE)."""
    name, directory = f'jobs{index}', args.output / 'jobs'
    def prepare():
        directory.mkdir(exist_ok=True)
        for capture in order:
            if capture in jobs:
                shutil.rmtree(jobs[capture]['output'])
            jobs[capture] = prepare_capture(**kwargs[capture])
        path = directory / f'{index}.jsonl'
        path.write_text(''.join(json.dumps(job_line(jobs[capture])) + '\n' for capture in order))
        command = [str(engine), 'gen', str(checkpoint), '--jobs', str(path)]
        return {'name': name, 'cmd': command, 'timeout_s': CAPTURE_TIMEOUT_S,
                'stdout': str(directory / f'{index}.stdout.log'), 'stderr': str(directory / f'{index}.stderr.log'),
                'env_set': {k: v for k, v in env.items() if os.environ.get(k) != v},
                'env_unset': sorted(k for k in os.environ if k not in env)}
    return name, prepare


def jobs_done(path):
    """Job indexes that the engine process reported as finished."""
    text = Path(path).read_text() if Path(path).exists() else ''
    return {int(m[1]) for m in re.finditer(r'^job (\d+) done ', text, re.M)}


def run_job_mode(schedule, separate, args, report, engine, checkpoint, finished=None):
    """Candidate captures in schedule run as gen --jobs processes, one per load, with a canary each;
    captures in separate keep their own process. All units share as few GPU leases as fit.
    Returns {name: finished-capture inputs} as run_in_leases, canaries included."""
    kwargs, jobs, units, processes = dict(schedule), {}, [], []
    env = capture_environment(False)
    for index, (key, order, source) in enumerate(job_groups(schedule)):
        canary = f'canary_{source}'
        kwargs[canary] = {**kwargs[source], 'output': args.output / canary}
        order = order + [canary]
        units.append(jobs_unit(index, order, kwargs, jobs, args, engine, checkpoint, env))
        processes.append({'name': f'jobs{index}', 'load': {'ctx': key[0], 'prefill': key[1]}, 'jobs': order,
                          'canary': {'source': source, 'name': canary}})
    units += [capture_unit(name, value, jobs) for name, value in separate]
    report['capture_mode']['processes'] = processes
    done = {}
    try:
        run_units_in_leases(units, args, report, done)
    finally:
        # A separate capture (the baseline) that finished stays usable when a later lease fails.
        if finished is not None:
            finished.update({name: (jobs[name], *done[name]) for name, _ in separate if name in done})
    result = {}
    for process in processes:
        item, index = done[process['name']]
        plan = json.loads((args.output / 'leases' / f'{index}.plan.json').read_text())
        unit = next(job for job in plan['jobs'] if job['name'] == process['name'])
        jobs_file = Path(unit['cmd'][-1])
        process.update({'jobs_file': str(jobs_file), 'jobs_sha256': digest(jobs_file), 'command': unit['cmd'],
                        'returncode': item['returncode'], 'seconds': item['seconds']})
        finished = jobs_done(unit['stdout'])
        for position, name in enumerate(process['jobs']):
            extra = {'process': {'name': process['name'], 'job_index': position, 'jobs_file': str(jobs_file),
                                 'job_done': position in finished}}
            result[name] = (jobs[name], item, index, extra)
    for name, _ in separate:
        item, index = done[name]
        result[name] = (jobs[name], item, index, {})
    return result


def check_canaries(args, report):
    """A canary repeats its source capture as the last job of the same process. Any difference
    means that state from an earlier job reached a later one."""
    checks = []
    for process in report['capture_mode']['processes']:
        source, canary = process['canary']['source'], process['canary']['name']
        try:
            compare_dumps(args.output / source / 'dump', args.output / canary / 'dump')
            if report['captures'][source]['output_ids'] != report['captures'][canary]['output_ids']:
                raise ValueError('output IDs differ')
        except ValueError as error:
            raise ValueError(f'canary {canary} differs from {source} in process {process["name"]} ({error}); '
                             'state leaked between jobs') from error
        checks.append({'process': process['name'], 'source': source, 'canary': canary, 'match': True})
    report['capture_mode']['canaries'] = checks


def baseline_points(path):
    points = {}
    text = path.read_text()
    quality = '- Quality checkpoint:' in text
    columns = 6 if quality else 3
    pattern = r'\|\s*(\d+)\s*\|' + r'\s*([\d.eE+-]+)\s*\|' * columns
    for line in text.splitlines():
        match = re.fullmatch(pattern, line)
        if match:
            layer = int(match[1])
            values = match.groups()[4:] if quality else match.groups()[1:]
            for pos, value in zip(POSITIONS, values):
                if (layer, pos) in points:
                    raise ValueError('duplicate oracle baseline point')
                points[layer, pos] = float(value)
    if set(points) != POINTS or any(not math.isfinite(v) or v <= 0 for v in points.values()):
        raise ValueError('oracle baseline must have 144 finite positive points')
    return points


def parse_oracle(text, layer):
    found = re.findall(r'layer (\d+) pos (\d+): relL2 vs fp64 (\S+)', text)
    points = {(int(l), int(p)): float(v) for l, p, v in found}
    if len(found) != 3 or set(points) != {(layer, p) for p in POSITIONS}:
        raise ValueError(f'oracle layer {layer} needs exactly positions 0,14,23')
    if any(not math.isfinite(v) or v < 0 for v in points.values()):
        raise ValueError('nonfinite or negative oracle error')
    return points


def check_oracle(points, baseline):
    if set(points) != POINTS or set(baseline) != POINTS:
        raise ValueError('oracle requires all 144 points')
    for key, value in points.items():
        if not math.isfinite(value) or value < 0 or value > min(2 * baseline[key], 1e-3):
            raise ValueError(f'oracle point exceeds limit: {key}: {value}')
    mean = sum(points.values()) / len(POINTS)
    if mean > 1.10 * sum(baseline.values()) / len(POINTS):
        raise ValueError(f'oracle mean exceeds baseline x1.10: {mean}')
    return mean


def oracle(checkpoint, dump, ids, baseline, python, output):
    points, runs = {}, []
    env = {**os.environ, 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}
    for layer in range(48):
        cmd = ['flock', '-w', '7200', str(ORACLE_LOCK), 'nice', '-n', '5', 'timeout', '300',
               str(python), str(ROOT / 'ref/oracle.py'),
               str(checkpoint), str(dump), str(layer), '--pos', '0,14,23', '--ids', str(ids)]
        start = time.time_ns()
        with (output / f'oracle-{layer:02d}.log').open('wb') as log:
            done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        if done.returncode:
            raise ValueError(f'oracle layer {layer} failed: {done.returncode}')
        path = output / f'oracle-{layer:02d}.log'
        regular(path, start)
        points = {**points, **parse_oracle(path.read_text(), layer)}
        runs.append({'command': cmd, 'started_ns': start, 'returncode': done.returncode,
                     'log_sha256': digest(path)})
    mean = check_oracle(points, baseline_points(baseline))
    return {'mean_relL2': mean, 'runs': runs, 'lock_path': str(ORACLE_LOCK),
            'baseline_sha256': digest(baseline),
            'oracle_sha256': digest(ROOT / 'ref/oracle.py'),
            'points': {f'{l}:{p}': v for (l, p), v in points.items()}}


def require_steps(steps):
    if set(steps) != {'gate1', 'gate2', 'gate3'} or not all(v is True for v in steps.values()):
        raise ValueError('all three gates must complete')


def checkpoint_identity(checkpoint):
    names = ('config.json', 'quantization_config.json', 'model.safetensors.index.json')
    return {'path': str(checkpoint), 'metadata_sha256': {name: digest(checkpoint / name) for name in names}}


def baseline_options(policy, prefill):
    # The named baseline capture: one token, chunk eight, no draft head (see take() in run).
    return {'tokens': 1, 'chunk': 8, 'spec': 0, 'prefill': prefill, 'native_full': None,
            'mtp_prefix': 0 if policy['mtp_prefix_capable'] else None}


def baseline_cache_plan(args, binary, checkpoint, ids, prompt_rows, prefill, policy):
    """Key inputs for a reusable baseline capture. The binary is the file that runs."""
    import baseline_cache as bc
    cache = Path(args.baseline_cache).expanduser().absolute()
    memo = bc.FileMemo(cache)
    env = capture_environment(reference=True)
    dump = Path('/baseline/dump')
    options = baseline_options(policy, prefill)
    command = capture_command(binary, checkpoint, ids, dump, prompt_rows, options['tokens'], options['chunk'],
                              options['spec'], options['prefill'], options['native_full'], options['mtp_prefix'])
    record = getattr(args, 'baseline_build_record', None)
    inputs = bc.key_inputs(
        binary_sha256=digest(binary), build_record_sha256=digest(record) if record else None,
        checkpoint=bc.checkpoint_files(checkpoint, memo), ids_sha256=digest(ids),
        command=bc.normalize_command(command, binary, checkpoint, ids, dump),
        environment=bc.output_environment(cache_environment(env), CACHE_PUBLIC_KNOBS), platform=bc.platform_identity(binary, env, memo))
    return {'module': bc, 'dir': cache, 'inputs': inputs, 'key': bc.cache_key(inputs), 'rows': prompt_rows,
            'binary': binary, 'checkpoint': checkpoint, 'ids': ids, 'tool_sha256': digest(Path(bc.__file__))}


def lookup_baseline_cache(plan, args):
    """Verify the entry before any capture, so the gate knows whether the baseline needs a GPU run."""
    bc = plan['module']
    found, refused = bc.lookup(plan['dir'], plan['key'], plan['inputs'], plan['rows'])
    use = bc.record_use(plan['dir'], plan['key'], args.baseline_recheck, {'output': str(args.output)}) if found else None
    return {**plan, 'found': found, 'refused': refused, 'use': use,
            'fresh_capture': found is None or use['due']}


def cached_baseline(plan, args, report, take_baseline):
    """Returns the baseline dump directory: a verified cache entry or a fresh capture."""
    bc, cache, key = plan['module'], plan['dir'], plan['key']
    found, refused, use = plan['found'], plan['refused'], plan['use']
    state = {'dir': str(cache), 'key': key, 'inputs': plan['inputs'], 'tool_sha256': plan['tool_sha256'],
             'refused': refused, 'stored': False}
    if found:
        report['baseline_source'] = 'cache'
        report['baseline_cache'] = {**state, **found, 'recheck': use}
        if use['due']:
            take_baseline()
            fresh = report['captures']['baseline']
            try:
                compare_dumps(Path(found['entry'])/'dump', args.output/'baseline/dump')
                if fresh['output_ids'] != found['output_ids']:
                    raise ValueError('output IDs differ')
            except ValueError as error:
                report['baseline_cache']['recheck'] = {**use, 'match': False}
                retired = bc.retire(cache, key, f'drift on recheck by {args.output}: {error}')
                raise ValueError(f'baseline cache drift: a fresh capture differs from cache entry {key} '
                                 f'({error}); the entry is retired to {retired}') from error
            report['baseline_cache']['recheck'] = {**use, 'match': True}
        return Path(found['entry'])/'dump'
    report['baseline_source'] = 'fresh'
    report['baseline_cache'] = state
    take_baseline()
    return store_fresh_baseline(plan, args, report, state)


def store_fresh_baseline(plan, args, report, state):
    """Copies the checked fresh baseline capture (report['captures']['baseline']) into the cache."""
    bc, cache, key = plan['module'], plan['dir'], plan['key']
    record = report['captures']['baseline']
    # Store only what the key describes: the same command and the same unchanged binary.
    command = bc.normalize_command(record['command'], plan['binary'], plan['checkpoint'], plan['ids'],
                                   args.output/'baseline/dump')
    if command != plan['inputs']['command'] or digest(plan['binary']) != plan['inputs']['binary_sha256']:
        raise ValueError('baseline capture does not match its cache key')
    try:
        entry = bc.store(cache, key, plan['inputs'], args.output/'baseline', record,
                         bc.provenance(record, args.output, report['gate_tool_sha256']))
    except OSError as error:
        # A fresh capture stays valid evidence when the cache cannot take it.
        report['baseline_cache'] = {**state, 'store_error': str(error)}
    else:
        report['baseline_cache'] = {**state, 'stored': entry is not None, 'entry': str(entry or cache/'entries'/key)}
    return args.output/'baseline/dump'


def lease_extra(item, lease):
    return {'lease': {'index': lease, 'timeout_s': item['timeout_s'], 'process_started_ns': item['started_ns'],
                      'seconds': item['seconds']}}


def keep_fresh_baseline(plan, args, report, finished):
    """A later lease failed (for example its queue wait expired). A baseline capture that finished
    and passes its checks stays valid: store it, so the next gate does not capture it again."""
    if plan is None or plan['found'] is not None or finished is None:
        return
    job, item, lease = finished
    try:
        report['captures']['baseline'] = finish_capture(job, item['returncode'], item['ended_ns'], lease_extra(item, lease))
        store_fresh_baseline(plan, args, report, {'dir': str(plan['dir']), 'key': plan['key'],
                                                  'inputs': plan['inputs'], 'tool_sha256': plan['tool_sha256'],
                                                  'refused': plan['refused'], 'stored': False})
    except (OSError, ValueError) as error:
        report['baseline_cache'] = {**report.get('baseline_cache', {}), 'early_store_error': str(error)}
    report['baseline_source'] = 'fresh'
    save(args.output / 'report.json', report)


def recheck_cache_entry(plan, report):
    value = report['baseline_cache']
    if digest(Path(plan['module'].__file__)) != plan['tool_sha256']:
        raise ValueError('baseline cache tool changed during gates')
    if report.get('baseline_source') == 'cache':
        current = plan['module'].verify(Path(value['entry']), plan['key'], plan['inputs'], plan['rows'])
        if current['manifest_sha256'] != value['manifest_sha256']:
            raise ValueError('baseline cache entry changed during gates')


def prepare_batch(args,engine,ids,baseline_engine,mode='baseline'):
    options=tuple(getattr(args,name,None) for name in
                  ('batch_gates_binary','batch_build_record','batch_prompt_list'))
    if all(value is None for value in options): return None
    if any(value is None for value in options) or args.engine_build_record is None:
        raise ValueError('batch supplement needs all three batch options and --engine-build-record')
    # The ablated-self mode has no stock reference; its batch streams compare
    # with its own (ablated) candidate dumps, as in every batch supplement.
    if baseline_engine is None and mode != 'ablated-self':
        raise ValueError('batch supplement needs a distinct named baseline binary')
    import batch_gate_report as batch
    probe,probe_record,prompts=(Path(value).resolve(strict=True) for value in options)
    build_record=args.engine_build_record.resolve(strict=True)
    files=tuple(path for path in (Path(batch.__file__).resolve(),batch.BUILDER.resolve(),engine,ids,
                baseline_engine,probe,probe_record,prompts,build_record) if path is not None)
    return {'module':batch,'inputs':{str(path):batch.file_record(path) for path in files},
            'kwargs':{'probe':probe,'probe_build_record':probe_record,'prompts':prompts,
                      'build_record':build_record,'primary_ids':ids}}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'status': 'FAIL', 'steps': {}, 'captures': {},
              'started_ns': time.time_ns(), 'gate_tool_sha256': digest(Path(__file__))}
    sidecar_source = os.environ.get('CRUCIBLE_MTP_SIDECAR')
    try:
        report['disk_free_bytes'] = disk_preflight(args.output)
        pinned = {}
        args = copy.copy(args)
        engine, pinned['candidate'] = pin_binary(args.engine.resolve(strict=True), args.output / 'bin', 'candidate',
                                                 args.engine_build_record)
        args.engine = engine
        report['pinned_binaries'] = pinned
        # Before any subprocess: a bad sidecar must fail before the GPU wait.
        report['mtp_sidecar'] = pin_sidecar(args.output / 'bin')
        checkpoint = args.checkpoint.resolve(strict=True)
        ids = args.ids.resolve(strict=True)
        prompt_rows = len(parse_ids(ids.read_text()))
        report = {**report, 'binary': str(engine), 'binary_sha256': digest(engine),
                  'checkpoint': checkpoint_identity(checkpoint), 'prompt_sha256': digest(ids),
                  'baseline_name': args.baseline_name}
        mtp_prefix = getattr(args, 'mtp_prefix', None)
        if mtp_prefix is not None and (type(mtp_prefix) is not int or not 0 <= mtp_prefix <= 262144):
            raise ValueError('MTP prefix cap must be an integer in [0,262144]')
        report['requested_mtp_prefix'] = mtp_prefix
        policy = full_head_policy(engine, args.engine_build_record)
        report['mtp_prefix'] = (0 if mtp_prefix is None else mtp_prefix) if policy['mtp_prefix_capable'] else None
        report['full_head_capability'] = policy
        if mtp_prefix is not None and not policy['mtp_prefix_capable']:
            raise ValueError('candidate usage must advertise --mtp-prefix before this option can be gated')
        save(args.output / 'report.json', report)
        if args.tokens < 2 or args.tokens > 4096:
            raise ValueError('Gate 3 needs 2 through 4096 output IDs')
        baseline_prefill = getattr(args, 'baseline_prefill_rows', 0)
        if (type(baseline_prefill) is not int or baseline_prefill < 0 or
                (baseline_prefill and not args.baseline_engine)):
            raise ValueError('baseline prefill rows need a nonnegative integer and a named baseline binary')
        report['baseline_prefill_rows'] = baseline_prefill
        report['gate1_mode'] = mode = gate1_mode(args, os.environ)
        queue_wait = getattr(args, 'queue_wait', QUEUE_WAIT_S)
        if type(queue_wait) is not int or queue_wait < 1:
            raise ValueError('--queue-wait needs a positive integer of seconds')
        report['queue_wait_s'] = queue_wait
        lease_seconds = getattr(args, 'lease_seconds', LEASE_LIMIT_S)
        if type(lease_seconds) is not int or not LEASE_LIMIT_S <= lease_seconds <= LEASE_MAX_S:
            raise ValueError(f'--lease-seconds needs an integer of {LEASE_LIMIT_S} to {LEASE_MAX_S}')
        report['lease_seconds'] = lease_seconds
        if getattr(args, 'baseline_cache', None) is not None:
            if mode != 'baseline':
                raise ValueError('the baseline cache needs a named baseline binary (--baseline-engine)')
            if type(getattr(args, 'baseline_recheck', 0)) is not int or args.baseline_recheck < 0:
                raise ValueError('--baseline-recheck needs a nonnegative integer')
        if args.prefill_rows < 0 or not args.baseline_name.strip():
            raise ValueError('prefill rows and baseline name are invalid')
        baseline_engine = None
        if args.baseline_engine:
            baseline_engine, pinned['baseline'] = pin_binary(args.baseline_engine.resolve(strict=True), args.output / 'bin',
                                                             'baseline', getattr(args, 'baseline_build_record', None))
            args.baseline_engine = baseline_engine
        batch_plan=prepare_batch(args,engine,ids,baseline_engine,mode)
        if batch_plan is not None:
            report['batch']={'status':'PENDING','report_path':str(args.output/'batch/report.json'),
                             'inputs':batch_plan['inputs']}
        if baseline_engine:
            report['baseline_binary_sha256'] = digest(baseline_engine)
            if report['baseline_binary_sha256'] == report['binary_sha256']:
                raise ValueError('candidate binary cannot serve as its own named baseline')
        elif mode == 'oracle':
            text = args.oracle_baseline.read_text()
            # The published oracle points belong to this exact 24-row prompt and checkpoint.
            match = re.search(r'- Prompt: `([^`]+)`', text)
            if (prompt_rows != 24 or not match or digest(Path(match[1])) != digest(ids)
                    or not re.search(r'- (?:Quality checkpoint|Checkpoint): `' + re.escape(str(checkpoint)) + r'`', text)):
                raise ValueError('oracle baseline prompt or checkpoint differs')
        long_plan = None
        long_options = ('long_bitwise_ids_files','long_kl_ids_files','long_baseline_engine',
                        'long_baseline_name','long_candidate_probe','long_baseline_probe','kl_max',
                        'long_resident_candidate_probe','long_resident_baseline_probe')
        if (any(getattr(args,name,None) is not None for name in long_options) or
                getattr(args,'long_prefill_rows',4096) != 4096 or getattr(args,'long_tokens',1) != 1 or
                getattr(args,'long_resident_shard','none')!='none' or
                getattr(args,'long_resident_16k_control_timeout',300)!=300):
            import long_capture,long_run
            if getattr(args,'long_baseline_engine',None) is not None:
                args.long_baseline_engine, pinned['long_baseline'] = pin_binary(
                    args.long_baseline_engine.resolve(strict=True), args.output / 'bin', 'long_baseline')
            long_plan = long_capture.prepare(args,engine)
            report['long'] = long_plan
            save(args.output/'report.json',report)
        baseline_policy = full_head_policy(baseline_engine,getattr(args,'baseline_build_record',None)) if baseline_engine else None
        report['baseline_full_head_capability'] = baseline_policy
        cache_plan = None
        if getattr(args, 'baseline_cache', None) is not None:
            cache_plan = baseline_cache_plan(args, baseline_engine, checkpoint, ids, prompt_rows,
                                             baseline_prefill, baseline_policy)
        def capture_kwargs(name, binary=engine, tokens=1, chunk=8, spec=0, prefill=0):
            return dict(engine=binary, checkpoint=checkpoint, ids=ids, output=args.output / name,
                        prompt_rows=prompt_rows, tokens=tokens, chunk=chunk, spec=spec, prefill=prefill,
                        gpu_run=args.gpu_run, priority=args.priority, queue_wait=queue_wait,
                        native_full=policy['native'] if binary == engine else None,
                        mtp_prefix=((mtp_prefix if mtp_prefix is not None else 0) if binary == engine
                                    else 0) if (policy if binary == engine else baseline_policy)['mtp_prefix_capable'] else None,
                        reference=binary != engine)
        if cache_plan:
            cache_plan = lookup_baseline_cache(cache_plan, args)
        requested_mode = getattr(args, 'capture_mode', 'auto')
        if requested_mode == 'jobs' and not policy['jobs_capable']:
            raise ValueError('candidate usage does not advertise gen CKPT --jobs FILE; use --capture-mode separate')
        used_mode = 'jobs' if policy['jobs_capable'] and requested_mode != 'separate' else 'separate'
        report['capture_mode'] = {'requested': requested_mode, 'used': used_mode,
                                  'candidate_jobs_capable': policy['jobs_capable']}
        # Job mode always runs its processes in leases.
        if lease_seconds != LEASE_LIMIT_S and not getattr(args, 'lease_batch', False) and used_mode != 'jobs':
            raise ValueError('--lease-seconds has an effect only with --lease-batch or job mode')
        # Same captures, same order, one process each; only the GPU lease grouping changes.
        schedule = [('candidate', capture_kwargs('candidate', prefill=args.prefill_rows))]
        if baseline_engine and (not cache_plan or cache_plan['fresh_capture']):
            schedule.append(('baseline', capture_kwargs('baseline', binary=baseline_engine, prefill=baseline_prefill)))
        # ablated-self always repeats; --repeat-candidate adds the same second capture to the other modes.
        repeat = mode == 'ablated-self' or bool(getattr(args, 'repeat_candidate', False))
        report['repeat_candidate'] = repeat
        if repeat:
            schedule.append(('candidate_repeat', capture_kwargs('candidate_repeat', prefill=args.prefill_rows)))
        schedule += [('chunk1', capture_kwargs('chunk1', chunk=1)), ('chunk8', capture_kwargs('chunk8', chunk=8)),
                     ('serial', capture_kwargs('serial', tokens=args.tokens, prefill=args.prefill_rows)),
                     ('spec2', capture_kwargs('spec2', tokens=args.tokens, spec=2, prefill=args.prefill_rows))]
        batched, batch_kwargs, finished = None, dict(schedule), {}
        try:
            if used_mode == 'jobs':
                # The candidate captures share one load per --ctx and --prefill; the baseline binary and a
                # self-repeat for Gate 1 keep their own process.
                shared = [(n, v) for n, v in schedule if v['engine'] == engine and n != 'candidate_repeat']
                batched = run_job_mode(shared, [(n, v) for n, v in schedule if (n, v) not in shared],
                                       args, report, engine, checkpoint, finished)
            elif getattr(args, 'lease_batch', False):
                batched = run_in_leases(schedule, args, report, finished)
        except ValueError:
            keep_fresh_baseline(cache_plan, args, report, finished.get('baseline'))
            raise
        def take(name, **options):
            if batched is not None:
                # Canaries are not in the schedule; check_canaries compares them with their source.
                if name in batch_kwargs and capture_kwargs(name, **options) != batch_kwargs[name]:
                    raise ValueError(f'capture {name} differs from the batch schedule')
                job, item, lease, extra = batched[name]
                value = finish_capture(job, item['returncode'], item['ended_ns'], {**lease_extra(item, lease), **extra})
            else:
                value = capture(**capture_kwargs(name, **options))
            report['captures'][name] = value
            save(args.output / 'report.json', report)
        take('candidate', prefill=args.prefill_rows)
        if baseline_engine and cache_plan:
            reference = cached_baseline(cache_plan, args, report,
                                        lambda: take('baseline', binary=baseline_engine, prefill=baseline_prefill))
            save(args.output / 'report.json', report)
            compare_dumps(args.output/'candidate/dump', reference)
        elif baseline_engine:
            take('baseline', binary=baseline_engine, prefill=baseline_prefill)
            compare_dumps(args.output/'candidate/dump', args.output/'baseline/dump')
        elif mode == 'oracle':
            report['oracle'] = oracle(checkpoint, args.output/'candidate/dump', ids,
                                      args.oracle_baseline, args.oracle_python, args.output)
        else:
            take('candidate_repeat', prefill=args.prefill_rows)
            compare_dumps(args.output/'candidate/dump', args.output/'candidate_repeat/dump')
        if repeat and mode != 'ablated-self':
            take('candidate_repeat', prefill=args.prefill_rows)
            compare_dumps(args.output/'candidate/dump', args.output/'candidate_repeat/dump')
        report['steps']['gate1'] = True
        take('chunk1', chunk=1)
        take('chunk8', chunk=8)
        # Without bulk prefill, 'candidate' and 'chunk8' run the same command: a free run-to-run check.
        # The flag is set only after the compare passes, so a failed report never shows the check as done.
        if args.prefill_rows == 0:
            try:
                compare_dumps(args.output/'candidate/dump', args.output/'chunk8/dump')
            except ValueError as error:
                raise ValueError(f'twin check candidate vs chunk8: {error}') from error
        report['twin_check'] = args.prefill_rows == 0
        compare_dumps(args.output/'chunk1/dump', args.output/'chunk8/dump')
        report['steps']['gate2'] = True
        take('serial', tokens=args.tokens, prefill=args.prefill_rows)
        take('spec2', tokens=args.tokens, spec=2, prefill=args.prefill_rows)
        prompt_reference = 'candidate' if args.prefill_rows else 'chunk8'
        for name in ('serial', 'spec2'):
            compare_dumps(args.output/prompt_reference/'dump', args.output/name/'dump', prefix_rows=prompt_rows)
        serial = check_ids(args.output/'serial/ids.txt', args.tokens, report['captures']['serial']['started_ns'])
        spec = check_ids(args.output/'spec2/ids.txt', args.tokens, report['captures']['spec2']['started_ns'])
        if serial != spec:
            raise ValueError('Gate 3 serial and speculative IDs differ')
        report['steps']['gate3'] = True
        if used_mode == 'jobs':
            for process in report['capture_mode']['processes']:
                take(process['canary']['name'])
            check_canaries(args, report)
        require_steps(report['steps'])
        if long_plan is not None:
            long_run.run(args,long_plan,args.output/'long',policy)
        if batch_plan is not None:
            batch=batch_plan['module']
            batch.recheck(batch_plan['inputs'])
            value=batch.run(checkpoint=checkpoint,engine=engine,output=args.output/'batch',
                            baseline_dump=args.output/'candidate/dump',gpu_run=args.gpu_run,
                            priority=args.priority,queue_wait=queue_wait,
                            # The record's -o names the built file; the run and every sha use the pinned copy.
                            engine_built=Path(pinned['candidate']['source']),**batch_plan['kwargs'])
            if not isinstance(value,dict) or value.get('status')!='PASS' or value.get('supplemental') is not True:
                raise ValueError('batch supplement did not complete all raw artifact checks')
            report['batch']={**report['batch'],**value}
            batch.recheck(batch_plan['inputs'])
        if digest(Path(__file__)) != report['gate_tool_sha256']:
            raise ValueError('gate tool changed during captures')
        # The source files are only informational: the captures ran from the pinned copies.
        for entry in pinned.values():
            entry['source_sha256_at_end'] = digest(Path(entry['source']))
        if digest(engine) != report['binary_sha256'] or checkpoint_identity(checkpoint) != report['checkpoint']:
            raise ValueError('engine or checkpoint metadata changed during gates')
        if baseline_engine and digest(baseline_engine) != report['baseline_binary_sha256']:
            raise ValueError('baseline binary changed during gates')
        if digest(ids) != report['prompt_sha256']:
            raise ValueError('prompt changed during gates')
        if mtp_sidecar_record(os.environ) != report['mtp_sidecar']:
            raise ValueError('MTP sidecar changed during gates')
        if cache_plan:
            recheck_cache_entry(cache_plan, report)
        for name, value in report['captures'].items():
            current = check_dumps(args.output/name/'dump', value['dump_rows'], value['started_ns'])
            if current != value['dumps']:
                raise ValueError(f'dumps changed after capture: {name}')
            if check_ids(args.output/name/'ids.txt', len(value['output_ids']), value['started_ns']) != value['output_ids']:
                raise ValueError(f'IDs changed after capture: {name}')
            if digest(args.output/name/'stdout.log') != value['stdout_sha256']:
                raise ValueError(f'engine log changed after capture: {name}')
        report['status'] = 'PASS'
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report['error'] = str(error)
        raise
    finally:
        if sidecar_source is None:
            os.environ.pop('CRUCIBLE_MTP_SIDECAR', None)
        else:
            os.environ['CRUCIBLE_MTP_SIDECAR'] = sidecar_source
        save(args.output / 'report.json', {**report, 'ended_ns': time.time_ns()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--engine-build-record', type=Path, help='hashed candidate build JSON; required for legacy and batch modes')
    ap.add_argument('--baseline-build-record',type=Path,help='audited legacy build record for a named baseline')
    ap.add_argument('--batch-gates-binary',type=Path,help='qualified supplemental batch probe, after primary gates')
    ap.add_argument('--batch-build-record',type=Path,help='probe build JSON from the fixed dual builder')
    ap.add_argument('--batch-prompt-list',type=Path,help='eight paths, one per line; first must equal --ids')
    ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baseline-name', required=True, help='baseline commit or named oracle source')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--ablated-numerics', action='store_true',
                       help='CRUCIBLE_ABLATE candidate: Gate 1 repeats the candidate capture bit for bit')
    group.add_argument('--baseline-engine', type=Path, help='capture a fresh named baseline for Gate 1(a)')
    group.add_argument('--oracle-baseline', type=Path, help='144-point Markdown table for Gate 1(b)')
    ap.add_argument('--oracle-python', type=Path, default=work_paths.refenv_python())
    ap.add_argument('--gpu-run', dest='gpu_run', type=Path, default=work_paths.gpu_run())
    ap.add_argument('--priority', type=int, choices=(0, 1, 2, 3), default=3)
    ap.add_argument('--prefill-rows', type=int, default=0, help='bulk rows for Gate 1 and live Gate 3 continuation')
    ap.add_argument('--baseline-prefill-rows', type=int, default=0,
                    help='explicit bulk rows for the independent named Gate 1(a) baseline; default chunk-eight path')
    ap.add_argument('--tokens', type=int, default=160)
    ap.add_argument('--mtp-prefix', type=int, default=None,
                    help='candidate cold MTP warmup cap, integer 0..262144; 0 retains full warmup')
    ap.add_argument('--long-bitwise-ids-files',type=Path,nargs='+',help='fresh native final full-logit/complete-ID equality')
    ap.add_argument('--long-kl-ids-files',type=Path,nargs='+',help='clipped sampled KL and separate native final KL')
    ap.add_argument('--long-baseline-engine',type=Path,help='independent named main for long checks; defaults to Gate 1(a) baseline')
    ap.add_argument('--long-baseline-name',help='required with a separate long baseline')
    ap.add_argument('--long-candidate-probe',type=Path,help='diagnostic adapter bound to the candidate build')
    ap.add_argument('--long-baseline-probe',type=Path,help='diagnostic adapter bound to independent main')
    ap.add_argument('--long-resident-candidate-probe',type=Path,help='resident producer bound to the candidate source')
    ap.add_argument('--long-resident-baseline-probe',type=Path,help='resident producer bound to the independent main source')
    ap.add_argument('--long-resident-shard',choices=('none','control','all'),default='none',
                    help='split selected resident roles into one request per GPU grant')
    ap.add_argument('--long-resident-16k-control-timeout',type=int,choices=(300,600),default=300,
                    help='parent-authorized 16K control exception: 600 seconds at priority two')
    ap.add_argument('--long-prefill-rows',type=int,default=4096)
    ap.add_argument('--long-tokens',type=int,default=1,help='native long generation count, separate from primary Gate 3')
    ap.add_argument('--kl-max',type=float,help='optional absolute directional aggregate median limit')
    ap.add_argument('--baseline-cache', type=Path, nargs='?', const=BASELINE_CACHE_DIR, default=None,
                    help=f'reuse verified named-baseline captures from DIR (default {BASELINE_CACHE_DIR})')
    ap.add_argument('--baseline-recheck', type=int, default=0, metavar='N',
                    help='with --baseline-cache: capture 1 of every N reuses fresh and compare bit for bit')
    ap.add_argument('--lease-batch', action='store_true',
                    help='run the captures (one process each) in as few GPU leases of 300 s as fit')
    ap.add_argument('--capture-mode', choices=('auto', 'jobs', 'separate'), default='auto',
                    help='jobs: the candidate captures share one engine load per --ctx and --prefill '
                         '(gen CKPT --jobs FILE), with a canary repeat in each process; auto uses jobs '
                         'when the candidate usage advertises it; separate: one process per capture')
    ap.add_argument('--queue-wait', type=int, default=QUEUE_WAIT_S, metavar='SECONDS',
                    help=f'gpu-run wait for the GPU lock on every ticket (default {QUEUE_WAIT_S}); a full queue can need more')
    ap.add_argument('--lease-seconds', type=int, default=LEASE_LIMIT_S, metavar='SECONDS',
                    help=f'with --lease-batch: length of each GPU lease, {LEASE_LIMIT_S} to {LEASE_MAX_S} '
                         f'(default {LEASE_LIMIT_S}); 600 fits all six captures of a 4.05 bulk gate')
    ap.add_argument('--repeat-candidate', action='store_true',
                    help='capture the candidate a second time; the dumps must be bitwise equal (costs one capture)')
    ap.add_argument('--keep-dumps', action='store_true',
                    help='keep dump binaries after PASS; by default they are deleted once report.json has their hashes')
    args = ap.parse_args()
    try:
        run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print(f'PASS: all three gates; evidence: {args.output}/report.json')
    if not args.keep_dumps:
        try:
            pruned = prune_dumps(args.output)
        except (OSError, ValueError) as error:
            print(f'warning: dumps not pruned: {error}', file=sys.stderr)
        else:
            print(f'pruned {pruned["files"]} dump files ({pruned["bytes"] / 10**9:.1f} GB); hashes stay in report.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
