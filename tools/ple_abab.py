#!/usr/bin/env python3
"""Interleaved A/B timing for CRUCIBLE_PLE_HOST inside one GPU lease.

Usage: ple_abab.py ENGINE CKPT IDS OUT.jsonl --ab off,hot --pairs 6 [--budget 280] [-- extra gen args]
       ple_abab.py ENGINE CKPT IDS OUT.jsonl --configs off,hot,hot,off [...]   (an explicit run list)
       ple_abab.py ... -- pfbench RUN...   (crucible pfbench; each RUN reports its second, rows-hot pass)
       ple_abab.py --summary OUT.jsonl --ab off,hot   (the order-corrected table of a finished run; no GPU)
The second arm of a pair runs about 0.7% faster (Oct 25), so --ab X,Y --pairs N runs ABBA blocks (X Y, Y X, Y X, X Y, ...)
with equal numbers of XY and YX pairs; an odd N is rounded up to the next even number. An explicit --configs list is
read as consecutive pairs and the tool names a list whose pairs are not balanced. The summary gives the order-corrected
B/A and the order term (tools/ab_order.py); quote the corrected ratio.
Configs: off (resident table), hot (host gather, page cache as left by the previous run),
cold (host gather after POSIX_FADV_DONTNEED on the n-gram file, so every row is a first touch),
warm (host gather after build/bench/ple_host_cold reads the prompt's rows into the page cache).
Each run records prefill and decode tok/s from the engine, the busy sclk mean (samples above
1 GHz from amdgpu hwmon, every 20 ms), the sclk median, the energy (J) and mean socket power (W) of the busy samples
(fastgate_lib.ClockSampler), the platform profile, the pair index and order, and the free device memory the engine
reports after warmup.
"""
import json, os, re, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_order  # noqa: E402
import fastgate_lib as lib  # noqa: E402

# None on a host without the GPU (the tests and the summary run there)
HWMON = next(iter(Path('/sys/class/drm/card1/device/hwmon').glob('hwmon*/freq1_input')), None)


def clock_sampler():
    return lib.ClockSampler()


def read_platform_profile():
    return lib.read_platform_profile()


def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


WARM_TOOL = Path(__file__).resolve().parents[1] / 'build/bench/ple_host_cold'
extra_ids = None


def warm(ckpt, ids_file):
    # Row geometry of the two packs.
    geometry = ('26736', '102', '320001536') if '3.05' in str(ckpt) else ('6328', '122', '320001536')
    subprocess.run([str(WARM_TOOL), str(Path(ckpt) / 'ngram_embedding.safetensors'), *geometry, ids_file],
                   check=True, capture_output=True)


def sample(stop, out):
    while HWMON and not stop.is_set():
        try:
            out.append(int(HWMON.read_text()) / 1e6)
        except OSError:
            pass
        time.sleep(0.02)


def disk_read_mb_s(seconds=1.0):
    """Other processes' NVMe reads just before a run: cold rows queue behind them."""
    def sectors():
        return int(Path('/sys/block/nvme0n1/stat').read_text().split()[2])
    a = sectors()
    time.sleep(seconds)
    return round((sectors() - a) * 512 / seconds / 1e6, 1)


def run(engine, ckpt, ids, config, extra):
    env = dict(os.environ, CRUCIBLE_PLE_HOST='0' if config == 'off' else '1')
    if config == 'cold':
        drop(Path(ckpt) / 'ngram_embedding.safetensors')
    if config == 'warm':  # the prompt's rows resident, as in a long-lived server
        warm(ckpt, extra_ids)
    disk_before = disk_read_mb_s()
    samples, stop = [], threading.Event()
    t = threading.Thread(target=sample, args=(stop, samples))
    t.start()
    start = time.time()
    clock_cm = clock_sampler()
    clock_cm.__enter__()
    if extra and extra[0] == 'pfbench':  # pfbench RUN...: the second pass of each run, rows hot
        cmd = [engine, 'pfbench', ckpt, '--prefill', '4096', *extra[1:]]
    else:
        cmd = [engine, 'gen', ckpt, '--ids', ids, *extra]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    pf = re.findall(r'^(\S+): (\d+) tok .* ([\d.]+) s ([\d.]+) tok/s$', p.stdout, re.M)
    stop.set()
    t.join()
    clock = clock_cm.summary(start)
    clock_cm.__exit__(None, None, None)
    m = re.search(r'prefill (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\), decode (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\)',
                  p.stdout + p.stderr)
    free = re.search(r'memory ready: device free (\d+)', p.stderr)
    avail = re.search(r'memory ready: MemAvailable:\s+(\d+) kB', p.stderr)
    busy = [s for s in samples if s > 1000]
    # CRUCIBLE_PLE_TRACE lines: time the forward waited for the CPU gather.
    trace = [(int(a), float(b), float(c)) for a, b, c in
             re.findall(r'ple host gather: (\d+) rows, ([\d.]+) ms since start, ([\d.]+) ms waited', p.stderr)]
    ids_line = re.search(r'^ids: (.*)$', p.stdout, re.M)
    return {
        'config': config, 'rc': p.returncode, 'wall_s': round(time.time() - start, 2),
        'prefill_tok_s': float(m[3]) if m else None, 'prefill_s': float(m[2]) if m else None,
        'decode_tok_s': float(m[6]) if m else None, 'decode_tokens': int(m[4]) if m else None,
        'sclk_busy_mhz': round(sum(busy) / len(busy), 1) if busy else None, 'sclk_samples': len(busy),
        'sclk_median_mhz': clock.get('median_mhz'), 'energy_j': clock.get('energy_j'), 'mean_w': clock.get('mean_w'),
        'platform_profile': read_platform_profile(),
        'device_free': int(free[1]) if free else None, 'mem_available_kb': int(avail[1]) if avail else None,
        'ids_sha': __import__('hashlib').sha256(ids_line[1].encode()).hexdigest()[:16] if ids_line else None,
        'disk_read_mb_s_before': disk_before,
        'pfbench': [dict(tag=t, tokens=int(n), s=float(x), tok_s=float(y)) for t, n, x, y in pf],
        'gather_big_ms': [round(b, 3) for r, b, _ in trace if r >= 64],
        'wait_big_ms': [round(c, 3) for r, _, c in trace if r >= 64],
        'wait_small_ms_sum': round(sum(c for r, _, c in trace if r < 64), 3),
        'stderr_tail': p.stderr[-400:] if p.returncode else '',
    }


def plan(argv):
    """The run list from --ab X,Y --pairs N (ABBA blocks, even N) or from --configs (as given)."""
    if '--ab' in argv:
        first, second = argv[argv.index('--ab') + 1].split(',')
        wanted = int(argv[argv.index('--pairs') + 1]) if '--pairs' in argv else 6
        pairs, rounded = ab_order.even_pairs(wanted)
        if rounded:
            print(f'ple_abab: --pairs {wanted} is odd; running {pairs} pairs so that the {first} {second} and '
                  f'{second} {first} pairs are equal in number (the second arm of a pair runs about 0.7% faster).', flush=True)
        return ab_order.sequence(first, second, pairs)
    configs = argv[argv.index('--configs') + 1].split(',')
    pairs = [(configs[i], configs[i + 1]) for i in range(0, len(configs) - 1, 2)]
    if any(pairs.count(p) != pairs.count(p[::-1]) for p in pairs):
        print('ple_abab: the pairs of --configs are not balanced (unequal XY and YX counts); use --ab X,Y --pairs N', flush=True)
    return configs


def summarize(path, a, b):
    """(text, aggregate) of a finished out file: the order-corrected comparison of config a (A) and config b (B). A pair
    needs one run of each config with rc 0. The order of a pair is AB when a ran first."""
    import abab
    by_pair = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        by_pair.setdefault(r.get('pair'), []).append(r)
    runs_a, runs_b, orders = [], [], []
    for pair, rows in sorted(by_pair.items(), key=lambda kv: kv[0]):
        got = {r['config']: r for r in rows if r.get('rc') == 0}
        if a not in got or b not in got:
            continue
        runs_a.append(got[a])
        runs_b.append(got[b])
        orders.append('AB' if rows[0]['config'] == a else 'BA')
    def view(r):
        return dict(prefill_tok_s=r.get('prefill_tok_s'), decode_tok_s=r.get('decode_tok_s'),
                    sclk_mhz=r.get('sclk_median_mhz'), energy_j=r.get('energy_j'), mean_w=r.get('mean_w'))
    va, vb = [view(r) for r in runs_a], [view(r) for r in runs_b]
    metrics = [m for m in (('prefill_tok_s', 'prefill (tok/s)'), ('decode_tok_s', 'decode (tok/s)'))
               if all(v[m[0]] for v in va + vb)]
    agg = abab.aggregate(va, vb, metrics, orders=orders)
    profiles = sorted({r['platform_profile'] for r in runs_a + runs_b if r.get('platform_profile')})
    return f'A = {a}, B = {b}\n' + abab.format_table(agg, len(orders), orders, profiles), agg


def main():
    argv = sys.argv[1:]
    if '--summary' in argv:
        a, b = argv[argv.index('--ab') + 1].split(',')
        print(summarize(argv[argv.index('--summary') + 1], a, b)[0])
        return
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    engine, ckpt, ids_file, out = argv[:4]
    configs = plan(argv)
    budget = float(argv[argv.index('--budget') + 1]) if '--budget' in argv else 280
    global extra_ids
    extra_ids = ids_file
    ids = ','.join(re.findall(r'\d+', Path(ids_file).read_text()))
    began, longest = time.time(), 0.0
    with open(out, 'a') as f:
        for idx, config in enumerate(configs):
            # Stop only between pairs: a half pair would leave one order unbalanced.
            if idx % 2 == 0 and time.time() - began + longest * 2.4 > budget:
                print('stop: lease budget', flush=True)
                break
            r = run(engine, ckpt, ids, config, extra)
            longest = max(longest, r['wall_s'])
            r['pair'] = idx // 2
            r['order'] = 'AB' if configs[idx // 2 * 2] == configs[0] else 'BA'
            r['time'] = time.strftime('%H:%M:%S')
            f.write(json.dumps(r) + '\n')
            f.flush()
            print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
