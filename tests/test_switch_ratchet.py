"""Ratchets for the switch registry: raw environment reads and the switch count may only go down.

engine/switches.h is the only reader of the environment. RAW_READS lists the raw reads in engine/ and kernels/
that existed when the ratchet landed (Oct 10, train 17 b8c28f64). The list may only get shorter: a new raw read
fails, and an entry that no longer matches the code fails too, so a cleanup must delete its entry.
SWITCH_CEILING is the number of distinct CRUCIBLE_* names in engine/ and kernels/. It may only go down: a new name
fails, and a removed name fails until the ceiling is lowered to the new count.
"""
from collections import Counter
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = 'engine/switches.h'
# A raw read: getenv (also std::getenv and secure_getenv) or the environ array.
RAW_READ = re.compile(r'\b(?:secure_)?getenv\s*\(\s*([^()]*?)\s*\)|\b(environ)\b')

# Lower it when a switch name leaves engine/ and kernels/. Never raise it. Target: 65 or fewer (user, Oct 10).
SWITCH_CEILING = 56  # merge rule: retire-pf19 left 54, CRUCIBLE_DEBUG_NONFINITE adds one,
# CRUCIBLE_PLE_BF16 (16-bit n-gram table) adds one

# (file, argument of the raw read): number of reads. Delete entries as reads move to engine/switches.h.
RAW_READS = {}

# bench/ C++ sources (Oct 10, 46 reads): the same rule, a separate allow-list. Benches may read their own
# non-CRUCIBLE variables; a cleanup moves CRUCIBLE_* reads to engine/switches.h and deletes the entry.
BENCH_SUFFIXES = ('.h', '.hip', '.cpp', '.cc', '.inc')
BENCH_RAW_READS = {
    ('bench/batch_spec_trials.h', '"CRUCIBLE_PROFILE"'): 1,
    ('bench/compact_transition_gate.hip', '"CRUCIBLE_COMPACT_STATE"'): 1,
    ('bench/forward_rows.hip', '"FORWARD_ROWS_PROFILE"'): 1,
    ('bench/gdn_scan_pf.hip', '"GDN_HEADS"'): 2,
    ('bench/packed_rows.hip', '"VERIFY_SPLITS"'): 1,
    ('bench/prefill_dense_half.hip', '"DH_22"'): 1,
    ('bench/prefill_dense_half.hip', '"DH_HALF"'): 1,
    ('bench/prefill_dense_half.hip', '"DH_X22"'): 1,
    ('bench/prefill_gdn.hip', '"PREFILL_INITIAL"'): 1,
    ('bench/prefill_gdn.hip', '"PREFILL_LANES"'): 1,
    ('bench/prefill_gdn.hip', '"PREFILL_MICRO"'): 1,
    ('bench/prefill_gdn.hip', '"PREFILL_PACKED"'): 1,
    ('bench/prefill_gdn.hip', '"PREFILL_PERSISTENT"'): 1,
    ('bench/prefill_gdn.hip', '"PREFILL_PERSISTENT_REF"'): 1,
    ('bench/prefill_gdn.hip', '"PREFILL_PROFILE"'): 2,
    ('bench/prefill_qsa.hip', '"QSA_KO"'): 3,
    ('bench/qsa_trace.hip', '"CRUCIBLE_QSA_MIGRATION_SETTLE"'): 1,
    ('bench/qsa_trace.hip', '"CRUCIBLE_QSA_READY_FILE"'): 1,
    ('bench/qsa_trace.hip', '"HSA_XNACK"'): 1,
    ('bench/quality_startup.hip', '"CRUCIBLE_SETTLE"'): 1,
    ('bench/spec_sweep.hip', '"CRUCIBLE_SETTLE"'): 2,
    ('bench/spec_sweep.hip', '"CRUCIBLE_SWEEP_CASE_COUNTERS"'): 1,
    ('bench/spec_sweep.hip', '"CRUCIBLE_SWEEP_IDS"'): 1,
    ('bench/spec_sweep.hip', '"CRUCIBLE_SWEEP_TIMINGS"'): 1,
    ('bench/statecache_gate.hip', '"CRUCIBLE_STATECACHE_GATE_DENSE"'): 1,
}


def sources(directories=('engine', 'kernels'), suffixes=None):
    for directory in directories:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and (suffixes is None or path.suffix in suffixes):
                yield path.relative_to(ROOT).as_posix(), path.read_text(errors='replace')


def raw_reads(directories=('engine', 'kernels'), suffixes=None):
    found = Counter()
    for name, text in sources(directories, suffixes):
        if name != REGISTRY:
            for match in RAW_READ.finditer(text):
                found[(name, match.group(1) or match.group(2))] += 1
    return found


def switch_names():
    return {name for _, text in sources() for name in re.findall(r'CRUCIBLE_[A-Z0-9_]+', text)}


class SwitchRatchetTests(unittest.TestCase):
    def test_no_new_raw_environment_read(self):
        new = {key: count for key, count in raw_reads().items() if count > RAW_READS.get(key, 0)}
        self.assertEqual(new, {}, f'read the environment through {REGISTRY}, not getenv or environ')

    def test_no_stale_raw_read_entry(self):
        found = raw_reads()
        stale = {key: count for key, count in RAW_READS.items() if found.get(key, 0) < count}
        self.assertEqual(stale, {}, 'a raw read is gone: delete or lower its RAW_READS entry')

    def test_bench_raw_reads_only_go_down(self):
        found = raw_reads(('bench',), BENCH_SUFFIXES)
        new = {key: count for key, count in found.items() if count > BENCH_RAW_READS.get(key, 0)}
        stale = {key: count for key, count in BENCH_RAW_READS.items() if found.get(key, 0) < count}
        self.assertEqual(new, {}, f'read the environment through {REGISTRY}, not getenv or environ')
        self.assertEqual(stale, {}, 'a bench raw read is gone: delete or lower its BENCH_RAW_READS entry')

    def test_switch_count_only_goes_down(self):
        count = len(switch_names())
        self.assertLessEqual(count, SWITCH_CEILING, 'a new CRUCIBLE_* switch needs a removed one')
        self.assertEqual(count, SWITCH_CEILING, f'the count fell to {count}: lower SWITCH_CEILING to it')


if __name__ == '__main__':
    unittest.main()
