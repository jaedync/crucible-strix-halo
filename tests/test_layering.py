"""Include direction and file ownership for engine/ and kernels/.

Rules (tools/layout.py): includes go down only (gfx1151 < exl3 < layers <
model/flashnext < runtime < serve < cli < engine.hip < bench/kernels); a
kernel header includes only kernel headers, gfx1151 helpers and leaves;
engine code never includes bench/ or tests/; every quoted include resolves to
a tracked file. The symbol ratchet (tools/layout_symbols.py) adds one
more rule: no use of a name that only a higher module defines.

BASELINE holds the violations of Oct 10 (b8c28f64 with main f9d61771, after
main's map decisions). It may only shrink: the test fails on a new violation
and on an entry that no longer occurs. Keys use
the target paths of tools/layout_map.tsv, so they stay valid after the move.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import layout  # noqa: E402
import layout_symbols  # noqa: E402

BASELINE = frozenset({
    'kernel: engine/layers/kernels/hc_multirow.h -> engine/layers/ops.h',
    'kernel: engine/layers/kernels/prefill_gdn.h -> engine/layers/numerics.h',
    'kernel: engine/layers/kernels/prefill_gdn.h -> engine/layers/ops.h',
    'kernel: engine/layers/kernels/prefill_moe.h -> engine/model/flashnext/prefill_launch.h',
    'kernel: engine/layers/kernels/prefill_qsa.h -> engine/layers/ops.h',
    'kernel: engine/layers/kernels/prefill_qsa_score.h -> engine/layers/qsa_long.h',
    'upward: engine/layers/persistent_gdn.h -> engine/model/flashnext/model.h',
    'upward: engine/layers/prefill_moe.h -> engine/model/flashnext/model.h',
})

# Top-level directories with a charter. kernels/ leaves with the layout move.
TOP_LEVEL = {'bench', 'deploy', 'docs', 'engine', 'ref', 'server',
             'tests', 'tools', 'training'} | ({'kernels'} if (layout.ROOT / 'kernels').is_dir() else set())

HOW_TO_FIX = (
    'Include only your own module or a lower one (see tools/layout.py RANKS and '
    'the module README charters). Do not add a baseline entry for new code.')

# Symbol ratchet. engine.hip is one translation unit, so a file can use a name that a
# higher module defines without any #include. SYMBOL_BASELINE holds the uses of Oct 11 (train 19
# head c17a47f1 plus a later move). It may only shrink. The fix for an entry is to move the
# definition down (DraftAcceptProb and the prefill row limits).
SYMBOL_BASELINE = frozenset({
    # cli code is textually inside engine.hip, after class Engine and its helpers (rank 7).
    'symbol: engine/cli/batch_cli.h uses Engine (engine/engine.hip)',
    'symbol: engine/cli/batch_cli.h uses Now (engine/engine.hip)',
    'symbol: engine/cli/gen_cli.h uses CopyRows (engine/engine.hip)',
    'symbol: engine/cli/gen_cli.h uses Engine (engine/engine.hip)',
    'symbol: engine/cli/gen_cli.h uses Now (engine/engine.hip)',
    'symbol: engine/cli/gen_cli.h uses ParseIds (engine/engine.hip)',
    'symbol: engine/cli/gen_jobs.h uses Engine (engine/engine.hip)',
    'symbol: engine/cli/gen_jobs.h uses Now (engine/engine.hip)',
    'symbol: engine/cli/prefill_bench.h uses Engine (engine/engine.hip)',
    'symbol: engine/cli/prefill_bench.h uses Now (engine/engine.hip)',
    'symbol: engine/cli/prefill_bench.h uses ParseIds (engine/engine.hip)',
    # Same facts as the include baseline above (model structs used by layer code).
    'symbol: engine/layers/kernels/prefill_moe.h uses kMaxPrefillLaunchRows (engine/model/flashnext/prefill_launch.h)',
    'symbol: engine/layers/persistent_gdn.h uses Hc (engine/model/flashnext/model.h)',
    'symbol: engine/layers/prefill_moe.h uses Lin (engine/model/flashnext/model.h)',
    # Runtime fragments and serve use class Engine and its free helpers from engine.hip.
    'symbol: engine/runtime/conversation_snapshot.h uses Engine (engine/engine.hip)',
    'symbol: engine/runtime/spec_steps.h uses Now (engine/engine.hip)',
    'symbol: engine/serve/batched_serve.h uses CopyRows (engine/engine.hip)',
    'symbol: engine/serve/batched_serve.h uses Engine (engine/engine.hip)',
    'symbol: engine/serve/serve.h uses Engine (engine/engine.hip)',
    # The serial serve loop calls the CLI generation loop. The serial path retires in train 20.
    'symbol: engine/serve/serve.h uses RunGen (engine/cli/gen_cli.h)',
})

HOW_TO_FIX_SYMBOL = (
    'A module may not use a name that only a higher module defines (tools/layout.py RANKS). '
    'Move the definition down to the lowest module that uses it. Do not add an allow-list '
    'entry for new code. Allow-list now:\n  ' + '\n  '.join(sorted(SYMBOL_BASELINE)))


def owned_paths(tracked):
    return sorted(p for p in tracked if p.startswith(layout.SOURCE_DIRS)
                  and not p.endswith('/README.md'))


class LayeringRatchetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = frozenset(layout.violations())

    def test_no_new_violation(self):
        new = sorted(self.found - BASELINE)
        self.assertEqual(new, [], f'new include violations. {HOW_TO_FIX}')

    def test_baseline_has_no_stale_entry(self):
        stale = sorted(BASELINE - self.found)
        self.assertEqual(stale, [], 'fixed violations: delete them from BASELINE')


class SymbolRatchetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = frozenset(layout_symbols.violations())

    def test_no_new_symbol_use(self):
        new = sorted(self.found - SYMBOL_BASELINE)
        self.assertEqual(new, [], f'new symbol uses. {HOW_TO_FIX_SYMBOL}')

    def test_symbol_baseline_has_no_stale_entry(self):
        stale = sorted(SYMBOL_BASELINE - self.found)
        self.assertEqual(stale, [], 'fixed symbol uses: delete them from SYMBOL_BASELINE')


class OwnershipTest(unittest.TestCase):
    def setUp(self):
        self.rows = layout.load_map()
        self.tracked = layout.tracked_files()

    def test_every_file_has_exactly_one_row(self):
        lay = layout.Layout(self.rows)
        missing = [p for p in owned_paths(self.tracked) if lay.row(p) is None]
        self.assertEqual(missing, [], f'add a row to {layout.MAP_PATH} for each new file')

    def test_rows_are_unique(self):
        for column in ('old', 'new'):
            values = [getattr(r, column) for r in self.rows]
            dupes = sorted({v for v in values if values.count(v) > 1})
            self.assertEqual(dupes, [], f'duplicate {column} paths in {layout.MAP_PATH}')

    def test_no_stale_row(self):
        stale = [r.old for r in self.rows
                 if r.old not in self.tracked and r.new not in self.tracked]
        self.assertEqual(stale, [], f'delete rows of removed files from {layout.MAP_PATH}')


def ownership_table(text):
    """Rows of the table under '## Code ownership' in docs/ARCHITECTURE.md."""
    section = text.split('## Code ownership', 1)[1].split('\n## ', 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip().strip('`') for c in line.strip().strip('|').split('|')]
        if line.startswith('| `') and len(cells) == 5:
            rows.append(cells)
    return rows


class ArchitectureTableTest(unittest.TestCase):
    """Each file -> one map row -> one module -> one ownership table row."""

    def test_each_module_has_exactly_one_row(self):
        text = (layout.ROOT / 'docs/ARCHITECTURE.md').read_text(encoding='utf-8')
        rows = ownership_table(text)
        modules = [r[1] for r in rows]
        self.assertEqual(sorted(modules), sorted(layout.RANKS), 'one table row per module')
        lay = layout.Layout(layout.load_map())
        tracked = layout.tracked_files()
        for directory, module, rank, charter, _ in rows:
            with self.subTest(module=module):
                self.assertEqual(directory, layout.module_dir(module) + '/')
                self.assertEqual(int(rank), layout.RANKS[module])
                source = lay.by_new.get(charter)
                self.assertTrue(charter in tracked or (source and source.old in tracked),
                                f'{charter}: no charter file and no map row that moves one')

    def test_engine_charters_name_no_full_switch(self):
        # Charters move into engine/, where the switch audits read every file
        # (server/tests/test_train13_audit.py). Write switch names without the prefix.
        lay = layout.Layout(layout.load_map())
        charters = [p for p in layout.tracked_files()
                    if (lay.canonical(p) or p).startswith(layout.SOURCE_DIRS) and p.endswith('.md')]
        self.assertTrue(charters)
        for path in charters:
            text = (layout.ROOT / path).read_text(encoding='utf-8')
            self.assertEqual(re.findall(r'CRUCIBLE_[A-Z0-9_]+', text), [], path)

    def test_leaves_include_only_the_standard_library(self):
        lay = layout.Layout(layout.load_map())
        leaves = [p for p in layout.tracked_files() if lay.canonical(p) in layout.LEAVES]
        self.assertTrue(leaves)
        for path in leaves:
            text = (layout.ROOT / path).read_text(encoding='utf-8')
            system = re.findall(r'^[ \t]*#[ \t]*include[ \t]*<([^>]+)>', text, re.M)
            self.assertEqual(layout.quoted_includes(text), [], path)
            self.assertEqual([h for h in system if '/' in h or h.startswith('hip')], [], path)

    def test_every_top_level_directory_has_a_charter(self):
        tracked = layout.tracked_files()
        missing = sorted(t for t in TOP_LEVEL if f'{t}/README.md' not in tracked)
        self.assertEqual(missing, [], 'add a README.md charter to each top-level directory')
        if (layout.ROOT / '.git').exists():  # a git archive tree also holds work directories
            tops = {p.split('/', 1)[0] for p in tracked if '/' in p and not p.startswith('.')}
            self.assertEqual(sorted(tops), sorted(TOP_LEVEL), 'a new top-level directory: add it here')


def row(path, module):
    return layout.Row(path, path, module, '', '')


class IncludeRuleTest(unittest.TestCase):
    def test_rules(self):
        cases = [
            (row('engine/layers/kernels/a.h', 'layers/kernels'),
             row('engine/gfx1151/common.h', 'gfx1151'), None),
            (row('engine/layers/kernels/a.h', 'layers/kernels'),
             row('engine/exl3/kernels/b.h', 'exl3/kernels'), None),
            (row('engine/exl3/kernels/b.h', 'exl3/kernels'),
             row('engine/layers/kernels/a.h', 'layers/kernels'), 'upward'),
            (row('engine/layers/kernels/a.h', 'layers/kernels'),
             row('engine/layers/ops.h', 'layers'), 'kernel'),
            (row('engine/layers/ops.h', 'layers'),
             row('engine/runtime/x.h', 'runtime'), 'upward'),
            (row('engine/serve/s.h', 'serve'), row('engine/runtime/x.h', 'runtime'), None),
            (row('engine/gfx1151/common.h', 'gfx1151'),
             row('engine/model/flashnext/geometry.h', 'model/flashnext'), None),
            (row('engine/layers/kernels/a.h', 'layers/kernels'),
             row('engine/switches.h', 'engine'), None),
            (row('engine/switches.h', 'engine'), row('engine/gfx1151/common.h', 'gfx1151'), 'leaf'),
        ]
        for src, dst, want in cases:
            with self.subTest(src=src.new, dst=dst.new):
                self.assertEqual(layout.include_rule(src, dst), want)

    def test_includes_resolve_relative_to_the_file(self):
        self.assertEqual(layout.resolve('kernels/a.h', '../engine/ops.h'), 'engine/ops.h')
        self.assertEqual(layout.quoted_includes('#include <x>\n  # include "a/b.h"\n'), ['a/b.h'])


if __name__ == '__main__':
    unittest.main()
