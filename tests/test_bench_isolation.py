"""Ratchet R3: benches and probes use public entries, not engine internals.

Counts per tracked source file outside engine/, that may only go down:
- engine: a program (`.hip`, `.cpp`, `.cu`, `.cc`) that compiles all of engine/engine.hip, directly or through
  included files (a bench helper header, another program); `"..."` and `<...>` includes and `#define X "path"`
  are resolved against the file's directory and the repository root;
- main: `#define main ...` in any file (it renames the main of an included program; S2 puts the engine's
  main under a build guard);
- internal: reads of engine internals (plans, layer groups, `.L[`, `->L[`, friend structs) in a file that
  reaches engine.hip;
- schedule: layer schedule literals in bench/ (`l % 4 == 3`, `(l & 3) == 3`, `(l + 1) % 4`, a loop bound of
  48 layers), with layer index names only.
A new site fails test_no_new_site; a count below ALLOWED fails the stale test.
Run `python3 tests/test_bench_isolation.py` to print the current counts.

Limits (written, not tested): an engine path from the build (`-D CRUCIBLE_ENGINE_HEADER=...`), a path built by
macro concatenation, an alias of an internal name, a reference that keeps `m_.L` under another name, and the
bench tier comment are not checked. `#if 0` blocks count.
"""
import posixpath
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model_geometry import SUFFIXES, blank_lexemes, source_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SKIPPED_TOPS = {"engine", "build", "node_modules"}
PROGRAMS = {".hip", ".cpp", ".cu", ".cc"}
ENGINE_HIP = "engine/engine.hip"
INCLUDE = re.compile(r'^[ \t]*#[ \t]*(?:include[ \t]*[<"]([^>"\n]+)[>"]|define[ \t]+\w+[ \t]+"([^"\n]+)")', re.M)
MAIN = re.compile(r"^[ \t]*#[ \t]*define[ \t]+main\b", re.M)
INTERNAL = re.compile(r"\b(?:PlansFor|PrefillPlansFor|LayerGroups|RunGroup|BuildLayer|EngineStateIO|"
                      r"ConversationStateIO)\b|(?:\.|->)L\[")
# Layer index names only, so `i % 4 == 0` (span sizes) and `h < 48` (heads) do not count.
LAYER = r"\b(?:l|li|layer)\b"
SCHEDULE = re.compile(LAYER + r"\s*\)?\s*(?:%\s*4|&\s*3)\s*\)?\s*[!=]=\s*[03]\b"
                      r"|\(\s*" + LAYER + r"\s*\+\s*1\s*\)\s*%\s*4\b"
                      r"|;\s*" + LAYER + r"\s*<\s*48\s*;|;\s*48\s*>\s*" + LAYER)


def targets(rel, code):
    """Repository paths that a file includes (both resolutions of each include)."""
    out = set()
    for m in INCLUDE.finditer(code):
        name = m.group(1) or m.group(2)
        out.add(posixpath.normpath(posixpath.join(posixpath.dirname(rel), name)))
        out.add(posixpath.normpath(name))
    return out


def scan(root=ROOT):
    texts = {}
    for top in sorted(p.name for p in root.iterdir() if p.is_dir()):
        if top not in SKIPPED_TOPS and not top.startswith("."):
            for rel, path in source_files(root, (top,), SUFFIXES):
                texts[rel] = blank_lexemes(path.read_text(errors="replace"))
    includes = {rel: targets(rel, code) for rel, (code, _) in texts.items()}
    # A copy kept under results/ may name the engine by a path that does not resolve from its directory.
    reaches = {rel for rel, t in includes.items() if any(x == ENGINE_HIP or x.endswith("/" + ENGINE_HIP) for x in t)}
    while True:  # the files that include a file that reaches the engine
        more = {rel for rel, t in includes.items() if rel not in reaches and t & reaches}
        if not more:
            break
        reaches |= more
    found = {}
    for rel, (_, bare) in texts.items():
        counts = {"main": len(MAIN.findall(bare))}
        if rel in reaches:
            counts.update(engine=int(Path(rel).suffix in PROGRAMS), internal=len(INTERNAL.findall(bare)))
        if rel.startswith("bench/"):
            counts["schedule"] = len(SCHEDULE.findall(bare))
        counts = {k: v for k, v in counts.items() if v}
        if counts:
            found[rel] = counts
    return found


# Frozen on train 19: 37 programs compile engine.hip (34 directly, 3 through
# bench/qsa_engine_long.hip); 41 files had `#define main` (34 engine includers, 7 that include another
# bench program). After an earlier change (s2-guard 6f17774a, train 20): 13 files have `#define main` (the 6 cross-source
# engine harnesses and the 7 that include another bench program); 18 files read internals (122 reads); 13 schedule
# literals in 8 bench files.
# Targets: main 13 after S2, internal 0 and
# schedule 0; engine only down.
ALLOWED = {
    'bench/batch_gates.hip': {'engine': 1},
    'bench/batch_profile.hip': {'engine': 1},
    'bench/batch_routing.hip': {'engine': 1, 'internal': 5},
    'bench/batch_snapshot_gate.hip': {'engine': 1, 'internal': 11},
    'bench/batch_sparse_gate.hip': {'engine': 1},
    'bench/batch_spec_probe.hip': {'engine': 1},
    'bench/batch_switch_probe.hip': {'engine': 1},
    'bench/compact_transition_gate.hip': {'main': 1, 'engine': 1, 'internal': 3},
    'bench/exact_dense_rows.hip': {'schedule': 2},
    'bench/exact_projection_rows.hip': {'schedule': 2},
    'bench/exl3_gemm.hip': {'schedule': 2},
    'bench/exl3_gemm_cycles_widths.hip': {'main': 1},
    'bench/exl3_gemv.hip': {'schedule': 2},
    'bench/forward_rows.hip': {'engine': 1, 'internal': 15},
    'bench/hc_engine_gates.hip': {'engine': 1},
    'bench/long_probe.hip': {'main': 1, 'engine': 1},
    'bench/mega_ablate.hip': {'main': 1},
    'bench/mega_ablate2.hip': {'main': 1},
    'bench/mega_dense.hip': {'schedule': 1},
    'bench/moe_body_window.hip': {'main': 1},
    'bench/moe_shared_first.hip': {'engine': 1, 'internal': 16},
    'bench/moe_wmma_count.hip': {'main': 1},
    'bench/mtp_dump.hip': {'engine': 1},
    'bench/packed_rows.hip': {'engine': 1, 'internal': 6},
    'bench/prefill_moe_loader.hip': {'main': 1},
    'bench/prefill_state_gates.hip': {'engine': 1, 'internal': 1},
    'bench/qsa_engine_long.hip': {'main': 1, 'engine': 1, 'internal': 6, 'schedule': 1},
    'bench/qsa_spec_long.hip': {'engine': 1, 'internal': 6},
    'bench/qsa_trace.hip': {'engine': 1, 'internal': 2},
    'bench/quality_decode.hip': {'engine': 1},
    'bench/quality_dispatch.hip': {'engine': 1, 'internal': 2},
    'bench/quality_startup.hip': {'engine': 1, 'internal': 3},
    'bench/resident_probe.hip': {'main': 1, 'engine': 1},
    'bench/sampling_logits.hip': {'engine': 1},
    'bench/session_replay_probe.hip': {'engine': 1, 'internal': 2},
    'bench/slot_checkpoint_gate.hip': {'engine': 1, 'internal': 6},
    'bench/spec_sweep.hip': {'main': 1, 'engine': 1},
    'bench/split2048.hip': {'engine': 1, 'internal': 2},
    'bench/statecache_gate.hip': {'engine': 1, 'internal': 8},
    'bench/suite/resident.hip': {'main': 1, 'engine': 1},
    'bench/vision_bench.hip': {'engine': 1},
    'tests/draft_defaults_test.hip': {'engine': 1},
    'tests/draft_head_api_test.hip': {'engine': 1},
    'tests/prefill_stride_test.hip': {'main': 1},
}


def totals(found):
    out = {}
    for counts in found.values():
        for kind, n in counts.items():
            out[kind] = out.get(kind, 0) + n
    files = {kind: sum(1 for c in found.values() if c.get(kind)) for kind in out}
    return out, files


class BenchIsolationRatchet(unittest.TestCase):
    def test_no_new_site(self):
        new = [f"{path} {kind}: {n} > {ALLOWED.get(path, {}).get(kind, 0)}"
               for path, counts in scan().items() for kind, n in counts.items()
               if n > ALLOWED.get(path, {}).get(kind, 0)]
        self.assertEqual(new, [], "time a kernel or layer through its public entry, not engine.hip")

    def test_allow_list_has_no_stale_entry(self):
        found = scan()
        stale = [f"{path} {kind}: allowed {n}, found {found.get(path, {}).get(kind, 0)}"
                 for path, counts in ALLOWED.items() for kind, n in counts.items()
                 if found.get(path, {}).get(kind, 0) < n]
        self.assertEqual(stale, [], "lower or remove these ALLOWED entries")

    def test_scan_kinds(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            files = {
                "bench/a.hip": ('#define main CrucibleMain\n#include "../engine/engine.hip"\n#undef main\n'
                                "auto& P = e.PlansFor(R); g(P.L[l].a_in); // RunGroup\n"
                                "for (int l = 0; l < 48; ++l) if (l % 4 == 3) f((l + 1) % 4);\n"),
                "bench/b.hip": ('#ifndef CRUCIBLE_ENGINE_HEADER\n#define CRUCIBLE_ENGINE_HEADER "../engine/engine.hip"\n'
                                '#endif\n#define main X\n#include CRUCIBLE_ENGINE_HEADER\n'),
                "bench/c.hip": '#include "../engine/layers/kernels/prefill_gdn.h"\n// engine/engine.hip\n',
                "bench/f.hip": '#define main WmmaRateMain\n#include "wmma_rate.hip"\n',
                "tests/d.hip": 'f("not an include: engine/engine.hip.bak");\nfor (int i = 0; i < 48; ++i) {}\n',
                "bench/e.hip": "for (int h = 0; h < 48; ++h) if (i % 4 == 0 || c % 4 == 3) {}\n",
                "engine/e.hip": '#include "../engine/engine.hip"\n',
            }
            for rel, text in files.items():
                Path(tmp, rel).parent.mkdir(parents=True, exist_ok=True)
                Path(tmp, rel).write_text(text)
            self.assertEqual(scan(Path(tmp)), {
                "bench/a.hip": {"engine": 1, "main": 1, "internal": 2, "schedule": 3},
                "bench/b.hip": {"engine": 1, "main": 1},
                "bench/f.hip": {"main": 1},
            })


    def test_scan_include_forms(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            files = {
                "bench/a.cpp": "#include <engine/engine.hip>\nauto* p = e->L[0];\n",
                "bench/b.hip": '#include "../engine/./engine.hip"\nif ((l & 3) == 3) {}\n',
                "bench/support/engine_lib.h": '#define main LibMain\n#include "../../engine/engine.hip"\n',
                "bench/c.hip": '#include "support/engine_lib.h"\nauto& P = e.PlansFor(1);\n',
                "bench/d.hip": '#include "c.hip"\n',
                "bench/e.hip": '#include "support/other.h"\n',
                "bench/support/other.h": "int x;\n",
            }
            for rel, text in files.items():
                Path(tmp, rel).parent.mkdir(parents=True, exist_ok=True)
                Path(tmp, rel).write_text(text)
            self.assertEqual(scan(Path(tmp)), {
                "bench/a.cpp": {"engine": 1, "internal": 1},
                "bench/b.hip": {"engine": 1, "schedule": 1},
                "bench/c.hip": {"engine": 1, "internal": 1},
                "bench/d.hip": {"engine": 1},
                "bench/support/engine_lib.h": {"main": 1},
            })

    def test_scan_reads_tracked_files_only(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bench").mkdir()
            for name in ("a.hip", "scratch.hip"):
                Path(tmp, "bench", name).write_text('#define main M\n#include "../engine/engine.hip"\n')
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(["git", "add", "bench/a.hip"], cwd=tmp, check=True)
            self.assertEqual(sorted(scan(Path(tmp))), ["bench/a.hip"])

if __name__ == "__main__":
    if sys.argv[1:] == []:
        found = scan()
        for path, counts in found.items():
            print(f"    {path!r}: {counts},")
        counts, files = totals(found)
        print(f"# sites {counts}, files {files}")
    else:
        unittest.main()
