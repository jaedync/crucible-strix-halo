"""Ratchet: model geometry literals outside the model config.

Model geometry (sizes, the layer schedule, the PLE layer) is a named constant in the model config
engine/model/flashnext/geometry.h, not a new literal. ALLOWED lists the sites
that existed when the ratchet landed, per file and token. The list may only get shorter: the test
fails on a new site and on a stale entry (a count above today's), so each cleanup removes entries.
Run `python3 tests/test_model_geometry.py` to print the current counts (`--bench` for bench/).

Counted tokens: the unambiguous trunk literals (hidden 2560, wide 10240, QSA q 12288, GDN value and
QSA output 6144, QSA index 640, vocabulary 248320, EOS 248044) and three schedule patterns. Values
that also have other meanings (2048, 512, 320, 256, 128, 48, 24) are not counted.
"""
import collections
import re
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import layout  # noqa: E402
SCANNED = ("engine", "kernels")
SUFFIXES = {".h", ".hpp", ".hip", ".cu", ".inc", ".cc", ".cpp"}
# The model config (tools/layout_map.tsv). Only this file may hold the values.
CONFIG = "engine/model/flashnext/geometry.h"
# One pass over string literals, character literals and comments, so "//" inside a string is not a comment.
LEXEMES = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.S)
# An integer suffix (2560u, 2560ull) still counts.
LITERAL = re.compile(r"(?<![\w.])(2560|10240|12288|6144|640|248320|248044)[uUlL]*(?!\w)")
# Patterns read the code with its strings, because tensor names hold the PLE layer.
PATTERNS = {
    "l%4==3": re.compile(r"\b(?:l|li|layer)\s*%\s*4\s*\)?\s*==\s*3\b"),
    "l==1": re.compile(r"\b(?:l|li|layer)\s*==\s*1\b"),
    "layers.1.": re.compile(r"layers\.1\."),
}


def blank_lexemes(text):
    """Return (code with its strings, code without strings or char literals), both without comments.

    Comments keep their newlines, so a multi-line comment cannot join two code lines."""
    code, bare, end = [], [], 0
    for m in LEXEMES.finditer(text):
        lexeme = m.group()
        between = text[end:m.start()]
        end = m.end()
        if lexeme[0] in "\"'":
            code.append(between + lexeme)
            bare.append(between + lexeme[0] * 2)
        else:
            blank = re.sub(r"[^\n]", " ", lexeme)
            code.append(between + blank)
            bare.append(between + blank)
    return "".join(code) + text[end:], "".join(bare) + text[end:]


def source_files(root, tops, suffixes=SUFFIXES):
    """Yield (rel, path) of the tracked source files under tops, sorted.

    Tracked files only (git ls-files; a tree without git skips .gitignore entries), so scratch files and
    ignored fixtures do not count."""
    for rel in sorted(layout.tracked_files(root)):
        path = root / rel
        if rel.split("/", 1)[0] in tops and Path(rel).suffix in suffixes and path.is_file():
            yield rel, path


def scan(root=ROOT, tops=SCANNED):
    """Return {path: {token: count}} for every scanned file with at least one site."""
    found = {}
    for rel, path in source_files(root, tops):
        if rel == CONFIG:
            continue
        code, bare = blank_lexemes(path.read_text())
        counts = collections.Counter(m.group(1) for m in LITERAL.finditer(bare))
        for name, pattern in PATTERNS.items():
            counts[name] += len(pattern.findall(code))
        counts = {k: v for k, v in sorted(counts.items()) if v}
        if counts:
            found[rel] = counts
    return found


# 117 sites in 20 files when the ratchet landed (b8c28f64), 1 after the model config.
ALLOWED = {
    # The prefill GEMM input staging: the widest input row (10240) without a model name yet.
    'engine/engine.hip': {'10240': 1},
}


# Ratchet R4: the same literals in bench/, frozen on train 19.
# Benches name the geometry from the model config; this list may only get shorter.
BENCH = ("bench",)
BENCH_ALLOWED = {
    'bench/batch_routing.hip': {'l==1': 1},
    'bench/d2h_logits.hip': {'248320': 1},
    'bench/exact_dense_rows.hip': {'layers.1.': 1},
    'bench/exact_projection_rows.hip': {'l%4==3': 1},
    'bench/exl3_dense_stage.hip': {'2560': 2, '6144': 1},
    'bench/exl3_gemm.hip': {'2560': 1},
    'bench/exl3_gemv.hip': {'10240': 1},
    'bench/forward_rows.hip': {'10240': 2},
    'bench/gdn_vscan.hip': {'10240': 5, '6144': 5},
    'bench/isa_decode_schedule.hip': {'10240': 1, '2560': 1, '640': 1},
    'bench/mega_ablate.hip': {'10240': 1, 'l==1': 2},
    'bench/mega_ablate2.hip': {'10240': 1},
    'bench/mega_dense.hip': {'10240': 1, 'l%4==3': 1},
    'bench/membw.hip': {'640': 1},
    'bench/moe_body_window.hip': {'2560': 1, '640': 1},
    'bench/ple_gather_bench.cpp': {'248044': 1},
    'bench/ple_host_cold.cpp': {'248044': 1},
    'bench/prefill_dense_epi.hip': {'6144': 1},
    'bench/prefill_dense_half.hip': {'2560': 1},
    'bench/prefill_gdn.hip': {'10240': 6, '6144': 13},
    'bench/prefill_moe_mid.hip': {'640': 3},
    'bench/prefill_qsa.hip': {'12288': 2, '6144': 5, '640': 1},
    'bench/qsa_capture_test.hip': {'12288': 1, '6144': 1, '640': 1},
    'bench/qsa_engine_long.hip': {'10240': 1, 'l%4==3': 1},
    'bench/qsa_flash_long.hip': {'6144': 5},
    'bench/qsa_flash_pipe.hip': {'6144': 4},
    'bench/qsa_long.hip': {'12288': 3, '6144': 6, '640': 3},
    'bench/qsa_prefill_score.hip': {'640': 3},
    'bench/qsa_prep_half.hip': {'6144': 1},
    'bench/qsa_top_threads.hip': {'6144': 1},
    'bench/quality_codec.hip': {'2560': 6, 'layers.1.': 1},
    'bench/static_layout.cc': {'2560': 1, 'layers.1.': 1},
    'bench/vision_bench.hip': {'10240': 1},
}


def _excess(found, allowed):
    return [f"{path} {token}: {count} > {allowed.get(path, {}).get(token, 0)}"
            for path, counts in found.items() for token, count in counts.items()
            if count > allowed.get(path, {}).get(token, 0)]


def _stale(found, allowed):
    return [f"{path} {token}: allowed {count}, found {found.get(path, {}).get(token, 0)}"
            for path, counts in allowed.items() for token, count in counts.items()
            if found.get(path, {}).get(token, 0) < count]


class ModelGeometryRatchet(unittest.TestCase):
    def test_no_new_geometry_literal_in_bench(self):
        self.assertEqual(_excess(scan(tops=BENCH), BENCH_ALLOWED), [], f"name the value from {CONFIG}")

    def test_bench_allow_list_has_no_stale_entry(self):
        self.assertEqual(_stale(scan(tops=BENCH), BENCH_ALLOWED), [], "lower or remove these BENCH_ALLOWED entries")

    def test_no_new_geometry_literal(self):
        new = [f"{path} {token}: {count} > {ALLOWED.get(path, {}).get(token, 0)}"
               for path, counts in scan().items() for token, count in counts.items()
               if count > ALLOWED.get(path, {}).get(token, 0)]
        self.assertEqual(new, [], f"name the value in {CONFIG} instead of a literal")

    def test_allow_list_has_no_stale_entry(self):
        found = scan()
        stale = [f"{path} {token}: allowed {count}, found {found.get(path, {}).get(token, 0)}"
                 for path, counts in ALLOWED.items() for token, count in counts.items()
                 if found.get(path, {}).get(token, 0) < count]
        self.assertEqual(stale, [], "lower or remove these ALLOWED entries")

    def test_scan_counts_code_not_comments_or_messages(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp, "engine")
            Path(tmp, CONFIG).parent.mkdir(parents=True)
            (engine / "a.h").write_text('// 2560\nint x = 2560; /* 6144\n */ f("[0, 248320)");\n'
                                        'g(p + "layers.1.ple.", (l % 4) == 3, x2560, 2560.f);\nif (l==1) {}\n')
            Path(tmp, CONFIG).write_text("constexpr int kHidden = 2560;\n")
            self.assertEqual(scan(Path(tmp)), {"engine/a.h": {"2560": 2, "l%4==3": 1, "l==1": 1, "layers.1.": 1}})

    def test_scan_counts_suffixes_and_code_after_a_string_with_slashes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp, "engine")).mkdir()
            Path(tmp, "engine", "a.h").write_text(
                'size_t a = 2560ull, b = 6144u, c = 640L;\n'
                'f("http://host", 10240); g("/* not a comment", 12288); // 248320\n'
                "char q = '\"'; int d = 248044;\n")
            self.assertEqual(scan(Path(tmp)), {"engine/a.h": {
                "10240": 1, "12288": 1, "248044": 1, "2560": 1, "6144": 1, "640": 1}})

    def test_scan_counts_code_between_a_quote_char_and_a_string(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp, "engine")).mkdir()
            Path(tmp, "engine", "a.h").write_text("char q = '\"'; int d = 248044; f(\"6144\");\n")
            self.assertEqual(scan(Path(tmp)), {"engine/a.h": {"248044": 1}})

    def test_scan_reads_tracked_files_only(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "engine").mkdir()
            for name in ("a.h", "b.hpp", "c.cu", "untracked.h"):
                Path(tmp, "engine", name).write_text("int x = 2560;\n")
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(["git", "add", "engine/a.h", "engine/b.hpp", "engine/c.cu"], cwd=tmp, check=True)
            self.assertEqual(sorted(scan(Path(tmp))), ["engine/a.h", "engine/b.hpp", "engine/c.cu"])

    def test_scan_skips_only_the_config_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, CONFIG).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp, CONFIG).write_text("constexpr int kHidden = 2560;\n")
            Path(tmp, "engine/model/flashnext/weight_validation.h").write_text("int x = 2560;\n")
            self.assertEqual(scan(Path(tmp)), {"engine/model/flashnext/weight_validation.h": {"2560": 1}})


if __name__ == "__main__":
    if sys.argv[1:] in ([], ["--bench"]):
        total = 0
        for path, counts in scan(tops=BENCH if sys.argv[1:] else SCANNED).items():
            print(f"    {path!r}: {counts},")
            total += sum(counts.values())
        print(f"# {total} sites")
    else:
        unittest.main()
