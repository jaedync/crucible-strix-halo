"""Ratchet R2: the EXL3 format seam.

The EXL3 transforms (scale by suh, 128-point Hadamard, scale by svh) and weight types belong in engine/exl3/.
Counts per tracked file in engine/ and bench/, outside the owner directories, that may only go down:
- wht: calls of `Wht128` and its variants (`Wht128Dpp(`, `Wht128<N>(`); owners engine/exl3/, engine/gfx1151/;
- def: definitions of a `Wht128...` function (a new copy of the transform); same owners; baseline 0;
- scale: index reads of the `suh` and `svh` scale vectors (`suh[`, `svh_o[`, ...); owner engine/exl3/;
- type: definitions of `struct Lin` and `struct ExpertPtrs` (with `alignas`, attributes, `final` or a base);
  owner engine/exl3/.
A new site fails test_no_new_format_site; a count below ALLOWED fails the stale test.
Run `python3 tests/test_format_seam.py` to print the current counts.

Limits (written, not tested): deliberate bypasses are not seen, for example a pointer alias
(`const __half* s = j.suh; s[c]`), pointer arithmetic (`*(j.svh + c)`), a `#define` alias of `Wht128`, or the
same butterfly written under another name. tests/ is not scanned. `#if 0` blocks count.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model_geometry import SUFFIXES, blank_lexemes, source_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("engine", "bench")
TRANSFORM_OWNERS = ("engine/exl3/", "engine/gfx1151/")
OWNERS = {"wht": TRANSFORM_OWNERS, "def": TRANSFORM_OWNERS, "scale": ("engine/exl3/",), "type": ("engine/exl3/",)}
WHT = re.compile(r"\bWht128\w*\s*(?:<[^<>;(){}]*>)?\s*\(")
# A name after one of these words is a declaration or definition, not a call.
DECLARES = re.compile(r"(?:\b(?:void|auto|float|inline|__forceinline__|__device__|__host__|static|constexpr)"
                      r"|>)\s*$")
PATTERNS = {
    "scale": re.compile(r"\b(?:suh|svh)\w*\s*\["),
    "type": re.compile(r"\bstruct\s+(?:(?:alignas\s*\([^)]*\)|\[\[[^\]]*\]\]|__attribute__\s*\(\(.*?\)\))\s*)*"
                       r"(?:Lin|ExpertPtrs)\s*(?:final\s*)?(?::[^{;]*)?\{"),
}


def count_file(rel, text):
    _, bare = blank_lexemes(text)
    counts = {"wht": 0, "def": 0}
    for m in WHT.finditer(bare):
        counts["def" if DECLARES.search(bare[max(0, m.start() - 40):m.start()]) else "wht"] += 1
    counts.update({name: len(p.findall(bare)) for name, p in PATTERNS.items()})
    return {k: v for k, v in counts.items() if v and not rel.startswith(OWNERS[k])}


def scan(root=ROOT):
    found = {}
    for rel, path in source_files(root, SCANNED, SUFFIXES):
        counts = count_file(rel, path.read_text())
        if counts:
            found[rel] = counts
    return found


# Frozen on train 19: engine wht 22, scale 45, type 2; bench wht 9,
# scale 32. Train 20 (dead-kern104 a9040407 moves bench-only kernels to bench/kernels/): engine wht 18,
# scale 37, type 2; bench wht 13, scale 40. Target in engine: 0 (S5 adds engine/exl3/transform.h).
ALLOWED = {
    'engine/engine.hip': {'scale': 6},
    'engine/layers/kernels/prefill_gdn.h': {'wht': 3, 'scale': 3},
    'engine/layers/kernels/prefill_qsa.h': {'wht': 1, 'scale': 1},
    'engine/layers/kernels/prefill_qsa_union.h': {'wht': 1, 'scale': 1},
    # +1 wht, +1 scale (packs): MoeFusedKernelWide copies MoeFusedKernel's down prep, because a
    # shared function changed the train 23 kernels' ISA (5842e3bb).
    'engine/layers/moe_rows.h': {'wht': 4, 'scale': 8},
    # New (decode-fuse 0d12f8cb): the R=1 MoE stage does the input transform inside the expert launch, so the
    # launch count drops; moving its transform calls to engine/exl3/ is an earlier change and must keep the stage's ISA.
    'engine/layers/moe_stage.h': {'wht': 4, 'scale': 6},
    'engine/layers/ops.h': {'wht': 1, 'scale': 3, 'type': 1},
    'engine/layers/persistent_ops.h': {'wht': 2, 'scale': 2},
    'engine/layers/prefill_moe.h': {'wht': 5, 'scale': 10},
    'engine/layers/qsa_long.h': {'wht': 1, 'scale': 1},
    'engine/model/flashnext/model.h': {'scale': 2, 'type': 1},
    'engine/runtime/draft_dense.h': {'wht': 1, 'scale': 1},
    'bench/kernels/exl3_moe_mid_probe.h': {'wht': 3, 'scale': 3},
    'bench/kernels/moe_rows.h': {'wht': 3, 'scale': 7},
    'bench/kernels/qsa_decode.h': {'wht': 1, 'scale': 1},
    'bench/moe_rows.hip': {'scale': 12},
    'bench/prefill_dense_half.hip': {'wht': 3, 'scale': 12},
    'bench/prefill_moe_gemm.hip': {'wht': 2, 'scale': 2},
    'bench/prefill_moe_mid.hip': {'scale': 2},
    'bench/qsa_flash_pipe.h': {'wht': 1, 'scale': 1},
}


def totals(found, prefix=""):
    out = {}
    for path, counts in found.items():
        if path.startswith(prefix):
            for kind, n in counts.items():
                out[kind] = out.get(kind, 0) + n
    return out


class FormatSeamRatchet(unittest.TestCase):
    def test_no_new_format_site(self):
        new = [f"{path} {kind}: {n} > {ALLOWED.get(path, {}).get(kind, 0)}"
               for path, counts in scan().items() for kind, n in counts.items()
               if n > ALLOWED.get(path, {}).get(kind, 0)]
        self.assertEqual(new, [], "put EXL3 transforms and weight types in engine/exl3/")

    def test_allow_list_has_no_stale_entry(self):
        found = scan()
        stale = [f"{path} {kind}: allowed {n}, found {found.get(path, {}).get(kind, 0)}"
                 for path, counts in ALLOWED.items() for kind, n in counts.items()
                 if found.get(path, {}).get(kind, 0) < n]
        self.assertEqual(stale, [], "lower or remove these ALLOWED entries")

    def test_scan_kinds_and_owners(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            files = {
                "engine/layers/a.h": (
                    "__device__ void f(float v[4]) { Wht128(v); Wht128Dpp(v); }\n"
                    "// Wht128(v); suh[0]\n"
                    "x = RoundH(RoundH(s) * __half2float(j.suh[c0 + i])); y = d.svh [c]; z = a.suh_out[h];\n"
                    "struct Lin {\n  int I;\n};\nstruct ExpertPtrs { int K[3]; };\nstruct LinJob { int I; };\n"
                    'f("Wht128(v)"); int suh = 0;\n'),
                "engine/gfx1151/common.h": "__device__ __forceinline__ void Wht128(float v[4]) {}\n",
                "engine/exl3/exl3.h": "Wht128(v); j.suh[0];\nstruct Lin { int I; };\n",
                "bench/b.hip": "void Wht128(float v[4]) {}\nWht128(v);\n",
                "server/c.h": "Wht128(v);\n",
            }
            for rel, text in files.items():
                Path(tmp, rel).parent.mkdir(parents=True, exist_ok=True)
                Path(tmp, rel).write_text(text)
            self.assertEqual(scan(Path(tmp)), {
                "bench/b.hip": {"wht": 1, "def": 1},
                "engine/layers/a.h": {"wht": 2, "scale": 3, "type": 2},
            })


    def test_scan_copies_and_forms(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            files = {
                "engine/layers/a.cu": ("__device__ __forceinline__ void Wht128Local(float v[4]) { Wht128<4>(v); }\n"
                                       "static inline auto Wht128B(float* v) -> void {}\n"
                                       "struct alignas(16) Lin { int I; };\n"
                                       "struct ExpertPtrs final : Base { int K; };\n"
                                       "struct Lin;\n"),
                "engine/gfx1151/b.hpp": "void Wht128(float v[4]) {}\nx = j.suh[0];\nstruct ExpertPtrs { int K; };\n",
                "engine/exl3/c.h": "void Wht128Copy(float v[4]) {}\n",
            }
            for rel, text in files.items():
                Path(tmp, rel).parent.mkdir(parents=True, exist_ok=True)
                Path(tmp, rel).write_text(text)
            self.assertEqual(scan(Path(tmp)), {
                "engine/gfx1151/b.hpp": {"scale": 1, "type": 1},
                "engine/layers/a.cu": {"wht": 1, "def": 2, "type": 2},
            })

if __name__ == "__main__":
    if sys.argv[1:] == []:
        found = scan()
        for path, counts in found.items():
            print(f"    {path!r}: {counts},")
        print(f"# engine {totals(found, 'engine/')}, bench {totals(found, 'bench/')}")
    else:
        unittest.main()
