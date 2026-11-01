#!/usr/bin/env python3
"""Copy reference contractions with only logical indices and barrier substitutions."""
import posixpath
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = (ROOT / 'engine/layers/ops.h').read_text()
# Repo paths, so that the include lines follow a layout move.
OUT = 'engine/layers/persistent_ops.h'
INCLUDES = ('engine/layers/ops.h', 'engine/gfx1151/persistent_sync.h', 'engine/layers/kernels/persistent_gdn_load.h')
EXL = (ROOT / 'engine/exl3/exl3.h').read_text()


def body(text, name):
    start = text.index(name + '(')
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[brace + 1:end - 1]


def adapt(text):
    text = text.replace('threadIdx.x', 'ct.t').replace('blockIdx.x', 'task')
    text = text.replace('blockDim.x', '256')
    text = text.replace('__syncthreads()', 'ComputeBarrier(ct)')
    text = re.sub(r'BlockSum<256>\(([^;]+), red\)', r'PersistentBlockSum(\1, red, ct)', text)
    return text


FUNCTIONS = [
    (OPS, 'HcDotsKernel', 'template <int R>\n__device__ __forceinline__ void PersistentHcDots(ComputeContext& ct, int task, const __half* fn, int M, const float* H, float* dots, float* ss)'),
    (OPS, 'HcMixKernel', 'template <int R>\n__device__ __forceinline__ void PersistentHcMix(ComputeContext& ct, int task, const __half* up, const float* w1, const float* dots, int M, const float* ss, const float* H, float* mixed, float* post)'),
    (OPS, 'GdnKernel', '__device__ __forceinline__ void PersistentGdn(ComputeContext& ct, int task, GdnArgs a)'),
    (EXL, 'LinInKernel', '__device__ __forceinline__ void PersistentLinIn(ComputeContext& ct, int task, const LinJob* jobs, const XfItem* items, int n)'),
    (EXL, 'LinOutKernel', '__device__ __forceinline__ void PersistentLinOut(ComputeContext& ct, int task, const LinJob* jobs, const XfItem* items, int n)'),
]
HEADER = '''// Generated reference contractions. Regenerate with tools/generate_persistent_ops.py.
// Only logical indices and compute-only synchronization differ from the source.
#pragma once
''' + ''.join(f'#include "{posixpath.relpath(p, posixpath.dirname(OUT))}"\n' for p in INCLUDES) + '''namespace crucible {
'''
def scheduled_body(source, name):
    reference = body(source, name)
    if name == 'GdnKernel':
        # The persistent caller supplies one state; it has no batch row to select.
        binding = '\n  const GdnArgs a = GdnRowArgs(args, blockIdx.y);'
        if reference.count(binding) != 1:
            raise ValueError('the GDN state binding changed; review the persistent adapter')
        reference = reference.replace(binding, '', 1)
    result = adapt(reference)
    # Lookahead progress belongs to the per-op kernels; the persistent side
    # requires the lookahead off, so it publishes nothing.
    result = re.sub(r'\n[ \t]*(?:if constexpr \([^)]*\) )?PublishWeights\([^;]*\);', '', result)
    if name == 'GdnKernel':
        # The reference template's default cache policy: plain state loads and stores.
        result = '\n  constexpr int CACHE = 0;\n  constexpr bool STREAM_STORE = false;' + result
        result = '\n#if CRUCIBLE_PERSISTENT_GDN_SERIAL\n  a.R = 1; a.replay = 0; a.commit = 1;\n#endif\n' + result
        for term in ('ii', 'i'):
            result = result.replace(f'mem += S[{term}] * kq[{term}];',
                                    f'mem += S[{term}] * PersistentCoefficient(kq + {term}, mem);')
        result = result.replace('S[i] += kq[i] * delta;',
                                'S[i] += PersistentCoefficient(kq + i, y) * delta;')
        result = result.replace('y += S[i] * qq[i];',
                                'y += S[i] * PersistentCoefficient(qq + i, y);')
    return result


def render():
    return HEADER + '\n'.join(signature + ' {' + scheduled_body(source, name) + '}\n'
                              for source, name, signature in FUNCTIONS) + '}  // namespace crucible\n'


if __name__ == '__main__':
    path = ROOT / OUT
    path.write_text(render())
    print(path)
