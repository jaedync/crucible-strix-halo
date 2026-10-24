"""The reference of the engine's BF16 row path (CRUCIBLE_PLE_BF16) on a converted table."""
import numpy as np

import convert
from codec import bf16_to_f32, engine_value_bf16
from common import read_header
from synth import PREFIX, f32_to_bf16, write_shard_set

N, R = 3, 1000  # rows of 320 B, so shard edges and page edges fall inside rows


def test_engine_row_math_reads_global_rows_across_shard_edges(tmp_path):
    rng = np.random.default_rng(11)
    shards = f32_to_bf16(rng.standard_normal((N, R, 160)).astype(np.float32) * 0.02)
    write_shard_set(str(tmp_path / "src"), shards, [[2, 0], [1]])
    out = tmp_path / "t.safetensors"
    assert convert.main([str(tmp_path / "src"), str(out), "--revision", "r", "--chunk-mb", "0.01",
                         "--rate-mb", "1e6"]) == 0
    header, base = read_header(str(out))
    lo, hi = header[f"{PREFIX}.weight"]["data_offsets"]
    assert (lo, hi) == (0, N * R * 320) and out.stat().st_size == base + hi
    raw = out.read_bytes()
    for shard, row in [(0, 0), (0, R - 1), (1, 0), (1, R - 1), (2, 0), (N - 1, R - 1)]:
        g = shard * R + row  # global row of the 6-bit table
        offset = base + g * 320  # engine: CheckPleBf16File offset + row * kPleBf16RowBytes
        got = np.frombuffer(raw[offset:offset + 320], np.uint16)
        assert np.array_equal(got, shards[shard, row])


def test_engine_value_is_the_bf16_number_in_the_fp16_normal_range():
    raw = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    v = bf16_to_f32(raw)
    e = engine_value_bf16(raw)
    finite = np.isfinite(v)
    a = np.abs(v)
    normal = finite & ((a == 0) | ((a >= 2.0 ** -14) & (a <= 65504)))
    assert np.array_equal(e[normal].view(np.uint32), v[normal].view(np.uint32))
    # Outside that range the fp16 rounding changes the value: overflow to inf, and fp16 subnormals.
    assert np.all(np.isinf(e[finite & (a > 65504)]))
    small = finite & (a > 0) & (a < 2.0 ** -14)
    assert np.all(np.abs(e[small] - v[small]) <= 2.0 ** -25)

