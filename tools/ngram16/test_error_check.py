import json

import numpy as np
import pytest

import convert
import error_check
from synth import f32_to_bf16, make_table, write_shard_set

SHARDS, SHARD_ROWS = 8, 130
ROWS = SHARDS * SHARD_ROWS


@pytest.fixture
def tables(tmp_path):
    q6 = make_table(str(tmp_path / "q6.safetensors"), 6, ROWS, None, seed=1)
    q5 = make_table(str(tmp_path / "q5.safetensors"), 5, ROWS, SHARD_ROWS, seed=2)
    rng = np.random.default_rng(5)
    # The "16-bit" rows: the K=6 rows plus 1% noise, so K=6 is close and K=5 is unrelated.
    x = q6 + rng.standard_normal(q6.shape).astype(np.float32) * 0.01 * np.sqrt((q6 ** 2).mean())
    write_shard_set(str(tmp_path / "src"), f32_to_bf16(x).reshape(SHARDS, SHARD_ROWS, 160),
                    [[0, 1, 2], [5, 3], [4, 6, 7]])
    out = tmp_path / "bf16.safetensors"
    assert convert.main([str(tmp_path / "src"), str(out), "--chunk-mb", "0.01", "--rate-mb", "1e6"]) == 0
    return tmp_path, out


def check(tmp_path, out, *extra):
    js = tmp_path / "res.json"
    rc = error_check.main([str(out), str(tmp_path / "src"), "--table", f"k6={tmp_path / 'q6.safetensors'}",
                           "--table", f"k5={tmp_path / 'q5.safetensors'}", "--blocks", "32",
                           "--block-rows", "16", "--json", str(js), *extra])
    assert rc == 0
    return json.loads(js.read_text())


def test_matching_table_has_small_error_and_shift_control_is_large(tables):
    tmp_path, out = tables
    res = check(tmp_path, out)
    k6 = res["tables"]["k6"]["stats"]
    assert res["rows"] == 32 * 16
    assert 0.005 < k6["rel_rms"] < 0.02
    assert k6["cos_p1"] > 0.999
    assert k6["shift_control"]["rel_rms"] > 1.0
    assert res["tables"]["k5"]["stats"]["rel_rms"] > 1.0
    assert res["bitwise"]["mismatch_rows"] == 0 and res["bitwise"]["rows"] == 2 * 17
    assert res["bitwise"]["boundary_rows"] == 16 and res["bitwise"]["boundary_mismatch_rows"] == 0
    heads = [h["rows"] for h in k6["per_head"]]
    assert heads == [32] * 16


def test_sampling_is_fixed_and_inside_heads(tables):
    tmp_path, out = tables
    a = check(tmp_path, out)
    b = check(tmp_path, out)
    assert a["tables"]["k6"]["stats"] == b["tables"]["k6"]["stats"]
    offs, sizes = np.array(a["head_offsets"]), np.array(a["head_sizes"])
    starts = error_check.sample_blocks(offs, sizes, ROWS, 32, 16, 20261018)
    h0 = np.searchsorted(offs, starts, side="right") - 1
    h1 = np.searchsorted(offs, starts + 16, side="right") - 1
    assert np.array_equal(h0, h1)
    assert np.bincount(h0, minlength=16).tolist() == [2] * 16


def test_corrupt_converted_row_is_found(tables):
    tmp_path, out = tables
    _, base = error_check.read_header(str(out))
    with open(out, "r+b") as f:  # flip one bit in the first row of shard 1
        f.seek(base + SHARD_ROWS * 320)
        b = f.read(1)
        f.seek(base + SHARD_ROWS * 320)
        f.write(bytes([b[0] ^ 1]))
    res = check(tmp_path, out)
    assert res["bitwise"]["boundary_mismatch_rows"] == 1
