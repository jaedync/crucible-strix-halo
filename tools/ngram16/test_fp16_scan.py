"""fp16_scan.py on small synthetic tables (no real data)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import fp16_scan
from codec import ROW_DIM, bf16_to_f32
from synth import f32_to_bf16, write_safetensors

NAME = "ngram.weight"


def table(tmp_path, bits: np.ndarray, extra: dict | None = None) -> str:
    path = str(tmp_path / "ngram_embedding_bf16.safetensors")
    write_safetensors(path, {NAME: ("BF16", bits.reshape(-1, ROW_DIM)), **(extra or {})}, {"format": "pt"})
    return path


def rows(n: int, seed: int = 3) -> np.ndarray:
    # Real-like values (|v| up to a few hundred) and both signs.
    rng = np.random.default_rng(seed)
    return f32_to_bf16(rng.standard_normal(n * ROW_DIM).astype(np.float32) * 40).reshape(-1)


def scan(path, tmp_path, capsys, *extra):
    out = tmp_path / "scan.jsonl"
    code = fp16_scan.main([path, "--json", str(out), *extra])
    printed = capsys.readouterr().out.strip().splitlines()
    record = json.loads(out.read_text().splitlines()[-1])
    # Guard log lines (MemAvailable) come first; the JSON line is the one before the RESULT line.
    assert json.loads(printed[-2]) == record
    return code, record, printed[-1]


def test_clean_table_passes_with_the_exact_max(tmp_path, capsys):
    # 2.5 MiB of data over 1 MiB chunks and 2 threads: a full wave, then a short last chunk.
    bits = rows(8192)
    bits[-1] = f32_to_bf16(np.array([-65280.0], np.float32))[0]  # the largest bf16 that fp16 holds
    code, rec, result = scan(table(tmp_path, bits), tmp_path, capsys, "--chunk-mib", "1", "--threads", "2")
    want = float(np.abs(bf16_to_f32(bits)).max())
    assert code == 0 and rec["status"] == "PASS"
    assert rec["max_abs"] == want == 65280.0
    assert (rec["nonfinite"], rec["above_fp16_max"], rec["values"]) == (0, 0, bits.size)
    assert rec["file"] == "ngram_embedding_bf16.safetensors" and str(tmp_path) not in json.dumps(rec)
    assert result == "RESULT ng16-fp16-scan PASS max_abs=65280 nonfinite=0"


@pytest.mark.parametrize("value, nonfinite, over", [
    (np.nan, 1, 0), (-np.inf, 1, 0), (np.inf, 1, 0), (65536.0, 0, 1), (-1e30, 0, 1)])
def test_bad_value_in_any_chunk_fails(tmp_path, capsys, value, nonfinite, over):
    for where in (0, 524288, 8192 * ROW_DIM - 1):  # first chunk, a chunk edge (1 MiB), the last value
        bits = rows(8192)
        bits[where] = f32_to_bf16(np.array([value], np.float32))[0]
        code, rec, result = scan(table(tmp_path, bits), tmp_path, capsys, "--chunk-mib", "1", "--threads", "3")
        assert code == 1 and rec["status"] == "FAIL", where
        assert (rec["nonfinite"], rec["above_fp16_max"]) == (nonfinite, over)
        assert result.startswith("RESULT ng16-fp16-scan FAIL ")


def test_a_nan_payload_with_the_sign_bit_counts(tmp_path, capsys):
    bits = rows(64)
    bits[5] = 0xFFC1
    code, rec, _ = scan(table(tmp_path, bits), tmp_path, capsys)
    assert code == 1 and rec["nonfinite"] == 1 and rec["max_abs"] == float(np.abs(bf16_to_f32(np.delete(bits, 5))).max())


def test_other_dtypes_are_not_scanned(tmp_path, capsys):
    other = {"meta.multipliers": ("F32", np.full((4,), np.nan, np.float32))}
    code, rec, _ = scan(table(tmp_path, rows(64), other), tmp_path, capsys)
    assert code == 0 and rec["tensors"] == [NAME]


def test_offsets_outside_the_file_are_an_error(tmp_path, capsys):
    path = table(tmp_path, rows(4))
    with open(path, "r+b") as f:
        f.truncate(f.seek(0, 2) - 2)
    assert fp16_scan.main([path]) == 2
    assert capsys.readouterr().out.strip() == "RESULT ng16-fp16-scan FAIL error"


def test_chunk_stats_matches_numpy():
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 1 << 16, 100_000, dtype=np.uint16)
    top, nonfinite, over = fp16_scan.chunk_stats(bits, np.empty_like(bits))
    values = bf16_to_f32(bits)
    finite = np.abs(values[np.isfinite(values)])
    assert top == int((bits[np.isfinite(values)] & 0x7FFF).max())
    assert nonfinite == int((~np.isfinite(values)).sum())
    assert over == int((finite > fp16_scan.FP16_MAX).sum())
