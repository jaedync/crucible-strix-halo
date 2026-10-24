"""Synthetic shard sets and quantized tables for the tests (no real data)."""
from __future__ import annotations

import os

import numpy as np

import codec
from common import encode_header

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def write_safetensors(path: str, tensors: dict, metadata: dict | None = None) -> None:
    """tensors: name -> (dtype string, numpy array with the right bytes)."""
    header, blobs, off = {}, [], 0
    if metadata:
        header["__metadata__"] = metadata
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    with open(path, "wb") as f:
        f.write(encode_header(header))
        for b in blobs:
            f.write(b)


def write_shard_set(src: str, shards: np.ndarray, layout: list[list[int]]) -> None:
    """shards (N, R, 160) uint16 bf16 bits; layout: shard indices per file, in file order."""
    os.makedirs(src, exist_ok=True)
    rng = np.random.default_rng(7)
    for f, idxs in enumerate(layout):
        t = {f"model.layers.{f}.other.weight": ("BF16", rng.integers(0, 65535, (3, 5), dtype=np.uint16))}
        for i in idxs:
            t[f"{PREFIX}.shard_{i}.weight"] = ("BF16", shards[i])
        write_safetensors(os.path.join(src, f"model-{f + 1:05d}-of-{len(layout):05d}.safetensors"), t)


def heads(rows: int, H: int = 16) -> tuple[np.ndarray, np.ndarray]:
    size = rows // H
    return np.arange(H, dtype=np.int64) * size, np.full(H, size, dtype=np.int64)


def make_table(path: str, K: int, rows: int, shard_rows: int | None, seed: int):
    """Random packed rows plus head bias in an exl3_ngram_trellis file; returns decoded rows."""
    rng = np.random.default_rng(seed)
    W = codec.words_per_row(K)
    packed = rng.integers(0, 65536, size=(rows, W), dtype=np.uint16)
    packed[:, 0] = (rng.random(rows) * 0.02 + 0.005).astype(np.float16).view(np.uint16)
    offs, sizes = heads(rows)
    bias = (rng.standard_normal((16, 160)) * 0.003).astype(np.float16)
    t = {f"{PREFIX}.head_bias": ("F16", bias), f"{PREFIX}.head_offsets": ("I64", offs),
         f"{PREFIX}.head_vocab_sizes": ("I64", sizes),
         f"{PREFIX}.layer_multipliers": ("I64", np.array([3, 5, 7], np.int64))}
    if shard_rows:
        for s in range(rows // shard_rows):
            t[f"{PREFIX}.shard_{s}.trellis"] = ("I16", packed[s * shard_rows:(s + 1) * shard_rows])
    else:
        t[f"{PREFIX}.trellis"] = ("I16", packed)
    meta = {"format": "exl3_ngram_trellis", "version": "1", "K": str(K), "codebook": "mul1",
            "row_dim": "160", "rows": str(rows)}
    write_safetensors(path, t, meta)
    hd = codec.head_of_rows(np.arange(rows), offs)
    return codec.dequant_rows(packed, K, codec.mul1_codebook(), bias[hd])
