"""NumPy port of the exl3_ngram_trellis row codec (exllamav3 ngram_codec.py, torch).

A packed row is (1 + 10*K) little-endian 16-bit words. Word 0 is the fp16 row
scale. The other words hold a 160*K-bit ring: stream bits [i*K, (i+1)*K) are
the low K bits of the 16-bit trellis state of position i. State i is the 16
ring bits that end at stream bit (i+1)*K - 1 (mod 160*K). Decode:

    row[i] = mul1(state_i) * scale + head_bias[head(row)]

in float32 from fp16 operands, like dequant_rows() in the torch reference.
There is no rotation and no global codebook scale: the quantizer folds its
per-row codebook scale into the stored fp16 row scale.
"""
from __future__ import annotations

import numpy as np

ROW_DIM = 160
MUL1 = 0x83DCD12D


def words_per_row(K: int) -> int:
    if not 1 <= K <= 8:
        raise ValueError("K must be 1..8")
    return 1 + ROW_DIM * K // 16


def mul1_codebook() -> np.ndarray:
    """All 65536 mul1 codebook values as fp16, the same as the torch reference."""
    s = np.arange(65536, dtype=np.uint64)
    prod = (s * np.uint64(MUL1)) & np.uint64(0xFFFFFFFF)
    bsum = sum((prod >> np.uint64(sh)) & np.uint64(255) for sh in (0, 8, 16, 24))
    # The torch reference computes in fp32 then rounds to fp16; fp32 holds
    # h * inv exactly (11 x 11 significant bits), so only the add rounds.
    h = (1024 + bsum).astype(np.float32)
    inv = np.array([0x1EEE], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    bias = np.array([0xC931], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    return (h * inv + bias).astype(np.float16)


def _state_sources(K: int) -> np.ndarray:
    """(160, 16) stream bit index of state bit m at position i."""
    i = np.arange(ROW_DIM)[:, None]
    m = np.arange(16)[None, :]
    return ((i - m // K) % ROW_DIM) * K + m % K


def unpack_rows(packed: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """packed (N, 1 + 10*K) int16/uint16 -> (states (N, 160) uint16, scales (N,) fp16)."""
    packed = np.ascontiguousarray(packed)
    if packed.ndim != 2 or packed.shape[1] != words_per_row(K) or packed.dtype.itemsize != 2:
        raise ValueError(f"packed must be (N, {words_per_row(K)}) 16-bit words")
    words = packed.view("<u2")
    scales = words[:, 0].copy().view(np.float16)
    stream_bytes = np.ascontiguousarray(words[:, 1:]).view(np.uint8)
    # Little-endian words and little bit order give stream bit b at index b.
    bits = np.unpackbits(stream_bytes, axis=1, bitorder="little")
    gathered = bits[:, _state_sources(K)]                       # (N, 160, 16), bit m last
    states = np.packbits(gathered, axis=2, bitorder="little")   # (N, 160, 2) bytes
    return np.ascontiguousarray(states).view("<u2")[..., 0], scales


def head_of_rows(rows: np.ndarray, head_offsets: np.ndarray) -> np.ndarray:
    """Head index of each global row, like torch.searchsorted(right=True) - 1."""
    h = np.searchsorted(np.asarray(head_offsets), np.asarray(rows), side="right") - 1
    return np.clip(h, 0, len(head_offsets) - 1)


def dequant_rows(packed: np.ndarray, K: int, codebook: np.ndarray,
                 bias: np.ndarray | None = None) -> np.ndarray:
    """packed (N, 1 + 10*K); bias (N, 160) fp16 per row or None -> (N, 160) float32."""
    states, scales = unpack_rows(packed, K)
    out = codebook[states].astype(np.float32) * scales.astype(np.float32)[:, None]
    if bias is not None:
        out = out + np.asarray(bias).astype(np.float32)
    return out


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit patterns -> float32 (exact)."""
    return (np.asarray(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def engine_value_bf16(raw: np.ndarray) -> np.ndarray:
    """The value of the engine's BF16 row path (CRUCIBLE_PLE_BF16) for uint16 bf16 bit patterns.

    The original model has no head bias (the HF checkpoint holds only the shard rows, the
    multipliers, the head offsets and the vocab sizes), so the value is the bf16 number. The
    engine rounds it to fp16 like every PLE value; that is exact in the fp16 normal range.
    """
    return bf16_to_f32(raw).astype(np.float16).astype(np.float32)
