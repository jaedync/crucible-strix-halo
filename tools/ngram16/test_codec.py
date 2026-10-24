import os
import sys

import numpy as np
import pytest

import codec

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = np.load(os.path.join(HERE, "fixtures", "codec_ref.npz"))
REPO_REF = os.path.join(HERE, "..", "..", "ref")


def test_codebook_matches_torch_reference():
    assert np.array_equal(codec.mul1_codebook().view(np.uint16), FIX["codebook"])


def test_codebook_matches_exl3_linear_reference():
    sys.path.insert(0, REPO_REF)
    try:
        import exl3_ref
    finally:
        sys.path.remove(REPO_REF)
    states = np.arange(65536, dtype=np.uint16)
    want = exl3_ref.decode_codebook(states, "mul1")
    assert np.array_equal(codec.mul1_codebook().view(np.uint16), want.view(np.uint16))


@pytest.mark.parametrize("K", [3, 5, 6, 8])
def test_unpack_matches_torch_reference(K):
    states, scales = codec.unpack_rows(FIX[f"k{K}_packed"], K)
    assert states.dtype == np.uint16
    assert np.array_equal(states, FIX[f"k{K}_states"])
    assert np.array_equal(scales.view(np.uint16), FIX[f"k{K}_scales"])


@pytest.mark.parametrize("K", [3, 5, 6, 8])
def test_dequant_matches_torch_reference_bitwise(K):
    bias = FIX[f"k{K}_bias"].view(np.float16)
    out = codec.dequant_rows(FIX[f"k{K}_packed"], K, codec.mul1_codebook(), bias)
    assert out.dtype == np.float32
    assert np.array_equal(out.view(np.uint32), FIX[f"k{K}_deq"].view(np.uint32))


def test_state_window_is_the_ring_bits():
    # Low K bits of state i are symbol i; the next bits are symbols i-1, i-2, ...
    K = 5
    rng = np.random.default_rng(1)
    packed = rng.integers(0, 65536, size=(4, codec.words_per_row(K)), dtype=np.uint16)
    states, _ = codec.unpack_rows(packed, K)
    bits = np.unpackbits(packed[:, 1:].astype("<u2").view(np.uint8), axis=1, bitorder="little")
    sym = (bits.reshape(4, 160, K) << np.arange(K)).sum(-1)
    for i in (0, 1, 2, 3, 159):
        want = (sym[:, i] | sym[:, i - 1] << 5 | sym[:, i - 2] << 10 | (sym[:, i - 3] & 1) << 15)
        assert np.array_equal(states[:, i], want)


def test_unpack_rejects_wrong_width():
    with pytest.raises(ValueError):
        codec.unpack_rows(np.zeros((2, 50), np.int16), 5)


def test_head_of_rows_like_searchsorted_right():
    offs = np.array([0, 10, 25])
    rows = np.array([0, 9, 10, 24, 25, 99])
    assert codec.head_of_rows(rows, offs).tolist() == [0, 0, 1, 1, 2, 2]


def test_bf16_to_f32():
    x = np.array([1.0, -2.5, 3.140625, 0.0], np.float32)
    raw = (x.view(np.uint32) >> 16).astype(np.uint16)
    assert np.array_equal(codec.bf16_to_f32(raw), x)
