"""NumPy reference for EXL3 linear tensors, source commit d3739fd.

Codebook values and decoder states are bit-exact. reconstruct models the Python
get_weight_tensor path, not the different rounding of fused CUDA reconstruction.
matvec follows the default FP16 activation transform / decoded matmul / output
transform order. Its middle dot uses FP32 BLAS, not NVIDIA MMA accumulation.

The EXL3-INT8 mul24 codebook (github.com/jyohukuchan/rocm_exl3_forme doc/exl3_int8.md, "mul24 形式") is selected
by the .mul1 value 0x00DCD12D: x = state * 0xDCD12D mod 2^32, B = its byte sum, the WMMA B fragment tile order
(frag_perm). Two value forms of one state: 'mul24' is the fp16 mul1 formula on B (the quantizer's reconstruction,
the default), 'int8mul24' the INT8 grid q = (B - 509) >> 2 times S_FIX in float32.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np

HAD_SIZE = 128
HAD_SCALE = np.float32(0.088388347648)
MUL1 = 0x83DCD12D
MCG = 0xCBAC1FED
MUL24 = 0x00DCD12D
S_FIX = np.float32(4 * 887 / 131072)  # 0.027069091796875, the INT8 grid step of EXL3-INT8
MUL24_FORMS = ('mul24', 'int8mul24')


def _bitrate(bits: float) -> tuple[int, bool]:
    bits = float(bits)
    if not np.isfinite(bits) or bits * 2 != int(bits * 2):
        raise ValueError('EXL3 bitrate must be integer or half-integer')
    whole = int(bits)
    half = bits != whole
    if not (1 <= whole <= (3 if half else 8)):
        raise ValueError('Supported bitrates: 1..8 and 1.5, 2.5, 3.5')
    return whole, half


def tensor_core_perm() -> np.ndarray:
    """Position i in decoder order maps to row-major tile element perm[i]."""
    lane = np.arange(32)
    rows = 2 * (lane % 4)
    cols = lane // 4
    return np.stack([rows * 16 + cols, (rows + 1) * 16 + cols,
                     (rows + 8) * 16 + cols, (rows + 9) * 16 + cols,
                     rows * 16 + cols + 8, (rows + 1) * 16 + cols + 8,
                     (rows + 8) * 16 + cols + 8, (rows + 9) * 16 + cols + 8], axis=-1).reshape(256)


def frag_perm() -> np.ndarray:
    """mul24 tile order: position t = 8 L + j maps to row-major element (8 (L // 16) + j) * 16 + L % 16."""
    t = np.arange(256)
    return (8 * (t // 128) + t % 8) * 16 + (t // 8) % 16


def tile_perm(codebook: str) -> np.ndarray:
    return frag_perm() if codebook in MUL24_FORMS else tensor_core_perm()


def decode_tile_symbols(packed: np.ndarray, bits: float | None = None) -> np.ndarray:
    """Return 256 uint16 decoder states per tile, in tensor-core order.

    packed[..., 16*K] contains little-endian int16 storage words. Despite the
    name, the result is the full sliding 16-bit state, not just its K new bits.
    uint32 words are consumed MSB first; state i ends at the end of symbol i.
    The 16-bit window wraps around within the same tile.
    """
    packed = np.asarray(packed)
    if packed.ndim < 1 or packed.dtype.kind not in 'iu' or packed.dtype.itemsize != 2:
        raise ValueError('packed must contain 16-bit integers')
    bits = packed.shape[-1] / 16 if bits is None else bits
    whole, half = _bitrate(bits)
    if packed.shape[-1] != 16 * float(bits):
        raise ValueError('Packed word count does not match bitrate')
    words16 = packed.astype('<u2', copy=False)
    words = words16[..., 0::2].astype(np.uint32) | (words16[..., 1::2].astype(np.uint32) << 16)
    widths = whole + (np.arange(256) % 2 if half else np.zeros(256, dtype=np.int64))
    ends = np.cumsum(widths) + int(256 * float(bits))
    starts = ends - 16
    first = starts // 32
    last = (ends - 1) // 32
    shift = (last + 1) * 32 - ends
    a = words[..., first % words.shape[-1]].astype(np.uint64)
    b = words[..., last % words.shape[-1]].astype(np.uint64)
    return (((a << 32 | b) >> shift.astype(np.uint64)) & 65535).astype(np.uint16)


def _byte_sum(states: np.ndarray, mult: int) -> np.ndarray:
    product = (np.asarray(states, dtype=np.uint16).astype(np.uint64) * mult) & 0xffffffff
    return sum((product >> shift) & 255 for shift in [0, 8, 16, 24])


def int8mul24_q(states: np.ndarray) -> np.ndarray:
    """The INT8 weights q = (B - 509) >> 2 (arithmetic shift, -128 .. 127) of mul24 states."""
    return ((_byte_sum(states, MUL24).astype(np.int64) - 509) >> 2).astype(np.int8)


def decode_codebook(states: np.ndarray, codebook: str = 'mul1') -> np.ndarray:
    """Decode states to exact fp16 codebook values, including half FMA rounding.

    'int8mul24' returns float32 q * S_FIX (exact: |q * 887| < 2^17)."""
    states = np.asarray(states, dtype=np.uint16).astype(np.uint64)
    if codebook == 'int8mul24':
        return int8mul24_q(states).astype(np.float32) * S_FIX
    if codebook in ['mul1', 'mul24']:
        byte_sum = _byte_sum(states, MUL1 if codebook == 'mul1' else MUL24)
        h = (np.uint16(0x6400) + byte_sum.astype(np.uint16)).view(np.float16)
        inv = np.array(0x1eee, dtype=np.uint16).view(np.float16).astype(np.float64)
        bias = np.array(0xc931, dtype=np.uint16).view(np.float16).astype(np.float64)
        # Binary64 represents these fp16 operands' product + sum exactly.
        return (h.astype(np.float64) * inv + bias).astype(np.float16)
    if codebook not in ['mcg', '3inst']:
        raise ValueError('Unknown codebook')
    product = (states * (MCG if codebook == 'mcg' else 89226354) +
               (0 if codebook == 'mcg' else 64248484)) & 0xffffffff
    value = (product & 0x8fff8fff) ^ 0x3b603b60
    low = (value & 65535).astype(np.uint16).view(np.float16)
    high = (value >> 16).astype(np.uint16).view(np.float16)
    return (low.astype(np.float32) + high.astype(np.float32)).astype(np.float16)


def stored_codebook(stored: Mapping[str, np.ndarray]) -> str:
    """The codebook that a tensor's marker selects: 'mul1', 'mul24', 'mcg' or '3inst'. Any other marker value raises."""
    if 'mul1' in stored and 'mcg' in stored:
        raise ValueError('Ambiguous codebook markers')
    if 'mcg' in stored or 'mul1' not in stored:
        codebook = 'mcg' if 'mcg' in stored else '3inst'
        if codebook == 'mcg':
            marker = np.asarray(stored['mcg'])
            if marker.size != 1 or (int(marker.item()) & 0xffffffff) != MCG:
                raise ValueError('Unsupported codebook multiplier marker')
        return codebook
    marker = np.asarray(stored['mul1'])
    if marker.size != 1:
        raise ValueError(f'Unsupported codebook multiplier marker of {marker.size} values')
    return codebook_of(int(marker.item()))


def codebook_of(marker: int) -> str:
    """'mul1' or 'mul24' for a .mul1 value (signed or unsigned 32-bit); any other value raises."""
    value = int(marker) & 0xffffffff
    if value not in (MUL1, MUL24):
        raise ValueError(f'Unsupported codebook multiplier marker {value:#010x}: '
                         f'mul1 is 0x83DCD12D, mul24 is 0x00DCD12D')
    return 'mul1' if value == MUL1 else 'mul24'


def _stored(stored: Mapping[str, np.ndarray],
            codebook: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    trellis = np.asarray(stored['trellis'])
    suh, svh = np.asarray(stored['suh']), np.asarray(stored['svh'])
    if trellis.ndim != 3 or trellis.shape[:2] != (suh.size // 16, svh.size // 16):
        raise ValueError('Trellis shape disagrees with scales')
    if suh.ndim != 1 or svh.ndim != 1 or suh.size % 128 or svh.size % 128:
        raise ValueError('Linear dimensions must be multiples of 128')
    if suh.dtype != np.float16 or svh.dtype != np.float16:
        raise ValueError('suh and svh must be fp16')
    marked = stored_codebook(stored)
    if codebook is None or codebook == marked:
        codebook = marked
    elif not (marked == 'mul24' and codebook in MUL24_FORMS):
        raise ValueError(f'Codebook {codebook} does not match the marker ({marked}); only a mul24 marker has the '
                         f'forms {MUL24_FORMS}')
    if trellis.shape[-1] % 16 and codebook != 'mul1':
        raise ValueError('Half-integer bitrates require mul1')
    return trellis, suh, svh, codebook


def reconstruct_inner(stored_tensors: Mapping[str, np.ndarray], codebook: str | None = None) -> np.ndarray:
    """Return the rotated-basis decoded matrix [input, output]: fp16, float32 for 'int8mul24'.

    codebook: None takes the marker's codebook ('mul24' for a mul24 marker); 'int8mul24' needs a mul24 marker."""
    trellis, suh, svh, codebook = _stored(stored_tensors, codebook)
    tiles = trellis.reshape(-1, trellis.shape[-1])
    perm = tile_perm(codebook)
    decoded = np.empty((tiles.shape[0], 256), dtype=np.float32 if codebook == 'int8mul24' else np.float16)
    for first in range(0, tiles.shape[0], 1024):
        states = decode_tile_symbols(tiles[first:first + 1024])
        decoded[first:first + 1024, perm] = decode_codebook(states, codebook)
    return decoded.reshape(trellis.shape[0], trellis.shape[1], 16, 16).transpose(0, 2, 1, 3).reshape(suh.size, svh.size)


def _had_matrix() -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float32)
    for _ in range(7):
        h = np.block([[h, h], [h, -h]])
    return h * HAD_SCALE


def reconstruct(stored_tensors: Mapping[str, np.ndarray], codebook: str | None = None) -> np.ndarray:
    """get_weight_tensor rounding: fp32 Had matmul, fp16 after each transform/scale."""
    _, suh, svh, _ = _stored(stored_tensors, codebook)
    inner = reconstruct_inner(stored_tensors, codebook)
    h = _had_matrix()
    left = (h @ inner.astype(np.float32).reshape(-1, 128, svh.size)).reshape(inner.shape).astype(np.float16)
    left_scaled = (left.astype(np.float32) * suh.astype(np.float32)[:, None]).astype(np.float16)
    right = (left_scaled.astype(np.float32).reshape(suh.size, -1, 128) @ h).reshape(inner.shape).astype(np.float16)
    return (right.astype(np.float32) * svh.astype(np.float32)[None, :]).astype(np.float16).astype(np.float32)


def hadamard_kernel(x: np.ndarray, pre_scale: np.ndarray | None = None,
                    post_scale: np.ndarray | None = None) -> np.ndarray:
    """had_hf / had_ff natural-order 128 butterfly and its conversion points."""
    x = np.asarray(x)
    if x.dtype not in [np.dtype('float16'), np.dtype('float32')] or x.shape[-1] % 128:
        raise ValueError('Hadamard input must be fp16/fp32 with 128-divisible width')
    dtype = x.dtype
    scaled = x if pre_scale is None else (x.astype(np.float32) * np.asarray(pre_scale, dtype=np.float32)).astype(dtype)
    v = scaled.astype(np.float32).reshape(*x.shape[:-1], -1, 128)
    for stride in [1, 2, 4, 8, 16, 32, 64]:
        pair = v.reshape(*v.shape[:-1], -1, 2, stride)
        low, high = pair[..., 0, :], pair[..., 1, :]
        v = np.stack([low + high, low - high], axis=-2).reshape(v.shape)
    result = (v * HAD_SCALE).reshape(x.shape).astype(dtype)
    if post_scale is not None:
        result = (result.astype(np.float32) * np.asarray(post_scale, dtype=np.float32)).astype(dtype)
    return result


def matvec(x: np.ndarray, stored_tensors: Mapping[str, np.ndarray],
           out_dtype: np.dtype = np.float16, codebook: str | None = None) -> np.ndarray:
    """GEMV stage order; FP32 BLAS dot is a portable, not MMA-bit-exact, oracle.

    out_dtype is activation metadata, not encoded in the EXL3 weight tensors.
    Routed expert down projections use float32 output in ExLlamaV3.
    codebook: as for reconstruct_inner.
    """
    _, suh, svh, codebook = _stored(stored_tensors, codebook)
    x = np.asarray(x, dtype=np.float16)
    if x.shape != (suh.size,):
        raise ValueError('x must be one input vector')
    xh = hadamard_kernel(x, pre_scale=suh)
    dtype = np.dtype(out_dtype)
    if dtype not in [np.dtype('float16'), np.dtype('float32')]:
        raise ValueError('out_dtype must be fp16 or fp32')
    rotated = (xh.astype(np.float32) @ reconstruct_inner(stored_tensors, codebook).astype(np.float32)).astype(dtype)
    output = hadamard_kernel(rotated, post_scale=svh)
    if 'bias' in stored_tensors:
        output = (output.astype(np.float32) + np.asarray(stored_tensors['bias'], dtype=np.float32)).astype(dtype)
    return output
