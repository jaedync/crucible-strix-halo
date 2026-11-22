# EXL3 linear format and executable reference

## Scope and verdict

The decoder is validated against two real checkpoint tensors. Layer 0 QKV is **5 bpw**, not 3 bpw. Routed expert 0 down is **3 bpw**. Relative Frobenius errors versus the original BF16 weights are **0.03710084** and **0.14183896**, respectively. Their correlations are **0.99931177** and **0.98989475**. Both complete reconstructed matrices match upstream's CPU pure-Torch Hadamard/reconstruction operations exactly in this environment. These results validate storage decoding, not NVIDIA GPU accumulation equivalence. [Validation: `ref/fixtures/validation.json:5-37`; `ref/test_exl3_ref.py:102-137`.]

Source baseline: ExLlamaV3 commit `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`, cloned with the commands in section 7. Checkpoint version is 1.4.4. The branch name `3.05bpw_h5_ng5` resolves to `69e33439ae950f17bcbe95c98f117d80f759ab6d`. The original model is pinned to `de4b8e4d43b917e7706784d8bb445c9af86a3540`, rather than floating `main`. [`docs/EXL3_CHECKPOINT_METADATA.json:1-20`; `ref/make_fixtures.py:13-23`.]

### Source citation key

All upstream references below are file:line references relative to that source checkout, also available at `https://github.com/turboderp-org/exllamav3/blob/d3739fd393337b1ff4d6c2a342b12f0c87a9592f/<path>`.

| Key | File |
| --- | --- |
| Q | `exllamav3/modules/quant/exl3_lib/quantize.py` |
| L | `exllamav3/modules/quant/exl3.py` |
| P | `exllamav3/exllamav3_ext/quant/pack.cu` |
| F | `exllamav3/exllamav3_ext/quant/frac.cu` |
| D | `exllamav3/exllamav3_ext/quant/exl3_dq.cuh` |
| C | `exllamav3/exllamav3_ext/quant/codebook.cuh` |
| R | `exllamav3/exllamav3_ext/quant/reconstruct.cu` |
| H | `exllamav3/exllamav3_ext/quant/hadamard_inner.cuh` |
| V | `exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh` |
| G | `exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh` |
| M | `docs/EXL3_CHECKPOINT_METADATA.json`, committed compact checkpoint header evidence |

Thus `C:57-88` means `exllamav3/exllamav3_ext/quant/codebook.cuh:57-88`, not a prose-only citation.

## 1. Stored tensors and axes

Use `I = in_features`, `O = out_features`, and `K = bits per weight`. The decoder matrix has orientation **`[I, O]`**, so a row vector computes `y = x @ W`. Both dimensions are multiples of 128 for the supported transformed linear format. Each tile is 16 input rows by 16 output columns. [`L:181-201, 228-248`; `R:160-183`; `Q:831-845`.]

For a linear prefix `name`, this checkpoint stores:

| Tensor | Safetensors dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `name.trellis` | I16 | `[I/16, O/16, 16*K]` | Packed circular trellis bitstreams, one per 16x16 tile. Signed storage is not signed quantization. Reinterpret as unsigned words. |
| `name.suh` | F16 | `[I]` | Input-side signed scale vector. Contains signs **and magnitudes**, not just random signs. |
| `name.svh` | F16 | `[O]` | Output-side signed scale vector, including per-channel output scaling when enabled. |
| `name.mul1` | I32 | `[]` | Codebook presence marker. Expected unsigned bit pattern `0x83DCD12D` (signed int32 `-2082680531`). |

These types/shapes are checked by `L:41-81`, serialized by `Q:1663-1687`, and observed in `M:23-62, 285-325`. There is **no separate persistent global scale or output-scale tensor** in this format. `scale` is explicitly rejected by `L:38`, and the old scale/sign serialization is commented out at `Q:1669-1677`.

Other supported cases:

* Optional `name.mcg`, I32 scalar, expected bit pattern `0xCBAC1FED`, selects MCG instead of mul1. With neither marker, the legacy 3INST codebook is selected. The multiplier values are not read by upstream inference, only their presence is used. The NumPy reference rejects unknown marker values to avoid accepting an unsupported format. [`L:78-81`; `Q:1679-1687`; `R:185-190`; `ref/exl3_ref.py:89-107`.]
* Optional `name.bias`, conventionally F16 `[O]`, is added after the output transform. A supplied F32 bias is converted to F16 on construction. Neither fixture has bias. [`L:49, 219-223`; `exllamav3/exllamav3_ext/libtorch/linear.cpp:47-48`; `M:23-62, 285-325`.]
* Older files can have `name.su` and `name.sv`, I16 packed sign vectors of `I/16` and `O/16` words. Bit `j` of word `t` controls element `16*t+j`, with zero mapping to +1 and one to -1. Upstream unpacks them to F16. The supplied NumPy reference targets this checkpoint's unpacked `suh/svh`, not this old sign-only serialization. [`L:40-46, 62-63, 153-170`; `P:179-206`.]

### Per-tensor K, not the model's average

`K = trellis.shape[-1] / 16`. Integer rates 1 through 8 are supported. Half-integer 1.5, 2.5, 3.5 are supported by the reconstruction kernel, require mul1, and alternate `floor(K)` and `floor(K)+1` new bits. There is no K value attached to every tile. A tensor is uniform-rate, but different tensors may have different rates. [`L:66-81`; `Q:81-97`; `R:115-145`.]

The metadata's `bits: 3.05` is a model-level target, not a decoder argument. Per-tensor `tensor_storage[name].bits_per_weight` and stored shapes identify individual choices, but upstream inference derives K from the tensor itself. Here QKV has last dimension 80, hence K=5; expert down has last dimension 48, hence K=3. The shared expert projections also use K=5. [`M:6-19, 51-62, 171-182, 211-222, 251-262, 313-325`; `L:66-68`; the downloaded `quantization_config.json`, keys `tensor_storage.<name>.bits_per_weight`.]

## 2. Tile order, element order and bitstream

### Matrix tile order

The stored tile `(a,b)` reconstructs `W_hat[16*a:16*a+16, 16*b:16*b+16]`. Tiles are contiguous row-major in `(input_tile, output_tile)`: flat tile index is `a*(O/16)+b`. There is no extra Morton order or expert-specific tile permutation in the checkpoint. [`R:27-37, 87-92`; `G:150-152`; `Q:812, 833-845`.]

### Element order inside a tile

The 256 ring positions are in NVIDIA tensor-core fragment order, **not row-major**. For position `i = 8*t+j`, `0 <= t < 32`, `0 <= j < 8`:

```text
r0 = 2*(t % 4)
c0 = t // 4
r  = r0 + (j & 1) + (8 if (j & 2) else 0)
c  = c0 + (8 if (j & 4) else 0)
W_hat[16*a+r, 16*b+c] = codebook(state[i])
```

Equivalently each lane contributes `(r0,c0), (r0+1,c0), (r0+8,c0), (r0+9,c0), (r0,c0+8), (r0+1,c0+8), (r0+8,c0+8), (r0+9,c0+8)`. [`Q:22-43`; `F:36-43`; `R:49-83`.]

### Exact packing convention

Let `d_i = K` for integer K, or `floor(K)+(i % 2)` for half-integer K. Let `E_i = sum(d_0..d_i)` and `B = sum(d_0..d_255) = 256*K`. The packer appends `state[i] & ((1<<d_i)-1)` to a B-bit stream, **MSB first within that symbol**. Stream bits are consumed MSB first within each uint32 word. [`P:28-55`; `F:47-66`; `Q:81-90`.]

Physical memory is little-endian, but that does **not** make the logical stream LSB first. Pair stored uint16 words as:

```text
u32[q] = uint16(stored[2*q]) | (uint32(uint16(stored[2*q+1])) << 16)
stream bit p = (u32[p//32] >> (31-(p%32))) & 1
```

The integer packer creates MSB-first uint16 values then swaps their order within each uint32 with `SWAP16`. Thus if the first 32 stream bits spell hex `0x01234567`, the two stored uint16 words are `0x4567, 0x0123`, and the bytes are `67 45 23 01`. Do not byte-swap the uint32 again. [`P:41-55`; `exllamav3/exllamav3_ext/util.cuh:72`; `D:27-32`; `F:60-65`.]

### State extraction and wraparound

The full decoder state is the **16 stream bits ending at the end of symbol i**, not just the d_i new bits:

```text
state[i] = sum(stream[(E_i-16+j) % B] << (15-j) for j in 0..15)
```

The window wraps within the **same tile**. Initial positions read the tile's tail. No external initial state or extra 16-bit prefix is stored. For integer K, `E_i=(i+1)*K`; for half-integer K, `E_i=(i+1)*floor(K)+(i+1)//2`. [`D:18-33`; `P:116-136`; `F:18-33, 77-81`.]

An executable uint32 extraction, matching the normal scalar decoder, is:

```text
e = E_i + B
s = e - 16
q0 = s // 32
q1 = (e-1) // 32
shift = (q1+1)*32 - e
A = u32[q0 % (B//32)]
Bword = u32[q1 % (B//32)]
state = (((uint64(A)<<32) | Bword) >> shift) & 0xffff
```

If q0=q1, both words are the same. The extra ring length makes indices nonnegative. `decode_tile_symbols` implements this expression in NumPy and supports leading batch/tile dimensions. [`D:18-33`; `ref/exl3_ref.py:38-65`.]

Note: `F:9-13` has stale comments saying `16*K bits` and `2*K uint32`; actual 256-weight storage is **256*K bits, 16*K uint16, 8*K uint32**, as proved by the allocation, packing loops and runtime shape checks. Do not copy those comment factors into a kernel. [`F:50-65, 100-105`; `Q:93-97`; `G:54`.]

## 3. Bit-exact procedural codebooks

All integer products below wrap modulo 2^32. A state is unsigned 16-bit, promoted to uint32. All final codebook outputs are IEEE binary16, round-to-nearest ties-to-even. Upstream converts them to float32 only when a consumer requires it. [`C:57-88, 125-128`; `D:18-33`.]

### mul1 (cb=2)

```text
p = uint32(state * 0x83DCD12D)
b = (p&255) + ((p>>8)&255) + ((p>>16)&255) + ((p>>24)&255)
h = bitcast_fp16(uint16(0x6400+b))  # exact numeric value 1024+b
inv  = bitcast_fp16(0x1eee)         # 0.00676727294921875
bias = bitcast_fp16(0xc931)         # -10.3828125
value = fp16_fma(h, inv, bias)      # ONE rounding, not half multiply then half add
```

The four bytes are **unsigned**. `__dp4a`'s unsigned overload is selected by uint32 operands. `b` is in 0..1020, so `0x6400+b` is a valid half encoding of `1024+b`, with no integer-to-half conversion. A HIP implementation can sum four masked bytes instead of using a GPU dot instruction. Use a genuine half FMA or compute the product and sum exactly then round once to half. The reference uses binary64 for that last exact operation; exhaustive 65536-state testing agrees bit-for-bit with upstream's pure-Torch `mul1_codebook`. [`C:76-88`; `exllamav3/modules/quant/exl3_lib/ngram_codec.py:29-37`; `ref/exl3_ref.py:68-86`; `ref/test_exl3_ref.py:78-87`.]

### mcg (cb=1) and legacy 3inst (cb=0)

```text
mcg:   p = uint32(state * 0xCBAC1FED)
3inst: p = uint32(state * 89226354 + 64248484)
u = (p & 0x8fff8fff) ^ 0x3b603b60
lo = bitcast_fp16(uint16(u))
hi = bitcast_fp16(uint16(u >> 16))
value = fp16_add(lo, hi)            # one half addition rounding
```

The PTX `lop3` expression with immediate `0x6a` implements `(p & mask) ^ constant`. Pair decoders use the same arithmetic lane-wise, not a separate codebook. Both alternative decoders are tested over all 65536 states against a scalar bitcast/arithmetic oracle. [`C:57-74, 93-123`; `ref/test_exl3_ref.py:47-61`.]

### EXL3-INT8 mul24 (not upstream; the Hy4 full pack)

Spec: github.com/jyohukuchan/rocm_exl3_forme `doc/exl3_int8.md` ("mul24 形式") and `research/codebook/README.md`. Upstream
exllamav3 cannot read it. The pack `kusuriya-labs/Hy4-flash-EXL3-INT8-4bpw` uses it for every linear. The tensors, the
bitstream and the state extraction are the ones above. Two things differ:

* The `.mul1` value is `0x00DCD12D` (the low 24 bits of the mul1 constant; signed int32 `14471469`).
  `p = uint32(state * 0x00DCD12D)`, `b` = the byte sum as for mul1.
* The element order inside a tile is the WMMA B fragment order: position `i = 8*t+j` goes to
  `r = 8*(t // 16) + j`, `c = t % 16` (equivalently `r = 8*(i // 128) + i % 8`, `c = (i // 8) % 16`).

The value of a state has two forms. `mul24` (fp16): the mul1 formula on `b`, `fp16_fma(1024+b, 0x1eee, 0xc931)`; this
is the quantizer's reconstruction and the default of `ref/exl3_ref.py` and the engine (`kCbMul24`). `int8mul24`: the INT8
grid `q = (b - 509) >> 2` (arithmetic, -128..127) times `S_FIX = 4*887*2^-17 = 0.027069091796875` (`kCbMul24Int8`). `q` is the
rounding of the fp16 value to the grid for every `b`. Any other `.mul1` value is refused (`ref/exl3_ref.py codebook_of`,
`engine/exl3/kernels/codebook.h Exl3CodebookOfMarker`).

## 4. Hadamards, scales and operation order

Let `H_I` and `H_O` be block-diagonal, normalized 128x128 natural-order Sylvester Hadamards. A block element is `(-1)^popcount(r&c)/sqrt(128)`; these are symmetric and their own inverse in exact arithmetic. They operate on consecutive blocks, with no full-dimension transform or bit-reversal permutation. [`Q:15, 401-419`; `exllamav3/util/hadamard.py:36-44, 116-130, 143-146`; `H:22-49, 119-149`.]

In exact real arithmetic:

```text
W = diag(suh) @ H_I @ W_hat @ H_O @ diag(svh)
x_h = (x * suh) @ H_I
y_h = x_h @ W_hat
y = (y_h @ H_O) * svh + bias
```

The sign/scale vectors are outside the Hadamards: **input multiply before input transform; output multiply after output transform**. A sign-only interpretation is wrong. The global quantizer normalization is folded into suh. [`L:189-196, 219-248`; `V:208-226, 451-483`; `Q:1365-1389`.]

### `out_scales: always`

The quantizer computes relative per-output-channel RMS values, incorporates them into sv, divides weights by sv, then applies the output Hadamard. It subsequently computes input-channel scales, incorporates them into su, divides by su, and applies the input Hadamard. Global scale search multiplies the rotated weights and divides su by the same global scale. Finally the su/sv vectors are saved as F16 suh/svh. Scale refitting can further modify these vectors without changing the trellis. A decoder must use the saved vectors verbatim, not recompute RMS values or apply an additional codebook scale `1.24371088`. [`Q:1346-1389, 1647-1667`; `M:6-19`.]

### Conversion and rounding boundaries

**Plain inner reconstruction:** each tile is decoded directly into F16 `W_hat`; no Hadamards or scales occur in `ext.reconstruct`. [`R:41-43, 87-92`; `L:226-235`.]

**Upstream `get_weight_tensor`:**

1. Decode F16 W_hat.
2. Left Hadamard matmul in F32; cast result to F16.
3. Multiply by F16 suh; round to F16.
4. Right Hadamard matmul in F32; cast result to F16.
5. Multiply by F16 svh; round to F16.

`reconstruct(stored_tensors)` implements those boundaries and promotes the final F16 values exactly to F32 `[I,O]`. This is the fixture oracle. It is not the CUDA fused reconstruction rounding path. [`L:239-248`; `Q:401-419`; `ref/exl3_ref.py:129-137`.]

**Ordinary activation transforms:** a half input multiplied by suh is rounded to half before the butterfly. The 4-element local butterfly and 32-lane butterfly use F32 adds/subtracts, in increasing XOR-stride order. Normalize with the F32 constant `0.088388347648f`, cast to half once, then multiply post-scale in half if present. A F32 activation path retains F32 throughout, multiplying F16 scales promoted to F32. The standalone API uses `scale * 0.088388347648f`. [`H:108-145, 166-207`; `exllamav3/exllamav3_ext/quant/hadamard.cu:129-161`.]

`matvec(x, stored_tensors, out_dtype=np.float16)` uses those activation transforms around an F32 BLAS dot. `out_dtype=np.float32` models the output-side F32 transform. This parameter is module/activation metadata, **not encoded in the trellis**. Routed down projections are configured to return F32. [`ref/exl3_ref.py:140-178`; `exllamav3/modules/block_sparse_mlp.py:361-371`; `exllamav3/exllamav3_ext/libtorch/linear.cpp:56-66`.]

## 5. Accumulation precision is path-dependent

There is no single architecture-independent bit-exact EXL3 output arithmetic. The following paths share the same bit-exact decoded half weights:

* **Small-M GEMV:** FP16 `mma.m16n8k16` accumulator fragments. Narrow CFG0 splits K across 16 warps, folds half accumulators into F32 every 4 input tiles; wide CFG1 splits across 8 warps and folds every 2 tiles. Each warp receives a consecutive `ceil((I/16)/WK)`-tile K range. F32 cross-warp sums are accumulated in increasing warp order. Half output is rounded before the output Hadamard; F32 output is retained. Turing uses two k8 instructions and adds another rounding boundary. [`V:12-14, 38-73, 192-195, 230-237, 396-451, 465-480`.]
* **Regular fused EXL3 GEMM:** normally F32 MMA accumulators. On sm_86, the implementation uses F16 MMA accumulators folded to F32 once per configured K slice. Slice size, cross-block reduction and kernel choices depend on the dispatch/tuning. Output partial sums can pass through half global storage for half-output paths. The shared-memory output-Hadamard variant transforms its F32 reduced sums directly, so it does **not** necessarily round y_h to half before the transform. [`G:13-22, 44-56, 608-630, 644-670, 814-846, 904-921`.]
* **Reconstruct then dense GEMM:** upstream can choose a dense kernel using 32-term half-accumulator partials folded to F32, or cuBLAS with `CUBLAS_COMPUTE_32F`. This choice depends on device probes, shapes and environment. [`exllamav3/exllamav3_ext/hgemm_f16acc.cu:1-11, 535-567`; `exllamav3/exllamav3_ext/hgemm.cu:65-75`; `L:198-223`.]
* **Fused original-basis reconstruction:** the left 128-Hadamard uses half additions/subtractions, an early half normalization, then half warp butterflies. The right transform starts with F32 local additions, converts/normalizes to half, then performs half warp butterflies. It applies suh and svh as two half multiplies after both transforms. Those rounding points differ from `get_weight_tensor`. Used for sufficiently large prefill batches, not a canonical mathematical weight definition. [`R:206-218, 304-376`; `L:182-190`.]

The small-M GEMV selector covers integer K=2,3,4 and the half-integer mul1 rates, not K=5 QKV. Default choices depend on shape/device; forcing the same choice is necessary before comparing final numerical outputs. Experimental int8-activation GEMV adds further activation quantization and is not the baseline reference. [`exllamav3/exllamav3_ext/quant/exl3_gemv.cu:46-72, 112-121`; `exllamav3/exllamav3_ext/quant/exl3_gemm.cu:188-198, 230-241`.]

For gfx1151 kernels, the bit-exact requirement can be met for state extraction and decoded F16 codebook values. Match the documented activation conversion points. Do not claim bit-identical final outputs against an unspecified NVIDIA kernel. The NumPy middle dot intentionally does not emulate NVIDIA's architecture-specific MMA internal summation. [`ref/exl3_ref.py:1-8, 159-178`; paths cited above.]

## 6. Actual layer 0 MoE names and layouts

All listed EXL3 tensors are in `model-00001-of-00007.safetensors`. The quantized checkpoint has **separate per-expert projections**, not a stacked expert dimension in trellis. Upstream creates one Linear per expert and builds pointer tables for fused execution. Pointer tables are runtime objects, not checkpoint tensor layouts. [`M:20-282`; `exllamav3/modules/block_sparse_mlp.py:308-378, 637-646`.]

Full router names:

* `model.language_model.layers.0.mlp.gate.weight`: F16 `[512,2560]`, plain router, HF `[output,input]` convention.
* `model.language_model.layers.0.mlp.shared_expert_gate.weight`: F16 `[1,2560]`, plain shared-expert scalar gate.

[`M:263-284`; ordinary load transposes HF weights at `exllamav3/modules/linear.py:195, 230`.]

Full EXL3 linear prefixes, each with **`.trellis`, `.suh`, `.svh`, `.mul1`**:

| Prefix | `[I,O]` | Trellis shape | K |
| --- | --- | --- | --- |
| `model.language_model.layers.0.mlp.shared_expert.gate_proj` | `[2560,640]` | `[160,40,80]` | 5 |
| `model.language_model.layers.0.mlp.shared_expert.up_proj` | `[2560,640]` | `[160,40,80]` | 5 |
| `model.language_model.layers.0.mlp.shared_expert.down_proj` | `[640,2560]` | `[40,160,80]` | 5 |
| `model.language_model.layers.0.mlp.experts.0.gate_proj` | `[2560,640]` | `[160,40,48]` | 3 |
| `model.language_model.layers.0.mlp.experts.0.up_proj` | `[2560,640]` | `[160,40,48]` | 3 |
| `model.language_model.layers.0.mlp.experts.0.down_proj` | `[640,2560]` | `[40,160,48]` | 3 |

[`M:23-262`.]

The original BF16 checkpoint instead contains stacked `model.language_model.layers.0.mlp.experts.down_proj`, shape `[512,2560,640]`, in `model-00003-of-00131.safetensors`. Its expert 0 slab is `[O,I]`, so it is transposed for comparison. The generator reads only that slab. Original QKV is `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`, `[10240,2560]`, also transposed for comparison. [`ref/make_fixtures.py:16-21, 47-50`; ranged headers of the original checkpoint, made during the audit (not in this repository).]

## 7. Validation and reproduction

### Exact commands executed

```bash
git clone -q https://github.com/turboderp-org/exllamav3.git /tmp/crucible-exllamav3
git -C /tmp/crucible-exllamav3 rev-parse HEAD
# d3739fd393337b1ff4d6c2a342b12f0c87a9592f

cd /path/to/crucible
uv venv --python 3.12 /path/to/crucible/.venv
uv pip install --python /path/to/crucible/.venv/bin/python numpy pytest torch
uv pip install --python .venv/bin/python pytest-cov
.venv/bin/python ref/make_fixtures.py
EXLLAMAV3_SRC=/tmp/crucible-exllamav3 .venv/bin/python -m pytest \
  ref/test_exl3_ref.py -q -s --cov=exl3_ref --cov-report=term-missing
```

Environment: CPython 3.12.12, macOS arm64; NumPy 2.5.3, pytest 9.1.1, Torch 2.14.1, pytest-cov 7.1.0. The installed macOS torch wheel has CPU support and does not require CUDA. Every upstream validation tensor is explicitly on CPU. No full package import or CUDA extension build was attempted: tests extract unchanged upstream function bodies with Python AST to bypass CUDA-only package initialization. The tests execute upstream `mul1_codebook`, `tensor_core_perm`, `sylvester`, `get_hadamard_dt`, `preapply_had_l`, and `preapply_had_r`. [`ref/test_exl3_ref.py:64-100`; source functions cited above.]

No complete pure-Torch **linear trellis** unpacker exists in the inspected quantization/reconstruction paths: packing/unpacking and reconstruction call the CUDA extension. The pure-Torch ngram row packer has a different LSB-first format and is not a valid linear packing oracle. Tests therefore check linear packing against an independent literal MSB-first bit-ring oracle for all 11 supported rates, including boundary wraparound, while checking the shared mul1 codebook exhaustively against upstream. [`Q:1077-1092`; `L:226-235`; `exllamav3/modules/quant/exl3_lib/ngram_codec.py:40-74`; `ref/test_exl3_ref.py:24-39, 82-93`.]

### Results

**22 tests passed. `ref/exl3_ref.py` statement coverage: 94%.** Pure-Torch reconstructed weights versus NumPy: max absolute difference 0, unequal elements 0 for both matrices. All 65536 mul1 states match upstream exactly; all 65536 mcg and 3inst states match independent scalar bitcast oracles. Fixture tests reconstruct the raw stored values and compare every element with the saved full-matrix oracle. [`ref/test_exl3_ref.py:24-39, 55-71, 82-93, 115-156`.]

| Tensor | K | Relative Frobenius error | Pearson correlation |
| --- | --- | --- | --- |
| Layer 0 QKV | 5 | 0.03710083556334977 | 0.9993117734948177 |
| Layer 0 expert 0 down | 3 | 0.14183896026714896 | 0.9898947545643074 |

The metric uses `||W_reconstructed-W_BF16.T||_F / ||W_BF16||_F`, and correlation is over all flattened elements, computed in float64. These are weight-space errors, not end-to-end model quality or activation/Hessian-weighted errors. [`ref/make_fixtures.py:36-38`; `ref/fixtures/validation.json:5-37`.]

### Range reader and fixtures

`ref/checkpoint_ranges.py` requests bytes 0..7, parses the little-endian uint64 header length, then requests exactly the JSON header. Tensor `data_offsets` are relative to `8+header_length`; for a stacked BF16 expert, the reader computes the contiguous expert slab before issuing the payload range. A 200 response, an unexpected Content-Range, or a wrong payload length raises an error **before a full shard can be consumed**. BF16 conversion is bitwise `uint32(bits)<<16`, viewed as F32. [`ref/checkpoint_ranges.py:13-31, 34-71`; `ref/test_exl3_ref.py:158-191`.]

Two local, git-ignored fixtures contain the complete original raw EXL3 tensors and complete reconstructed W:

* `ref/fixtures/layer0_qkv.npz`: 63,566,219 bytes.
* `ref/fixtures/layer0_expert0_down.npz`: 3,560,530 bytes.

Total **67,126,749 bytes (64.02 MiB)**, slightly above the requested approximate 60 MB cap. Full lossless matrices were retained rather than truncating or reducing precision. The archives use standard ZIP BZIP2 members, readable by `np.load`. `W` is stored F16 because the upstream reconstructed result is F16; `fixture['W'].astype(np.float32)` reproduces every returned reference value exactly. The raw tensors retain their native dtypes. Fixtures include metrics and the SHA256 of the original F32-promoted BF16 tensor. [`ref/make_fixtures.py:24-33, 52-60`; `ref/fixtures/validation.json:15, 32`; `.gitignore:5`.]

Fixture generation is pinned to repository commit SHAs, uses fixed ZIP member timestamps, and requires no GPU. Run `ref/make_fixtures.py` to regenerate after cloning Crucible. Tests skip fixture-dependent cases with a clear instruction when the files are absent. CPU source cross-checks require the upstream clone or `EXLLAMAV3_SRC`. [`ref/make_fixtures.py:12-33`; `ref/test_exl3_ref.py:66-67, 105-106, 120-121, 138-139`.]

## 8. Kernel-author caveats and unresolved verification

1. **Decoded state/codebook identity is verified. Final GPU arithmetic is not.** No CUDA kernel or gfx1151 kernel was run. The portable matvec uses F32 BLAS and is an operation-order oracle, not a NVIDIA MMA simulator.
2. **Choose a reconstruction target.** The fixture W targets `get_weight_tensor`, which matches upstream pure-Torch CPU operations exactly here. Fused CUDA original-basis reconstruction deliberately uses different half rounding. Those outputs cannot all be one bit-identical W oracle.
3. **Choose activation output dtype and GEMV/GEMM variant explicitly.** Routed expert down defaults to F32 output. K=5 QKV does not enter the specialized small-M GEMV path. Generic GEMM shared-memory output Hadamards can retain F32 y_h instead of rounding it first.
4. **Only half-integer fractional rates are part of this executable reference.** Generic period-16 masks exist in the fractional packer, but the deployed inference instances use the alternating `0xAAAA` pattern. The tensor shape alone cannot encode an arbitrary mask.
5. **The pure-Torch ngram codec is not the linear bitstream codec.** Reuse its mul1 codebook test, not its row bit packing.
6. **Marker values are locked constants, but upstream checks presence.** Supporting a new multiplier would require changing the GPU codebook, not simply reading a different stored scalar.

Evidence: accumulation paths in section 5; `Q:81-90`; `L:78-81`; `Q:1679-1687`; ngram format at `exllamav3/modules/quant/exl3_lib/ngram_codec.py:5-14, 40-74`. No storage permutation, extra tile state, separate global scale, or missing output-scale tensor remains ambiguous for the two tested tensors.
