// QSA prefill attention over row tiles. Four consecutive query rows share one
// workgroup and one key list: the union of their selected blocks (plus causal
// tails, or all blocks for dense rows). Every key and value is loaded once per
// tile instead of once per row, and the 4 rows x 12 heads fill three WMMA row
// tiles. A per-block bit mask keeps each row's own selection exact.
#pragma once
#include "prefill_qsa.h"
#include "qsa_union_lds.h"
#include <cstdlib>
#include <string>

namespace crucible::prefill {
constexpr int kUnionRows = 4;
constexpr int kUnionCap = kUnionRows * kSelBlocks + 2 * kUnionRows;  // sel + tail blocks
constexpr int kUnionKeys = 32;                                    // keys per flash step
constexpr int kUnionPad = kUnionKeys + 8;

struct QsaUnion {
  int* list;   // [tiles][kUnionCap]: block | row mask << 24, ascending blocks
  int* count;  // [tiles]
};

// mask[b] bit t: row t of the tile attends block b. Dense rows see blocks
// 0..p/4; sparse rows see their selected blocks and blocks nb..p/4 (the tail).
__global__ void __launch_bounds__(256) QsaUnionBuildKernel(QsaArgs a, QsaUnion u) {
  extern __shared__ unsigned char mask[];
  __shared__ int cnt[16][256], total[16], base[16];
  const int tile = blockIdx.x, r0 = tile * kUnionRows, t = threadIdx.x;
  const int rows = min(kUnionRows, a.R - r0);
  const int blocks = (a.pos0 + r0 + rows - 1) / 4 + 1;
  for (int b = t; b < blocks; b += 256) mask[b] = 0;
  __syncthreads();
  for (int i = 0; i < rows; ++i) {
    const int r = r0 + i, p = a.pos0 + r, ns = a.nsel[r];
    if (ns < 0) {
      for (int b = t; b <= p / 4; b += 256) mask[b] |= 1 << i;
    } else {
      for (int s = t; s < ns; s += 256) mask[a.sel[(size_t)r * kSelBlocks + s]] |= 1 << i;
      for (int b = (p + 1) / 4 + t; b <= p / 4; b += 256) mask[b] |= 1 << i;
    }
    __syncthreads();
  }
  // Stable sort by mask (15 down to 1), then block: each thread counts the
  // masks of one contiguous span, thread m scans mask m over the spans, and
  // each thread writes its span. Steps of one mask then leave whole WMMA row
  // tiles idle in the flash kernel. The order is deterministic.
  const int span = (blocks + 255) / 256, b0 = t * span, b1 = min(blocks, b0 + span);
  for (int m = 0; m < 16; ++m) cnt[m][t] = 0;
  for (int b = b0; b < b1; ++b) ++cnt[mask[b]][t];
  __syncthreads();
  if (t < 16) {
    int run = 0;
    for (int i = 0; i < 256; ++i) { const int c = cnt[t][i]; cnt[t][i] = run; run += c; }
    total[t] = run;
  }
  __syncthreads();
  if (t == 0) {
    int run = 0;
    for (int m = 15; m >= 1; --m) { base[m] = run; run += total[m]; }
    u.count[tile] = run;
  }
  __syncthreads();
  int* list = u.list + (size_t)tile * kUnionCap;
  for (int b = b0; b < b1; ++b) {
    const int m = mask[b];
    if (m) list[base[m] + cnt[m][t]++] = b | m << 24;
  }
}

// One workgroup per (row tile, KV group). M = 4 rows x 12 heads = 48 query
// rows in three WMMA tiles, m = row * 12 + head. Each step takes 32 keys:
// waves 0..5 compute the 48x32 scores (hi/lo q, exact fp16 k), all threads run
// the online softmax, then each wave applies P (hi/lo) to 32 of the 256 value
// dimensions with V transposed through LDS.
// LO adds the low halves of q and P (QsaPrefill::q_lo).
template <bool LO = true>
__global__ void __launch_bounds__(256) QsaUnionFlashKernel(QsaPrefill a, QsaUnion u) {
  __shared__ float score[48][kUnionKeys], maxima[48], sums[48], scale[48];
  __shared__ __attribute__((aligned(16))) _Float16 phi[48][kUnionPad], plo[LO ? 48 : 1][kUnionPad];
  __shared__ __attribute__((aligned(16))) _Float16 vt[256][kUnionPad];
  __shared__ int keypos[kUnionKeys], keymask[kUnionKeys], stepmask;
  const QsaArgs q = a.input;
  const int tile = blockIdx.x, g = blockIdx.y, r0 = tile * kUnionRows;
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const int rows = min(kUnionRows, q.R - r0);
  const int n = u.count[tile] * 4;
  const int* list = u.list + (size_t)tile * kUnionCap;
  if (t < 48) { maxima[t] = -INFINITY; sums[t] = 0.f; }
  F8 out[3][2] = {};
  // QK operand rows for waves 0..5: M tile mt = wave / 2, key tile nt = wave % 2.
  const int mt = wave >> 1, nt = wave & 1;
  const int qm = mt * 16 + (lane & 15), qrow = qm / 12, qhead = qm % 12;
  const bool qvalid = qrow < rows;
  const size_t qi = ((size_t)(r0 + min(qrow, rows - 1)) * 24 + g * 12 + qhead) * 256;
  for (int start = 0; start < n; start += kUnionKeys) {
    __syncthreads();
    if (t < kUnionKeys) {
      const int idx = start + t;
      const int e = idx < n ? list[idx >> 2] : 0;
      // Keys past the last row of this call are never visible. Marking them
      // absent keeps the K/V loads inside the cache (ctx need not be a
      // multiple of 4) and off uninitialized rows, whose V could be non-finite.
      const int pos = (e & 0xffffff) * 4 + (idx & 3);
      keypos[t] = idx < n && pos < q.pos0 + q.R ? pos : -1;
      int km = idx < n ? (e >> 24) & 0xff : 0;
      keymask[t] = km;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) km |= __shfl_xor(km, o);
      if (t == 0) stepmask = km;
    }
    __syncthreads();
    // M tile mt holds query rows m = 16 mt .. 16 mt + 15, i.e. tile rows
    // {0,1}, {1,2}, {2,3}. A tile with no row in this step's masks has no
    // visible key: its scores are -inf, its softmax state is unchanged
    // (scale 1, sum + 0) and its P V update is skipped.
    const int sm = stepmask;
    const bool live0 = sm & 3, live1 = sm & 6, live2 = sm & 12;
    // V^T: 32 keys x 256 dims, 16-byte loads, transposed into LDS. Lanes of a
    // wave take different keys, so their 2-byte stores hit different banks.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int part = t + i * 256, key = part & 31, d0 = (part >> 5) * 8;
      const int pos = keypos[key];
      uint4 raw = {};
      if (pos >= 0) raw = *reinterpret_cast<const uint4*>(q.c.v + ((size_t)pos * 2 + g) * 256 + d0);
      const _Float16* h = reinterpret_cast<const _Float16*>(&raw);
#pragma unroll
      for (int k = 0; k < 8; ++k) vt[d0 + k][key] = h[k];
    }
    if (wave < 6 && !(mt == 0 ? live0 : mt == 1 ? live1 : live2)) {
#pragma unroll
      for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][nt * 16 + (lane & 15)] = -INFINITY;
    } else if (wave < 6) {
      const int key = nt * 16 + (lane & 15), pos = keypos[key];
      const __half* kp = q.c.k + ((size_t)max(pos, 0) * 2 + g) * 256;
      F8 high = {}, low = {};
#pragma unroll 4
      for (int k = 0; k < 256; k += 16) {
        const H16 ah = qvalid ? HalfFragment(a.q_hi + qi + k) : H16{};
        const H16 b = pos >= 0 ? HalfFragment(kp + k) : H16{};
        high = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high);
        if constexpr (LO) {
          const H16 al = qvalid ? HalfFragment(a.q_lo + qi + k) : H16{};
          low = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b, low);
        }
      }
      F8 dots = high;
      if constexpr (LO) dots = high + low * (1.f / kResidualScale);
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int m = mt * 16 + FragmentRow(z), row = m / 12;
        const bool visible = pos >= 0 && row < rows && (keymask[key] >> row & 1)
            && pos <= q.pos0 + r0 + row;
        score[m][key] = visible ? dots[z] * .0625f : -INFINITY;
      }
    }
    __syncthreads();
    // Online softmax: four threads per query row, eight keys each.
    if (t < 192 && ((t >> 6) == 0 ? live0 : (t >> 6) == 1 ? live1 : live2)) {
      const int m = t >> 2, part = t & 3;
      float maximum = -INFINITY;
#pragma unroll
      for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, score[m][part * 8 + i]);
      maximum = fmaxf(maximum, __shfl_xor(maximum, 1));
      maximum = fmaxf(maximum, __shfl_xor(maximum, 2));
      const float old_max = maxima[m], new_max = fmaxf(old_max, maximum);
      // A row with no visible key yet keeps zero weights.
      const float base = new_max == -INFINITY ? 0.f : new_max;
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int col = part * 8 + i;
        const float p = __expf(score[m][col] - base);
        const _Float16 h = (_Float16)p;
        phi[m][col] = h;
        if constexpr (LO) plo[m][col] = (_Float16)((p - (float)h) * kResidualScale);
        sum += p;
      }
      sum += __shfl_xor(sum, 1);
      sum += __shfl_xor(sum, 2);
      if (part == 0) {
        const float s = old_max == -INFINITY ? 0.f : __expf(old_max - base);
        maxima[m] = new_max; scale[m] = s; sums[m] = sums[m] * s + sum;
      }
    }
    __syncthreads();
    // P V: wave owns value dims [32 wave, 32 wave + 32) for all 48 query rows.
#pragma unroll
    for (int m3 = 0; m3 < 3; ++m3) {
      if (!(m3 == 0 ? live0 : m3 == 1 ? live1 : live2)) continue;
      F8 high[2] = {}, low[2] = {};
#pragma unroll
      for (int k = 0; k < kUnionKeys; k += 16) {
        const H16 ah = *reinterpret_cast<const H16*>(&phi[m3 * 16 + (lane & 15)][k]);
#pragma unroll
        for (int d2 = 0; d2 < 2; ++d2) {
          const H16 b = *reinterpret_cast<const H16*>(&vt[wave * 32 + d2 * 16 + (lane & 15)][k]);
          high[d2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high[d2]);
          if constexpr (LO) {
            const H16 al = *reinterpret_cast<const H16*>(&plo[m3 * 16 + (lane & 15)][k]);
            low[d2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, b, low[d2]);
          }
        }
      }
#pragma unroll
      for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
        for (int z = 0; z < 8; ++z)
          out[m3][d2][z] = LO ? out[m3][d2][z] * scale[m3 * 16 + FragmentRow(z)]
              + high[d2][z] + low[d2][z] * (1.f / kResidualScale)
              : out[m3][d2][z] * scale[m3 * 16 + FragmentRow(z)] + high[d2][z];
    }
  }
#pragma unroll
  for (int m3 = 0; m3 < 3; ++m3)
#pragma unroll
    for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int m = m3 * 16 + FragmentRow(z), row = m / 12, head = m % 12;
        const int d = wave * 32 + d2 * 16 + (lane & 15);
        if (row < rows)
          a.context[((size_t)(r0 + row) * 24 + g * 12 + head) * 256 + d] = out[m3][d2][z] / sums[m];
      }
}

constexpr int kUnionScoreBytes = 48 * kUnionKeys * 4, kUnionPhiBytes = 48 * kUnionPad * 2;
constexpr int kUnionVtBytes = 256 * kUnionPad * 2;
constexpr int kUnionLoopBytes = kUnionScoreBytes + kUnionPhiBytes + kUnionVtBytes;
// The fp16 gate outputs of 48 query rows reuse the dead loop buffers.
static_assert(48 * 256 * 2 <= kUnionLoopBytes);

// Default no-lo union flash (half-q). Waves 0..2 hold q chunks 0..7 of M tile `wave` in
// VGPRs Waves 0..2 hold q chunks 0..7 of M tile `wave` in
// VGPRs and run the first half of each QK chain; waves 3..5 hold chunks 8..15
// and finish the chain from the fp32 partial sums in `score` (the WMMA C
// operand round-trips exactly). Waves 3..7 transpose V during the first half.
// All eight waves then run the production softmax and P V, so each score and
// output element keeps the production operands and accumulation order.
// Gate fuses QsaPrefillGateKernel into the epilogue (see below).
// Grid (2, tiles): the two KV groups of a tile run side by side and read the
// two halves of the same K and V rows at the same steps (DRAM locality). CU
// mode keeps the 8 waves on one CU, so the 3 waves that load the same K
// fragments share its L0, and barriers need no L0 invalidate. Occupancy stays
// 2 workgroups per CU (LDS and VGPRs).
#ifdef __HIP_DEVICE_COMPILE__
#define QSA_UNION_CU_MODE __attribute__((target("cumode")))
#else
#define QSA_UNION_CU_MODE  // the host compile does not know the feature
#endif
template <bool Gate = true>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8))) QSA_UNION_CU_MODE
QsaUnionHalfQKernel(QsaPrefill a, QsaUnion u) {
  __shared__ __attribute__((aligned(16))) unsigned char smem[kUnionLoopBytes];
  __shared__ float maxima[48], sums[48], scale[48];
  __shared__ int keypos[kUnionKeys], keymask[kUnionKeys], stepmask;
  auto score = reinterpret_cast<float(*)[kUnionKeys]>(smem);
  auto phi = reinterpret_cast<_Float16(*)[kUnionPad]>(smem + kUnionScoreBytes);
  auto vt = reinterpret_cast<_Float16(*)[kUnionPad]>(smem + kUnionScoreBytes + kUnionPhiBytes);
  const QsaArgs q = a.input;
  const int tile = blockIdx.y, g = blockIdx.x, r0 = tile * kUnionRows;
  const int t = threadIdx.x, lane = t & 31, wave = __builtin_amdgcn_readfirstlane(t >> 5);
  const int rows = min(kUnionRows, q.R - r0);
  const int n = u.count[tile] * 4;
  const int* list = u.list + (size_t)tile * kUnionCap;
  if (t < 48) { maxima[t] = -INFINITY; sums[t] = 0.f; }
  F8 out[3][2] = {};
  const bool first = wave < 3, second = wave >= 3 && wave < 6;
  const int mt = first ? wave : second ? wave - 3 : 0, c0 = second ? 8 : 0;
  const int qm = mt * 16 + (lane & 15), qrow = qm / 12, qhead = qm % 12;
  // Rows past the tile load a clamped row. Their scores are never visible.
  const size_t qi = ((size_t)(r0 + min(qrow, rows - 1)) * 24 + g * 12 + qhead) * 256 + c0 * 16;
  // Tile row of each fragment element, 4 bits per z. Rebuilt from an opaque
  // copy each step so the compiler does not hoist eight live row values.
  unsigned rowpack = 0;
#pragma unroll
  for (int z = 0; z < 8; ++z) rowpack |= (unsigned)((mt * 16 + FragmentRow(z)) / 12) << (4 * z);
  H16 qf[8];
  if (wave < 6) {
#pragma unroll
    for (int c = 0; c < 8; ++c) qf[c] = HalfFragment(a.q_hi + qi + c * 16);
  }
  for (int start = 0; start < n; start += kUnionKeys) {
    __syncthreads();
    if (t < kUnionKeys) {
      const int idx = start + t;
      const int e = idx < n ? list[idx >> 2] : 0;
      const int pos = (e & 0xffffff) * 4 + (idx & 3);
      keypos[t] = idx < n && pos < q.pos0 + q.R ? pos : -1;
      int km = idx < n ? (e >> 24) & 0xff : 0;
      keymask[t] = km;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) km |= __shfl_xor(km, o);
      if (t == 0) stepmask = km;
    }
    __syncthreads();
    const int sm = stepmask;
    const bool live0 = sm & 3, live1 = sm & 6, live2 = sm & 12;
    const bool live = mt == 0 ? live0 : mt == 1 ? live1 : live2;
    const int pos0 = keypos[lane & 15], pos1 = keypos[16 + (lane & 15)];
    if (first) {
      if (live) {
        // The chains of both key tiles run interleaved, each in the order
        // c = 0..7. Fragments c + 1 load before the WMMAs of c, so the loads
        // of one chain overlap the other chain's WMMA.
        // Absent keys read position 0. Each key feeds only its own score
        // column, which the visibility test sets to -inf.
        const __half* kp0 = q.c.k + ((size_t)max(pos0, 0) * 2 + g) * 256;
        const __half* kp1 = q.c.k + ((size_t)max(pos1, 0) * 2 + g) * 256;
        F8 high[2] = {};
        H16 b0 = HalfFragment(kp0), b1 = HalfFragment(kp1);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          high[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], b0, high[0]);
          high[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], b1, high[1]);
          if (c < 7) { b0 = HalfFragment(kp0 + (c + 1) * 16); b1 = HalfFragment(kp1 + (c + 1) * 16); }
          // More loads in flight spill VGPRs (191 of 192 used).
          __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][nt * 16 + (lane & 15)] = high[nt][z];
      }
    } else {
      // V^T: 32 keys x 256 dims in 1024 16-byte parts over 160 threads.
      for (int part = t - 96; part < 1024; part += 160) {
        const int key = part & 31, v0 = (part >> 5) * 8;
        const int pos = keypos[key];
        uint4 raw = {};
        if (pos >= 0) raw = *reinterpret_cast<const uint4*>(q.c.v + ((size_t)pos * 2 + g) * 256 + v0);
        const _Float16* h = reinterpret_cast<const _Float16*>(&raw);
#pragma unroll
        for (int k = 0; k < 8; ++k) vt[v0 + k][key] = h[k];
      }
    }
    __syncthreads();
    if (second) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int key = nt * 16 + (lane & 15), pos = nt ? pos1 : pos0;
        if (!live) {
#pragma unroll
          for (int z = 0; z < 8; ++z) score[mt * 16 + FragmentRow(z)][key] = -INFINITY;
          continue;
        }
        F8 high;
#pragma unroll
        for (int z = 0; z < 8; ++z) high[z] = score[mt * 16 + FragmentRow(z)][key];
        unsigned rz = rowpack;
        asm volatile("" : "+v"(rz));
        const __half* kp = q.c.k + ((size_t)max(pos, 0) * 2 + g) * 256 + 128;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          high = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], HalfFragment(kp + c * 16), high);
          if (c % 4 == 3) __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for (int z = 0; z < 8; ++z) {
          const int m = mt * 16 + FragmentRow(z), row = rz >> (4 * z) & 15;
          const bool visible = pos >= 0 && row < rows && (keymask[key] >> row & 1)
              && pos <= q.pos0 + r0 + row;
          score[m][key] = visible ? high[z] * .0625f : -INFINITY;
        }
      }
    }
    __syncthreads();
    if (t < 192 && ((t >> 6) == 0 ? live0 : (t >> 6) == 1 ? live1 : live2)) {
      const int m = t >> 2, part = t & 3;
      float maximum = -INFINITY;
#pragma unroll
      for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, score[m][part * 8 + i]);
      maximum = fmaxf(maximum, __shfl_xor(maximum, 1));
      maximum = fmaxf(maximum, __shfl_xor(maximum, 2));
      const float old_max = maxima[m], new_max = fmaxf(old_max, maximum);
      const float base = new_max == -INFINITY ? 0.f : new_max;
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int col = part * 8 + i;
        const float p = __expf(score[m][col] - base);
        phi[m][col] = (_Float16)p;
        sum += p;
      }
      sum += __shfl_xor(sum, 1);
      sum += __shfl_xor(sum, 2);
      if (part == 0) {
        const float s = old_max == -INFINITY ? 0.f : __expf(old_max - base);
        maxima[m] = new_max; scale[m] = s; sums[m] = sums[m] * s + sum;
      }
    }
    __syncthreads();
#pragma unroll
    for (int m3 = 0; m3 < 3; ++m3) {
      if (!(m3 == 0 ? live0 : m3 == 1 ? live1 : live2)) continue;
      F8 high[2] = {};
#pragma unroll
      for (int k = 0; k < kUnionKeys; k += 16) {
        const H16 ah = *reinterpret_cast<const H16*>(&phi[m3 * 16 + (lane & 15)][k]);
#pragma unroll
        for (int d2 = 0; d2 < 2; ++d2) {
          const H16 b = *reinterpret_cast<const H16*>(&vt[wave * 32 + d2 * 16 + (lane & 15)][k]);
          high[d2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, b, high[d2]);
        }
      }
#pragma unroll
      for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
        for (int z = 0; z < 8; ++z)
          out[m3][d2][z] = out[m3][d2][z] * scale[m3 * 16 + FragmentRow(z)] + high[d2][z];
    }
  }
  if constexpr (!Gate) {
#pragma unroll
    for (int m3 = 0; m3 < 3; ++m3)
#pragma unroll
      for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
        for (int z = 0; z < 8; ++z) {
          const int m = m3 * 16 + FragmentRow(z), row = m / 12, head = m % 12;
          const int d = wave * 32 + d2 * 16 + (lane & 15);
          if (row < rows)
            a.context[((size_t)(r0 + row) * 24 + g * 12 + head) * 256 + d] = out[m3][d2][z] / sums[m];
        }
    return;
  }
  // QsaPrefillGateKernel, element by element, on the context values that it
  // would read back: x = context * sigmoid(gate), then RoundH(RoundH(x) * suh_o),
  // which fp16 holds exactly. The fp16 rows reuse the dead loop buffers.
  auto gated = reinterpret_cast<_Float16(*)[256]>(smem);
  __syncthreads();
#pragma unroll
  for (int m3 = 0; m3 < 3; ++m3)
#pragma unroll
    for (int d2 = 0; d2 < 2; ++d2)
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const int m = m3 * 16 + FragmentRow(z), row = m / 12, h = g * 12 + m % 12;
        const int d = wave * 32 + d2 * 16 + (lane & 15);
        if (row < rows) {
          const float context = out[m3][d2][z] / sums[m];
          const float gate = Sigmoid(q.q[((size_t)(r0 + row) * 24 + h) * 512 + 256 + d]);
          const float x = context * gate;
          gated[m][d] = (_Float16)(RoundH(x) * __half2float(q.suh_o[h * 256 + d]));
        }
      }
  __syncthreads();
  // Wht128 over each 128-dim half, with the gate kernel's lane mapping.
  for (int job = wave; job < 96; job += 8) {
    const int m = job >> 1, half = job & 1, row = m / 12, h = g * 12 + m % 12;
    if (row >= rows) continue;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = (float)gated[m][half * 128 + lane * 4 + i];
    Wht128(v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      q.xh_o[OutProjIndex(r0 + row, h * 256 + half * 128 + lane * 4 + i, q.pack_rows, q.pack_off)] =
          __float2half(v[i] * kHadNorm);
  }
}


inline size_t QsaUnionListInts(int rows) {
  return (size_t)((rows + kUnionRows - 1) / kUnionRows) * kUnionCap;
}

// Same work as LaunchQsaFlash for one part (pos0 + R <= context).
inline void LaunchQsaUnionFlash(QsaPrefill a, QsaUnion u, hipStream_t stream = nullptr) {
  const auto q = a.input;
  const int tiles = (q.R + kUnionRows - 1) / kUnionRows;
  const int blocks = (q.pos0 + q.R - 1) / 4 + 1;
  QsaUnionBuildKernel<<<tiles, 256, blocks, stream>>>(q, u);
  if (a.q_lo) QsaUnionFlashKernel<true><<<dim3(tiles, 2), 256, 0, stream>>>(a, u);
  else QsaUnionHalfQKernel<false><<<dim3(2, tiles), 256, 0, stream>>>(a, u);
}

// LaunchQsaUnionFlash followed by QsaPrefillGateKernel. The no-lo path runs
// both in one kernel.
inline void LaunchQsaUnionFlashGate(QsaPrefill a, QsaUnion u, hipStream_t stream = nullptr) {
  const auto q = a.input;
  const int tiles = (q.R + kUnionRows - 1) / kUnionRows;
  const int blocks = (q.pos0 + q.R - 1) / 4 + 1;
  if (a.q_lo) {
    LaunchQsaUnionFlash(a, u, stream);
    QsaPrefillGateKernel<<<dim3(24, q.R), 256, 0, stream>>>(a);
    return;
  }
  QsaUnionBuildKernel<<<tiles, 256, blocks, stream>>>(q, u);
  QsaUnionHalfQKernel<true><<<dim3(2, tiles), 256, 0, stream>>>(a, u);
}

// LaunchQsaPrefill with the row-tile union flash in place of the per-row one.
template <class Mark>
inline void LaunchQsaPrefillUnion(QsaPrefill a, QsaUnion u, Mark mark, hipStream_t stream = nullptr, int skip = 0) {
  const auto q = a.input;
  if (q.R < 1 || q.R > 2048 || q.pos0 < 0 || q.sstride < (q.pos0 + q.R) / 4
      || !a.context || !a.q_hi || !u.list || !u.count
      || ((q.pos0 + q.R) / 4 > kQsaCandidates && !a.select_scratch)) {
    std::fprintf(stderr, "QSA prefill requires 1..2048 rows and sufficient score storage\n");
    std::abort();
  }
  // The block mask lives in LDS; past about 196K positions use the per-row flash.
  // The choice depends on the part's end only, so skipped rows do not change it.
  if (!QsaUnionFits(q.pos0, q.R)) return LaunchQsaPrefill(a, mark, stream, skip);
  if (skip < 0 || skip >= q.R || skip % 16) { std::fprintf(stderr, "QSA prefill skip must be a multiple of 16 below R\n"); std::abort(); }
  LaunchQsaPrefillCache(q, stream, true);
  QsaPrefillQueryKernel<2><<<dim3(12, q.R - skip), 256, 0, stream>>>(a, skip);
  mark("qsa_prep");
  a.input = QsaOutputRows(q, skip);
  const auto out = a.input;
  // Context is dead until the flash stage writes it, so select may keep its
  // query tiles there: QsaQueryTileFloats(R) <= R * 6144 floats.
  LaunchQsaPrefillSelect(out, a.select_scratch, a.context, stream);
  mark("qsa_select");
  // The gate runs inside the flash stage, so "qsa_flash" includes it.
  LaunchQsaUnionFlashGate(a, u, stream);
  mark("qsa_flash");
}
}  // namespace crucible::prefill
