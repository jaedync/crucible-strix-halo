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

// QsaUnionStageKernel: the arithmetic of QsaUnionHalfQKernel (same steps,
// chains, softmax, P V and gate), with the key and value loads off the
// critical path. Waves 6 and 7 load the K and V
// rows of later steps into registers (16 b128 loads per lane in flight, one
// step ahead) and write them to LDS; waves 0..5 each run one full 16-chunk QK
// chain (M tile wave / 2, key tile wave % 2) with all of q in VGPRs and K from
// LDS. The fp32 C operand of a full chain equals the half chains with the LDS
// round trip. Three barriers a step instead of five. WGP mode: the barriers
// wait for LDS only (StageBarrier), so the loads stay in flight across them.
// Map picks the lanes of the loads (StageLoad); 1 is the default, 0 is the
// plain map (bench only, same bits). Mut (bench only): 1 swaps QK chunks 0
// and 1, 2 hides key 31 of each step, 3 swaps the two P V key chunks. Each
// one must change the output bytes. Mut 4 (bench timing only, wrong output)
// skips QK, softmax and P V: the time of the loads and LDS stores alone.
constexpr int kStageKPad = 264;  // halves per staged key row: 528 B puts 16 key rows on distinct banks
constexpr int kStageKBytes = kUnionKeys * kStageKPad * 2;
constexpr int kStageListBytes = kUnionCap * (int)sizeof(int);
constexpr int kStageLoopBytes = kUnionLoopBytes + kStageKBytes;
constexpr int kStageWaves = 8, kStageLoaders = 2;

__device__ __forceinline__ void StageBarrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The key of lane `key` in step j of a tile list (QsaUnionHalfQKernel's keypos and keymask).
__device__ __forceinline__ void StageKey(const int* list, int n, int end, int j, int key, int& pos, int& mask) {
  const int idx = j * kUnionKeys + key;
  const int e = idx < n ? list[idx >> 2] : 0;
  const int p = (e & 0xffffff) * 4 + (idx & 3);
  pos = idx < n && p < end ? p : -1;
  mask = idx < n ? (e >> 24) & 0xff : 0;
}

// One half (128 dims) of the K or V row of position pos for KV group g, through a buffer resource over
// positions [0, end) (StageRsrc). An absent key (pos < 0) reads out of range, which gives zeros without
// memory traffic and without a branch, so the waitcnt pass keeps all 16 loads in flight.
typedef unsigned StageU4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ __amdgpu_buffer_rsrc_t StageRsrc(const __half* cache, int end) {
  return StreamRsrc(reinterpret_cast<const uint32_t*>(cache), (uint32_t)end * 1024);
}
// Map 0: load i of a lane reads dims [8 i, 8 i + 8) of the half row of key `lane` (pa). Map 1: load i reads key
// lo + 16 (i >> 3) (pa for i < 8, pb after) at chunk 2 (i & 7) + hi (lo = lane & 15, hi = lane >> 4), so the
// two lane halves read neighbouring 16-byte chunks of the same 16 rows: 16 cache lines per load, not 32.
template <int Map>
__device__ __forceinline__ void StageLoad(StageU4 (&r)[16], __amdgpu_buffer_rsrc_t rs, int end, int pa, int pb,
                                          int g, int half, int hi) {
  const int base = g * 512 + half * 256 + (Map == 1 ? hi * 16 : 0), out = end * 1024;
  const int va = pa >= 0 ? pa * 1024 + base : out, vb = pb >= 0 ? pb * 1024 + base : out;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    r[i] = __builtin_amdgcn_raw_buffer_load_b128(rs, Map == 0 || i < 8 ? va : vb, Map == 0 ? i * 16 : (i & 7) * 32, 0);
}

template <bool Gate = true, int Mut = 0, int Map = 1>
__global__ void __launch_bounds__(256) QsaUnionStageKernel(QsaPrefill a, QsaUnion u) {
  __shared__ __attribute__((aligned(16))) unsigned char smem[kStageLoopBytes];
  __shared__ int slist[kUnionCap];
  __shared__ float maxima[48], sums[48], scale[48];
  // Key positions and masks of steps s .. s + 2 live at once (slot s & 3).
  __shared__ int keypos[4][kUnionKeys], keymask[4][kUnionKeys], stepmask[4];
  auto score = reinterpret_cast<float(*)[kUnionKeys]>(smem);
  auto phi = reinterpret_cast<_Float16(*)[kUnionPad]>(smem + kUnionScoreBytes);
  auto vt = reinterpret_cast<_Float16(*)[kUnionPad]>(smem + kUnionScoreBytes + kUnionPhiBytes);
  auto kst = reinterpret_cast<_Float16(*)[kStageKPad]>(smem + kUnionLoopBytes);
  const QsaArgs q = a.input;
  const int tile = blockIdx.y, g = blockIdx.x, r0 = tile * kUnionRows;
  const int t = threadIdx.x, lane = t & 31, wave = __builtin_amdgcn_readfirstlane(t >> 5);
  const int rows = min(kUnionRows, q.R - r0);
  const int n = u.count[tile] * 4, steps = (n + kUnionKeys - 1) / kUnionKeys, end = q.pos0 + q.R;
  const int* list = u.list + (size_t)tile * kUnionCap;
  for (int i = t; i < n / 4; i += 256) slist[i] = list[i];
  if (t < 48) { maxima[t] = -INFINITY; sums[t] = 0.f; }
  F8 out[3][2] = {};
  // P V of step s for this wave's 32 value dims (QsaUnionHalfQKernel's code).
  auto pv = [&](int s) {
    if constexpr (Mut == 4) return;
    const int sm = stepmask[s & 3];
    const bool live0 = sm & 3, live1 = sm & 6, live2 = sm & 12;
#pragma unroll
    for (int m3 = 0; m3 < 3; ++m3) {
      if (!(m3 == 0 ? live0 : m3 == 1 ? live1 : live2)) continue;
      F8 high[2] = {};
#pragma unroll
      for (int kk = 0; kk < kUnionKeys; kk += 16) {
        const int k = Mut == 3 ? kUnionKeys - 16 - kk : kk;
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
  };
  if (wave >= kStageWaves - kStageLoaders) {
    // Loaders: wave 6 + half holds dims [128 half, 128 half + 128) of the 32 keys (StageLoad maps the lanes).
    const int half = wave - (kStageWaves - kStageLoaders), lo = lane & 15, hi = lane >> 4;
    const auto krs = StageRsrc(q.c.k, end), vrs = StageRsrc(q.c.v, end);
    auto key_of = [&](int i) { return Map == 0 ? lane : lo + 16 * (i >> 3); };
    auto chunk_of = [&](int i) { return Map == 0 ? i : 2 * (i & 7) + hi; };
    StageU4 kr[16], vr[16];
    int ka, kb;  // the StageLoad positions of kr
    auto publish = [&](int j, int pos, int mask) {
      if (half == 0) {
        keypos[j & 3][lane] = pos; keymask[j & 3][lane] = mask;
        int km = mask;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) km |= __shfl_xor(km, o);
        if (lane == 0) stepmask[j & 3] = km;
      }
    };
    // The StageLoad positions of step j; publishes key `lane` of step j.
    auto keys = [&](int j, int& pa, int& pb) {
      int ma, mb;
      StageKey(slist, n, end, j, Map == 0 ? lane : lo, pa, ma);
      if (Map == 1) StageKey(slist, n, end, j, 16 + lo, pb, mb);
      else { pb = pa; mb = ma; }
      publish(j, Map == 1 && hi ? pb : pa, Map == 1 && hi ? mb : ma);
    };
    auto store_k = [&] {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        *reinterpret_cast<StageU4*>(&kst[key_of(i)][half * 128 + chunk_of(i) * 8]) = kr[i];
    };
    auto store_v = [&] {
#pragma unroll
      for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int k = 0; k < 8; ++k)
          reinterpret_cast<unsigned short(*)[kUnionPad]>(vt)[half * 128 + chunk_of(i) * 8 + k][key_of(i)] =
              (unsigned short)(vr[i][k >> 1] >> (16 * (k & 1)));
    };
    StageBarrier();  // slist
    int a0, b0, a1, b1;
    keys(0, a0, b0);
    keys(1, a1, b1);
    StageLoad<Map>(kr, krs, end, a0, b0, g, half, hi);
    store_k();
    StageLoad<Map>(vr, vrs, end, a0, b0, g, half, hi);
    StageLoad<Map>(kr, krs, end, ka = a1, kb = b1, g, half, hi);
    StageBarrier();
    for (int s = 0; s < steps; ++s) {
      // QK phase: V(s) to LDS, V(s + 1) loads (the positions of K(s + 1)), keys of step s + 2.
      store_v();
      int a2, b2;
      keys(s + 2, a2, b2);
      StageLoad<Map>(vr, vrs, end, ka, kb, g, half, hi);
      StageBarrier();
      // Softmax phase: K(s + 1) to LDS, K(s + 2) loads.
      store_k();
      StageLoad<Map>(kr, krs, end, ka = a2, kb = b2, g, half, hi);
      StageBarrier();
      pv(s);
      StageBarrier();
    }
  } else {
    const int mt = wave >> 1, nt = wave & 1;
    const int qm = mt * 16 + (lane & 15), qrow = qm / 12, qhead = qm % 12;
    // Rows past the tile load a clamped row. Their scores are never visible.
    const size_t qi = ((size_t)(r0 + min(qrow, rows - 1)) * 24 + g * 12 + qhead) * 256;
    H16 qf[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) qf[c] = HalfFragment(a.q_hi + qi + c * 16);
    unsigned rowpack = 0;
#pragma unroll
    for (int z = 0; z < 8; ++z) rowpack |= (unsigned)((mt * 16 + FragmentRow(z)) / 12) << (4 * z);
    StageBarrier();  // slist
    StageBarrier();
    for (int s = 0; s < steps; ++s) {
      const int slot = s & 3, sm = stepmask[slot];
      const bool live0 = sm & 3, live1 = sm & 6, live2 = sm & 12;
      if (Mut != 4 && (mt == 0 ? live0 : mt == 1 ? live1 : live2)) {
        const int key = nt * 16 + (lane & 15);
        const _Float16* kp = kst[key];
        // K chunk c + 4 loads into the slot of chunk c after its WMMA: four chunks in flight from LDS.
        H16 kb[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) kb[c] = *reinterpret_cast<const H16*>(kp + c * 16);
        F8 high = {};
#pragma unroll
        for (int cc = 0; cc < 16; ++cc) {
          const int c = Mut == 1 && cc < 2 ? 1 - cc : cc;
          high = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qf[c], kb[c & 3], high);
          if (c + 4 < 16) kb[c & 3] = *reinterpret_cast<const H16*>(kp + (c + 4) * 16);
          __builtin_amdgcn_sched_barrier(0);
        }
        const int pos = keypos[slot][key], km = Mut == 2 && key == kUnionKeys - 1 ? 0 : keymask[slot][key];
        unsigned rz = rowpack;
        asm volatile("" : "+v"(rz));
#pragma unroll
        for (int z = 0; z < 8; ++z) {
          const int m = mt * 16 + FragmentRow(z), row = rz >> (4 * z) & 15;
          // QsaUnionHalfQKernel's test, without the short-circuit branches.
          const bool visible = (pos >= 0) & (row < rows) & ((km >> row & 1) != 0) & (pos <= q.pos0 + r0 + row);
          score[m][key] = visible ? high[z] * .0625f : -INFINITY;
        }
      }
      StageBarrier();
      // Online softmax (QsaUnionHalfQKernel's code): four threads per query row, eight keys each.
      if (Mut != 4 && ((t >> 6) == 0 ? live0 : (t >> 6) == 1 ? live1 : live2)) {
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
      StageBarrier();
      pv(s);
      StageBarrier();
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
  // The fused gate of QsaUnionHalfQKernel<true>, element by element.
  auto gated = reinterpret_cast<_Float16(*)[256]>(smem);
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
  StageBarrier();
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

// qsa-attn: QsaUnionStageKernel in place of QsaUnionHalfQKernel (same bits). On since its gate of Oct 31
// (token gates 4.05 and 3.05 pass, exact-extend load 6 of 6, cold prefill +2.62% at 32K and +3.86% at 130K);
// a constexpr, so no new CRUCIBLE_* switch. false gives the old kernel.
constexpr bool kQsaUnionStage = true;

// Same work as LaunchQsaFlash for one part (pos0 + R <= context).
inline void LaunchQsaUnionFlash(QsaPrefill a, QsaUnion u, hipStream_t stream = nullptr) {
  const auto q = a.input;
  const int tiles = (q.R + kUnionRows - 1) / kUnionRows;
  const int blocks = (q.pos0 + q.R - 1) / 4 + 1;
  QsaUnionBuildKernel<<<tiles, 256, blocks, stream>>>(q, u);
  if (a.q_lo) QsaUnionFlashKernel<true><<<dim3(tiles, 2), 256, 0, stream>>>(a, u);
  else if constexpr (kQsaUnionStage) QsaUnionStageKernel<false><<<dim3(2, tiles), 256, 0, stream>>>(a, u);
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
  if constexpr (kQsaUnionStage) QsaUnionStageKernel<true><<<dim3(2, tiles), 256, 0, stream>>>(a, u);
  else QsaUnionHalfQKernel<true><<<dim3(2, tiles), 256, 0, stream>>>(a, u);
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
