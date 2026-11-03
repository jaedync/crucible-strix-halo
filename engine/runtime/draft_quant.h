// Low-bit draft head (CRUCIBLE_DRAFT_HEAD=q4 or q2). Drafts only: the trunk verifies
// every proposed token with its full EXL3 head, so this head changes the drafts and the speed, never the output.
//
// Per scored row, three kernels after the head's input transform (LinInKernel of the block head group):
//   DraftQGemvKernel     coarse logits of every retained token from a 4- or 2-bit copy of the head rows
//                        (draft_quant_host.h format), and the maximum of each 128-token block;
//   DraftQRescoreKernel  the kDraftQBlocks best blocks by coarse maximum, recomputed from the original EXL3 tiles
//                        (LinGemvBody of exl3.h on the engine-order block head, kDraftQSplits K splits);
//   DraftQChoiceKernel   the output transform of those blocks (LinOutLane), their logits into `logits` (so the sampled
//                        drafts read exact values for the best blocks), and the choice (DraftChoiceKernel's rule)
//                        over the recomputed logits only.
#pragma once

#include <algorithm>
#include <thread>
#include <vector>

#include "../exl3/exl3.h"
#include "../exl3/dense_rows.h"
#include "draft_ops.h"
#include "draft_quant_host.h"

namespace crucible {

inline constexpr int kDraftQMaxRows = 8;    // scored rows of one pass (kMaxRows)
inline constexpr int kDraftQMaxBlocks = 2048;  // blocks of 128 tokens in a head (kVocab / 128 = 1940)

// Coarse logits: one workgroup per 128-token block, 16 rows per wave, P rows per wave step so that the 32 lanes read
// 5 uint4 each (the next step's loads are in flight during the dots). Each weight pair is unpacked once and used for
// all NR input rows (NR = n rounded up to 1, 2, 4 or 8; the rows past n are zero and not stored). xh: n rows of the
// transformed input (fp16). logits: [n][V]; maxima: [n][V / 128]. The strided row format (draft_quant_host.h) keeps
// the LDS reads of x conflict-free: lane c reads half2 word q * L + c.
template <int BITS, int NR>
__global__ void __launch_bounds__(256) DraftQGemvKernel(const uint4* __restrict__ words, const __half* __restrict__ scales,
                                                        const __half* __restrict__ xh, int n, int V,
                                                        float* __restrict__ logits, float* __restrict__ maxima) {
  constexpr int L = DraftQLoads(kHidden, BITS);  // uint4 per row
  constexpr int P = 160 / L, PPW = 16 / BITS, NS = 16 / P, H = kHidden / 2;
  static_assert(P * L == 160 && 16 % P == 0 && NR <= kDraftQMaxRows);
  constexpr uint32_t kMask = ((1u << BITS) - 1) * 0x10001u;
  extern __shared__ uint32_t xs_raw[];  // [NR][kHidden / 2] half2 (the engine's one dynamic LDS name and type)
  const __half2* xs = reinterpret_cast<const __half2*>(xs_raw);
  __shared__ float wave_max[8][kDraftQMaxRows];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  for (int i = threadIdx.x; i < NR * H; i += 256) xs_raw[i] = i < n * H ? reinterpret_cast<const uint32_t*>(xh)[i] : 0u;
  __syncthreads();
  const __half2 bias = __float2half2_rn(1024.f), center = __float2half2_rn(DraftQCenter(BITS));
  float best[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) best[r] = -INFINITY;
  const int base = blockIdx.x * 128 + wave * 16;
  uint4 raw[5];
  float scale[5];
  auto load = [&](int step, uint4 (&w)[5], float (&sc)[5]) {
    const int row0 = base + step * P;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      const int c = lane + 32 * j;
      w[j] = LoadNT(words + (size_t)row0 * L + c);
      sc[j] = __half2float(scales[(size_t)row0 * L + c]);
    }
  };
  load(0, raw, scale);
#pragma unroll 1
  for (int step = 0; step < NS; ++step) {
    // One row: the next step's loads fly during the dots. More rows: enough ALU work per load, and the registers of
    // the second set would spill.
    constexpr bool kAhead = NR == 1;
    uint4 next[5];
    float next_scale[5];
    if (kAhead && step + 1 < NS) load(step + 1, next, next_scale);
    const int row0 = base + step * P;
    float acc[NR][P];
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
      for (int q = 0; q < P; ++q) acc[r][q] = 0.f;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
      const int c = lane + 32 * j, p = c / L, k = c % L;
      const uint32_t w4[4] = {raw[j].x, raw[j].y, raw[j].z, raw[j].w};
      float part[NR];
#pragma unroll
      for (int r = 0; r < NR; ++r) part[r] = 0.f;
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
        for (int jj = 0; jj < PPW; ++jj) {
          uint32_t h = ((w4[wi] >> (jj * BITS)) & kMask) | 0x64006400u;
          const __half2 v = __hsub2(__hsub2(*reinterpret_cast<__half2*>(&h), bias), center);
#pragma unroll
          for (int r = 0; r < NR; ++r) part[r] = Dot2(v, xs[r * H + (wi * PPW + jj) * L + k], part[r]);
        }
        // Keeps the x reads of one word together: without it the scheduler hoists all of them and spills (NR > 1).
        __builtin_amdgcn_sched_barrier(0);
      }
#pragma unroll
      for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int q = 0; q < P; ++q) acc[r][q] += p == q ? part[r] * scale[j] : 0.f;
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      if (r >= n) break;
#pragma unroll
      for (int q = 0; q < P; ++q) {
        const float v = WaveSum(acc[r][q]);
        if (lane == 0) logits[(size_t)r * V + row0 + q] = v;
        best[r] = fmaxf(best[r], v);
      }
    }
    if (step + 1 < NS) {
      if constexpr (kAhead) {
#pragma unroll
        for (int j = 0; j < 5; ++j) raw[j] = next[j], scale[j] = next_scale[j];
      } else {
        load(step + 1, raw, scale);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < NR; ++r)
      if (r < n) wave_max[wave][r] = best[r];
  }
  __syncthreads();
  if (threadIdx.x < n) {
    float m = -INFINITY;
    for (int w = 0; w < 8; ++w) m = fmaxf(m, wave_max[w][threadIdx.x]);
    maxima[(size_t)threadIdx.x * (V / 128) + blockIdx.x] = m;
  }
}

// Grid (kDraftQSplits, kDraftQBlocks, n): split s of the slot-th best block of row r, from the engine-order block
// head (`base`: tiles, I, K; xh = row 0). Writes partial[((r * kDraftQBlocks + slot) * kDraftQSplits + s) * 128 + c]
// and chosen[r * kDraftQBlocks + slot] (the head block index, -1 when the head has fewer blocks).
template <int K>
__global__ void __launch_bounds__(256) DraftQRescoreKernel(LinJob base, const float* __restrict__ maxima, int blocks,
                                                           int* __restrict__ chosen, float* __restrict__ partial) {
  __shared__ float m[kDraftQMaxBlocks];
  __shared__ unsigned char taken[kDraftQMaxBlocks];
  __shared__ int pick;
  __shared__ uint32_t xs[(kHidden / 16 / kDraftQSplits) * 8];
  const int s = blockIdx.x, slot = blockIdx.y, r = blockIdx.z;
  for (int i = threadIdx.x; i < blocks; i += blockDim.x) m[i] = maxima[(size_t)r * blocks + i], taken[i] = 0;
  __syncthreads();
  // Wave 0 takes the best remaining block slot + 1 times (DraftQBefore order, so the pick has DraftQRank == slot):
  // per round a strided scan per lane and a wave reduction; far cheaper than a rank count per block.
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    int found = -1;
    for (int round = 0; round <= slot; ++round) {
      float bv = -INFINITY;
      int bi = -1;
      for (int j = lane; j < blocks; j += 32)
        if (!taken[j] && (bi < 0 || DraftQBefore(m[j], j, bv, bi))) bv = m[j], bi = j;
#pragma unroll
      for (int k = 1; k < 32; k <<= 1) {
        const float ov = __shfl_xor(bv, k);
        const int oi = __shfl_xor(bi, k);
        if (oi >= 0 && (bi < 0 || DraftQBefore(ov, oi, bv, bi))) bv = ov, bi = oi;
      }
      found = bi;
      if (bi < 0) break;
      if (lane == 0) taken[bi] = 1;
      __builtin_amdgcn_wave_barrier();
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    }
    if (lane == 0) pick = found;
  }
  __syncthreads();
  const int block = pick;
  if (s == 0 && threadIdx.x == 0) chosen[r * kDraftQBlocks + slot] = block;
  if (block < 0) return;
  LinJob job = base;
  job.tiles = base.tiles + (size_t)block * 128 * base.I * K / 32;
  job.xh = base.xh + (size_t)r * base.I;
  job.partial = partial + (size_t)(r * kDraftQBlocks + slot) * kDraftQSplits * 128;
  job.O = 128;
  job.S = kDraftQSplits;
  job.R = 1;
  job.cache_policy = 0;
  LinGemvBody<K, 1>(job, GemvItem{0, s, 0, 8}, xs, false);
}

// One workgroup per row r, one wave per rescored block: LinOutBlock's arithmetic (fixed split order, Wht128,
// kHadNorm * svh), the logits into logits[r * V + block * 128 + c], and DraftChoiceKernel's top-2 rule over the
// recomputed logits. svh: the block head's (head block order); ids: vocabulary block ID of each head block.
__global__ void __launch_bounds__(256) DraftQChoiceKernel(const float* __restrict__ partial, const int* __restrict__ chosen,
                                                          const __half* __restrict__ svh, const int* __restrict__ ids,
                                                          int V, float* __restrict__ logits, DraftChoice* out) {
  static_assert(kDraftQBlocks == 8);
  __shared__ float best[8], second[8];
  __shared__ int tokens[8];
  const int r = blockIdx.x, lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int block = chosen[r * kDraftQBlocks + wave];
  float b = -INFINITY, s = -INFINITY;
  int id = 0;
  if (block >= 0) {
    const float* p = partial + (size_t)(r * kDraftQBlocks + wave) * kDraftQSplits * 128 + lane * 4;
    float sum[4], v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) sum[i] = 0.f;
    for (int k = 0; k < kDraftQSplits; ++k)
#pragma unroll
      for (int i = 0; i < 4; ++i) sum[i] += p[k * 128 + i];
    __half scale[4];
    LinScaleLane(svh, block * 128 + lane * 4, scale);
    LinOutLane(sum, scale, v);  // LinOutBlock's split sum, Wht128 and kHadNorm * svh (0 + sum adds nothing)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = block * 128 + lane * 4 + i;
      logits[(size_t)r * V + c] = v[i];
      const int token = ids[block] * 128 + lane * 4 + i;
      if (v[i] > b || (v[i] == b && token < id)) {
        s = b;
        b = v[i];
        id = token;
      } else {
        s = fmaxf(s, v[i]);
      }
    }
  }
#pragma unroll
  for (int k = 1; k < 32; k <<= 1) {
    const float ob = __shfl_xor(b, k), os = __shfl_xor(s, k);
    const int oi = __shfl_xor(id, k);
    if (ob > b || (ob == b && oi < id)) {
      s = fmaxf(b, os);
      b = ob;
      id = oi;
    } else {
      s = fmaxf(s, ob);
    }
  }
  if (lane == 0) {
    best[wave] = b;
    second[wave] = s;
    tokens[wave] = id;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    b = best[0];
    s = second[0];
    id = tokens[0];
    for (int w = 1; w < 8; ++w) {
      if (best[w] > b || (best[w] == b && tokens[w] < id)) {
        s = fmaxf(b, second[w]);
        b = best[w];
        id = tokens[w];
      } else {
        s = fmaxf(s, best[w]);
      }
    }
    out[r] = {id, b - s};
  }
}

// The low-bit copy of a block draft head and its scratch. Built once at load from the full EXL3 head (`head`, any
// tile order); `weights` is the engine-order block head (BlockDraftHead::Weights()) that the rescoring reads.
class DraftQuantHead {
 public:
  template <typename Linear>
  DraftQuantHead(const Linear& head, const Linear& weights, const std::vector<int>& blocks, int bits)
      : bits_(bits), blocks_((int)blocks.size()), V_((int)blocks.size() * 128), K_(weights.K),
        tiles_(weights.tiles), svh_(weights.svh) {
    if ((bits != 4 && bits != 2) || head.I != kHidden || weights.I != kHidden || (K_ != 5 && K_ != 6) ||
        head.K != K_ || blocks_ < 1 || blocks_ > kDraftQMaxBlocks || weights.O != V_ || weights.front_s) {
      std::fprintf(stderr, "low-bit draft head needs a 2560-input K5 or K6 block head of 1..%d blocks\n",
                   kDraftQMaxBlocks);
      std::exit(2);
    }
    const int wpr = DraftQWords(kHidden, bits), ng = DraftQScales(kHidden, bits);
    std::vector<uint32_t> words((size_t)V_ * wpr);
    std::vector<_Float16> scales((size_t)V_ * ng);
    constexpr int kChunk = 32;  // blocks per decode pass: 21 MB of fp16 rows
    __half* dense;
    int* ids;
    HIP_OK(hipMalloc(&dense, (size_t)kChunk * 128 * kHidden * sizeof(__half)));
    HIP_OK(hipMalloc(&ids, kChunk * sizeof(int)));
    std::vector<_Float16> host((size_t)kChunk * 128 * kHidden);
    const int threads = std::clamp((int)std::thread::hardware_concurrency(), 1, 16);
    for (int c0 = 0; c0 < blocks_; c0 += kChunk) {
      const int count = std::min(kChunk, blocks_ - c0);
      HIP_OK(hipMemcpy(ids, blocks.data() + c0, count * sizeof(int), hipMemcpyHostToDevice));
      if (K_ == 6)
        Exl3DenseRowsKernel<6><<<dim3(kHidden / 16, count), 256>>>(head.tiles, head.svh, ids, kHidden, head.O,
                                                                  head.front_s, dense);
      else
        Exl3DenseRowsKernel<5><<<dim3(kHidden / 16, count), 256>>>(head.tiles, head.svh, ids, kHidden, head.O,
                                                                  head.front_s, dense);
      HIP_OK(hipGetLastError());
      HIP_OK(hipMemcpy(host.data(), dense, (size_t)count * 128 * kHidden * sizeof(__half), hipMemcpyDeviceToHost));
      const int rows = count * 128;
      std::vector<std::thread> pool;
      for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
          std::vector<float> w(kHidden);
          for (int i = t; i < rows; i += threads) {
            for (int e = 0; e < kHidden; ++e) w[e] = float(host[(size_t)i * kHidden + e]);
            const size_t row = (size_t)c0 * 128 + i;
            DraftQuantRow(w.data(), kHidden, bits, words.data() + row * wpr, scales.data() + row * ng);
          }
        });
      for (auto& thread : pool) thread.join();
    }
    HIP_OK(hipFree(dense));
    HIP_OK(hipFree(ids));
    HIP_OK(hipMalloc(&words_, words.size() * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&scales_, scales.size() * sizeof(__half)));
    HIP_OK(hipMemcpy(words_, words.data(), words.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(scales_, scales.data(), scales.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&maxima_, (size_t)kDraftQMaxRows * blocks_ * sizeof(float)));
    HIP_OK(hipMalloc(&chosen_, (size_t)kDraftQMaxRows * kDraftQBlocks * sizeof(int)));
    HIP_OK(hipMalloc(&partial_, (size_t)kDraftQMaxRows * kDraftQBlocks * kDraftQSplits * 128 * sizeof(float)));
  }
  ~DraftQuantHead() {
    for (void* p : {(void*)words_, (void*)scales_, (void*)maxima_, (void*)chosen_, (void*)partial_})
      if (p) HIP_OK(hipFree(p));
  }
  DraftQuantHead(const DraftQuantHead&) = delete;
  DraftQuantHead& operator=(const DraftQuantHead&) = delete;

  // Scores n rows (1..kDraftQMaxRows) of the transformed input xh ([n][kHidden] fp16, the block head group's xh):
  // logits [n][Vocabulary()] (coarse, the rescored blocks exact) and one choice per row. ids: draft_blocks_.
  void Score(const __half* xh, int n, float* logits, const int* ids, DraftChoice* choice) const {
    if (n < 1 || n > kDraftQMaxRows) std::abort();
    Coarse(xh, n, logits);
    Rescore(xh, n);
    Choose(n, logits, ids, choice);
  }
  // The three passes of Score, public for the stage times of tests/draft_quant_test.hip.
  void Rescore(const __half* xh, int n) const {
    LinJob base{};
    base.tiles = tiles_;
    base.xh = const_cast<__half*>(xh);
    base.I = kHidden;
    base.K = K_;
    const dim3 grid(kDraftQSplits, kDraftQBlocks, n);
    if (K_ == 6) DraftQRescoreKernel<6><<<grid, 256>>>(base, maxima_, blocks_, chosen_, partial_);
    else DraftQRescoreKernel<5><<<grid, 256>>>(base, maxima_, blocks_, chosen_, partial_);
  }
  void Choose(int n, float* logits, const int* ids, DraftChoice* choice) const {
    DraftQChoiceKernel<<<n, 256>>>(partial_, chosen_, svh_, ids, V_, logits, choice);
  }
  // The coarse pass alone (DraftQGemvKernel): logits and block maxima of n rows.
  void Coarse(const __half* xh, int n, float* logits) const {
    if (bits_ == 4) Launch<4>(xh, n, logits);
    else Launch<2>(xh, n, logits);
  }
  int Vocabulary() const { return V_; }
  int Bits() const { return bits_; }
  // The last Score's rescored head blocks [n][kDraftQBlocks] and coarse block maxima [n][blocks] (tests).
  const int* Chosen() const { return chosen_; }
  const float* Maxima() const { return maxima_; }
  // Device bytes of the low-bit copy (words and scales), as the load line prints them.
  size_t Bytes() const {
    return (size_t)V_ * (DraftQWords(kHidden, bits_) * sizeof(uint32_t) + DraftQScales(kHidden, bits_) * sizeof(_Float16));
  }

 private:
  template <int BITS>
  void Launch(const __half* xh, int n, float* logits) const {
    const auto* w = reinterpret_cast<const uint4*>(words_);
    const int rows = n <= 1 ? 1 : n <= 2 ? 2 : n <= 4 ? 4 : 8;
    const size_t lds = (size_t)rows * kHidden * sizeof(__half);
    if (rows == 1) DraftQGemvKernel<BITS, 1><<<blocks_, 256, lds>>>(w, scales_, xh, n, V_, logits, maxima_);
    else if (rows == 2) DraftQGemvKernel<BITS, 2><<<blocks_, 256, lds>>>(w, scales_, xh, n, V_, logits, maxima_);
    else if (rows == 4) DraftQGemvKernel<BITS, 4><<<blocks_, 256, lds>>>(w, scales_, xh, n, V_, logits, maxima_);
    else DraftQGemvKernel<BITS, 8><<<blocks_, 256, lds>>>(w, scales_, xh, n, V_, logits, maxima_);
  }

  int bits_, blocks_, V_, K_;
  const uint32_t* tiles_;
  const __half* svh_;
  uint32_t* words_ = nullptr;
  __half* scales_ = nullptr;
  float* maxima_ = nullptr;
  int* chosen_ = nullptr;
  float* partial_ = nullptr;
};

}  // namespace crucible
