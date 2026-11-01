// Half path of the fused dense prefill GEMM (prefill-gemm).
//
// PrefillGemmOutKernel decodes every EXL3 tile again for each 128-row item (64 times per 8,192-row
// launch), and the decode shares the SIMD with the WMMAs. Here GemmDecodeCopyKernel decodes each
// tile once per launch into its fp16 WMMA B operand (the bytes that GemmBFragment<true> builds from
// DecodeLane), and HalfGemmOutKernel, an fp16 GEMM with A and B staged in LDS, reads that copy.
// Every output keeps its WMMA chain (k blocks in ascending order, one WMMA per 16 k values, an
// fp32 accumulator from +0) and the fused epilogue of PrefillGemmOutKernel, so the results are
// bitwise equal (bench/gemm_out_next.hip checks every byte against the EXL3 kernel).
#pragma once
#include "exl3.h"
#include "front_order.h"
#include "prefill_items.h"
#include "kernels/exl3_gemm.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace crucible {

// The copy of one matrix: [k block a][output column o][16 k values] halves (the packed layout of
// the transposed weight), 512 bytes per tile.
__host__ __device__ inline size_t HalfCopyIndex(int a, int o, int O) { return ((size_t)a * O + o) * 16; }

// One wave per tile; tile t = a * NB + b, so consecutive waves write adjacent 512-byte runs.
// FRONT: the tiles are in the front order of split S (front_order.h).
template <int K, bool FRONT>
__global__ void __launch_bounds__(256) GemmDecodeCopyKernel(const uint32_t* __restrict__ tiles,
    __half* __restrict__ copy, int A, int NB, int S) {
  __shared__ uint32_t ws[8 * kGemmWaveWords];
  constexpr int T = TileDwords(K);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const size_t t = (size_t)blockIdx.x * 8 + wave;
  if (t >= (size_t)A * NB) return;  // GemmBFragment syncs the wave only
  const int a = (int)(t / NB), b = (int)(t % NB);
  const size_t src = FRONT ? FrontTile(b, a, A, NB, S) : (size_t)b * A + a;
  const LaneWindow<K> win(lane);
  const uint32_t* p = tiles + src * T;
  const uint32_t w0 = p[win.d0], w1 = p[(win.d0 + 1) % T], w2 = K >= 5 ? p[(win.d0 + 2) % T] : 0;
  __half2 h[4];
  DecodeLane<K, kMulTrue16>(w0, w1, w2, win.shift, h);
  const GemmH16 bf = GemmBFragment<true>(h, ws + wave * kGemmWaveWords, lane);
  if (lane < 16) *reinterpret_cast<GemmH16*>(copy + HalfCopyIndex(a, b * 16 + lane, NB * 16)) = bf;
}

// One matrix of a group: K3..6, front split S (0: engine order).
inline void LaunchHalfCopy(int K, int S, const uint32_t* tiles, __half* copy, int I, int O) {
  const int A = I / 16, NB = O / 16;
  const unsigned blocks = (unsigned)(((size_t)A * NB + 7) / 8);
  auto go = [&](auto kc) {
    constexpr int KK = decltype(kc)::value;
    if (S > 0) GemmDecodeCopyKernel<KK, true><<<blocks, 256>>>(tiles, copy, A, NB, S);
    else GemmDecodeCopyKernel<KK, false><<<blocks, 256>>>(tiles, copy, A, NB, 0);
  };
  if (K == 3) go(std::integral_constant<int, 3>{});
  else if (K == 4) go(std::integral_constant<int, 4>{});
  else if (K == 5) go(std::integral_constant<int, 5>{});
  else if (K == 6) go(std::integral_constant<int, 6>{});
  else { std::fprintf(stderr, "internal error: half prefill GEMM copy with K%d\n", K); std::abort(); }
}

struct HalfGemmJob {
  const __half* x;  // packed input [I/16][Xrows][16]
  const __half* w;  // decode copy [I/16][O][16]
  int I, O, M, Xrows;
};

// LDS plane: 128 rows (or columns) x 16 bytes (8 k values), plus a pad that moves the second plane
// of a k block to other banks.
constexpr int kHalfPlane = 128 * 16 + 64;    // bytes
constexpr int kHalfKBlock = 4 * kHalfPlane;  // A low, A high, B low, B high
constexpr int kHalfStagePad = 128 + 4;       // floats per staged output row
// k blocks per LDS stage. One beats two (bench/gemm_out_next.hip, 8,192 rows: GDN in 17.0 against
// 18.5 ms): the smaller LDS stages fit more blocks on a CU.
constexpr int kHalfKC = 1;
// Row blocks per item group (HalfItems): 16 for the wide input groups, 8 for the 2,560-column
// output projections (bench: out 7.5 ms with 8, 8.2 ms with 16).
inline int HalfGroupFor(int columns) { return columns > 4096 ? 16 : 8; }
// Smaller launches keep the EXL3 kernels: the copy costs a fixed decode pass per launch.
constexpr int kHalfMinRows = 2048;

// Main loop and epilogue options (prefill-dense; bench
// bench/prefill_dense_gemm.hip, 8,192 rows at the 100 W limit, against the trunk kernel: GDN input
// 19.2 to 18.2 ms, QSA input 15.8 to 15.2 ms, output projection 7.6 to 7.2 ms; every byte equal).
// kHalfDepth: register sets of the global prefetch. With 2, the loads of stage c + 2 go out in stage
// c, so they have two stages of WMMAs to arrive before their LDS stores (240 VGPRs, 6 waves per SIMD).
// kHalfEpi 0: the epilogue of PrefillGemmOutKernel as written; 1: the same with DPP lane moves for the
// output transform (Wht128Dpp, the same arithmetic and bits).
// kHalfPut: where the LDS stores of the next stage go in a stage (HalfGemmOutBody, PP; 1 is 0.5 to
// 1% faster than 0).
// Slower, not kept: a third LDS stage (15 to 20%: fewer blocks fit on a WGP) and 256-row blocks of 8
// waves (2 to 4%).
constexpr int kHalfDepth = 2, kHalfEpi = 1, kHalfPut = 1;

// 128 rows x 128 columns per block, 4 waves in a 2 x 2 grid; wave (wr, wc) owns rows m0 + 64 wr
// (4 row tiles) and columns b0 * 16 + 64 wc (4 column tiles). One k block (16 k values) per LDS
// stage, two stages, one barrier per stage. Every output keeps its WMMA chain: k blocks in
// ascending order, one WMMA per 16 k values, an fp32 accumulator from +0; DG and EP change only
// when the loads run and how the lanes move data. EPI (at most one): as PrefillGemmOutKernel's.
template <int DG, int EP, int PP, class... EPI>
__device__ __forceinline__ void HalfGemmOutBody(const LinJob* __restrict__ lin,
    const HalfGemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n, EPI... epi) {
  static_assert(sizeof...(EPI) <= 1);
  static_assert((DG == 1 || DG == 2) && (EP == 0 || EP == 1) && (PP == 0 || PP == 1));
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  constexpr int kStage = kHalfKBlock;
  static_assert(2 * kStage >= 2 * 16 * kHalfStagePad * 4, "the epilogue stage reuses the GEMM stages");
  __shared__ __attribute__((aligned(16))) char lds[2 * kStage];
  if ((int)blockIdx.x >= n) return;
  const Exl3GemmItem it = items[blockIdx.x];
  const HalfGemmJob j = jobs[it.job];
  const LinJob& L = lin[it.job];
  const int tid = threadIdx.x, lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5), wr = wave >> 1, wc = wave & 1;
  const int A = j.I / 16, c0 = it.b0 * 16;
  // Global loads: per k block, the A run (128 rows x 32 B) and the B run (128 columns x 32 B) are
  // 256 chunks of 16 bytes each; thread t moves chunks t and t + 128 of both.
  const uint32_t xbytes = (uint32_t)j.Xrows * j.I * 2;
  const auto xrs = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), xbytes);
  const auto wrs = StreamRsrc(reinterpret_cast<const uint32_t*>(j.w), (uint32_t)((size_t)j.O * j.I * 2));
  uint32_t xo[2], wo[2], lo[2];
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int chunk = tid + q * 128, row = chunk >> 1, half = chunk & 1;
    xo[q] = it.m0 + row < j.M ? (uint32_t)(it.m0 + row) * 32 + half * 16 : xbytes;
    wo[q] = (uint32_t)(c0 + row) * 32 + half * 16;
    lo[q] = half * kHalfPlane + row * 16;
  }
  const uint32_t astep = (uint32_t)j.Xrows * 32, bstep = (uint32_t)j.O * 32;
  U4 g[DG][4];
  // Stage t's k block goes through register set t % DG (S below, a constant: a register array takes
  // only constant indices). Past the last k block the raw buffer loads return 0 (the range check
  // covers the k offset in the lane offset), and those stages are never read.
  auto fetch = [&](int t, auto S) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      g[S][q] = __builtin_amdgcn_raw_buffer_load_b128(xrs, xo[q] + t * astep, 0, 0);
      g[S][2 + q] = __builtin_amdgcn_raw_buffer_load_b128(wrs, wo[q] + t * bstep, 0, 0);
    }
  };
  auto put = [&](int t, auto S) {
    char* base = lds + (t & 1) * kStage;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      *reinterpret_cast<U4*>(base + lo[q]) = g[S][q];
      *reinterpret_cast<U4*>(base + 2 * kHalfPlane + lo[q]) = g[S][2 + q];
    }
  };
  using S0 = std::integral_constant<int, 0>;
  using S1 = std::integral_constant<int, DG - 1>;
  GemmF8 acc[4][4];
#pragma unroll
  for (int n4 = 0; n4 < 4; ++n4)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[n4][r] = GemmF8{};
  const int ar = wr * 64 + (lane & 15), bc = wc * 64 + (lane & 15);
  // Prologue: stage 0 in LDS, stage 1 in flight with DG 2.
  fetch(0, S0{});
  put(0, S0{});
  if constexpr (DG == 2) fetch(1, S1{});
  GemmGroupSync();
  // Stage c; S = c % DG.
  auto step = [&](int c, auto S) {
    constexpr int o = DG == 2 ? 1 - decltype(S)::value : 0;
    fetch(c + DG, S);  // set (c + DG) % DG = S
    const char* base = lds + (c & 1) * kStage;
    GemmH16 af[4], bf[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      auto* p = reinterpret_cast<U4*>(&af[r]);
      p[0] = *reinterpret_cast<const U4*>(base + (ar + r * 16) * 16);
      p[1] = *reinterpret_cast<const U4*>(base + kHalfPlane + (ar + r * 16) * 16);
    }
#pragma unroll
    for (int n4 = 0; n4 < 4; ++n4) {
      auto* p = reinterpret_cast<U4*>(&bf[n4]);
      p[0] = *reinterpret_cast<const U4*>(base + 2 * kHalfPlane + (bc + n4 * 16) * 16);
      p[1] = *reinterpret_cast<const U4*>(base + 3 * kHalfPlane + (bc + n4 * 16) * 16);
    }
    // PP 0: the stores of stage c + 1 go after every WMMA of stage c (they wait for the prefetch,
    // vmcnt). PP 1: they go after the first 8 WMMAs, so they finish before the barrier. Stage c + 1's
    // LDS stage is the stage of c - 1, which every wave left before the last barrier.
#pragma unroll
    for (int n4 = 0; n4 < 4; ++n4) {
      if (PP == 1 && n4 == 2) {
        __builtin_amdgcn_sched_barrier(0);
        put(c + 1, std::integral_constant<int, o>{});  // set (c + 1) % DG
        __builtin_amdgcn_sched_barrier(0);
      }
#pragma unroll
      for (int r = 0; r < 4; ++r)
        acc[n4][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[r], bf[n4], acc[n4][r]);
    }
    // One k block of fragments live at a time (the next block's LDS reads would need 64 more
    // VGPRs next to the 128 accumulators and the global prefetch).
    asm volatile("" ::: "memory");
    if constexpr (PP == 0) {
      __builtin_amdgcn_sched_barrier(0);
      put(c + 1, std::integral_constant<int, o>{});  // set (c + 1) % DG
    }
    GemmGroupSync();  // LDS only: __syncthreads would also invalidate the L0 cache each stage
  };
  for (int c = 0; c < A; c += DG) {
    step(c, S0{});
    if constexpr (DG == 2) if (c + 1 < A) step(c + 1, S1{});
  }
  __syncthreads();
  // Epilogue of PrefillGemmOutKernel on whole 128-column rows: row tile rt of both row halves goes
  // to two 16-row stages, then each wave transforms 8 of the 32 staged rows.
  float* stage = reinterpret_cast<float*>(lds);
  const int c = c0 + lane * 4;
  float svh[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) svh[i] = __half2float(L.svh[c + i]);
#pragma unroll
  for (int rt = 0; rt < 4; ++rt) {
    if (rt > 0) __syncthreads();  // the readers of the last stage are done
#pragma unroll
    for (int n4 = 0; n4 < 4; ++n4)
#pragma unroll
      for (int z = 0; z < 8; ++z)
        stage[(wr * 16 + 2 * z + (lane >> 4)) * kHalfStagePad + wc * 64 + n4 * 16 + (lane & 15)] = acc[n4][rt][z];
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 8; ++h) {
      const int lr = wave * 8 + h, row = it.m0 + (lr >> 4) * 64 + rt * 16 + (lr & 15);
      if (row >= j.M) continue;  // wave-uniform: Wht128Dpp needs full waves
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = 0.f + stage[lr * kHalfStagePad + lane * 4 + i];
      if constexpr (EP == 1) Wht128Dpp(v);
      else Wht128(v);
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] *= kHadNorm * svh[i];
      if (L.epi == kEpiStore) {
        float* d = L.dst + (size_t)row * L.ldd + c;
#pragma unroll
        for (int i = 0; i < 4; ++i) d[i] = v[i];
      } else {
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
          const float gs = L.aux[row * kStreams + s];
          float* d = L.dst + (size_t)row * (L.ldd ? L.ldd : kWide) + s * kHidden + c;
#pragma unroll
          for (int i = 0; i < 4; ++i) d[i] += gs * v[i];
        }
      }
    }
  }
  if constexpr (sizeof...(EPI) == 1) {
    __syncthreads();  // the block's stores are visible to the whole block
    (epi.Run(L.dst, L.ldd, it.m0, min(it.m0 + 128, j.M), c0), ...);
  }
}

// The engine kernel (its name is in tools/devsplit_groups.tsv). KC: k blocks per LDS stage, only 1
// (two are slower: the larger stages fit fewer blocks on a CU).
template <int KC, class... EPI>
__global__ void __launch_bounds__(128) HalfGemmOutKernel(const LinJob* __restrict__ lin,
    const HalfGemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n, EPI... epi) {
  static_assert(KC == 1);
  HalfGemmOutBody<kHalfDepth, kHalfEpi, kHalfPut>(lin, jobs, items, n, epi...);
}

inline std::vector<Exl3GemmItem> HalfItems(const std::vector<HalfGemmJob>& jobs, int G) {
  std::vector<int> M, O;
  for (const auto& j : jobs) { M.push_back(j.M); O.push_back(j.O); }
  std::vector<Exl3GemmItem> v;
  for (const auto& it : PrefillHalfItemsOf(M.data(), O.data(), (int)jobs.size(), G)) v.push_back({it.job, 0, it.m0, it.b0});
  return v;
}

}  // namespace crucible
