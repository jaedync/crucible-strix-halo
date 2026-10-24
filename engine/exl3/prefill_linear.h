// Batched EXL3 plans share the engine's transforms and epilogues with decode.
#pragma once
#include "exl3.h"
#include "front_order.h"
#include "prefill_items.h"
#include "kernels/exl3_gemm.h"
#include "../switches.h"
#include <algorithm>
#include <type_traits>
#include <vector>

namespace crucible {
// Widest EXL3 K the prefill GEMM bodies handle. K6 (the 4.05 checkpoint) uses
// separate kernel instantiations, so K3..5 keep their register budgets.
constexpr int kPrefillMaxK = 6;
// K7 and K8 (the h6_ng8 packs) run in groups of one width through the fused dense kernel
// (PrefillGemmOutWideKernel) and the KMAX 7 and 8 MoE kernels (prefill_moe.h).
inline bool PrefillWidthSupported(int K) { return (K >= 3 && K <= kPrefillMaxK) || K == 7 || K == 8; }
// Row tiles per GEMM item: 2, 4, 8 or 16; other values fall back to 2.
inline int ParseRows(int n) { return n == 4 || n == 8 || n == 16 ? n : 2; }
template<class T> inline T* PrefillUpload(const std::vector<T>& values) {
  if (values.empty()) return nullptr;
  T* dst;
  HIP_OK(hipMalloc(&dst, values.size()*sizeof(T)));
  HIP_OK(hipMemcpy(dst, values.data(), values.size()*sizeof(T), hipMemcpyHostToDevice));
  return dst;
}

// Packed GEMM input [I/16][R][16]: one k block of all rows is contiguous, so a
// wave's A fragment is one 512-byte read instead of 16 row-strided reads.
__device__ __forceinline__ size_t PackedIndex(int r, int c, int R) {
  return ((size_t)(c >> 4) * R + r) * 16 + (c & 15);
}
// Input transform item w. Without a list (items == nullptr) every job has the
// same R and I, and item w is the w-th of the list that Build would write:
// jobs in order, rows in order, then 128-column blocks. A list costs 12 bytes a
// row and block (about 220 MiB in a 4096-row plan, linear in R).
__host__ __device__ inline XfItem PrefillInItem(const LinJob* jobs, const XfItem* items, int w) {
  if (items) return items[w];
  const PrefillInIndex i = PrefillInItemAt(jobs[0].R, jobs[0].I, w);
  return {i.job, i.r, i.blk};
}
__global__ void __launch_bounds__(256) PrefillLinInPackedKernel(const LinJob* __restrict__ jobs,
    const XfItem* __restrict__ items, int n) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (w >= n) return;
  const int lane = threadIdx.x & 31;
  const XfItem it = PrefillInItem(jobs, items, w);
  const LinJob& j = jobs[it.job];
  const int c0 = it.blk * 128 + lane * 4;
  const float* s = j.src + (size_t)it.r * j.lds + c0;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = RoundH(RoundH(s[i]) * __half2float(j.suh[c0 + i]));
  Wht128(v);
  __half* d = j.xh + PackedIndex(it.r, c0, j.R);
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = __float2half(v[i] * kHadNorm);
}
// The HC mix already rounds x to fp16. Reading that half copy removes the
// wider store and reload without changing a transform's rounding boundary.
template<bool PACKED>
__global__ void __launch_bounds__(256) PrefillLinInHalfKernel(const LinJob* __restrict__ jobs,
    const XfItem* __restrict__ items, int n, const __half* __restrict__ source) {
  const int w=blockIdx.x*8+(threadIdx.x>>5);
  if (w>=n) return;
  const XfItem it=PrefillInItem(jobs,items,w); const LinJob& j=jobs[it.job];
  const int c=it.blk*128+(threadIdx.x&31)*4;
  const __half* src=source+(size_t)it.r*j.lds+c;
  float v[4];
#pragma unroll
  for (int i=0;i<4;++i) v[i]=RoundH(__half2float(src[i])*__half2float(j.suh[c+i]));
  Wht128(v);
  __half* dst=j.xh+(PACKED ? PackedIndex(it.r,c,j.R) : (size_t)it.r*j.I+c);
#pragma unroll
  for (int i=0;i<4;++i) dst[i]=__float2half(v[i]*kHadNorm);
}

// Repacks a row-major [R][I] fp16 input that another kernel wrote (I % 128
// == 0). A wave moves 4 rows x 8 column groups: it reads 256-byte runs of
// each row and writes whole 128-byte lines of the packed layout. The old
// row-strided order ran at about 10 GB/s at R4096.
__host__ __device__ inline int PackRowsThreads(int R, int I) { return (R + 3) / 4 * (I / 128) * 32; }
__global__ void __launch_bounds__(256) PrefillPackRowsKernel(const __half* src, __half* dst,
                                                            int R, int I) {
  const int g = blockIdx.x * 256 + threadIdx.x;
  if (g >= PackRowsThreads(R, I)) return;
  const int run = g >> 5, groups = I / 128;
  const int r = run / groups * 4 + (g & 3), a = run % groups * 8 + (g >> 2 & 7);
  if (r >= R) return;
  const uint4* p = reinterpret_cast<const uint4*>(src + (size_t)r * I + a * 16);
  uint4* q = reinterpret_cast<uint4*>(dst + ((size_t)a * R + r) * 16);
  q[0] = p[0]; q[1] = p[1];
}

// GEMM item shape: row tiles per item and the activation layout.
struct PrefillGemmShape {
  int rm = 2;
  bool packed = false;
};
inline void LaunchPrefillGemm(PrefillGemmShape g, const Exl3GemmJob* jobs,
                              const Exl3GemmItem* items, int count) {
  // exl3-gemm's measured dense choice: packed, register A order, no skips.
  if (g.packed) {
    if (g.rm == 16) Exl3GemmKernel<16,true,false,1,true,true,false,false><<<count,128>>>(jobs,items,count);
    else if (g.rm == 8) Exl3GemmKernel<8,true,false,1,true,true,false,false><<<count,128>>>(jobs,items,count);
    else if (g.rm == 4) Exl3GemmKernel<4,true,false,1,true,true,false,false><<<count,128>>>(jobs,items,count);
    else Exl3GemmKernel<2,true,false,1,true,true,false,false><<<count,128>>>(jobs,items,count);
    return;
  }
  if (g.rm == 16) Exl3GemmKernel<16,true,false,1,false,false,true><<<count,128>>>(jobs,items,count);
  else if (g.rm == 8) Exl3GemmKernel<8,true,false,1,false,false,true><<<count,128>>>(jobs,items,count);
  else if (g.rm == 4) Exl3GemmKernel<4,true,false,1,false,false,true><<<count,128>>>(jobs,items,count);
  else Exl3GemmKernel<2,true,false,1,false><<<count,128>>>(jobs,items,count);
}

// k-step walk over the tiles of one column block in the front order
// (front_order.h, split S): FrontTile(b, a) = base(a) + Col(b). base walks
// tile f in a chunk, then chunk row q (S * B/8 chunks apart), then split s,
// all in scalar registers.
struct PrefillTileWalk {
  int base = 0, f = 0, q = 0, s = 0, chunks = 0, jump = 0, split = 0;
  __device__ explicit PrefillTileWalk(const Exl3GemmJob& j) {
    const int A = j.I / 16, groups = j.O / 16 / kFrontCols;
    chunks = j.front_s > 0 ? A / j.front_s / kFrontTiles : 0;
    split = groups * kFrontCols * kFrontTiles;
    jump = j.front_s * split;
  }
  __device__ static int Col(int b) {
    return (b / kFrontCols) * kFrontCols * kFrontTiles + (b % kFrontCols) * kFrontTiles;
  }
  // Past the last tile base points at the next split or past the matrix;
  // those prefetches are unused and out-of-range loads return 0.
  __device__ void Next() {
    if (++f < kFrontTiles) { ++base; return; }
    f = 0;
    if (++q < chunks) { base += jump - (kFrontTiles - 1); return; }
    q = 0; ++s; base = s * split;
  }
};

// Packed GEMM body: NB column tiles per wave, so each A fragment feeds NB
// WMMAs, and the A loads of a k step are issued before its trellis decode,
// which then hides their latency. Each output tile sees the same WMMA
// sequence as in Exl3GemmBody, so results are bitwise equal. Row tiles past
// M are skipped. S must be 1. With keep, the accumulators go to keep[n * RM
// + r] instead of partial. HALF_OUT stores the partial as fp16 (round to
// nearest even; magnitudes above 65504 saturate, NaN stays NaN). FRONT reads
// tiles in the front order (j.front_s > 0); the WMMA sequence is the same, so
// the results are bitwise equal. COPY > 0 only makes a separate instantiation (the same code): see
// PrefillMoeGemmBody.
template<int K,int RM,int NB,bool EARLY_A=true,bool SKIP=true,bool ORDER=SKIP,bool HALF_OUT=false,bool FRONT=false,int COPY=0> __device__ __forceinline__ void PrefillWideBody(const Exl3GemmJob& j,
    const Exl3GemmItem& it, uint32_t* ws, GemmF8* keep=nullptr) {
  constexpr int T=TileDwords(K);
  const int lane=threadIdx.x&31, wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int b=it.b0+wave*NB;
  if (b*16>=j.O) return;
  const int A=j.I/16;
  const LaneWindow<K> win(lane);
  __amdgpu_buffer_rsrc_t rs[NB];
#pragma unroll
  for (int n=0; n<NB; ++n) {
    if constexpr (FRONT) {
      const size_t col=PrefillTileWalk::Col(b+n), total=(size_t)A*(j.O/16);
      rs[n]=StreamRsrc(j.tiles+col*T,(uint32_t)((total-col)*T*4));
    } else {
      rs[n]=StreamRsrc(j.tiles+(size_t)(b+n)*A*T,A*T*4);
    }
  }
  PrefillTileWalk walk(j);
  GemmF8 acc[NB][RM]={};
  const uint32_t xbytes=(uint32_t)j.Xrows*j.I*2;
  const auto xrs=StreamRsrc(reinterpret_cast<const uint32_t*>(j.x),xbytes);
  const int astep=j.Xrows*32;
  uint32_t xo[RM];
#pragma unroll
  for (int r=0; r<RM; ++r) {
    const int row=it.m0+r*16+(lane&15);
    xo[r]=row<j.M ? (uint32_t)row*32 : xbytes;
  }
  uint32_t* w=ws+wave*kGemmWaveWords;
  const int o0=win.d0*4, o1=((win.d0+1)%T)*4, o2=((win.d0+2)%T)*4;
  uint32_t raw[NB][3];
#pragma unroll
  for (int n=0; n<NB; ++n) {
    raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0,0,0);
    raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1,0,0);
    raw[n][2]=K>=5 ? __builtin_amdgcn_raw_buffer_load_b32(rs[n],o2,0,0) : 0;
  }
  for (int a=0; a<A; ++a) {
    GemmH16 afr[RM];
    auto load_a=[&] {
#pragma unroll
      for (int r=0; r<RM; ++r) {
        if (SKIP && it.m0+r*16>=j.M) break;
        afr[r]=GemmAFragment<false>(xrs,xo[r],a,nullptr,0,astep);
      }
      // Keeps the A loads ahead of the weight refills (and, with EARLY_A, the
      // decode). Dense full tiles schedule better without it.
      if constexpr (ORDER) asm volatile("" ::: "memory");
    };
    if constexpr (EARLY_A) load_a();
    int next=(a+1)*T*4;
    if constexpr (FRONT) { walk.Next(); next=walk.base*T*4; }
    GemmH16 bfs[EARLY_A ? 1 : NB];
    if constexpr (!EARLY_A) {
      // Exl3GemmBody's order: decode and exchange first, then A, then refill.
      uint32_t wv[NB][3];
#pragma unroll
      for (int n=0; n<NB; ++n) {
        wv[n][0]=raw[n][0]; wv[n][1]=raw[n][1]; wv[n][2]=raw[n][2];
        __half2 p[4];
        DecodeLane<K,kMulTrue16>(wv[n][0],wv[n][1],wv[n][2],win.shift,p);
        bfs[n]=GemmBFragment<true>(p,w,lane);
      }
      load_a();
    }
#pragma unroll
    for (int n=0; n<NB; ++n) {
      const uint32_t w0=raw[n][0], w1=raw[n][1], w2=raw[n][2];
      raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0+next,0,0);
      raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1+next,0,0);
      if constexpr (K>=5) raw[n][2]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o2+next,0,0);
      GemmH16 bf;
      if constexpr (EARLY_A) {
        __half2 p[4];
        DecodeLane<K,kMulTrue16>(w0,w1,w2,win.shift,p);
        bf=GemmBFragment<true>(p,w,lane);
      } else {
        bf=bfs[n];
      }
#pragma unroll
      for (int r=0; r<RM; ++r) {
        if (SKIP && it.m0+r*16>=j.M) break;
        acc[n][r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r],bf,acc[n][r]);
      }
    }
  }
  if (keep) {
#pragma unroll
    for (int n=0; n<NB; ++n)
#pragma unroll
      for (int r=0; r<RM; ++r) keep[n*RM+r]=acc[n][r];
    return;
  }
  constexpr uint32_t kOutBytes=HALF_OUT ? 2 : 4;
  const auto ps=StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial),(uint32_t)j.M*j.O*kOutBytes);
#pragma unroll
  for (int n=0; n<NB; ++n) {
    const uint32_t po=((uint32_t)(it.m0+lane/16)*j.O+(b+n)*16+(lane&15))*kOutBytes;
#pragma unroll
    for (int r=0; r<RM; ++r) {
      if (SKIP && it.m0+r*16>=j.M) break;
#pragma unroll
      for (int z=0; z<8; ++z) {
        if constexpr (HALF_OUT) {
          const float v=acc[n][r][z];
          const __half h=__float2half(fabsf(v)>65504.f ? copysignf(65504.f,v) : v);
          __builtin_amdgcn_raw_buffer_store_b16(__half_as_ushort(h),ps,
                                                po+(uint32_t)(r*16+2*z)*j.O*2,0,0);
        } else {
          __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[n][r][z]),ps,
                                                po+(uint32_t)(r*16+2*z)*j.O*4,0,0);
        }
      }
    }
  }
}

// Packed dense GEMM with the output transform fused. A block owns 128 output
// columns (8 / NB waves of NB column tiles; NB = 2 with early A loads is about
// 5 percent faster at 4K), so after the k loop it stages each 16-row tile in LDS
// and runs LinOutBlock's arithmetic (S = 1) on whole 128-column blocks: the
// fp32 partial never goes to memory. Results are bitwise equal to the GEMM
// plus LinOutKernel. Needs O % 128 == 0 (PrefillLinear::Build checks it).
constexpr int kFusedWaves = 8, kFusedPad = 16 * kFusedWaves + 4;
// NB column tiles per wave (PrefillWideBody), 8 / NB waves per block.
// K6 instantiations take only K6 jobs: a K6 branch next to K3..5 makes the
// K3..5 code spill (544 B scratch at RM8 NB2).
// EPI (at most one): a caller-supplied epilogue for one group (the GDN input
// projection, an earlier change). It runs after the block has stored its rows, with
// Run(dst, ldd, first row, end row, first column). With no EPI the kernel is
// the shared dense kernel.
template <int RM, int NB, bool EARLY, bool K6 = false, bool FRONT = false, class... EPI>
__global__ void __launch_bounds__(32 * kFusedWaves / NB) PrefillGemmOutKernel(const LinJob* __restrict__ lin,
    const Exl3GemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n, EPI... epi) {
  static_assert(sizeof...(EPI) <= 1);
  constexpr int kWaves = kFusedWaves / NB;
  static_assert(kWaves * kGemmWaveWords <= 16 * kFusedPad);
  __shared__ float stage[16 * kFusedPad];
  if ((int)blockIdx.x >= n) return;
  const Exl3GemmItem it = items[blockIdx.x];
  const Exl3GemmJob j = jobs[it.job];
  const LinJob& L = lin[it.job];
  uint32_t* ws = reinterpret_cast<uint32_t*>(stage);
  GemmF8 acc[NB * RM];
  if constexpr (K6) PrefillWideBody<6,RM,NB,EARLY,false,false,false,FRONT>(j,it,ws,acc);
  else if (j.K == 3) PrefillWideBody<3,RM,NB,EARLY,false,false,false,FRONT>(j,it,ws,acc);
  else if (j.K == 4) PrefillWideBody<4,RM,NB,EARLY,false,false,false,FRONT>(j,it,ws,acc);
  else PrefillWideBody<5,RM,NB,EARLY,false,false,false,FRONT>(j,it,ws,acc);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, c = it.b0 * 16 + lane * 4;
  float svh[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) svh[i] = __half2float(L.svh[c + i]);
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int row0 = it.m0 + r * 16;
    if (row0 >= j.M) break;
    __syncthreads();  // B exchanges and earlier readers are done with stage
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int z = 0; z < 8; ++z)
        stage[(2 * z + (lane >> 4)) * kFusedPad + (wave * NB + nb) * 16 + (lane & 15)] = acc[nb * RM + r][z];
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 16 / kWaves; ++h) {
      const int lr = wave * (16 / kWaves) + h, row = row0 + lr;
      if (row >= j.M) break;
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = 0.f + stage[lr * kFusedPad + lane * 4 + i];
      Wht128(v);
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] *= kHadNorm * svh[i];
      if (L.epi == kEpiStore) {
        float* d = L.dst + (size_t)row * L.ldd + c;
#pragma unroll
        for (int i = 0; i < 4; ++i) d[i] = v[i];
      } else {
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
          const float g = L.aux[row * kStreams + s];
          float* d = L.dst + (size_t)row * (L.ldd ? L.ldd : kWide) + s * kHidden + c;
#pragma unroll
          for (int i = 0; i < 4; ++i) d[i] += g * v[i];
        }
      }
    }
  }
  if constexpr (sizeof...(EPI) == 1) {
    static_assert(NB == 2, "GdnPrepEpi is built and tested with NB 2 only");
    __syncthreads();  // the block's stores are visible to the whole block
    (epi.Run(L.dst, L.ldd, it.m0, min(it.m0 + RM * 16, j.M), it.b0 * 16), ...);
  }
}

// PrefillGemmOutKernel for groups whose jobs all have the width KW (7 or 8, the h6_ng8 packs); one
// width per instantiation, as for K6. A copy of PrefillGemmOutKernel, not a shared inline store
// function: that refactor changed the code of the train 23 kernels (isa_same).
template <int RM, int NB, bool EARLY, int KW, bool FRONT = false, class... EPI>
__global__ void __launch_bounds__(32 * kFusedWaves / NB) PrefillGemmOutWideKernel(const LinJob* __restrict__ lin,
    const Exl3GemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n, EPI... epi) {
  static_assert(sizeof...(EPI) <= 1 && (KW == 7 || KW == 8));
  constexpr int kWaves = kFusedWaves / NB;
  static_assert(kWaves * kGemmWaveWords <= 16 * kFusedPad);
  __shared__ float stage[16 * kFusedPad];
  if ((int)blockIdx.x >= n) return;
  const Exl3GemmItem it = items[blockIdx.x];
  const Exl3GemmJob j = jobs[it.job];
  const LinJob& L = lin[it.job];
  uint32_t* ws = reinterpret_cast<uint32_t*>(stage);
  GemmF8 acc[NB * RM];
  PrefillWideBody<KW,RM,NB,EARLY,false,false,false,FRONT>(j,it,ws,acc);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, c = it.b0 * 16 + lane * 4;
  float svh[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) svh[i] = __half2float(L.svh[c + i]);
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int row0 = it.m0 + r * 16;
    if (row0 >= j.M) break;
    __syncthreads();  // B exchanges and earlier readers are done with stage
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int z = 0; z < 8; ++z)
        stage[(2 * z + (lane >> 4)) * kFusedPad + (wave * NB + nb) * 16 + (lane & 15)] = acc[nb * RM + r][z];
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 16 / kWaves; ++h) {
      const int lr = wave * (16 / kWaves) + h, row = row0 + lr;
      if (row >= j.M) break;
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = 0.f + stage[lr * kFusedPad + lane * 4 + i];
      Wht128(v);
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] *= kHadNorm * svh[i];
      if (L.epi == kEpiStore) {
        float* d = L.dst + (size_t)row * L.ldd + c;
#pragma unroll
        for (int i = 0; i < 4; ++i) d[i] = v[i];
      } else {
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
          const float g = L.aux[row * kStreams + s];
          float* d = L.dst + (size_t)row * (L.ldd ? L.ldd : kWide) + s * kHidden + c;
#pragma unroll
          for (int i = 0; i < 4; ++i) d[i] += g * v[i];
        }
      }
    }
  }
  if constexpr (sizeof...(EPI) == 1) {
    static_assert(NB == 2, "GdnPrepEpi is built and tested with NB 2 only");
    __syncthreads();  // the block's stores are visible to the whole block
    (epi.Run(L.dst, L.ldd, it.m0, min(it.m0 + RM * 16, j.M), it.b0 * 16), ...);
  }
}

// front: every job of the group has front-order tiles (default body only).
template <class... EPI>
inline void LaunchPrefillGemmOut(int rm, const LinJob* lin, const Exl3GemmJob* jobs,
                                 const Exl3GemmItem* items, int count, bool k6 = false, bool front = false,
                                 const EPI&... epi) {
  if (k6) {
    // K6 (4.05) uses the default body variant only.
    auto g6 = [&](auto f) {
      constexpr bool F = decltype(f)::value;
      if (rm == 8) PrefillGemmOutKernel<8, 2, true, true, F, EPI...><<<count, 32 * kFusedWaves / 2>>>(lin, jobs, items, count, epi...);
      else if (rm == 4) PrefillGemmOutKernel<4, 2, true, true, F, EPI...><<<count, 32 * kFusedWaves / 2>>>(lin, jobs, items, count, epi...);
      else if (rm == 2) PrefillGemmOutKernel<2, 2, true, true, F, EPI...><<<count, 32 * kFusedWaves / 2>>>(lin, jobs, items, count, epi...);
      else { std::fprintf(stderr, "K6 fused output GEMM supports RM 2, 4 and 8\n"); std::abort(); }
    };
    if (front) g6(std::true_type{});
    else g6(std::false_type{});
    return;
  }
  // The body: two N blocks and the early A loads.
  auto go = [&](auto rmc) {
    PrefillGemmOutKernel<decltype(rmc)::value, 2, true, false, false, EPI...><<<count, 32 * kFusedWaves / 2>>>(lin, jobs, items, count, epi...);
  };
  if (front) {
    auto gf = [&](auto rmc) {
      PrefillGemmOutKernel<decltype(rmc)::value, 2, true, false, true, EPI...><<<count, 32 * kFusedWaves / 2>>>(lin, jobs, items, count, epi...);
    };
    if (rm == 16) gf(std::integral_constant<int,16>{});
    else if (rm == 8) gf(std::integral_constant<int,8>{});
    else if (rm == 4) gf(std::integral_constant<int,4>{});
    else gf(std::integral_constant<int,2>{});
    return;
  }
  if (rm == 16) go(std::integral_constant<int,16>{});
  else if (rm == 8) go(std::integral_constant<int,8>{});
  else if (rm == 4) go(std::integral_constant<int,4>{});
  else go(std::integral_constant<int,2>{});
}

// A group of one wide width kw (7 or 8), engine tile order, RM 2, 4 or 8 (as K6).
template <class... EPI>
inline void LaunchPrefillGemmOutWide(int kw, int rm, const LinJob* lin, const Exl3GemmJob* jobs,
                                     const Exl3GemmItem* items, int count, const EPI&... epi) {
  auto go = [&](auto kc) {
    constexpr int KW = decltype(kc)::value;
    if (rm == 8) PrefillGemmOutWideKernel<8, 2, true, KW, false, EPI...><<<count, 32 * kFusedWaves / 2>>>(lin, jobs, items, count, epi...);
    else if (rm == 4) PrefillGemmOutWideKernel<4, 2, true, KW, false, EPI...><<<count, 32 * kFusedWaves / 2>>>(lin, jobs, items, count, epi...);
    else if (rm == 2) PrefillGemmOutWideKernel<2, 2, true, KW, false, EPI...><<<count, 32 * kFusedWaves / 2>>>(lin, jobs, items, count, epi...);
    else { std::fprintf(stderr, "K%d fused output GEMM supports RM 2, 4 and 8\n", KW); std::abort(); }
  };
  if (kw == 7) go(std::integral_constant<int, 7>{});
  else if (kw == 8) go(std::integral_constant<int, 8>{});
  else { std::fprintf(stderr, "internal error: wide prefill GEMM with K%d\n", kw); std::abort(); }
}

// Per-group GEMM shapes. By default they follow the chunk rows, as measured by
// exl3-gemm and in the engine: about 10R/512 rows per expert, and out
// projections need enough items for 40 CUs. CRUCIBLE_PF_TILES
// (moe_rm,moe_packed,in_rm,in_packed,out_rm,out_packed) fixes them for A/B tests.
struct PrefillTiles {
  bool fixed = false;
  PrefillGemmShape moe{2, true}, in{8, true}, out{4, true};
  // MoE row tiles for 4096-row chunks: 6 (96 rows) fits the about 80 rows per
  // expert, and RM8 with two column tiles per wave spills in the queue
  // kernel. CRUCIBLE_PF_MOE_RM (2, 4, 6, 7 or 8) overrides it for tests.
  PrefillGemmShape Moe(int R) const {
    static const int big=crucible::SwitchText("CRUCIBLE_PF_MOE_RM") ? std::atoi(crucible::SwitchText("CRUCIBLE_PF_MOE_RM")) : 6;
    return fixed ? moe : PrefillGemmShape{R >= 4096 ? big : R >= 2048 ? 4 : 2, true};
  }
  // With the fused output transform, RM16 items run about 2x slower than RM8
  // in the engine at 4K (8-wave blocks; cause not found yet), so dense groups
  // stop at RM8.
  PrefillGemmShape In(int R) const { return fixed ? in : PrefillGemmShape{8, true}; }
  PrefillGemmShape Out(int R) const {
    return fixed ? out : PrefillGemmShape{R > 1024 ? 8 : 4, true};
  }
};
inline PrefillTiles PrefillTilesFromEnv(PrefillTiles t) {
  const char* e = crucible::SwitchText("CRUCIBLE_PF_TILES");
  int v[6];
  if (!e || std::sscanf(e, "%d,%d,%d,%d,%d,%d", v, v+1, v+2, v+3, v+4, v+5) != 6) return t;
  return {true, {std::min(8, ParseRows(v[0])), v[1] != 0}, {ParseRows(v[2]), v[3] != 0},
          {ParseRows(v[4]), v[5] != 0}};
}

struct PrefillLinear {
  std::vector<LinJob> jobs;
  LinJob* linear = nullptr;
  Exl3GemmJob* gemm = nullptr;
  Exl3GemmItem* items = nullptr;
  XfItem *input = nullptr, *output = nullptr;
  int count = 0, nin = 0, nout = 0;
  int count6 = 0;            // split: items [0, count6) belong to K6 jobs
  PrefillGemmShape shape;
  bool fused = false;        // packed shape: the GEMM applies the output transform
  bool front = false;        // tiles in the front order (fused path only)
  bool k6 = false;           // every job has K6 (fused output kernels take one width class)
  bool split = false;        // K6 and K3..5 jobs: one fused launch per width class
  int wide = 0;              // 7 or 8: every job has that width (PrefillGemmOutWideKernel)

  size_t DeviceBytes() const {
    return jobs.size() * (sizeof(LinJob) + sizeof(Exl3GemmJob)) +
           (size_t)count * sizeof(Exl3GemmItem) + (size_t)((input ? nin : 0) + nout) * sizeof(XfItem);
  }

  // Releases the device plan arrays (the engine's plan cache evicts with it).
  void Free() {
    for (void* p : {(void*)linear, (void*)gemm, (void*)items, (void*)input, (void*)output})
      if (p) HIP_OK(hipFree(p));
    linear = nullptr; gemm = nullptr; items = nullptr; input = nullptr; output = nullptr;
  }
  __half* repack = nullptr;  // packed copy of an input written row-major elsewhere

  // With a packed shape and no input transform, `repack` receives the
  // packed copy of jobs[0].xh before the GEMM.
  // front_s: per job, 0 for engine order or the front-order split (Lin::front_s).
  void Build(std::vector<LinJob> source, PrefillGemmShape g, __half* repack_dst, float* partial,
             size_t partial_cap, __half* xh, size_t xh_cap, const std::vector<int>& front_s = {}) {
    size_t po = 0, xo = 0;
    nin = 0;
    shape = g; repack = g.packed ? repack_dst : nullptr; front = false;
    bool fusable = g.packed;
    for (const LinJob& job : source) fusable = fusable && job.O % (16 * kFusedWaves) == 0;
    std::vector<int> widths;
    for (const LinJob& job : source) widths.push_back(job.K);
    const PrefillFusedClasses classes = PrefillFusedClassesOf(widths.data(), (int)widths.size(), fusable);
    fused = classes.fused; k6 = classes.k6; split = classes.split;
    wide = PrefillWideKOf(widths.data(), (int)widths.size());
    if (wide < 0 || (wide > 0 && !fused)) {
      std::fprintf(stderr,"prefill GEMM: K7 and K8 need groups of one width with the fused packed GEMM\n");
      std::exit(2);
    }
    const int step = fused ? kFusedWaves : 4;
    // Packed input transforms derive their items (PrefillInItem) when every
    // job has the same R and I; LinInKernel and mixed groups read a list.
    const bool listed = !g.packed || std::any_of(source.begin(), source.end(), [&](const LinJob& j) {
      return j.R != source[0].R || j.I != source[0].I;
    });
    std::vector<Exl3GemmJob> gj;
    std::vector<Exl3GemmItem> gi;
    std::vector<XfItem> in, out;
    for (size_t j = 0; j < source.size(); ++j) {
      LinJob job = source[j];
      if (!PrefillWidthSupported(job.K)) {
        std::fprintf(stderr,"prefill GEMM supports EXL3 K3..8 only\n");std::abort();
      }
      job.S = 1; job.partial = partial+po; job.xh = xh+xo;
      po += (size_t)job.R*job.O; xo += (size_t)job.R*job.I;
      jobs.push_back(job);
      const int fs = j < front_s.size() ? front_s[j] : 0;
      if (j > 0 && (fs > 0) != front) {
        std::fprintf(stderr,"prefill linear group mixes front and engine tile orders\n"); std::exit(2);
      }
      front = fs > 0;
      if (front && (!fused || wide || !FrontFitsShape(job.I/16,job.O/16,fs))) {
        std::fprintf(stderr,"front tile order needs the fused packed prefill GEMM\n"); std::exit(2);
      }
      gj.push_back({job.tiles,repack ? repack : job.xh,job.partial,nullptr,job.I,job.O,job.R,1,job.K,job.R,fs});
      for (int r = 0; r < job.R; r += 16*g.rm)
        for (int b = 0; b < job.O/16; b += step) gi.push_back({(int)j,0,r,b});
      // The fused GEMM applies the output transform itself, so Run never
      // reads output items there (about 322 MiB per 4096-row plan).
      nin += job.R*(job.I/128);
      for (int r = 0; r < job.R; ++r) {
        if (listed) for (int b = 0; b < job.I/128; ++b) in.push_back({(int)j,r,b});
        if (!fused) for (int b = 0; b < job.O/128; ++b) out.push_back({(int)j,r,b});
      }
    }
    // Split: K6 items first. The order of items does not change any output.
    count6 = split ? (int)(std::stable_partition(gi.begin(), gi.end(), [&](const Exl3GemmItem& it) {
      return source[it.job].K == 6;
    }) - gi.begin()) : 0;
    if (po > partial_cap || xo > xh_cap || (repack && (source.size() != 1 || source[0].I % 128))) {
      std::fprintf(stderr,"prefill linear workspace is too small\n"); std::exit(1);
    }
    linear=PrefillUpload(jobs); gemm=PrefillUpload(gj); items=PrefillUpload(gi);
    input=PrefillUpload(in); output=PrefillUpload(out);
    count=gi.size(); nout=out.size();
  }
  // mark(stage) runs after the input transform and after the GEMM.
  // input_packed: the producer already wrote `repack` (no repack kernel).
  // epi (at most one): the fused GEMM's caller epilogue (PrefillGemmOutKernel EPI).
  template <class Mark, class... EPI>
  void Run(bool transform_input, bool transform_output, Mark mark, bool input_packed = false,
           const __half* input_half = nullptr, const EPI&... epi) const {
    if (sizeof...(EPI) && !(fused && transform_output)) {
      std::fprintf(stderr,"prefill linear: an epilogue needs the fused output transform\n"); std::abort();
    }
    if (input_half && !transform_input) {
      std::fprintf(stderr,"prefill linear: fp16 source needs the input transform\n"); std::abort();
    }
    if (input_packed && (transform_input || !repack)) {
      std::fprintf(stderr,"prefill linear: packed input needs a repack target\n"); std::abort();
    }
    if (transform_input) {
      if (input_half) {
        if (shape.packed) PrefillLinInHalfKernel<true><<<(nin+7)/8,256>>>(linear,input,nin,input_half);
        else PrefillLinInHalfKernel<false><<<(nin+7)/8,256>>>(linear,input,nin,input_half);
      } else if (shape.packed) PrefillLinInPackedKernel<<<(nin+7)/8,256>>>(linear,input,nin);
      else LinInKernel<<<(nin+7)/8,256>>>(linear,input,nin);
    } else if (repack && !input_packed) {
      const LinJob& j = jobs[0];
      PrefillPackRowsKernel<<<(PackRowsThreads(j.R,j.I)+255)/256,256>>>(j.xh,repack,j.R,j.I);
    }
    mark("lin_in");
    if (fused && transform_output) {
      // A split group (4.05 QSA input) ran unfused with LinOutKernel before
      // same values, 27 ms more per 4,096 rows.
      if (wide) {
        LaunchPrefillGemmOutWide(wide,shape.rm,linear,gemm,items,count,epi...);
      } else if (split) {
        LaunchPrefillGemmOut(shape.rm,linear,gemm,items,count6,true,front,epi...);
        LaunchPrefillGemmOut(shape.rm,linear,gemm,items+count6,count-count6,false,front,epi...);
      } else LaunchPrefillGemmOut(shape.rm,linear,gemm,items,count,k6,front,epi...);
      mark("lin_gemm");
      return;
    }
    if (fused) { std::fprintf(stderr,"fused prefill linear needs the output transform\n"); std::abort(); }
    LaunchPrefillGemm(shape,gemm,items,count);
    mark("lin_gemm");
    if (transform_output) LinOutKernel<<<(nout+7)/8,256>>>(linear,output,nout);
  }
  void Run(bool transform_input = true, bool transform_output = true) const {
    Run(transform_input, transform_output, [](const char*) {});
  }
};

}  // namespace crucible
