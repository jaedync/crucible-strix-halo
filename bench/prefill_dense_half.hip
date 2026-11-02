// Transient fp16 copy of the dense EXL3 weights for bulk prefill (a
// short-lived variant of a dense fp16 weight copy). Compares
// the engine's fused EXL3 GEMM (PrefillGemmOutKernel) with a decode-to-copy
// kernel plus a half GEMM that keeps the A path, the k order of every
// output and the fused epilogue, so the outputs must be bitwise equal.
// Usage: prefill_dense_half [M=4096] [K=5] [I=2560] [O=16384]
#include "prefill_common.h"
#include "../engine/exl3/prefill_linear.h"
using namespace crucible;
using namespace prefill_bench;

// One wave decodes one tile into its WMMA B fragments. Copy layout:
// [column tile b][row tile a][16 columns][16 k] halves (engine tile order).
template <int K>
__global__ void __launch_bounds__(128) DecodeCopyKernel(const uint32_t* tiles, __half* copy, size_t n) {
  __shared__ uint32_t ws[kGemmWeightWords];
  constexpr int T = 8 * K;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const LaneWindow<K> win(lane);
  uint32_t* w = ws + wave * kGemmWaveWords;
  for (size_t t = (size_t)blockIdx.x * 4 + wave; t < n; t += (size_t)gridDim.x * 4) {
    const uint32_t* src = tiles + t * T;
    const uint32_t w0 = src[win.d0], w1 = src[(win.d0 + 1) % T], w2 = K >= 5 ? src[(win.d0 + 2) % T] : 0;
    __half2 p[4];
    DecodeLane<K, kMulTrue16>(w0, w1, w2, win.shift, p);
    const GemmH16 bf = GemmBFragment<true>(p, w, lane);
    if (lane < 16) *reinterpret_cast<GemmH16*>(copy + (t * 16 + lane) * 16) = bf;
  }
}

// PrefillWideBody's A path and per-output WMMA chain with B from the copy,
// one k step ahead.
template <int RM, int NB>
__device__ __forceinline__ void HalfBody(const Exl3GemmJob& j, const Exl3GemmItem& it, const __half* copy,
                                         GemmF8* acc) {
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int b = it.b0 + wave * NB, A = j.I / 16;
  const auto brs = StreamRsrc(reinterpret_cast<const uint32_t*>(copy), (uint32_t)((size_t)A * (j.O / 16) * 512));
  uint32_t bo[NB];
#pragma unroll
  for (int n = 0; n < NB; ++n) bo[n] = ((uint32_t)(b + n) * A * 16 + (lane & 15)) * 32;
  const uint32_t xbytes = (uint32_t)j.Xrows * j.I * 2;
  const auto xrs = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), xbytes);
  const int astep = j.Xrows * 32;
  uint32_t xo[RM];
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int row = it.m0 + r * 16 + (lane & 15);
    xo[r] = row < j.M ? (uint32_t)row * 32 : xbytes;
  }
#pragma unroll
  for (int i = 0; i < NB * RM; ++i) acc[i] = GemmF8{};
  auto load_b = [&](GemmH16 (&f)[NB], int a) {
#pragma unroll
    for (int n = 0; n < NB; ++n) {
      auto* fp = reinterpret_cast<U4*>(&f[n]);
      fp[0] = __builtin_amdgcn_raw_buffer_load_b128(brs, bo[n], a * 512, 0);
      fp[1] = __builtin_amdgcn_raw_buffer_load_b128(brs, bo[n] + 16, a * 512, 0);
    }
  };
  GemmH16 bf[NB];
  load_b(bf, 0);
  for (int a = 0; a < A; ++a) {
    GemmH16 afr[RM], bn[NB];
#pragma unroll
    for (int r = 0; r < RM; ++r) afr[r] = GemmAFragment<false>(xrs, xo[r], a, nullptr, 0, astep);
    load_b(bn, a + 1 < A ? a + 1 : a);
#pragma unroll
    for (int n = 0; n < NB; ++n)
#pragma unroll
      for (int r = 0; r < RM; ++r)
        acc[n * RM + r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r], bf[n], acc[n * RM + r]);
#pragma unroll
    for (int n = 0; n < NB; ++n) bf[n] = bn[n];
  }
}

// PrefillGemmOutKernel with HalfBody (kEpiStore epilogue only).
template <int RM, int NB>
__global__ void __launch_bounds__(32 * kFusedWaves / NB) HalfOutKernel(const LinJob* __restrict__ lin,
    const Exl3GemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n, const __half* copy) {
  constexpr int kWaves = kFusedWaves / NB;
  __shared__ float stage[16 * kFusedPad];
  if ((int)blockIdx.x >= n) return;
  const Exl3GemmItem it = items[blockIdx.x];
  const Exl3GemmJob j = jobs[it.job];
  const LinJob& L = lin[it.job];
  GemmF8 acc[NB * RM];
  HalfBody<RM, NB>(j, it, copy, acc);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, c = it.b0 * 16 + lane * 4;
  float svh[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) svh[i] = __half2float(L.svh[c + i]);
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int row0 = it.m0 + r * 16;
    if (row0 >= j.M) break;
    __syncthreads();
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
      float* d = L.dst + (size_t)row * L.ldd + c;
#pragma unroll
      for (int i = 0; i < 4; ++i) d[i] = v[i];
    }
  }
}

// 2x2 waves per 128x128 item: wave (wr, wc) owns rows m0 + 64 wr (RM4) and
// column tiles b0 + 4 wc (NB4). Each output keeps its k chain, so the result
// is bitwise equal to HalfBody. BL: B fragments go through LDS; each wave
// loads two of the block's eight column tiles per k step (as the EXL3 body
// decodes two), double-buffered with one barrier per k step.
template <bool BL>
__global__ void __launch_bounds__(128) Half22Kernel(const LinJob* __restrict__ lin,
    const Exl3GemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n, const __half* copy) {
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  __shared__ float stage[16 * kFusedPad];
  __shared__ __attribute__((aligned(16))) GemmH16 bls[BL ? 2 : 1][8][16];
  if ((int)blockIdx.x >= n) return;
  const Exl3GemmItem it = items[blockIdx.x];
  const Exl3GemmJob j = jobs[it.job];
  const LinJob& L = lin[it.job];
  const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int wr = wave >> 1, wc = wave & 1, A = j.I / 16;
  const auto brs = StreamRsrc(reinterpret_cast<const uint32_t*>(copy), (uint32_t)((size_t)A * (j.O / 16) * 512));
  const uint32_t xbytes = (uint32_t)j.Xrows * j.I * 2;
  const auto xrs = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), xbytes);
  const int astep = j.Xrows * 32;
  uint32_t xo[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int row = it.m0 + wr * 64 + r * 16 + (lane & 15);
    xo[r] = row < j.M ? (uint32_t)row * 32 : xbytes;
  }
  GemmF8 acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = GemmF8{};
  // Direct: this wave's four tiles. LDS: tiles 2 wave, 2 wave + 1 of the block.
  constexpr int NL = BL ? 2 : 4;
  uint32_t bo[NL];
#pragma unroll
  for (int q = 0; q < NL; ++q) {
    const int tb = BL ? it.b0 + 2 * wave + q : it.b0 + 4 * wc + q;
    bo[q] = ((uint32_t)tb * A * 16 + (lane & 15)) * 32;
  }
  auto load_b = [&](GemmH16 (&f)[NL], int a) {
#pragma unroll
    for (int q = 0; q < NL; ++q) {
      auto* fp = reinterpret_cast<U4*>(&f[q]);
      fp[0] = __builtin_amdgcn_raw_buffer_load_b128(brs, bo[q], a * 512, 0);
      fp[1] = __builtin_amdgcn_raw_buffer_load_b128(brs, bo[q] + 16, a * 512, 0);
    }
  };
  GemmH16 bf[NL];
  load_b(bf, 0);
  if constexpr (BL) {
    if (lane < 16) {
#pragma unroll
      for (int q = 0; q < NL; ++q) bls[0][2 * wave + q][lane] = bf[q];
    }
    load_b(bf, A > 1 ? 1 : 0);
    __syncthreads();
  }
  for (int a = 0; a < A; ++a) {
    GemmH16 afr[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) afr[r] = GemmAFragment<false>(xrs, xo[r], a, nullptr, 0, astep);
    if constexpr (BL) {
      const int cur = a & 1;
      GemmH16 bn[NL];
      load_b(bn, a + 2 < A ? a + 2 : a);
      GemmH16 bb[4];
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) bb[nb] = bls[cur][4 * wc + nb][lane & 15];
#pragma unroll
      for (int nb = 0; nb < 4; ++nb)
#pragma unroll
        for (int r = 0; r < 4; ++r)
          acc[nb * 4 + r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r], bb[nb], acc[nb * 4 + r]);
      if (lane < 16 && a + 1 < A) {
#pragma unroll
        for (int q = 0; q < NL; ++q) bls[cur ^ 1][2 * wave + q][lane] = bf[q];
      }
#pragma unroll
      for (int q = 0; q < NL; ++q) bf[q] = bn[q];
      __syncthreads();
    } else {
      GemmH16 bn[NL];
      load_b(bn, a + 1 < A ? a + 1 : a);
#pragma unroll
      for (int nb = 0; nb < 4; ++nb)
#pragma unroll
        for (int r = 0; r < 4; ++r)
          acc[nb * 4 + r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r], bf[nb], acc[nb * 4 + r]);
#pragma unroll
      for (int q = 0; q < NL; ++q) bf[q] = bn[q];
    }
  }
  const int c = it.b0 * 16 + lane * 4;
  float svh[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) svh[i] = __half2float(L.svh[c + i]);
  for (int r8 = 0; r8 < 8; ++r8) {
    const int row0 = it.m0 + r8 * 16;
    if (row0 >= j.M) break;
    __syncthreads();
    if (wr == r8 >> 2) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r != (r8 & 3)) continue;
#pragma unroll
        for (int nb = 0; nb < 4; ++nb)
#pragma unroll
          for (int z = 0; z < 8; ++z)
            stage[(2 * z + (lane >> 4)) * kFusedPad + (4 * wc + nb) * 16 + (lane & 15)] = acc[nb * 4 + r][z];
      }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const int lr = wave * 4 + h, row = row0 + lr;
      if (row >= j.M) break;
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = 0.f + stage[lr * kFusedPad + lane * 4 + i];
      Wht128(v);
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] *= kHadNorm * svh[i];
      float* d = L.dst + (size_t)row * L.ldd + c;
#pragma unroll
      for (int i = 0; i < 4; ++i) d[i] = v[i];
    }
  }
}

// EXL3 2x2: PrefillWideBody's decode (EARLY_A, full tiles) with the 2x2 wave
// layout of Half22Kernel. BL = false: each wave decodes its own four column
// tiles (every tile is decoded by two waves). BL = true: each wave decodes two
// of the block's eight tiles and shares them through LDS (double-buffered,
// one barrier per k step). Same k chain per output, so bitwise equal.
template <int K, bool BL>
__global__ void __launch_bounds__(128) Exl3x22Kernel(const LinJob* __restrict__ lin,
    const Exl3GemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n) {
  constexpr int T = 8 * K, ND = BL ? 2 : 4;
  static_assert(4 * kGemmWaveWords <= 16 * kFusedPad);
  __shared__ float stage[16 * kFusedPad];
  __shared__ __attribute__((aligned(16))) GemmH16 bls[BL ? 2 : 1][8][16];
  if ((int)blockIdx.x >= n) return;
  const Exl3GemmItem it = items[blockIdx.x];
  const Exl3GemmJob j = jobs[it.job];
  const LinJob& L = lin[it.job];
  const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int wr = wave >> 1, wc = wave & 1, A = j.I / 16;
  const LaneWindow<K> win(lane);
  __amdgpu_buffer_rsrc_t rs[ND];
#pragma unroll
  for (int q = 0; q < ND; ++q) {
    const int tb = BL ? it.b0 + 2 * wave + q : it.b0 + 4 * wc + q;
    rs[q] = StreamRsrc(j.tiles + (size_t)tb * A * T, A * T * 4);
  }
  const uint32_t xbytes = (uint32_t)j.Xrows * j.I * 2;
  const auto xrs = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), xbytes);
  const int astep = j.Xrows * 32;
  uint32_t xo[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int row = it.m0 + wr * 64 + r * 16 + (lane & 15);
    xo[r] = row < j.M ? (uint32_t)row * 32 : xbytes;
  }
  uint32_t* w = reinterpret_cast<uint32_t*>(stage) + wave * kGemmWaveWords;
  const int o0 = win.d0 * 4, o1 = ((win.d0 + 1) % T) * 4, o2 = ((win.d0 + 2) % T) * 4;
  uint32_t raw[ND][3];
#pragma unroll
  for (int q = 0; q < ND; ++q) {
    raw[q][0] = __builtin_amdgcn_raw_buffer_load_b32(rs[q], o0, 0, 0);
    raw[q][1] = __builtin_amdgcn_raw_buffer_load_b32(rs[q], o1, 0, 0);
    raw[q][2] = K >= 5 ? __builtin_amdgcn_raw_buffer_load_b32(rs[q], o2, 0, 0) : 0;
  }
  GemmF8 acc[4][4] = {};
  for (int a = 0; a < A; ++a) {
    GemmH16 afr[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) afr[r] = GemmAFragment<false>(xrs, xo[r], a, nullptr, 0, astep);
    const int next = (a + 1) * T * 4;
    GemmH16 bf[ND];
#pragma unroll
    for (int q = 0; q < ND; ++q) {
      const uint32_t w0 = raw[q][0], w1 = raw[q][1], w2 = raw[q][2];
      raw[q][0] = __builtin_amdgcn_raw_buffer_load_b32(rs[q], o0 + next, 0, 0);
      raw[q][1] = __builtin_amdgcn_raw_buffer_load_b32(rs[q], o1 + next, 0, 0);
      if constexpr (K >= 5) raw[q][2] = __builtin_amdgcn_raw_buffer_load_b32(rs[q], o2 + next, 0, 0);
      __half2 p[4];
      DecodeLane<K, kMulTrue16>(w0, w1, w2, win.shift, p);
      bf[q] = GemmBFragment<true>(p, w, lane);
    }
    GemmH16 bb[4];
    if constexpr (BL) {
      const int cur = a & 1;
      if (lane < 16) {
#pragma unroll
        for (int q = 0; q < ND; ++q) bls[cur][2 * wave + q][lane] = bf[q];
      }
      __syncthreads();
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) bb[nb] = bls[cur][4 * wc + nb][lane & 15];
    } else {
#pragma unroll
      for (int nb = 0; nb < 4; ++nb) bb[nb] = bf[nb];
    }
#pragma unroll
    for (int nb = 0; nb < 4; ++nb)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        acc[nb][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[r], bb[nb], acc[nb][r]);
  }
  const int c = it.b0 * 16 + lane * 4;
  float svh[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) svh[i] = __half2float(L.svh[c + i]);
  for (int r8 = 0; r8 < 8; ++r8) {
    const int row0 = it.m0 + r8 * 16;
    if (row0 >= j.M) break;
    __syncthreads();
    if (wr == r8 >> 2) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r != (r8 & 3)) continue;
#pragma unroll
        for (int nb = 0; nb < 4; ++nb)
#pragma unroll
          for (int z = 0; z < 8; ++z)
            stage[(2 * z + (lane >> 4)) * kFusedPad + (4 * wc + nb) * 16 + (lane & 15)] = acc[nb][r][z];
      }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const int lr = wave * 4 + h, row = row0 + lr;
      if (row >= j.M) break;
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = 0.f + stage[lr * kFusedPad + lane * 4 + i];
      Wht128(v);
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] *= kHadNorm * svh[i];
      float* d = L.dst + (size_t)row * L.ldd + c;
#pragma unroll
      for (int i = 0; i < 4; ++i) d[i] = v[i];
    }
  }
}

// group > 0: row blocks of `group` items, all columns of a group before the next.
std::vector<Exl3GemmItem> Items(int M, int O, int rm, bool column_major, int group = 0) {
  std::vector<Exl3GemmItem> items;
  if (group > 0) {
    for (int r0 = 0; r0 < M; r0 += 16 * rm * group)
      for (int b = 0; b < O / 16; b += kFusedWaves)
        for (int r = r0; r < std::min(M, r0 + 16 * rm * group); r += 16 * rm) items.push_back({0, 0, r, b});
  } else if (column_major) {
    for (int b = 0; b < O / 16; b += kFusedWaves)
      for (int r = 0; r < M; r += 16 * rm) items.push_back({0, 0, r, b});
  } else {
    for (int r = 0; r < M; r += 16 * rm)
      for (int b = 0; b < O / 16; b += kFusedWaves) items.push_back({0, 0, r, b});
  }
  return items;
}

template <int K>
int Run(int M, int I, int O) {
  const int A = I / 16, B = O / 16;
  const size_t ntiles = (size_t)A * B;
  std::mt19937 rng(7);
  std::vector<uint32_t> ht(ntiles * 8 * K);
  for (auto& v : ht) v = rng();
  Buffer<uint32_t> tiles(ht);
  const auto hx = Half(Random((size_t)M * I, .1f, 11));
  Buffer<__half> x(hx);  // random values, so the packed order needs no permutation
  auto sv = Random(O, .05f, 13);
  for (auto& v : sv) v += 1.f;
  Buffer<__half> svh(Half(sv));
  Buffer<float> d1((size_t)M * O), d2((size_t)M * O);
  Buffer<__half> copy(ntiles * 256);
  LinJob lj{};
  lj.tiles = tiles.p; lj.svh = svh.p; lj.dst = d1.p; lj.I = I; lj.O = O; lj.K = K; lj.S = 1; lj.R = M;
  lj.ldd = O; lj.epi = kEpiStore;
  LinJob lh = lj; lh.dst = d2.p;
  Buffer<LinJob> lin1(std::vector<LinJob>{lj}), lin2(std::vector<LinJob>{lh});
  Buffer<Exl3GemmJob> job(std::vector<Exl3GemmJob>{{tiles.p, x.p, nullptr, nullptr, I, O, M, 1, K, M}});
  const double tflop = 2.0 * M * I * O * 1e-12;
  const auto rows8 = Items(M, O, 8, false);
  Buffer<Exl3GemmItem> it8(rows8);
  const int n8 = (int)rows8.size();
  const bool k6 = K == 6;
  auto exl3 = [&] { LaunchPrefillGemmOut(8, lin1.p, job.p, it8.p, n8, k6); };
  auto decode = [&] { DecodeCopyKernel<K><<<2048, 128>>>(tiles.p, copy.p, ntiles); };
  exl3(); decode(); CheckLaunch();
  const auto ref = d1.Get();
  const double te = Time(exl3), td = Time(decode);
  EnergyHook("dense", "exl3_fused_rm8", exl3);
  std::printf("M=%d I=%d O=%d K=%d: exl3 fused RM8 %.1f us (%.1f TFLOPS); decode copy %.1f us (%.1f MB)\n",
              M, I, O, K, te, tflop / (te * 1e-6), td, ntiles * 512 / 1e6);
  bool all = true;
  for (int g : {-1, 2, 4, 8}) {
    const auto items = Items(M, O, 8, g < 0, g < 0 ? 0 : g);
    Buffer<Exl3GemmItem> di(items);
    const int n = (int)items.size();
    auto go = [&] { LaunchPrefillGemmOut(8, lin2.p, job.p, di.p, n, k6); };
    d2.Zero(); go(); CheckLaunch();
    const bool same = d2.Get() == ref;
    all = all && same;
    const double t = Time(go);
    std::printf("  exl3 order %s%d bitwise=%d %.1f us (%.1f TFLOPS) = %.3f of row-major\n", g < 0 ? "col-major" : "group",
                g < 0 ? 0 : g, (int)same, t, tflop / (t * 1e-6), t / te);
  }
  auto half = [&](const char* name, int rm, bool cm, auto launch_fn) {
    const auto items = Items(M, O, rm, cm);
    Buffer<Exl3GemmItem> di(items);
    const int n = (int)items.size();
    auto go = [&] { launch_fn(di.p, n); };
    d2.Zero(); go(); CheckLaunch();
    const bool same = d2.Get() == ref;
    all = all && same;
    const double t = Time(go);
    std::printf("  half %-12s %s bitwise=%d %.1f us (%.1f TFLOPS), with decode %.1f us = %.3f of exl3\n",
                name, cm ? "col-major" : "row-major", (int)same, t, tflop / (t * 1e-6), t + td, (t + td) / te);
  };
#define HV(RM_, NB_)                                                                                    \
  for (bool cm : {false, true})                                                                         \
    half("rm" #RM_ " nb" #NB_, RM_, cm, [&](const Exl3GemmItem* di, int n) {                           \
      HalfOutKernel<RM_, NB_><<<n, 32 * kFusedWaves / NB_>>>(lin2.p, job.p, di, n, copy.p);            \
    });
  if (std::getenv("DH_HALF")) { HV(8, 2) HV(8, 1) HV(4, 2) HV(16, 1) }
  if (std::getenv("DH_X22")) {
    for (bool cm : {false, true}) {
      half("x22 direct", 8, cm, [&](const Exl3GemmItem* di, int n) {
        Exl3x22Kernel<K, false><<<n, 128>>>(lin2.p, job.p, di, n); });
      half("x22 B-lds", 8, cm, [&](const Exl3GemmItem* di, int n) {
        Exl3x22Kernel<K, true><<<n, 128>>>(lin2.p, job.p, di, n); });
    }
  }
  if (std::getenv("DH_22")) {
    HV(8, 2)
    for (bool cm : {false, true}) {
      half("22 direct", 8, cm, [&](const Exl3GemmItem* di, int n) {
        Half22Kernel<false><<<n, 128>>>(lin2.p, job.p, di, n, copy.p); });
      half("22 B-lds", 8, cm, [&](const Exl3GemmItem* di, int n) {
        Half22Kernel<true><<<n, 128>>>(lin2.p, job.p, di, n, copy.p); });
    }
  }
#undef HV
  return all ? 0 : 8;
}

int main(int argc, char** argv) {
  const int M = argc > 1 ? std::atoi(argv[1]) : 4096, K = argc > 2 ? std::atoi(argv[2]) : 5;
  const int I = argc > 3 ? std::atoi(argv[3]) : 2560, O = argc > 4 ? std::atoi(argv[4]) : 16384;
  if (M < 16 || M % 16 || I % 128 || O % 128) return 2;
  if (K == 3) return Run<3>(M, I, O);
  if (K == 4) return Run<4>(M, I, O);
  if (K == 5) return Run<5>(M, I, O);
  if (K == 6) return Run<6>(M, I, O);
  return 2;
}
