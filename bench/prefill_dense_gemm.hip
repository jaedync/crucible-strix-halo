// prefill-dense (from bench/gemm_out_next.hip): sustained timing of the dense prefill GEMMs under the
// package power limit, and new exact variants against HalfGemmOutKernel.
// Usage: prefill_dense_gemm sustain [R=8192] [ROUNDS=3] [SECONDS=1.5]
//
// prefill-gemm: the engine's fused dense prefill GEMM (LaunchPrefillGemmOut, K6, front tile
// order, the 4.05 pack) against the half path of engine/exl3/prefill_half.h (decode copy plus
// an LDS-staged fp16 GEMM), at the engine's shapes:
//   gdn_in  qkv (2560 -> 10240) and z (2560 -> 6144), store epilogue (and the GDN prep epilogue)
//   qsa_in  q (2560 -> 12288), k and v (2560 -> 512), store epilogue
//   qsa_idx q, k, v K6 and the indexer (2560 -> 640) K4: the engine runs it as two launches (split)
//   out     6144 -> 2560, residual epilogue into the 4-stream stack
// check: every output byte of the half path must equal the engine's at each R; the k-order
// mutant must differ. time: interleaved rounds; "cold" writes 256 MiB before each launch.
// Usage: gemm_out_next check [R ...]  |  gemm_out_next time [R=8192] [ROUNDS=3] [G=16]
#include "prefill_common.h"
#include "../engine/exl3/decode_split.h"
#include "../engine/layers/kernels/prefill_gdn.h"
#include "../engine/exl3/prefill_linear.h"
#include "../engine/exl3/prefill_half.h"
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
using namespace crucible;

namespace crucible {
// The trunk kernel (7a2d295e engine/exl3/prefill_half.h), verbatim: the reference for check and time.
template <int KC, class... EPI>
__global__ void __launch_bounds__(128) HalfGemmOutRef(const LinJob* __restrict__ lin,
    const HalfGemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n, EPI... epi) {
  static_assert(sizeof...(EPI) <= 1);
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  constexpr int kStage = KC * kHalfKBlock;
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
  U4 g[KC][4];
  auto fetch = [&](int a0) {
#pragma unroll
    for (int kc = 0; kc < KC; ++kc)
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        // The k offset goes in the lane offset: the range check covers it there (the last stage's
        // fetch past the end must read 0, not memory after the buffer).
        g[kc][q] = __builtin_amdgcn_raw_buffer_load_b128(xrs, xo[q] + (a0 + kc) * astep, 0, 0);
        g[kc][2 + q] = __builtin_amdgcn_raw_buffer_load_b128(wrs, wo[q] + (a0 + kc) * bstep, 0, 0);
      }
  };
  auto put = [&](int s) {
#pragma unroll
    for (int kc = 0; kc < KC; ++kc)
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        char* base = lds + s * kStage + kc * kHalfKBlock;
        *reinterpret_cast<U4*>(base + lo[q]) = g[kc][q];
        *reinterpret_cast<U4*>(base + 2 * kHalfPlane + lo[q]) = g[kc][2 + q];
      }
  };
  GemmF8 acc[4][4];
#pragma unroll
  for (int n4 = 0; n4 < 4; ++n4)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[n4][r] = GemmF8{};
  const int ar = wr * 64 + (lane & 15), bc = wc * 64 + (lane & 15);
  fetch(0);
  put(0);
  GemmGroupSync();
  const int stages = A / KC;
  for (int c = 0; c < stages; ++c) {
    const int s = c & 1;
    // The last stage fetches past the end (raw buffer loads return 0) and stores to the idle stage:
    // no branch, so the prefetch registers need no spill around it.
    fetch((c + 1) * KC);
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
      const char* base = lds + s * kStage + kc * kHalfKBlock;
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
#pragma unroll
      for (int n4 = 0; n4 < 4; ++n4)
#pragma unroll
        for (int r = 0; r < 4; ++r)
          acc[n4][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[r], bf[n4], acc[n4][r]);
      // One k block of fragments live at a time (the next block's LDS reads would need 64 more
      // VGPRs next to the 128 accumulators and the 32 of the global prefetch).
      asm volatile("" ::: "memory");
    }
    // The stores wait for the prefetch (vmcnt): keep them after every WMMA of the stage.
    __builtin_amdgcn_sched_barrier(0);
    put(s ^ 1);
    GemmGroupSync();  // LDS only: __syncthreads would also invalidate the L0 cache each stage
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
      if (row >= j.M) continue;
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = 0.f + stage[lr * kHalfStagePad + lane * 4 + i];
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

// The main-loop and epilogue variants of HalfGemmOutBody (engine/exl3/prefill_half.h).
// PP 0 puts the next stage's LDS stores after the WMMAs, 1 after half of them. MR: 128 only (the
// 256-row blocks of 8 waves were 2 to 4% slower).
template <int MR, int DG, int EP, int PP, class... EPI>
__global__ void __launch_bounds__(128) HalfGemmVar(const LinJob* __restrict__ lin,
    const HalfGemmJob* __restrict__ jobs, const Exl3GemmItem* __restrict__ items, int n, EPI... epi) {
  static_assert(MR == 128);
  HalfGemmOutBody<DG, EP, PP>(lin, jobs, items, n, epi...);
}
}  // namespace crucible
using namespace prefill_bench;

namespace {
constexpr int kK = 6;

Buffer<char>& Flush() { static Buffer<char> b((size_t)256 << 20); return b; }

template <class F> double TimeCold(F launch, int n) {
  launch(); CheckLaunch();
  std::vector<double> us;
  hipEvent_t a, b; HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
  for (int i = 0; i < n; ++i) {
    HIP_OK(hipMemsetAsync(Flush().p, i & 0xFF, Flush().n));
    HIP_OK(hipEventRecord(a)); launch(); HIP_OK(hipEventRecord(b));
    HIP_OK(hipEventSynchronize(b));
    float ms = 0; HIP_OK(hipEventElapsedTime(&ms, a, b)); us.push_back(ms * 1000.);
  }
  HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

std::vector<uint32_t> RandomWords(size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<uint32_t> w(n);
  for (auto& v : w) v = rng();
  return w;
}

struct Mat {
  int I, O, S, K;
  std::unique_ptr<Buffer<uint32_t>> tiles;
  std::unique_ptr<Buffer<__half>> svh, copy, copym;
  std::unique_ptr<Buffer<float>> dst, dst2;
};

// One fused group: jobs share x (packed [I/16][R][16]) and R; each has its own matrix and dst.
struct Case {
  std::string name;
  int R, I, epi;
  std::vector<Mat> m;
  std::unique_ptr<Buffer<__half>> x, xm;  // xm: x with k blocks 2i and 2i + 1 swapped (the mutant)
  std::unique_ptr<Buffer<float>> aux;
  std::vector<float> h0;  // residual stack start
  std::unique_ptr<Buffer<LinJob>> lin, lin2;
  std::unique_ptr<Buffer<Exl3GemmJob>> ej;
  std::unique_ptr<Buffer<HalfGemmJob>> hj, hjm;
  std::unique_ptr<Buffer<Exl3GemmItem>> eit, hit;
  int ecount = 0, hcount = 0, count6 = 0;  // count6: the engine's K6 items come first
  double flop = 0;

  // ks: the EXL3 width of each job (default kK for all).
  Case(std::string n, int R_, int I_, const std::vector<int>& outs, int epi_, int G, std::vector<int> ks = {})
      : name(n), R(R_), I(I_), epi(epi_) {
    if (ks.empty()) ks.assign(outs.size(), kK);
    std::vector<std::pair<int, int>> io;
    for (int O : outs) io.push_back({I, O});
    const std::vector<int> splits = DecodeSplits(io);
    x = std::make_unique<Buffer<__half>>(Half(Random((size_t)R * I, .1f, 11)));
    xm = std::make_unique<Buffer<__half>>(SwapK(x->Get(), I / 16, (size_t)R * 16));
    if (epi == kEpiResidual) {
      aux = std::make_unique<Buffer<float>>(Random((size_t)R * kStreams, .5f, 19));
      h0 = Random((size_t)R * kWide, 1.f, 17);
    }
    std::vector<LinJob> l1, l2;
    std::vector<Exl3GemmJob> e;
    std::vector<HalfGemmJob> h, hm;
    std::vector<Exl3GemmItem> items;
    for (size_t k = 0; k < outs.size(); ++k) {
      const int O = outs[k];
      const int K = ks[k];
      Mat mt{I, O, splits[k], K};
      mt.tiles = std::make_unique<Buffer<uint32_t>>(RandomWords((size_t)I / 16 * (O / 16) * 8 * K, 7 + k));
      mt.svh = std::make_unique<Buffer<__half>>(Half(Random(O, .05f, 13 + k)));
      mt.copy = std::make_unique<Buffer<__half>>((size_t)I * O);
      mt.copym = std::make_unique<Buffer<__half>>((size_t)I * O);
      const size_t dn = epi == kEpiStore ? (size_t)R * O : (size_t)R * kWide;
      mt.dst = std::make_unique<Buffer<float>>(dn);
      mt.dst2 = std::make_unique<Buffer<float>>(dn);
      LinJob j{};
      j.tiles = mt.tiles->p; j.svh = mt.svh->p; j.aux = aux ? aux->p : nullptr; j.I = I; j.O = O; j.K = K;
      j.S = 1; j.R = R; j.ldd = epi == kEpiStore ? O : kWide; j.epi = epi;
      j.dst = mt.dst->p; l1.push_back(j);
      j.dst = mt.dst2->p; l2.push_back(j);
      e.push_back({mt.tiles->p, x->p, nullptr, nullptr, I, O, R, 1, K, R, mt.S});
      h.push_back({x->p, mt.copy->p, I, O, R, R});
      hm.push_back({xm->p, mt.copym->p, I, O, R, R});
      for (int r = 0; r < R; r += 128)
        for (int b = 0; b < O / 16; b += kFusedWaves) items.push_back({(int)k, 0, r, b});
      flop += 2.0 * R * I * O;
      m.push_back(std::move(mt));
    }
    // Engine order (PrefillLinear::Build): jobs in order, rows, then column blocks; K6 jobs first.
    std::vector<Exl3GemmItem> eo;
    for (int six : {1, 0})
      for (size_t k = 0; k < outs.size(); ++k)
        for (const auto& v : items) if (v.job == (int)k && (ks[k] == 6) == (six == 1)) eo.push_back(v);
    for (const auto& v : eo) count6 += ks[v.job] == 6;
    lin = std::make_unique<Buffer<LinJob>>(l1);
    lin2 = std::make_unique<Buffer<LinJob>>(l2);
    ej = std::make_unique<Buffer<Exl3GemmJob>>(e);
    hj = std::make_unique<Buffer<HalfGemmJob>>(h);
    hjm = std::make_unique<Buffer<HalfGemmJob>>(hm);
    eit = std::make_unique<Buffer<Exl3GemmItem>>(eo);
    const auto ho = HalfItems(h, G);
    hit = std::make_unique<Buffer<Exl3GemmItem>>(ho);
    ecount = (int)eo.size(); hcount = (int)ho.size();
    Reset();
  }
  void Reset() {
    if (epi != kEpiResidual) return;
    for (auto& mt : m) { mt.dst->Set(h0); mt.dst2->Set(h0); }
  }
  template <class... EPI> void Engine(const EPI&... epi) const {
    if (count6 == ecount) { LaunchPrefillGemmOut(8, lin->p, ej->p, eit->p, ecount, true, true, epi...); return; }
    LaunchPrefillGemmOut(8, lin->p, ej->p, eit->p, count6, true, true, epi...);
    LaunchPrefillGemmOut(8, lin->p, ej->p, eit->p + count6, ecount - count6, false, true, epi...);
  }
  void Copy() const {
    for (const auto& mt : m) LaunchHalfCopy(mt.K, mt.S, mt.tiles->p, mt.copy->p, mt.I, mt.O);
  }
  // The mutant's copy: the decode copy with the same k block swap as xm. Every output then sums the
  // same products in another order, so its bytes must differ.
  void MutantCopy() const {
    Copy(); CheckLaunch();
    for (const auto& mt : m) mt.copym->Set(SwapK(mt.copy->Get(), mt.I / 16, (size_t)mt.O * 16));
  }
  template <int KC, class... EPI> void HalfRun(const EPI&... epi) const {
    HalfGemmOutKernel<KC, EPI...><<<hcount, 128>>>(lin2->p, hj->p, hit->p, hcount, epi...);
  }
  // Ref writes dst (lin), the variants dst2 (lin2).
  template <class... EPI> void RefRun(const EPI&... epi) const {
    HalfGemmOutRef<1, EPI...><<<hcount, 128>>>(lin->p, hj->p, hit->p, hcount, epi...);
  }
  template <int MR, int DG, int EP, int PP, class... EPI> void VarRun(bool mutant, const EPI&... epi) const {
    HalfGemmVar<MR, DG, EP, PP, EPI...><<<hcount, 128>>>(lin2->p, mutant ? hjm->p : hj->p, hit->p, hcount, epi...);
  }
  void MutantRun() const { HalfGemmOutKernel<kHalfKC><<<hcount, 128>>>(lin2->p, hjm->p, hit->p, hcount); }
  // Swaps k blocks 2i and 2i + 1 of a [blocks][per] array.
  static std::vector<__half> SwapK(std::vector<__half> v, int blocks, size_t per) {
    for (int a = 0; a + 1 < blocks; a += 2)
      std::swap_ranges(v.begin() + a * per, v.begin() + (a + 1) * per, v.begin() + (a + 1) * per);
    return v;
  }
};

bool Same(const Buffer<float>& a, const Buffer<float>& b) {
  const auto x = a.Get(), y = b.Get();
  return std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}

void CheckCase(Case& c, int& fails) {
  c.Reset();
  c.Engine(); c.Copy(); c.HalfRun<kHalfKC>(); CheckLaunch();
  bool same = true;
  for (auto& mt : c.m) same = same && Same(*mt.dst, *mt.dst2);
  c.MutantCopy();
  c.Reset();
  c.Engine(); c.MutantRun(); CheckLaunch();
  bool mut_same = true;
  for (auto& mt : c.m) mut_same = mut_same && Same(*mt.dst, *mt.dst2);
  std::printf("check %-8s R=%-6d half %s, mutant %s\n", c.name.c_str(), c.R, same ? "SAME" : "DIFF",
              mut_same ? "SAME" : "DIFF");
  fails += !same + mut_same;
}

// The GDN input group with the GDN prep epilogue (q, k parts of 2,048 rows).
void CheckGdnEpi(int R, int& fails) {
  Case c("gdn_epi", R, kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore, 16);
  const size_t parts = (R + 2047) / 2048;
  Buffer<float> q1(parts * prefill::kGdnPrepPartFloats), k1(parts * prefill::kGdnPrepPartFloats);
  Buffer<float> q2(parts * prefill::kGdnPrepPartFloats), k2(parts * prefill::kGdnPrepPartFloats);
  Buffer<float> conv(Random((size_t)kGdnQkvWidth * 4, .3f, 23));
  q1.Nan(); k1.Nan(); q2.Nan(); k2.Nan();
  c.Engine(prefill::GdnPrepEpi{c.m[0].dst->p, q1.p, k1.p, conv.p, R}); CheckLaunch();
  c.Copy(); c.HalfRun<kHalfKC>(prefill::GdnPrepEpi{c.m[0].dst2->p, q2.p, k2.p, conv.p, R}); CheckLaunch();
  bool same = Same(*c.m[0].dst, *c.m[0].dst2) && Same(*c.m[1].dst, *c.m[1].dst2);
  const auto a = q1.Get(), b = q2.Get(), ka = k1.Get(), kb = k2.Get();
  same = same && !std::memcmp(a.data(), b.data(), a.size() * 4) && !std::memcmp(ka.data(), kb.data(), ka.size() * 4);
  std::printf("check gdn_epi  R=%-6d half %s\n", R, same ? "SAME" : "DIFF");
  fails += !same;
}


// Knockout copies of the main loop (NOT exact, timing only): KO bit 0 skips the global loads, bit 1
// skips the LDS fragment reads in the loop (fragments read once), bit 2 skips the per-stage barrier,
// bit 3 skips the LDS stores. The epilogue writes the raw accumulators.
template <int KO>
__global__ void __launch_bounds__(128) HalfGemmKO(const HalfGemmJob* __restrict__ jobs,
    const Exl3GemmItem* __restrict__ items, int n, float* __restrict__ out) {
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  constexpr int kStage = kHalfKBlock;
  __shared__ __attribute__((aligned(16))) char lds[2 * kStage];
  if ((int)blockIdx.x >= n) return;
  const Exl3GemmItem it = items[blockIdx.x];
  const HalfGemmJob j = jobs[it.job];
  const int tid = threadIdx.x, lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5), wr = wave >> 1, wc = wave & 1;
  const int A = j.I / 16, c0 = it.b0 * 16;
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
  U4 g[4];
  auto fetch = [&](int a0) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      g[q] = __builtin_amdgcn_raw_buffer_load_b128(xrs, xo[q] + a0 * astep, 0, 0);
      g[2 + q] = __builtin_amdgcn_raw_buffer_load_b128(wrs, wo[q] + a0 * bstep, 0, 0);
    }
  };
  auto put = [&](int s) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      char* base = lds + s * kStage;
      *reinterpret_cast<U4*>(base + lo[q]) = g[q];
      *reinterpret_cast<U4*>(base + 2 * kHalfPlane + lo[q]) = g[2 + q];
    }
  };
  GemmF8 acc[4][4];
#pragma unroll
  for (int n4 = 0; n4 < 4; ++n4)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[n4][r] = GemmF8{};
  const int ar = wr * 64 + (lane & 15), bc = wc * 64 + (lane & 15);
  fetch(0);
  put(0);
  GemmGroupSync();
  GemmH16 af[4], bf[4];
  auto frags = [&](const char* base) {
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
  };
  frags(lds);
  for (int c = 0; c < A; ++c) {
    const int s = c & 1;
    if constexpr (!(KO & 1)) fetch(c + 1);
    if constexpr (!(KO & 2)) frags(lds + s * kStage);
#pragma unroll
    for (int n4 = 0; n4 < 4; ++n4)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        acc[n4][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[r], bf[n4], acc[n4][r]);
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (!(KO & 8)) put(s ^ 1);
    if constexpr (!(KO & 4)) GemmGroupSync();
  }
  float* o = out + (size_t)blockIdx.x * 128 * 128 + tid * 8;
#pragma unroll
  for (int n4 = 0; n4 < 4; ++n4)
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int z = 0; z < 8; ++z) o[(n4 * 4 + r) * 128 * 8 * 0 + z] += acc[n4][r][z];
}

// Sustained timing: each variant runs back to back for about `sec` seconds, so the chip reaches the
// package power limit as in a long prefill. Prints us per call, TFLOPS, mean sclk and power, and the
// energy per call (mean power x time per call).
template <class F> void Sustain(const char* shape, const std::string& name, double flop, F go, double sec, int round) {
  go(); CheckLaunch(); HIP_OK(hipDeviceSynchronize());
  hipEvent_t a, b; HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
  HIP_OK(hipEventRecord(a)); go(); HIP_OK(hipEventRecord(b)); HIP_OK(hipEventSynchronize(b));
  float one = 0; HIP_OK(hipEventElapsedTime(&one, a, b));
  const int n = std::max(3, (int)(sec * 1000. / std::max(one, 0.01f)));
  for (int i = 0; i < n / 4; ++i) go();  // ramp to the power limit
  HIP_OK(hipDeviceSynchronize());
  bench::Sampler s(10);
  s.Start();
  HIP_OK(hipEventRecord(a));
  for (int i = 0; i < n; ++i) go();
  HIP_OK(hipEventRecord(b)); HIP_OK(hipEventSynchronize(b));
  s.Stop();
  float ms = 0; HIP_OK(hipEventElapsedTime(&ms, a, b));
  const double us = ms * 1000. / n;
  std::printf("sustain round %d %-8s %-22s %9.1f us %5.1f TF n=%-4d %s\n", round, shape, name.c_str(), us,
              flop / (us * 1e6), n, s.Summary().c_str());
  std::fflush(stdout);
  HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
}

// Variants of HalfGemmOutBody: every output byte equal to the trunk kernel (HalfGemmOutRef) at each R; the
// k-order mutant (x and the copy with k blocks 2i and 2i + 1 swapped) must differ.
template <int MR, int DG, int EP, int PP> void VarCheck(Case& c, int& fails, const char* tag) {
  c.Copy(); c.MutantCopy(); c.Reset();
  c.RefRun(); c.VarRun<MR, DG, EP, PP>(false); CheckLaunch();
  bool same = true;
  for (auto& mt : c.m) same = same && Same(*mt.dst, *mt.dst2);
  c.Reset();
  c.RefRun(); c.VarRun<MR, DG, EP, PP>(true); CheckLaunch();
  bool mut = true;
  for (auto& mt : c.m) mut = mut && Same(*mt.dst, *mt.dst2);
  std::printf("vcheck %-8s R=%-6d %-6s %s, mutant %s\n", c.name.c_str(), c.R, tag, same ? "SAME" : "DIFF", mut ? "SAME" : "DIFF");
  fails += !same + mut;
}
template <int MR, int DG, int EP, int PP> void VarCheckEpi(int R, int& fails, const char* tag) {
  Case c("gdn_epi", R, kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore, 16);
  const size_t parts = (R + 2047) / 2048;
  Buffer<float> q1(parts * prefill::kGdnPrepPartFloats), k1(parts * prefill::kGdnPrepPartFloats);
  Buffer<float> q2(parts * prefill::kGdnPrepPartFloats), k2(parts * prefill::kGdnPrepPartFloats);
  Buffer<float> conv(Random((size_t)kGdnQkvWidth * 4, .3f, 23));
  q1.Nan(); k1.Nan(); q2.Nan(); k2.Nan();
  c.Copy();
  c.RefRun(prefill::GdnPrepEpi{c.m[0].dst->p, q1.p, k1.p, conv.p, R});
  c.VarRun<MR, DG, EP, PP>(false, prefill::GdnPrepEpi{c.m[0].dst2->p, q2.p, k2.p, conv.p, R}); CheckLaunch();
  bool same = Same(*c.m[0].dst, *c.m[0].dst2) && Same(*c.m[1].dst, *c.m[1].dst2);
  const auto a = q1.Get(), b = q2.Get(), ka = k1.Get(), kb = k2.Get();
  // The reference (128-row tiles) leaves the first 3 rows of each tile to the halo kernel (NaN here); a
  // 256-row variant writes some of them. Every value that the reference wrote must be equal bit for bit,
  // and the variant may leave out only rows that the reference left out.
  auto eq = [](const std::vector<float>& r, const std::vector<float>& v) {
    for (size_t i = 0; i < r.size(); ++i) {
      uint32_t x, y;
      std::memcpy(&x, &r[i], 4); std::memcpy(&y, &v[i], 4);
      if (std::isnan(r[i]) ? (std::isnan(v[i]) && x != y) : x != y) return false;
    }
    return true;
  };
  same = same && eq(a, b) && eq(ka, kb);
  std::printf("vcheck gdn_epi  R=%-6d %-6s %s\n", R, tag, same ? "SAME" : "DIFF");
  fails += !same;
}
#define PD_VARIANTS(X) X(128, 1, 0, 0) X(128, 1, 1, 0) X(128, 2, 0, 0) X(128, 2, 1, 0) X(128, 2, 1, 1)
int VarCheckMain(int argc, char** argv) {
  std::vector<int> rows;
  for (int i = 2; i < argc; ++i) rows.push_back(std::atoi(argv[i]));
  if (rows.empty()) rows = {16, 97, 4096, 8192};
  int fails = 0;
  for (int R : rows) {
    Case gdn("gdn_in", R, kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore, 16);
    Case qsa("qsa_idx", R, kHidden, {kQsaQWidth, kQsaKvWidth, kQsaKvWidth, kQsaIdxWidth}, kEpiStore, 16, {6, 6, 6, 4});
    Case out("out", R, kGdnValueWidth, {kHidden}, kEpiResidual, 8);
#define PD_CHECK(a, b, e, f) VarCheck<a, b, e, f>(gdn, fails, #a "_" #b #e #f); VarCheck<a, b, e, f>(qsa, fails, #a "_" #b #e #f); \
    VarCheck<a, b, e, f>(out, fails, #a "_" #b #e #f); VarCheckEpi<a, b, e, f>(R, fails, #a "_" #b #e #f);
    PD_VARIANTS(PD_CHECK)
  }
  std::printf("RESULT vcheck fails=%d\n", fails);
  return fails ? 8 : 0;
}

// Item group sizes for the engine variant (kHalfDepth, kHalfEpi, kHalfPut).
int VarGroupMain(int argc, char** argv) {
  const int R = argc > 2 ? std::atoi(argv[2]) : 8192;
  const double sec = argc > 3 ? std::atof(argv[3]) : 1.0;
  for (int round = 1; round <= 2; ++round)
    for (int G : {4, 8, 16, 32}) {
      const std::string tag = "G" + std::to_string(G);
      {
        Case gdn("gdn_in", R, kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore, G);
        gdn.Copy(); CheckLaunch();
        Sustain("gdn_in", tag.c_str(), gdn.flop, [&] { gdn.VarRun<128, kHalfDepth, kHalfEpi, kHalfPut>(false); }, sec, round);
      }
      {
        Case qsa("qsa_idx", R, kHidden, {kQsaQWidth, kQsaKvWidth, kQsaKvWidth, kQsaIdxWidth}, kEpiStore, G, {6, 6, 6, 4});
        qsa.Copy(); CheckLaunch();
        Sustain("qsa_in", tag.c_str(), qsa.flop, [&] { qsa.VarRun<128, kHalfDepth, kHalfEpi, kHalfPut>(false); }, sec, round);
      }
      {
        Case out("out", R, kGdnValueWidth, {kHidden}, kEpiResidual, G);
        out.Copy(); CheckLaunch();
        Sustain("out", tag.c_str(), out.flop, [&] { out.VarRun<128, kHalfDepth, kHalfEpi, kHalfPut>(false); }, sec, round);
      }
    }
  return 0;
}

int VarTimeMain(int argc, char** argv) {
  const int R = argc > 2 ? std::atoi(argv[2]) : 8192, rounds = argc > 3 ? std::atoi(argv[3]) : 2;
  const double sec = argc > 4 ? std::atof(argv[4]) : 1.0;
  Case gdn("gdn_in", R, kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore, 16);
  Case qsa("qsa_idx", R, kHidden, {kQsaQWidth, kQsaKvWidth, kQsaKvWidth, kQsaIdxWidth}, kEpiStore, 16, {6, 6, 6, 4});
  Case out("out", R, kGdnValueWidth, {kHidden}, kEpiResidual, 8);
  const size_t parts = (R + 2047) / 2048;
  Buffer<float> q1(parts * prefill::kGdnPrepPartFloats), k1(parts * prefill::kGdnPrepPartFloats);
  Buffer<float> conv(Random((size_t)kGdnQkvWidth * 4, .3f, 23));
  const prefill::GdnPrepEpi epi{gdn.m[0].dst2->p, q1.p, k1.p, conv.p, R};
  gdn.Copy(); qsa.Copy(); out.Copy(); CheckLaunch();
  for (int round = 1; round <= rounds; ++round) {
    Sustain("gdn_epi", "ref", gdn.flop, [&] { HalfGemmOutRef<1, prefill::GdnPrepEpi><<<gdn.hcount, 128>>>(gdn.lin2->p, gdn.hj->p, gdn.hit->p, gdn.hcount, epi); }, sec, round);
#define PD_T1(a, b, e, f) Sustain("gdn_epi", #a "_" #b #e #f, gdn.flop, [&] { gdn.VarRun<a, b, e, f>(false, epi); }, sec, round);
    PD_VARIANTS(PD_T1)
    Sustain("qsa_in", "ref", qsa.flop, [&] { qsa.RefRun(); }, sec, round);
#define PD_T2(a, b, e, f) Sustain("qsa_in", #a "_" #b #e #f, qsa.flop, [&] { qsa.VarRun<a, b, e, f>(false); }, sec, round);
    PD_VARIANTS(PD_T2)
    Sustain("out", "ref", out.flop, [&] { out.RefRun(); }, sec, round);
#define PD_T3(a, b, e, f) Sustain("out", #a "_" #b #e #f, out.flop, [&] { out.VarRun<a, b, e, f>(false); }, sec, round);
    PD_VARIANTS(PD_T3)
  }
  return 0;
}

// Item group sizes (row blocks per group, HalfItems): same outputs, other L2 and MALL reuse.
int GroupMain(int argc, char** argv) {
  const int R = argc > 2 ? std::atoi(argv[2]) : 8192, rounds = argc > 3 ? std::atoi(argv[3]) : 2;
  const double sec = argc > 4 ? std::atof(argv[4]) : 1.0;
  struct Sh { const char* n; int I; std::vector<int> outs; int epi; std::vector<int> gs; };
  const std::vector<Sh> shapes{{"gdn_in", kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore, {8, 16, 32, 64}},
                               {"qsa_in", kHidden, {kQsaQWidth, kQsaKvWidth, kQsaKvWidth}, kEpiStore, {16, 32}},
                               {"out", kGdnValueWidth, {kHidden}, kEpiResidual, {4, 8, 16, 32}}};
  for (const auto& sh : shapes) {
    std::vector<std::unique_ptr<Case>> cs;
    for (int G : sh.gs) { cs.push_back(std::make_unique<Case>(sh.n, R, sh.I, sh.outs, sh.epi, G)); cs.back()->Copy(); }
    CheckLaunch();
    for (int round = 1; round <= rounds; ++round)
      for (size_t i = 0; i < cs.size(); ++i) {
        Case* c = cs[i].get();
        Sustain(sh.n, "half_G" + std::to_string(sh.gs[i]), c->flop, [c] { c->HalfRun<1>(); }, sec, round);
      }
  }
  return 0;
}

int KoMain(int argc, char** argv) {
  const int R = argc > 2 ? std::atoi(argv[2]) : 8192, rounds = argc > 3 ? std::atoi(argv[3]) : 2;
  const double sec = argc > 4 ? std::atof(argv[4]) : 1.0;
  Case gdn("gdn_in", R, kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore, 16);
  gdn.Copy(); CheckLaunch();
  Buffer<float> out((size_t)gdn.hcount * 128 * 128);
  for (int round = 1; round <= rounds; ++round) {
    Sustain("gdn_in", "ref_store", gdn.flop, [&] { gdn.RefRun(); }, sec, round);
    Sustain("gdn_in", "ko0_full", gdn.flop, [&] { HalfGemmKO<0><<<gdn.hcount, 128>>>(gdn.hj->p, gdn.hit->p, gdn.hcount, out.p); }, sec, round);
    Sustain("gdn_in", "ko1_noglobal", gdn.flop, [&] { HalfGemmKO<1><<<gdn.hcount, 128>>>(gdn.hj->p, gdn.hit->p, gdn.hcount, out.p); }, sec, round);
    Sustain("gdn_in", "ko2_nofrag", gdn.flop, [&] { HalfGemmKO<2><<<gdn.hcount, 128>>>(gdn.hj->p, gdn.hit->p, gdn.hcount, out.p); }, sec, round);
    Sustain("gdn_in", "ko4_nobar", gdn.flop, [&] { HalfGemmKO<4><<<gdn.hcount, 128>>>(gdn.hj->p, gdn.hit->p, gdn.hcount, out.p); }, sec, round);
    Sustain("gdn_in", "ko9_noglob_nostore", gdn.flop, [&] { HalfGemmKO<9><<<gdn.hcount, 128>>>(gdn.hj->p, gdn.hit->p, gdn.hcount, out.p); }, sec, round);
    Sustain("gdn_in", "ko15_wmma_only", gdn.flop, [&] { HalfGemmKO<15><<<gdn.hcount, 128>>>(gdn.hj->p, gdn.hit->p, gdn.hcount, out.p); }, sec, round);
  }
  return 0;
}

int SustainMain(int argc, char** argv) {
  const int R = argc > 2 ? std::atoi(argv[2]) : 8192, rounds = argc > 3 ? std::atoi(argv[3]) : 3;
  const double sec = argc > 4 ? std::atof(argv[4]) : 1.5;
  Case gdn("gdn_in", R, kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore, 16);
  Case qsa("qsa_in", R, kHidden, {kQsaQWidth, kQsaKvWidth, kQsaKvWidth}, kEpiStore, 16);
  Case out("out", R, kGdnValueWidth, {kHidden}, kEpiResidual, 8);
  Case outs("out_store", R, kGdnValueWidth, {kHidden}, kEpiStore, 8);
  const size_t parts = (R + 2047) / 2048;
  Buffer<float> q1(parts * prefill::kGdnPrepPartFloats), k1(parts * prefill::kGdnPrepPartFloats);
  Buffer<float> conv(Random((size_t)kGdnQkvWidth * 4, .3f, 23));
  const prefill::GdnPrepEpi epi{gdn.m[0].dst2->p, q1.p, k1.p, conv.p, R};
  gdn.Copy(); qsa.Copy(); out.Copy(); outs.Copy(); CheckLaunch();
  for (int round = 1; round <= rounds; ++round) {
    Sustain("gdn_epi", "ref", gdn.flop, [&] { gdn.RefRun(epi); }, sec, round);
    Sustain("gdn_epi", "half", gdn.flop, [&] { gdn.HalfRun<1>(epi); }, sec, round);
    Sustain("gdn_in", "ref", gdn.flop, [&] { gdn.RefRun(); }, sec, round);
    Sustain("gdn_in", "half", gdn.flop, [&] { gdn.HalfRun<1>(); }, sec, round);
    Sustain("qsa_in", "ref", qsa.flop, [&] { qsa.RefRun(); }, sec, round);
    Sustain("qsa_in", "half", qsa.flop, [&] { qsa.HalfRun<1>(); }, sec, round);
    Sustain("out", "ref", out.flop, [&] { out.RefRun(); }, sec, round);
    Sustain("out", "half", out.flop, [&] { out.HalfRun<1>(); }, sec, round);
    Sustain("out_store", "ref", outs.flop, [&] { outs.RefRun(); }, sec, round);
    Sustain("out_store", "half", outs.flop, [&] { outs.HalfRun<1>(); }, sec, round);
    Sustain("gdn_in", "copy", gdn.flop, [&] { gdn.Copy(); }, sec, round);
  }
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "check";
  if (mode == "sustain") return SustainMain(argc, argv);
  if (mode == "group") return GroupMain(argc, argv);
  if (mode == "ko") return KoMain(argc, argv);
  if (mode == "vcheck") return VarCheckMain(argc, argv);
  if (mode == "vtime") return VarTimeMain(argc, argv);
  if (mode == "vgroup") return VarGroupMain(argc, argv);
  int fails = 0;
  if (mode == "check") {
    std::vector<int> rows;
    for (int i = 2; i < argc; ++i) rows.push_back(std::atoi(argv[i]));
    if (rows.empty()) rows = {16, 97, 4096, 8192, 32768};
    for (int R : rows) {
      { Case c("gdn_in", R, kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore, 16); CheckCase(c, fails); }
      { Case c("qsa_in", R, kHidden, {kQsaQWidth, kQsaKvWidth, kQsaKvWidth}, kEpiStore, 16); CheckCase(c, fails); }
      { Case c("qsa_idx", R, kHidden, {kQsaQWidth, kQsaKvWidth, kQsaKvWidth, kQsaIdxWidth}, kEpiStore, 16, {6, 6, 6, 4});
        CheckCase(c, fails); }
      { Case c("out", R, kGdnValueWidth, {kHidden}, kEpiResidual, 16); CheckCase(c, fails); }
      CheckGdnEpi(R, fails);
    }
    std::printf("RESULT check fails=%d\n", fails);
    return fails ? 8 : 0;
  }
  const int R = argc > 2 ? std::atoi(argv[2]) : 8192, rounds = argc > 3 ? std::atoi(argv[3]) : 3;
  std::vector<int> gs;
  for (int i = 4; i < argc; ++i) gs.push_back(std::atoi(argv[i]));
  if (gs.empty()) gs = {16};
  struct Shape { const char* n; int I; std::vector<int> outs; int epi; };
  const std::vector<Shape> shapes{{"gdn_in", kHidden, {kGdnQkvWidth, kGdnValueWidth}, kEpiStore},
                                  {"qsa_in", kHidden, {kQsaQWidth, kQsaKvWidth, kQsaKvWidth}, kEpiStore},
                                  {"out", kGdnValueWidth, {kHidden}, kEpiResidual}};
  for (const auto& sh : shapes) {
    std::vector<std::unique_ptr<Case>> cs;
    for (int G : gs) cs.push_back(std::make_unique<Case>(sh.n, R, sh.I, sh.outs, sh.epi, G));
    for (int round = 1; round <= rounds; ++round) {
      struct V { std::string n; std::function<void()> go; };
      std::vector<V> vs{{"engine", [&] { cs[0]->Engine(); }}, {"copy", [&] { cs[0]->Copy(); }}};
      for (size_t i = 0; i < gs.size(); ++i) {
        Case* c = cs[i].get();
        const std::string g = "G" + std::to_string(gs[i]);
        vs.push_back({"half1_" + g, [c] { c->HalfRun<1>(); }});
      }
      for (auto& v : vs) {
        bench::Sampler sclk(5);
        sclk.Start(); const double t = TimeCold(v.go, 5); sclk.Stop();
        std::printf("R=%d round %d %-7s %-12s %9.1f us %5.1f TF %4.0f MHz\n", R, round, sh.n, v.n.c_str(), t,
                    cs[0]->flop / (t * 1e6), sclk.MeanMhz());
      }
    }
  }
  return 0;
}
