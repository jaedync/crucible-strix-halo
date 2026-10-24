// GPU test of the K7, K8 and K3.5 EXL3 paths (the h6_ng8 packs) against an independent CPU oracle on
// random tiles (CpuWeight: the circular-bit codec, as in bench/exl3_wide_codecs.hip; K3.5, code 35:
// a port of exllamav3 frac.cu, MASK 0xAAAA, as in tests/lane_window_test.cpp):
//   1. DecodeLane: every weight of every tile, bitwise, K2..K8 and K3.5.
//   2. Decode GEMV: LinGemvKernel and LinPolicyKernel with MaxK 8 (K7, K8, and K3/K6 through the
//      MaxK 8 dispatch), R 1/2/4/8, split S 1/2, cache policy 0/4; fp64 reference.
//   3. Prefill dense GEMM: PrefillWideBody K7/K8, RM 2/4/8; fp64 reference.
//   4. Prefill MoE GEMM: PrefillMoeGemmKernel<RM, packed, KMAX 7|8> with K4, K5 and KMAX jobs in one
//      launch (4.18: routed K4/K5 with a K8 shared expert), and KMAX kPrefillMoeK7Half with K3, K3.5
//      and K7 jobs (3.18, 3.68); fp64 reference. The fp16 form (HALF) bit-equal to the fp16 of the
//      fp32 form; the kPrefillMoeK7Half K3 and K7 jobs bit-equal to the KMAX 7 kernel.
//   5. Decode MoE bodies of MoeFusedKernelWide<R, KR, 7> (MoeAnyBody, MoeAnyBodyChunked) for the
//      3.18/3.68 pairs K3+7, K3.5+7, K4+7: routed rows 1..R at KR, shared rows R at K7, items of 16
//      and 8 column blocks, cache policy 0/4, full-span and 40-tile chunked x staging; fp64
//      reference. A mutant (one flipped tile word) must fail.
// Exit 0 and "packs_wide_test: PASS" when every case passes.
#include "../bench/prefill_common.h"
#include "../engine/layers/prefill_moe.h"
#include "../engine/layers/moe_rows.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

using namespace crucible;
using namespace prefill_bench;

namespace {

// exllamav3 frac.cu, MASK 0xAAAA: the ring bit where position pos's 16-bit window ends (K = KA + 0.5).
int FracEnd(int pos, int ka) {
  int s = (pos >> 4) * (16 * ka + 8);
  for (int j = 0; j <= (pos & 15); ++j) s += ka + ((0xAAAAu >> j) & 1);
  return s;
}

// Weight (input i, output o) of a matrix of I/16 x O/16 tiles in engine order (column-block-major).
// K is a width code: 2..8, or 35 for K3.5.
float CpuWeight(const std::vector<uint32_t>& tiles, int I, int K, int i, int o) {
  const int r = i & 15, c = o & 15, lane = 4 * (c & 7) + (r & 7) / 2;
  const int pos = 8 * lane + (r & 1) + ((r & 8) ? 2 : 0) + ((c & 8) ? 4 : 0), T = TileDwords(K), bits = 32 * T;
  const auto* t = tiles.data() + ((size_t)(o / 16) * (I / 16) + i / 16) * T;
  const int end = IsHalfK(K) ? FracEnd(pos, K / 10) : (pos + 1) * K;
  uint32_t state = 0;
  for (int z = 0; z < 16; ++z) {
    const int bit = (end - 16 + z + bits) % bits;
    state = (state << 1) | ((t[bit / 32] >> (31 - bit % 32)) & 1);
  }
  const uint32_t p = state * kMul1;
  const int sum = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24);
  return __half2float(__float2half((float)((1024. + sum) * 0.00676727294921875 - 10.3828125)));
}

struct Mat {
  int I, O, K;
  std::vector<uint32_t> tiles;
  std::vector<float> w;  // [I][O]
};

Mat RandomMat(int I, int O, int K, unsigned seed) {
  Mat m{I, O, K, std::vector<uint32_t>((size_t)I * O / 256 * TileDwords(K)), std::vector<float>((size_t)I * O)};
  std::mt19937 rng(seed);
  for (auto& x : m.tiles) x = rng();
  for (int i = 0; i < I; ++i)
    for (int o = 0; o < O; ++o) m.w[(size_t)i * O + o] = CpuWeight(m.tiles, I, K, i, o);
  return m;
}

int failures = 0, cases = 0;

void Report(bool ok, const char* what, int K, int a, int b, int c, double err) {
  ++cases;
  if (!ok) ++failures;
  std::printf("%s K=%d %d %d %d err=%.3e %s\n", what, K, a, b, c, err, ok ? "PASS" : "FAIL");
}

// Relative error against fp64 y[r][o] = sum_i x[r][i] w[i][o]; got is [R][O].
double GemmError(const Mat& m, const std::vector<__half>& x, int R, const std::vector<float>& got) {
  double worst = 0;
  for (int r = 0; r < R; ++r)
    for (int o = 0; o < m.O; ++o) {
      double ref = 0, mag = 0;
      for (int i = 0; i < m.I; ++i) {
        const double p = Value(x[(size_t)r * m.I + i]) * m.w[(size_t)i * m.O + o];
        ref += p;
        mag += std::fabs(p);
      }
      const double e = std::fabs(got[(size_t)r * m.O + o] - ref) / (mag + 1e-30);
      if (!std::isfinite(e)) return INFINITY;
      worst = std::max(worst, e);
    }
  return worst;
}

// ---- 1. DecodeLane -------------------------------------------------------------------------------

template <int K>
__global__ void DecodeAllKernel(const uint32_t* tiles, __half* out) {
  constexpr int T = TileDwords(K);
  const int lane = threadIdx.x, tile = blockIdx.x;
  const LaneWindow<K> win(lane);
  const uint32_t* t = tiles + (size_t)tile * T;
  __half2 P[4];
  DecodeLane<K, kMulTrue16>(t[win.d0], t[(win.d0 + 1) % T], t[(win.d0 + 2) % T], win.shift, P);
  // P0 rows (r0, r0+1) col c0, P1 rows (r0+8, r0+9) col c0, P2 col c0+8, P3 rows r0+8.. col c0+8.
  const int r0 = 2 * (lane % 4), c0 = lane / 4;
  __half* o = out + (size_t)tile * 256;
  const int rows[4] = {r0, r0 + 8, r0, r0 + 8}, cols[4] = {c0, c0, c0 + 8, c0 + 8};
  for (int k = 0; k < 4; ++k) {
    o[rows[k] * 16 + cols[k]] = __low2half(P[k]);
    o[(rows[k] + 1) * 16 + cols[k]] = __high2half(P[k]);
  }
}

template <int K>
void DecodeCase() {
  const int I = 64, O = 64, tiles = (I / 16) * (O / 16);
  const Mat m = RandomMat(I, O, K, 100 + K);
  Buffer<uint32_t> dt(m.tiles);
  Buffer<__half> out((size_t)tiles * 256);
  out.Nan();
  DecodeAllKernel<K><<<tiles, 32>>>(dt.p, out.p);
  CheckLaunch();
  const auto got = out.Get();
  long bad = 0;
  for (int i = 0; i < I; ++i)
    for (int o = 0; o < O; ++o) {
      const size_t tile = (size_t)(o / 16) * (I / 16) + i / 16;
      const __half g = got[tile * 256 + (i % 16) * 16 + o % 16];
      const __half want = __float2half(m.w[(size_t)i * O + o]);
      bad += __half_as_ushort(g) != __half_as_ushort(want);
    }
  Report(bad == 0, "decode-lane bitwise", K, I, O, 0, (double)bad);
}

// ---- 2. Decode GEMV --------------------------------------------------------------------------------

template <int R>
void GemvCase(const Mat& m, int S, int cache, bool policy) {
  const auto x = Half(Random((size_t)R * m.I, .1f, 7 + R));
  Buffer<uint32_t> dt(m.tiles);
  Buffer<__half> dx(x);
  Buffer<float> partial((size_t)S * R * m.O);
  partial.Nan();
  LinJob j{};
  j.tiles = dt.p;
  j.xh = dx.p;
  j.partial = partial.p;
  j.cache_policy = cache;
  j.I = m.I; j.O = m.O; j.K = m.K; j.S = S; j.R = R;
  Buffer<LinJob> dj(std::vector<LinJob>{j});
  std::vector<GemvItem> items;
  for (int s = 0; s < S; ++s)
    for (int b = 0; b < m.O / 16; b += 8) items.push_back({0, s, b, 8});
  Buffer<GemvItem> di(items);
  const size_t lds = (size_t)R * (m.I / 16 / S) * 32;
  if (policy) LinPolicyKernel<R, kMaxWideDecodeK><<<items.size(), 256, lds>>>(dj.p, di.p, items.size(), false, false);
  else LinGemvKernel<R, kMaxWideDecodeK><<<items.size(), 256, lds>>>(dj.p, di.p, items.size());
  CheckLaunch();
  const auto raw = partial.Get();
  std::vector<float> got((size_t)R * m.O, 0.f);
  for (int s = 0; s < S; ++s)
    for (size_t i = 0; i < got.size(); ++i) got[i] += raw[(size_t)s * R * m.O + i];
  const double e = GemmError(m, x, R, got);
  Report(e < 1e-5, policy ? "gemv-policy" : "gemv", m.K, R, S, cache, e);
}

void GemvCases(int K) {
  const Mat m = RandomMat(512, 256, K, 200 + K);
  for (const int S : {1, 2})
    for (const int cache : {0, 4}) {
      GemvCase<1>(m, S, cache, false);
      GemvCase<2>(m, S, cache, false);
      GemvCase<4>(m, S, cache, false);
      GemvCase<8>(m, S, cache, false);
      GemvCase<1>(m, S, cache, true);
      GemvCase<4>(m, S, cache, true);
    }
}

// ---- 3. Prefill dense GEMM ---------------------------------------------------------------------------

std::vector<__half> PackX(const std::vector<__half>& x, int M, int I) {
  std::vector<__half> packed(x.size());
  for (int r = 0; r < M; ++r)
    for (int i = 0; i < I; ++i) packed[((size_t)(i / 16) * M + r) * 16 + i % 16] = x[(size_t)r * I + i];
  return packed;
}

template <int K, int RM, int NB>
__global__ void WideBodyKernel(Exl3GemmJob j, const Exl3GemmItem* items, int n) {
  __shared__ uint32_t ws[kGemmWeightWords];
  if ((int)blockIdx.x >= n) return;
  PrefillWideBody<K, RM, NB>(j, items[blockIdx.x], ws);
}

template <int K, int RM>
void PrefillCase(const Mat& m, int M) {
  const auto x = Half(Random((size_t)M * m.I, .1f, 19 + M));
  Buffer<uint32_t> dt(m.tiles);
  Buffer<__half> dx(PackX(x, M, m.I));
  Buffer<float> y((size_t)M * m.O);
  y.Nan();
  Exl3GemmJob j{dt.p, dx.p, y.p, nullptr, m.I, m.O, M, 1, K, M};
  std::vector<Exl3GemmItem> items;
  for (int r = 0; r < M; r += 16 * RM)
    for (int b = 0; b < m.O / 16; b += 4 * 2) items.push_back({0, 0, r, b});
  Buffer<Exl3GemmItem> di(items);
  WideBodyKernel<K, RM, 2><<<items.size(), 128>>>(j, di.p, (int)items.size());
  CheckLaunch();
  const double e = GemmError(m, x, M, y.Get());
  Report(e < 1e-5, "prefill-dense", K, RM, M, 0, e);
}

template <int K>
void PrefillCases() {
  const Mat m = RandomMat(256, 128, K, 300 + K);
  for (const int M : {16, 17, 64, 129, 256}) {
    PrefillCase<K, 2>(m, M);
    PrefillCase<K, 4>(m, M);
    PrefillCase<K, 8>(m, M);
  }
}

// ---- 4. Prefill MoE GEMM -----------------------------------------------------------------------------

// One PrefillMoeGemmKernel<RM, packed, KMAX, HALF> launch over the jobs of mats (inputs xs, M rows);
// each job's partial as raw bits (fp32, or fp16 with HALF).
template <int RM, int KMAX, bool HALF>
std::vector<std::vector<uint32_t>> MoeLaunch(const std::vector<Mat>& mats, const std::vector<std::vector<__half>>& xs,
                                             int M) {
  using Out = std::conditional_t<HALF, __half, float>;
  std::vector<std::unique_ptr<Buffer<uint32_t>>> dts;
  std::vector<std::unique_ptr<Buffer<__half>>> dxs;
  std::vector<std::unique_ptr<Buffer<Out>>> ys;
  std::vector<Exl3GemmJob> jobs;
  std::vector<Exl3GemmItem> items;
  for (size_t n = 0; n < mats.size(); ++n) {
    const Mat& m = mats[n];
    dts.push_back(std::make_unique<Buffer<uint32_t>>(m.tiles));
    dxs.push_back(std::make_unique<Buffer<__half>>(PackX(xs[n], M, m.I)));
    ys.push_back(std::make_unique<Buffer<Out>>((size_t)M * m.O));
    ys[n]->Nan();
    jobs.push_back({dts[n]->p, dxs[n]->p, reinterpret_cast<float*>(ys[n]->p), nullptr, m.I, m.O, M, 1, m.K, M});
    for (int r = 0; r < M; r += 16 * RM)
      for (int b = 0; b < m.O / 16; b += 4 * kMoeNb) items.push_back({(int)n, 0, r, b});
  }
  Buffer<Exl3GemmJob> dj(jobs);
  Buffer<Exl3GemmItem> di(items);
  Buffer<int> count(std::vector<int>{(int)items.size()});
  PrefillMoeGemmKernel<RM, true, KMAX, HALF><<<64, 128>>>(dj.p, di.p, count.p);
  CheckLaunch();
  std::vector<std::vector<uint32_t>> out;
  for (const auto& y : ys) {
    const std::vector<Out> v = y->Get();
    std::vector<uint32_t> bits(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
      if constexpr (HALF) bits[i] = __half_as_ushort(v[i]);
      else std::memcpy(&bits[i], &v[i], 4);
    }
    out.push_back(std::move(bits));
  }
  return out;
}

// Values of a and b that differ in their bits (a and b of the same size).
double BitsDiffer(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  double n = a.size() == b.size() ? 0 : 1;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) n += a[i] != b[i];
  return n;
}

// Jobs of widths K4, K5 and KMAX; for kPrefillMoeK7Half K3, K3.5 and K7 (a 3.18 layer, the K3.5
// layers of 3.68 with one more width). Checks:
//   - fp32 partials against the fp64 reference;
//   - the fp16 form (HALF, the down projection): bit-equal to the saturated fp16 of the fp32 partial;
//   - kPrefillMoeK7Half: its K3 and K7 jobs bit-equal to the KMAX 7 kernel (fp32 and fp16).
template <int RM, int KMAX>
void MoeCase(int M) {
  const int I = 256, O = 128;
  const std::vector<int> widths = KMAX == kPrefillMoeK7Half ? std::vector<int>{3, 35, 7} : std::vector<int>{4, 5, KMAX};
  std::vector<Mat> mats;
  std::vector<std::vector<__half>> xs;
  for (size_t n = 0; n < widths.size(); ++n) {
    mats.push_back(RandomMat(I, O, widths[n], 400 + 100 * n + M));
    xs.push_back(Half(Random((size_t)M * I, .1f, 50 + n)));
  }
  const auto full = MoeLaunch<RM, KMAX, false>(mats, xs, M);
  const auto half = MoeLaunch<RM, KMAX, true>(mats, xs, M);
  for (size_t n = 0; n < mats.size(); ++n) {
    std::vector<float> y(full[n].size());
    std::vector<uint32_t> want(full[n].size());
    for (size_t i = 0; i < y.size(); ++i) {
      std::memcpy(&y[i], &full[n][i], 4);
      const float v = std::fabs(y[i]) > 65504.f ? std::copysign(65504.f, y[i]) : y[i];
      want[i] = __half_as_ushort(__float2half(v));
    }
    const double e = GemmError(mats[n], xs[n], M, y);
    Report(e < 1e-5, "prefill-moe", mats[n].K, RM, M, KMAX, e);
    const double d = BitsDiffer(half[n], want);
    Report(d == 0, "prefill-moe fp16 out", mats[n].K, RM, M, KMAX, d);
  }
  if constexpr (KMAX == kPrefillMoeK7Half) {
    const std::vector<Mat> same{mats[0], mats[2]};
    const std::vector<std::vector<__half>> same_x{xs[0], xs[2]};
    const auto full7 = MoeLaunch<RM, 7, false>(same, same_x, M);
    const auto half7 = MoeLaunch<RM, 7, true>(same, same_x, M);
    for (size_t n = 0; n < same.size(); ++n) {
      const size_t k = n ? 2 : 0;
      Report(BitsDiffer(full[k], full7[n]) == 0, "prefill-moe K7half=KMAX7", same[n].K, RM, M, 32,
             BitsDiffer(full[k], full7[n]));
      Report(BitsDiffer(half[k], half7[n]) == 0, "prefill-moe K7half=KMAX7", same[n].K, RM, M, 16,
             BitsDiffer(half[k], half7[n]));
    }
  }
}

template <int KMAX>
void MoeCases() {
  for (const int M : {16, 40, 96, 200}) {
    MoeCase<2, KMAX>(M);
    MoeCase<4, KMAX>(M);
    MoeCase<6, KMAX>(M);
  }
}

// ---- 5. Decode MoE bodies -------------------------------------------------------------------------

// The gate/up branch of MoeFusedKernelWide<R, KR, KS, kMoePf, CACHE, kMoeCols, XC> for one job.
template <int R, int KR, int KS, int CACHE, int XC>
__global__ void __launch_bounds__(256) MoeBodyKernel(const LinJob* jobs, const GemvItem* items) {
  extern __shared__ uint32_t xs[];
  const GemvItem item = items[blockIdx.x];
  const LinJob& job = jobs[item.job];
  const int K = __builtin_amdgcn_readfirstlane(job.K);
  const int rows = __builtin_amdgcn_readfirstlane(job.R);
  if constexpr (XC > 0) MoeAnyBodyChunked<R, KR, KS, kMoePf, kMoeCols, CACHE, XC>(K, rows, job, item, xs);
  else MoeAnyBody<R, KR, KS, kMoePf, kMoeCols, CACHE>(K, rows, job, item, xs, [&] { MoeFillX(job, item, rows, xs); });
}

// rows x m.I input, one job of width m.K; mutant: flip one tile word on the device copy.
template <int R, int KR, int KS, int CACHE, int XC>
double MoeBodyError(const Mat& m, int rows, bool mutant = false) {
  const auto x = Half(Random((size_t)rows * m.I, .1f, 900 + rows + m.K));
  std::vector<uint32_t> tiles = m.tiles;
  if (mutant) tiles[tiles.size() / 2 + 3] ^= 0x00100000u;
  Buffer<uint32_t> dt(tiles);
  Buffer<__half> dx(x);
  Buffer<float> partial((size_t)rows * m.O);
  partial.Nan();
  LinJob j{};
  j.tiles = dt.p;
  j.xh = dx.p;
  j.partial = partial.p;
  j.cache_policy = CACHE;
  j.I = m.I; j.O = m.O; j.K = m.K; j.S = 1; j.R = rows;
  Buffer<LinJob> dj(std::vector<LinJob>{j});
  std::vector<GemvItem> items;
  constexpr int ib = kMoeItemBlocks<kMoeCols>;
  for (int b = 0; b < m.O / 16; b += ib) items.push_back({0, 0, b, std::min(ib, m.O / 16 - b)});
  Buffer<GemvItem> di(items);
  const size_t lds = MoeFusedLds(R, XC, (size_t)R * (m.I / 16) * 32, 0);
  MoeBodyKernel<R, KR, KS, CACHE, XC><<<items.size(), 256, lds>>>(dj.p, di.p);
  CheckLaunch();
  return GemmError(m, x, rows, partial.Get());
}

// Routed rows 1, R / 2 and R at KR; shared rows R at KS. I 640 and 1280 (spans of one and two
// 40-tile chunks), O 256 (items of 16 blocks) and 128 (8 blocks).
template <int R, int KR, int KS, int CACHE, int XC>
void MoeBodyCases(const std::vector<Mat>& routed, const std::vector<Mat>& shared) {
  std::vector<int> counts{1};
  for (const int rows : {(R + 1) / 2, R})
    if (rows != counts.back()) counts.push_back(rows);
  for (const Mat& m : routed)
    for (const int rows : counts) {
      const double e = MoeBodyError<R, KR, KS, CACHE, XC>(m, rows);
      Report(e < 1e-5, XC ? "moe-body-chunked routed" : "moe-body routed", m.K, R, rows, m.I * 1000 + m.O, e);
    }
  for (const Mat& m : shared) {
    const double e = MoeBodyError<R, KR, KS, CACHE, XC>(m, R);
    Report(e < 1e-5, XC ? "moe-body-chunked shared" : "moe-body shared", m.K, R, R, m.I * 1000 + m.O, e);
  }
}

template <int KR>
void MoeBodyPair() {
  constexpr int KS = 7;
  std::vector<Mat> routed, shared;
  for (const auto& [I, O] : {std::pair{640, 256}, std::pair{1280, 128}}) {
    routed.push_back(RandomMat(I, O, KR, 700 + KR + I + O));
    shared.push_back(RandomMat(I, O, KS, 800 + KR + I + O));
  }
  MoeBodyCases<1, KR, KS, 0, 0>(routed, shared);
  MoeBodyCases<2, KR, KS, 0, 0>(routed, shared);
  MoeBodyCases<5, KR, KS, 0, 0>(routed, shared);
  MoeBodyCases<8, KR, KS, 0, 0>(routed, shared);
  MoeBodyCases<8, KR, KS, 4, 0>(routed, shared);
  // The engine stages K3 and K3.5 routed layers in chunks from R = 2 on (MoeFusedWideLaunch).
  if constexpr (KR == 3 || KR == 35) {
    MoeBodyCases<2, KR, KS, 0, kMoeChunk>(routed, shared);
    MoeBodyCases<8, KR, KS, 0, kMoeChunk>(routed, shared);
    MoeBodyCases<8, KR, KS, 4, kMoeChunk>(routed, shared);
  }
  // The check must reject a wrong tile (routed and shared).
  for (const Mat* m : {&routed[0], &shared[0]}) {
    const double e = MoeBodyError<8, KR, KS, 0, 0>(*m, 8, true);
    Report(!(e < 1e-5), "moe-body mutant rejected", m->K, 8, 8, 0, e);
  }
}

}  // namespace

int main() {
  DecodeCase<2>(); DecodeCase<3>(); DecodeCase<4>(); DecodeCase<5>();
  DecodeCase<6>(); DecodeCase<7>(); DecodeCase<8>(); DecodeCase<35>();
  for (const int K : {3, 6, 7, 8}) GemvCases(K);
  PrefillCases<7>();
  PrefillCases<8>();
  MoeCases<7>();
  MoeCases<8>();
  MoeCases<kPrefillMoeK7Half>();
  MoeBodyPair<3>();
  MoeBodyPair<35>();
  MoeBodyPair<4>();
  std::printf("packs_wide_test: %d cases, %d failures: %s\n", cases, failures, failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
