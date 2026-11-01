// prefill-dense: where the dense prefill GEMM time goes besides the
// WMMA core. Times the engine's fused EXL3 GEMM (LaunchPrefillGemmOut, front
// tile order) with its residual and store epilogues, at RM8 and RM4, and at
// three row counts (the round tail: 120 items of 4 waves fit at once), plus
// the GDN a/b fp16 projection (O = 96) kernels. RM4 must be bitwise equal to
// RM8, and every a/b kernel to PrefillDenseBigKernel.
//
// Each round times every variant once (median of 5 launches) with the mean
// sclk of a 5 ms hwmon sampler (bench::Sampler), so the variants interleave (ABAB).
// Usage: prefill_dense_epi K [ROUNDS=5]
#include "prefill_common.h"
#include "../engine/exl3/decode_split.h"
#include "kernels/prefill_hc.h"
#include "../engine/exl3/prefill_linear.h"
#include <functional>
using namespace crucible;
using namespace prefill_bench;

// One fused GEMM case: M rows, I to O, epilogue, row tiles per item.
struct GemmCase {
  int M, I, O, epi, rm;
  std::vector<Exl3GemmItem> items;
  Buffer<uint32_t> tiles;
  Buffer<__half> x, svh;
  Buffer<float> dst, aux;
  Buffer<LinJob> lin;
  Buffer<Exl3GemmJob> job;
  Buffer<Exl3GemmItem> it;
  std::vector<float> h0;
  GemmCase(int M_, int I_, int O_, int K, int epi_, int rm_)
      : M(M_), I(I_), O(O_), epi(epi_), rm(rm_), items(Items(M_, O_, rm_)),
        tiles(RandomWords((size_t)I_ / 16 * (O_ / 16) * 8 * K)), x(Half(Random((size_t)M_ * I_, .1f, 11))),
        svh(Half(Random(O_, .05f, 13))), dst(DstCount(M_, O_, epi_)), aux(Random((size_t)M_ * kStreams, .5f, 19)),
        lin(std::vector<LinJob>{Lin(K)}), job(std::vector<Exl3GemmJob>{Job(K)}), it(items),
        h0(Random(DstCount(M_, O_, epi_), 1.f, 17)) { dst.Set(h0); }
  static size_t DstCount(int M, int O, int epi) { return epi == kEpiStore ? (size_t)M * O : (size_t)M * kWide; }
  static std::vector<uint32_t> RandomWords(size_t n) {
    std::mt19937 rng(7);
    std::vector<uint32_t> w(n);
    for (auto& v : w) v = rng();
    return w;
  }
  static std::vector<Exl3GemmItem> Items(int M, int O, int rm) {
    std::vector<Exl3GemmItem> v;
    for (int r = 0; r < M; r += 16 * rm)
      for (int b = 0; b < O / 16; b += kFusedWaves) v.push_back({0, 0, r, b});
    return v;
  }
  int Split() const { return SplitFor(I / 16, DecodeTarget({{I, O}})); }
  LinJob Lin(int K) const {
    LinJob j{};
    j.tiles = tiles.p; j.svh = svh.p; j.dst = dst.p; j.aux = aux.p; j.I = I; j.O = O; j.K = K; j.S = 1; j.R = M;
    j.ldd = epi == kEpiStore ? O : kWide; j.epi = epi;
    return j;
  }
  Exl3GemmJob Job(int K) const { return {tiles.p, x.p, nullptr, nullptr, I, O, M, 1, K, M, Split()}; }
  void Run(int K) const { LaunchPrefillGemmOut(rm, lin.p, job.p, it.p, (int)items.size(), K == 6, true); }
  std::vector<float> Output(int K) { dst.Set(h0); Run(K); CheckLaunch(); return dst.Get(); }
};

struct Variant { std::string name; std::function<void()> go; double flop; };

int main(int argc, char** argv) {
  const int K = argc > 1 ? std::atoi(argv[1]) : 5, rounds = argc > 2 ? std::atoi(argv[2]) : 5;
  if (K < 3 || K > 6 || rounds < 1) return 2;
  constexpr int kOutI = 6144, kInO = 16384, kAbO = 96;
  GemmCase res8(4096, kOutI, kHidden, K, kEpiResidual, 8), res4(4096, kOutI, kHidden, K, kEpiResidual, 4);
  GemmCase sto8(4096, kOutI, kHidden, K, kEpiStore, 8);
  GemmCase lo(3584, kOutI, kHidden, K, kEpiResidual, 8), hi(4608, kOutI, kHidden, K, kEpiResidual, 8);
  GemmCase in8(4096, kHidden, kInO, K, kEpiStore, 8);
  int bad = 0;
  const bool rm4_same = res4.Output(K) == res8.Output(K);
  bad += !rm4_same;
  std::printf("out RM4 bitwise to RM8: %d (front split %d)\n", (int)rm4_same, res8.Split());
  // a/b projection: x [4096][2560] fp16, W [96][2560] fp16.
  const int R = 4096;
  Buffer<__half> ax(Half(Random((size_t)R * kHidden, .5f, 7))), aw(Half(Random((size_t)kAbO * kHidden, .02f, 427)));
  Buffer<float> ay1((size_t)R * kAbO), ay2((size_t)R * kAbO);
  const dim3 ag((R + 127) / 128, 1);
  auto big = [&] { PrefillDenseBigKernel<<<ag, 256>>>(aw.p, kAbO, ax.p, ay1.p, kAbO, R); };
  auto fast = [&] { PrefillDenseFastKernel<2><<<ag, 256>>>(aw.p, kAbO, ax.p, ay2.p, kAbO, R); };
  big(); CheckLaunch(); const auto ab_ref = ay1.Get();
  fast(); CheckLaunch(); const bool fast_same = ay2.Get() == ab_ref;
  bad += !fast_same;
  std::printf("ab Fast2 bitwise to Big: %d\n", (int)fast_same);
  const double fo = 2.0 * kOutI * kHidden, fi = 2.0 * kHidden * kInO, fa = 2.0 * kHidden * kAbO;
  std::vector<Variant> vs{
      {"out_res_rm8_m4096", [&] { res8.Run(K); }, fo * 4096}, {"out_sto_rm8_m4096", [&] { sto8.Run(K); }, fo * 4096},
      {"out_res_rm4_m4096", [&] { res4.Run(K); }, fo * 4096}, {"out_res_rm8_m3584", [&] { lo.Run(K); }, fo * 3584},
      {"out_res_rm8_m4608", [&] { hi.Run(K); }, fo * 4608}, {"in16384_sto_rm8_m4096", [&] { in8.Run(K); }, fi * 4096},
      {"ab_big_m4096", big, fa * R}, {"ab_fast2_m4096", fast, fa * R}};
  for (int round = 1; round <= rounds; ++round)
    for (const auto& v : vs) {
      bench::Sampler sclk(5);
      sclk.Start(); const double t = Time(v.go); sclk.Stop(); const double f = sclk.MeanMhz();
      std::printf("K=%d round %d %-24s %9.1f us %5.1f TF %4.0f MHz\n", K, round, v.name.c_str(), t,
                  v.flop / (t * 1e6), f);
    }
  std::printf("RESULT K=%d bitwise_all=%d\n", K, bad == 0);
  return bad ? 8 : 0;
}
