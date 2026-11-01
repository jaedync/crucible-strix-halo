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
#include <cstring>
#include <functional>
#include <memory>
#include <string>
using namespace crucible;
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
}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "check";
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
