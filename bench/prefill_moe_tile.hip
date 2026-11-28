// prefill-moe: the 192-row tile kernels against trunk's MoE path on the same
// inputs. Trunk (A): the plan with tile_rows 96, then the pair kernel (3.05 and 4.05 K sets) or FX + Mid (other
// K sets), then the RM6 fp16 down kernel. Tile path (B): the plan with tile_rows 192, PrefillMoeTilePairKernel,
// PrefillMoeTileDownKernel. xd (the down input) and pd (the down output) must be equal byte for byte; each
// mutant must differ. Outputs are NaN-filled before every run, so an unwritten value cannot pass.
#include "prefill_common.h"
#include "exl3_moe_mid_cold.h"
#include "../engine/layers/prefill_moe_tile.h"
#include <cstdint>
#include <cstring>
#include <array>
#include <cstdio>
#include <numeric>
#include <string>
using namespace crucible;
using namespace prefill_bench;

namespace {
// One MoE width set: routed K kr, shared K ks (codes 25 and 35 are K2.5 and K3.5), and its KMAX code.
struct KSet {
  const char* name;
  int kr, ks;
  int Kmax() const { return PrefillMoeKmax(std::array<int, 2>{kr, ks}); }
  bool Pair() const { return (kr == 3 && ks == 5) || (kr == 4 && ks == 6); }
};
// Every width set of kSupportedMoeWidths (decode_support.h): K2, K2.5, K3, K3.5, K4, K5 routed; K5..K8 shared.
const KSet kSets[] = {{"3/5", 3, 5},  {"4/6", 4, 6},  {"4/8", 4, 8},   {"5/8", 5, 8},   {"3/7", 3, 7},
                      {"3.5/7", 35, 7}, {"4/7", 4, 7}, {"2/6", 2, 6}, {"2.5/6", 25, 6}, {"3/6", 3, 6}};

size_t MatrixWords(int K) { return (size_t)(kHidden / 16) * (kMoeInter / 16) * TileDwords(K); }

// prefill-rows: the real routes of one layer from a CRUCIBLE_ROUTE_DUMP file (records {layer, R, R x 10 ids}),
// the launches of that layer in prompt order. Empty when the file has no record of the layer.
std::vector<int> DumpedRoutes(const std::string& path, int layer) {
  std::vector<int> out;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return out;
  int head[2];
  while (std::fread(head, sizeof(int), 2, f) == 2 && head[1] > 0) {
    std::vector<int> ids((size_t)head[1] * kTopK);
    if (std::fread(ids.data(), sizeof(int), ids.size(), f) != ids.size()) break;
    if (head[0] == layer) out.insert(out.end(), ids.begin(), ids.end());
  }
  std::fclose(f);
  return out;
}
// Routing of a fixture: "uniform", "skew", or "PATH@LAYER" (DumpedRoutes; the first R rows of the layer's launches).
struct Routing {
  bool skew = false;
  std::vector<int> dumped;
};

// prefill-rows diagnostic (timex): every routed job reads its X from the start of the side's buffer, so all experts
// share about 1.3 MB of hot X per side and the X reads hit the caches. Not exact (no byte check): it bounds what a
// cut of the X traffic can give. Jobs 0 .. 2 n - 1 are the routed gate/up jobs (2 e + p); n = kExperts.
__global__ void AliasXKernel(Exl3GemmJob* jobs, int n, int per, const __half* b0, const __half* b1) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * per) jobs[i].x = per == 2 && (i & 1) ? b1 : b0;
}

struct Fixture {
  const bool skew;
  const int R, total;
  const KSet set;
  std::vector<int> sel, cnt, off, pairs, ids;
  std::vector<size_t> woff;
  Buffer<int> ds, dc, d_o, dp, di, tc, ic, inv, fi, fu, fx, fr, fc;
  Buffer<__half> source, scale, xg, xu, xd, pd;
  Buffer<uint32_t> dw;
  Buffer<float> pg, pu;
  Buffer<LinJob> gu, down;
  Buffer<Exl3GemmJob> gj, dj;
  Buffer<Exl3GemmItem> gi, itdown;
  Buffer<ExpertPtrs> ep;
  PrefillMoe a{};
  Lin sg{}, su{}, sd{};
  const std::vector<int>* dumped = nullptr;
  Fixture(int rows, KSet ks, const Routing& routing)
      : skew(routing.skew), dumped(routing.dumped.empty() ? nullptr : &routing.dumped), R(rows), total(rows * kSlots), set(ks), sel(R * kTopK), cnt(kExperts), off(kExperts + 1),
        ids(R * kTopK), ds(sel.size()), dc(kExperts), d_o(kExperts + 1), dp(sel.size()), di(sel.size()),
        tc(kExperts * ((R * kTopK + 255) / 256)), ic(2), inv(sel.size()), fi(2 * ((total + 15) / 16 + kExperts + 1)),
        fu(kExperts + 1), fx(kExperts + 1), fr((total + 15) / 16 + kExperts + 1), fc(3),
        source(Half(Random((size_t)R * kHidden, .5f, 21))), scale((size_t)(kExperts + 1) * 3 * kHidden),
        xg((size_t)total * kHidden), xu(xg.n), xd((size_t)total * kMoeInter), pd((size_t)2 * total * kHidden),
        dw(2 * ((size_t)kExperts * MatrixWords(ks.kr) + MatrixWords(ks.ks))), pg((size_t)total * kMoeInter), pu(pg.n),
        gu((kExperts + 1) * 2), down(kExperts + 1), gj(gu.n), dj(down.n),
        gi(((total + 15) / 16 + kExperts + 1) * 20), itdown(gi.n * 2), ep(kExperts) {
    std::mt19937 rng(7);
    BuildRouting(rng);
    BuildWeights(rng);
    BuildJobs();
  }
  // skew: expert = 512 u^3 (a few experts get thousands of rows, many get 0 to 7); else uniform top-10.
  void BuildRouting(std::mt19937& rng) {
    std::vector<int> ex(kExperts);
    std::iota(ex.begin(), ex.end(), 0);
    std::uniform_real_distribution<double> u(0, 1);
    for (int r = 0; r < R; ++r)
      for (int k = 0; k < kTopK; ++k) {
        if (dumped) {
          const int e = (*dumped)[(size_t)r * kTopK + k];
          sel[r * kTopK + k] = e;
          ++cnt[e];
          continue;
        }
        if (skew) {
          int e;
          do {
            const double x = u(rng);
            e = std::min(kExperts - 1, (int)(kExperts * x * x * x));
          } while (std::find(sel.begin() + r * kTopK, sel.begin() + r * kTopK + k, e) != sel.begin() + r * kTopK + k);
          sel[r * kTopK + k] = e;
          ++cnt[e];
          continue;
        }
        std::swap(ex[k], ex[k + rng() % (kExperts - k)]);
        sel[r * kTopK + k] = ex[k];
        ++cnt[ex[k]];
      }
    std::vector<std::vector<int>> by(kExperts);
    for (int i = 0; i < R * kTopK; ++i) by[sel[i]].push_back(i);
    for (int e = 0; e < kExperts; ++e) {
      off[e + 1] = off[e] + cnt[e];
      pairs.insert(pairs.end(), by[e].begin(), by[e].end());
    }
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = pairs[i] / kTopK;
    ds.Set(sel); dc.Set(cnt); d_o.Set(off); dp.Set(pairs); di.Set(ids);
  }
  // Random trellis bits (every bit string is a valid code). The down projection reuses the gate tiles (the same
  // tile count). Gate and up scales of about 0.2 give an rms of about 1, as in the model (prefill_moe_mid.hip).
  void BuildWeights(std::mt19937& rng) {
    std::vector<uint32_t> w(dw.n);
    for (auto& v : w) v = rng();
    dw.Set(w);
    auto sv = Random(scale.n, .004f, 23);
    for (auto& v : sv) v = (v < 0 ? -1.f : 1.f) * (.012f + std::fabs(v));
    for (int e = 0; e <= kExperts; ++e)
      for (int p = 0; p < 2; ++p)
        for (int i = 0; i < kHidden; ++i) sv[(size_t)(e * 3 + p) * kHidden + i] *= 13.f;
    scale.Set(Half(sv));
    woff.resize((kExperts + 1) * 2);
    size_t pos = 0;
    for (int e = 0; e <= kExperts; ++e)
      for (int p = 0; p < 2; ++p) {
        woff[2 * e + p] = pos;
        pos += MatrixWords(e == kExperts ? set.ks : set.kr);
      }
  }
  Lin Make(int e, int p) {
    Lin l{};
    l.I = kHidden; l.O = kMoeInter; l.K = e == kExperts ? set.ks : set.kr;
    l.tiles = dw.p + woff[2 * e + (p & 1)];
    l.suh = scale.p + (size_t)(e * 3 + p) * kHidden; l.svh = l.suh;
    return l;
  }
  void BuildJobs() {
    std::vector<ExpertPtrs> experts(kExperts);
    for (int e = 0; e < kExperts; ++e) {
      const Lin g = Make(e, 0), u = Make(e, 1), d = Make(e, 2);
      experts[e] = ExpertPtrs{{g.tiles, u.tiles, d.tiles}, {g.suh, u.suh, d.suh}, {g.svh, u.svh, d.svh}, {g.K, u.K, d.K}};
    }
    ep.Set(experts);
    sg = Make(kExperts, 0); su = Make(kExperts, 1); sd = Make(kExperts, 2);
    a.dispatch = {ds.p, tc.p, dc.p, d_o.p, dp.p, di.p, R};
    a.gu = gu.p; a.down = down.p; a.g_gu = gj.p; a.g_down = dj.p; a.i_gu = gi.p; a.i_down = itdown.p;
    a.item_counts = ic.p; a.inverse = inv.p; a.packed = true;
    a.pg = pg.p; a.pu = pu.p; a.pd = reinterpret_cast<float*>(pd.p); a.xg = xg.p; a.xu = xu.p; a.xd = xd.p;
    a.xh = source.p;
    a.fx_items = fi.p; a.fx_uoff = fu.p; a.fx_xoff = fx.p; a.fx_ready = fr.p; a.fx_count = fc.p;
  }
  // The fp16 pd of expert e starts at float offset off[e] x kHidden (PrefillMoeJobs: a.pd is a float pointer),
  // so only the first M x kHidden halves of each expert's region are written.
  std::vector<__half> PdWritten(const std::vector<__half>& pdv) const {
    std::vector<__half> v;
    v.reserve((size_t)total * kHidden);
    auto take = [&](size_t o, size_t m) { v.insert(v.end(), pdv.begin() + 2 * o * kHidden, pdv.begin() + (2 * o + m) * kHidden); };
    for (int e = 0; e < kExperts; ++e) take(off[e], cnt[e]);
    take((size_t)R * kTopK, R);
    return v;
  }
  bool alias_x = false;
  void Plan(int rows) {
    a.tile_rows = rows;
    HIP_OK(hipMemsetAsync(ic.p, 0, 8));
    PrefillMoeFxPlanKernel<<<1, 512>>>(a, ep.p, sg, su, sd);
    if (alias_x) {
      AliasXKernel<<<(2 * kExperts + 255) / 256, 256>>>(gj.p, kExperts, 2, xg.p, xu.p);
      AliasXKernel<<<(kExperts + 255) / 256, 256>>>(dj.p, kExperts, 1, xd.p, xd.p);
    }
  }
  // Trunk (prefill_engine.inc before this branch): RM6, fused, half pd.
  void TrunkGateUp() {
    Plan(96);
    if (set.Pair()) LaunchPrefillMoePair(6, a, set.ks == 6);
    else {
      LaunchPrefillMoeFxK(6, a, set.Kmax());
      PrefillMoeMidKernel<<<(total * 5 + 7) / 8, 256>>>(a);
    }
  }
  void TrunkDown() { LaunchPrefillMoeGemmK({6, true}, a.g_down, a.i_down, a.item_counts + 1, set.Kmax(), true); }
  template <int MUT = 0> void TileGateUp(int lead = kTileFxLead) {
    Plan(kTileRows);
    if constexpr (MUT == 0) LaunchPrefillMoeTilePair(set.Kmax(), a, lead);
    else if (set.Kmax() == 6) PrefillMoeTilePairKernel<6, MUT><<<512, 512>>>(a, lead);
    else PrefillMoeTilePairKernel<5, MUT><<<512, 512>>>(a, lead);
  }
  // moe-fused split (diagnostic): the X transform part alone (no G claims), then the G part alone on the X items of
  // that run (every X item claimed and ready). MUT: the TileBody diagnostics (kTileMut*).
  int ng_saved = -1;
  void TileFxOnly() {
    Plan(kTileRows);
    if (ng_saved < 0) {
      HIP_OK(hipMemcpy(&ng_saved, ic.p, sizeof(int), hipMemcpyDeviceToHost));
    }
    HIP_OK(hipMemsetAsync(ic.p, 0, sizeof(int)));
    PrefillMoeTilePairKernel<6><<<512, 512>>>(a, kFxLead);
  }
  template <int MUT = 0> void TileGOnly() {
    HIP_OK(hipMemcpyAsync(ic.p, &ng_saved, sizeof(int), hipMemcpyHostToDevice));
    HIP_OK(hipMemsetAsync(fc.p + 1, 0, sizeof(int)));
    PrefillMoeTilePairKernel<6, MUT><<<512, 512>>>(a, kFxLead);
  }
  // moe-pc: the engine's launches for a tile launch with KMAX 5 or 6 (prefill_engine.inc: the fat gate/up kernel,
  // then the tile down kernel).
  void PcGateUp(int lead = kTileFxLead) {
    Plan(kTileRows);
    LaunchPrefillMoePcEngine(set.Kmax(), a, lead);
  }
  void PcDown() { LaunchPrefillMoePcDownEngine(set.Kmax(), a.g_down, a.i_down, a.item_counts + 1); }
  // moe-lds: the engine's fat gate/up kernel with the old 32-word A plane skew (the same bytes on other LDS banks).
  void PcGateUpPlane32(int lead = kTileFxLead) {
    Plan(kTileRows);
    LaunchPrefillMoePcFatPair<kTileMutPlane32>(set.Kmax(), a, lead);
  }
  template <int MUT = 0> void TileDown() {
    if constexpr (MUT == 0) LaunchPrefillMoeTileDown(set.Kmax(), a.g_down, a.i_down, a.item_counts + 1);
    else if (set.Kmax() == 6) PrefillMoeTileDownKernel<6, MUT><<<1024, 256>>>(a.g_down, a.i_down, a.item_counts + 1);
    else PrefillMoeTileDownKernel<5, MUT><<<1024, 256>>>(a.g_down, a.i_down, a.item_counts + 1);
  }
};

size_t DiffCount(const std::vector<__half>& x, const std::vector<__half>& y) {
  size_t n = 0;
  for (size_t i = 0; i < x.size(); ++i) n += std::memcmp(&x[i], &y[i], sizeof(__half)) != 0;
  return n;
}
// Two-compiler check: the differences that are only the sign of a zero (+0 against -0).
size_t ZeroSignDiffCount(const std::vector<__half>& x, const std::vector<__half>& y) {
  size_t n = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    uint16_t p, q;
    std::memcpy(&p, &x[i], 2);
    std::memcpy(&q, &y[i], 2);
    n += p != q && (p & 0x7fff) == 0 && (q & 0x7fff) == 0;
  }
  return n;
}
// Two-compiler check: FNV-1a of the bytes, so two builds (two compilers) can compare their outputs.
unsigned long long Fnv(const std::vector<__half>& x) {
  unsigned long long h = 1469598103934665603ull;
  const auto* b = reinterpret_cast<const unsigned char*>(x.data());
  for (size_t i = 0; i < x.size() * sizeof(__half); ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}
size_t Unwritten(const std::vector<__half>& x) {
  size_t n = 0;
  for (const auto h : x) n += !std::isfinite(Value(h));
  return n;
}

template <class F> double ColdTime(F run, const gm::ColdCache& cold, const std::string& label, int reps = 7) {
  // hwmon clock averages lag short kernels: warm each path for 0.5 s before the cold samples.
  const auto start = std::chrono::steady_clock::now();
  do {
    gm::Flush(&cold);
    run();
    CheckLaunch();
  } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < .5);
  bench::EventTimer t;
  std::vector<double> ms;
  bench::Sampler clocks;
  clocks.Start();
  for (int i = 0; i < reps; ++i) {
    gm::Flush(&cold);
    t.Begin();
    run();
    ms.push_back(t.EndMs());
  }
  clocks.Stop();
  std::printf("clocks-%s %s\n", label.c_str(), clocks.Summary().c_str());
  return bench::Median(ms);
}

// Byte checks of one fixture. Mutants run only for the 3.05 and 4.05 sets (instantiated for KMAX 5 and 6).
bool Check(Fixture& f, const char* tag) {
  const char* name = f.set.name;
  f.xd.Nan(); f.pd.Nan(); f.xg.Nan(); f.xu.Nan();
  f.TrunkGateUp(); f.TrunkDown(); CheckLaunch();
  const auto xref = f.xd.Get(), pref = f.PdWritten(f.pd.Get());
  const auto gref = f.xg.Get(), uref = f.xu.Get();
  bool ok = Unwritten(xref) == 0 && Unwritten(pref) == 0;
  std::printf("check set=%s R=%d routing=%s variant=trunk unwritten-xd=%zu unwritten-pd=%zu xd-hash=%016llx "
              "pd-hash=%016llx %s\n", name, f.R, tag, Unwritten(xref), Unwritten(pref), Fnv(xref), Fnv(pref),
              ok ? "PASS" : "FAIL");
  auto run = [&](const char* variant, auto gateup, auto down, bool expect_equal) {
    f.xd.Nan(); f.pd.Nan(); f.xg.Nan(); f.xu.Nan();
    gateup(); CheckLaunch();
    const auto xd = f.xd.Get();
    // Info (not a pass condition): the fp16 X inputs of gate and up, and the xd bytes that differ only in a zero's sign.
    const auto xg = f.xg.Get(), xu = f.xu.Get();
    std::printf("info set=%s R=%d routing=%s variant=%s xg-diff=%zu xu-diff=%zu xg-zero-sign=%zu xu-zero-sign=%zu "
                "xd-zero-sign=%zu xd-hash=%016llx\n", name, f.R, tag, variant, DiffCount(xg, gref), DiffCount(xu, uref),
                ZeroSignDiffCount(xg, gref), ZeroSignDiffCount(xu, uref), ZeroSignDiffCount(xd, xref), Fnv(xd));
    // The down kernel reads trunk's xd, so a gate/up mutant does not hide a down difference and back.
    HIP_OK(hipMemcpy(f.xd.p, xref.data(), xref.size() * sizeof(__half), hipMemcpyHostToDevice));
    down(); CheckLaunch();
    const auto pdv = f.PdWritten(f.pd.Get());
    const size_t dx = DiffCount(xd, xref), dp = DiffCount(pdv, pref);
    const bool pass = expect_equal ? dx == 0 && dp == 0 : dx + dp > 0;
    ok = ok && pass;
    std::printf("check set=%s R=%d routing=%s variant=%s xd-diff=%zu/%zu pd-diff=%zu/%zu expect=%s %s\n", name, f.R,
                tag, variant, dx, xd.size(), dp, pdv.size(), expect_equal ? "equal" : "differ", pass ? "PASS" : "FAIL");
  };
  run("tile", [&] { f.TileGateUp(); }, [&] { f.TileDown(); }, true);
  // Two-compiler check: the engine's tile launch (kPrefillMoePc) must give trunk's bytes too, with every compiler.
  if (PrefillMoePcOn(f.set.Kmax())) run("pc", [&] { f.PcGateUp(); }, [&] { f.PcDown(); }, true);
  if (f.set.Pair()) {
    run("mut-no-suh", [&] { f.TileGateUp<1>(); }, [&] { f.TileDown(); }, false);
    run("mut-swap", [&] { f.TileGateUp<2>(); }, [&] { f.TileDown(); }, false);
    run("mut-korder-gateup", [&] { f.TileGateUp<3>(); }, [&] { f.TileDown(); }, false);
    // The down mutant alone: gate/up is the correct tile kernel, so only pd may differ.
    run("mut-korder-down", [&] { f.TileGateUp(); }, [&] { f.TileDown<3>(); }, false);
  }
  std::printf("checks set=%s R=%d routing=%s %s\n", name, f.R, tag, ok ? "PASS" : "FAIL");
  std::fflush(stdout);
  return ok;
}

// ABAB timing: trunk gate/up (A) and tile gate/up (B), trunk down and tile down, each cold (MALL flushed).
// The gate/up times include the plan kernel on both sides. ms per 32K prompt = ms per call x 48 x 32768 / R.
void Time(Fixture& f, const char* tag, int pairs) {
  const auto cold = gm::MakeColdCache();
  const double scale = 48. * 32768 / f.R;
  const std::string id = std::string(f.set.name) + "-" + std::to_string(f.R);
  // One same-shape warm-up per arm.
  ColdTime([&] { f.TrunkGateUp(); }, cold, "warm-A-gateup-" + id);
  ColdTime([&] { f.TileGateUp(); }, cold, "warm-B-gateup-" + id);
  std::vector<double> ga, gb, da, db;
  for (int p = 0; p < pairs; ++p) {
    ga.push_back(ColdTime([&] { f.TrunkGateUp(); }, cold, "A-gateup-" + id));
    gb.push_back(ColdTime([&] { f.TileGateUp(); }, cold, "B-gateup-" + id));
    f.Plan(96); CheckLaunch();
    da.push_back(ColdTime([&] { f.TrunkDown(); }, cold, "A-down-" + id));
    f.Plan(kTileRows); CheckLaunch();
    db.push_back(ColdTime([&] { f.TileDown(); }, cold, "B-down-" + id));
    std::printf("abab set=%s R=%d routing=%s pair=%d gateup A=%.4f B=%.4f down A=%.4f B=%.4f ms-per-call\n",
                f.set.name, f.R, tag, p, ga.back(), gb.back(), da.back(), db.back());
    std::fflush(stdout);
  }
  const double GA = bench::Median(ga), GB = bench::Median(gb), DA = bench::Median(da), DB = bench::Median(db);
  int gw = 0, dw = 0;
  for (int p = 0; p < pairs; ++p) { gw += gb[p] < ga[p]; dw += db[p] < da[p]; }
  const double gflop = 4. * f.total * kHidden * kMoeInter * 1e-9, dflop = gflop / 2;
  std::printf("result set=%s R=%d routing=%s gateup A=%.4f B=%.4f ms (%.1f/%.1f TFLOPS) B-faster=%d/%d save=%.0f "
              "ms-per-32k | down A=%.4f B=%.4f ms (%.1f/%.1f TFLOPS) B-faster=%d/%d save=%.0f ms-per-32k\n",
              f.set.name, f.R, tag, GA, GB, gflop / GA, gflop / GB, gw, pairs, (GA - GB) * scale, DA, DB, dflop / DA,
              dflop / DB, dw, pairs, (DA - DB) * scale);
  std::fflush(stdout);
  // Energy loop (ENERGY_S): the trunk gate/up + down of one layer, MALL warm. Plan(96) first: TrunkDown reads that plan.
  f.Plan(96); CheckLaunch();
  EnergyHook("moe", "trunk_gateup_down", [&] { f.TrunkGateUp(); f.TrunkDown(); });
  cold.Free();
}

template <class P, class F> double ColdTimePre(P pre, F run, const gm::ColdCache& cold, int reps = 7) {
  bench::EventTimer t;
  std::vector<double> ms;
  for (int i = 0; i < reps + 2; ++i) {
    gm::Flush(&cold);
    pre();
    CheckLaunch();
    HIP_OK(hipDeviceSynchronize());
    t.Begin();
    run();
    const double v = t.EndMs();
    CheckLaunch();
    if (i >= 2) ms.push_back(v);
  }
  return bench::Median(ms);
}

// moe-fused split: where the time of the tile kernels goes (4.05 set, KMAX 6). Each line is a cold median.
void Split(Fixture& f, const char* tag, int pairs) {
  const auto cold = gm::MakeColdCache();
  const double scale = 48. * 32768 / f.R;
  auto none = [] {};
  auto line = [&](const char* what, double ms) {
    std::printf("split set=%s R=%d routing=%s %-22s %.4f ms  (%.0f ms-per-32k)\n", f.set.name, f.R, tag, what, ms,
                ms * scale);
    std::fflush(stdout);
  };
  // Byte check of the exact variants against trunk (xd from gate/up, pd from down on trunk's xd).
  f.xd.Nan(); f.pd.Nan();
  f.TrunkGateUp(); f.TrunkDown(); CheckLaunch();
  const auto xref = f.xd.Get(), pref = f.PdWritten(f.pd.Get());
  auto bytes = [&](const char* what, auto gateup, auto down) {
    f.xd.Nan(); f.pd.Nan();
    gateup(); CheckLaunch();
    const auto xd = f.xd.Get();
    HIP_OK(hipMemcpy(f.xd.p, xref.data(), xref.size() * sizeof(__half), hipMemcpyHostToDevice));
    down(); CheckLaunch();
    const size_t dx = DiffCount(xd, xref), dp = DiffCount(f.PdWritten(f.pd.Get()), pref);
    std::printf("bytes set=%s R=%d routing=%s %s xd-diff=%zu pd-diff=%zu %s\n", f.set.name, f.R, tag, what, dx, dp,
                dx + dp == 0 ? "PASS" : "FAIL");
  };
  bytes("tile", [&] { f.TileGateUp(); }, [&] { f.TileDown(); });
  bytes("deepW", [&] { f.TileGateUp<kTileMutDeepW>(); }, [&] { f.TileDown<kTileMutDeepW>(); });
  bytes("deepA", [&] { f.TileGateUp<kTileMutDeepA>(); }, [&] { f.TileDown<kTileMutDeepA>(); });
  bytes("deep", [&] { f.TileGateUp<kTileMutDeep>(); }, [&] { f.TileDown<kTileMutDeep>(); });
  bytes("mut-korder", [&] { f.TileGateUp<3>(); }, [&] { f.TileDown<3>(); });
  bench::Sampler clocks;
  clocks.Start();
  for (int p = 0; p < pairs; ++p) {
    auto fx = [&] { f.TileFxOnly(); };
    line("B-g-oneA", ColdTimePre(fx, [&] { f.TileGOnly<kTileMutOneA>(); }, cold));
    line("B-g-oneB", ColdTimePre(fx, [&] { f.TileGOnly<kTileMutOneB>(); }, cold));
    line("B-g-oneAB", ColdTimePre(fx, [&] { f.TileGOnly<kTileMutOneAB>(); }, cold));
    line("B-g-bare", ColdTimePre(fx, [&] { f.TileGOnly<kTileMutBare>(); }, cold));
    line("B-g-nosync", ColdTimePre(fx, [&] { f.TileGOnly<kTileMutNoSync>(); }, cold));
    line("B-down-oneA", ColdTimePre([&] { f.Plan(kTileRows); }, [&] { f.TileDown<kTileMutOneA>(); }, cold));
    line("B-down-oneB", ColdTimePre([&] { f.Plan(kTileRows); }, [&] { f.TileDown<kTileMutOneB>(); }, cold));
    line("B-down-oneAB", ColdTimePre([&] { f.Plan(kTileRows); }, [&] { f.TileDown<kTileMutOneAB>(); }, cold));
    line("B-down-bare", ColdTimePre([&] { f.Plan(kTileRows); }, [&] { f.TileDown<kTileMutBare>(); }, cold));
    line("B-gateup-deepW", ColdTimePre(none, [&] { f.TileGateUp<kTileMutDeepW>(); }, cold));
    line("B-gateup-deepA", ColdTimePre(none, [&] { f.TileGateUp<kTileMutDeepA>(); }, cold));
    line("B-gateup-deep", ColdTimePre(none, [&] { f.TileGateUp<kTileMutDeep>(); }, cold));
    line("B-gateup-nosync", ColdTimePre(none, [&] { f.TileGateUp<kTileMutNoSync>(); }, cold));
    line("B-down-deepW", ColdTimePre([&] { f.Plan(kTileRows); }, [&] { f.TileDown<kTileMutDeepW>(); }, cold));
    line("B-down-deepA", ColdTimePre([&] { f.Plan(kTileRows); }, [&] { f.TileDown<kTileMutDeepA>(); }, cold));
    line("B-down-deep", ColdTimePre([&] { f.Plan(kTileRows); }, [&] { f.TileDown<kTileMutDeep>(); }, cold));
    line("B-down-nosync", ColdTimePre([&] { f.Plan(kTileRows); }, [&] { f.TileDown<kTileMutNoSync>(); }, cold));
    line("A-gateup", ColdTimePre(none, [&] { f.TrunkGateUp(); }, cold));
    line("B-gateup", ColdTimePre(none, [&] { f.TileGateUp(); }, cold));
    line("B-fx-only", ColdTimePre(none, [&] { f.TileFxOnly(); }, cold));
    line("B-g-only", ColdTimePre(fx, [&] { f.TileGOnly(); }, cold));
    line("B-g-nodecode", ColdTimePre(fx, [&] { f.TileGOnly<kTileMutNoDecode>(); }, cold));
    line("B-g-nowmma", ColdTimePre(fx, [&] { f.TileGOnly<kTileMutNoWmma>(); }, cold));
    line("B-g-noA", ColdTimePre(fx, [&] { f.TileGOnly<kTileMutNoA>(); }, cold));
    line("B-g-noW", ColdTimePre(fx, [&] { f.TileGOnly<kTileMutNoW>(); }, cold));
    auto plan = [&] { f.Plan(kTileRows); };
    line("A-down", ColdTimePre([&] { f.Plan(96); }, [&] { f.TrunkDown(); }, cold));
    line("B-down", ColdTimePre(plan, [&] { f.TileDown(); }, cold));
    line("B-down-nodecode", ColdTimePre(plan, [&] { f.TileDown<kTileMutNoDecode>(); }, cold));
    line("B-down-nowmma", ColdTimePre(plan, [&] { f.TileDown<kTileMutNoWmma>(); }, cold));
    line("B-down-noA", ColdTimePre(plan, [&] { f.TileDown<kTileMutNoA>(); }, cold));
    line("B-down-noW", ColdTimePre(plan, [&] { f.TileDown<kTileMutNoW>(); }, cold));
    f.alias_x = true;
    line("B-g-only-xcache", ColdTimePre(fx, [&] { f.TileGOnly(); }, cold));
    line("B-down-xcache", ColdTimePre(plan, [&] { f.TileDown(); }, cold));
    f.alias_x = false;
  }
  clocks.Stop();
  std::printf("clocks-split %s\n", clocks.Summary().c_str());
  cold.Free();
}

// moe-fused conc (diagnostic): gate/up and down of one launch-layer on two streams at once, with GA gate/up blocks
// (16 waves) and GD down blocks (8 waves), against the two kernels one after the other. Down reads an xd that gate/up
// writes at the same time (timing only). Bounds what one kernel with both queues can give.
void Conc(Fixture& f, const char* tag, int pairs) {
  const auto cold = gm::MakeColdCache();
  const double scale = 48. * 32768 / f.R;
  hipStream_t s1, s2;
  HIP_OK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  HIP_OK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
  auto host_ms = [&](auto run) {
    std::vector<double> ms;
    for (int i = 0; i < 9; ++i) {
      gm::Flush(&cold);
      f.Plan(kTileRows);
      HIP_OK(hipDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      run();
      HIP_OK(hipDeviceSynchronize());
      CheckLaunch();
      if (i >= 2) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return bench::Median(ms);
  };
  auto line = [&](const std::string& what, double ms) {
    std::printf("conc set=%s R=%d routing=%s %-22s %.4f ms  (%.0f ms-per-32k)\n", f.set.name, f.R, tag, what.c_str(), ms,
                ms * scale);
    std::fflush(stdout);
  };
  const int ga_list[] = {40, 30, 20, 10}, gd_list[] = {80, 60, 40, 20};
  for (int p = 0; p < pairs; ++p) {
    line("seq", host_ms([&] {
      PrefillMoeTilePairKernel<6><<<512, 512, 0, s1>>>(f.a, kFxLead);
      PrefillMoeTileDownKernel<6><<<1024, 256, 0, s1>>>(f.a.g_down, f.a.i_down, f.a.item_counts + 1);
    }));
    line("gateup-only", host_ms([&] { PrefillMoeTilePairKernel<6><<<512, 512, 0, s1>>>(f.a, kFxLead); }));
    line("down-only", host_ms([&] {
      PrefillMoeTileDownKernel<6><<<1024, 256, 0, s1>>>(f.a.g_down, f.a.i_down, f.a.item_counts + 1);
    }));
    for (const int ga : ga_list)
      for (const int gd : gd_list) {
        // 8 waves per SIMD: 2 gate/up or 4 down blocks per WGP, 20 WGPs (40 + 80 oversubscribes, the rest fit).
        if (ga * 2 + gd > 80 && !(ga == 40 && gd == 80)) continue;
        line("conc-ga" + std::to_string(ga) + "-gd" + std::to_string(gd), host_ms([&] {
          PrefillMoeTilePairKernel<6><<<ga, 512, 0, s1>>>(f.a, kFxLead);
          PrefillMoeTileDownKernel<6><<<gd, 256, 0, s2>>>(f.a.g_down, f.a.i_down, f.a.item_counts + 1);
        }));
      }
  }
  HIP_OK(hipStreamDestroy(s1));
  HIP_OK(hipStreamDestroy(s2));
  cold.Free();
}

// moe-fused pipe: byte check and timing of the load pipeline of depth 2 and 3 (TileBodyPipe) and of the X lead.
void Pipe(Fixture& f, const char* tag, int pairs) {
  const auto cold = gm::MakeColdCache();
  const double scale = 48. * 32768 / f.R;
  f.xd.Nan(); f.pd.Nan();
  f.TrunkGateUp(); f.TrunkDown(); CheckLaunch();
  const auto xref = f.xd.Get(), pref = f.PdWritten(f.pd.Get());
  bool all = true;
  auto bytes = [&](const char* what, auto gateup, auto down, bool expect_equal) {
    f.xd.Nan(); f.pd.Nan();
    gateup(); CheckLaunch();
    const auto xd = f.xd.Get();
    HIP_OK(hipMemcpy(f.xd.p, xref.data(), xref.size() * sizeof(__half), hipMemcpyHostToDevice));
    down(); CheckLaunch();
    const auto pdv = f.PdWritten(f.pd.Get());
    const size_t dx = DiffCount(xd, xref), dp = DiffCount(pdv, pref);
    const bool pass = expect_equal ? dx + dp == 0 && Unwritten(xd) == 0 && Unwritten(pdv) == 0 : dx + dp > 0;
    all = all && pass;
    std::printf("bytes set=%s R=%d routing=%s %s xd-diff=%zu pd-diff=%zu expect=%s %s\n", f.set.name, f.R, tag, what, dx,
                dp, expect_equal ? "equal" : "differ", pass ? "PASS" : "FAIL");
    std::fflush(stdout);
  };
  bytes("tile", [&] { f.TileGateUp(); }, [&] { f.TileDown(); }, true);
  bytes("pipe2", [&] { f.TileGateUp<kTileMutPipe2>(); }, [&] { f.TileDown<kTileMutPipe2>(); }, true);
  bytes("pipe3", [&] { f.TileGateUp<kTileMutPipe3>(); }, [&] { f.TileDown<kTileMutPipe3>(); }, true);
  bytes("tile-lead32", [&] { f.TileGateUp(32); }, [&] { f.TileDown(); }, true);
  bytes("tile-lead512", [&] { f.TileGateUp(512); }, [&] { f.TileDown(); }, true);
  bytes("pipe1", [&] { f.TileGateUp<kTileMutPipe1>(); }, [&] { f.TileDown<kTileMutPipe1>(); }, true);
  bytes("fast1", [&] { f.TileGateUp<kTileMutFast1>(); }, [&] { f.TileDown<kTileMutFast1>(); }, true);
  bytes("fast2", [&] { f.TileGateUp<kTileMutFast2>(); }, [&] { f.TileDown<kTileMutFast2>(); }, true);
  bytes("fast2-lead0", [&] { f.TileGateUp<kTileMutFast2>(0); }, [&] { f.TileDown<kTileMutFast2>(); }, true);
  bytes("fast1-lead32", [&] { f.TileGateUp<kTileMutFast1>(32); }, [&] { f.TileDown<kTileMutFast1>(); }, true);
  bytes("mut-korder", [&] { f.TileGateUp<3>(); }, [&] { f.TileDown<3>(); }, false);
  std::printf("bytes-all set=%s R=%d routing=%s %s\n", f.set.name, f.R, tag, all ? "PASS" : "FAIL");
  auto none = [] {};
  auto plan = [&] { f.Plan(kTileRows); };
  auto line = [&](const char* what, double ms) {
    std::printf("pipe set=%s R=%d routing=%s %-18s %.4f ms  (%.0f ms-per-32k)\n", f.set.name, f.R, tag, what, ms,
                ms * scale);
    std::fflush(stdout);
  };
  bench::Sampler clocks;
  clocks.Start();
  for (int p = 0; p < pairs; ++p) {
    line("B-gateup", ColdTimePre(none, [&] { f.TileGateUp(); }, cold));
    line("pipe1-gateup", ColdTimePre(none, [&] { f.TileGateUp<kTileMutPipe1>(); }, cold));
    line("pipe2-gateup", ColdTimePre(none, [&] { f.TileGateUp<kTileMutPipe2>(); }, cold));
    line("fast1-gateup", ColdTimePre(none, [&] { f.TileGateUp<kTileMutFast1>(); }, cold));
    line("fast2-gateup", ColdTimePre(none, [&] { f.TileGateUp<kTileMutFast2>(); }, cold));
    line("B-down", ColdTimePre(plan, [&] { f.TileDown(); }, cold));
    line("pipe1-down", ColdTimePre(plan, [&] { f.TileDown<kTileMutPipe1>(); }, cold));
    line("pipe2-down", ColdTimePre(plan, [&] { f.TileDown<kTileMutPipe2>(); }, cold));
    line("fast1-down", ColdTimePre(plan, [&] { f.TileDown<kTileMutFast1>(); }, cold));
    line("fast2-down", ColdTimePre(plan, [&] { f.TileDown<kTileMutFast2>(); }, cold));
    for (const int lead : {0, 16, 32, 64}) {
      const std::string b = "B-gateup-lead" + std::to_string(lead), q = "fast1-gateup-lead" + std::to_string(lead),
                        r = "fast2-gateup-lead" + std::to_string(lead);
      line(b.c_str(), ColdTimePre(none, [&] { f.TileGateUp(lead); }, cold));
      line(q.c_str(), ColdTimePre(none, [&] { f.TileGateUp<kTileMutFast1>(lead); }, cold));
      line(r.c_str(), ColdTimePre(none, [&] { f.TileGateUp<kTileMutFast2>(lead); }, cold));
    }
    line("A-gateup", ColdTimePre(none, [&] { f.TrunkGateUp(); }, cold));
    line("A-down", ColdTimePre([&] { f.Plan(96); }, [&] { f.TrunkDown(); }, cold));
  }
  clocks.Stop();
  std::printf("clocks-pipe %s\n", clocks.Summary().c_str());
  cold.Free();
}

std::vector<int> List(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p;) {
    char* e = nullptr;
    const long x = std::strtol(p, &e, 10);
    if (e == p || x < 1 || x > 32768) return {};
    v.push_back((int)x);
    if (*e && *e != ',') return {};
    p = *e ? e + 1 : e;
  }
  return v;
}
std::vector<KSet> Sets(const std::string& s) {
  if (s == "all") return {std::begin(kSets), std::end(kSets)};
  std::vector<KSet> v;
  size_t p = 0;
  while (p <= s.size()) {
    const size_t q = std::min(s.find(',', p), s.size());
    const std::string n = s.substr(p, q - p);
    bool found = false;
    for (const auto& k : kSets) if (n == k.name) { v.push_back(k); found = true; }
    if (!found) return {};
    p = q + 1;
  }
  return v;
}
void PrintResources() {
  hipFuncAttributes at{};
  HIP_OK(hipFuncGetAttributes(&at, reinterpret_cast<const void*>(PrefillMoeTilePairKernel<6>)));
  std::printf("resources TilePair<6> vgpr=%d max-threads=%d lds=%zu scratch=%zu\n", at.numRegs, at.maxThreadsPerBlock,
              at.sharedSizeBytes, at.localSizeBytes);
  HIP_OK(hipFuncGetAttributes(&at, reinterpret_cast<const void*>(PrefillMoeTilePairKernel<6, kTileMutPipe2>)));
  std::printf("resources TilePair<6,pipe2> vgpr=%d lds=%zu scratch=%zu\n", at.numRegs, at.sharedSizeBytes, at.localSizeBytes);
  HIP_OK(hipFuncGetAttributes(&at, reinterpret_cast<const void*>(PrefillMoeTilePairKernel<6, kTileMutPipe3>)));
  std::printf("resources TilePair<6,pipe3> vgpr=%d lds=%zu scratch=%zu\n", at.numRegs, at.sharedSizeBytes, at.localSizeBytes);
  HIP_OK(hipFuncGetAttributes(&at, reinterpret_cast<const void*>(PrefillMoeTileDownKernel<6, kTileMutPipe2>)));
  std::printf("resources TileDown<6,pipe2> vgpr=%d lds=%zu scratch=%zu\n", at.numRegs, at.sharedSizeBytes, at.localSizeBytes);
  HIP_OK(hipFuncGetAttributes(&at, reinterpret_cast<const void*>(PrefillMoeTileDownKernel<6, kTileMutPipe3>)));
  std::printf("resources TileDown<6,pipe3> vgpr=%d lds=%zu scratch=%zu\n", at.numRegs, at.sharedSizeBytes, at.localSizeBytes);
  for (const auto& [name, fn] : {std::pair{"TilePair<6,fast1>", (const void*)PrefillMoeTilePairKernel<6, kTileMutFast1>},
                                 std::pair{"TilePair<6,fast2>", (const void*)PrefillMoeTilePairKernel<6, kTileMutFast2>},
                                 std::pair{"TileDown<6,fast1>", (const void*)PrefillMoeTileDownKernel<6, kTileMutFast1>},
                                 std::pair{"TileDown<6,fast2>", (const void*)PrefillMoeTileDownKernel<6, kTileMutFast2>}}) {
    HIP_OK(hipFuncGetAttributes(&at, fn));
    std::printf("resources %s vgpr=%d lds=%zu scratch=%zu\n", name, at.numRegs, at.sharedSizeBytes, at.localSizeBytes);
  }
  HIP_OK(hipFuncGetAttributes(&at, reinterpret_cast<const void*>(PrefillMoeTilePairKernel<6, kTileMutDeep>)));
  std::printf("resources TilePair<6,deep> vgpr=%d lds=%zu scratch=%zu\n", at.numRegs, at.sharedSizeBytes, at.localSizeBytes);
  HIP_OK(hipFuncGetAttributes(&at, reinterpret_cast<const void*>(PrefillMoeTileDownKernel<6, kTileMutDeep>)));
  std::printf("resources TileDown<6,deep> vgpr=%d lds=%zu scratch=%zu\n", at.numRegs, at.sharedSizeBytes, at.localSizeBytes);
  HIP_OK(hipFuncGetAttributes(&at, reinterpret_cast<const void*>(PrefillMoeTileDownKernel<6>)));
  std::printf("resources TileDown<6> vgpr=%d max-threads=%d lds=%zu scratch=%zu\n", at.numRegs, at.maxThreadsPerBlock,
              at.sharedSizeBytes, at.localSizeBytes);
}
// p1-decode and p1-critpath: arms of the engine's MoE expert
// kernels (the fat gate/up kernel and the tile down kernel, KMAX 6), each measured against the engine kernels (base).
// One harness, two arm tables: `dec` (exact variants and levers; rule equal: their bytes must be trunk's) and `crit`
// (removal mutants; rule info: wrong outputs on purpose, the time they save bounds the cost they remove).
// Arm names say what the arm runs in this build (kTileDecodeSched, engine/layers/prefill_moe_tile.h):
// - base, aa: the engine kernels (aa: base again, the noise floor). Since p1-decsw they run the scheduled decode.
// - old (switch on) or sched (switch off): the other form of the decode switch (MUT kMutDecodeSched): with the switch on
//   it is the old decode, a reference arm (not a lever); with the switch off it is the scheduled decode variant.
// - sched-p0, sched-d3, early, early1, early2: the scheduled decode (kMutDecodeNew) with one more exact change.
// Each arm prints a crit-arm line with the decode form of its two kernels (sched, old or none), so a read-out
// does not take a reference arm for a lever.
using DecFn = void (*)(Fixture&);
template <int MUT, int D = kPcDepth, int PRIO = kPcPrio> void DecGu(Fixture& f) {
  if constexpr (MUT == 0 && D == kPcDepth && PRIO == kPcPrio) LaunchPrefillMoePcEngine(6, f.a, kTileFxLead);
  else PrefillMoePcFatPairKernel<6, MUT, kPcFatXDist, D, PRIO><<<512, 32 * kFatPairWaves>>>(f.a, kTileFxLead);
}
template <int MUT> void DecDown(Fixture& f) {
  if constexpr (MUT == 0) f.PcDown();
  else PrefillMoeTileDownKernel<6, MUT><<<1024, 256>>>(f.a.g_down, f.a.i_down, f.a.item_counts + 1);
}
// The decode form of a kernel MUT: none (the decode or the whole producer removed), sched or old.
template <int G> constexpr const char* GuDecode() {
  return G == kPcMutNoDecode || PcBareProducer(G) ? "none" : TileSched<G>() ? "sched" : "old";
}
template <int D> constexpr const char* DownDecode() {
  return D == kTileMutNoDecode || D == kTileMutBare ? "none" : TileSched<D>() ? "sched" : "old";
}
// The name of the arm with MUT kMutDecodeSched in both kernels (the other form of the switch).
constexpr const char* kDecSwitchArm = kTileDecodeSched ? "old" : "sched";
struct DecArm {
  const char* name;
  DecFn gu, down;
  const void *gk, *dk;  // the kernels (resources)
  const char* rule;     // byte rule: equal (trunk's bytes) or info (wrong outputs on purpose)
  const char *gdec, *ddec;  // the decode form of the gate/up and the down kernel
};
// G, GD, GP: the gate/up MUT, producer depth and producer priority; D: the down MUT.
template <int G, int D, int GD = kPcDepth, int GP = kPcPrio> DecArm MakeDecArm(const char* name, const char* rule) {
  return {name, DecGu<G, GD, GP>, DecDown<D>,
          reinterpret_cast<const void*>(PrefillMoePcFatPairKernel<6, G, kPcFatXDist, GD, GP>),
          reinterpret_cast<const void*>(PrefillMoeTileDownKernel<6, D>), rule, GuDecode<G>(), DownDecode<D>()};
}
const std::vector<DecArm>& DecArms() {
  static const std::vector<DecArm> arms = {
      MakeDecArm<0, 0>("base", "equal"),
      MakeDecArm<0, 0>("aa", "equal"),
      MakeDecArm<kMutDecodeSched, kMutDecodeSched>(kDecSwitchArm, "equal"),
      MakeDecArm<kPcMutNoDecode, kTileMutNoDecode>("nodecode", "info"),
      // the scheduled decode with the gate/up producers at priority 0 or at register depth 3, and the down kernel with
      // the early fragment reads (all exact)
      MakeDecArm<kMutDecodeNew, kMutDecodeNew, kPcDepth, 0>("sched-p0", "equal"),
      MakeDecArm<kMutDecodeNew, kMutDecodeNew, 3, kPcPrio>("sched-d3", "equal"),
      MakeDecArm<kMutDecodeNew, kMutDecodeEarly>("early", "equal"),
      // early with fewer fragments read before the decode (1 or 2 A fragments and the first B fragment)
      MakeDecArm<kMutDecodeNew, kMutDecodeEarly1>("early1", "equal"),
      MakeDecArm<kMutDecodeNew, kMutDecodeEarly2>("early2", "equal"),
  };
  return arms;
}
// p1-critpath: each mutant removes one cost and keeps the rest (the comments at kPcMutNoDecode .. kPcMutNoPad).
const std::vector<DecArm>& CritArms() {
  static const std::vector<DecArm> arms = {
      MakeDecArm<0, 0>("base", "equal"),
      MakeDecArm<0, 0>("aa", "equal"),
      MakeDecArm<kPcMutNoDecode, kTileMutNoDecode>("nodecode", "info"),  // raw weight words stored, no decode VALU
      MakeDecArm<kPcMutNoStage, 0>("nostage", "info"),                   // gate/up epilogue stage stores skipped
      MakeDecArm<kPcMutNoPad, 0>("nopad", "info"),                       // stage stores with no bank conflict
      MakeDecArm<kPcMutNoOut, kTileMutNoStore>("noout", "info"),         // gate/up xd and down pd stores skipped
      MakeDecArm<kPcMutNoSync, kTileMutNoSync>("nosync", "info"),        // k-loop barriers removed in all waves
      MakeDecArm<kPcMutBare, kTileMutBare>("bare", "info"),              // ref: producers take only the barriers
      MakeDecArm<kPcMutWmmaOnly, kTileMutBare>("wmma", "info"),          // ref floor: WMMAs and barriers only
  };
  return arms;
}
double DecEpochS() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
void DecResources(const std::vector<DecArm>& arms) {
  std::printf("crit-switch kTileDecodeSched=%d switch-arm=%s\n", (int)kTileDecodeSched, kDecSwitchArm);
  for (const auto& arm : arms) {
    std::printf("crit-arm arm=%s gateup-decode=%s down-decode=%s rule=%s\n", arm.name, arm.gdec, arm.ddec, arm.rule);
    for (int k = 0; k < 2; ++k) {
      hipFuncAttributes at{};
      HIP_OK(hipFuncGetAttributes(&at, k == 0 ? arm.gk : arm.dk));
      std::printf("crit-res arm=%s kernel=%s vgpr=%d lds=%zu scratch=%zu\n", arm.name, k == 0 ? "gateup" : "down",
                  at.numRegs, at.sharedSizeBytes, at.localSizeBytes);
    }
  }
  std::fflush(stdout);
}
// Bytes: trunk's RM6 path gives xd and pd; each arm runs gate/up (its xd), then down on trunk's xd (its pd). Rules:
// equal (a pass condition), differ (a pass condition: the check sees a change), info (no condition). Twice per arm, so
// a variant with a race shows it.
bool DecCheck(Fixture& f, const char* tag, const std::vector<DecArm>& arms) {
  const char* name = f.set.name;
  f.xd.Nan(); f.pd.Nan(); f.xg.Nan(); f.xu.Nan();
  f.TrunkGateUp(); f.TrunkDown(); CheckLaunch();
  const auto xref = f.xd.Get(), pref = f.PdWritten(f.pd.Get());
  bool ok = Unwritten(xref) == 0 && Unwritten(pref) == 0;
  std::printf("crit-check set=%s R=%d routing=%s variant=trunk unwritten-xd=%zu unwritten-pd=%zu xd-hash=%016llx "
              "pd-hash=%016llx %s\n", name, f.R, tag, Unwritten(xref), Unwritten(pref), Fnv(xref), Fnv(pref),
              ok ? "PASS" : "FAIL");
  auto arm = [&](const char* variant, DecFn gu, DecFn down, const char* rule) {
    f.xd.Nan(); f.pd.Nan();
    f.Plan(kTileRows);
    gu(f); CheckLaunch();
    const auto xd = f.xd.Get();
    HIP_OK(hipMemcpy(f.xd.p, xref.data(), xref.size() * sizeof(__half), hipMemcpyHostToDevice));
    down(f); CheckLaunch();
    const auto pdv = f.PdWritten(f.pd.Get());
    const size_t dx = DiffCount(xd, xref), dp = DiffCount(pdv, pref);
    bool pass = true;
    if (!std::strcmp(rule, "equal")) pass = dx == 0 && dp == 0 && Unwritten(xd) == 0 && Unwritten(pdv) == 0;
    if (!std::strcmp(rule, "differ")) pass = dx + dp > 0;
    ok = ok && pass;
    std::printf("crit-check set=%s R=%d routing=%s variant=%s xd-diff=%zu/%zu pd-diff=%zu/%zu unwritten-xd=%zu "
                "unwritten-pd=%zu xd-hash=%016llx pd-hash=%016llx rule=%s %s\n", name, f.R, tag, variant, dx, xd.size(),
                dp, pdv.size(), Unwritten(xd), Unwritten(pdv), Fnv(xd), Fnv(pdv), rule, pass ? "PASS" : "FAIL");
  };
  for (int rep = 0; rep < 2; ++rep)
    for (const auto& a : arms) arm(a.name, a.gu, a.down, a.rule);
  arm("mut-korder-down", DecGu<0>, DecDown<3>, "differ");
  arm("mut-korder-gateup", DecGu<3>, DecDown<0>, "differ");
  std::printf("crit-checks set=%s R=%d routing=%s %s\n", name, f.R, tag, ok ? "PASS" : "FAIL");
  std::fflush(stdout);
  return ok;
}
struct DecRow {
  double g, d, t0, t1, mhz, watts, temp;
};
// One measure of an arm: gate/up, then down, each a cold median of 7 (ColdTimePre: MALL flushed, the plan kernel
// before the timer), with the hwmon clock, power and temperature of the measure.
DecRow DecMeasure(Fixture& f, const DecArm& arm, const gm::ColdCache& cold) {
  auto plan = [&] { f.Plan(kTileRows); };
  DecRow r{};
  bench::Sampler s;
  r.t0 = DecEpochS();
  s.Start();
  r.g = ColdTimePre(plan, [&] { arm.gu(f); }, cold);
  r.d = ColdTimePre(plan, [&] { arm.down(f); }, cold);
  s.Stop();
  r.t1 = DecEpochS();
  r.mhz = s.MeanMhz(); r.watts = s.MeanWatts(); r.temp = s.MaxTemp();
  return r;
}
// ABAB: in pair p, base and arm m one after the other; the order flips with the parity of p + m.
void DecTime(Fixture& f, const char* tag, int pairs, double settle, const std::vector<DecArm>& arms) {
  const auto cold = gm::MakeColdCache();
  // Settle (SETTLE_S seconds, default 20): the base arm in a loop first, so the ABAB starts at the steady power
  // limit (the first run of Nov 26 started at 105 to 113 W and its first 3 pairs carried the A/A outlier).
  {
    bench::Sampler s;
    int calls = 0;
    const double t0 = DecEpochS();
    s.Start();
    while (DecEpochS() - t0 < settle) {
      gm::Flush(&cold);
      f.Plan(kTileRows);
      arms[0].gu(f);
      f.Plan(kTileRows);
      arms[0].down(f);
      HIP_OK(hipDeviceSynchronize());
      ++calls;
    }
    CheckLaunch();
    s.Stop();
    std::printf("crit-settle set=%s R=%d routing=%s secs=%.1f calls=%d sclk=%.0f pwr=%.1f temp=%.0f\n", f.set.name, f.R,
                tag, settle, calls, s.MeanMhz(), s.MeanWatts(), s.MaxTemp());
  }
  for (const auto& a : arms) DecMeasure(f, a, cold);  // warm-up of each arm at this shape
  auto row = [&](int p, const DecArm& a, const char* vs, int slot, const DecRow& r) {
    std::printf("crit-row set=%s R=%d routing=%s pair=%d arm=%s vs=%s slot=%d gateup=%.4f down=%.4f t0=%.3f t1=%.3f "
                "sclk=%.0f pwr=%.1f temp=%.0f\n", f.set.name, f.R, tag, p, a.name, vs, slot, r.g, r.d, r.t0, r.t1, r.mhz,
                r.watts, r.temp);
  };
  for (int p = 0; p < pairs; ++p) {
    for (size_t m = 1; m < arms.size(); ++m) {
      const bool base_first = (p + m) % 2 == 0;
      const DecArm& first = base_first ? arms[0] : arms[m];
      const DecArm& second = base_first ? arms[m] : arms[0];
      const DecRow r0 = DecMeasure(f, first, cold), r1 = DecMeasure(f, second, cold);
      row(p, first, arms[m].name, 0, r0);
      row(p, second, arms[m].name, 1, r1);
    }
    std::fflush(stdout);
  }
  cold.Free();
}
// Clock and power: each arm of ORDER in a loop for SECS seconds (flush, plan, gate/up, plan, down), base between the
// others.
void DecPower(Fixture& f, const char* tag, double secs, const std::vector<DecArm>& arms,
              const std::vector<const char*>& order) {
  const auto cold = gm::MakeColdCache();
  int idx = 0;
  for (const char* want : order) {
    const DecArm* arm = nullptr;
    for (const auto& a : arms) if (!std::strcmp(a.name, want)) arm = &a;
    if (!arm) continue;
    bench::EventTimer t;
    std::vector<double> g, d;
    bench::Sampler s;
    const double t0 = DecEpochS();
    s.Start();
    while (DecEpochS() - t0 < secs) {
      gm::Flush(&cold);
      f.Plan(kTileRows);
      HIP_OK(hipDeviceSynchronize());
      t.Begin();
      arm->gu(f);
      g.push_back(t.EndMs());
      f.Plan(kTileRows);
      HIP_OK(hipDeviceSynchronize());
      t.Begin();
      arm->down(f);
      d.push_back(t.EndMs());
    }
    CheckLaunch();
    s.Stop();
    const double t1 = DecEpochS();
    std::printf("crit-power set=%s R=%d routing=%s idx=%d arm=%s calls=%zu gateup=%.4f down=%.4f t0=%.3f t1=%.3f "
                "sclk=%.0f pwr=%.1f temp=%.0f\n", f.set.name, f.R, tag, idx++, arm->name, g.size(), bench::Median(g),
                bench::Median(d), t0, t1, s.MeanMhz(), s.MeanWatts(), s.MaxTemp());
    std::fflush(stdout);
  }
  cold.Free();
}
}  // namespace

// moe-lds: the engine's MoE kernels (fat gate/up, tile down) with the A plane skew of the
// switch (B: kTileAPlaneSkew16, 16 words) against the old 32-word skew (A: kTileMutPlane32). Bytes first: trunk's RM6
// path, B and A give the same xd and pd (pd from trunk's xd), and a k-order mutant differs. Then ABAB cold timing,
// the arm order flipped in every other pair. Exit 1 on a byte FAIL (no timing then).
bool LdsCheck(Fixture& f, const char* tag) {
  const char* name = f.set.name;
  f.xd.Nan(); f.pd.Nan(); f.xg.Nan(); f.xu.Nan();
  f.TrunkGateUp(); f.TrunkDown(); CheckLaunch();
  const auto xref = f.xd.Get(), pref = f.PdWritten(f.pd.Get());
  bool ok = Unwritten(xref) == 0 && Unwritten(pref) == 0;
  std::printf("lds-check set=%s R=%d routing=%s variant=trunk unwritten-xd=%zu unwritten-pd=%zu xd-hash=%016llx "
              "pd-hash=%016llx %s\n", name, f.R, tag, Unwritten(xref), Unwritten(pref), Fnv(xref), Fnv(pref),
              ok ? "PASS" : "FAIL");
  auto arm = [&](const char* variant, auto gateup, auto down, bool expect_equal) {
    f.xd.Nan(); f.pd.Nan();
    gateup(); CheckLaunch();
    const auto xd = f.xd.Get();
    HIP_OK(hipMemcpy(f.xd.p, xref.data(), xref.size() * sizeof(__half), hipMemcpyHostToDevice));
    down(); CheckLaunch();
    const auto pdv = f.PdWritten(f.pd.Get());
    const size_t dx = DiffCount(xd, xref), dp = DiffCount(pdv, pref);
    const bool pass = expect_equal ? dx == 0 && dp == 0 : dx + dp > 0;
    ok = ok && pass;
    std::printf("lds-check set=%s R=%d routing=%s variant=%s xd-diff=%zu/%zu pd-diff=%zu/%zu xd-hash=%016llx "
                "pd-hash=%016llx expect=%s %s\n", name, f.R, tag, variant, dx, xd.size(), dp, pdv.size(), Fnv(xd),
                Fnv(pdv), expect_equal ? "equal" : "differ", pass ? "PASS" : "FAIL");
  };
  arm(kTileAPlaneSkew16 ? "skew16" : "switch-skew32", [&] { f.PcGateUp(); }, [&] { f.PcDown(); }, true);
  arm("skew32", [&] { f.PcGateUpPlane32(); }, [&] { f.TileDown<kTileMutPlane32>(); }, true);
  arm("mut-korder-down", [&] { f.PcGateUp(); }, [&] { f.TileDown<3>(); }, false);
  std::printf("lds-checks set=%s R=%d routing=%s %s\n", name, f.R, tag, ok ? "PASS" : "FAIL");
  std::fflush(stdout);
  return ok;
}
void LdsTime(Fixture& f, const char* tag, int pairs) {
  const auto cold = gm::MakeColdCache();
  const double scale = 48. * 32768 / f.R;
  auto none = [] {};
  auto plan = [&] { f.Plan(kTileRows); };
  auto ga = [&] { return ColdTimePre(none, [&] { f.PcGateUpPlane32(); }, cold); };
  auto gb = [&] { return ColdTimePre(none, [&] { f.PcGateUp(); }, cold); };
  auto da = [&] { return ColdTimePre(plan, [&] { f.TileDown<kTileMutPlane32>(); }, cold); };
  auto db = [&] { return ColdTimePre(plan, [&] { f.PcDown(); }, cold); };
  ga(); gb(); da(); db();  // warm-up of each arm at this shape
  std::vector<double> GA, GB, DA, DB;
  bench::Sampler clocks;
  clocks.Start();
  for (int p = 0; p < pairs; ++p) {
    if (p % 2 == 0) { GA.push_back(ga()); GB.push_back(gb()); DA.push_back(da()); DB.push_back(db()); }
    else { GB.push_back(gb()); GA.push_back(ga()); DB.push_back(db()); DA.push_back(da()); }
    std::printf("lds-abab set=%s R=%d routing=%s pair=%d gateup A=%.4f B=%.4f down A=%.4f B=%.4f ms-per-call\n",
                f.set.name, f.R, tag, p, GA.back(), GB.back(), DA.back(), DB.back());
    std::fflush(stdout);
  }
  clocks.Stop();
  const double ma = bench::Median(GA), mb = bench::Median(GB), na = bench::Median(DA), nb = bench::Median(DB);
  int gw = 0, dw = 0;
  for (int p = 0; p < pairs; ++p) { gw += GB[p] < GA[p]; dw += DB[p] < DA[p]; }
  const double sum_a = ma + na, sum_b = mb + nb;
  std::printf("lds-result set=%s R=%d routing=%s gateup A=%.4f B=%.4f ms B/A=%.4f B-faster=%d/%d | down A=%.4f B=%.4f "
              "ms B/A=%.4f B-faster=%d/%d | moe save=%.2f%% %.0f ms-per-32k\n", f.set.name, f.R, tag, ma, mb, mb / ma, gw,
              pairs, na, nb, nb / na, dw, pairs, 100. * (sum_a - sum_b) / sum_a, (sum_a - sum_b) * scale);
  std::printf("clocks-lds %s\n", clocks.Summary().c_str());
  std::fflush(stdout);
  cold.Free();
}

// prefill_moe_tile check|time|timex|lds|dec|crit ROWS SETS [uniform|skew|PATH@LAYER] [PAIRS] [SETTLE_S] [POWER_S]
//   dec: p1-decode arms, crit: p1-critpath removal mutants (set 4/6 only; one harness):
//   the switch and arm lines, resources, the byte check of every arm (twice), and only
//   when every check passes a settle loop (SETTLE_S, default 20 s), ABAB of every arm against base, then the clock and
//   power loops (POWER_S seconds per arm, default 0: none). After a byte FAIL the set has no timing rows (exit 1).
//   SETTLE_S and POWER_S are arguments, not environment variables (tests/test_switch_ratchet.py: no new raw read).
//   lds: moe-lds byte check (trunk = A plane skew of the switch = skew 32) and ABAB of the two skews (only KMAX 5 and 6
//   sets). The switch kTileAPlaneSkew16 is off, so both lds arms run the same 32-word skew (an A/A).
//   timex: time with every routed job's X from one cached region (diagnostic bound of the X traffic, not exact).
//   PATH@LAYER: the routes of one layer from a CRUCIBLE_ROUTE_DUMP file of the engine (prefill-rows).
//   ROWS: "16,96,97,4096,8192,32768"; SETS: "all" or names ("3/5,4/6,3.5/7"); time adds ABAB pairs (default 5).
int main(int argc, char** argv) {
  if (argc < 4 || argc > 8) return 2;
  const std::string mode = argv[1];
  const auto rows = List(argv[2]);
  const auto sets = Sets(argv[3]);
  Routing routing;
  routing.skew = argc > 4 && !std::strcmp(argv[4], "skew");
  const std::string route_arg = argc > 4 ? argv[4] : "uniform";
  if ((mode != "check" && mode != "time" && mode != "timex" && mode != "split" && mode != "conc" && mode != "pipe" &&
       mode != "lds" && mode != "dec" && mode != "crit") || rows.empty() || sets.empty()) return 2;
  if (const auto at = route_arg.rfind('@'); at != std::string::npos) {
    routing.dumped = DumpedRoutes(route_arg.substr(0, at), std::atoi(route_arg.c_str() + at + 1));
    for (const int R : rows)
      if ((size_t)R * kTopK > routing.dumped.size()) {
        std::fprintf(stderr, "%s has %zu routed rows, fewer than %d\n", route_arg.c_str(), routing.dumped.size() / kTopK, R);
        return 2;
      }
  } else if (!routing.skew && route_arg != "uniform") {
    return 2;
  }
  const int pairs = argc > 5 ? std::atoi(argv[5]) : 5;
  if (pairs < 1 || pairs > 50) return 2;
  const double settle = argc > 6 ? std::atof(argv[6]) : 20.0, power_s = argc > 7 ? std::atof(argv[7]) : 0.0;
  const bool arms_mode = mode == "dec" || mode == "crit";
  if (argc > 6 && !arms_mode) return 2;
  if (settle < 0 || settle > 600 || power_s < 0 || power_s > 600) return 2;
  const auto& arms = mode == "crit" ? CritArms() : DecArms();
  const std::vector<const char*> power_order =
      mode == "crit" ? std::vector<const char*>{"base", "nodecode", "base", "nosync", "base", "bare", "base", "wmma",
                                                "base", "noout", "base"}
                     : std::vector<const char*>{"base", kDecSwitchArm, "base", "nodecode", "base", kDecSwitchArm, "base"};
  if (arms_mode) DecResources(arms);
  else PrintResources();
  const std::string tag_text = routing.dumped.empty() ? (routing.skew ? "skew" : "uniform")
                                                      : "layer" + route_arg.substr(route_arg.rfind('@') + 1);
  const char* tag = tag_text.c_str();
  int rc = 0;
  for (const auto& s : sets)
    for (const int R : rows) {
      Fixture f(R, s, routing);
      if (mode == "timex") {  // diagnostic: X from the caches, no byte check (the outputs are not the model's)
        f.alias_x = true;
        Time(f, (tag_text + "-xcache").c_str(), pairs);
        continue;
      }
      if (mode == "pipe") {
        Pipe(f, tag, pairs);
        continue;
      }
      if (arms_mode) {
        if (f.set.Kmax() != 6 || !PrefillMoePcOn(6)) {
          std::printf("%s set=%s skipped (KMAX 6 only)\n", mode.c_str(), f.set.name);
          continue;
        }
        const bool ok = DecCheck(f, tag, arms);
        rc |= !ok;
        if (!ok) continue;  // no timing after a byte FAIL: the read-outs FAIL on the crit-check lines
        DecTime(f, tag, pairs, settle, arms);
        if (power_s > 0) DecPower(f, tag, power_s, arms, power_order);
        std::printf("crit-done set=%s R=%d routing=%s\n", f.set.name, f.R, tag);
        continue;
      }
      if (mode == "conc") {
        Conc(f, tag, pairs);
        continue;
      }
      if (mode == "lds") {
        if (!PrefillMoePcOn(f.set.Kmax())) { std::printf("lds set=%s skipped (no pc kernels)\n", f.set.name); continue; }
        const bool ok = LdsCheck(f, tag);
        rc |= !ok;
        if (ok) LdsTime(f, tag, pairs);
        continue;
      }
      if (mode == "split") {  // diagnostic: no byte check of the mutants (their outputs are wrong on purpose)
        Split(f, tag, pairs);
        continue;
      }
      const bool ok = Check(f, tag);
      rc |= !ok;
      if (ok && mode == "time") Time(f, tag, pairs);
    }
  std::printf("bench %s %s\n", rc ? "FAIL" : "PASS", tag);
  return rc;
}
