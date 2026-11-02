// qsa-attn: the QSA prefill attention (union build + union flash with the fused gate) at the engine's shapes.
//
//   qsa_attn check                    byte check of the engine kernels against the verbatim copies of the v0.30.0 kernels
//                                     (bench/kernels/qsa_attn_trunk.h) on 9 shapes; mutants must DIFF
//   qsa_attn time START [PAIRS]       one bulk launch of 8,192 rows from position START (4 core parts of 2,048
//                                     rows): ABBA rounds of trunk against the staged kernel, sclk per sample
//   qsa_attn capture PREFIX [PAIRS]   the same on one captured part (CRUCIBLE_QSA_CAPTURE_INDEX_ONLY files)
//
// Arms: T = trunk build + QsaUnionHalfQKernel<true> per part (the engine today); S = engine build +
// QsaUnionStageKernel<true> per part (S0: its Map 0 lanes); T1 and S1 = one build + flash call over the whole
// launch (exact: tiles start on the same 4-row grid); B = build only; SM = stage flash with the loads and LDS
// stores only (Mut 4, wrong output); TL and SL = flash only, every block remapped into the first kLocal blocks
// (L2 resident, wrong output): the time without DRAM and MALL traffic. Selections are synthetic with the
// measured overlap of real ones (4-row union about 860 blocks), or a capture.
// Lines: "check ... SAME|DIFF", "mutant ... DIFF|SAME", "time ...", "result ...", last line "bench PASS|FAIL".
#include "prefill_common.h"
#include "kernels/qsa_attn_trunk.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>

using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

namespace {
constexpr int kPart = 2048, kLaunch = 4 * kPart, kLocal = 128;  // kLocal blocks of K and V: 1 MB, L2 resident
constexpr int kOut = kQHeads * kHeadDim;  // xh_o row: 24 heads x 256
constexpr double kDrift = .25;

template <class T> std::vector<T> Read(const std::string& path, size_t count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in || in.tellg() != (std::streamoff)(count * sizeof(T))) {
    std::fprintf(stderr, "wrong capture size: %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<T> x(count);
  in.seekg(0);
  if (!in.read(reinterpret_cast<char*>(x.data()), count * sizeof(T))) std::exit(2);
  return x;
}

// Selections of rows [pos0, pos0 + R): rows with at most 512 completed blocks are dense (-1). The sparse rows
// keep a set of 512 blocks that drifts: each row replaces a member with probability `drift` (half from the
// last 512 blocks, half anywhere), the newest completed block joins, and blocks 0..3 stay (attention sink).
// kDrift gives a 4-row union of 862 blocks and 108 steps at 120K (854 at 24K), as the real 16K to 128K captures
// (860 blocks).
void Synthetic(int pos0, int R, double drift, std::vector<int>& sel, std::vector<int>& nsel) {
  std::mt19937 rng(7 + pos0);
  std::uniform_real_distribution<double> u01(0, 1);
  sel.assign((size_t)R * kSelBlocks, 0);
  nsel.assign(R, -1);
  std::vector<char> in;
  std::vector<int> set;
  int have = 0;  // completed blocks the set knows
  for (int r = 0; r < R; ++r) {
    const int nb = (pos0 + r + 1) / 4;
    if (nb <= kSelBlocks) continue;
    if (set.empty()) {
      in.assign(nb + R / 4 + 8, 0);
      for (int b = nb - kSelBlocks; b < nb; ++b) { set.push_back(b); in[b] = 1; }
      for (int b = 0; b < 4; ++b) if (!in[b]) { in[set[4 + b]] = 0; set[4 + b] = b; in[b] = 1; }
      have = nb;
    }
    auto fresh = [&] {
      while (true) {
        const int b = u01(rng) < .5 ? nb - 1 - (int)(rng() % kSelBlocks) : (int)(rng() % nb);
        if (b >= 0 && !in[b]) return b;
      }
    };
    for (; have < nb; ++have) {  // the newest completed block joins
      size_t i;
      do i = rng() % set.size(); while (set[i] < 4);
      in[set[i]] = 0; set[i] = have; in[have] = 1;
    }
    for (auto& b : set) {
      if (b < 4 || b == nb - 1 || u01(rng) >= drift) continue;
      const int f = fresh();
      in[b] = 0; b = f; in[f] = 1;
    }
    std::vector<int> row(set);
    std::sort(row.begin(), row.end());
    std::copy(row.begin(), row.end(), sel.begin() + (size_t)r * kSelBlocks);
    nsel[r] = kSelBlocks;
  }
}

struct Work { double tiles = 0, blocks = 0, steps = 0, live = 0; };
Work Count(const std::vector<int>& list, const std::vector<int>& count) {
  Work w;
  for (size_t tile = 0; tile < count.size(); ++tile) {
    const int n = count[tile] * 4;
    w.tiles += 1; w.blocks += count[tile];
    for (int start = 0; start < n; start += kUnionKeys) {
      int sm = 0;
      for (int i = start; i < std::min(n, start + kUnionKeys); ++i) sm |= list[tile * kUnionCap + i / 4] >> 24 & 0xff;
      w.steps += 1;
      w.live += (sm & 3 ? 1 : 0) + (sm & 6 ? 1 : 0) + (sm & 12 ? 1 : 0);
    }
  }
  return w;
}

// One problem: rows [pos0, pos0 + R) over a cache of pos0 + R positions, selections given.
struct Problem {
  int pos0, R;
  Buffer<__half> k, v, qhi, suh, xh;
  Buffer<float> gates, context;
  Buffer<int> sel, nsel, list, count;
  Problem(int p0, int rows, const std::vector<int>& hsel, const std::vector<int>& hnsel)
      : pos0(p0), R(rows),
        k(Half(Random((size_t)(p0 + rows) * kQsaKvWidth, 1.f, 35))),
        v(Half(Random((size_t)(p0 + rows) * kQsaKvWidth, .4f, 36))),
        qhi(Half(Random((size_t)rows * kOut, .4f, 31))),
        suh(Half(Suh())), xh((size_t)rows * kOut),
        gates(Random((size_t)rows * kQsaQWidth, 1.f, 33)), context((size_t)rows * kOut),
        sel(hsel), nsel(hnsel), list(QsaUnionListInts(rows)), count((rows + kUnionRows - 1) / kUnionRows) {}
  static std::vector<__half> Half(const std::vector<float>& x) { return prefill_bench::Half(x); }
  static std::vector<float> Suh() {
    auto s = Random(kOut, .3f, 34);
    for (auto& x : s) x += 1.f;
    return s;
  }
  // The arguments of the part at row offset o with n rows (the engine's QsaOutputRows view).
  QsaPrefill Args(int o, int n) {
    QsaArgs a{};
    a.c.k = k.p; a.c.v = v.p; a.pos0 = pos0 + o; a.R = n;
    a.sel = sel.p + (size_t)o * kSelBlocks; a.nsel = nsel.p + o;
    a.q = gates.p + (size_t)o * kQsaQWidth; a.suh_o = suh.p; a.xh_o = xh.p + (size_t)o * kOut;
    return QsaPrefill{a, context.p + (size_t)o * kOut, qhi.p + (size_t)o * kOut, nullptr};
  }
  QsaUnion Union(int o) { return {list.p + (size_t)(o / kUnionRows) * kUnionCap, count.p + o / kUnionRows}; }
};

enum Kernel { kTrunk, kStage, kStageM1, kStageM2, kStageM3, kStage0, kStageMem };

void Build(bool trunk, const QsaArgs& q, QsaUnion u) {
  const int tiles = (q.R + kUnionRows - 1) / kUnionRows, blocks = (q.pos0 + q.R - 1) / 4 + 1;
  if (trunk) trunk27e0::QsaUnionBuildKernel<<<tiles, 256, blocks>>>(q, u);
  else QsaUnionBuildKernel<<<tiles, 256, blocks>>>(q, u);
}

template <bool Gate> void Flash(Kernel kernel, const QsaPrefill& f, QsaUnion u) {
  const dim3 grid(2, (f.input.R + kUnionRows - 1) / kUnionRows);
  switch (kernel) {
    case kTrunk: trunk27e0::QsaUnionHalfQKernel<Gate><<<grid, 256>>>(f, u); break;
    case kStage: QsaUnionStageKernel<Gate, 0><<<grid, 256>>>(f, u); break;
    case kStageM1: QsaUnionStageKernel<Gate, 1><<<grid, 256>>>(f, u); break;
    case kStageM2: QsaUnionStageKernel<Gate, 2><<<grid, 256>>>(f, u); break;
    case kStageM3: QsaUnionStageKernel<Gate, 3><<<grid, 256>>>(f, u); break;
    case kStage0: QsaUnionStageKernel<Gate, 0, 0><<<grid, 256>>>(f, u); break;
    case kStageMem: QsaUnionStageKernel<Gate, 4><<<grid, 256>>>(f, u); break;
  }
}

// Build + flash over parts of `part` rows (the engine's core parts; part == R is one call).
template <bool Gate> void Run(Problem& p, Kernel kernel, int part) {
  for (int o = 0; o < p.R; o += part) {
    const int n = std::min(part, p.R - o);
    const auto f = p.Args(o, n);
    Build(kernel == kTrunk, f.input, p.Union(o));
    Flash<Gate>(kernel, f, p.Union(o));
  }
}

template <class T> bool Same(const std::vector<T>& x, const std::vector<T>& y) {
  return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}
template <class T> size_t Differ(const std::vector<T>& x, const std::vector<T>& y) {
  size_t d = 0;
  for (size_t i = 0; i < x.size(); ++i) d += std::memcmp(&x[i], &y[i], sizeof(T)) != 0;
  return d;
}

int Check() {
  struct Shape { int pos0, R; const char* name; };
  const Shape shapes[] = {
      {0, 2048, "dense part"},        {130, 37, "dense, odd start, partial tile"},
      {2048, 2048, "dense to sparse"}, {6000, 97, "sparse, partial tile"},
      {24576, 2048, "24K part"},       {26624, 2045, "24K part, partial tile"},
      {65538, 64, "64K, start 2 mod 4"}, {122880, 2048, "120K part"}, {129024, 2048, "126K part"}};
  int bad = 0;
  for (const auto& s : shapes) {
    std::vector<int> hsel, hnsel;
    Synthetic(s.pos0, s.R, kDrift, hsel, hnsel);
    Problem p(s.pos0, s.R, hsel, hnsel);
    auto gated = [&](Kernel kernel) { p.xh.Nan(); Run<true>(p, kernel, kPart); CheckLaunch(); return p.xh.Get(); };
    auto context = [&](Kernel kernel) {
      p.context.Nan(); Run<false>(p, kernel, kPart); CheckLaunch(); return p.context.Get();
    };
    const auto xt = gated(kTrunk), xs = gated(kStage), x0 = gated(kStage0);
    const auto ct = context(kTrunk), cs = context(kStage), c0 = context(kStage0);
    // The engine's default kernel and build (kQsaUnionStage off) against the trunk copy.
    p.xh.Nan();
    { const auto f = p.Args(0, p.R); LaunchQsaUnionFlashGate(f, p.Union(0)); CheckLaunch(); }
    const auto xe = p.xh.Get();
    const bool ok = Same(xt, xs) && Same(ct, cs) && Same(xt, xe) && Same(xt, x0) && Same(ct, c0);
    bad += !ok;
    std::printf("check pos0=%d R=%d (%s) gated=%s context=%s map0_gated=%s map0_context=%s engine_default=%s "
        "diff_gated=%zu diff_context=%zu %s\n", s.pos0, s.R, s.name, Same(xt, xs) ? "SAME" : "DIFF",
        Same(ct, cs) ? "SAME" : "DIFF", Same(xt, x0) ? "SAME" : "DIFF", Same(ct, c0) ? "SAME" : "DIFF",
        Same(xt, xe) ? "SAME" : "DIFF", Differ(xt, xs), Differ(ct, cs), ok ? "SAME" : "DIFF");
    std::fflush(stdout);
    if (s.pos0 >= 24576 && s.R == kPart) {
      for (Kernel m : {kStageM1, kStageM2, kStageM3}) {
        const auto xm = gated(m);
        const bool differs = !Same(xt, xm);
        bad += !differs;
        std::printf("mutant %d pos0=%d R=%d %s (%zu bytes differ)\n", (int)m - (int)kStage, s.pos0, s.R,
            differs ? "DIFF" : "SAME", Differ(xt, xm));
      }
    }
  }
  // One call over a whole launch keeps the bits of four part calls (tiles on the same grid).
  for (int start : {0, 24576}) {
    std::vector<int> hsel, hnsel;
    Synthetic(start, kLaunch, kDrift, hsel, hnsel);
    Problem p(start, kLaunch, hsel, hnsel);
    p.xh.Nan(); Run<true>(p, kTrunk, kPart); CheckLaunch();
    const auto parts = p.xh.Get();
    p.xh.Nan(); Run<true>(p, kStage, kLaunch); CheckLaunch();
    const auto one = p.xh.Get();
    const bool ok = Same(parts, one);
    bad += !ok;
    std::printf("check launch start=%d rows=%d one stage call vs four trunk part calls %s\n", start, kLaunch,
        ok ? "SAME" : "DIFF");
  }
  std::printf("bench %s\n", bad ? "FAIL" : "PASS");
  return bad ? 9 : 0;
}

int Time(int pos0, int R, std::vector<int> hsel, std::vector<int> hnsel, int pairs, const std::string& source) {
  Problem p(pos0, R, hsel, hnsel);
  // Union statistics of the part grid.
  Run<true>(p, kTrunk, kPart); CheckLaunch();
  const auto list = p.list.Get(), counts = p.count.Get();
  const Work w = Count(list, counts);
  // Knockout lists: the same steps and masks, every block in [0, kLocal).
  std::vector<int> local = list;
  for (auto& e : local) e = (e & ~0xffffff) | ((e & 0xffffff) % kLocal);
  auto flash_only = [&](Kernel kernel) {
    for (int o = 0; o < p.R; o += kPart) Flash<true>(kernel, p.Args(o, std::min(kPart, p.R - o)), p.Union(o));
  };
  std::map<std::string, std::function<void()>> arms = {
      {"T", [&] { Run<true>(p, kTrunk, kPart); }},
      {"S", [&] { Run<true>(p, kStage, kPart); }},
      {"T1", [&] { Run<true>(p, kTrunk, p.R); }},
      {"S1", [&] { Run<true>(p, kStage, p.R); }},
      {"S0", [&] { Run<true>(p, kStage0, kPart); }},
      {"SM", [&] { flash_only(kStageMem); }},
      {"B", [&] {
         for (int o = 0; o < p.R; o += kPart) Build(false, p.Args(o, std::min(kPart, p.R - o)).input, p.Union(o));
       }},
  };
  const std::vector<std::string> order = {"T", "S", "T1", "S1", "S0", "B", "SM"};
  std::map<std::string, std::vector<double>> us, mhz, mj;
  std::vector<double> all_mhz;
  bench::Sampler sampler;
  auto sample = [&](const std::string& name, const std::function<void()>& f) {
    sampler.Start();
    const double t = prefill_bench::Time(f, 15);
    sampler.Stop();
    // Energy of one call: the mean package power of the sample times the call time.
    const std::string summary = sampler.Summary();
    double watts = 0;
    if (const char* w = std::strstr(summary.c_str(), "pwr=")) std::sscanf(w, "pwr=%lf", &watts);
    us[name].push_back(t); mhz[name].push_back(sampler.MeanMhz()); mj[name].push_back(watts * t / 1000.);
    if (sampler.MeanMhz() > 0) all_mhz.push_back(sampler.MeanMhz());  // a short arm can end before the first sample
    std::printf("time source=%s pos0=%d R=%d arm=%s us=%.1f mJ=%.1f %s\n", source.c_str(), pos0, R, name.c_str(), t,
        watts * t / 1000., summary.c_str());
    std::fflush(stdout);
  };
  for (int round = 0; round < pairs; ++round) {
    // ABBA over the arm list: even rounds forward, odd rounds backward.
    std::vector<std::string> o = order;
    if (round & 1) std::reverse(o.begin(), o.end());
    for (const auto& name : o) sample(name, arms[name]);
  }
  // Knockouts last: they rewrite the lists.
  Run<true>(p, kTrunk, kPart); CheckLaunch();
  p.list.Set(local);
  for (int round = 0; round < pairs; ++round) {
    sample(round & 1 ? "SL" : "TL", [&] { flash_only(round & 1 ? kStage : kTrunk); });
    sample(round & 1 ? "TL" : "SL", [&] { flash_only(round & 1 ? kTrunk : kStage); });
  }
  p.list.Set(list);
  auto med = [&](const std::string& n) { return bench::Median(us[n]); };
  auto mmhz = [&](const std::string& n) { return bench::Median(mhz[n]); };
  auto mmj = [&](const std::string& n) { return bench::Median(mj[n]); };
  const double sclk = bench::Median(all_mhz);
  int faster = 0;
  for (size_t i = 0; i < us["T"].size(); ++i) faster += us["S"][i] < us["T"][i];
  const double kv_gb = w.blocks * 4 * 2 * kHeadDim * 2 * 2 / 1e9;  // K and V, both groups
  std::printf("result source=%s pos0=%d R=%d tiles=%.0f union_blocks_per_tile=%.1f steps_per_tile=%.1f "
      "live_mtiles_per_step=%.2f T_us=%.1f S_us=%.1f S/T=%.3f T1/T=%.3f S1/T=%.3f S0/T=%.3f SM/S=%.3f build_us=%.1f "
      "TL_us=%.1f SL_us=%.1f TL/T=%.3f SL/S=%.3f kv_gb=%.3f T_kv_gbps=%.0f S_kv_gbps=%.0f S_faster=%d/%zu "
      "sclk_T=%.0f sclk_S=%.0f sclk=%.0f sclk_ok=%d T_mJ=%.1f S_mJ=%.1f\n",
      source.c_str(), pos0, R, w.tiles, w.blocks / w.tiles, w.steps / w.tiles, w.live / w.steps, med("T"), med("S"),
      med("S") / med("T"), med("T1") / med("T"), med("S1") / med("T"), med("S0") / med("T"), med("SM") / med("S"),
      med("B"), med("TL"), med("SL"),
      med("TL") / med("T"), med("SL") / med("S"), kv_gb, kv_gb / med("T") * 1e6, kv_gb / med("S") * 1e6, faster,
      us["T"].size(), mmhz("T"), mmhz("S"), sclk, sclk >= 2200, mmj("T"), mmj("S"));
  // Timing runs below a median sclk of 2,200 MHz do not count (brief).
  std::printf("bench %s\n", sclk >= 2200 ? "PASS" : "FAIL sclk");
  return sclk >= 2200 ? 0 : 3;
}
}  // namespace

int main(int argc, char** argv) {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) return 2;
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "check" && argc == 2) return Check();
  if (mode == "time" && (argc == 3 || argc == 4)) {
    const int start = std::stoi(argv[2]), pairs = argc == 4 ? std::stoi(argv[3]) : 5;
    if (start < 0 || start % 4 || start > 262144 - kLaunch || pairs < 1) return 2;
    std::vector<int> hsel, hnsel;
    Synthetic(start, kLaunch, kDrift, hsel, hnsel);
    return Time(start, kLaunch, hsel, hnsel, pairs, "synthetic");
  }
  if (mode == "capture" && (argc == 3 || argc == 4)) {
    const std::string prefix = argv[2];
    const int pairs = argc == 4 ? std::stoi(argv[3]) : 5;
    int pos0 = 0, R = 0;
    std::ifstream meta(prefix + ".meta");
    if (!(meta >> pos0 >> R) || pos0 < 0 || R < 1 || R > kPart || pairs < 1) return 2;
    return Time(pos0, R, Read<int>(prefix + ".sel.i32", (size_t)R * kSelBlocks), Read<int>(prefix + ".nsel.i32", R),
        pairs, prefix);
  }
  std::fprintf(stderr, "usage: qsa_attn check | time START [PAIRS] | capture PREFIX [PAIRS]\n");
  return 2;
}
