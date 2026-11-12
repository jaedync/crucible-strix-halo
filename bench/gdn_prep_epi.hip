// the GDN q/k prep in the epilogue of the GDN input GEMM (GdnPrepEpi)
// against the separate prep (GdnLanesPrepQkKernel). The engine's fused EXL3
// GEMM (LaunchPrefillGemmOut, front tile order) writes q/k/v and z of the
// whole launch; then 2,048-row parts run LaunchGdnLanes (vscan, SGPR scan).
// q, k, the state, the history ring, y and the gated out-projection input must
// match bit for bit. The row counts put tile edges (rows 0, 127, 128, a last
// partial tile) and part edges in the check. Then ABAB event timing.
// Usage: gdn_prep_epi [--k K] [--pairs N] [rows ...]
// A build with -DGDN_PREP_EPI_BENCH_MUTANT skips the halo rows: it must FAIL.
// the "cut" check runs the epilogue path with the engine's lazy-scale anchor (2,048) twice: whole
// parts, and the part that holds a cut p as two pieces (crucible::PrefillPieces, qk_rows, LaunchGdnPrepQkHalo), with
// the separate gate (LaunchGdnLanes) and the fused scan gate (LaunchGdnScanGate). The state with its scale floats,
// the history ring and the gated out-projection input must be equal. A build with -DGDN_CUT_BENCH_MUTANT reads the
// piece's q and k without qk_rows: it must FAIL.
#include "prefill_common.h"
#include "../engine/model/flashnext/prefill_launch.h"
#include "../engine/exl3/decode_split.h"
#include "../engine/layers/prefill_hc.h"
#include "../engine/exl3/prefill_linear.h"
#include "../engine/layers/kernels/prefill_gdn.h"
#include <cstring>
#include <string>
using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;
namespace {
constexpr int kCore = 2048, kRm = 8, kQkvO = kGdnQkvWidth, kZO = kGdnValueWidth;

std::vector<uint32_t> RandomWords(size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<uint32_t> w(n);
  for (auto& v : w) v = rng();
  return w;
}

// The engine's GDN input group: the q/k/v job and the z job, both kEpiStore.
struct Gemm {
  int M, K;
  Buffer<uint32_t> tq, tz;
  Buffer<__half> x, sq, sz;
  Buffer<float> qkv, z;
  Buffer<LinJob> lin;
  Buffer<Exl3GemmJob> job;
  Buffer<Exl3GemmItem> it;
  int n;
  Gemm(int M_, int K_)
      : M(M_), K(K_), tq(RandomWords((size_t)kHidden / 16 * (kQkvO / 16) * 8 * K_, 7)),
        tz(RandomWords((size_t)kHidden / 16 * (kZO / 16) * 8 * K_, 8)), x(Half(Random((size_t)M_ * kHidden, .1f, 11))),
        sq(Half(Random(kQkvO, .05f, 13))), sz(Half(Random(kZO, .05f, 14))), qkv((size_t)M_ * kQkvO), z((size_t)M_ * kZO),
        lin(std::vector<LinJob>{Lin(tq.p, sq.p, qkv.p, kQkvO), Lin(tz.p, sz.p, z.p, kZO)}),
        job(std::vector<Exl3GemmJob>{Job(tq.p, kQkvO), Job(tz.p, kZO)}), it(Items()), n((int)it.n) {}
  LinJob Lin(const uint32_t* t, const __half* s, float* d, int O) const {
    LinJob j{};
    j.tiles = t; j.svh = s; j.dst = d; j.I = kHidden; j.O = O; j.K = K; j.S = 1; j.R = M; j.ldd = O; j.epi = kEpiStore;
    return j;
  }
  Exl3GemmJob Job(const uint32_t* t, int O) const {
    return {t, x.p, nullptr, nullptr, kHidden, O, M, 1, K, M, SplitFor(kHidden / 16, DecodeTarget({{kHidden, O}}))};
  }
  std::vector<Exl3GemmItem> Items() const {
    std::vector<Exl3GemmItem> v;
    for (int j = 0; j < 2; ++j)
      for (int r = 0; r < M; r += 16 * kRm)
        for (int b = 0; b < (j ? kZO : kQkvO) / 16; b += kFusedWaves) v.push_back({j, 0, r, b});
    return v;
  }
  void Run() const { LaunchPrefillGemmOut(kRm, lin.p, job.p, it.p, n, K == 6, true); }
  void Run(const GdnPrepEpi& e) const { LaunchPrefillGemmOut(kRm, lin.p, job.p, it.p, n, K == 6, true, e); }
};

struct Gdn {
  int R;
  Buffer<float> ab, conv, A_log, dt_bias, gamma, hist, hist0, state, state0, carry;
  Buffer<unsigned long long> prog;
  unsigned gen = 0;
  Buffer<__half> suh, xh;
  Buffer<float> q, k, qf, kf, v, y, g, beta;
  explicit Gdn(int rows)
      : R(rows), ab(Random((size_t)rows * 96, 1.f, 3)), conv(Random(kQkvO * 4, .5f, 4)), A_log(Random(48, .5f, 5)),
        dt_bias(Random(48, .5f, 6)), gamma(Random(128, .1f, 7)), hist(16 * kQkvO), hist0(Random(16 * kQkvO, 1.f, 8)),
        state(kGdnStateFloats + kGdnScaleFloats), state0(Init()), carry(48), prog(48 * 4), suh(Half(Random(kZO, 1.f, 10))),
        xh((size_t)rows * kZO), q(48 * (size_t)kCore * 128), k(48 * (size_t)kCore * 128),
        qf(Parts() * kGdnPrepPartFloats), kf(Parts() * kGdnPrepPartFloats), v(48 * (size_t)kCore * 128),
        y(48 * (size_t)kCore * 128), g(48 * kCore), beta(48 * kCore) {}
  size_t Parts() const { return (size_t)(R + kCore - 1) / kCore; }
  static std::vector<float> Init() {
    auto x = Random(kGdnStateFloats + kGdnScaleFloats, .01f, 9);
    std::fill(x.begin() + kGdnStateFloats, x.end(), 0.f);  // a zeroed scale means 1
    return x;
  }
  void Reset() {
    HIP_OK(hipMemcpy(state.p, state0.p, state.n * 4, hipMemcpyDeviceToDevice));
    HIP_OK(hipMemcpy(hist.p, hist0.p, hist.n * 4, hipMemcpyDeviceToDevice));
    HIP_OK(hipMemset(xh.p, 0xff, xh.n * 2));
    HIP_OK(hipMemset(y.p, 0, y.n * 4));
    qf.Zero(); kf.Zero();
  }
  GdnPrefill Full(const Gemm& m, bool packed, int pos0) {
    GdnPrefill a{};
    a.input = {m.qkv.p, m.z.p, nullptr, nullptr, conv.p, A_log.p, dt_bias.p, gamma.p, hist.p, state.p, suh.p,
               xh.p, nullptr, pos0, R, 0, R};
    if (packed) a.input.pack_rows = R;
    a.ab_logits = ab.p;
    a.q = q.p; a.k = k.p; a.v = v.p; a.y = y.p; a.g = g.p; a.beta = beta.p;
    return a;
  }
};

// engine/runtime/prefill_engine.inc: 2,048-row parts on one stream. halo > 0: the epilogue path.
// qk (optional): the reference q and k of every part, [part][16][rows of the part][128].
void Parts(GdnPrefill full, Gdn& d, int halo, std::vector<float>* qk = nullptr) {
  for (int o = 0; o < full.input.R; o += kCore) {
    const int n = std::min(kCore, full.input.R - o);
    GdnPrefill a = full;
    a.input.qkv += (size_t)o * kQkvO; a.input.z += (size_t)o * kZO;
    if (a.input.pack_rows) a.input.pack_off = o; else a.input.xh_out += (size_t)o * kZO;
    a.input.pos0 += o; a.input.R = a.input.commit = n;
    a.ab_logits += (size_t)o * 96; a.chunks = (n + 63) / 64;
    if (halo) { a.q = d.qf.p + (o / kCore) * kGdnPrepPartFloats; a.k = d.kf.p + (o / kCore) * kGdnPrepPartFloats; }
    LaunchGdnLanes(a, nullptr, true, halo);
    if (qk) {
      const size_t f = (size_t)16 * a.chunks * 64 * 128, at = qk->size();
      qk->resize(at + 2 * f);
      HIP_OK(hipMemcpy(qk->data() + at, a.q, f * 4, hipMemcpyDeviceToHost));
      HIP_OK(hipMemcpy(qk->data() + at + f, a.k, f * 4, hipMemcpyDeviceToHost));
    }
  }
}

// b-cut2: the epilogue path as prefill_engine.inc runs it with a cut (0: none) and the lazy-scale anchor.
void PartsCut(GdnPrefill full, Gdn& d, int halo, int cut, bool fused) {
  full.anchor = kCore; full.carry = d.carry.p;
  const auto args = [&](int o, int n) {
    GdnPrefill a = full;
    a.input.qkv += (size_t)o * kQkvO; a.input.z += (size_t)o * kZO;
    if (a.input.pack_rows) a.input.pack_off = o; else a.input.xh_out += (size_t)o * kZO;
    a.input.pos0 += o; a.input.R = a.input.commit = n;
    a.ab_logits += (size_t)o * 96; a.chunks = (n + 63) / 64;
    return a;
  };
  for (const auto& piece : crucible::PrefillPieces(full.input.R, cut)) {
    const int o = piece.offset, n = piece.rows;
    GdnPrefill a = args(o, n);
    const size_t part = (size_t)(piece.part / kCore) * kGdnPrepPartFloats;
    a.q = d.qf.p + part + (size_t)(o - piece.part) * 128; a.k = d.kf.p + part + (size_t)(o - piece.part) * 128;
    if (n != piece.part_rows) {
#ifndef GDN_CUT_BENCH_MUTANT
      a.qk_rows = (piece.part_rows + 63) & ~63;
#else
      a.qk_rows = a.chunks * 64;
#endif
      if (o == piece.part) {
        GdnPrefill w = args(o, piece.part_rows);
        w.q = d.qf.p + part; w.k = d.kf.p + part;
        LaunchGdnPrepQkHalo(w, nullptr, halo);
      }
    }
    if (fused) LaunchGdnScanGate(a, d.prog.p, ++d.gen ? d.gen : ++d.gen, nullptr, halo);
    else LaunchGdnLanes(a, nullptr, true, halo);
  }
}

struct Result { std::vector<float> qk, state, hist, y; std::vector<__half> xh; };
Result RunCut(const Gemm& m, Gdn& d, bool packed, int pos0, int cut, bool fused) {
  d.Reset();
  m.Run(GdnPrepEpi{m.qkv.p, d.qf.p, d.kf.p, d.conv.p, d.R});
  PartsCut(d.Full(m, packed, pos0), d, 16 * kRm, cut, fused);
  CheckLaunch();
  Result out;
  out.state = d.state.Get(); out.hist = d.hist.Get(); out.xh = d.xh.Get();
  return out;
}
Result Run(const Gemm& m, Gdn& d, bool packed, int pos0, bool epi) {
  d.Reset();
  GdnPrefill full = d.Full(m, packed, pos0);
  Result out;
  if (epi) {
    m.Run(GdnPrepEpi{m.qkv.p, d.qf.p, d.kf.p, d.conv.p, d.R});
    Parts(full, d, 16 * kRm);
    const std::vector<float> qf = d.qf.Get(), kf = d.kf.Get();
    for (int o = 0; o < d.R; o += kCore) {  // the same order as Parts(qk)
      const size_t f = (size_t)16 * ((std::min(kCore, d.R - o) + 63) / 64 * 64) * 128, at = (o / kCore) * kGdnPrepPartFloats;
      out.qk.insert(out.qk.end(), qf.begin() + at, qf.begin() + at + f);
      out.qk.insert(out.qk.end(), kf.begin() + at, kf.begin() + at + f);
    }
  } else {
    m.Run();
    Parts(full, d, 0, &out.qk);
  }
  CheckLaunch();
  out.state = d.state.Get(); out.hist = d.hist.Get(); out.y = d.y.Get(); out.xh = d.xh.Get();
  return out;
}
template <class T> bool Same(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
}  // namespace

int main(int argc, char** argv) {
  int pairs = 7;
  std::vector<int> ks, sizes;
  bool cut_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--cut") cut_only = true;
    else if (s == "--pairs" && i + 1 < argc) pairs = std::stoi(argv[++i]);
    else if (s == "--k" && i + 1 < argc) ks.push_back(std::stoi(argv[++i]));
    else sizes.push_back(std::stoi(s));
  }
  if (ks.empty()) ks = {4, 6};
  if (sizes.empty()) sizes = {1, 3, 4, 5, 127, 128, 129, 200, 2047, 2048, 2049, 4100, 8192};
  bool ok = true;
  for (int K : ks) {
    if (K < 3 || K > 6) return 2;
    for (int R : sizes) {
      if (R < 1 || R > 16384) return 2;
      Gemm m(R, K);
      Gdn d(R);
      // b-cut2: a cut inside a part against whole parts, both scan paths, on the 4-row grid from a launch start
      // on that grid (PrefillPreCutValid); pos0 2,000 puts an anchor inside the first part.
      bool cut_same = true;
      int cuts = 0;
      for (int cut : {4, 60, 1000, 2044, 2052, 4096 + 1000})
        for (int pos0 : {0, 100, 2000})
          for (int fused = 0; fused < 2; ++fused) {
            if (cut >= R || (fused && !GdnScanGateFits())) continue;
            const int packed = (cut / 4 + pos0) % 2;
            const Result a = RunCut(m, d, packed, pos0, 0, fused), b = RunCut(m, d, packed, pos0, cut, fused);
            const bool eq = Same(a.state, b.state) && Same(a.hist, b.hist) && Same(a.xh, b.xh);
            if (!eq) std::printf("K=%d R=%d cut=%d pos0=%d fused=%d DIFFER\n", K, R, cut, pos0, fused);
            cut_same = cut_same && eq;
            ++cuts;
          }
      ok = ok && cut_same;
      std::printf("K=%d R=%d cut=%s (%d cases)", K, R, cut_same ? "EQUAL" : "DIFFER", cuts);
      if (cut_only) { std::printf("\n"); std::fflush(stdout); continue; }
      std::printf("  ");
      bool same = true;
      for (int packed = 0; packed < 2; ++packed)
        for (int pos0 : {0, 2, 5, 100}) {
          const Result a = Run(m, d, packed, pos0, false), b = Run(m, d, packed, pos0, true);
          same = same && Same(a.qk, b.qk) && Same(a.state, b.state) && Same(a.hist, b.hist) && Same(a.y, b.y) &&
                 Same(a.xh, b.xh);
        }
      ok = ok && same;
      std::printf("K=%d R=%d bitwise=%s", K, R, same ? "EQUAL" : "DIFFER");
      if (R >= 4096 && pairs > 0) {  // ABAB; the state keeps evolving, which does not change the work
        const GdnPrefill full = d.Full(m, true, 0);
        const GdnPrepEpi e{m.qkv.p, d.qf.p, d.kf.p, d.conv.p, R};
        std::vector<double> t0, t1, g0, g1, p0, p1;
        for (int rep = 0; rep < pairs; ++rep) {
          t0.push_back(Time([&] { m.Run(); Parts(full, d, 0); }, 3));
          t1.push_back(Time([&] { m.Run(e); Parts(full, d, 16 * kRm); }, 3));
          g0.push_back(Time([&] { m.Run(); }, 3));
          g1.push_back(Time([&] { m.Run(e); }, 3));
          p0.push_back(Time([&] { Parts(full, d, 0); }, 3));
          p1.push_back(Time([&] { Parts(full, d, 16 * kRm); }, 3));
        }
        const double a = bench::Median(t0), b = bench::Median(t1);
        std::printf(" | gemm+gdn %.1f -> %.1f us (%.3f, ms_per4096_36L=%+.1f) gemm %.1f -> %.1f us gdn %.1f -> %.1f us",
                    a, b, b / a, (b - a) * 36 / 1000. * 4096. / R, bench::Median(g0), bench::Median(g1),
                    bench::Median(p0), bench::Median(p1));
      }
      std::printf("\n");
      std::fflush(stdout);
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
