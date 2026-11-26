// GPU checks of CRUCIBLE_HC_MIX_HI (engine/layers/hc_mix_hi.h). Exit 0 and a last line
// "hc_mix_hi_gpu_test: PASS" when every check holds.
//   hc_mix_hi_gpu_test          the bit checks (about 10 s)
//   hc_mix_hi_gpu_test time     the bit checks, then kernel times (median us, interleaved)
//
// Bulk prefill (R > 8):
//   1. PrefillHcMixHiKernel (FB 0 and 1, RT 1 and 2) and LaunchPrefillHcMix(hi) equal PrefillHcMixWideKernel on a
//      zero tlo, bitwise (mixed and xh). LaunchPrefillHcMix(off) equals PrefillHcMixWideKernel on the real tlo.
//   2. Rows do not depend on R or on the row offset of the launch (a 97-row launch at rows 0 and 4000).
//   3. Control: the high-half mix differs from the exact mix (and the size of the change is printed).
//   4. LaunchPrefillHcMidPack(hi) writes the trunk thi and post and does not write tlo; (off) equals the trunk mid.
// WMMA read (decode, verify, chunked prefill; fp16 row-major, fp16 fragment order, int8):
//   5. With hi_only, every row of an R-row call (R = 1..16) equals the 1-row call of that row, bitwise (x, post).
//   6. HcW2Launch with hi_only equals HcWLaunch with hi_only (both mids), and post does not change with hi_only.
//   7. Control: hi_only changes x.
#include "../bench/prefill_common.h"
#include "../engine/layers/hc_int8.h"
#include "../engine/layers/hc_wmma2.h"
#include "../engine/layers/prefill_hc_hi.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

using namespace crucible;
using namespace prefill_bench;

namespace {

constexpr int kM = kRank + kStreams, kLdh = kWide + 64;
int g_fails = 0;

template <class T> bool Same(const std::vector<T>& a, const std::vector<T>& b, size_t off = 0, size_t n = 0) {
  if (!n) n = b.size();
  return std::memcmp(a.data() + off, b.data() + off, n * sizeof(T)) == 0;
}
void Expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_fails;
}

struct Bulk {
  int R;
  Buffer<float> H, w1, dots, ss, post, mixed;
  Buffer<__half> fn, fnf, up, upf, thi, tlo, zero, xh;
  explicit Bulk(int rows)
      : R(rows), H((size_t)rows * kLdh), w1(Random(kWide, .05f, 319)), dots((size_t)rows * kM * 4), ss(rows * 4),
        post(rows * 4), mixed((size_t)rows * kHidden), fn(Half(Random((size_t)kM * kWide, .02f, 313))),
        fnf((size_t)(kM + 15) / 16 * 16 * kWide), up(Half(Random((size_t)kWide * kRank, .03f, 317))),
        upf((size_t)kWide * kRank), thi((size_t)rows * kRank), tlo(thi.n), zero(thi.n), xh((size_t)rows * kHidden) {
    auto h = Random((size_t)rows * kLdh, .5f, 311);
    for (size_t i = 0; i < h.size(); i += 97) h[i] *= 40.f;  // a spread of magnitudes, like the trunk streams
    H.Set(h);
    auto w = w1.Get();
    for (auto& v : w) v += 1.f;
    w1.Set(w);
    HcFragFnKernel<<<((kM + 15) / 16 * 16 * kWide + 255) / 256, 256>>>(fn.p, kM, fnf.p);
    HcFragUpKernel<<<(kWide * kRank + 255) / 256, 256>>>(up.p, upf.p);
    // The engine's bulk dots and mid (engine.hip HcRead, default numerics), so t has the engine's values.
    if (R >= 512)
      PrefillHcDotsWsKernel<64, 0, true, 2, 2, false, 4, true><<<kStreams * ((R + 63) / 64), (kHcWsCompute + 2) * 32>>>(
          fnf.p, kM, H.p, dots.p, ss.p, R, kLdh);
    else
      PrefillHcDotsWsKernel<64, 0, true, 2, 4, false><<<kStreams * ((R + kHcRows - 1) / kHcRows), (kHcWsCompute + 2) * 32>>>(
          fn.p, kM, H.p, dots.p, ss.p, R, kLdh);
    PrefillHcMidPackKernel<<<R, 256>>>(dots.p, kM, ss.p, thi.p, tlo.p, post.p, R);
    CheckLaunch();
  }
  struct Out { std::vector<float> mixed; std::vector<__half> xh; };
  template <class F> Out Run(F launch) {
    mixed.Nan();
    xh.Nan();
    launch();
    CheckLaunch();
    return {mixed.Get(), xh.Get()};
  }
};

template <bool FB, int RT> void HiKernel(Bulk& b, int r0, int R) {
  PrefillHcMixHiKernel<FB, RT><<<dim3(kHidden / 16, (R + 128 * RT - 1) / (128 * RT)), 256>>>(
      FB ? b.upf.p : b.up.p, b.w1.p, b.thi.p + (size_t)r0 * kRank, b.ss.p + r0 * 4, b.H.p + (size_t)r0 * kLdh,
      b.mixed.p + (size_t)r0 * kHidden, R, b.xh.p + (size_t)r0 * kHidden, kLdh);
}
void WideKernel(Bulk& b, const __half* tlo, bool fb) {
  if (fb)
    PrefillHcMixWideKernel<true, true><<<dim3(kHidden / 16, (b.R + 127) / 128), 256>>>(b.upf.p, b.w1.p, b.thi.p, tlo,
        b.ss.p, b.H.p, b.mixed.p, b.R, b.xh.p, kLdh);
  else
    PrefillHcMixWideKernel<true><<<dim3(kHidden / 16, (b.R + 127) / 128), 256>>>(b.up.p, b.w1.p, b.thi.p, tlo,
        b.ss.p, b.H.p, b.mixed.p, b.R, b.xh.p, kLdh);
}

// The mid pack launcher against the trunk mid of the Bulk constructor (thi, tlo, post).
void CheckMidPack(Bulk& b) {
  const std::string at = " R=" + std::to_string(b.R);
  const auto thi = b.thi.Get(), tlo = b.tlo.Get();
  const auto post = b.post.Get();
  Buffer<__half> h2(b.thi.n), l2(b.tlo.n);
  Buffer<float> p2(b.post.n);
  for (bool hi : {false, true}) {
    h2.Nan();
    l2.Nan();
    p2.Nan();
    LaunchPrefillHcMidPack(hi, b.dots.p, kM, b.ss.p, h2.p, l2.p, p2.p, b.R);
    CheckLaunch();
    const auto lo = l2.Get();
    bool untouched = true;  // Nan() sets every byte to 0xFF
    for (const auto& v : lo) {
      unsigned char c[2];
      std::memcpy(c, &v, 2);
      untouched &= c[0] == 0xFF && c[1] == 0xFF;
    }
    const bool lo_ok = hi ? untouched : Same(lo, tlo);
    Expect(Same(h2.Get(), thi) && Same(p2.Get(), post) && lo_ok,
           std::string("mid launcher ") + (hi ? "hi: trunk thi and post, tlo not written" : "off = trunk mid") + at);
  }
}

void CheckBulk(int R) {
  Bulk b(R);
  const std::string at = " R=" + std::to_string(R);
  const auto exact = b.Run([&] { WideKernel(b, b.tlo.p, true); });
  const auto zero = b.Run([&] { WideKernel(b, b.zero.p, true); });
  auto same = [&](const Bulk::Out& o, const Bulk::Out& ref) { return Same(o.mixed, ref.mixed) && Same(o.xh, ref.xh); };
  Expect(same(b.Run([&] { WideKernel(b, b.zero.p, false); }), zero), "wide row-major up = fragment up, zero tlo" + at);
  Expect(same(b.Run([&] { HiKernel<true, 2>(b, 0, R); }), zero), "hi FB1 RT2 = wide on zero tlo" + at);
  Expect(same(b.Run([&] { HiKernel<true, 1>(b, 0, R); }), zero), "hi FB1 RT1 = wide on zero tlo" + at);
  Expect(same(b.Run([&] { HiKernel<false, 2>(b, 0, R); }), zero), "hi FB0 RT2 = wide on zero tlo" + at);
  auto launch = [&](bool hi, bool fb) {
    return b.Run([&] { LaunchPrefillHcMix(hi, b.up.p, fb ? b.upf.p : nullptr, b.w1.p, b.thi.p, b.tlo.p, b.ss.p, b.H.p,
                                          b.mixed.p, R, b.xh.p, kLdh); });
  };
  Expect(same(launch(true, true), zero) && same(launch(true, false), zero), "launcher hi = wide on zero tlo" + at);
  Expect(same(launch(false, true), exact) && same(launch(false, false), exact), "launcher off = wide (trunk)" + at);
  // xh only (the engine's half_only call): the same bits.
  {
    b.xh.Nan();
    LaunchPrefillHcMix(true, b.up.p, b.upf.p, b.w1.p, b.thi.p, b.tlo.p, b.ss.p, b.H.p, nullptr, R, b.xh.p, kLdh);
    CheckLaunch();
    Expect(Same(b.xh.Get(), zero.xh), "launcher hi, xh only" + at);
  }
  // Row independence: shorter launches at row 0 and at an offset give the rows of the full launch.
  for (int r0 : {0, 4000}) {
    const int n = 97;
    if (r0 + n > R) continue;
    b.mixed.Nan();
    b.xh.Nan();
    HiKernel<true, 2>(b, r0, n);
    CheckLaunch();
    const auto m = b.mixed.Get();
    const auto x = b.xh.Get();
    Expect(Same(m, zero.mixed, (size_t)r0 * kHidden, (size_t)n * kHidden) &&
               Same(x, zero.xh, (size_t)r0 * kHidden, (size_t)n * kHidden),
           "hi rows " + std::to_string(r0) + ".." + std::to_string(r0 + n - 1) + " alone = in the full launch" + at);
  }
  // Control and size of the change: the high-half mix against the exact mix on the fp16 output.
  size_t flips = 0;
  for (size_t i = 0; i < zero.xh.size(); ++i) flips += std::memcmp(&zero.xh[i], &exact.xh[i], 2) != 0;
  const auto e = Compare(zero.xh, exact.xh);
  std::printf("size R=%d xh rel_l2=%.3e changed=%.2f%% max_abs=%.3e\n", R, e.rel_l2, 100. * flips / zero.xh.size(),
              e.max_abs);
  Expect(flips > 0, "control: hi differs from the exact mix" + at);
  CheckMidPack(b);
}

// The WMMA read on one site (the inject site, M = kRank + 4).
struct Site {
  Buffer<__half> fn, up, fnf, upf;
  Buffer<float> w1, H, x, post;
  Buffer<uint8_t> fq, uq;
  HcWScratch sc2 = HcW2Alloc(), sc1 = HcWAlloc();
  unsigned epoch = 0;
  Site() : fn((size_t)kM * kWide), up((size_t)kWide * kRank), fnf((size_t)kHcWPad * kWide), upf((size_t)kWide * kRank),
           w1(Random(kWide, .4f, 7)), H(Random((size_t)kHcWRows * kWide, .5f, 11)), x((size_t)kHcWRows * kHidden),
           post(kHcWRows * kStreams), fq(Hc8Bytes(kM, kWide, kHc8FnGroup)), uq(Hc8Bytes(kWide, kRank, kHc8UpGroup)) {
    // W' from the int8 blocks, so the fp16 and int8 reads see the same weights (hc_int8.h).
    auto f0 = Random((size_t)kM * kWide, .02f, 3), u0 = Random((size_t)kWide * kRank, .03f, 5);
    std::vector<_Float16> fh(f0.begin(), f0.end()), uh(u0.begin(), u0.end()), fw(fh.size()), uw(uh.size());
    std::vector<uint8_t> blk(fq.n, 0);
    Hc8Quantize(fh.data(), kM, kWide, kHc8FnGroup, blk.data(), fw.data());
    fq.Set(blk);
    blk.assign(uq.n, 0);
    Hc8Quantize(uh.data(), kWide, kRank, kHc8UpGroup, blk.data(), uw.data());
    uq.Set(blk);
    HIP_OK(hipMemcpy(fn.p, fw.data(), fw.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(up.p, uw.data(), uw.size() * 2, hipMemcpyHostToDevice));
    auto w = w1.Get();
    for (auto& v : w) v += 1.f;
    w1.Set(w);
    HcFragFnKernel<<<(kHcWPad * kWide + 255) / 256, 256>>>(fn.p, kM, fnf.p);
    HcFragUpKernel<<<(kWide * kRank + 255) / 256, 256>>>(up.p, upf.p);
    CheckLaunch();
  }
  // mode 0: fp16 row-major, 1: fp16 fragment order, 2: int8. v1: HcWLaunch (hc_wmma.h).
  std::pair<std::vector<float>, std::vector<float>> Run(int mode, bool hi, int row0, int R, bool v1 = false) {
    x.Nan();
    post.Nan();
    const float* h = H.p + (size_t)row0 * kWide;
    const HcW8 q8 = mode == 2 ? HcW8{fq.p, uq.p} : HcW8{};
    if (v1) {
      sc1.hi_only = hi;
      HcWLaunch(fn.p, kM, up.p, w1.p, h, R, x.p, post.p, sc1, ++epoch, q8);
    } else {
      sc2.hi_only = hi;
      HcW2Launch(fn.p, mode == 1 ? fnf.p : nullptr, kM, up.p, mode == 1 ? upf.p : nullptr, w1.p, h, R, x.p, post.p,
                 sc2, ++epoch, q8);
    }
    CheckLaunch();
    auto xs = x.Get(), ps = post.Get();
    xs.resize((size_t)R * kHidden);
    ps.resize((size_t)R * kStreams);
    return {xs, ps};
  }
};

void CheckWmmaRead() {
  Site s;
  for (int mode = 0; mode < 3; ++mode) {
    const std::string at = std::string(" mode=") + (mode == 0 ? "fp16" : mode == 1 ? "fp16-frag" : "int8");
    std::vector<std::vector<float>> x1(kHcWRows), p1(kHcWRows);
    for (int r = 0; r < kHcWRows; ++r) std::tie(x1[r], p1[r]) = s.Run(mode, true, r, 1);
    bool rows_ok = true, v1_ok = true, post_ok = true, differs = false;
    for (int R = 1; R <= kHcWRows; ++R) {
      const auto [x, p] = s.Run(mode, true, 0, R);
      for (int r = 0; r < R; ++r)
        rows_ok &= std::memcmp(x.data() + (size_t)r * kHidden, x1[r].data(), kHidden * 4) == 0 &&
                   std::memcmp(p.data() + r * kStreams, p1[r].data(), kStreams * 4) == 0;
      const auto [xv, pv] = s.Run(mode, true, 0, R, true);
      v1_ok &= Same(xv, x) && Same(pv, p);
      const auto [xe, pe] = s.Run(mode, false, 0, R);
      post_ok &= Same(pe, p);
      differs |= !Same(xe, x);
    }
    Expect(rows_ok, "wmma read hi: every row of R = 1..16 = its 1-row call" + at);
    Expect(v1_ok, "wmma read hi: HcW2Launch = HcWLaunch" + at);
    Expect(post_ok, "wmma read: post does not change with hi_only" + at);
    Expect(differs, "control: hi_only changes x" + at);
  }
}

// Median of interleaved runs (us). Each call runs once per round, in a fixed order.
void Time() {
  for (int R : {4096, 16512}) {
    Bulk b(R);
    struct Case { const char* name; std::function<void()> f; std::vector<float> us; };
    std::vector<Case> cases = {
        {"wide_exact_fb", [&] { WideKernel(b, b.tlo.p, true); }, {}},
        {"hi_fb_rt2", [&] { HiKernel<true, 2>(b, 0, R); }, {}},
        {"hi_fb_rt1", [&] { HiKernel<true, 1>(b, 0, R); }, {}},
        {"mid_off", [&] { LaunchPrefillHcMidPack(false, b.dots.p, kM, b.ss.p, b.thi.p, b.tlo.p, b.post.p, R); }, {}},
        {"mid_hi", [&] { LaunchPrefillHcMidPack(true, b.dots.p, kM, b.ss.p, b.thi.p, b.tlo.p, b.post.p, R); }, {}},
    };
    bench::EventTimer t;
    for (int round = 0; round < 21; ++round)
      for (auto& c : cases) {
        t.Begin();
        c.f();
        const float ms = t.EndMs();
        if (round) c.us.push_back(ms * 1000.f);  // round 0 warms up
      }
    for (auto& c : cases) {
      std::sort(c.us.begin(), c.us.end());
      std::printf("time bulk R=%d %s median_us=%.1f\n", R, c.name, c.us[c.us.size() / 2]);
    }
  }
  Site s;
  for (int mode : {1, 2})
    for (int R : {1, 8}) {
      std::vector<float> on, off;
      bench::EventTimer t;
      for (int round = 0; round < 41; ++round)
        for (bool hi : {false, true}) {
          s.sc2.hi_only = hi;
          const HcW8 q8 = mode == 2 ? HcW8{s.fq.p, s.uq.p} : HcW8{};
          t.Begin();
          HcW2Launch(s.fn.p, mode == 1 ? s.fnf.p : nullptr, kM, s.up.p, mode == 1 ? s.upf.p : nullptr, s.w1.p, s.H.p, R,
                     s.x.p, s.post.p, s.sc2, ++s.epoch, q8);
          const float ms = t.EndMs();
          if (round) (hi ? on : off).push_back(ms * 1000.f);
        }
      std::sort(on.begin(), on.end());
      std::sort(off.begin(), off.end());
      std::printf("time wmma read mode=%s R=%d off_median_us=%.1f hi_median_us=%.1f\n", mode == 1 ? "fp16-frag" : "int8", R,
                  off[off.size() / 2], on[on.size() / 2]);
    }
}

}  // namespace

int main(int argc, char** argv) {
  for (int R : {9, 97, 4096, 16512}) CheckBulk(R);
  CheckWmmaRead();
  if (argc > 1 && std::string(argv[1]) == "time") Time();
  std::printf("hc_mix_hi_gpu_test: %s (%d failed)\n", g_fails ? "FAIL" : "PASS", g_fails);
  return g_fails ? 1 : 0;
}
