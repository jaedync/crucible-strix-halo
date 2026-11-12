// GdnKernel (A, the present path) against GdnVerifyKernel (B, CRUCIBLE_GDN_VERIFY) for decode, verify and
// packed passes.
//
// gdn_verify check            byte check: state, qkv ring, pending rows and out_proj input bitwise equal for
//                             state cache 0, 4 and 6, total rows 1..16 (1 span of 1..8 rows, 2 spans of 8 + 1..8,
//                             and 3 or 4 spans), several replay and commit counts, ring start and 32K, two passes
//                             (the second replays the pending rows of the first). Then each kernel mutant
//                             (MUT 1..8) must differ. Prints "SUMMARY PASS" or "SUMMARY FAIL".
// gdn_verify time [pairs] [passes] [la] [spans]
//                             timing: whole passes (36 GDN layers, own state per layer, so the state never stays
//                             in the MALL), R = 1..8 at replay R / 2, ABBA-rotated pairs of A and B (rows per
//                             input batch 2), and B4/B8 (rows per batch 4 and 8) at R = 2, 4, 8. sclk per run.
//                             la = 1: a lookahead kernel (4 blocks, 24 MiB window, as the engine row policy) runs
//                             beside the pass and its fetched bytes per pass are printed (the prefetch that a
//                             shorter GDN takes from the next kernels).
//                             spans > 1: packed passes of that many spans of R rows each (the batched verify of
//                             several streams, grid 48 x spans).
#include "common.h"
#include "../engine/layers/gdn_verify.h"
#include <cstring>
#include <functional>
#include <random>
#include <string>

using namespace crucible;

namespace {

constexpr int kGdnLayers = kLayers - kLayers / kQsaPeriod;  // 36
constexpr size_t kState = (size_t)kGdnHeads * kGdnDim * kGdnDim;
constexpr size_t kHist = (size_t)16 * kGdnQkvWidth;
constexpr size_t kPendN = (size_t)kGdnHeads * kMaxRows * kPend;

template <class T> T* Dev(size_t n) {
  T* p = nullptr;
  HIP_CHECK(hipMalloc(&p, n * sizeof(T)));
  return p;
}
template <class T> void Up(T* d, const std::vector<T>& h) {
  HIP_CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
}
template <class T> std::vector<T> Down(const T* d, size_t n) {
  std::vector<T> h(n);
  HIP_CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}
std::vector<float> Normal(size_t n, float sd, uint32_t seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.f, sd);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}
std::vector<__half> ToHalf(const std::vector<float>& v) {
  std::vector<__half> h(v.size());
  for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
  return h;
}
std::vector<float> PendInit(size_t units, uint32_t seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> n(0.f, 1.f);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  std::vector<float> p(units * kPendN);
  for (size_t b = 0; b < p.size(); b += kPend) {
    double s = 0;
    for (int i = 0; i < 128; ++i) s += (double)(p[b + i] = n(g)) * p[b + i];
    for (int i = 0; i < 128; ++i) p[b + i] = (float)(p[b + i] / std::sqrt(s));
    for (int i = 128; i < 256; ++i) p[b + i] = 0.3f * n(g);
    p[b + 256] = 0.5f + 0.5f * u(g);
    p[b + 257] = u(g);
  }
  return p;
}

}  // namespace

// One span of a pass: rows, replay and commit (commit 1 is the engine's verify; others are the decode/prefill uses).
struct SpanShape {
  int R, replay, commit;
};
struct Pass {
  std::vector<SpanShape> spans;
  int pos0;
};

struct Fixture {
  int layers, spans;
  size_t units;
  __half *ab, *suh, *xh;
  float *conv, *alog, *dt, *gamma, *qkv, *z, *x, *hist, *state, *pend;
  std::vector<float> state0, hist0, pend0;
  std::vector<__half> xh0;

  // ab_pitch: bytes between the a/b weights of two layers (the lookahead arena in timing; 0 = packed).
  Fixture(int L_, int S_, uint8_t* arena = nullptr, size_t ab_pitch = 0) : layers(L_), spans(S_) {
    const size_t L = layers, M = kMaxRows;
    units = L * spans;
    const size_t U = units;
    const size_t abn = (size_t)96 * kHidden;
    if (arena) {
      ab = reinterpret_cast<__half*>(arena);
      for (size_t l = 0; l < L; ++l)
        Up(reinterpret_cast<__half*>(arena + l * ab_pitch), ToHalf(Normal(abn, .05f, 100 + l)));
    } else {
      ab = Dev<__half>(L * abn);
      Up(ab, ToHalf(Normal(L * abn, .05f, 1)));
    }
    pitch = arena ? ab_pitch / sizeof(__half) : abn;
    suh = Dev<__half>(L * kGdnValueWidth);
    conv = Dev<float>(L * kGdnQkvWidth * 4);
    alog = Dev<float>(L * kGdnHeads);
    dt = Dev<float>(L * kGdnHeads);
    gamma = Dev<float>(L * kGdnDim);
    qkv = Dev<float>(U * M * kGdnQkvWidth);
    z = Dev<float>(U * M * kGdnValueWidth);
    x = Dev<float>(U * M * kHidden);
    xh = Dev<__half>(U * M * kGdnValueWidth);
    hist = Dev<float>(U * kHist);
    state = Dev<float>(U * kState);
    pend = Dev<float>(U * kPendN);
    std::vector<float> s(L * kGdnValueWidth);
    for (size_t i = 0; i < s.size(); ++i) s[i] = (i * 2654435761u >> 7) & 1 ? 1.f : -1.f;
    Up(suh, ToHalf(s));
    Up(conv, Normal(L * kGdnQkvWidth * 4, .25f, 2));
    // A_log from -5 to 1.5 as the model (slow and fast heads), dt_bias near 0.
    auto al = Normal(L * kGdnHeads, 1.5f, 3);
    for (auto& v : al) v = std::fmin(1.5f, std::fmax(-5.f, v - 1.f));
    Up(alog, al);
    Up(dt, Normal(L * kGdnHeads, .5f, 4));
    auto g = Normal(L * kGdnDim, .1f, 5);
    for (auto& v : g) v += 1.f;
    Up(gamma, g);
    Up(qkv, Normal(U * M * kGdnQkvWidth, .3f, 6));
    Up(z, Normal(U * M * kGdnValueWidth, .3f, 7));
    Up(x, Normal(U * M * kHidden, .1f, 8));
    state0 = Normal(U * kState, .05f, 9);
    hist0 = Normal(U * kHist, .3f, 10);
    pend0 = PendInit(U, 11);
    xh0 = std::vector<__half>(U * M * kGdnValueWidth, __float2half(0.f));
    Reset();
  }
  size_t pitch;
  void Reset() {
    Up(state, state0);
    Up(hist, hist0);
    Up(pend, pend0);
    Up(xh, xh0);
    HIP_CHECK(hipDeviceSynchronize());
  }
  GdnArgs Args(int l, int s, const SpanShape& sh, int pos0) const {
    const size_t u = (size_t)l * spans + s, M = kMaxRows;
    GdnArgs a{};
    a.qkv = qkv + u * M * kGdnQkvWidth;
    a.z = z + u * M * kGdnValueWidth;
    a.x = x + u * M * kHidden;
    a.ab = ab + (size_t)l * pitch;
    a.conv = conv + (size_t)l * kGdnQkvWidth * 4;
    a.A_log = alog + (size_t)l * kGdnHeads;
    a.dt_bias = dt + (size_t)l * kGdnHeads;
    a.gamma = gamma + (size_t)l * kGdnDim;
    a.hist = hist + u * kHist;
    a.state = state + u * kState;
    a.suh_out = suh + (size_t)l * kGdnValueWidth;
    a.xh_out = xh + u * M * kGdnValueWidth;
    a.pend = pend + u * kPendN;
    a.pos0 = pos0 + 11 * s;
    a.R = sh.R;
    a.replay = sh.replay;
    a.commit = sh.commit;
    return a;
  }
};

using Packed = PackedStateArgs<GdnArgs, kMaxRows>;

// pos0 advances by the rows of the first pass for the second (the engine's next forward).
template <int CACHE, bool SS>
void PassA(const Fixture& f, const Pass& p) {
  const int n = (int)p.spans.size();
  for (int l = 0; l < f.layers; ++l) {
    if (n == 1) {
      GdnKernel<CACHE, SS><<<kGdnHeads, 256>>>(f.Args(l, 0, p.spans[0], p.pos0));
    } else {
      Packed q{};
      for (int s = 0; s < n; ++s) q.rows[s] = f.Args(l, s, p.spans[s], p.pos0);
      GdnKernel<CACHE, SS, Packed><<<dim3(kGdnHeads, n), 256>>>(q);
    }
  }
  HIP_CHECK(hipGetLastError());
}
template <int CACHE, bool SS, int MUT = 0, int RB = 2>
void PassB(const Fixture& f, const Pass& p) {
  const int n = (int)p.spans.size();
  for (int l = 0; l < f.layers; ++l) {
    if (n == 1) {
      GdnVerifyKernel<CACHE, SS, GdnArgs, MUT, RB><<<kGdnHeads, kGdnVerifyThreads>>>(
          f.Args(l, 0, p.spans[0], p.pos0));
    } else {
      Packed q{};
      for (int s = 0; s < n; ++s) q.rows[s] = f.Args(l, s, p.spans[s], p.pos0);
      GdnVerifyKernel<CACHE, SS, Packed, MUT, RB><<<dim3(kGdnHeads, n), kGdnVerifyThreads>>>(q);
    }
  }
  HIP_CHECK(hipGetLastError());
}
// The engine launch function (one span): the same kernel as PassB<..., 0, 2>.
void PassEngine(const Fixture& f, const Pass& p, int cache) {
  for (int l = 0; l < f.layers; ++l) {
    if (p.spans.size() == 1) LaunchGdnVerify(f.Args(l, 0, p.spans[0], p.pos0), 1, cache);
    else {
      Packed q{};
      for (size_t s = 0; s < p.spans.size(); ++s) q.rows[s] = f.Args(l, (int)s, p.spans[s], p.pos0);
      LaunchGdnVerify(q, (unsigned)p.spans.size(), cache);
    }
  }
  HIP_CHECK(hipGetLastError());
}

using PassFn = void (*)(const Fixture&, const Pass&);

template <class K> void PrintAttr(const char* name, K kernel) {
  hipFuncAttributes at{};
  HIP_CHECK(hipFuncGetAttributes(&at, reinterpret_cast<const void*>(kernel)));
  std::printf("attr %s vgprs=%d scratch=%zu lds=%zu maxthreads=%d\n", name, at.numRegs, at.localSizeBytes,
              at.sharedSizeBytes, at.maxThreadsPerBlock);
}
void PrintAttrs() {
  PrintAttr("A_solo_c4", GdnKernel<4, false, GdnArgs>);
  PrintAttr("B_solo_c0", GdnVerifyKernel<0, false, GdnArgs>);
  PrintAttr("B_solo_c4", GdnVerifyKernel<4, false, GdnArgs>);
  PrintAttr("B_solo_c6", GdnVerifyKernel<4, true, GdnArgs>);
  PrintAttr("B_packed_c4", GdnVerifyKernel<4, false, Packed>);
  PrintAttr("B4_solo_c4", GdnVerifyKernel<4, false, GdnArgs, 0, 4>);
  PrintAttr("B8_solo_c4", GdnVerifyKernel<4, false, GdnArgs, 0, 8>);
  int occ = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &occ, reinterpret_cast<const void*>(GdnVerifyKernel<4, false, GdnArgs>), kGdnVerifyThreads, 0));
  std::printf("attr B_solo_c4 occupancy_blocks_per_cu=%d (runtime estimate)\n", occ);
}

struct Snap {
  std::vector<float> state, hist, pend;
  std::vector<__half> xh;
};
Snap Take(const Fixture& f) {
  HIP_CHECK(hipDeviceSynchronize());
  return {Down(f.state, f.units * kState), Down(f.hist, f.units * kHist), Down(f.pend, f.units * kPendN),
          Down(f.xh, f.units * kMaxRows * kGdnValueWidth)};
}
template <class T> size_t Diff(const std::vector<T>& a, const std::vector<T>& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
  return n;
}
size_t Compare(const Snap& a, const Snap& b, char* buf, size_t len) {
  const size_t s = Diff(a.state, b.state), h = Diff(a.hist, b.hist), p = Diff(a.pend, b.pend),
               x = Diff(a.xh, b.xh);
  std::snprintf(buf, len, "state=%zu hist=%zu pend=%zu xh=%zu", s, h, p, x);
  return s + h + p + x;
}
// Two forwards: the second starts R rows later and replays the first's pending rows (replay = R - commit of the
// first, capped at its accepted count here: the shape's own replay is kept when it is smaller).
Snap Run(Fixture& f, const std::function<void(const Pass&)>& fn, const Pass& p) {
  f.Reset();
  fn(p);
  Pass q = p;
  for (size_t s = 0; s < q.spans.size(); ++s) {
    const int pending = std::max(0, p.spans[s].R - p.spans[s].commit);
    q.spans[s].replay = std::min(pending, std::max(p.spans[s].replay, 1));
  }
  int adv = 0;
  for (const auto& s : p.spans) adv = std::max(adv, s.R);
  q.pos0 = p.pos0 + adv;
  fn(q);
  return Take(f);
}

std::string Describe(const Pass& p) {
  std::string s = "pos=" + std::to_string(p.pos0) + " spans=";
  int total = 0;
  for (size_t i = 0; i < p.spans.size(); ++i) {
    const auto& v = p.spans[i];
    s += (i ? "," : "") + std::to_string(v.R) + "/" + std::to_string(v.replay) + "/" + std::to_string(v.commit);
    total += v.R;
  }
  return "rows=" + std::to_string(total) + " " + s + " (R/replay/commit)";
}

std::vector<Pass> CheckPasses() {
  std::vector<Pass> out;
  for (int pos : {1, 32768}) {
    for (int R = 1; R <= kMaxRows; ++R) {
      // replay 0, a middle value and the maximum; commit 1 (verify), R (decode/prefill) and a middle value.
      for (int replay : {0, R / 2 + 1, kMaxRows - 1})
        for (int commit : {1, R, (R + 1) / 2}) out.push_back({{{R, std::min(replay, kMaxRows), commit}}, pos});
    }
    // Packed: 2 spans of 8 + 1..8 rows (total 9..16), 3 and 4 spans.
    for (int R2 = 1; R2 <= kMaxRows; ++R2) out.push_back({{{8, 5, 1}, {R2, R2 % 8, 1}}, pos});
    out.push_back({{{4, 2, 1}, {3, 1, 1}, {2, 7, 2}}, pos});
    out.push_back({{{4, 3, 1}, {4, 0, 1}, {4, 2, 4}, {4, 7, 1}}, pos});
  }
  return out;
}

int Check() {
  // 3 layers (the per-layer arithmetic does not depend on the layer), 4 span slots.
  Fixture f(3, 4);
  char buf[200];
  int bad = 0, runs = 0;
  struct V {
    const char* cache;
    int c;
    PassFn a, b;
  };
  const V kV[] = {{"0", 0, PassA<0, false>, PassB<0, false>},
                  {"4", 4, PassA<4, false>, PassB<4, false>},
                  {"6", 6, PassA<4, true>, PassB<4, true>}};
  const auto passes = CheckPasses();
  for (const auto& v : kV) {
    for (const auto& p : passes) {
      const Snap a = Run(f, [&](const Pass& q) { v.a(f, q); }, p);
      const Snap b = Run(f, [&](const Pass& q) { v.b(f, q); }, p);
      const size_t n = Compare(a, b, buf, sizeof buf);
      ++runs;
      if (n || p.pos0 == 32768) std::printf("check cache=%s %s %s %s\n", v.cache, Describe(p).c_str(), n ? "FAIL" : "PASS", buf);
      bad += n != 0;
    }
    // The engine launch function and the RB = 4 and 8 bodies on a few shapes.
    for (const Pass& p : {Pass{{{4, 3, 1}}, 32768}, Pass{{{8, 7, 1}}, 32768}, Pass{{{8, 5, 1}, {8, 2, 1}}, 1}}) {
      const Snap a = Run(f, [&](const Pass& q) { v.a(f, q); }, p);
      const Snap e = Run(f, [&](const Pass& q) { PassEngine(f, q, v.c); }, p);
      const Snap b2 = Run(f, [&](const Pass& q) {
        if (v.c == 0) PassB<0, false, 0, 4>(f, q);
        else if (v.c == 4) PassB<4, false, 0, 4>(f, q);
        else PassB<4, true, 0, 4>(f, q);
      }, p);
      const Snap b8 = Run(f, [&](const Pass& q) {
        if (v.c == 0) PassB<0, false, 0, 8>(f, q);
        else if (v.c == 4) PassB<4, false, 0, 8>(f, q);
        else PassB<4, true, 0, 8>(f, q);
      }, p);
      for (const auto* s : {&e, &b2, &b8}) {
        const size_t n = Compare(a, *s, buf, sizeof buf);
        const char* name = s == &e ? "engine-launch" : s == &b2 ? "RB4" : "RB8";
        std::printf("check cache=%s %s %s %s %s\n", v.cache, name, Describe(p).c_str(), n ? "FAIL" : "PASS", buf);
        bad += n != 0;
        ++runs;
      }
    }
  }
  std::printf("checks %d run, %d failing\n", runs, bad);
  // Mutants: each must differ from A on at least one of its shapes (cache 4, the policy default).
  struct M {
    int id;
    PassFn b;
    const char* what;
  };
  const M kM[] = {{1, PassB<4, false, 1>, "one 128-long mem chain"},
                  {2, PassB<4, false, 2>, "state stored one row late"},
                  {3, PassB<4, false, 3>, "pending beta = decay"},
                  {4, PassB<4, false, 4>, "no eps in the output norm"},
                  {5, PassB<4, false, 5>, "no q/k ring write"},
                  {6, PassB<4, false, 6>, "first replay row skips decay"},
                  {7, PassB<4, false, 7>, "a/b chains reversed"},
                  {8, PassB<4, false, 8>, "one ulp on delta of the last row"}};
  const Pass kMp[] = {{{{4, 3, 1}}, 32768}, {{{1, 0, 1}}, 32768}, {{{8, 5, 3}}, 1}, {{{8, 5, 1}, {6, 2, 1}}, 32768}};
  int missed = 0;
  for (const auto& m : kM) {
    size_t total = 0;
    std::string detail;
    for (const auto& p : kMp) {
      const Snap a = Run(f, [&](const Pass& q) { PassA<4, false>(f, q); }, p);
      const Snap b = Run(f, [&](const Pass& q) { m.b(f, q); }, p);
      total += Compare(a, b, buf, sizeof buf);
      detail += std::string(" [") + Describe(p) + " " + buf + "]";
    }
    std::printf("mutant %d (%s) %s%s\n", m.id, m.what, total ? "DETECTED" : "MISSED", detail.c_str());
    missed += total == 0;
  }
  const bool ok = bad == 0 && missed == 0;
  std::printf("SUMMARY %s (checks failing %d, mutants missed %d)\n", ok ? "PASS" : "FAIL", bad, missed);
  std::fflush(stdout);
  return ok ? 0 : 3;
}

// ---- timing ----

struct La {
  LookaheadState* s = nullptr;
  hipStream_t stream{};
  uint8_t* base;
  size_t bytes;
  La(uint8_t* b, size_t n) : base(b), bytes(n) {
    LookaheadState st{};
    st.base = base;
    st.total = bytes;
    HIP_CHECK(hipMalloc(&s, sizeof(st)));
    HIP_CHECK(hipMemcpy(s, &st, sizeof(st), hipMemcpyHostToDevice));
    LookaheadInitKernel<<<1, 1>>>(s);
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK(hipDeviceSynchronize());
  }
  void Start() {
    LookaheadResetKernel<<<1, 1>>>(s, (unsigned long long)24 << 20);
    HIP_CHECK(hipDeviceSynchronize());
    LookaheadKernel<<<4, 256, 0, stream>>>(s);
  }
  unsigned long long Stop() {
    LookaheadStopKernel<<<1, 1>>>(s);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipDeviceSynchronize());
    LookaheadState st{};
    HIP_CHECK(hipMemcpy(&st, s, sizeof(st), hipMemcpyDeviceToHost));
    return st.fetched;
  }
};

struct Timed {
  double ms, mhz, la_mb;
};
Timed TimeRun(Fixture& f, PassFn fn, const Pass& p, int passes, bench::Sampler& smp, La* la) {
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  for (int i = 0; i < 2; ++i) fn(f, p);
  HIP_CHECK(hipDeviceSynchronize());
  smp.Start();
  std::vector<double> ms;
  double la_bytes = 0;
  for (int i = 0; i < passes; ++i) {
    if (la) la->Start();
    HIP_CHECK(hipEventRecord(e0));
    fn(f, p);
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    if (la) la_bytes += (double)la->Stop();
    float t = 0;
    HIP_CHECK(hipEventElapsedTime(&t, e0, e1));
    ms.push_back(t);
  }
  smp.Stop();
  HIP_CHECK(hipEventDestroy(e0));
  HIP_CHECK(hipEventDestroy(e1));
  return {bench::Median(ms), smp.MeanMhz(), la ? la_bytes / passes / 1e6 : 0.0};
}

int Time(int pairs, int passes, bool use_la, int spans) {
  // a/b weights of layer l at l * 32 MiB in a lookahead arena (the engine publishes the a/b weights' offset).
  const size_t pitch = (size_t)32 << 20;
  uint8_t* arena = nullptr;
  const size_t arena_bytes = pitch * (kGdnLayers + 1);
  HIP_CHECK(hipMalloc(&arena, arena_bytes));
  HIP_CHECK(hipMemset(arena, 0x3c, arena_bytes));
  Fixture f(kGdnLayers, spans, arena, pitch);
  La la(arena, arena_bytes);
  bench::Sampler smp;
  struct Var {
    const char* name;
    PassFn fn;
  };
  const Var kA{"A", PassA<4, false>}, kB{"B", PassB<4, false>}, kB2{"B4", PassB<4, false, 0, 4>},
      kB8{"B8", PassB<4, false, 0, 8>};
  std::printf("time pairs=%d passes=%d la=%d spans=%d (ms per pass of %d layers, us per launch)\n", pairs, passes,
              use_la, spans, kGdnLayers);
  for (int R = 1; R <= kMaxRows; ++R) {
    const Pass p{std::vector<SpanShape>(spans, SpanShape{R, R / 2, 1}), 32768};
    std::vector<Var> vars{kA, kB};
    if (R == 2 || R == 4 || R == 8) {
      vars.push_back(kB2);
      vars.push_back(kB8);
    }
    const size_t nv = vars.size();
    std::vector<std::vector<double>> t(nv), mb(nv);
    for (int k = 0; k < pairs; ++k) {
      // Rotated order: pair k starts at variant k % nv, odd pairs reversed (ABBA for two variants).
      std::vector<size_t> order;
      for (size_t i = 0; i < nv; ++i) order.push_back((k + i) % nv);
      if (k & 1) std::reverse(order.begin(), order.end());
      std::string line;
      for (size_t i : order) {
        const Timed r = TimeRun(f, vars[i].fn, p, passes, smp, use_la ? &la : nullptr);
        t[i].push_back(r.ms);
        mb[i].push_back(r.la_mb);
        char b[96];
        std::snprintf(b, sizeof b, " %s=%.4f ms (%.0f MHz%s%.1f MB)", vars[i].name, r.ms, r.mhz,
                      use_la ? ", la " : "", use_la ? r.la_mb : 0.0);
        line += b;
      }
      std::printf("pair R=%d replay=%d %d%s\n", R, R / 2, k, line.c_str());
    }
    std::string line;
    const double a = bench::Median(t[0]);
    for (size_t i = 0; i < nv; ++i) {
      const double m = bench::Median(t[i]);
      char b[160];
      std::snprintf(b, sizeof b, " %s=%.4f ms %.2f us/launch ratio=%.3f%s", vars[i].name, m, m * 1000 / kGdnLayers,
                    m / a, use_la ? (" la_mb=" + std::to_string(bench::Median(mb[i]))).c_str() : "");
      line += b;
    }
    std::printf("time R=%d replay=%d%s\n", R, R / 2, line.c_str());
    std::fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "check";
  PrintAttrs();
  std::fflush(stdout);
  if (mode == "check") return Check();
  if (mode == "time") {
    const int pairs = argc > 2 ? std::atoi(argv[2]) : 6, passes = argc > 3 ? std::atoi(argv[3]) : 20;
    const bool la = argc > 4 && std::atoi(argv[4]) != 0;
    const int spans = argc > 5 ? std::atoi(argv[5]) : 1;
    if (spans < 1 || spans > kMaxRows) {
      std::fprintf(stderr, "spans needs 1..%d\n", kMaxRows);
      return 2;
    }
    return Time(pairs, passes, la, spans);
  }
  std::fprintf(stderr, "usage: gdn_verify check | time [pairs] [passes] [la] [spans]\n");
  return 2;
}
