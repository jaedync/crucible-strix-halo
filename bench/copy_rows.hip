// copy_rows (copy-e E3): the row-templated decode kernels of a lone copy span at 16, 24 and 32 rows, on synthetic
// weights at the engine's shapes. A pass above 16 rows has two forms on today's kernels:
//   one launch of R rows (the ring with the lifted cap of ring_rows.h, DenseRowsKernel<R>), or
//   row groups of 16 (two launches over the same weights: the host form, each row bitwise its 16-row value).
// HC (HcW2Launch) has only the group form: its WMMA row tile is 16 rows (engine.hip HcRead loops over the tiles).
//
//   copy_rows [K] [reps] [check|time|all] [rows list, default 16,24,32]
//
// check: every row of a one-launch pass of `rows` rows equals the same row alone (R = 1), byte for byte: the ring for
//   every split partial, the dense rows for every output. Exit 1 when a check fails.
// time: median ms of one launch (or one group pair) per shape and form, cycling over weight copies (>= 384 MB) so every
//   launch reads cold tiles. "ring/pass", "dense/pass" and "hc/pass" sum one forward (36 GDN layers, 12 QSA layers,
//   the head; router x 48, GDN a/b x 36, PLE kv x 1; HC 2 x 48 + 1 sites). The sclk mean is printed.
// No engine state, no model file, no user data.
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../engine/exl3/lookahead.h"
#include "../engine/exl3/ring_rows.h"
#include "../engine/exl3/decode_split.h"
#include "../engine/layers/dense_rows.h"
#include "../engine/layers/hc_wmma2.h"
#include "../engine/layers/prefill_hc.h"
#include "../engine/model/flashnext/geometry.h"
#include "common.h"

using namespace crucible;

namespace {

constexpr int kGrid = 80;  // Engine::kRingBlocks
constexpr int kRowsMax = ring_rows::kRingRowsMax;  // 32

__global__ void FillWords(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
    z ^= z >> 13;
    z *= 0x5bd1e995u;
    z ^= z >> 15;
    p[i] = z;
  }
}

// Activations in [-2, 2): the scale of transformed decode inputs.
__global__ void FillHalf(__half* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2246822519u) ^ seed;
    z ^= z >> 15;
    z *= 0x2c1b3c6du;
    z ^= z >> 12;
    p[i] = __float2half(((z & 0xffff) / 65536.f - 0.5f) * 4.f);
  }
}

template <class T>
T* Alloc(size_t n) {
  T* p = nullptr;
  HIP_CHECK(hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  HIP_CHECK(hipMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

struct Mat { int I, O, K; };

struct Shape {
  std::string name;
  std::vector<Mat> mats;
  int per_pass;  // groups of this shape in one forward
};

// One group: weight copies, activations for kRowsMax rows, partials.
struct RingGroup {
  std::vector<LinJob> jobs;         // copy 0
  std::vector<LinJob*> d_jobs;      // per copy
  std::vector<GemvItem> items;
  GemvItem* d_items = nullptr;
  size_t tile_bytes = 0;
  int maxspan = 0;
  int copies = 1;
};

RingGroup Make(const Shape& sh, int copies, uint32_t seed) {
  RingGroup g;
  g.copies = copies;
  std::vector<std::pair<int, int>> io;
  for (const Mat& m : sh.mats) io.push_back({m.I, m.O});
  const std::vector<int> splits = DecodeSplits(io);
  std::vector<std::vector<uint32_t*>> tiles(copies);
  for (size_t j = 0; j < sh.mats.size(); ++j) {
    const Mat& m = sh.mats[j];
    LinJob job{};
    job.I = m.I;
    job.O = m.O;
    job.K = m.K;
    job.S = splits[j];
    const int A = m.I / 16;
    if (!FrontFitsShape(A, m.O / 16, job.S)) {
      std::fprintf(stderr, "%s job %zu: the front order does not fit\n", sh.name.c_str(), j);
      std::exit(2);
    }
    const size_t words = (size_t)A * (m.O / 16) * 8 * m.K;
    g.tile_bytes += words * 4;
    for (int c = 0; c < copies; ++c) {
      uint32_t* t = Alloc<uint32_t>(words);
      FillWords<<<1024, 256>>>(t, words, seed + 977 * (uint32_t)j + 31 * (uint32_t)c);
      tiles[c].push_back(t);
    }
    job.xh = Alloc<__half>((size_t)kRowsMax * m.I);
    job.partial = Alloc<float>((size_t)job.S * kRowsMax * m.O);
    g.maxspan = std::max(g.maxspan, A / job.S);
    for (int s = 0; s < job.S; ++s)
      for (int b0 = 0; b0 < m.O / 16; b0 += 8) g.items.push_back({(int)j, s, b0, std::min(8, m.O / 16 - b0)});
    g.jobs.push_back(job);
  }
  for (int c = 0; c < copies; ++c) {
    std::vector<LinJob> js = g.jobs;
    for (size_t j = 0; j < js.size(); ++j) js[j].tiles = tiles[c][j];
    if (c == 0) g.jobs = js;
    LinJob* d = Alloc<LinJob>(js.size());
    HIP_CHECK(hipMemcpy(d, js.data(), js.size() * sizeof(LinJob), hipMemcpyHostToDevice));
    g.d_jobs.push_back(d);
  }
  g.d_items = Alloc<GemvItem>(g.items.size());
  HIP_CHECK(hipMemcpy(g.d_items, g.items.data(), g.items.size() * sizeof(GemvItem), hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());
  return g;
}

// R rows of the row-major activations; job.R and the template R agree.
void SetRows(RingGroup& g, int R) {
  for (auto& j : g.jobs) j.R = R;
  for (LinJob* d : g.d_jobs) {
    std::vector<LinJob> js = g.jobs;
    for (size_t j = 0; j < js.size(); ++j) {
      LinJob tmp;
      HIP_CHECK(hipMemcpy(&tmp, d + j, sizeof(LinJob), hipMemcpyDeviceToHost));
      js[j].tiles = tmp.tiles;
    }
    HIP_CHECK(hipMemcpy(d, js.data(), js.size() * sizeof(LinJob), hipMemcpyHostToDevice));
  }
}


using Launch = void (*)(const LinJob*, const RingGroup&, int);

template <int R>
void Ring(const LinJob* jobs, const RingGroup& g, int) {
  const int n = (int)g.items.size(), grid = std::min(kGrid, n);
  if constexpr (R == 1) {
    const size_t lds = (size_t)g.maxspan * 32;
    LinRingFlowKernel<true><<<grid, 320, 2 * lds>>>(jobs, g.d_items, n, (int)(lds / 4), false, false);
  } else {
    LinRingRowsKernel<R><<<grid, 320>>>(jobs, g.d_items, n, false);
  }
}

Launch RingFor(int R) {
  switch (R) {
    case 1: return Ring<1>;
    case 8: return Ring<8>;
    case 16: return Ring<16>;
    case 24: return Ring<24>;
    case 32: return Ring<32>;
    default: return nullptr;
  }
}

// The engine's kernel for a pass of `rows` rows in one launch.
int KernelRows(int rows) { return rows <= kMaxRows ? rows : rows <= kPackedRows ? kPackedRows : (rows + 7) / 8 * 8; }

void SetR(std::vector<LinJob>& js, int R) { for (auto& j : js) j.R = R; }

// Device job arrays for each copy with R set; hi: xh and partial moved to rows 16.. (the second group).
std::vector<LinJob*> JobsFor(const RingGroup& g, int R, bool hi, std::vector<float*>* parts = nullptr) {
  std::vector<LinJob*> out;
  for (int c = 0; c < g.copies; ++c) {
    std::vector<LinJob> js = g.jobs;
    for (size_t j = 0; j < js.size(); ++j) {
      LinJob tmp;
      HIP_CHECK(hipMemcpy(&tmp, g.d_jobs[c] + j, sizeof(LinJob), hipMemcpyDeviceToHost));
      js[j].tiles = tmp.tiles;
      js[j].R = R;
      if (hi) {
        js[j].xh = g.jobs[j].xh + (size_t)kPackedRows * js[j].I;
        if (parts) js[j].partial = (*parts)[j];
      }
    }
    LinJob* d = Alloc<LinJob>(js.size());
    HIP_CHECK(hipMemcpy(d, js.data(), js.size() * sizeof(LinJob), hipMemcpyHostToDevice));
    out.push_back(d);
  }
  return out;
}

std::vector<float> Partials(const float* p, const LinJob& job, int R) {
  std::vector<float> v((size_t)job.S * R * job.O);
  HIP_CHECK(hipMemcpy(v.data(), p, v.size() * 4, hipMemcpyDeviceToHost));
  return v;
}

// Rows 0..rows-1 random (seed), rows rows..kRowsMax-1 NaN.
void FillX(const RingGroup& g, int rows, uint32_t seed) {
  for (const LinJob& j : g.jobs) {
    HIP_CHECK(hipMemset(j.xh, 0xff, (size_t)kRowsMax * j.I * 2));
    FillHalf<<<256, 256>>>(j.xh, (size_t)rows * j.I, seed);
  }
  HIP_CHECK(hipDeviceSynchronize());
}

int fails = 0;

bool CheckRing(RingGroup& g, int rows) {
  const int KR = KernelRows(rows);
  const Launch many = RingFor(KR), one = RingFor(1);
  if (!many) return true;
  FillX(g, rows, 1234 + rows);
  std::vector<std::vector<__half>> x(g.jobs.size());
  for (size_t j = 0; j < g.jobs.size(); ++j) {
    x[j].resize((size_t)rows * g.jobs[j].I);
    HIP_CHECK(hipMemcpy(x[j].data(), g.jobs[j].xh, x[j].size() * 2, hipMemcpyDeviceToHost));
  }
  for (const LinJob& j : g.jobs) HIP_CHECK(hipMemset(j.partial, 0xff, (size_t)j.S * KR * j.O * 4));
  std::vector<LinJob*> jm = JobsFor(g, KR, false), j1 = JobsFor(g, 1, false);
  many(jm[0], g, 0);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::vector<float>> got(g.jobs.size());
  for (size_t j = 0; j < g.jobs.size(); ++j) got[j] = Partials(g.jobs[j].partial, g.jobs[j], KR);
  long bad = 0;
  for (int r = 0; r < rows; ++r) {
    for (size_t j = 0; j < g.jobs.size(); ++j)
      HIP_CHECK(hipMemcpy(g.jobs[j].xh, x[j].data() + (size_t)r * g.jobs[j].I, g.jobs[j].I * 2, hipMemcpyHostToDevice));
    for (const LinJob& j : g.jobs) HIP_CHECK(hipMemset(j.partial, 0xff, (size_t)j.S * j.O * 4));
    one(j1[0], g, 0);
    HIP_CHECK(hipDeviceSynchronize());
    for (size_t j = 0; j < g.jobs.size(); ++j) {
      const LinJob& job = g.jobs[j];
      const std::vector<float> alone = Partials(job.partial, job, 1);
      for (int s = 0; s < job.S; ++s)
        if (std::memcmp(&alone[(size_t)s * job.O], &got[j][((size_t)s * KR + r) * job.O], job.O * 4) != 0) ++bad;
    }
  }
  for (LinJob* d : jm) HIP_CHECK(hipFree(d));
  for (LinJob* d : j1) HIP_CHECK(hipFree(d));
  std::printf("check ring rows=%2d (kernel %2d): %s", rows, KR, bad ? "DIFFERENT" : "SAME");
  if (bad) std::printf(" (%ld split rows differ)", bad);
  std::printf("\n");
  if (bad) ++fails;
  return !bad;
}

// One timed form: a list of (launch, jobs per copy).
struct Form { std::vector<std::pair<Launch, std::vector<LinJob*>>> parts; };

double TimeForm(const RingGroup& g, const Form& f, int reps) {
  for (int c = 0; c < g.copies; ++c)
    for (const auto& [l, js] : f.parts) l(js[c], g, c);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<double> ms;
  bench::EventTimer t;
  for (int i = 0; i < reps; ++i) {
    t.Begin();
    for (const auto& [l, js] : f.parts) l(js[i % g.copies], g, i % g.copies);
    ms.push_back(t.EndMs());
  }
  return bench::Median(ms);
}

// ---- dense rows (router O 513, GDN a/b O 96, PLE kv O 12800; I = kHidden) ----
template <int R>
void DenseL(const __half* W, int O, const float* x, float* y) {
  if constexpr (R == 1) DenseGemvKernel<1><<<(O + 7) / 8, 256>>>(W, O, x, kHidden, y, O);
  else LaunchDenseRows<R, 1>(W, O, x, kHidden, y, O);
}
using DLaunch = void (*)(const __half*, int, const float*, float*);
DLaunch DenseFor(int R) {
  switch (R) {
    case 1: return DenseL<1>;
    case 8: return DenseL<8>;
    case 16: return DenseL<16>;
    case 24: return DenseL<24>;
    case 32: return DenseL<32>;
    default: return nullptr;
  }
}

__global__ void FillHalfW(__half* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
    z ^= z >> 13; z *= 0x5bd1e995u; z ^= z >> 15;
    p[i] = __float2half(((z & 0xffff) / 65536.f - 0.5f) * 0.1f);
  }
}
__global__ void FillFloat(float* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2246822519u) ^ seed;
    z ^= z >> 15; z *= 0x2c1b3c6du; z ^= z >> 12;
    p[i] = ((z & 0xffffff) / 16777216.f - 0.5f) * scale;
  }
}

struct DenseShape { const char* name; int O, per_pass; };

void DenseRun(const DenseShape& d, const std::vector<int>& rlist, int reps, bool check, bool time,
              std::map<std::string, double>& pass) {
  const size_t wn = (size_t)d.O * kHidden;
  const int copies = (int)std::max<size_t>(1, (384u << 20) / (wn * 2) + 1);
  std::vector<__half*> W(copies);
  for (int c = 0; c < copies; ++c) { W[c] = Alloc<__half>(wn); FillHalfW<<<1024, 256>>>(W[c], wn, 17 + c); }
  float* x = Alloc<float>((size_t)kRowsMax * kHidden);
  float* y = Alloc<float>((size_t)kRowsMax * d.O);
  float* y1 = Alloc<float>(d.O);
  FillFloat<<<256, 256>>>(x, (size_t)kRowsMax * kHidden, 5, 4.f);
  HIP_CHECK(hipDeviceSynchronize());
  for (int rows : rlist) {
    const int KR = rows <= 16 ? 16 : (rows + 7) / 8 * 8;
    if (check && KR > 16) {
      HIP_CHECK(hipMemset(y, 0xff, (size_t)KR * d.O * 4));
      DenseFor(KR)(W[0], d.O, x, y);
      std::vector<float> got((size_t)KR * d.O), one(d.O);
      HIP_CHECK(hipMemcpy(got.data(), y, got.size() * 4, hipMemcpyDeviceToHost));
      long bad = 0;
      for (int r = 0; r < rows; ++r) {
        DenseFor(1)(W[0], d.O, x + (size_t)r * kHidden, y1);
        HIP_CHECK(hipMemcpy(one.data(), y1, d.O * 4, hipMemcpyDeviceToHost));
        if (std::memcmp(one.data(), &got[(size_t)r * d.O], d.O * 4)) ++bad;
      }
      std::printf("check dense %-6s rows=%2d (kernel %2d): %s\n", d.name, rows, KR, bad ? "DIFFERENT" : "SAME");
      if (bad) ++fails;
    }
    if (!time) continue;
    auto t = [&](auto&& f) {
      for (int c = 0; c < copies; ++c) f(c);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<double> ms;
      bench::EventTimer e;
      for (int i = 0; i < reps; ++i) { e.Begin(); f(i % copies); ms.push_back(e.EndMs()); }
      return bench::Median(ms);
    };
    const double one = t([&](int c) { DenseFor(KR)(W[c], d.O, x, y); });
    std::printf("time dense %-6s rows=%2d one launch (kernel %2d)  %.4f ms\n", d.name, rows, KR, one);
    pass["dense one " + std::to_string(rows)] += d.per_pass * one;
    if (rows > 16) {
      const DLaunch hi = DenseFor(rows - 16 <= 8 ? 8 : 16);
      const double two = t([&](int c) { DenseFor(16)(W[c], d.O, x, y); hi(W[c], d.O, x + 16 * kHidden, y + 16 * d.O); });
      std::printf("time dense %-6s rows=%2d groups 16+%d            %.4f ms\n", d.name, rows, rows - 16 <= 8 ? 8 : 16, two);
      pass["dense groups " + std::to_string(rows)] += d.per_pass * two;
    }
  }
  for (__half* w : W) HIP_CHECK(hipFree(w));
  HIP_CHECK(hipFree(x)); HIP_CHECK(hipFree(y)); HIP_CHECK(hipFree(y1));
}

// ---- HC read (HcW2Launch, fp16 fragment order, the inject site) ----
void HcRun(const std::vector<int>& rlist, int reps, std::map<std::string, double>& pass) {
  constexpr int kM = kRank + kStreams;
  const size_t site = sizeof(__half) * ((size_t)kM * kWide + (size_t)kWide * kRank);
  const int copies = (int)std::max<size_t>(2, (384u << 20) / site + 1);
  struct Site { __half *fn, *up, *fnf, *upf; float* w1; };
  std::vector<Site> s(copies);
  for (int c = 0; c < copies; ++c) {
    s[c].fn = Alloc<__half>((size_t)kM * kWide); s[c].up = Alloc<__half>((size_t)kWide * kRank);
    s[c].fnf = Alloc<__half>((size_t)kHcWPad * kWide); s[c].upf = Alloc<__half>((size_t)kWide * kRank);
    s[c].w1 = Alloc<float>(kWide);
    FillHalfW<<<1024, 256>>>(s[c].fn, (size_t)kM * kWide, 3 + c);
    FillHalfW<<<1024, 256>>>(s[c].up, (size_t)kWide * kRank, 7 + c);
    FillFloat<<<64, 256>>>(s[c].w1, kWide, 11 + c, 0.4f);
    HcFragFnKernel<<<(kHcWPad * kWide + 255) / 256, 256>>>(s[c].fn, kM, s[c].fnf);
    HcFragUpKernel<<<(kWide * kRank + 255) / 256, 256>>>(s[c].up, s[c].upf);
  }
  float* H = Alloc<float>((size_t)kRowsMax * kWide);
  float* x = Alloc<float>((size_t)kRowsMax * kHidden);
  float* post = Alloc<float>((size_t)kRowsMax * kStreams);
  FillFloat<<<256, 256>>>(H, (size_t)kRowsMax * kWide, 5, 2.f);
  const HcWScratch sc = HcW2Alloc();
  unsigned epoch = 0;
  HIP_CHECK(hipDeviceSynchronize());
  auto launch = [&](int c, int R) {
    for (int o = 0; o < R; o += kHcWRows)  // engine.hip HcRead: one 16-row tile a launch
      HcW2Launch(s[c].fn, s[c].fnf, kM, s[c].up, s[c].upf, s[c].w1, H + (size_t)o * kWide, std::min(kHcWRows, R - o),
                 x + (size_t)o * kHidden, post + o * kStreams, sc, ++epoch);
  };
  for (int rows : rlist) {
    for (int c = 0; c < copies; ++c) launch(c, rows);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> ms;
    bench::EventTimer e;
    for (int i = 0; i < reps; ++i) { e.Begin(); launch(i % copies, rows); ms.push_back(e.EndMs()); }
    const double t = bench::Median(ms);
    std::printf("time hc site rows=%2d launches %d  %.4f ms\n", rows, (rows + kHcWRows - 1) / kHcWRows, t);
    pass["hc " + std::to_string(rows)] += (2 * kLayers + 1) * t;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int K = argc > 1 ? std::atoi(argv[1]) : 6;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 30;
  const std::string mode = argc > 3 ? argv[3] : "all";
  std::vector<int> rlist;
  for (const char* p = argc > 4 ? argv[4] : "16,24,32"; *p;) {
    rlist.push_back(std::atoi(p));
    const char* q = std::strchr(p, ',');
    p = q ? q + 1 : p + std::strlen(p);
  }
  if (K < 4 || K > 6) { std::fprintf(stderr, "K must be 4..6\n"); return 2; }
  for (int r : rlist)
    if (r != 8 && r != 16 && r != 24 && r != 32) { std::fprintf(stderr, "rows %d: use 8, 16, 24 or 32\n", r); return 2; }
  const int Ki = K - 2;
  constexpr int kQsaLayers = kLayers / kQsaPeriod, kGdnLayers = kLayers - kQsaLayers;
  const std::vector<Shape> shapes = {
      {"gdn_in", {{kHidden, kGdnQkvWidth, K}, {kHidden, kGdnValueWidth, K}}, kGdnLayers},
      {"out", {{kMixerOutWidth, kHidden, K}}, kLayers},
      {"qsa_in", {{kHidden, kQsaQWidth, K}, {kHidden, kQsaKvWidth, K}, {kHidden, kQsaKvWidth, K},
                  {kHidden, kQsaIdxWidth, Ki}}, kQsaLayers},
      {"head", {{kHidden, kVocab, K}}, 1},
  };
  const std::vector<DenseShape> dense = {{"router", kExperts + 1, kLayers}, {"gdn_ab", 96, kGdnLayers}, {"ple_kv", 12800, 1}};
  const bool check = mode == "check" || mode == "all", time = mode == "time" || mode == "all";
  std::printf("copy_rows K=%d (indexer K=%d) reps=%d mode=%s\n", K, Ki, reps, mode.c_str());
  if (check)
    for (size_t s = 0; s < 3; ++s) {
      RingGroup g = Make(shapes[s], 1, 7);
      for (int rows : rlist)
        if (rows > 1) CheckRing(g, rows);
    }
  std::map<std::string, double> pass;
  bench::Sampler sampler;
  if (time) sampler.Start();
  if (time)
    for (const Shape& sh : shapes) {
      size_t bytes = 0;
      for (const Mat& m : sh.mats) bytes += (size_t)(m.I / 16) * (m.O / 16) * 32 * m.K;
      const int copies = (int)std::max<size_t>(1, (384u << 20) / bytes + 1);
      RingGroup g = Make(sh, copies, 11);
      FillX(g, kRowsMax, 5);
      std::vector<float*> parts;
      for (const LinJob& j : g.jobs) parts.push_back(Alloc<float>((size_t)j.S * kRowsMax * j.O));
      for (int rows : rlist) {
        const int KR = KernelRows(rows);
        Form one{{{RingFor(KR), JobsFor(g, KR, false)}}};
        const double t1 = TimeForm(g, one, reps);
        const double gb = g.tile_bytes / 1e9;
        std::printf("time ring %-7s rows=%2d one launch (kernel %2d)  %.4f ms  %.1f GB/s\n", sh.name.c_str(), rows, KR, t1,
                    gb / t1 * 1e3);
        pass["ring one " + std::to_string(rows)] += sh.per_pass * t1;
        if (rows > 16) {
          const int hi = rows - 16 <= 8 ? 8 : 16;
          Form two{{{RingFor(16), JobsFor(g, 16, false)}, {RingFor(hi), JobsFor(g, hi, true, &parts)}}};
          const double t2 = TimeForm(g, two, reps);
          std::printf("time ring %-7s rows=%2d groups 16+%d            %.4f ms  %.1f GB/s\n", sh.name.c_str(), rows, hi, t2,
                      gb / t2 * 1e3);
          pass["ring groups " + std::to_string(rows)] += sh.per_pass * t2;
        }
      }
    }
  for (const DenseShape& d : dense) DenseRun(d, rlist, reps, check, time, pass);
  if (time) HcRun(rlist, reps, pass);
  if (time) {
    sampler.Stop();
    for (const auto& [name, t] : pass) std::printf("pass %s  %.2f ms\n", name.c_str(), t);
    std::printf("clocks: %s\n", sampler.Summary().c_str());
    std::printf("sclk_mean_mhz %.0f\n", sampler.MeanMhz());
  }
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
