// p418_dense (p418-kernel): the R=1 dense decode kernels of the 4.05 (K6) and 4.18 (K8) packs, on synthetic weights
// at the engine's shapes (GDN in, mixer out, QSA in; the head is K6 on both packs).
//
//   p418_dense [reps] [check|time|all] [grids, default 80]
//
// Forms (job.cache_policy 4, as the day policy; in and out transforms off in every form):
//   ring6F  LinRingFlowKernel<true>        K6 front order   (4.05 today)
//   ring6E  LinRingFlowKernel<false>       K6 engine order
//   pol8E   LinPolicyKernel<1, 8, false>   K8 engine order  (4.18 today)
//   pol8F   LinPolicyKernel<1, 8, true>    K8 front order
//   ring8E  LinRingFlowKernel<false, 8>    K8 engine order  (the wide ring)
//   ring8F  LinRingFlowKernel<true, 8>     K8 front order   (the wide ring)
// check: the split partials of ring8E equal pol8E, and ring8F on a front-order copy of the same matrix equals pol8E,
//   byte for byte, at R=1. Exit 1 when a check fails.
// time: median ms of one launch per shape and form, cycling over weight copies (>= 384 MB) so every launch reads cold
//   tiles; "pass" sums one forward (36 GDN in, 48 out, 12 QSA in). The sclk mean is printed.
// No engine state, no model file, no user data.
#include <cstring>
#include <string>
#include <vector>

#include "../engine/exl3/exl3_ring.h"
#include "../engine/exl3/decode_split.h"
#include "../engine/model/flashnext/geometry.h"
#include "common.h"

using namespace crucible;

namespace {

__global__ void FillWords(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
    z ^= z >> 13;
    z *= 0x5bd1e995u;
    z ^= z >> 15;
    p[i] = z;
  }
}

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
struct Shape { std::string name; std::vector<Mat> mats; int per_pass; };

struct Group {
  std::vector<LinJob> jobs;      // copy 0
  std::vector<LinJob*> d_jobs;   // per copy
  LinJob* d_front = nullptr;     // copy 0 with a front-order copy of its tiles
  std::vector<GemvItem> items;
  GemvItem* d_items = nullptr;
  size_t tile_bytes = 0;
  int maxspan = 0;
  int copies = 1;
};

Group Make(const Shape& sh, int copies, uint32_t seed, bool front_copy) {
  Group g;
  g.copies = copies;
  std::vector<std::pair<int, int>> io;
  for (const Mat& m : sh.mats) io.push_back({m.I, m.O});
  const std::vector<int> splits = DecodeSplits(io);
  std::vector<std::vector<uint32_t*>> tiles(copies);
  std::vector<uint32_t*> front;
  for (size_t j = 0; j < sh.mats.size(); ++j) {
    const Mat& m = sh.mats[j];
    LinJob job{};
    job.I = m.I;
    job.O = m.O;
    job.K = m.K;
    job.S = splits[j];
    job.R = 1;
    job.cache_policy = 4;
    const int A = m.I / 16, B = m.O / 16;
    if (!FrontFitsShape(A, B, job.S)) {
      std::fprintf(stderr, "%s job %zu: the front order does not fit\n", sh.name.c_str(), j);
      std::exit(2);
    }
    const size_t T = 8 * m.K, words = (size_t)A * B * T;
    g.tile_bytes += words * 4;
    for (int c = 0; c < copies; ++c) {
      uint32_t* t = Alloc<uint32_t>(words);
      FillWords<<<1024, 256>>>(t, words, seed + 977 * (uint32_t)j + 31 * (uint32_t)c);
      tiles[c].push_back(t);
    }
    if (front_copy) {
      // Engine order holds tile (a, b) at b * A + a; the front copy moves it to FrontTile (the loader's rule).
      std::vector<uint32_t> e(words), f(words);
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(e.data(), tiles[0][j], words * 4, hipMemcpyDeviceToHost));
      for (int b = 0; b < B; ++b)
        for (int a = 0; a < A; ++a)
          std::memcpy(&f[FrontTile(b, a, A, B, job.S) * T], &e[((size_t)b * A + a) * T], T * 4);
      uint32_t* t = Alloc<uint32_t>(words);
      HIP_CHECK(hipMemcpy(t, f.data(), words * 4, hipMemcpyHostToDevice));
      front.push_back(t);
    }
    job.xh = Alloc<__half>((size_t)m.I);
    job.partial = Alloc<float>((size_t)job.S * m.O);
    g.maxspan = std::max(g.maxspan, A / job.S);
    for (int s = 0; s < job.S; ++s)
      for (int b0 = 0; b0 < B; b0 += 8) g.items.push_back({(int)j, s, b0, std::min(8, B - b0)});
    g.jobs.push_back(job);
  }
  auto upload = [&](const std::vector<uint32_t*>& t) {
    std::vector<LinJob> js = g.jobs;
    for (size_t j = 0; j < js.size(); ++j) js[j].tiles = t[j];
    LinJob* d = Alloc<LinJob>(js.size());
    HIP_CHECK(hipMemcpy(d, js.data(), js.size() * sizeof(LinJob), hipMemcpyHostToDevice));
    return std::make_pair(d, js);
  };
  for (int c = 0; c < copies; ++c) {
    auto [d, js] = upload(tiles[c]);
    if (c == 0) g.jobs = js;
    g.d_jobs.push_back(d);
  }
  if (front_copy) g.d_front = upload(front).first;
  g.d_items = Alloc<GemvItem>(g.items.size());
  HIP_CHECK(hipMemcpy(g.d_items, g.items.data(), g.items.size() * sizeof(GemvItem), hipMemcpyHostToDevice));
  for (const LinJob& j : g.jobs) FillHalf<<<256, 256>>>(j.xh, (size_t)j.I, 5 + j.O);
  HIP_CHECK(hipDeviceSynchronize());
  return g;
}

enum Form { kRing6F, kRing6E, kPol8E, kPol8F, kRing8E, kRing8F, kForms };
const char* kNames[kForms] = {"ring6F", "ring6E", "pol8E", "pol8F", "ring8E", "ring8F"};

void Launch(Form f, const LinJob* jobs, const Group& g, int grid) {
  const int n = (int)g.items.size(), rg = std::min(grid, n);
  const size_t lds = (size_t)g.maxspan * 32;
  const int words = (int)(lds / 4);
  switch (f) {
    case kRing6F: LinRingFlowKernel<true><<<rg, 320, 2 * lds>>>(jobs, g.d_items, n, words, false, false); break;
    case kRing6E: LinRingFlowKernel<false><<<rg, 320, 2 * lds>>>(jobs, g.d_items, n, words, false, false); break;
    case kPol8E: LinPolicyKernel<1, kMaxWideDecodeK><<<n, 256, lds>>>(jobs, g.d_items, n, false, false); break;
    case kPol8F: LinPolicyKernel<1, kMaxWideDecodeK, true><<<n, 256, lds>>>(jobs, g.d_items, n, false, false); break;
    case kRing8E:
      LinRingFlowKernel<false, kMaxWideDecodeK><<<rg, 320, 2 * lds>>>(jobs, g.d_items, n, words, false, false);
      break;
    case kRing8F:
      LinRingFlowKernel<true, kMaxWideDecodeK><<<rg, 320, 2 * lds>>>(jobs, g.d_items, n, words, false, false);
      break;
    default: break;
  }
  HIP_CHECK(hipGetLastError());
}

std::vector<std::vector<float>> Run(Form f, const LinJob* jobs, const Group& g, int grid) {
  for (const LinJob& j : g.jobs) HIP_CHECK(hipMemset(j.partial, 0xff, (size_t)j.S * j.O * 4));
  Launch(f, jobs, g, grid);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::vector<float>> out;
  for (const LinJob& j : g.jobs) {
    out.emplace_back((size_t)j.S * j.O);
    HIP_CHECK(hipMemcpy(out.back().data(), j.partial, out.back().size() * 4, hipMemcpyDeviceToHost));
  }
  return out;
}

int fails = 0;

void Check(const Shape& sh, int grid) {
  Group g = Make(sh, 1, 7, true);
  const auto ref = Run(kPol8E, g.d_jobs[0], g, grid);
  const std::pair<Form, const LinJob*> forms[] = {{kRing8E, g.d_jobs[0]}, {kRing8F, g.d_front}, {kPol8F, g.d_front}};
  for (const auto& [f, jobs] : forms) {
    const auto got = Run(f, jobs, g, grid);
    long bad = 0, nan = 0;
    for (size_t j = 0; j < ref.size(); ++j) {
      for (size_t i = 0; i < ref[j].size(); ++i) {
        if (std::memcmp(&ref[j][i], &got[j][i], 4) != 0) ++bad;
        if (got[j][i] != got[j][i]) ++nan;
      }
    }
    std::printf("check %-7s %-6s grid %d against pol8E: %s", sh.name.c_str(), kNames[f], grid, bad ? "DIFFERENT" : "SAME");
    if (bad || nan) std::printf(" (%ld values differ, %ld NaN)", bad, nan);
    std::printf("\n");
    if (bad || nan) ++fails;
  }
}

double Time(Form f, const Group& g, int grid, int reps) {
  for (int c = 0; c < g.copies; ++c) Launch(f, g.d_jobs[c], g, grid);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<double> ms;
  bench::EventTimer t;
  for (int i = 0; i < reps; ++i) {
    t.Begin();
    Launch(f, g.d_jobs[i % g.copies], g, grid);
    ms.push_back(t.EndMs());
  }
  return bench::Median(ms);
}

}  // namespace

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 30;
  const std::string mode = argc > 2 ? argv[2] : "all";
  std::vector<int> grids;
  for (const char* p = argc > 3 ? argv[3] : "80"; *p;) {
    grids.push_back(std::atoi(p));
    const char* q = std::strchr(p, ',');
    p = q ? q + 1 : p + std::strlen(p);
  }
  constexpr int kQsaLayers = kLayers / kQsaPeriod, kGdnLayers = kLayers - kQsaLayers;
  auto shapes = [&](int K, int Ki) {
    return std::vector<Shape>{
        {"gdn_in", {{kHidden, kGdnQkvWidth, K}, {kHidden, kGdnValueWidth, K}}, kGdnLayers},
        {"out", {{kMixerOutWidth, kHidden, K}}, kLayers},
        {"qsa_in", {{kHidden, kQsaQWidth, K}, {kHidden, kQsaKvWidth, K}, {kHidden, kQsaKvWidth, K},
                    {kHidden, kQsaIdxWidth, Ki}}, kQsaLayers},
    };
  };
  const std::vector<Shape> s6 = shapes(6, 4), s8 = shapes(8, 8);
  const bool check = mode == "check" || mode == "all", time = mode == "time" || mode == "all";
  std::printf("p418_dense reps=%d mode=%s CUs=%d\n", reps, mode.c_str(), bench::NumCUs());
  if (check)
    for (const Shape& sh : s8)
      for (int grid : grids) Check(sh, grid);
  if (time) {
    bench::Sampler sampler;
    sampler.Start();
    std::vector<double> pass(kForms * grids.size(), 0.0), gbs(kForms, 0.0);
    double bytes6 = 0, bytes8 = 0;
    for (int w : {6, 8}) {
      for (const Shape& sh : w == 6 ? s6 : s8) {
        size_t bytes = 0;
        for (const Mat& m : sh.mats) bytes += (size_t)(m.I / 16) * (m.O / 16) * 32 * m.K;
        const int copies = (int)std::max<size_t>(1, (384u << 20) / bytes + 1);
        Group g = Make(sh, copies, 11, false);
        (w == 6 ? bytes6 : bytes8) += (double)g.tile_bytes * sh.per_pass;
        std::printf("shape %-7s K%d %.2f MB items %zu maxspan %d splits", sh.name.c_str(), w, g.tile_bytes / 1e6,
                    g.items.size(), g.maxspan);
        for (const LinJob& j : g.jobs) std::printf(" %d", j.S);
        std::printf("\n");
        const Form fs6[] = {kRing6F, kRing6E}, fs8[] = {kPol8E, kPol8F, kRing8E, kRing8F};
        const Form* fs = w == 6 ? fs6 : fs8;
        const int nf = w == 6 ? 2 : 4;
        for (int i = 0; i < nf; ++i)
          for (size_t gi = 0; gi < grids.size(); ++gi) {
            const bool ring = fs[i] != kPol8E && fs[i] != kPol8F;
            if (!ring && gi > 0) {
              pass[fs[i] * grids.size() + gi] = pass[fs[i] * grids.size()];
              continue;
            }
            const double ms = Time(fs[i], g, grids[gi], reps);
            pass[fs[i] * grids.size() + gi] += ms * sh.per_pass;
            std::printf("time %-7s %-6s grid %3d  %.4f ms  %.1f GB/s\n", sh.name.c_str(), kNames[fs[i]],
                        ring ? grids[gi] : (int)g.items.size(), ms, g.tile_bytes / 1e6 / ms);
          }
      }
    }
    for (int f = 0; f < kForms; ++f)
      for (size_t gi = 0; gi < grids.size(); ++gi) {
        const double b = f < kPol8E ? bytes6 : bytes8, ms = pass[f * grids.size() + gi];
        std::printf("pass %-6s grid %3d  %.3f ms  %.2f GB  %.1f GB/s\n", kNames[f], grids[gi], ms, b / 1e9,
                    b / 1e6 / ms);
      }
    sampler.Stop();
    std::printf("sclk_mean_mhz %.0f\n", sampler.MeanMhz());
  }
  std::printf("fails %d\n", fails);
  return fails ? 1 : 0;
}
