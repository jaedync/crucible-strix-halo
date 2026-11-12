// Setup shared by bench/verify_moe.hip and bench/moe_sg_engine.hip: random EXL3 expert weights at the 4.05
// widths, Engine::Build (fixedS) and the MoE part of Engine::BuildLayer, and the route dump reader.
// Include it in one translation unit only (anonymous namespace).
#pragma once
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "../engine/model/flashnext/model.h"
#include "../engine/layers/ops.h"
#include "../engine/layers/moe_shared_first.h"
#include "../engine/exl3/plans.h"

using namespace crucible;

namespace {

constexpr int kKR = 4, kKS = 6;

__global__ void FillWords(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t z = (i + 1) * 0x9E3779B97F4A7C15ull + seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    p[i] = (uint32_t)(z ^ (z >> 31));
  }
}

__global__ void FillHalves(__half* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
    z ^= z >> 13;
    z *= 0x5bd1e995u;
    z ^= z >> 15;
    const float v = (0.5f + (z & 0xffff) / 65536.f) * scale;
    p[i] = __float2half((z >> 16) & 1 ? v : -v);
  }
}

__global__ void FillFloats(float* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2246822519u) ^ seed;
    z ^= z >> 15;
    z *= 0x2c1b3c6du;
    z ^= z >> 12;
    p[i] = ((z & 0xffffff) / 16777216.f - 0.5f) * scale;
  }
}

template <typename T>
T* Alloc(size_t n) {
  T* p;
  HIP_OK(hipMalloc(&p, n * sizeof(T)));
  return p;
}
uint32_t* Words(size_t n, uint32_t seed) {
  uint32_t* p = Alloc<uint32_t>(n);
  FillWords<<<1024, 256>>>(p, n, seed);
  return p;
}
__half* Halves(size_t n, uint32_t seed, float scale) {
  __half* p = Alloc<__half>(n);
  FillHalves<<<256, 256>>>(p, n, seed, scale);
  return p;
}
size_t TileWords(int I, int O, int K) { return (size_t)I * O / 256 * 8 * K; }
Lin MakeLin(int I, int O, int K, uint32_t seed) {
  Lin l;
  l.I = I, l.O = O, l.K = K, l.tiles = Words(TileWords(I, O, K), seed);
  l.suh = Halves(I, seed + 1, 1.f), l.svh = Halves(O, seed + 2, 0.02f);
  return l;
}

struct LayerW {
  ExpertPtrs* d_experts = nullptr;
  std::vector<ExpertPtrs> host;
  Lin sg, su, sd;
};

LinJob Job(const Lin& L, const float* src, int lds, int R) {
  LinJob j{};
  j.tiles = L.tiles, j.suh = L.suh, j.svh = L.svh, j.src = src;
  j.I = L.I, j.O = L.O, j.K = L.K, j.S = 1, j.R = R, j.lds = lds, j.epi = kEpiStore, j.cache_policy = 4;
  return j;
}

// Engine::Build with fixedS (decode policy cache 4).
void Build(Group& g, std::vector<LinJob> jobs, int R, float* partial, __half* xh, int ib, int shared_job) {
  size_t poff = 0, xoff = 0;
  int maxspan = 0;
  std::vector<GemvItem> items;
  for (size_t ji = 0; ji < jobs.size(); ++ji) {
    LinJob& j = jobs[ji];
    j.partial = partial + poff;
    poff += (size_t)j.S * j.R * j.O;
    j.xh = xh + xoff;
    xoff += (size_t)j.R * j.I;
    maxspan = std::max(maxspan, j.I / 16 / j.S);
    for (int s = 0; s < j.S; ++s)
      for (int b0 = 0; b0 < j.O / 16; b0 += ib) items.push_back({(int)ji, s, b0, std::min(ib, j.O / 16 - b0)});
  }
  g.jobs = jobs;
  g.cache = 4;
  g.d_jobs = Upload(jobs);
  g.d_items = Upload(shared_job < 0 ? items : MoeSharedFirstItems(items, shared_job));
  g.nitems = (int)items.size();
  g.R = R;
  g.lds = (size_t)R * maxspan * 32;
}

struct Plan {
  Group gu, dn;
};

// Random weights of nl layers: 512 routed experts (K4 tiles, 256-byte aligned as the loader's) and the K6 shared
// expert per layer.
std::vector<LayerW> MakeLayers(int nl) {
  const size_t gw = TileWords(kHidden, kMoeInter, kKR), dw = TileWords(kMoeInter, kHidden, kKR);
  std::vector<LayerW> layers(nl);
  for (int l = 0; l < nl; ++l) {
    uint32_t* g = Words(gw * kExperts, 100 * l + 1);
    uint32_t* u = Words(gw * kExperts, 100 * l + 2);
    uint32_t* d = Words(dw * kExperts, 100 * l + 3);
    __half* gs = Halves((size_t)kExperts * (kHidden + kMoeInter), 100 * l + 4, 1.f);
    __half* us = Halves((size_t)kExperts * (kHidden + kMoeInter), 100 * l + 5, 1.f);
    __half* ds = Halves((size_t)kExperts * (kHidden + kMoeInter), 100 * l + 6, 1.f);
    auto& h = layers[l].host;
    h.resize(kExperts);
    for (int e = 0; e < kExperts; ++e) {
      const size_t so = (size_t)e * (kHidden + kMoeInter);
      h[e].tiles[0] = g + gw * e, h[e].tiles[1] = u + gw * e, h[e].tiles[2] = d + dw * e;
      h[e].suh[0] = gs + so, h[e].svh[0] = gs + so + kHidden;
      h[e].suh[1] = us + so, h[e].svh[1] = us + so + kHidden;
      h[e].suh[2] = ds + so, h[e].svh[2] = ds + so + kMoeInter;
      h[e].K[0] = h[e].K[1] = h[e].K[2] = kKR;
    }
    layers[l].d_experts = Upload(h);
    layers[l].sg = MakeLin(kHidden, kMoeInter, kKS, 100 * l + 7);
    layers[l].su = MakeLin(kHidden, kMoeInter, kKS, 100 * l + 10);
    layers[l].sd = MakeLin(kMoeInter, kHidden, kKS, 100 * l + 13);
  }
  return layers;
}

struct VmScratch {
  float *x, *partial, *H, *post, *wts;
  __half* xh;
  int *map, *mapj, *rowl;
  unsigned* done;
};

// The down group's xh at its own region 8 Mi halves up (verify_moe; the gate/up inputs of 16 rows take 6.6 Mi).
constexpr size_t kBenchDnXh = size_t(8) << 20;

// Engine::BuildLayer, MoE part, for a pass of R = 1..kPackedRows rows. dn_xh: halves from s.xh to the down
// group's xh (0: the gate/up base, as Engine::Build without an xh offset).
Plan MakePlan(const LayerW& L, int R, const VmScratch& s, int ib, size_t dn_xh) {
  const ExpertPtrs& e0 = L.host[0];
  Lin rg, ru, rd;
  rg.I = kHidden, rg.O = kMoeInter, rg.K = e0.K[0], rg.tiles = (uint32_t*)e0.tiles[0];
  rg.suh = (__half*)e0.suh[0], rg.svh = (__half*)e0.svh[0];
  ru = rg, ru.tiles = (uint32_t*)e0.tiles[1], ru.suh = (__half*)e0.suh[1], ru.svh = (__half*)e0.svh[1];
  rd.I = kMoeInter, rd.O = kHidden, rd.K = e0.K[2], rd.tiles = (uint32_t*)e0.tiles[2];
  rd.suh = (__half*)e0.suh[2], rd.svh = (__half*)e0.svh[2];
  std::vector<LinJob> gu, dn;
  const int U = R * kTopK, nr = std::min(R, kMaxRows);
  for (int u = 0; u < MoeExpertJobs(R); ++u) {
    const bool sh = u >= U;
    gu.push_back(Job(sh ? L.sg : rg, s.x, kHidden, nr));
    gu.push_back(Job(sh ? L.su : ru, s.x, kHidden, nr));
    dn.push_back(Job(sh ? L.sd : rd, nullptr, 0, nr));
  }
  Plan p;
  const bool shared_first = R > 1;
  Build(p.gu, gu, nr, s.partial, s.xh, ib, shared_first ? 2 * R * kTopK : -1);
  size_t gu_partial = 0;
  for (const LinJob& j : p.gu.jobs) gu_partial += (size_t)j.S * j.R * j.O;
  Build(p.dn, dn, nr, s.partial + gu_partial, s.xh + dn_xh, ib, shared_first ? R * kTopK : -1);
  return p;
}

// Routes: per decode token, 48 layers x 10 ids in router order.
std::vector<std::vector<int>> Tokens(const std::string& path) {
  std::vector<std::vector<int>> out;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::vector<int> cur(kLayers * kTopK, -1);
  int head[2];
  while (std::fread(head, sizeof(int), 2, f) == 2) {
    std::vector<int> ids((size_t)head[1] * kTopK);
    if (std::fread(ids.data(), sizeof(int), ids.size(), f) != ids.size()) break;
    if (head[1] != 1 || head[0] >= kLayers) continue;
    std::copy(ids.begin(), ids.end(), cur.begin() + head[0] * kTopK);
    if (head[0] == kLayers - 1) out.push_back(cur);
  }
  std::fclose(f);
  return out;
}

}  // namespace
