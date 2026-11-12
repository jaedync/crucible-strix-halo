// GPU test of the low-bit draft head (engine/runtime/draft_quant.h): on random K6 and K5 EXL3 heads, the rescored
// blocks are the 8 best coarse blocks, their logits equal the EXL3 logits of the full head, the coarse logits stay
// close, and the choice follows DraftChoiceKernel's top-2 rule over the rescored logits. Then the time of one
// scored row on a 64K head (q4, q2, and the EXL3 block head GEMV for scale). Run it under a GPU lease.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "../engine/model/flashnext/model.h"
#include "../engine/runtime/draft_adapt.h"
#include "../engine/runtime/draft_ops.h"
#include "../engine/runtime/draft_quant.h"

using namespace crucible;

template <typename T> T* Put(const std::vector<T>& v) {
  T* p;
  HIP_OK(hipMalloc(&p, v.size() * sizeof(T)));
  HIP_OK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}

template <typename T> std::vector<T> Get(const T* p, size_t n) {
  std::vector<T> v(n);
  HIP_OK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
  return v;
}

constexpr int I = 2560;

// The EXL3 head of `rows` input rows: LinIn, LinGemv (S = 1), LinOut. Returns logits [rows][O]; xh gets the
// transformed input ([rows][I], the same for every head with this suh).
struct Exl3 {
  LinJob j{};
  LinJob* job;
  GemvItem* items;
  XfItem *in, *out;
  int nitems, nin, nout;
  Exl3(const uint32_t* tiles, const __half* suh, const __half* svh, const float* src, int O, int K, int rows) {
    j.tiles = tiles; j.suh = suh; j.svh = svh; j.src = src;
    HIP_OK(hipMalloc(&j.xh, (size_t)rows * I * sizeof(__half)));
    HIP_OK(hipMalloc(&j.partial, (size_t)rows * O * sizeof(float)));
    HIP_OK(hipMalloc(&j.dst, (size_t)rows * O * sizeof(float)));
    j.I = I; j.O = O; j.K = K; j.S = 1; j.R = rows;
    j.lds = I; j.ldd = O; j.epi = kEpiStore;
    job = Put(std::vector<LinJob>{j});
    std::vector<GemvItem> it;
    std::vector<XfItem> vi, vo;
    for (int b = 0; b < O / 16; b += 8) it.push_back({0, 0, b, 8});
    for (int r = 0; r < rows; ++r) {
      for (int b = 0; b < I / 128; ++b) vi.push_back({0, r, b});
      for (int b = 0; b < O / 128; ++b) vo.push_back({0, r, b});
    }
    items = Put(it); in = Put(vi); out = Put(vo);
    nitems = it.size(); nin = vi.size(); nout = vo.size();
  }
  void Run(int rows) const {
    LinInKernel<<<(nin + 7) / 8, 256>>>(job, in, nin);
    if (rows == 1) LinGemvKernel<1><<<nitems, 256, (I / 16) * 32>>>(job, items, nitems);
    else LinGemvKernel<2><<<nitems, 256, 2 * (I / 16) * 32>>>(job, items, nitems);
    LinOutKernel<<<(nout + 7) / 8, 256>>>(job, out, nout);
  }
};

static void Check(int K, int bits) {
  constexpr int O = 40 * 128, rows = 2;
  const size_t words = (size_t)128 * I * K / 32;
  std::mt19937 rng(1000 * K + bits);
  std::vector<uint32_t> tiles(words * (O / 128));
  for (auto& x : tiles) x = rng();
  std::vector<__half> svh(O), suh(I);
  for (auto& x : svh) x = __float2half(0.5f + (rng() % 100) / 100.f);
  for (auto& x : suh) x = __float2half((rng() % 2 ? 1.f : -1.f) * 0.01f);
  std::vector<float> input((size_t)rows * I);
  for (auto& x : input) x = (int(rng() % 200) - 100) / 100.f;
  auto* dt = Put(tiles);
  auto* dsuh = Put(suh);
  auto* dsvh = Put(svh);
  auto* dsrc = Put(input);
  const Exl3 full(dt, dsuh, dsvh, dsrc, O, K, rows);
  full.Run(rows);
  const auto exact = Get(full.j.dst, (size_t)rows * O);
  // A selection of 30 of the 40 blocks, not in order of the vocabulary.
  std::vector<int> blocks(40);
  std::iota(blocks.begin(), blocks.end(), 0);
  std::shuffle(blocks.begin(), blocks.end(), rng);
  blocks.resize(30);
  std::sort(blocks.begin(), blocks.end());
  const Lin source{I, O, K, dt, dsuh, dsvh};
  BlockDraftHead block_head(source, blocks, false);
  DraftQuantHead q(source, block_head.Weights(), blocks, bits);
  const int V = (int)blocks.size() * 128;
  assert(q.Vocabulary() == V);
  float* logits;
  HIP_OK(hipMalloc(&logits, (size_t)rows * V * sizeof(float)));
  auto* choice = Put(std::vector<DraftChoice>(rows));
  q.Score(full.j.xh, rows, logits, block_head.Mapping(), choice);
  HIP_OK(hipDeviceSynchronize());
  const auto got = Get(logits, (size_t)rows * V);
  const auto chosen = Get(q.Chosen(), (size_t)rows * kDraftQBlocks);
  const auto maxima = Get(q.Maxima(), (size_t)rows * blocks.size());
  const auto picks = Get(choice, rows);
  double coarse_err = 0, coarse_norm = 0, fine_err = 0, fine_norm = 0;
  for (int r = 0; r < rows; ++r) {
    const int nb = (int)blocks.size();
    // The rescored blocks are the 8 best by coarse maximum (ties: the lower head block).
    std::vector<bool> rescored(nb, false);
    for (int slot = 0; slot < kDraftQBlocks; ++slot) {
      const int b = chosen[r * kDraftQBlocks + slot];
      assert(b >= 0 && b < nb && !rescored[b]);
      assert(DraftQRank(maxima.data() + r * nb, nb, b, nb) == slot);
      rescored[b] = true;
    }
    float best = -INFINITY, second = -INFINITY;
    int id = 0;
    for (int b = 0; b < nb; ++b) {
      float coarse_max = -INFINITY;
      for (int c = 0; c < 128; ++c) {
        const float e = exact[(size_t)r * O + blocks[b] * 128 + c], v = got[(size_t)r * V + b * 128 + c];
        if (rescored[b]) {
          fine_err += double(v - e) * (v - e), fine_norm += double(e) * e;
          const int token = blocks[b] * 128 + c;
          if (v > best || (v == best && token < id)) second = best, best = v, id = token;
          else second = std::max(second, v);
        } else {
          coarse_err += double(v - e) * (v - e), coarse_norm += double(e) * e;
          coarse_max = std::max(coarse_max, v);
        }
      }
      if (!rescored[b]) assert(coarse_max == maxima[r * nb + b]);
    }
    assert(picks[r].token == id && picks[r].margin == best - second);
  }
  const double coarse = std::sqrt(coarse_err / coarse_norm), fine = std::sqrt(fine_err / fine_norm);
  std::printf("K%d q%d: rescored relative error %.2e, coarse relative error %.4f\n", K, bits, fine, coarse);
  assert(fine < 1e-5);
  assert(coarse < (bits == 4 ? 0.15 : 0.5));
  HIP_OK(hipFree(logits));
}

// One scored row on a 64K K6 head: q4, q2 and the EXL3 block head GEMV (LinGemvKernel<1>, S = 1; the engine's
// tuned LinRingFlowKernel read 126 MB in 0.547 ms in the D1 trace).
static void Time() {
  constexpr int K = 6, O = 512 * 128;
  const size_t words = (size_t)128 * I * K / 32;
  std::mt19937 rng(5);
  std::vector<uint32_t> tiles(words * (O / 128));
  for (auto& x : tiles) x = rng();
  auto* dt = Put(tiles);
  tiles.clear();
  tiles.shrink_to_fit();
  auto* dsuh = Put(std::vector<__half>(I, __float2half(0.01f)));
  auto* dsvh = Put(std::vector<__half>(O, __float2half(1.f)));
  std::vector<float> input(I);
  for (auto& x : input) x = (int(rng() % 200) - 100) / 100.f;
  auto* dsrc = Put(input);
  const Exl3 full(dt, dsuh, dsvh, dsrc, O, K, 1);
  std::vector<int> blocks(512);
  std::iota(blocks.begin(), blocks.end(), 0);
  const Lin source{I, O, K, dt, dsuh, dsvh};
  // The block head copies the tiles; the source is the same layout, so the rescoring reads the source.
  float* logits;
  HIP_OK(hipMalloc(&logits, (size_t)4 * O * sizeof(float)));
  auto* choice = Put(std::vector<DraftChoice>(4));
  __half* xh4;  // 4 rows for the 4-row time (row 0 of the transformed input, 4 times)
  HIP_OK(hipMalloc(&xh4, (size_t)4 * I * sizeof(__half)));
  auto* ids = Put(blocks);
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  auto time = [&](auto&& call) {
    for (int i = 0; i < 20; ++i) call();
    HIP_OK(hipEventRecord(a));
    constexpr int n = 200;
    for (int i = 0; i < n; ++i) call();
    HIP_OK(hipEventRecord(b));
    HIP_OK(hipEventSynchronize(b));
    float ms;
    HIP_OK(hipEventElapsedTime(&ms, a, b));
    return 1000.f * ms / n;
  };
  const float exl3 = time([&] { full.Run(1); });
  for (int r = 0; r < 4; ++r)
    HIP_OK(hipMemcpy(xh4 + (size_t)r * I, full.j.xh, I * sizeof(__half), hipMemcpyDeviceToDevice));
  for (int bits : {4, 2}) {
    DraftQuantHead q(source, source, blocks, bits);
    const float us = time([&] { q.Score(full.j.xh, 1, logits, ids, choice); });
    const float coarse = time([&] { q.Coarse(full.j.xh, 1, logits); });
    const float rescore = time([&] { q.Rescore(full.j.xh, 1); });
    const float choose = time([&] { q.Choose(1, logits, ids, choice); });
    const float four = time([&] { q.Score(xh4, 4, logits, ids, choice); });
    std::printf("64K head, one row: q%d %.1f us (coarse pass %.1f us, %.1f MB copy, %.0f GB/s; rescore %.1f us; "
                "choice %.1f us), 4 rows %.1f us; EXL3 in+GEMV+out S=1 %.1f us (%.1f MB)\n", bits, us, coarse,
                q.Bytes() / 1e6, q.Bytes() / coarse / 1e3, rescore, choose, four, exl3, words * (O / 128) * 4 / 1e6);
  }
}

int main() {
  for (int K : {6, 5})
    for (int bits : {4, 2}) Check(K, bits);
  std::puts("PASS: rescored blocks are the 8 best coarse blocks, exact EXL3 logits, top-2 rule");
  Time();
}
