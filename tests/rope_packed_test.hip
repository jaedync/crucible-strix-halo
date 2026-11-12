// Y2 kernel contract: one packed launch may hold plain and YaRN spans.
// Each span reads its own table through PackedQsaArgs rows[s].rope, so the packed QsaPrepKernel and QsaPoolKernel
// must give, bit for bit, the bytes of one launch per span. Span 0 is plain, span 1 yarn:2, span 2 yarn:4.
// Build: hipcc -O3 -std=c++20 --offload-arch=gfx1151 -o build/tests/rope_packed_test tests/rope_packed_test.hip
#include "../bench/prefill_common.h"
#include "../engine/layers/ops.h"
#include <cstring>
#include <memory>

using namespace crucible;
using namespace prefill_bench;

namespace {
constexpr int kCtx = 4096;
template <class T> bool Same(const std::vector<T>& x, const std::vector<T>& y) {
  return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}

// One span's inputs and cache, filled from a seed; two copies of the same span give the same bytes.
struct Span {
  std::unique_ptr<Buffer<float>> q, k, v, idx, raw, pool;
  std::unique_ptr<Buffer<__half>> ck, cv;
  QsaArgs a{};
  Span(int pos0, int R, unsigned seed, const float* rope, const Buffer<float>& n256, const Buffer<float>& n128) {
    q = std::make_unique<Buffer<float>>(Random((size_t)R * 24 * 512, 2.f, seed));
    k = std::make_unique<Buffer<float>>(Random((size_t)R * 512, 1.f, seed + 1));
    v = std::make_unique<Buffer<float>>(Random((size_t)R * 512, 1.f, seed + 2));
    idx = std::make_unique<Buffer<float>>(Random((size_t)R * 640, 1.f, seed + 3));
    raw = std::make_unique<Buffer<float>>(Random((size_t)kCtx * 128, .5f, seed + 4));
    pool = std::make_unique<Buffer<float>>(std::vector<float>(QsaPoolFloats(kCtx), 0.f));
    ck = std::make_unique<Buffer<__half>>(Half(std::vector<float>((size_t)kCtx * 512, 0.f)));
    cv = std::make_unique<Buffer<__half>>(Half(std::vector<float>((size_t)kCtx * 512, 0.f)));
    a.q = q->p; a.k = k->p; a.v = v->p; a.idx = idx->p;
    a.q_norm1 = n256.p; a.k_norm1 = n256.p; a.iq_norm1 = n128.p; a.ik_norm1 = n128.p;
    a.c = QsaCache{ck->p, cv->p, raw->p, pool->p, 0};
    a.pos0 = pos0; a.R = R; a.rope = rope;
  }
  std::vector<std::vector<unsigned char>> Bytes() const {
    auto bytes = [](const auto& b) {
      const auto h = b->Get();
      std::vector<unsigned char> out(h.size() * sizeof(h[0]));
      std::memcpy(out.data(), h.data(), out.size());
      return out;
    };
    return {bytes(q), bytes(idx), bytes(ck), bytes(cv), bytes(raw), bytes(pool)};
  }
};
}  // namespace

int main() {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) return 2;
  float* tables = nullptr;
  HIP_OK(hipMalloc(&tables, kRopeTableModes * kRopeTableFloats * sizeof(float)));
  FillRopeTables(tables);
  std::vector<float> w256 = Random(256, .2f, 7), w128 = Random(128, .2f, 8);
  for (auto& w : w256) w += 1.f;
  for (auto& w : w128) w += 1.f;
  Buffer<float> n256(w256), n128(w128);
  // Spans of 1, 3 and 4 rows that end on a pooled block (pos0 + R multiple of 4) at different positions.
  const int pos0[3] = {3, 1001, 4088}, rows[3] = {1, 3, 4};
  int failures = 0;
  for (int order = 0; order < 2; ++order) {  // order 1: the modes swap places, so a span index cannot hide a bug
    std::vector<std::unique_ptr<Span>> alone, packed;
    PackedQsaArgs p{};
    int base = 0;
    for (int s = 0; s < 3; ++s) {
      const int mode = order ? 2 - s : s;
      const float* rope = tables + kRopeTableFloats * mode;
      alone.push_back(std::make_unique<Span>(pos0[s], rows[s], 100 + 10 * s, rope, n256, n128));
      packed.push_back(std::make_unique<Span>(pos0[s], rows[s], 100 + 10 * s, rope, n256, n128));
      p.rows[s] = packed.back()->a;
      p.base[s] = base;
      base += rows[s];
    }
    for (const auto& s : alone) {
      QsaPrepKernel<<<dim3(33, s->a.R), 256>>>(s->a);
      QsaPoolKernel<<<1, 256>>>(s->a);
    }
    QsaPrepKernel<<<dim3(33, 4, 3), 256>>>(p);
    QsaPoolKernel<<<dim3(1, 1, 3), 256>>>(p);
    CheckLaunch();
    for (int s = 0; s < 3; ++s) {
      const bool same = alone[s]->Bytes() == packed[s]->Bytes();
      std::printf("%s: order %d span %d mode %d packed = alone\n", same ? "PASS" : "FAIL", order, s, order ? 2 - s : s);
      failures += !same;
    }
    // The modes must matter: the YaRN span differs from the same span in plain mode.
    if (order == 0) {
      Span plain(pos0[1], rows[1], 110, tables, n256, n128);
      QsaPrepKernel<<<dim3(33, plain.a.R), 256>>>(plain.a);
      CheckLaunch();
      const bool differs = plain.Bytes()[0] != alone[1]->Bytes()[0];
      std::printf("%s: a yarn:2 span differs from plain\n", differs ? "PASS" : "FAIL");
      failures += !differs;
    }
  }
  std::printf("%s: rope_packed_test\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
