// AblateWriteKernel must equal the CPU reference bit for bit, for every row
// count and row stride, so that ablated runs keep row-count exactness.
#include "../engine/layers/ablation_device.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace crucible;

namespace {

std::vector<float> Values(size_t n, float scale, unsigned seed) {
  std::vector<float> v(n);
  uint32_t x = seed * 2654435761u + 1;
  for (float& f : v) {
    x = x * 1664525u + 1013904223u;
    f = scale * (float(x >> 8) / float(1u << 24) - 0.5f);
  }
  return v;
}

template <class T> T* Device(const std::vector<T>& h) {
  T* d;
  HIP_OK(hipMalloc(&d, h.size() * sizeof(T)));
  HIP_OK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

// stock: bit r marks a stock row (CRUCIBLE_ABLATE_MODE=request); it must equal AblateWriteStockRef.
void Check(int R, int ldh, uint32_t stock = 0) {
  std::vector<float> a = Values(kHidden, 1.f, 1);
  double n = 0;
  for (float v : a) n += double(v) * v;
  for (float& v : a) v = float(v / std::sqrt(n));
  std::vector<float> ab(a);
  for (float v : a) ab.push_back(2.5f * v);
  const std::vector<float> y = Values((size_t)R * kHidden, 4.f, 2 + R);
  const std::vector<float> post = Values((size_t)R * kStreams, 3.f, 3 + R);
  std::vector<float> H = Values((size_t)R * ldh, 50.f, 4 + R);
  std::vector<float> want(H);
  for (int r = 0; r < R; ++r) {
    if (r < 32 && (stock >> r & 1u))
      AblateWriteStockRef(y.data() + (size_t)r * kHidden, post.data() + r * kStreams, want.data() + (size_t)r * ldh);
    else
      AblateWriteRef(y.data() + (size_t)r * kHidden, ab.data(), ab.data() + kHidden, post.data() + r * kStreams,
                     want.data() + (size_t)r * ldh);
  }
  float *dy = Device(y), *dab = Device(ab), *dpost = Device(post), *dH = Device(H);
  AblateWriteKernel<<<R, kAblateThreads>>>(dy, dab, dpost, dH, ldh, stock);
  HIP_OK(hipGetLastError());
  std::vector<float> got(H.size());
  HIP_OK(hipMemcpy(got.data(), dH, got.size() * sizeof(float), hipMemcpyDeviceToHost));
  if (std::memcmp(got.data(), want.data(), got.size() * sizeof(float))) {
    std::fprintf(stderr, "FAIL: R=%d ldh=%d stock=%x differs from the reference\n", R, ldh, stock);
    std::exit(1);
  }
  for (float* p : {dy, dab, dpost, dH}) HIP_OK(hipFree(p));
}

}  // namespace

int main() {
  for (int R : {1, 3, 8, 37}) {
    Check(R, kWide);
    Check(R, kWide + 64);  // padded prefill stride; the padding must stay unchanged
  }
  // Mixed passes of per-request mode: stock rows first, last, alternating, and all but one.
  for (int R : {2, 4, 8, 16}) {
    Check(R, kWide, 1u);
    Check(R, kWide, 1u << (R - 1));
    Check(R, kWide, 0x5555u & ((1u << R) - 1));
    Check(R, kWide, ((1u << R) - 1) & ~2u);
  }
  std::puts("ablation_kernel_test: ok");
}
