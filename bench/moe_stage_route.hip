// MoeStageRouteRow (moe_stage.h) against MoeRouteKernel (ops.h): sel and the
// eleven weights bitwise, on random router rows.
//
//   moe_stage_route [rows]
//
// Row families: normal logits; logits on a coarse grid (many ties, also
// across the tenth place); every logit equal; signed zeros; +-inf mixed in.
// Rows with NaN are out of scope (both paths give non-finite outputs).
// Prints one line per family and exits 1 on any difference.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../engine/model/flashnext/model.h"
#include "../engine/layers/ops.h"
#include "../engine/layers/moe_stage.h"

using namespace crucible;

namespace {

__global__ void __launch_bounds__(256) StageRouteKernel(const float* logits, int* sel, float* wts) {
  __shared__ MoeRouteState<1> st;
  const int r = blockIdx.x;
  MoeStageRoute<1>(logits + (size_t)r * (kExperts + 1), st);
  if (threadIdx.x < kTopK) sel[r * kTopK + threadIdx.x] = st.sel[threadIdx.x];
  if (threadIdx.x < kSlots) wts[r * kSlots + threadIdx.x] = st.wts[threadIdx.x];
}

std::vector<float> Rows(int family, int rows, std::mt19937& rng) {
  std::vector<float> v((size_t)rows * (kExperts + 1));
  std::normal_distribution<float> n(0.f, 2.f);
  std::uniform_int_distribution<int> grid(-6, 6), pick(0, kExperts - 1);
  for (int r = 0; r < rows; ++r) {
    float* x = v.data() + (size_t)r * (kExperts + 1);
    for (int e = 0; e <= kExperts; ++e) {
      switch (family) {
        case 0: x[e] = n(rng); break;
        case 1: x[e] = 0.5f * (float)grid(rng); break;
        case 2: x[e] = 1.25f; break;
        case 3: x[e] = (rng() & 1) ? 0.f : -0.f; break;
        default: x[e] = n(rng); break;
      }
    }
    if (family == 3)
      for (int k = 0; k < 4; ++k) x[pick(rng)] = 0.5f * (float)grid(rng);
    if (family == 4) {
      for (int k = 0; k < 3; ++k) x[pick(rng)] = INFINITY;
      for (int k = 0; k < 3; ++k) x[pick(rng)] = -INFINITY;
    }
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::stoi(argv[1]) : 20000;
  const char* names[] = {"normal", "grid ties", "all equal", "signed zeros", "infinities"};
  std::mt19937 rng(1234);
  float *dl, *wa, *wb;
  int *sa, *sb;
  HIP_OK(hipMalloc(&dl, sizeof(float) * rows * (kExperts + 1)));
  HIP_OK(hipMalloc(&wa, sizeof(float) * rows * kSlots));
  HIP_OK(hipMalloc(&wb, sizeof(float) * rows * kSlots));
  HIP_OK(hipMalloc(&sa, sizeof(int) * rows * kTopK));
  HIP_OK(hipMalloc(&sb, sizeof(int) * rows * kTopK));
  bool all = true;
  for (int f = 0; f < 5; ++f) {
    const std::vector<float> h = Rows(f, rows, rng);
    HIP_OK(hipMemcpy(dl, h.data(), sizeof(float) * h.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(sa, 0xff, sizeof(int) * rows * kTopK));
    HIP_OK(hipMemset(sb, 0xff, sizeof(int) * rows * kTopK));
    MoeRouteKernel<<<rows, 512>>>(dl, sa, wa);
    StageRouteKernel<<<rows, 256>>>(dl, sb, wb);
    HIP_OK(hipDeviceSynchronize());
    std::vector<int> ha(rows * kTopK), hb(rows * kTopK);
    std::vector<float> fa(rows * kSlots), fb(rows * kSlots);
    HIP_OK(hipMemcpy(ha.data(), sa, sizeof(int) * ha.size(), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(hb.data(), sb, sizeof(int) * hb.size(), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(fa.data(), wa, sizeof(float) * fa.size(), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(fb.data(), wb, sizeof(float) * fb.size(), hipMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < rows; ++r) {
      const bool s = std::memcmp(&ha[r * kTopK], &hb[r * kTopK], sizeof(int) * kTopK) == 0;
      const bool w = std::memcmp(&fa[r * kSlots], &fb[r * kSlots], sizeof(float) * kSlots) == 0;
      if (!s || !w) {
        if (bad == 0) std::printf("  first bad row %d (sel %s, wts %s)\n", r, s ? "equal" : "DIFF", w ? "equal" : "DIFF");
        ++bad;
      }
    }
    std::printf("%-13s rows %d differ %d\n", names[f], rows, bad);
    all &= bad == 0;
  }
  std::printf("%s\n", all ? "BITWISE EQUAL" : "DIFFERENT");
  return all ? 0 : 1;
}
