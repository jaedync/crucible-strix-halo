// Y1 rope table: RopeAt with the plain table must give the bits of the old
// RoPE, which computed each inverse frequency with powf in the kernel. OldRope64 and OldRope64Axes are the old
// code of engine/layers/ops.h, copied verbatim. The YaRN tables must hold the header bits.
// Build: hipcc -O3 -std=c++20 --offload-arch=gfx1151 -o build/tests/rope_table_test tests/rope_table_test.hip
#include "../engine/layers/ops.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using namespace crucible;

namespace {
__device__ __forceinline__ void OldRope64(float* x, int p) {
  const int j = threadIdx.x;
  float c = 0.f, s = 0.f, a = 0.f, b = 0.f;
  if (j < kRotDim / 2) {
    const float inv = powf(kRopeTheta, -2.f * j / kRotDim);
    const float ang = (float)p * inv;
    c = cosf(ang);
    s = sinf(ang);
    a = x[j];
    b = x[j + 32];
  }
  __syncthreads();
  if (j < kRotDim / 2) {
    x[j] = a * c - b * s;
    x[j + 32] = b * c + a * s;
  }
  __syncthreads();
}

__device__ __forceinline__ void OldRope64Axes(float* x, int pt, int ph, int pw) {
  const int j = threadIdx.x;
  float c = 0.f, s = 0.f, a = 0.f, b = 0.f;
  if (j < kRotDim / 2) {
    const int p = j % 3 == 1 && j < 33 ? ph : j % 3 == 2 && j < 30 ? pw : pt;
    const float inv = powf(kRopeTheta, -2.f * j / kRotDim);
    const float ang = (float)p * inv;
    c = cosf(ang);
    s = sinf(ang);
    a = x[j];
    b = x[j + 32];
  }
  __syncthreads();
  if (j < kRotDim / 2) {
    x[j] = a * c - b * s;
    x[j + 32] = b * c + a * s;
  }
  __syncthreads();
}

__device__ __forceinline__ void OldRopeAt(float* x, const int* mrope, int p) {
  if (!mrope) OldRope64(x, p);
  else OldRope64Axes(x, mrope[4 * p], mrope[4 * p + 1], mrope[4 * p + 2]);
}

// One block per test row: x[256] rotated at position pos[i] (and the mRoPE row of pos[i] when mrope is set).
template <bool Old>
__global__ void __launch_bounds__(256) RopeRows(const float* in, float* out, const int* pos, const int* mrope,
                                                const float* rope) {
  __shared__ float x[256];
  const int i = blockIdx.x, j = threadIdx.x;
  x[j] = in[(size_t)i * 256 + j];
  __syncthreads();
  if constexpr (Old) OldRopeAt(x, mrope ? mrope + 4 * (size_t)i - 4 * (size_t)pos[i] : nullptr, pos[i]);
  else RopeAt(x, rope, mrope ? mrope + 4 * (size_t)i - 4 * (size_t)pos[i] : nullptr, pos[i]);
  out[(size_t)i * 256 + j] = x[j];
}

__global__ void OldInverse(float* inv) {
  const int j = threadIdx.x;
  if (j < kRotDim / 2) inv[j] = powf(kRopeTheta, -2.f * j / kRotDim);
}

template <class T> std::vector<T> Get(const T* p, size_t n) {
  std::vector<T> h(n);
  HIP_OK(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}
template <class T> T* Put(const std::vector<T>& h) {
  T* p = nullptr;
  HIP_OK(hipMalloc(&p, h.size() * sizeof(T)));
  HIP_OK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return p;
}

int failures = 0;
void Expect(bool ok, const char* what) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
  failures += !ok;
}

void CompareRows(const std::vector<int>& positions, bool images, const char* what) {
  const size_t n = positions.size();
  std::mt19937 rng(1234 + (unsigned)n + images);
  std::uniform_real_distribution<float> u(-3.f, 3.f);
  std::vector<float> x(n * 256);
  for (auto& v : x) v = u(rng);
  std::vector<int> axes;
  if (images) {
    std::uniform_int_distribution<int> q(0, 1 << 20);
    for (size_t i = 0; i < n; ++i) axes.insert(axes.end(), {q(rng), q(rng), q(rng), 0});
  }
  float* in = Put(x);
  int* pos = Put(positions);
  int* mrope = images ? Put(axes) : nullptr;
  float *a = nullptr, *b = nullptr;
  HIP_OK(hipMalloc(&a, x.size() * 4));
  HIP_OK(hipMalloc(&b, x.size() * 4));
  RopeRows<true><<<n, 256>>>(in, a, pos, mrope, nullptr);
  RopeRows<false><<<n, 256>>>(in, b, pos, mrope, PlainRopeTable());
  HIP_OK(hipDeviceSynchronize());
  const auto ha = Get(a, x.size()), hb = Get(b, x.size());
  Expect(std::memcmp(ha.data(), hb.data(), ha.size() * 4) == 0, what);
  for (float* p : {in, a, b}) HIP_OK(hipFree(p));
  HIP_OK(hipFree(pos));
  if (mrope) HIP_OK(hipFree(mrope));
}
// Review L4: YaRN rows against a host reference. The angle is the float32 product p * inv[j], as in the kernel and
// in Hugging Face (freqs in float32); cos and sin of that angle are taken in double and scaled by m. The check
// bounds the device cosf/sinf error at large angles. A float64 angle would differ by up to half an ulp of the
// float32 angle (about 0.03 rad at 524,287 for j = 0), which HF has too.
void CheckYarn(const float* tables, const std::vector<float>& host_tables, int mode, const char* what) {
  const std::vector<int> positions = {0, 1, 4095, 262143, 262144, 400000, 524287, 1048575};
  const size_t n = positions.size();
  std::mt19937 rng(99 + mode);
  std::uniform_real_distribution<float> u(-3.f, 3.f);
  std::vector<float> x(n * 256);
  for (auto& v : x) v = u(rng);
  float* in = Put(x);
  int* pos = Put(positions);
  float* out = nullptr;
  HIP_OK(hipMalloc(&out, x.size() * 4));
  RopeRows<false><<<n, 256>>>(in, out, pos, nullptr, tables + kRopeTableFloats * mode);
  HIP_OK(hipDeviceSynchronize());
  const auto got = Get(out, x.size());
  const float* t = host_tables.data() + kRopeTableFloats * mode;
  double worst = 0;
  bool rest = true;
  for (size_t i = 0; i < n; ++i) {
    const float* xi = x.data() + i * 256;
    const float* gi = got.data() + i * 256;
    for (int j = 0; j < 32; ++j) {
      const float ang = (float)positions[i] * t[j];
      const double c = std::cos((double)ang) * t[32], sn = std::sin((double)ang) * t[32];
      const double r0 = xi[j] * c - xi[j + 32] * sn, r1 = xi[j + 32] * c + xi[j] * sn;
      worst = std::max({worst, std::abs(gi[j] - r0), std::abs(gi[j + 32] - r1)});
    }
    for (int j = 64; j < 256; ++j) rest = rest && std::memcmp(&gi[j], &xi[j], 4) == 0;
  }
  std::printf("INFO: %s worst abs error %.3g (|x| <= 3)\n", what, worst);
  Expect(worst < 1e-5 && rest, what);
  for (float* p : {in, out}) HIP_OK(hipFree(p));
  HIP_OK(hipFree(pos));
}
}  // namespace

int main() {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) return 2;
  // The plain table: the old inverse frequencies bit for bit, multiplier 1, zero tail.
  float* old_inv = nullptr;
  HIP_OK(hipMalloc(&old_inv, 32 * 4));
  OldInverse<<<1, 64>>>(old_inv);
  HIP_OK(hipDeviceSynchronize());
  const auto inv = Get(old_inv, 32);
  const auto plain = Get(PlainRopeTable(), kRopeTableFloats);
  Expect(std::memcmp(inv.data(), plain.data(), 32 * 4) == 0, "plain table = old powf inverse frequencies");
  bool tail = plain[32] == 1.f;
  for (int i = 33; i < kRopeTableFloats; ++i) tail = tail && plain[i] == 0.f;
  Expect(tail, "plain multiplier 1, zero tail");

  // The engine tables: row 0 plain, rows 1 and 2 the header bits.
  float* tables = nullptr;
  HIP_OK(hipMalloc(&tables, kRopeTableModes * kRopeTableFloats * 4));
  FillRopeTables(tables);
  HIP_OK(hipDeviceSynchronize());
  const auto all = Get(tables, kRopeTableModes * kRopeTableFloats);
  Expect(std::memcmp(all.data(), plain.data(), kRopeTableFloats * 4) == 0, "engine table 0 = plain");
  bool yarn = true;
  for (int m = 1; m < kRopeTableModes; ++m)
    for (int i = 0; i < kRopeTableFloats; ++i) {
      uint32_t bits;
      std::memcpy(&bits, &all[m * kRopeTableFloats + i], 4);
      uint32_t want = 0;
      if (i < 33) std::memcpy(&want, &kYarnTable[m - 1][i], 4);
      yarn = yarn && bits == want;
    }
  Expect(yarn, "engine tables 1, 2 = rope_tables.h bits");
  CheckYarn(tables, all, 0, "plain rows vs host reference to 1048575");
  CheckYarn(tables, all, 1, "yarn:2 rows vs host reference to 1048575");
  CheckYarn(tables, all, 2, "yarn:4 rows vs host reference to 1048575");

  // Rotation: every position below 8192, then a spread up to 1,048,575, text and mRoPE.
  std::vector<int> low(8192), spread;
  for (int i = 0; i < 8192; ++i) low[i] = i;
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> any(0, 1048575);
  for (int i = 0; i < 8192; ++i) spread.push_back(any(rng));
  for (int p : {262143, 262144, 262145, 524287, 524288, 1048575}) spread.push_back(p);
  CompareRows(low, false, "RopeAt plain = old RoPE, positions 0..8191");
  CompareRows(spread, false, "RopeAt plain = old RoPE, 8198 positions to 1048575");
  CompareRows(low, true, "RopeAt plain = old RoPE, mRoPE axes");
  std::printf("%s: rope_table_test\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
