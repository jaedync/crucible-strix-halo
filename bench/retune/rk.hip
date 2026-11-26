// rocm10-retune kernel bench (Nov 23): the decode kernels that the retune changed, at the shapes of the 4.05 pack,
// on synthetic inputs (fixed seeds), outside the engine. Build one binary per source tree and compiler:
//
//   hipcc -O3 -std=c++20 --offload-arch=gfx1151 -I<tree>/engine bench/retune/rk.hip -o rk-<arm>
//   rk-<arm> [launches] [filter]   one line per case: case, median and p10 us per launch, FNV-1a hash of the output
//                                  (filter: name prefixes split by ';'; only those cases run)
//
// The weights cycle through copies of 128 MB or more in total, so each launch reads cold weights (as in a forward,
// where every layer has its own matrices). Two arms with the same hash on a case give bitwise equal outputs.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "layers/qsa_long.h"
#include "runtime/draft_quant.h"

using namespace crucible;

#define CK(x)                                                                         \
  do {                                                                                \
    hipError_t e_ = (x);                                                              \
    if (e_ != hipSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      std::exit(2);                                                                   \
    }                                                                                 \
  } while (0)

namespace {

__global__ void FillU32(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
    z ^= z >> 13;
    z *= 0x5bd1e995u;
    z ^= z >> 15;
    p[i] = z;
  }
}

// fp16 values in [-0.05, 0.05) (two per word), so dots stay finite.
__global__ void FillH(__half* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
    z ^= z >> 13;
    z *= 0x5bd1e995u;
    z ^= z >> 15;
    p[i] = __float2half(((z & 0xffff) / 65536.f - 0.5f) * 0.1f);
  }
}

__global__ void FillF(float* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2246822519u) ^ seed;
    z ^= z >> 15;
    z *= 0x2c1b3c6du;
    z ^= z >> 12;
    p[i] = ((z & 0xffffff) / 16777216.f - 0.5f) * 4.f;
  }
}

template <class T>
T* Dev(size_t n) {
  T* p = nullptr;
  CK(hipMalloc(&p, n * sizeof(T)));
  CK(hipMemset(p, 0, n * sizeof(T)));
  return p;
}

uint64_t Hash(const void* dev, size_t bytes) {
  std::vector<unsigned char> h(bytes);
  CK(hipMemcpy(h.data(), dev, bytes, hipMemcpyDeviceToHost));
  uint64_t x = 1469598103934665603ull;
  for (unsigned char c : h) x = (x ^ c) * 1099511628211ull;
  return x;
}

// Median and p10 of `launches` single-launch times (events around each launch). launch(i) runs copy i.
void Time(const char* name, int launches, const std::function<void(int)>& launch, const void* out, size_t bytes) {
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  for (int i = 0; i < 5; ++i) launch(i);
  CK(hipDeviceSynchronize());
  std::vector<float> us(launches);
  for (int i = 0; i < launches; ++i) {
    CK(hipEventRecord(a));
    launch(i);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, a, b));
    us[i] = ms * 1000.f;
  }
  CK(hipGetLastError());
  std::sort(us.begin(), us.end());
  // The output of copy 0 again, for the hash (every launch overwrote it).
  launch(0);
  CK(hipDeviceSynchronize());
  std::printf("case %-28s med_us %9.2f p10_us %9.2f hash %016llx\n", name, us[launches / 2], us[launches / 10],
              (unsigned long long)Hash(out, bytes));
  std::fflush(stdout);
  CK(hipEventDestroy(a));
  CK(hipEventDestroy(b));
}

std::vector<std::string> g_filter;

bool Want(const std::string& name) {
  if (g_filter.empty()) return true;
  for (const auto& f : g_filter)
    if (name.compare(0, f.size(), f) == 0) return true;
  return false;
}

int Copies(size_t bytes) { return (int)std::max<size_t>(2, (128u << 20) / bytes + 1); }

void DenseCase(int O, int launches) {
  if (!Want("DenseGemv<3> O=" + std::to_string(O))) return;
  constexpr int R = 3;
  const size_t wn = (size_t)O * kHidden;
  const int copies = Copies(wn * 2);
  __half* W = Dev<__half>(wn * copies);
  FillH<<<1024, 256>>>(W, wn * copies, 11u);
  float* x = Dev<float>((size_t)R * kHidden);
  FillF<<<64, 256>>>(x, (size_t)R * kHidden, 12u);
  float* y = Dev<float>((size_t)R * O);
  const std::string name = "DenseGemv<3> O=" + std::to_string(O);
  Time(name.c_str(), launches,
       [&](int i) { DenseGemvKernel<R><<<(O + 7) / 8, 256>>>(W + (size_t)(i % copies) * wn, O, x, kHidden, y, O); },
       y, (size_t)R * O * sizeof(float));
  CK(hipFree(W));
  CK(hipFree(x));
  CK(hipFree(y));
}

void HcDotsCase(int M, int launches) {
  if (!Want("HcDots<3> M=" + std::to_string(M))) return;
  constexpr int R = 3;
  const size_t wn = (size_t)M * kWide;
  const int copies = Copies(wn * 2);
  __half* fn = Dev<__half>(wn * copies);
  FillH<<<1024, 256>>>(fn, wn * copies, 21u);
  float* H = Dev<float>((size_t)R * kWide);
  FillF<<<64, 256>>>(H, (size_t)R * kWide, 22u);
  float* dots = Dev<float>((size_t)R * M * kStreams);
  float* ss = Dev<float>((size_t)R * kStreams);
  const std::string name = "HcDots<3> M=" + std::to_string(M);
  Time(name.c_str(), launches,
       [&](int i) {
         HcDotsKernel<R><<<((M + 1) * kStreams + 7) / 8, 256>>>(fn + (size_t)(i % copies) * wn, M, H, dots, ss);
       },
       dots, (size_t)R * M * kStreams * sizeof(float));
  // ss comes from the same launch; fold it into a second line so a change there shows too.
  std::printf("case %-28s hash %016llx\n", (name + " ss").c_str(),
              (unsigned long long)Hash(ss, (size_t)R * kStreams * sizeof(float)));
  CK(hipFree(fn));
  CK(hipFree(H));
  CK(hipFree(dots));
  CK(hipFree(ss));
}

template <int BITS, int NR>
void DraftCase(int n, int launches) {
  if (!Want("DraftQGemv<" + std::to_string(BITS) + "," + std::to_string(NR) + "> n=" + std::to_string(n))) return;
  const int V = kVocab, blocks = V / 128;
  constexpr int L = DraftQLoads(kHidden, BITS);
  const size_t words = (size_t)V * L;  // uint4 per copy
  const int copies = Copies(words * 16);
  uint4* w = Dev<uint4>(words * copies);
  FillU32<<<1024, 256>>>(reinterpret_cast<uint32_t*>(w), words * 4 * copies, 31u + BITS);
  __half* scales = Dev<__half>((size_t)V * L);
  FillH<<<1024, 256>>>(scales, (size_t)V * L, 32u);
  __half* xh = Dev<__half>((size_t)kDraftQMaxRows * kHidden);
  FillH<<<64, 256>>>(xh, (size_t)kDraftQMaxRows * kHidden, 33u);
  float* logits = Dev<float>((size_t)kDraftQMaxRows * V);
  float* maxima = Dev<float>((size_t)kDraftQMaxRows * blocks);
  const size_t lds = (size_t)NR * kHidden * sizeof(__half);
  const std::string name = "DraftQGemv<" + std::to_string(BITS) + "," + std::to_string(NR) + "> n=" + std::to_string(n);
  Time(name.c_str(), launches,
       [&](int i) {
         DraftQGemvKernel<BITS, NR><<<blocks, 256, lds>>>(w + (size_t)(i % copies) * words, scales, xh, n, V, logits,
                                                           maxima);
       },
       logits, (size_t)n * V * sizeof(float));
  std::printf("case %-28s hash %016llx\n", (name + " max").c_str(),
              (unsigned long long)Hash(maxima, (size_t)n * blocks * sizeof(float)));
  CK(hipFree(w));
  CK(hipFree(scales));
  CK(hipFree(xh));
  CK(hipFree(logits));
  CK(hipFree(maxima));
}

void QsaCase(int ctx, int launches) {
  if (!Want("QsaScoreRows<8> ctx=" + std::to_string(ctx))) return;
  constexpr int R = 8;
  const size_t pool = QsaPoolFloats(ctx + R);
  const int copies = Copies(pool * 4);
  float* ik = Dev<float>(pool * copies);
  FillF<<<1024, 256>>>(ik, pool * copies, 41u);
  float* idx = Dev<float>((size_t)R * kQsaIdxWidth);
  FillF<<<64, 256>>>(idx, (size_t)R * kQsaIdxWidth, 42u);
  const int nb = (ctx + R) / 4, sstride = (nb + 255) / 256 * 256;
  float* scores = Dev<float>((size_t)R * sstride);
  QsaArgs a{};
  a.idx = idx;
  a.scores = scores;
  a.sstride = sstride;
  a.pos0 = ctx;
  a.R = R;
  const std::string name = "QsaScoreRows<8> ctx=" + std::to_string(ctx);
  Time(name.c_str(), launches,
       [&](int i) {
         QsaArgs b = a;
         b.c.ik_pool = ik + (size_t)(i % copies) * pool;
         QsaScoreRowsKernel<R><<<(nb + 255) / 256, 256>>>(b);
       },
       scores, (size_t)R * sstride * sizeof(float));
  CK(hipFree(ik));
  CK(hipFree(idx));
  CK(hipFree(scores));
}

}  // namespace

int main(int argc, char** argv) {
  const int launches = argc > 1 ? std::atoi(argv[1]) : 200;
  if (argc > 2) {
    std::string f = argv[2];
    for (size_t at = 0; at <= f.size();) {
      const size_t comma = std::min(f.find(';', at), f.size());
      if (comma > at) g_filter.push_back(f.substr(at, comma - at));
      at = comma + 1;
    }
  }
  DenseCase(kExperts + 1, launches);
  DenseCase(12800, launches);
  HcDotsCase(kRank, launches);
  HcDotsCase(kRank + kStreams, launches);
  DraftCase<2, 1>(1, launches);
  DraftCase<2, 2>(2, launches);
  DraftCase<2, 4>(4, launches);
  DraftCase<2, 8>(8, launches);
  DraftCase<2, 8>(5, launches);
  DraftCase<4, 1>(1, launches);
  DraftCase<4, 2>(2, launches);
  DraftCase<4, 4>(4, launches);
  DraftCase<4, 8>(8, launches);
  DraftCase<4, 8>(5, launches);
  QsaCase(32768, launches);
  QsaCase(131072, launches);
  return 0;
}
