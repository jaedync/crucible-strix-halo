// Lever 3 of oracles/decode-bw-2026-10-24.md: does a CU mask on the lookahead
// stream (hipExtStreamCreateWithCUMask) make the main stream faster?
//
// Part 1, dispatch tax: 500 dependent tiny kernels on the main stream, alone,
// beside a 4-block persistent spinner (the lookahead's idle loop), and with
// the spinner and the main stream on disjoint CU masks.
// Part 2, layer mix: the engine's LookaheadKernel (engine/exl3/lookahead.h,
// 4 blocks, 24 MiB window as ResolveRowPolicy) runs beside a chain shaped like
// one 4.05 decode step: per layer the streaming kernels of lookahead.hip's
// "layer mix" plus 9 tiny dependent kernels (route chain, QSA). The chain
// kernels publish their weights as the engine kernels do (PublishWeights).
// A row: median ms over 8 reps (the pool rotates over 4 regions, so no rep
// starts warm), GB/s of the chain bytes, and the change against the row
// "lookahead, no masks" (the engine today).
//
//   la_mask [total_mib] [reps]
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "../engine/exl3/lookahead.h"

using namespace crucible;

__device__ uint32_t g_mask_sink;

namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads) ChunkKernel(const uint4* __restrict__ p, size_t n) {
  PublishWeights(p);
  uint32_t acc = 0;
  const size_t stride = (size_t)gridDim.x * kThreads;
#pragma unroll 8
  for (size_t i = (size_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) g_mask_sink = acc;
}

// A route/QSA-like kernel: one block, a short dependent body, no weights.
__global__ void TinyKernel(float* v) {
  float x = v[threadIdx.x];
#pragma unroll 4
  for (int i = 0; i < 64; ++i) x = x * 0.999f + 0.001f;
  v[threadIdx.x] = x;
}

// The lookahead's idle loop without reads: s_sleep and an atomic load, until stopped.
__global__ void IdleSpinKernel(const unsigned* stop) {
  while (!__hip_atomic_load(stop, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) __builtin_amdgcn_s_sleep(2);
}

struct Timer {
  hipEvent_t a, b;
  Timer() {
    HIP_OK(hipEventCreate(&a));
    HIP_OK(hipEventCreate(&b));
  }
  ~Timer() {
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
  }
  void Begin(hipStream_t s) { HIP_OK(hipEventRecord(a, s)); }
  double EndMs(hipStream_t s) {
    HIP_OK(hipEventRecord(b, s));
    HIP_OK(hipEventSynchronize(b));
    float ms = 0;
    HIP_OK(hipEventElapsedTime(&ms, a, b));
    return ms;
  }
};

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Mask of CUs [lo, hi) over `ncu` CUs, as uint32 words. Note for the engine:
// hipExtStreamCreateWithCUMask has no flags, so the stream is blocking and the
// null stream waits for its work. The engine's main stream is the null stream,
// so a masked lookahead stream there would deadlock (the lookahead kernel ends
// only on a stop from the main stream). Here no null-stream work is queued
// while a spinner or lookahead runs.
std::vector<uint32_t> Mask(int ncu, int lo, int hi) {
  std::vector<uint32_t> m((ncu + 31) / 32, 0);
  for (int c = lo; c < hi; ++c) m[c / 32] |= 1u << (c % 32);
  return m;
}

hipStream_t Stream(int ncu, int lo, int hi) {
  hipStream_t s;
  if (lo == 0 && hi == ncu) {
    HIP_OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  } else {
    std::vector<uint32_t> m = Mask(ncu, lo, hi);
    HIP_OK(hipExtStreamCreateWithCUMask(&s, (uint32_t)m.size(), m.data()));
  }
  return s;
}

struct Cfg {
  const char* name;
  int la_blocks;  // 0: no lookahead
  int la_cus;     // CUs of the lookahead stream (0: all, no mask)
  bool main_masked;  // main stream on the other CUs
};

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const size_t total = (size_t)(argc > 1 ? std::stoul(argv[1]) : 1024) << 20;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 8;
  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  const int ncu = prop.multiProcessorCount;
  std::printf("device %s, %d CUs\n", prop.gcnArchName, ncu);

  float* tiny;
  HIP_OK(hipMalloc(&tiny, 4096));
  HIP_OK(hipMemset(tiny, 0, 4096));
  unsigned* stop;
  HIP_OK(hipMalloc(&stop, 4));

  // Part 1: dispatch tax.
  std::printf("\n1. 500 dependent tiny kernels on the main stream (us per launch, median of %d)\n", reps);
  for (Cfg c : {Cfg{"alone", 0, 0, false}, Cfg{"spinner 4 blocks, no masks", 4, 0, false},
                Cfg{"spinner on 2 CUs, main all", 4, 2, false}, Cfg{"spinner on 2 CUs, main on the rest", 4, 2, true},
                Cfg{"spinner on 4 CUs, main on the rest", 4, 4, true}}) {
    hipStream_t main = Stream(ncu, 0, c.main_masked ? ncu - c.la_cus : ncu);
    hipStream_t side = Stream(ncu, c.la_cus ? ncu - c.la_cus : 0, ncu);
    std::vector<double> us;
    for (int r = 0; r < reps; ++r) {
      HIP_OK(hipMemset(stop, 0, 4));
      HIP_OK(hipDeviceSynchronize());
      if (c.la_blocks) IdleSpinKernel<<<c.la_blocks, kThreads, 0, side>>>(stop);
      Timer t;
      t.Begin(main);
      for (int i = 0; i < 500; ++i) TinyKernel<<<1, 64, 0, main>>>(tiny);
      us.push_back(1000.0 * t.EndMs(main) / 500);
      // Not hipMemcpy: a CU-masked stream is a blocking stream, so a null-stream
      // copy would wait for the spinner, which waits for this copy.
      static const unsigned one = 1;
      HIP_OK(hipMemcpyAsync(stop, &one, 4, hipMemcpyHostToDevice, main));
      HIP_OK(hipStreamSynchronize(main));
      HIP_OK(hipDeviceSynchronize());
    }
    std::printf("  %-40s %6.2f us\n", c.name, Median(us));
    HIP_OK(hipStreamDestroy(main));
    HIP_OK(hipStreamDestroy(side));
  }

  // Part 2: the layer mix with the engine's lookahead.
  constexpr int kRegions = 4;
  uint8_t* pool;
  HIP_OK(hipMalloc(&pool, total * kRegions));
  HIP_OK(hipMemset(pool, 1, total * kRegions));
  LookaheadState init{};
  init.total = total;
  LookaheadState* st;
  HIP_OK(hipMalloc(&st, sizeof(init)));
  // MB x 10 per streaming kernel of one layer (lookahead.hip), tiny kernels after each index in `tiny_after`.
  const std::vector<size_t> layer_mb10 = {66, 66, 262, 98, 66, 66, 26, 143, 71};
  const std::vector<int> tiny_after = {0, 0, 0, 5, 0, 0, 4, 0, 0};  // 5 QSA after attention, 4 route after router
  std::vector<size_t> chunks;
  std::vector<int> tinies;
  for (size_t b = 0;;) {
    bool full = false;
    for (size_t i = 0; i < layer_mb10.size(); ++i) {
      const size_t bytes = ((layer_mb10[i] << 20) / 10) & ~size_t(255);
      if (b + bytes > total) {
        full = true;
        break;
      }
      chunks.push_back(bytes);
      tinies.push_back(tiny_after[i]);
      b += bytes;
    }
    if (full) break;
  }
  size_t bytes = 0;
  int ntiny = 0;
  for (size_t i = 0; i < chunks.size(); ++i) bytes += chunks[i], ntiny += tinies[i];
  std::printf("\n2. layer mix: %zu streaming kernels (%.0f MB), %d tiny kernels; lookahead window 24 MiB\n",
              chunks.size(), bytes / 1e6, ntiny);
  std::printf("  %-44s %8s %8s %8s %8s\n", "config", "ms", "GB/s", "vs la", "fetched");
  double ref = 0;
  for (Cfg c : {Cfg{"no lookahead", 0, 0, false}, Cfg{"lookahead 4, no masks (engine)", 4, 0, false},
                Cfg{"lookahead 4 on 2 CUs, main all", 4, 2, false}, Cfg{"lookahead 4 on 2 CUs, main rest", 4, 2, true},
                Cfg{"lookahead 4 on 4 CUs, main all", 4, 4, false}, Cfg{"lookahead 4 on 4 CUs, main rest", 4, 4, true},
                Cfg{"lookahead 8 on 4 CUs, main rest", 8, 4, true}, Cfg{"lookahead 8, no masks", 8, 0, false}}) {
    hipStream_t main = Stream(ncu, 0, c.main_masked ? ncu - c.la_cus : ncu);
    hipStream_t side = Stream(ncu, c.la_cus ? ncu - c.la_cus : 0, ncu);
    std::vector<double> ms;
    double fetched = 0;
    for (int r = 0; r < reps + 1; ++r) {  // rep 0 is a warm-up
      uint8_t* region = pool + (size_t)(r % kRegions) * total;
      init.base = region;
      HIP_OK(hipMemcpy(st, &init, sizeof(init), hipMemcpyHostToDevice));
      LookaheadInitKernel<<<1, 1, 0, main>>>(st);
      HIP_OK(hipDeviceSynchronize());
      Timer t;
      t.Begin(main);
      if (c.la_blocks) {
        // As LookaheadRunner::Start: reset on the main stream, then the side stream waits for it.
        LookaheadResetKernel<<<1, 1, 0, main>>>(st, 24ull << 20);
        hipEvent_t ev;
        HIP_OK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
        HIP_OK(hipEventRecord(ev, main));
        HIP_OK(hipStreamWaitEvent(side, ev, 0));
        LookaheadKernel<<<c.la_blocks, 256, 0, side>>>(st);
        HIP_OK(hipEventDestroy(ev));
      }
      size_t off = 0;
      for (size_t i = 0; i < chunks.size(); ++i) {
        ChunkKernel<<<160, kThreads, 0, main>>>(reinterpret_cast<const uint4*>(region + off), chunks[i] / 16);
        off += chunks[i];
        for (int k = 0; k < tinies[i]; ++k) TinyKernel<<<1, 64, 0, main>>>(tiny);
      }
      const double m = t.EndMs(main);
      LookaheadStopKernel<<<1, 1, 0, main>>>(st);
      HIP_OK(hipDeviceSynchronize());
      LookaheadState back;
      HIP_OK(hipMemcpy(&back, st, sizeof(back), hipMemcpyDeviceToHost));
      if (r) ms.push_back(m), fetched += (double)back.fetched / bytes;
    }
    const double m = Median(ms);
    if (c.la_blocks == 4 && c.la_cus == 0) ref = m;
    std::printf("  %-44s %8.3f %8.1f %7s %7.1f%%\n", c.name, m, bytes / m / 1e6,
                ref ? (std::to_string(100.0 * (ref / m - 1)).substr(0, 5) + "%").c_str() : "-",
                100.0 * fetched / reps);
    HIP_OK(hipStreamDestroy(main));
    HIP_OK(hipStreamDestroy(side));
  }
  HIP_OK(hipGetLastError());
  return 0;
}
