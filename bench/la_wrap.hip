// Lever 4 of oracles/decode-bw-2026-10-24.md: keep the lookahead running
// across the decode step boundary (no Stop() after the head, no new launch at
// the next step), and let it wrap to the start of the static arena, so that it
// reads layer 0 during the head, the host round trip and the next step's
// embed. The engine today stops it at the end of every step (Head:
// lookahead_.Stop()) and starts it again in the next Forward.
//
// A step here: the "layer mix" chain of la_mask.hip over one 1 GiB region
// (the same region every step, as decode reads the same static arena; 1 GiB
// is far more than the 32 MiB MALL, so every step starts cold), then
// hipStreamSynchronize and a host wait (the argmax copy and the host work
// between steps). Rows: ms per step (median of the step times of 3 runs of
// 16 steps), against "stop + start per step".
//
//   Modes: 0 no lookahead; 1 stop + start per step (the engine); 2 wrap
//   (one launch for all steps; a lap kernel on the main stream at each step
//   start; the progress is virtual: lap * total + offset).
//
//   la_wrap [total_mib] [host_us]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "../engine/exl3/lookahead.h"

using namespace crucible;

__device__ uint32_t g_wrap_sink;

namespace {

constexpr int kThreads = 256;

// The wrap state: progress and next are virtual (lap * total + offset, and a
// slab count over laps of `nslab` slabs).
struct WrapState {
  const uint8_t* base;
  unsigned long long total, window, progress, fetched, lap;
  unsigned nslab, next, active;
};

__device__ __forceinline__ void PublishWrap(WrapState* s, const void* w) {
  if (!s || blockIdx.x != 0 || threadIdx.x != 0) return;
  const unsigned long long lap = __hip_atomic_load(&s->lap, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  const auto off = (unsigned long long)(reinterpret_cast<uintptr_t>(w) - reinterpret_cast<uintptr_t>(s->base));
  __hip_atomic_store(&s->progress, lap * s->total + off, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__global__ void __launch_bounds__(kThreads) ChunkKernel(const uint4* __restrict__ p, size_t n, WrapState* ws) {
  PublishWeights(p);  // the engine's lookahead (mode 1)
  PublishWrap(ws, p);  // the wrap lookahead (mode 2)
  uint32_t acc = 0;
  const size_t stride = (size_t)gridDim.x * kThreads;
#pragma unroll 8
  for (size_t i = (size_t)blockIdx.x * kThreads + threadIdx.x; i < n; i += stride) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) g_wrap_sink = acc;
}

__global__ void TinyKernel(float* v) {
  float x = v[threadIdx.x];
#pragma unroll 4
  for (int i = 0; i < 64; ++i) x = x * 0.999f + 0.001f;
  v[threadIdx.x] = x;
}

// Step start (main stream, before the first chunk): the next lap.
__global__ void WrapLapKernel(WrapState* s, unsigned long long lap) {
  __hip_atomic_store(&s->lap, lap, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(&s->progress, lap * s->total, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__global__ void WrapStopKernel(WrapState* s) {
  __hip_atomic_store(&s->active, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// LookaheadKernel with virtual offsets: slab k is lap k / nslab, arena offset
// (k % nslab) * kLookaheadSlab. Past the end of a lap it goes on with the
// next lap's first slabs, inside the same window.
__global__ void __launch_bounds__(256) WrapLookaheadKernel(WrapState* s) {
  __shared__ unsigned slab;
  __shared__ bool stop, skip;
  uint32_t acc = 0;
  unsigned long long fetched = 0;
  const unsigned long long total = s->total;
  const unsigned nslab = s->nslab;
  auto vslab = [&](unsigned long long p) {
    return (unsigned)((p / total) * nslab + (p % total) / kLookaheadSlab);
  };
  for (;;) {
    if (threadIdx.x == 0) {
      auto p = __hip_atomic_load(&s->progress, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      atomicMax(&s->next, vslab(p));
      slab = atomicAdd(&s->next, 1u);
      const unsigned long long v = (unsigned long long)(slab / nslab) * total +
                                   (unsigned long long)(slab % nslab) * kLookaheadSlab;
      do {
        stop = !__hip_atomic_load(&s->active, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        p = __hip_atomic_load(&s->progress, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (stop || v < p + s->window) break;
        __builtin_amdgcn_s_sleep(2);
      } while (true);
      skip = v + kLookaheadSlab <= p;
    }
    __syncthreads();
    if (stop) break;
    if (!skip) {
      const auto off = (unsigned long long)(slab % nslab) * kLookaheadSlab;
      const auto left = total - off;
      const size_t count = (left < kLookaheadSlab ? left : kLookaheadSlab) / 16;
      const uint4* p = reinterpret_cast<const uint4*>(s->base + off);
#pragma unroll 16
      for (size_t i = threadIdx.x; i < count; i += 256) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
      fetched += count * 16;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(&s->fetched, fetched);
  if (acc == 0x9e3779b9u) atomicExch(&g_wrap_sink, acc);
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void HostWait(int us) {
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() < us) {
  }
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const size_t total = (size_t)(argc > 1 ? std::stoul(argv[1]) : 1024) << 20;
  const int host_us = argc > 2 ? std::atoi(argv[2]) : 60;
  constexpr int kSteps = 16, kRuns = 3, kBlocks = 4;
  constexpr unsigned long long kWindow = 24ull << 20;
  uint8_t* region;
  HIP_OK(hipMalloc(&region, total));
  HIP_OK(hipMemset(region, 1, total));
  float* tiny;
  HIP_OK(hipMalloc(&tiny, 4096));
  HIP_OK(hipMemset(tiny, 0, 4096));
  hipStream_t main, side;
  HIP_OK(hipStreamCreateWithFlags(&main, hipStreamNonBlocking));
  HIP_OK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));
  hipEvent_t ready;
  HIP_OK(hipEventCreateWithFlags(&ready, hipEventDisableTiming));

  const std::vector<size_t> layer_mb10 = {66, 66, 262, 98, 66, 66, 26, 143, 71};
  const std::vector<int> tiny_after = {0, 0, 0, 5, 0, 0, 4, 0, 0};
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
  for (size_t c : chunks) bytes += c;
  LookaheadState la{};
  la.base = region;
  la.total = bytes;  // the chunk sum: no unread tail to prefetch
  LookaheadState* las;
  HIP_OK(hipMalloc(&las, sizeof(la)));
  HIP_OK(hipMemcpy(las, &la, sizeof(la), hipMemcpyHostToDevice));
  LookaheadInitKernel<<<1, 1, 0, main>>>(las);
  WrapState wi{};
  wi.base = region;
  wi.total = bytes;
  wi.window = kWindow;
  wi.nslab = (unsigned)((bytes + kLookaheadSlab - 1) / kLookaheadSlab);
  WrapState* ws;
  HIP_OK(hipMalloc(&ws, sizeof(wi)));

  std::printf("la_wrap: %zu chunks, %.0f MB per step, host wait %d us, %d steps x %d runs, lookahead %d blocks 24 MiB\n",
              chunks.size(), bytes / 1e6, host_us, kSteps, kRuns, kBlocks);
  const char* names[] = {"no lookahead", "stop + start per step (engine)", "wrap across steps"};
  double ref = 0;
  for (int mode : {0, 1, 2, 1, 2}) {  // modes 1 and 2 twice, interleaved
    std::vector<double> step_ms;
    double fetched = 0;
    for (int run = 0; run < kRuns + 1; ++run) {  // run 0 warms up
      HIP_OK(hipDeviceSynchronize());
      WrapState* wsp = mode == 2 ? ws : nullptr;
      if (mode == 2) {
        WrapState w0 = wi;
        w0.active = 1;
        HIP_OK(hipMemcpy(ws, &w0, sizeof(w0), hipMemcpyHostToDevice));
        WrapLookaheadKernel<<<kBlocks, 256, 0, side>>>(ws);
      }
      auto last = std::chrono::steady_clock::now();
      for (int step = 0; step < kSteps; ++step) {
        if (mode == 1) {  // LookaheadRunner::Start
          LookaheadResetKernel<<<1, 1, 0, main>>>(las, kWindow);
          HIP_OK(hipEventRecord(ready, main));
          HIP_OK(hipStreamWaitEvent(side, ready, 0));
          LookaheadKernel<<<kBlocks, 256, 0, side>>>(las);
        }
        if (mode == 2) WrapLapKernel<<<1, 1, 0, main>>>(ws, step);
        size_t off = 0;
        for (size_t i = 0; i < chunks.size(); ++i) {
          ChunkKernel<<<160, kThreads, 0, main>>>(reinterpret_cast<const uint4*>(region + off), chunks[i] / 16, wsp);
          off += chunks[i];
          for (int k = 0; k < tinies[i]; ++k) TinyKernel<<<1, 64, 0, main>>>(tiny);
        }
        if (mode == 1) {  // LookaheadRunner::Stop
          LookaheadStopKernel<<<1, 1, 0, main>>>(las);
          HIP_OK(hipStreamSynchronize(side));
        }
        HIP_OK(hipStreamSynchronize(main));  // the argmax copy
        HostWait(host_us);
        const auto now = std::chrono::steady_clock::now();
        if (run) step_ms.push_back(std::chrono::duration<double, std::milli>(now - last).count());
        last = now;
      }
      if (mode == 2) {
        WrapStopKernel<<<1, 1, 0, main>>>(ws);
        HIP_OK(hipDeviceSynchronize());
        WrapState back;
        HIP_OK(hipMemcpy(&back, ws, sizeof(back), hipMemcpyDeviceToHost));
        if (run) fetched += (double)back.fetched / ((double)bytes * kSteps);
      }
    }
    const double m = Median(step_ms);
    if (mode == 1 && !ref) ref = m;
    std::printf("  %-34s %8.3f ms per step  %7.1f GB/s  %s  fetched %s\n", names[mode], m, bytes / m / 1e6,
                ref ? (std::to_string(100.0 * (ref / m - 1)).substr(0, 5) + "%").c_str() : "-",
                mode == 2 ? (std::to_string(100.0 * fetched / kRuns).substr(0, 5) + "%").c_str() : "-");
  }
  HIP_OK(hipGetLastError());
  return 0;
}
