// Launch gap probe (moe-sg-fix next lever, timing only): how much main-stream time does a chain of N dependent kernels
// cost beyond their work, launched one by one on the stream or replayed as one HIP graph?
//   launch_gap [N=700] [ITERS=20]
// Each kernel has 120 blocks of 256 threads (the stream-grid shape) and spins about WORK_US microseconds
// (wall_clock64, a constant clock), then writes one word. Rows: WORK_US = 0, 5, 20, 50. Per row: us per chain
// (median of ITERS), us per kernel above the work, for the stream launches and for the graph replay.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_OK(x)                                                              \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void SpinKernel(unsigned* out, long long ticks) {
  const long long t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) {
  }
  if (threadIdx.x == 0) out[blockIdx.x] += 1;
}

static long long TicksPerUs() {
  int khz = 0;
  HIP_OK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
  return khz / 1000;
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 700;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
  unsigned* out;
  HIP_OK(hipMalloc(&out, 120 * sizeof(unsigned)));
  hipStream_t s;
  HIP_OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  const long long tpu = TicksPerUs();
  std::printf("ticks per us %lld, N %d kernels per chain, %d iters\n", tpu, n, iters);
  for (const int work_us : {0, 5, 20, 50}) {
    const long long ticks = work_us * tpu;
    auto chain = [&] {
      for (int i = 0; i < n; ++i) hipLaunchKernelGGL(SpinKernel, dim3(120), dim3(256), 0, s, out, ticks);
    };
    // One kernel alone: its time is the work plus one launch.
    std::vector<float> one, st, gr;
    for (int it = 0; it < iters; ++it) {
      HIP_OK(hipEventRecord(a, s));
      hipLaunchKernelGGL(SpinKernel, dim3(120), dim3(256), 0, s, out, ticks);
      HIP_OK(hipEventRecord(b, s));
      HIP_OK(hipEventSynchronize(b));
      float ms;
      HIP_OK(hipEventElapsedTime(&ms, a, b));
      one.push_back(ms * 1000);
    }
    for (int it = 0; it < iters; ++it) {
      HIP_OK(hipEventRecord(a, s));
      chain();
      HIP_OK(hipEventRecord(b, s));
      HIP_OK(hipEventSynchronize(b));
      float ms;
      HIP_OK(hipEventElapsedTime(&ms, a, b));
      st.push_back(ms * 1000);
    }
    hipGraph_t g;
    hipGraphExec_t ge;
    HIP_OK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    chain();
    HIP_OK(hipStreamEndCapture(s, &g));
    HIP_OK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    HIP_OK(hipGraphLaunch(ge, s));
    HIP_OK(hipStreamSynchronize(s));
    for (int it = 0; it < iters; ++it) {
      HIP_OK(hipEventRecord(a, s));
      HIP_OK(hipGraphLaunch(ge, s));
      HIP_OK(hipEventRecord(b, s));
      HIP_OK(hipEventSynchronize(b));
      float ms;
      HIP_OK(hipEventElapsedTime(&ms, a, b));
      gr.push_back(ms * 1000);
    }
    HIP_OK(hipGraphExecDestroy(ge));
    HIP_OK(hipGraphDestroy(g));
    auto med = [](std::vector<float> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    const float m1 = med(one), ms_ = med(st), mg = med(gr);
    std::printf("work %2d us: one kernel %.1f us | stream chain %.1f us, %.2f us per kernel above the work | "
                "graph chain %.1f us, %.2f us per kernel above the work\n",
                work_us, m1, ms_, ms_ / n - work_us, mg, mg / n - work_us);
  }
  HIP_OK(hipFree(out));
  return 0;
}
