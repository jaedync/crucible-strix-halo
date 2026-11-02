#pragma once
#include "common.h"
#include "../engine/gfx1151/common.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <thread>

namespace prefill_bench {
template <class T> struct Buffer {
  T* p = nullptr;
  size_t n;
  explicit Buffer(size_t count) : n(count) {
    HIP_OK(hipMalloc(&p, n * sizeof(T)));
    HIP_OK(hipMemset(p, 0, n * sizeof(T)));
  }
  explicit Buffer(const std::vector<T>& x) : Buffer(x.size()) { Set(x); }
  Buffer(const Buffer&) = delete;
  ~Buffer() { HIP_OK(hipFree(p)); }
  void Set(const std::vector<T>& x) {
    if (x.size() != n) std::abort();
    HIP_OK(hipMemcpy(p, x.data(), n * sizeof(T), hipMemcpyHostToDevice));
  }
  void Zero() { HIP_OK(hipMemset(p, 0, n * sizeof(T))); }
  // All bytes 0xFF: NaN for fp16 and fp32, so an unwritten output cannot pass a byte compare.
  void Nan() { HIP_OK(hipMemset(p, 0xFF, n * sizeof(T))); }
  std::vector<T> Get() const {
    std::vector<T> x(n);
    HIP_OK(hipMemcpy(x.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
    return x;
  }
};
inline std::vector<float> Random(size_t n, float scale, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.f, scale);
  std::vector<float> x(n);
  for (auto& v : x) v = normal(rng);
  return x;
}
inline std::vector<__half> Half(const std::vector<float>& x) {
  std::vector<__half> h(x.size());
  for (size_t i = 0; i < x.size(); ++i) h[i] = __float2half(x[i]);
  return h;
}
struct Error { double rel_l2, max_abs, max_scaled, max_relative; };
template <class T> double Value(T x) { return (double)x; }
inline double Value(__half x) { return __half2float(x); }
template <class A, class B> Error Compare(const std::vector<A>& x, const std::vector<B>& y) {
  double err = 0, norm = 0, max_abs = 0, max_scaled = 0, max_relative = 0, peak = 0;
  if (x.size() != y.size()) std::abort();
  for (auto v : y) peak = std::max(peak, std::abs(Value(v)));
  for (size_t i = 0; i < x.size(); ++i) {
    const double ref = Value(y[i]), delta = Value(x[i]) - ref;
    if (!std::isfinite(ref) || !std::isfinite(delta)) std::abort();
    const double relative = ref != 0 ? std::abs(delta / ref) : delta == 0 ? 0 : INFINITY;
    max_relative = std::max(max_relative, relative);
    err += delta * delta;
    norm += ref * ref;
    max_abs = std::max(max_abs, std::abs(delta));
    max_scaled = std::max(max_scaled, std::abs(delta) / std::max(std::abs(ref), peak * 1e-3));
  }
  return {std::sqrt(err / std::max(norm, 1e-30)), max_abs, max_scaled, max_relative};
}
template <class F> double Time(F launch, int repeat = 5) {
  launch();
  HIP_OK(hipDeviceSynchronize());
  std::vector<double> us;
  bench::EventTimer timer;
  for (int i = 0; i < repeat; ++i) {
    timer.Begin();
    launch();
    us.push_back(timer.EndMs() * 1000.);
  }
  return bench::Median(us);
}
// Energy loop of one kernel class (a private measurement script reads the line): run `call` back to back for ENERGY_S
// seconds and average the socket power (gpu_metrics v3.0, u32 mW at byte 112) over the last
// 70 percent of the loop. The SMU averages its own window, so the loop must be long (10 s or more). No-op without ENERGY_S.
inline double SocketW() {
  std::ifstream f("/sys/class/drm/card1/device/gpu_metrics", std::ios::binary);
  unsigned char buf[264];
  unsigned mw = 0;
  if (!f.read(reinterpret_cast<char*>(buf), 264)) return -1;
  std::memcpy(&mw, buf + 112, 4);
  return mw / 1000.;
}
template <class F> void EnergyHook(const char* cls, const char* what, F call) {
  const char* env = std::getenv("ENERGY_S");
  const double seconds = env ? std::atof(env) : 0;
  if (seconds <= 0) return;
  call();
  HIP_OK(hipDeviceSynchronize());
  std::atomic<bool> run{true};
  std::vector<double> w;
  std::thread sampler([&] {
    const auto t0 = std::chrono::steady_clock::now();
    while (run) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (std::chrono::steady_clock::now() - t0 > std::chrono::duration<double>(seconds * 0.3)) w.push_back(SocketW());
    }
  });
  bench::EventTimer timer;
  int calls = 0;
  timer.Begin();
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - t0 < std::chrono::duration<double>(seconds)) {
    for (int i = 0; i < 4; ++i) call();
    calls += 4;
    HIP_OK(hipDeviceSynchronize());
  }
  const double us = timer.EndMs() * 1000. / calls;
  run = false;
  sampler.join();
  double m = 0;
  for (double x : w) m += x;
  m /= std::max<size_t>(1, w.size());
  std::printf("energy class=%s what=%s calls=%d us_per_call=%.2f socket_w=%.2f mj_per_call=%.4f samples=%zu\n", cls, what, calls, us, m,
              us * m / 1000., w.size());
}
inline void CheckLaunch() { HIP_OK(hipGetLastError()); HIP_OK(hipDeviceSynchronize()); }
}  // namespace prefill_bench
