// Shared helpers for the gfx1151 calibration probes.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#define HIP_CHECK(expr)                                                       \
  do {                                                                        \
    hipError_t err_ = (expr);                                                 \
    if (err_ != hipSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr,      \
                   hipGetErrorString(err_));                                  \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace bench {

// Samples GPU clock, package power and edge temperature from amdgpu hwmon
// while a measurement runs, so every number can be read against the clock it
// actually ran at (sclk drops under sustained load at the 100 W limit).
class Sampler {
 public:
  explicit Sampler(int period_ms = 20) : period_ms_(period_ms) {
    base_ = FindHwmon();
  }
  void Start() {
    stop_ = false;
    samples_.clear();
    thread_ = std::thread([this] { Loop(); });
  }
  void Stop() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
  }
  // "sclk_mhz mean/max, power_w mean/max, temp_c max"
  std::string Summary() const {
    if (samples_.empty() || base_.empty()) return "sclk=? power=?";
    double f = 0, fmax = 0, p = 0, pmax = 0, tmax = 0;
    for (const auto& s : samples_) {
      f += s.mhz;
      fmax = std::max(fmax, s.mhz);
      p += s.watts;
      pmax = std::max(pmax, s.watts);
      tmax = std::max(tmax, s.temp_c);
    }
    double n = static_cast<double>(samples_.size());
    char buf[160];
    std::snprintf(buf, sizeof buf, "sclk=%.0f/%.0fMHz pwr=%.1f/%.1fW temp=%.0fC",
                  f / n, fmax, p / n, pmax, tmax);
    return buf;
  }
  double MeanMhz() const {
    if (samples_.empty()) return 0;
    double f = 0;
    for (const auto& s : samples_) f += s.mhz;
    return f / samples_.size();
  }
  double MeanWatts() const {
    if (samples_.empty()) return 0;
    double p = 0;
    for (const auto& s : samples_) p += s.watts;
    return p / samples_.size();
  }
  double MaxTemp() const {
    double t = 0;
    for (const auto& s : samples_) t = std::max(t, s.temp_c);
    return t;
  }
  size_t Count() const { return samples_.size(); }

 private:
  struct Sample {
    double mhz, watts, temp_c;
  };
  static std::string FindHwmon() {
    for (int i = 0; i < 16; ++i) {
      std::string p = "/sys/class/drm/card1/device/hwmon/hwmon" +
                      std::to_string(i) + "/";
      std::ifstream f(p + "freq1_input");
      if (f.good()) return p;
    }
    return "";
  }
  static double Read(const std::string& path) {
    std::ifstream f(path);
    double v = 0;
    f >> v;
    return v;
  }
  void Loop() {
    while (!stop_) {
      if (!base_.empty()) {
        samples_.push_back({Read(base_ + "freq1_input") / 1e6,
                            Read(base_ + "power1_average") / 1e6,
                            Read(base_ + "temp1_input") / 1e3});
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(period_ms_));
    }
  }
  int period_ms_;
  std::string base_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::vector<Sample> samples_;
};

struct EventTimer {
  hipEvent_t a, b;
  EventTimer() {
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
  }
  ~EventTimer() {
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
  }
  void Begin(hipStream_t s = nullptr) { HIP_CHECK(hipEventRecord(a, s)); }
  float EndMs(hipStream_t s = nullptr) {
    HIP_CHECK(hipEventRecord(b, s));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return ms;
  }
};

inline double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v[v.size() / 2];
}

inline int NumCUs() {
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, 0));
  return p.multiProcessorCount;
}

}  // namespace bench
