// Chunk-64 WY/UT versus the engine's exact row-by-row recurrent kernel.
#include "prefill_common.h"
#include <cstring>
#include "kernels/prefill_gdn_packed.h"
#include "kernels/prefill_gdn_lanes.h"
#include <filesystem>
using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

template <class T> std::vector<T> ReadFile(const std::string& path, size_t n) {
  std::vector<T> v(n);
  std::ifstream file(path, std::ios::binary);
  if (!file.read((char*)v.data(), n * sizeof(T))) {
    std::fprintf(stderr, "cannot read %s (%zu elements)\n", path.c_str(), n);
    std::exit(2);
  }
  return v;
}

struct Inputs {
  std::vector<float> qkv, z, x, conv, alog, dt, gamma, logits;
  std::vector<__half> ab, suh;
};

Inputs MakeInput(int rows, const std::string& fixture) {
  if (!fixture.empty()) {
    int fixture_rows = 0;
    std::ifstream metadata(fixture + "/rows.txt");
    if (!(metadata >> fixture_rows) || fixture_rows != rows) {
      std::fprintf(stderr, "fixture rows.txt must match the requested row count\n");
      std::exit(2);
    }
    auto f = [&](const char* n, size_t s) { return ReadFile<float>(fixture + "/" + n + ".f32", s); };
    return {f("qkv", rows * 10240), f("z", rows * 6144), f("x", rows * kHidden),
      f("conv", 10240 * 4), f("alog", 48), f("dt", 48), f("gamma", 128), f("ab_logits", rows * 96),
      ReadFile<__half>(fixture + "/ab.f16", 96 * kHidden),
      ReadFile<__half>(fixture + "/suh.f16", 6144)};
  }
  Inputs in{Random((size_t)rows * 10240, .3f, 12), Random((size_t)rows * 6144, .3f, 13),
    Random((size_t)rows * kHidden, .1f, 14), Random(10240 * 4, .25f, 15),
    Random(48, .3f, 16), Random(48, .2f, 17), std::vector<float>(128, 1.f), {},
    Half(Random(96 * kHidden, .05f, 18)), std::vector<__half>(6144, __float2half(1.f))};
  return in;
}

__global__ void AbKernel(const float* x, const __half* weights, float* out) {
  __shared__ float red[8];
  const int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
  float a = 0.f, b = 0.f;
  for (int i = t; i < kHidden; i += 256) {
    a += __half2float(weights[h * kHidden + i]) * x[r * kHidden + i];
    b += __half2float(weights[(48 + h) * kHidden + i]) * x[r * kHidden + i];
  }
  a = BlockSum<256>(a, red);
  b = BlockSum<256>(b, red);
  if (t == 0) { out[r * 96 + h] = a; out[r * 96 + 48 + h] = b; }
}

// This shares the exact state ownership and contraction order of GdnKernel.
// It exposes the un-gated fp32 readout without changing decode source files.
__global__ void __launch_bounds__(256) SerialPreparedKernel(GdnPrefill a) {
  __shared__ float part[2][128];
  const int h = blockIdx.x, t = threadIdx.x, j = t & 127, half = t >> 7;
  float state[64];
  float* dst = a.input.state + (size_t)h * 128 * 128 + half * 64 * 128;
#pragma unroll
  for (int i = 0; i < 64; ++i) state[i] = dst[i * 128 + j];
  for (int r = 0; r < a.input.R; ++r) {
    const size_t b = GdnBase(a, h, r / 64) + r % 64;
    const float* k = a.k + b * 128 + half * 64;
    const float* q = a.q + b * 128 + half * 64;
    const float decay = expf(a.g[b]);
    float memory = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) { state[i] *= decay; memory += state[i] * k[i]; }
    part[half][j] = memory;
    __syncthreads();
    const float delta = a.beta[b] * (a.v[b * 128 + j] - (part[0][j] + part[1][j]));
    __syncthreads();
    float y = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) { state[i] += k[i] * delta; y += state[i] * q[i]; }
    part[half][j] = y;
    __syncthreads();
    if (half == 0) a.y[b * 128 + j] = part[0][j] + part[1][j];
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 64; ++i) dst[i * 128 + j] = state[i];
}

struct PackedBuffer {
  Buffer<__half> hi, lo;
  explicit PackedBuffer(size_t n) : hi(n), lo(n) {}
  Packed Get() { return {hi.p, lo.p}; }
};

void Run(int rows, const std::string& fixture) {
  const auto in = MakeInput(rows, fixture);
  Buffer<float> qkv(in.qkv), z(in.z), x(in.x), conv(in.conv), alog(in.alog), dt(in.dt), gamma(in.gamma);
  Buffer<__half> ab(in.ab), suh(in.suh);
  Buffer<float> logits(rows * 96), hist(16 * 10240), state(48 * 128 * 128), pend(48 * kMaxRows * kPend);
  Buffer<__half> output((size_t)rows * 6144);
  const auto initial = std::getenv("PREFILL_INITIAL") ? Random(state.n, .01f, 91) : std::vector<float>(state.n, 0.f);
  state.Set(initial);
  AbKernel<<<dim3(48, rows), 256>>>(x.p, ab.p, logits.p);
  GdnArgs input{qkv.p, z.p, x.p, ab.p, conv.p, alog.p, dt.p, gamma.p, hist.p, state.p, suh.p,
    output.p, pend.p, 0, rows, 0, rows};
  const int chunks = (rows + 63) / 64;
  const size_t nv = (size_t)48 * chunks * 64 * 128, ng = nv / 128;
  Buffer<float> q(nv), k(nv), v(nv), w(nv), ut(nv), u(nv), y(nv), wk(nv), g(ng), beta(ng), lower(ng * 64), qk(ng * 64);
  GdnPrefill a{input, logits.p, q.p, k.p, v.p, w.p, ut.p, u.p, y.p, wk.p, g.p, beta.p, lower.p, qk.p, chunks};
  PackedBuffer pq(nv), pk(nv), pi(ng * 64), pqk(ng * 64), pwk(nv), pv(nv), pw(nv), pu(nv), pe(nv), ps(state.n);
  PackedGdn packed{a, pq.Get(), pk.Get(), pi.Get(), pqk.Get(), pwk.Get(), pv.Get(), pw.Get(), pu.Get(), pe.Get(), ps.Get()};
  const bool use_packed = std::getenv("PREFILL_PACKED") != nullptr;
  const bool use_micro = std::getenv("PREFILL_MICRO") != nullptr;
  const bool persistent = std::getenv("PREFILL_PERSISTENT") != nullptr;
  Buffer<float> chunk_y(48 * 64 * 128);
  auto chunk = [&] {
    if (persistent && std::getenv("PREFILL_LANES")) LaunchGdnLanesRef(a);
    else if (persistent) LaunchGdnPersistent(a);
    else if (use_micro) LaunchGdnMicro(a, chunk_y.p, true);
    else if (use_packed) LaunchGdnPacked(packed);
    else LaunchGdnPrefill(a);
  };
  chunk();
  CheckLaunch();
  const auto got_state = state.Get();
  const auto got_raw = y.Get();
  const auto got_out = output.Get();
  state.Set(initial); hist.Zero();
  auto serial = [&] {
    for (int r = 0; r < rows; ++r) {
      GdnArgs step = input;
      step.qkv += (size_t)r * 10240; step.z += (size_t)r * 6144; step.x += (size_t)r * kHidden;
      step.xh_out += (size_t)r * 6144; step.pos0 = r; step.R = step.commit = 1;
      GdnKernel<<<48, 256>>>(step);
    }
  };
  serial();
  CheckLaunch();
  const auto ref_state = state.Get();
  const auto ref_out = output.Get();
  const auto se = Compare(got_state, ref_state), oe = Compare(got_out, ref_out);
  state.Set(initial); hist.Zero();
  GdnPrefillPrepKernel<<<dim3(48, chunks * 64), 256>>>(a);
  SerialPreparedKernel<<<48, 256>>>(a);
  CheckLaunch();
  const auto prepared_state = state.Get();
  if (std::memcmp(prepared_state.data(), ref_state.data(), ref_state.size() * sizeof(float)) != 0) {
    std::fprintf(stderr, "prepared reference differs from GdnKernel state\n"); std::exit(4);
  }
  const auto raw_error = Compare(got_raw, y.Get());
  double max_row_l2 = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<__half> ga(got_out.begin() + r * 6144, got_out.begin() + (r + 1) * 6144);
    std::vector<__half> re(ref_out.begin() + r * 6144, ref_out.begin() + (r + 1) * 6144);
    max_row_l2 = std::max(max_row_l2, Compare(ga, re).rel_l2);
  }
  const double serial_us = Time([&] { state.Zero(); hist.Zero(); serial(); }, 3);
  const double chunk_us = Time([&] { state.Zero(); hist.Zero(); chunk(); }, 5);
  EnergyHook("gdn", "chunk64_prefill", chunk);
  if (std::getenv("PREFILL_PERSISTENT_REF")) {
    const double prepared_us=Time([&] {
      state.Zero();hist.Zero();
      GdnPrefillPrepKernel<<<dim3(48,chunks*64),256>>>(a);
      SerialPreparedKernel<<<48,256>>>(a);
      GdnPrefillGateKernel<<<dim3(48,rows),256>>>(a);
    },3);
    std::printf("persistent_prepared rows=%d us=%.3f\n",rows,prepared_us);
  }
  if (persistent && std::getenv("PREFILL_PROFILE")) {
    auto time = [&](const char* name, auto launch) { std::printf("stage %s us=%.3f\n", name, Time(launch)); };
    time("p_prep", [&] { GdnPrefillPrepKernel<<<dim3(48, chunks * 64), 256>>>(a); });
    time("p_column", [&] { state.Zero(); GdnPersistentColumnKernel<<<48, 128>>>(a); });
    time("l_prep", [&] { GdnLanesPrepKernel<<<dim3(16 + 24, chunks * 64), 256>>>(a); });
    time("p_lanes", [&] { state.Zero(); GdnScanLanesKernel<false><<<48 * 4, 128>>>(a); });
    time("p_lanes_lazy", [&] { state.Zero(); GdnScanLanesKernel<true><<<48 * 4, 128>>>(a); });
    time("l_gate", [&] { GdnLanesGateKernel<<<(rows * 48 + 7) / 8, 256>>>(a); });
    time("p_gate", [&] { GdnPrefillGateKernel<<<dim3(48, rows), 256>>>(a); });
    time("p_history", [&] { GdnPrefillHistoryKernel<<<40, 256>>>(a); });
  } else if (std::getenv("PREFILL_PROFILE")) {
    auto time = [&](const char* name, auto launch) { std::printf("stage %s us=%.3f\n", name, Time(launch)); };
    time("prep", [&] { GdnPrefillPrepKernel<<<dim3(48, rows), 256>>>(a); });
    time("prefix", [&] { GdnPrefillPrefixKernel<<<dim3(48, chunks), 128>>>(a); });
    time("gram", [&] { GdnPrefillGramKernel<<<dim3(4, chunks, 48), 128>>>(a); });
    time("inverse", [&] { GdnPrefillInverseKernel<<<dim3(48, chunks), 64>>>(a); });
    time("wy", [&] { GdnPrefillWyKernel<<<dim3(2, 4, 48 * chunks * 2), 128>>>(a); });
    time("carry_all", [&] { for (int c = 0; c < chunks; ++c) GdnPrefillCarryKernel<<<dim3(2, 4, 48), 128>>>(a, c); });
    time("output_all", [&] { for (int c = 0; c < chunks; ++c) GdnPrefillOutputKernel<<<dim3(2, 4, 48), 128>>>(a, c); });
    time("state_all", [&] { for (int c = 0; c < chunks; ++c) GdnPrefillStateKernel<<<dim3(2, 8, 48), 128>>>(a, c); });
    time("gate", [&] { GdnPrefillGateKernel<<<dim3(48, rows), 256>>>(a); });
  }
  std::printf("gdn packed=%d micro=%d rows=%d chunk_us=%.3f serial_us=%.3f speedup=%.2f out_relL2=%.3g out_max_row_relL2=%.3g "
    "out_max_scaled=%.3g out_max_relative=%.3g raw_relL2=%.3g raw_max_relative=%.3g "
    "state_relL2=%.3g state_max_scaled=%.3g state_max_relative=%.3g state_max_abs=%.3g\n",
    use_packed, use_micro, rows, chunk_us, serial_us, serial_us / chunk_us, oe.rel_l2, max_row_l2, oe.max_scaled,
    oe.max_relative, raw_error.rel_l2, raw_error.max_relative, se.rel_l2, se.max_scaled, se.max_relative, se.max_abs);
  if (!fixture.empty()) {
    const auto& raw = got_raw;
    std::vector<float> ordered((size_t)rows * 6144);
    for (int r = 0; r < rows; ++r) for (int h = 0; h < 48; ++h) for (int j = 0; j < 128; ++j)
      ordered[(size_t)r * 6144 + h * 128 + j] = raw[((size_t)h * chunks * 64 + r) * 128 + j];
    const auto oracle_y = ReadFile<double>(fixture + "/oracle_y.f64", ordered.size());
    const auto oracle_state = ReadFile<double>(fixture + "/oracle_state.f64", got_state.size());
    const auto ye = Compare(ordered, oracle_y), ste = Compare(got_state, oracle_state);
    const auto serial_error = Compare(ref_state, oracle_state);
    std::printf("oracle fixture=%s rows=%d raw_y_relL2=%.3g raw_y_max_scaled=%.3g raw_y_max_relative=%.3g "
      "state_relL2=%.3g state_max_scaled=%.3g state_max_relative=%.3g serial_state_relL2=%.3g\n",
      fixture.c_str(), rows, ye.rel_l2, ye.max_scaled, ye.max_relative, ste.rel_l2, ste.max_scaled,
      ste.max_relative, serial_error.rel_l2);
    if (ye.rel_l2 > 3e-6 || ste.rel_l2 > 3e-6) std::exit(5);
  }
  if (se.rel_l2 > 3e-6 || raw_error.rel_l2 > 3e-6 || oe.rel_l2 > 3e-4) std::exit(3);
}

int main(int argc, char** argv) {
  if (argc > 1 && (std::stoi(argv[1]) < 1 || std::stoi(argv[1]) > 2048)) {
    std::fprintf(stderr, "rows must be in 1..2048\n");
    return 2;
  }
  if (argc == 3) Run(std::stoi(argv[1]), argv[2]);
  else if (argc == 2) Run(std::stoi(argv[1]), "");
  else for (int rows : {256, 512, 1024, 2048}) Run(rows, "");
}
