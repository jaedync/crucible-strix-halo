// the GDN prefill scan with the carried lazy scale.
// 1. Cold: N rows from position 0 in 2,048-row parts. The engine scan (anchor 2,048, then GdnMaterializeKernel)
//    against the scan of the earlier engine (bench/kernels/prefill_gdn_trunk_scan.h): y of every row, the gated
//    out-projection input, the history ring and the final state must be equal bit for bit.
// 2. Split: rows [0, s), then rows [s, N) as a new launch (parts of 2,048 rows from s). The same four outputs
//    must be equal to the cold run bit for bit, at every s.
// 3. ABAB event timing of one 2,048-row part: the trunk scan against the engine scan.
// Usage: gdn_split [--pairs N] [N s ...]. A build with -DGDN_SPLIT_MUTANT runs the split with anchor 0 (a reset at
// each launch start, as trunk): check 2 must FAIL.
#include "prefill_common.h"
#include "kernels/prefill_gdn_trunk_scan.h"
#include <cstring>
#include <string>
using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;
namespace {
constexpr int kCore = 2048;

std::vector<float> HeadDecays() {
  // Per-head A_log from -5 to 1.5: slow heads keep the lazy scale above 2^-20 for a whole part (so the anchor
  // decides), fast heads renormalize every few rows.
  std::vector<float> x(48);
  for (int h = 0; h < 48; ++h) x[h] = -5.f + 6.5f * h / 47.f;
  return x;
}

struct Inputs {
  int N;
  Buffer<float> qkv, z, ab, conv, A_log, dt_bias, gamma, hist, hist0, state, state0, carry;
  Buffer<__half> suh, xh;
  Buffer<float> q, k, v, y, g, beta, Y;
  explicit Inputs(int rows)
      : N(rows), qkv(Random((size_t)rows * kGdnQkvWidth, 1.f, 1)), z(Random((size_t)rows * kGdnValueWidth, 1.f, 2)),
        ab(Random((size_t)rows * 96, 1.f, 3)), conv(Random(kGdnQkvWidth * 4, .5f, 4)), A_log(HeadDecays()),
        dt_bias(Random(48, .5f, 6)), gamma(Random(128, .1f, 7)), hist(16 * kGdnQkvWidth),
        hist0(Random(16 * kGdnQkvWidth, 1.f, 8)), state(kGdnStateFloats + kGdnScaleFloats), state0(Init()), carry(48),
        suh(Half(Random(kGdnValueWidth, 1.f, 10))), xh((size_t)rows * kGdnValueWidth), q(16 * (size_t)kCore * 128),
        k(16 * (size_t)kCore * 128), v(48 * (size_t)kCore * 128), y(48 * (size_t)kCore * 128), g(48 * kCore),
        beta(48 * kCore), Y((size_t)rows * kGdnValueWidth) {}
  static std::vector<float> Init() {
    auto x = Random(kGdnStateFloats + kGdnScaleFloats, .01f, 9);
    std::fill(x.begin() + kGdnStateFloats, x.end(), 0.f);  // a zeroed scale means 1
    return x;
  }
  void Reset() {
    HIP_OK(hipMemcpy(state.p, state0.p, state.n * 4, hipMemcpyDeviceToDevice));
    HIP_OK(hipMemcpy(hist.p, hist0.p, hist.n * 4, hipMemcpyDeviceToDevice));
    xh.Nan(); Y.Nan(); y.Zero();
  }
  GdnPrefill At(int pos0, int n) {
    GdnPrefill a{};
    a.input = {qkv.p + (size_t)pos0 * kGdnQkvWidth, z.p + (size_t)pos0 * kGdnValueWidth, nullptr, nullptr, conv.p, A_log.p,
               dt_bias.p, gamma.p, hist.p, state.p, suh.p, xh.p + (size_t)pos0 * kGdnValueWidth, nullptr, pos0, n, 0, n};
    a.ab_logits = ab.p + (size_t)pos0 * 96; a.chunks = (n + 63) / 64;
    a.q = q.p; a.k = k.p; a.v = v.p; a.y = y.p; a.g = g.p; a.beta = beta.p;
    a.carry = carry.p;
    return a;
  }
};

// y of a part ([48][chunks*64][128]) into Y[pos][48*128].
__global__ void CopyY(GdnPrefill a, float* Y) {
  const int r = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
  Y[((size_t)a.input.pos0 + r) * kGdnValueWidth + h * 128 + j] = a.y[((size_t)h * a.chunks * 64 + r) * 128 + j];
}

enum class Scan { Trunk, Engine, Mutant };
// One launch of rows [pos0, pos0 + rows), cut in 2,048-row parts from its start (engine/runtime/prefill_engine.inc).
void Launch(Inputs& in, int pos0, int rows, Scan scan) {
  for (int o = 0; o < rows; o += kCore) {
    const int n = std::min(kCore, rows - o);
    GdnPrefill a = in.At(pos0 + o, n);
    if (scan == Scan::Trunk) {
      a.anchor = 0;
      GdnLanesPrepQkKernel<<<dim3(16 + 1, a.chunks * 64 / kGdnPrepRows), 256>>>(a);
      GdnScanSgprTrunkKernel<<<48 * 4, 128>>>(a, a.k);
      GdnLanesGateKernel<true><<<(n + 7) / 8 * 48, 256>>>(a);
      GdnPrefillHistoryKernel<<<40, 256>>>(a);
    } else {
      a.anchor = scan == Scan::Mutant ? 0 : kCore;
      LaunchGdnLanes(a, nullptr, true);
    }
    CopyY<<<dim3(n, 48), 128>>>(a, in.Y.p);
  }
}

struct Result { std::vector<float> state, hist, Y; std::vector<__half> xh; };
Result Run(Inputs& in, int s, Scan scan) {
  in.Reset();
  if (s > 0) Launch(in, 0, s, scan);
  Launch(in, s, in.N - s, scan);
  if (scan != Scan::Trunk) LaunchGdnMaterialize({in.state.p});
  CheckLaunch();
  auto st = in.state.Get();
  for (int h = 0; h < 48; ++h) {
    const float c = st[kGdnStateFloats + h];
    if (scan != Scan::Trunk && c != 0.f && c != 1.f) { std::printf("scale %d not materialized: %g\n", h, c); std::abort(); }
  }
  st.resize(kGdnStateFloats);
  return {st, in.hist.Get(), in.Y.Get(), in.xh.Get()};
}
template <class T> bool Same(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
// The first position whose y differs, or -1.
int FirstRow(const Result& a, const Result& b) {
  for (size_t i = 0; i < a.Y.size(); ++i)
    if (std::memcmp(&a.Y[i], &b.Y[i], 4)) return int(i / kGdnValueWidth);
  return -1;
}
bool Report(const char* what, int N, int s, const Result& a, const Result& b) {
  const bool ok = Same(a.state, b.state) && Same(a.hist, b.hist) && Same(a.Y, b.Y) && Same(a.xh, b.xh);
  std::printf("%s N=%d s=%d state=%s hist=%s y=%s xh=%s first_y_row=%d -> %s\n", what, N, s,
              Same(a.state, b.state) ? "eq" : "DIFF", Same(a.hist, b.hist) ? "eq" : "DIFF",
              Same(a.Y, b.Y) ? "eq" : "DIFF", Same(a.xh, b.xh) ? "eq" : "DIFF", FirstRow(a, b), ok ? "EQUAL" : "DIFFER");
  return ok;
}
}  // namespace

int main(int argc, char** argv) {
  int pairs = 7;
  std::vector<int> args;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--pairs" && i + 1 < argc) pairs = std::stoi(argv[++i]);
    else args.push_back(std::stoi(s));
  }
  // The exact-ext set (tools/exact_extend.py), then edge cuts.
  std::vector<std::pair<int, std::vector<int>>> cases = {
      {8900, {512, 4096, 8192, 8300, 8899, 5001, 1, 3, 63, 64, 65, 2047, 2049, 6143, 6145}},
      {4100, {2048, 2047, 4099}},
      {34816, {32768, 30001}}};
  if (!args.empty()) cases = {{args[0], std::vector<int>(args.begin() + 1, args.end())}};
  bool cold_ok = true, split_ok = true;
  for (const auto& [N, splits] : cases) {
    Inputs in(N);
    const Result trunk = Run(in, 0, Scan::Trunk), cold = Run(in, 0, Scan::Engine);
    cold_ok = Report("cold-vs-trunk", N, 0, cold, trunk) && cold_ok;
    for (int s : splits) {
      if (s <= 0 || s >= N) return 2;
#ifdef GDN_SPLIT_MUTANT
      const Result split = Run(in, s, Scan::Mutant);
#else
      const Result split = Run(in, s, Scan::Engine);
#endif
      split_ok = Report("split-vs-cold", N, s, split, cold) && split_ok;
    }
    std::fflush(stdout);
  }
  {
    Inputs in(kCore);
    in.Reset();
    GdnPrefill part = in.At(0, kCore);
    GdnLanesPrepQkKernel<<<dim3(16 + 1, kCore / kGdnPrepRows), 256>>>(part);
    CheckLaunch();
    GdnPrefill eng = part;
    eng.anchor = kCore;
    std::vector<double> t, e;
    for (int rep = 0; rep < pairs; ++rep) {
      t.push_back(Time([&] { GdnScanSgprTrunkKernel<<<48 * 4, 128>>>(part, part.k); }, 5));
      e.push_back(Time([&] { GdnScanSgprKernel<<<48 * 4, 128>>>(eng, eng.k); }, 5));
    }
    const double a = bench::Median(t), b = bench::Median(e);
    std::printf("scan2048 trunk_us=%.1f engine_us=%.1f ratio=%.4f ms_per_32K_36L=%+.2f pairs=%d\n", a, b, b / a,
                (b - a) * 36 * 16 / 1000., pairs);
  }
  std::printf("cold=%s split=%s\n%s\n", cold_ok ? "PASS" : "FAIL", split_ok ? "PASS" : "FAIL",
              cold_ok && split_ok ? "PASS" : "FAIL");
  return cold_ok && split_ok ? 0 : 1;
}
