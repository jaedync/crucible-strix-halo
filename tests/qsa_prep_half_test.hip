// QsaPrefillQueryKernel (the only prefill query path since train 20) must
// write the q_hi and q_lo bits of the old path: the query units of QsaPrepKernel
// (norm and RoPE in place), then QsaPrefillPackKernel (bench/kernels) on the
// output rows [skip, R).
// Build: hipcc -O3 -std=c++20 --offload-arch=gfx1151 -o build/qsa_prep_half_test tests/qsa_prep_half_test.hip
#include "../bench/prefill_common.h"
#include "../bench/kernels/prefill_qsa_pack.h"
#include <cstring>

using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

namespace {
template <class T> bool Same(const std::vector<T>& x, const std::vector<T>& y) {
  return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}

void Check(int pos0, int R, int skip, bool lo) {
  const std::vector<float> q0 = Random((size_t)R * 24 * 512, 2.f, 51 + R);
  std::vector<float> norm = Random(256, .2f, 52);
  for (auto& w : norm) w += 1.f;
  Buffer<float> qa(q0), qb(q0), w(norm);
  const size_t n = (size_t)(R - skip) * 6144;
  Buffer<__half> hia(n), loa(n), hib(n), lob(n);
  QsaArgs a{};
  a.q = qa.p; a.q_norm1 = w.p; a.pos0 = pos0; a.R = R; a.rope = PlainRopeTable();
  // Old path (before train 20): query units only (grid x 24), then the pack on the output rows.
  QsaPrepKernel<<<dim3(24, R), 256>>>(a);
  QsaPrefill fa{QsaOutputRows(a, skip), nullptr, hia.p, lo ? loa.p : nullptr};
  QsaPrefillPackKernel<<<(R - skip) * 24, 256>>>(fa);
  CheckLaunch();
  QsaArgs b = a;
  b.q = qb.p;
  QsaPrefill fb{b, nullptr, hib.p, lo ? lob.p : nullptr};
  const auto run = [&](int hpb) {
    hib.Zero(); lob.Zero();
    if (hpb == 1) QsaPrefillQueryKernel<1><<<dim3(24, R - skip), 256>>>(fb, skip);
    if (hpb == 2) QsaPrefillQueryKernel<2><<<dim3(12, R - skip), 256>>>(fb, skip);
    if (hpb == 4) QsaPrefillQueryKernel<4><<<dim3(6, R - skip), 256>>>(fb, skip);
    if (hpb == 6) QsaPrefillQueryKernel<6><<<dim3(4, R - skip), 256>>>(fb, skip);
    CheckLaunch();
  };
  for (const int hpb : {1, 2, 4, 6}) {
    run(hpb);
    const bool hi_same = Same(hia.Get(), hib.Get());
    const bool lo_same = !lo || Same(loa.Get(), lob.Get());
    if (!hi_same || !lo_same) {
      std::fprintf(stderr, "FAIL: pos0=%d R=%d skip=%d lo=%d hpb=%d hi=%d lo=%d\n", pos0, R, skip, lo, hpb,
                   hi_same, lo_same);
      std::exit(1);
    }
  }
  std::printf("PASS: prep half pos0=%d R=%d skip=%d lo=%d\n", pos0, R, skip, lo);
}
}  // namespace

int main() {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) return 2;
  for (const bool lo : {false, true}) {
    Check(0, 1, 0, lo);
    Check(5, 37, 16, lo);
    Check(4096, 2048, 0, lo);
    Check(131000, 2048, 1024, lo);
  }
  std::printf("PASS: qsa_prep_half_test\n");
}
