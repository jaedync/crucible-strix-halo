// Time of the QSA prefill query preparation: the old path before train 20
// (query units of QsaPrepKernel, norm and RoPE in place, then QsaPrefillPackKernel
// from bench/kernels/prefill_qsa_pack.h) against QsaPrefillQueryKernel<HPB>
// (the default uses HPB 2), 2,048 rows, ABBA.
// Bits: tests/qsa_prep_half_test.hip. Usage: qsa_prep_half [POS0]
#include "prefill_common.h"
#include "kernels/prefill_qsa_pack.h"
#include <string>

using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

int main(int argc, char** argv) {
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0) return 2;
  const int R = 2048, pos0 = argc > 1 ? std::stoi(argv[1]) : 30720;
  std::vector<float> norm = Random(256, .2f, 52);
  for (auto& w : norm) w += 1.f;
  Buffer<float> q(Random((size_t)R * 24 * 512, 2.f, 51)), w(norm);
  Buffer<__half> hi((size_t)R * 6144);
  QsaArgs a{};
  a.q = q.p; a.q_norm1 = w.p; a.pos0 = pos0; a.R = R; a.rope = PlainRopeTable();
  const QsaPrefill f{a, nullptr, hi.p, nullptr};
  const auto base = [&] {
    QsaPrepKernel<<<dim3(24, R), 256>>>(a);
    QsaPrefillPackKernel<<<R * 24, 256>>>(f);
  };
  const auto report = [&](const char* name, auto launch) {
    const double b0 = Time(base), v0 = Time(launch), v1 = Time(launch), b1 = Time(base);
    std::printf("%-6s base_us=%.1f var_us=%.1f ratio=%.3f\n", name, (b0 + b1) / 2, (v0 + v1) / 2,
                (v0 + v1) / (b0 + b1));
  };
  report("hpb1", [&] { QsaPrefillQueryKernel<1><<<dim3(24, R), 256>>>(f, 0); });
  report("hpb2", [&] { QsaPrefillQueryKernel<2><<<dim3(12, R), 256>>>(f, 0); });
  report("hpb4", [&] { QsaPrefillQueryKernel<4><<<dim3(6, R), 256>>>(f, 0); });
  report("hpb6", [&] { QsaPrefillQueryKernel<6><<<dim3(4, R), 256>>>(f, 0); });
  CheckLaunch();
}
