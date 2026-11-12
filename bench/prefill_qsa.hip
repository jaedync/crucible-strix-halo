// Dense/sparse QSA prefill versus the engine's row-wise attention/reference.
#include "prefill_common.h"
#include "../engine/layers/kernels/prefill_qsa_union.h"
#include "kernels/qsa_decode.h"
#include "kernels/prefill_qsa_pack.h"
#include "qsa_union_ko.h"
using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;

double Oracle(QsaArgs a, const std::vector<float>& context) {
  std::vector<float> q((size_t)a.R * 12288);
  std::vector<__half> k((size_t)(a.pos0 + a.R) * 512), v(k.size());
  std::vector<int> selected(a.R * 512), counts(a.R);
  HIP_OK(hipMemcpy(q.data(), a.q, q.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(k.data(), a.c.k, k.size() * 2, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(v.data(), a.c.v, v.size() * 2, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(selected.data(), a.sel, selected.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(counts.data(), a.nsel, counts.size() * 4, hipMemcpyDeviceToHost));
  double maximum = 0;
  for (int r : {0, a.R / 2, a.R - 1}) for (int head : {0, 11, 12, 23}) {
    const int end = a.pos0 + r + 1, count = counts[r] < 0 ? end : counts[r] * 4 + end % 4;
    std::vector<int> positions(count);
    std::vector<double> logits(count), reference(256, 0.);
    double m = -INFINITY, denominator = 0.;
    for (int i = 0; i < count; ++i) {
      const int pos = counts[r] < 0 ? i : i < counts[r] * 4 ? selected[r * 512 + i / 4] * 4 + i % 4
          : end / 4 * 4 + i - counts[r] * 4;
      positions[i] = pos;
      double dot = 0.;
      for (int d = 0; d < 256; ++d)
        dot += q[((size_t)r * 24 + head) * 512 + d] * Value(k[((size_t)pos * 2 + head / 12) * 256 + d]);
      logits[i] = dot / 16.;
      m = std::max(m, logits[i]);
    }
    for (int i = 0; i < count; ++i) {
      const double p = std::exp(logits[i] - m);
      denominator += p;
      for (int d = 0; d < 256; ++d)
        reference[d] += p * Value(v[((size_t)positions[i] * 2 + head / 12) * 256 + d]);
    }
    for (auto& value : reference) value /= denominator;
    std::vector<float> got(context.begin() + ((size_t)r * 24 + head) * 256,
        context.begin() + ((size_t)r * 24 + head + 1) * 256);
    maximum = std::max(maximum, Compare(got, reference).rel_l2);
  }
  return maximum;
}

void Run(int rows, int past) {
  const int ctx = past + rows, blocks = ctx / 4;
  const auto query = Random((size_t)rows * 12288, .4f, 31);
  const auto index = Random((size_t)rows * 640, .4f, 32);
  Buffer<float> q(query), idx(index), key(Random((size_t)rows * 512, .4f, 33));
  Buffer<float> value(Random((size_t)rows * 512, .4f, 34));
  Buffer<float> norm256(std::vector<float>(256, 1.f)), norm128(std::vector<float>(128, 1.f));
  Buffer<__half> ck(Half(Random((size_t)ctx * 512, 1.f, 35)));
  Buffer<__half> cv(Half(Random((size_t)ctx * 512, .4f, 36)));
  Buffer<float> raw(Random((size_t)ctx * 128, .4f, 37)), pool(Random(PrefillPoolFloats(ctx), 1.f, 38));
  Buffer<float> scores((size_t)rows * std::max(1, blocks));
  Buffer<int> sel(rows * 512), nsel(rows);
  Buffer<__half> suh(std::vector<__half>(6144, __float2half(1.f))), output((size_t)rows * 6144);
  Buffer<float> context((size_t)rows * 6144);
  Buffer<__half> qhi((size_t)rows * 6144), qlo((size_t)rows * 6144);
  QsaArgs a{q.p, key.p, value.p, idx.p, norm256.p, norm256.p, norm128.p, norm128.p,
    {ck.p, cv.p, raw.p, pool.p}, scores.p, std::max(1, blocks), sel.p, nsel.p,
    suh.p, output.p, past, rows};
  a.rope = PlainRopeTable();
  QsaPrefill f{a, context.p, qhi.p, qlo.p};
  QsaPrepKernel<<<dim3(33, rows), 256>>>(a);
  QsaPoolKernel<<<1, 256>>>(a);
  CheckLaunch();
  const auto serial_pool = pool.Get();
  const int completed = blocks - past / 4;
  if (completed) QsaPrefillPoolKernel<<<completed, 256>>>(a);
  CheckLaunch();
  const auto pooled = pool.Get();
  if (serial_pool != pooled) { std::fprintf(stderr, "pool mismatch\n"); std::exit(3); }
  // The retained serial selector always uses its original row-major layout.
  std::vector<float> row_major_pool((size_t)std::max(1, blocks) * 128);
  for (int b = 0; b < blocks; ++b) for (int d = 0; d < 128; ++d)
    row_major_pool[(size_t)b * 128 + d] = pooled[PrefillPoolOffset(b, d)];
  Buffer<float> legacy_pool(row_major_pool);
  QsaArgs legacy = a;
  legacy.c.ik_pool = legacy_pool.p;
  QsaSelectKernel<<<rows, 1024>>>(legacy);
  QsaAttnKernel<<<dim3(24, rows), 256>>>(a);
  CheckLaunch();
  const auto ref_sel = sel.Get(), ref_ns = nsel.Get();
  const auto ref_out = output.Get();
  const auto ref_scores = scores.Get();
  if (blocks > 512) QsaScoreKernel<<<dim3((blocks+255)/256,rows),256>>>(a);
  QsaTopKernel<<<rows,1024>>>(a);
  QsaPrefillPackKernel<<<rows * 24, 256>>>(f);
  LaunchQsaFlash(f);
  QsaPrefillGateKernel<<<dim3(24, rows), 256>>>(f);
  CheckLaunch();
  if (nsel.Get() != ref_ns || sel.Get() != ref_sel) {
    std::fprintf(stderr, "selection mismatch rows=%d past=%d\n", rows, past);
    std::exit(4);
  }
  const auto got = output.Get();
  const auto sc = scores.Get();
  std::vector<float> checked_scores, checked_reference;
  for (int r = 0; r < rows; ++r) if ((past + r + 1) / 4 > 512)
    for (int b = 0; b < (past + r + 1) / 4; ++b) {
      checked_scores.push_back(sc[(size_t)r * a.sstride + b]);
      checked_reference.push_back(ref_scores[(size_t)r * a.sstride + b]);
    }
  const auto err = Compare(got, ref_out), score_err = Compare(checked_scores, checked_reference);
  const double oracle_error = Oracle(a, context.Get());
  const double prep_us = Time([&] { QsaPrepKernel<<<dim3(33, rows), 256>>>(a); });
  q.Set(query); idx.Set(index);
  QsaPrepKernel<<<dim3(33, rows), 256>>>(a);
  CheckLaunch();
  const double pool_us = completed ? Time([&] { QsaPrefillPoolKernel<<<completed, 256>>>(a); }) : 0;
  const double score_us = blocks > 512 ? Time([&] { QsaScoreKernel<<<dim3((blocks+255)/256,rows),256>>>(a); }) : 0;
  const double select_us = Time([&] { QsaTopKernel<<<rows,1024>>>(a); });
  const double flash_us = Time([&] {
    QsaPrefillPackKernel<<<rows * 24, 256>>>(f);
    LaunchQsaFlash(f);
    QsaPrefillGateKernel<<<dim3(24, rows), 256>>>(f);
  });
  const double serial_us = Time([&] { QsaAttnKernel<<<dim3(24, rows), 256>>>(a); }, 3);
  {
    QsaPrefillPackKernel<<<rows * 24, 256>>>(f);
    LaunchQsaFlash(f);
    CheckLaunch();
    const auto base_context = context.Get();
    Buffer<int> ulist(QsaUnionListInts(rows)), ucount((rows + kUnionRows - 1) / kUnionRows);
    QsaUnion un{ulist.p, ucount.p};
    context.Zero();
    LaunchQsaUnionFlash(f, un);
    CheckLaunch();
    const auto ue = Compare(context.Get(), base_context);
    const double uo = Oracle(a, context.Get());
    const auto counts = ucount.Get();
    double mean = 0; for (int c : counts) mean += c; mean /= counts.size();
    const double union_us = Time([&] { LaunchQsaUnionFlash(f, un); });
    const double base_us = Time([&] { LaunchQsaFlash(f); });
    EnergyHook("attention", "qsa_union_flash", [&] { LaunchQsaUnionFlash(f, un); });
    std::printf("union rows=%d past=%d union_us=%.3f flash_only_us=%.3f mean_union_blocks=%.1f ctx_relL2=%.3g oracle=%.3g\n",
      rows, past, union_us, base_us, mean, ue.rel_l2, uo);
    if (ue.rel_l2 > 1e-5 || uo > 5e-6) std::exit(8);
    if (std::getenv("QSA_KO")) {
      // Knockouts of the no-lo union kernel; KO 0 must match production bitwise.
      QsaPrefill h = f; h.q_lo = nullptr;
      QsaPrefillPackKernel<<<rows * 24, 256>>>(h);
      const int tiles = (rows + kUnionRows - 1) / kUnionRows;
      auto prod = [&] { QsaUnionFlashKernel<false><<<dim3(tiles, 2), 256>>>(h, un); };
      context.Zero(); prod(); CheckLaunch();
      const auto c0 = context.Get();
      std::printf("ko rows=%d past=%d prod_us=%.1f\n", rows, past, Time(prod));
      auto run = [&](auto ko) {
        constexpr int K = decltype(ko)::value;
        auto go = [&] { QsaUnionKoKernel<K><<<dim3(tiles, 2), 256>>>(h, un); };
        context.Zero(); go(); CheckLaunch();
        const bool same = context.Get() == c0;
        std::printf("ko rows=%d past=%d ko=%d us=%.1f bitwise=%d\n", rows, past, K, Time(go), (int)same);
      };
      auto var = [&](auto v) {
        constexpr int V = decltype(v)::value;
        auto go = [&] { QsaUnionVarKernel<V><<<dim3(tiles, 2), 256>>>(h, un); };
        context.Zero(); go(); CheckLaunch();
        const bool same = context.Get() == c0;
        std::printf("var rows=%d past=%d var=%d us=%.1f prod_us=%.1f bitwise=%d\n", rows, past, V, Time(go), Time(prod), (int)same);
      };
      auto kl = [&](auto v) {
        constexpr int V = decltype(v)::value;
        auto go = [&] { QsaUnionKldsKernel<V><<<dim3(tiles, 2), 256>>>(h, un); };
        context.Zero(); go(); CheckLaunch();
        const bool same = context.Get() == c0;
        const double a0 = Time(prod), a1 = Time(go), b1 = Time(go), b0 = Time(prod);
        std::printf("klds rows=%d past=%d var=%d us=%.1f prod_us=%.1f ratio=%.3f bitwise=%d\n", rows, past, V,
          (a1 + b1) / 2, (a0 + b0) / 2, (a1 + b1) / (a0 + b0), (int)same);
      };
      kl(std::integral_constant<int, 0>{});
      kl(std::integral_constant<int, 1>{});
      kl(std::integral_constant<int, 2>{});
      kl(std::integral_constant<int, 3>{});
      if (std::getenv("QSA_KO")[0] == '3') return;
      var(std::integral_constant<int, 0>{});
      var(std::integral_constant<int, kVarList>{});
      var(std::integral_constant<int, kVarVWaves>{});
      var(std::integral_constant<int, kVarList | kVarVWaves>{});
      var(std::integral_constant<int, kVarQk2>{});
      var(std::integral_constant<int, kVarList | kVarQk2>{});
      var(std::integral_constant<int, kVarList | kVarVWaves | kVarVb4>{});
      var(std::integral_constant<int, kVarVWaves | kVarVb4>{});
      var(std::integral_constant<int, kVarList | kVarQk2 | kVarVb4 | kVarU2>{});
      var(std::integral_constant<int, kVarList | kVarU2>{});
      if (std::getenv("QSA_KO")[0] == '2') return;
      run(std::integral_constant<int, 0>{});
      run(std::integral_constant<int, kKoVLoad>{});
      run(std::integral_constant<int, kKoVAll>{});
      run(std::integral_constant<int, kKoKLoad>{});
      run(std::integral_constant<int, kKoQLoad>{});
      run(std::integral_constant<int, kKoKLoad | kKoQLoad>{});
      run(std::integral_constant<int, kKoKLoad | kKoQLoad | kKoVLoad>{});
      run(std::integral_constant<int, kKoQk>{});
      run(std::integral_constant<int, kKoPv>{});
      run(std::integral_constant<int, kKoExp>{});
      run(std::integral_constant<int, kKoQk | kKoPv>{});
      run(std::integral_constant<int, kKoQk | kKoPv | kKoVAll | kKoExp>{});
    }
  }
  std::printf("qsa rows=%d past=%d prep_us=%.3f pool_us=%.3f score_us=%.3f select_us=%.3f flash_us=%.3f serial_attn_us=%.3f "
    "speedup=%.2f out_relL2=%.3g out_max_scaled=%.3g score_relL2=%.3g oracle_raw_max_relL2=%.3g selection_exact=1 pool_exact=1\n",
    rows, past, prep_us, pool_us, score_us, select_us, flash_us, serial_us, serial_us / flash_us,
    err.rel_l2, err.max_scaled, score_err.rel_l2, oracle_error);
  if (err.rel_l2 > 3e-4 || oracle_error > 5e-6) std::exit(5);
  // Full tied scores exercise lower-ID selection and the three-token tail.
  if (blocks > 512) {
    scores.Zero();
    QsaTopKernel<<<rows,1024>>>(a);
    CheckLaunch();
    const auto tied = sel.Get();
    for (int r = 0; r < rows; ++r) if ((past + r + 1) / 4 > 512)
      for (int b = 0; b < 512; ++b) if (tied[r * 512 + b] != b) std::exit(6);
    LaunchQsaFlash(f);
    CheckLaunch();
    if (Oracle(a, context.Get()) > 5e-6) std::exit(7);
  }
}

int main(int argc, char** argv) {
  if (argc >= 2 && (std::stoi(argv[1]) < 1 || std::stoi(argv[1]) > 2048)) {
    std::fprintf(stderr, "rows must be in 1..2048\n");
    return 2;
  }
  if (argc >= 3 && (std::stoi(argv[2]) < 0 || std::stoi(argv[2]) > 262144)) {
    std::fprintf(stderr, "past must be in 0..262144\n");
    return 2;
  }
  if (argc >= 2) Run(std::stoi(argv[1]), argc >= 3 ? std::stoi(argv[2]) : 0);
  else for (int rows : {256, 512, 1024, 2048}) { Run(rows, 0); Run(rows, 4096); }
}
