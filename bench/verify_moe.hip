// verify-moe: the verify-step MoE (R = 1..8 rows) on real routes, against its byte floor.
//
//   verify_moe ROUTES [layers] [iters] [rmin] [rmax] [variants] [stamp dump file]
//
// ROUTES: a CRUCIBLE_ROUTE_DUMP file; its R = 1 decode records give the routes of consecutive tokens, so a window
// of R tokens is the expert set of a verify step that accepts them. Weights are random EXL3 tiles at the 4.05
// widths (routed K4, shared K6) for `layers` layers of 512 experts (cold reads, as in a forward). Each call runs
// the engine sequence after the router: MoeRowsUniqueKernel, MoeRowsInSlotsKernel, MoeFusedKernel, combine.
// Variant 0 is the engine launch (engine.hip MoeFusedLaunch, BuildLayer item plan). Each other variant must give
// the same residual stack byte for byte on the same inputs.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <string>

#include "../engine/model/flashnext/model.h"
#include "../engine/layers/ops.h"
#include "../engine/layers/moe_shared_first.h"
#include "../engine/exl3/plans.h"
#include "kernels/moe_b128.h"
#include "verify_moe_plan.h"

using namespace crucible;

namespace {

const char* g_vm_dump = nullptr;

// Probe summary (variant 6) over calls nl..2nl-1: phase times in us from the first block start.
// g_vm_dump (argument 7) appends one line per block (call event us, item, job K and rows, stamps) for timeline analysis.
template <class Call>
void Phases(int R, const std::vector<Plan>& plans, Call call, int nl, double mb) {
  const Plan& p0 = plans[0];
  const int ngu = p0.gu.nitems, n = ngu + p0.dn.nitems;
  double a[8] = {};
  int calls = 0;
  FILE* dump = g_vm_dump ? std::fopen(g_vm_dump, "a") : nullptr;
  for (int i = nl; i < 2 * nl; ++i) {
    HIP_OK(hipMemset(g_vm_stamps, 0, sizeof(unsigned long long) * 3 * n));
    float ev_ms = 0;
    call(i, &ev_ms);
    HIP_OK(hipDeviceSynchronize());
    std::vector<unsigned long long> st(3 * n);
    HIP_OK(hipMemcpy(st.data(), g_vm_stamps, st.size() * 8, hipMemcpyDeviceToHost));
    if (dump) {
      const Plan& p = plans[i % nl];
      std::vector<LinJob> gj(p.gu.jobs.size()), dj(p.dn.jobs.size());
      std::vector<GemvItem> gi(p.gu.nitems), di(p.dn.nitems);
      HIP_OK(hipMemcpy(gj.data(), p.gu.d_jobs, gj.size() * sizeof(LinJob), hipMemcpyDeviceToHost));
      HIP_OK(hipMemcpy(dj.data(), p.dn.d_jobs, dj.size() * sizeof(LinJob), hipMemcpyDeviceToHost));
      HIP_OK(hipMemcpy(gi.data(), p.gu.d_items, gi.size() * sizeof(GemvItem), hipMemcpyDeviceToHost));
      HIP_OK(hipMemcpy(di.data(), p.dn.d_items, di.size() * sizeof(GemvItem), hipMemcpyDeviceToHost));
      for (int b = 0; b < n; ++b) {
        const bool g = b < ngu;
        const GemvItem& it = g ? gi[b] : di[b - ngu];
        const LinJob& j = g ? gj[it.job] : dj[it.job];
        std::fprintf(dump, "%d %d %.2f %d %d %d %d %d %d %d %llu %llu %llu\n", R, i, ev_ms * 1e3, b, g ? 0 : 1, it.job,
                     it.nb, j.K, j.R, j.I / 16 / j.S, st[3 * b], st[3 * b + 1], st[3 * b + 2]);
      }
    }
    unsigned long long t0 = ~0ull;
    for (int b = 0; b < n; ++b)
      if (st[3 * b]) t0 = std::min(t0, st[3 * b]);
    std::vector<double> gu_end, dn_start, dn_ready, dn_end, gu_dur, dn_dur;
    for (int b = 0; b < n; ++b) {
      if (!st[3 * b]) continue;
      const double s = (st[3 * b] - t0) / 100.0, r = (st[3 * b + 1] - t0) / 100.0, e = (st[3 * b + 2] - t0) / 100.0;
      if (b < ngu) gu_end.push_back(e), gu_dur.push_back(e - s);
      else dn_start.push_back(s), dn_ready.push_back(r), dn_end.push_back(e), dn_dur.push_back(e - r);
    }
    auto pct = [](std::vector<double>& v, double q) {
      std::sort(v.begin(), v.end());
      return v.empty() ? 0.0 : v[std::min(v.size() - 1, (size_t)(q * v.size()))];
    };
    a[0] += pct(gu_end, 0.5), a[1] += pct(gu_end, 1.0), a[2] += pct(dn_start, 0.5), a[3] += pct(dn_ready, 0.1);
    a[4] += pct(dn_ready, 0.5), a[5] += pct(dn_end, 1.0), a[6] += pct(gu_dur, 0.5), a[7] += pct(dn_dur, 0.5);
    ++calls;
  }
  if (dump) std::fclose(dump);
  for (double& x : a) x /= calls;
  std::printf("  phases R=%d: gu end p50 %.1f max %.1f | dn start p50 %.1f ready p10 %.1f p50 %.1f | end %.1f | "
              "gu block %.1f dn block %.1f us (100 MHz stamps; %.1f MB)\n",
              R, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], mb);
}

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: verify_moe ROUTES [layers] [iters] [rmin] [rmax] [variants]\n");
    return 2;
  }
  const auto toks = Tokens(argv[1]);
  const int nl = argc > 2 ? std::atoi(argv[2]) : 8;
  const int iters = argc > 3 ? std::atoi(argv[3]) : 192;
  const int rmin = argc > 4 ? std::atoi(argv[4]) : 1, rmax = argc > 5 ? std::atoi(argv[5]) : 8;
  // rmin may be a list ("1,4,8", moe-b128): then only those R run and rmax is ignored.
  std::vector<int> rlist;
  if (argc > 4 && std::strchr(argv[4], ',')) {
    for (const char* p = argv[4]; *p;) {
      rlist.push_back(std::atoi(p));
      const char* q = std::strchr(p, ',');
      p = q ? q + 1 : p + std::strlen(p);
    }
  } else {
    for (int R = rmin; R <= rmax; ++R) rlist.push_back(R);
  }
  const std::string vlist = argc > 6 ? argv[6] : "0";
  if (argc > 7) g_vm_dump = argv[7];
  std::printf("%zu route tokens, %d layers, %d calls per R\n", toks.size(), nl, iters);
  const size_t gw = TileWords(kHidden, kMoeInter, kKR), dw = TileWords(kMoeInter, kHidden, kKR);
  const std::vector<LayerW> layers = MakeLayers(nl);
  VmScratch s{};
  s.x = Alloc<float>(kMaxRows * kHidden);
  FillFloats<<<64, 256>>>(s.x, kMaxRows * kHidden, 7, 2.f);
  s.partial = Alloc<float>(size_t(16) << 20);
  s.xh = Alloc<__half>(size_t(16) << 20);
  s.H = Alloc<float>(kMaxRows * kWide);
  s.post = Alloc<float>(kMaxRows * kStreams);
  FillFloats<<<4, 64>>>(s.post, kMaxRows * kStreams, 9, 1.f);
  s.wts = Alloc<float>(kMaxRows * kSlots);
  FillFloats<<<4, 64>>>(s.wts, kMaxRows * kSlots, 11, 0.2f);
  s.map = Alloc<int>(kPackedRows * kTopK);
  s.mapj = Alloc<int>(kPackedRows * kTopK);
  s.rowl = Alloc<int>((kPackedRows * kTopK + 2) * kMaxRows);
  s.done = Alloc<unsigned>(kPackedRows * kTopK + 2);
  HIP_OK(hipMemset(s.done, 0, sizeof(unsigned) * (kPackedRows * kTopK + 2)));
  g_vm_stamps = Alloc<unsigned long long>(3 * 4096);
  float* H0 = Alloc<float>(kMaxRows * kWide);
  FillFloats<<<64, 256>>>(H0, kMaxRows * kWide, 13, 1.f);
  HIP_OK(hipDeviceSynchronize());
  const double expert_mb = (2.0 * gw + dw) * 4 / 1e6 + 3.0 * (kHidden + kMoeInter) * 2 / 1e6;
  const double shared_mb = 3.0 * TileWords(kHidden, kMoeInter, kKS) * 4 / 1e6;
  std::printf("expert %.4f MB, shared %.4f MB\n", expert_mb, shared_mb);
  hipEvent_t ev[3];
  for (auto& e : ev) HIP_OK(hipEventCreate(&e));
  std::vector<int> variants;
  for (size_t p = 0; p < vlist.size();) {
    size_t q = vlist.find(',', p);
    if (q == std::string::npos) q = vlist.size();
    variants.push_back(std::atoi(vlist.substr(p, q - p).c_str()));
    p = q + 1;
  }
  for (const int R : rlist) {
    // Selections: call i takes bench layer i % nl, model layer (7 i) % 48 and tokens [w, w + R).
    std::vector<int*> sels;
    double mb = 0;
    for (int i = 0; i < iters; ++i) {
      const int ml = (7 * i) % kLayers, w = (int)((size_t)i * 13 % (toks.size() - R));
      std::vector<int> sel;
      for (int r = 0; r < R; ++r) sel.insert(sel.end(), toks[w + r].begin() + ml * kTopK, toks[w + r].begin() + (ml + 1) * kTopK);
      std::vector<int> u = sel;
      std::sort(u.begin(), u.end());
      mb += (std::unique(u.begin(), u.end()) - u.begin()) * expert_mb + shared_mb;
      sels.push_back(Upload(sel));
    }
    mb /= iters;
    for (int v : variants) {
      const VerifyMoeVariant& V = VerifyMoeVariantOf(v);
      std::vector<Plan> plans;
      for (int l = 0; l < nl; ++l) plans.push_back(MakePlan(layers[l], R, s, V.item_blocks, kBenchDnXh));
      auto call = [&](int i, bool timed) {
        const int l = i % nl;
        const Plan& p = plans[l];
        MoeRowsUniqueKernel<<<1, 512>>>(sels[i], R, layers[l].d_experts, p.gu.d_jobs, p.dn.d_jobs, s.map, s.mapj,
                                        s.rowl, s.done);
        MoeRowsInSlotsKernel<<<MoeInSlotsBlocks(R), 256>>>(p.gu.d_jobs, s.map, s.mapj, R);
        if (timed) HIP_OK(hipEventRecord(ev[0], 0));
        if (!LaunchB128(v, R, p.gu, p.dn, s.done) && !LaunchFlow(v, R, p.gu, p.dn, s.done)) LaunchVerifyMoe(v, R, p.gu, p.dn, s.done);
        if (timed) HIP_OK(hipEventRecord(ev[1], 0));
        MoeRowsCombineKernel<<<(R * 20 + 7) / 8, 256>>>(p.dn.d_jobs, s.map, s.mapj, s.wts, s.post, s.H, R);
      };
      // Byte check on 2 nl calls: H after the combine.
      bool same = true;
      for (int i = 0; i < std::min(iters, 2 * nl); ++i) {
        HIP_OK(hipMemcpy(s.H, H0, sizeof(float) * kMaxRows * kWide, hipMemcpyDeviceToDevice));
        call(i, false);
        std::vector<float> h((size_t)R * kWide);
        HIP_OK(hipMemcpy(h.data(), s.H, h.size() * 4, hipMemcpyDeviceToHost));
        static std::map<std::pair<int, int>, std::vector<float>> keep;
        if (v == variants[0]) keep[{R, i}] = h;
        else same = same && std::memcmp(keep[{R, i}].data(), h.data(), h.size() * 4) == 0;
      }
      for (int i = 0; i < nl; ++i) call(i, false);
      HIP_OK(hipDeviceSynchronize());
      std::vector<double> ms;
      for (int i = 0; i < iters; ++i) {
        call(i, true);
        HIP_OK(hipEventSynchronize(ev[1]));
        float t;
        HIP_OK(hipEventElapsedTime(&t, ev[0], ev[1]));
        ms.push_back(t);
      }
      std::sort(ms.begin(), ms.end());
      HIP_OK(hipDeviceSynchronize());
      const double t0 = Now();
      for (int i = 0; i < iters; ++i) call(i, false);
      HIP_OK(hipDeviceSynchronize());
      const double chain_us = (Now() - t0) / iters * 1e6;
      if (v == 6 || v == 15 || v == 16 || v == 22 || v == 23)
        Phases(R, plans, [&](int i, float* t) {
          call(i, true);
          HIP_OK(hipEventSynchronize(ev[1]));
          HIP_OK(hipEventElapsedTime(t, ev[0], ev[1]));
        }, nl, mb);
      const double med = ms[ms.size() / 2] * 1e3;
      std::printf("R=%d v%d %-28s MB %.1f floor %.1f us | fused med %.1f us (%.0f GB/s, %.0f%%) p10 %.1f p90 %.1f | "
                  "chain stream %.1f us | %s\n",
                  R, v, V.name, mb, mb / 245.5e3 * 1e6, med, mb / med * 1e3, 100 * mb / 245.5e3 * 1e6 / med,
                  ms[ms.size() / 10] * 1e3, ms[ms.size() * 9 / 10] * 1e3, chain_us,
                  v == variants[0] ? "reference" : (FlowTimingOnly(v) || B128TimingOnly(v)) ? "timing only" : same ? "bitwise equal" : "DIFFERENT");
      std::fflush(stdout);
    }
  }
  return 0;
}
