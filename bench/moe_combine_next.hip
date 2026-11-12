// prefill-hc: PrefillMoeCombineKernel<true> (fp16 pd) against a candidate that
// loads all 11 slots first. Bytes of H must be equal; the k-order mutant must
// differ. Times: "cold" flushes L2 and MALL (a 256 MiB write) before each run.
// Usage: moe_combine_next [R ...]   (default 16 96 97 4096 8192).
#include "prefill_common.h"
#include "kernels/moe_combine_next.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>
using namespace crucible;
using namespace prefill_bench;

namespace {
constexpr int kLdh=kWide+64, kReps=10;
constexpr double kDramGbs=245.5;

Buffer<char>& Flush() { static Buffer<char> b((size_t)256<<20); return b; }
template<class F> double TimeCold(F launch,int n) {
  launch(); CheckLaunch();
  std::vector<double> us;
  hipEvent_t a,b; HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
  for (int i=0;i<n;++i) {
    HIP_OK(hipMemsetAsync(Flush().p,i&0xFF,Flush().n));
    HIP_OK(hipEventRecord(a)); launch(); HIP_OK(hipEventRecord(b));
    HIP_OK(hipEventSynchronize(b));
    float ms=0; HIP_OK(hipEventElapsedTime(&ms,a,b)); us.push_back(ms*1000.);
  }
  HIP_OK(hipEventDestroy(a)); HIP_OK(hipEventDestroy(b));
  std::sort(us.begin(),us.end());
  return us[us.size()/2];
}

void Run(int R,int& fails) {
  // Routing: 10 distinct experts per row, compact lists per expert in row order.
  std::mt19937 rng(503+R);
  std::vector<int> sel((size_t)R*kTopK),counts(kExperts+1,0),offsets(kExperts+1,0),inv((size_t)R*kTopK);
  std::vector<int> ids(kExperts); std::iota(ids.begin(),ids.end(),0);
  for (int r=0;r<R;++r) {
    std::shuffle(ids.begin(),ids.end(),rng);
    for (int k=0;k<kTopK;++k) { sel[(size_t)r*kTopK+k]=ids[k]; ++counts[ids[k]]; }
  }
  for (int e=1;e<=kExperts;++e) offsets[e]=offsets[e-1]+counts[e-1];
  std::vector<int> fill(offsets.begin(),offsets.end());
  for (size_t i=0;i<sel.size();++i) inv[i]=fill[sel[i]]++;
  // pd: per expert, rows of O fp16 values at float offset offsets[e] * kHidden (as PrefillMoeJobs lays it out).
  const size_t pd_floats=(size_t)(R*kTopK+R)*kHidden;
  Buffer<float> pd(pd_floats);
  {
    auto h=Half(Random(pd_floats*2,.05f,509));
    HIP_OK(hipMemcpy(pd.p,h.data(),pd_floats*4,hipMemcpyHostToDevice));
  }
  Buffer<__half> svh(Half(Random((size_t)(kExperts+1)*kHidden,1.f,521)));
  std::vector<LinJob> jobs(kExperts+1);
  for (int e=0;e<=kExperts;++e) {
    LinJob j{}; j.svh=svh.p+(size_t)e*kHidden; j.O=kHidden; j.I=kMoeInter; j.S=1;
    j.partial=pd.p+(size_t)(e<kExperts ? offsets[e] : R*kTopK)*kHidden; j.R=e<kExperts ? counts[e] : R;
    jobs[e]=j;
  }
  Buffer<LinJob> down(jobs);
  Buffer<int> dsel(sel),doff(offsets),dinv(inv);
  Buffer<float> wts(Random((size_t)R*kSlots,.3f,523)),post(Random((size_t)R*kStreams,1.f,527));
  const auto h0=Random((size_t)R*kLdh,.5f,529);
  Buffer<float> Href((size_t)R*kLdh),Hgot((size_t)R*kLdh);
  PrefillMoe a{}; a.dispatch.selected=dsel.p; a.dispatch.offsets=doff.p; a.dispatch.rows=R;
  a.inverse=dinv.p; a.down=down.p;
  const int grid=(R*20+7)/8;
  Href.Set(h0); PrefillMoeCombineKernel<true><<<grid,256>>>(a,wts.p,post.p,Href.p,kLdh); CheckLaunch();
  const auto ref=Href.Get();
  auto check=[&](const char* name,auto launch,bool expect_same) {
    Hgot.Set(h0); launch(); CheckLaunch();
    const auto got=Hgot.Get();
    const bool same=std::memcmp(got.data(),ref.data(),ref.size()*4)==0;
    std::printf("R=%d %s bitwise=%s%s\n",R,name,same ? "SAME" : "DIFF",same==expect_same ? "" : "  <-- FAIL");
    if (same!=expect_same) ++fails;
  };
  auto engine=[&] { PrefillMoeCombineKernel<true><<<grid,256>>>(a,wts.p,post.p,Hgot.p,kLdh); };
  auto next=[&] { PrefillMoeCombineNextKernel<0><<<grid,256>>>(a,wts.p,post.p,Hgot.p,kLdh); };
  auto lazy=[&] { PrefillMoeCombineNextKernel<0,1><<<grid,256>>>(a,wts.p,post.p,Hgot.p,kLdh); };
  auto nt=[&] { PrefillMoeCombineNextKernel<0,2><<<grid,256>>>(a,wts.p,post.p,Hgot.p,kLdh); };
  auto mutant=[&] { PrefillMoeCombineNextKernel<1><<<grid,256>>>(a,wts.p,post.p,Hgot.p,kLdh); };
  check("engine",engine,true);
  check("next_prefetch",next,true);
  check("next_addr_first",lazy,true);
  check("next_prefetch_nt",nt,true);
  check("mutant_korder",mutant,false);
  if (R<4096) return;
  const double bytes=(double)R*(kSlots*kHidden*2+2.*kStreams*kHidden*4);
  for (auto [name,f] : {std::pair<const char*,std::function<void()>>{"engine",engine},{"next_prefetch",next},{"next_addr_first",lazy},{"next_prefetch_nt",nt}}) {
    const double c=TimeCold(f,kReps), w=Time(f,kReps);
    std::printf("R=%d combine %-14s cold %8.1f us warm %8.1f us  %5.1f%% DRAM cold\n",R,name,c,w,bytes/c/1e3/kDramGbs*100);
  }
}
}  // namespace

int main(int argc,char** argv) {
  std::vector<int> rows;
  for (int i=1;i<argc;++i) rows.push_back(std::atoi(argv[i]));
  if (rows.empty()) rows={16,96,97,4096,8192};
  int fails=0;
  for (int R : rows) Run(R,fails);
  std::printf("fails=%d\n",fails);
  return fails ? 1 : 0;
}
