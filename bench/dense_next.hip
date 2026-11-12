// prefill-hc: the engine's prefill dense launches (LaunchDenseHalf) against
// candidate kernels, for the three prefill shapes: GDN a/b (O = 96), router
// (O = 513) and PLE kv (O = 12800). Every candidate must give the same bytes.
// Times: "cold" flushes L2 and MALL (a 256 MiB write) before each launch,
// like the engine, where x comes from an earlier kernel; "warm" repeats.
// Usage: dense_next [R ...]   (default 16 96 97 4096 8192).
#include "prefill_common.h"
#include "kernels/dense_next.h"
#include <cstring>
#include <memory>
using namespace crucible;
using namespace prefill_bench;

namespace {
constexpr double kWmmaTflops=44.6;

// engine.hip LaunchDenseHalf.
void Engine(const __half* W,int O,const __half* x,float* y,int R) {
  const dim3 g((R+127)/128,(O+127)/128);
  if (O<256 && R<=1024) PrefillDenseTileKernel<<<dim3((R+63)/64,(O+127)/128),256>>>(W,O,x,y,O,R);
  else if (O<256 || O>1024) PrefillDenseBigKernel<<<g,256>>>(W,O,x,y,O,R);
  else PrefillDenseFastKernel<2><<<g,256>>>(W,O,x,y,O,R);
}
template<int DEPTH,int RB,int MUT=0> auto Next(int G) {
  return [G](const __half* W,int O,const __half* x,float* y,int R) {
    const int blocks=(R+RB-1)/RB*((O+127)/128);
    DenseNextKernel<DEPTH,RB,MUT><<<blocks,RB*2>>>(W,O,x,y,O,R,G);
  };
}

template<int DEPTH,int RB> auto EngineSw(int G) {
  return [G](const __half* W,int O,const __half* x,float* y,int R) {
    PrefillDenseSwKernel<DEPTH,RB><<<(R+RB-1)/RB*((O+127)/128),RB*2>>>(W,O,x,y,O,R,G);
  };
}

constexpr int kReps=10;

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

// Input transform of a packed group of NJ jobs (engine: PrefillLinInHalfKernel<true>, no item list).
void RunLinIn(int R,int nj,int& fails) {
  const int I=kHidden, per=R*(I/128);
  Buffer<__half> src(Half(Random((size_t)R*I,1.f,421)));
  std::vector<std::unique_ptr<Buffer<__half>>> scales,ref,got;
  std::vector<LinJob> jr(nj),jg(nj);
  for (int j=0;j<nj;++j) {
    auto sv=Random(I,1.f,431+j); for (auto& v:sv) v=v<0 ? -1.f : 1.f;
    scales.emplace_back(new Buffer<__half>(Half(sv)));
    ref.emplace_back(new Buffer<__half>((size_t)R*I)); got.emplace_back(new Buffer<__half>((size_t)R*I));
    LinJob b{}; b.suh=scales.at(j)->p; b.I=I; b.R=R; b.lds=I; b.O=I; b.K=4; b.S=1;
    jr[j]=b; jr[j].xh=ref[j]->p; jg[j]=b; jg[j].xh=got[j]->p;
  }
  Buffer<LinJob> dr(jr),dg(jg);
  for (auto& b : ref) b->Nan();
  for (auto& b : got) b->Nan();
  auto engine=[&] { PrefillLinInHalfKernel<true><<<(nj*per+7)/8,256>>>(dr.p,nullptr,nj*per,src.p); };
  auto fused=[&] { PrefillLinInHalfJobsKernel<<<(per+7)/8,256>>>(dg.p,nj,per,src.p); };
  engine(); fused(); CheckLaunch();
  bool same=true;
  for (int j=0;j<nj;++j) { const auto a=ref[j]->Get(),b=got[j]->Get(); same=same && std::memcmp(a.data(),b.data(),a.size()*2)==0; }
  std::printf("R=%d linin jobs=%d fused bitwise=%s%s\n",R,nj,same ? "SAME" : "DIFF",same ? "" : "  <-- FAIL");
  if (!same) ++fails;
  if (R<4096) return;
  const int n=kReps;
  std::printf("R=%d linin jobs=%d engine cold %8.1f us warm %8.1f us | fused cold %8.1f us warm %8.1f us\n",R,nj,
              TimeCold(engine,n),Time(engine,n),TimeCold(fused,n),Time(fused,n));
}

int Run(int R,int& fails) {
  RunLinIn(R,2,fails);
  RunLinIn(R,4,fails);
  for (int O : {96,513,12800}) {
    if (O==12800 && R<4096 && R!=97) continue;
    Buffer<__half> W(Half(Random((size_t)O*kHidden,.02f,401+O))),x(Half(Random((size_t)R*kHidden,1.f,409)));
    Buffer<float> ref((size_t)R*O),got((size_t)R*O);
    ref.Nan(); Engine(W.p,O,x.p,ref.p,R); CheckLaunch();
    const auto r=ref.Get();
    auto check=[&](const char* name,auto launch,bool expect_same) {
      got.Nan(); launch(W.p,O,x.p,got.p,R); CheckLaunch();
      const auto v=got.Get();
      const bool same=std::memcmp(v.data(),r.data(),r.size()*4)==0;
      std::printf("R=%d O=%d %s bitwise=%s%s\n",R,O,name,same ? "SAME" : "DIFF",same==expect_same ? "" : "  <-- FAIL");
      if (same!=expect_same) ++fails;
    };
    check("next_d2_rb128_g1",Next<2,128>(1),true);
    check("next_d2_rb128_g8",Next<2,128>(8),true);
    check("next_d2_rb64_g4",Next<2,64>(4),true);
    check("next_d4_rb128_g4",Next<4,128>(4),true);
    check("engine_sw_d4_rb64_g8",EngineSw<4,64>(8),true);
    check("engine_sw_d4_rb128_g16",EngineSw<4,128>(16),true);
    check("mutant_korder",Next<2,128,1>(1),false);
    if (R<4096) continue;
    const int n=kReps;
    const double flop=2.*R*O*kHidden;
    auto line=[&](const char* name,auto launch) {
      auto f=[&] { launch(W.p,O,x.p,got.p,R); };
      const double c=TimeCold(f,n), w=Time(f,n);
      std::printf("R=%d O=%-5d %-22s cold %9.1f us  warm %9.1f us  %5.1f%% WMMA cold\n",R,O,name,c,w,flop/c/1e6/kWmmaTflops*100);
    };
    line("engine",Engine);
    line("next_d2_rb128_g1",Next<2,128>(1));
    line("next_d2_rb128_g2",Next<2,128>(2));
    line("next_d2_rb128_g4",Next<2,128>(4));
    line("next_d2_rb128_g8",Next<2,128>(8));
    line("next_d2_rb128_g16",Next<2,128>(16));
    line("next_d2_rb128_g64",Next<2,128>(64));
    line("next_d4_rb128_g4",Next<4,128>(4));
    line("next_d4_rb128_g16",Next<4,128>(16));
    line("next_d2_rb64_g1",Next<2,64>(1));
    line("next_d2_rb64_g8",Next<2,64>(8));
    line("next_d4_rb64_g8",Next<4,64>(8));
  }
  return 0;
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
