// prefill-hc: engine HC chain (dots, mid pack, mix) against candidate kernels.
// Every candidate must give the same bytes as the engine kernel it replaces.
// Usage: hc_next [R ...]   (default 16 96 97 4096 8192).
#include "prefill_common.h"
#include "kernels/hc_next.h"
#include <cstring>
using namespace crucible;
using namespace prefill_bench;

namespace {
constexpr int kM=kRank+kStreams, kLdh=kWide+64;
constexpr double kWmmaTflops=48.2, kDramGbs=245.5;

template<class T> bool Same(const Buffer<T>& a,const std::vector<T>& ref) {
  const auto got=a.Get();
  return std::memcmp(got.data(),ref.data(),ref.size()*sizeof(T))==0;
}

struct Inputs {
  int R;
  Buffer<float> H;
  Buffer<__half> fn,fnf,up,upf;
  Buffer<float> w1;
  explicit Inputs(int rows) : R(rows),H((size_t)rows*kLdh),fn(Half(Random((size_t)kM*kWide,.02f,313))),
      fnf((size_t)(kM+15)/16*16*kWide),up(Half(Random((size_t)kWide*kRank,.03f,317))),upf((size_t)kWide*kRank),
      w1(Random(kWide,.05f,319)) {
    // H rows with a spread of magnitudes, like the trunk streams.
    auto h=Random((size_t)rows*kLdh,.5f,311);
    for (size_t i=0;i<h.size();i+=97) h[i]*=40.f;
    H.Set(h);
    auto w=w1.Get(); for (auto& v:w) v+=1.f; w1.Set(w);
    HcFragFnKernel<<<((kM+15)/16*16*kWide+255)/256,256>>>(fn.p,kM,fnf.p);
    HcFragUpKernel<<<(kWide*kRank+255)/256,256>>>(up.p,upf.p);
    CheckLaunch();
  }
};

struct Outputs {
  Buffer<float> dots,ss,post,mixed; Buffer<__half> thi,tlo,xh;
  explicit Outputs(int R) : dots((size_t)R*kM*4),ss(R*4),post(R*4),mixed((size_t)R*kHidden),
      thi((size_t)R*kRank),tlo((size_t)R*kRank),xh((size_t)R*kHidden) {}
  void Nan() { dots.Nan(); ss.Nan(); post.Nan(); mixed.Nan(); thi.Nan(); tlo.Nan(); xh.Nan(); }
};

// The engine's launches (engine.hip HcRead, bulk path, default numerics).
void EngineDots(const Inputs& in,Outputs& o) {
  const int R=in.R;
  if (R>=512) PrefillHcDotsWsKernel<64,0,true,2,2,false,4,true><<<kStreams*((R+63)/64),(kHcWsCompute+2)*32>>>(
      in.fnf.p,kM,in.H.p,o.dots.p,o.ss.p,R,kLdh);
  else PrefillHcDotsWsKernel<64,0,true,2,4,false><<<kStreams*((R+kHcRows-1)/kHcRows),(kHcWsCompute+2)*32>>>(
      in.fn.p,kM,in.H.p,o.dots.p,o.ss.p,R,kLdh);
}
void EngineMid(const Inputs& in,Outputs& o) {
  PrefillHcMidPackKernel<<<in.R,256>>>(o.dots.p,kM,o.ss.p,o.thi.p,o.tlo.p,o.post.p,in.R);
}
void EngineMix(const Inputs& in,Outputs& o,bool mixed) {
  PrefillHcMixWideKernel<true,true><<<dim3(kHidden/16,(in.R+127)/128),256>>>(in.upf.p,in.w1.p,o.thi.p,o.tlo.p,
      o.ss.p,in.H.p,mixed ? o.mixed.p : nullptr,in.R,o.xh.p,kLdh);
}

template<int DT,bool PRE,int MUT=0,int SW=0,int DIAG=0> void MixNext(const Inputs& in,Outputs& o,bool mixed) {
  const dim3 grid=SW ? dim3(kHidden/16/DT*((in.R+127)/128)) : dim3(kHidden/16/DT,(in.R+127)/128);
  HcMixNextKernel<DT,PRE,MUT,SW,DIAG><<<grid,256>>>(in.upf.p,in.w1.p,o.thi.p,o.tlo.p,
      o.ss.p,in.H.p,mixed ? o.mixed.p : nullptr,in.R,o.xh.p,kLdh);
}
template<int KS,int SW> void MixLds(const Inputs& in,Outputs& o,bool mixed) {
  const int rbs=(in.R+63)/64;
  const dim3 grid=SW ? dim3(kHidden/64*rbs) : dim3(kHidden/64,rbs);
  HcMixLdsKernel<KS,SW><<<grid,256>>>(in.upf.p,in.w1.p,o.thi.p,o.tlo.p,o.ss.p,in.H.p,
      mixed ? o.mixed.p : nullptr,in.R,o.xh.p,kLdh);
}
template<int W,int HS,bool HALF=false> void MixUpLds(const Inputs& in,Outputs& o,bool mixed) {
  HcMixUpLdsKernel<W,HS,HALF><<<dim3(kHidden/16,(in.R+16*W-1)/(16*W)),W*32>>>(in.upf.p,in.w1.p,o.thi.p,o.tlo.p,
      o.ss.p,in.H.p,mixed ? o.mixed.p : nullptr,in.R,o.xh.p,kLdh);
}
template<int PD,int HS,int RT=1,bool HALF=false> void MixPipe(const Inputs& in,Outputs& o,bool mixed) {
  HcMixPipeKernel<PD,HS,RT,HALF><<<dim3(kHidden/16,(in.R+128*RT-1)/(128*RT)),256>>>(in.upf.p,in.w1.p,o.thi.p,o.tlo.p,
      o.ss.p,in.H.p,mixed ? o.mixed.p : nullptr,in.R,o.xh.p,kLdh);
}
template<bool SWAP,bool NT,int RTB=4,int LW=2,int PFB=1> void DotsNext(const Inputs& in,Outputs& o) {
  const int R=in.R, rows=16*RTB;
  const dim3 grid=SWAP ? dim3(kStreams*((R+rows-1)/rows)) : dim3((R+rows-1)/rows,kStreams);
  HcDotsNextKernel<SWAP,NT,RTB,LW,PFB><<<grid,(kHcWsCompute*RTB/4+LW)*32>>>(in.fnf.p,kM,in.H.p,o.dots.p,o.ss.p,R,kLdh);
}
// The engine dots with a stream-major grid (blocks of one stream run together, so fn of that stream stays in L2).
void DotsStreamMajor(const Inputs& in,Outputs& o) {
  const int R=in.R;
  PrefillHcDotsWsKernel<64,0,false,2,2,false,4,true><<<dim3((R+63)/64,kStreams),(kHcWsCompute+2)*32>>>(
      in.fnf.p,kM,in.H.p,o.dots.p,o.ss.p,R,kLdh);
}

constexpr int kReps=20;

int Run(int R) {
  Inputs in(R);
  Outputs ref(R),got(R);
  ref.Nan();
  EngineDots(in,ref); EngineMid(in,ref); EngineMix(in,ref,true); CheckLaunch();
  const auto rdots=ref.dots.Get(),rss=ref.ss.Get(),rpost=ref.post.Get(),rmixed=ref.mixed.Get();
  const auto rthi=ref.thi.Get(),rtlo=ref.tlo.Get(),rxh=ref.xh.Get();
  int fails=0;
  // Mix candidates read the reference t and ss.
  auto check_mix=[&](const char* name,auto launch,bool expect_same) {
    got.mixed.Nan(); got.xh.Nan();
    HIP_OK(hipMemcpy(got.thi.p,ref.thi.p,got.thi.n*2,hipMemcpyDeviceToDevice));
    HIP_OK(hipMemcpy(got.tlo.p,ref.tlo.p,got.tlo.n*2,hipMemcpyDeviceToDevice));
    HIP_OK(hipMemcpy(got.ss.p,ref.ss.p,got.ss.n*4,hipMemcpyDeviceToDevice));
    launch(in,got,true); CheckLaunch();
    const bool same=Same(got.mixed,rmixed) && Same(got.xh,rxh);
    std::printf("R=%d %s bitwise=%s%s\n",R,name,same ? "SAME" : "DIFF",same==expect_same ? "" : "  <-- FAIL");
    if (same!=expect_same) ++fails;
  };
  check_mix("mix_engine",EngineMix,true);
  check_mix("mix_dt1_pre",MixNext<1,true>,true);
  check_mix("mix_dt2",MixNext<2,false>,true);
  check_mix("mix_dt2_pre",MixNext<2,true>,true);
  check_mix("mix_sw2",MixNext<1,false,0,2>,true);
  check_mix("mix_sw4",MixNext<1,false,0,4>,true);
  check_mix("mix_sw4_pre",MixNext<1,true,0,4>,true);
  check_mix("mix_dt2_sw4",MixNext<2,false,0,4>,true);
  check_mix("mix_lds16",MixLds<16,0>,true);
  check_mix("mix_lds16_sw4",MixLds<16,4>,true);
  check_mix("mix_mutant_korder",MixNext<1,true,1>,false);
  check_mix("mix_pipe1",MixPipe<1,0>,true);
  check_mix("mix_pipe2",MixPipe<2,0>,true);
  check_mix("mix_ulds8",MixUpLds<8,0>,true);
  check_mix("mix_ulds8_h12",MixUpLds<8,12>,true);
  check_mix("mix_ulds16",MixUpLds<16,0>,true);
  check_mix("mix_ulds16_h12",MixUpLds<16,12>,true);
  check_mix("mix_pipe1_rt2",MixPipe<1,0,2>,true);
  check_mix("mix_pipe2_rt2",MixPipe<2,0,2>,true);
  {
    got.xh.Nan(); MixNext<1,false,0,0,8>(in,got,false); CheckLaunch();
    const auto x=got.xh.Get(); size_t flips=0;
    for (size_t i=0;i<x.size();++i) flips+=std::memcmp(&x[i],&rxh[i],2)!=0;
    const auto e=Compare(x,rxh);
    for (auto f : {MixPipe<1,0,1,true>,MixPipe<1,0,2,true>,MixUpLds<8,0,true>,MixUpLds<16,12,true>}) {
      got.xh.Nan(); f(in,got,false); CheckLaunch();
      const auto y=got.xh.Get();
      const bool same=std::memcmp(y.data(),x.data(),2*x.size())==0;
      std::printf("R=%d mix_half_pipe vs mix_halft bitwise=%s%s\n",R,same ? "SAME" : "DIFF",same ? "" : "  <-- FAIL");
      if (!same) ++fails;
    }
    std::printf("R=%d mix_halft (numerics change) rel_l2=%.3g max_rel=%.3g fp16_flips=%.4f%%\n",R,e.rel_l2,e.max_relative,100.*flips/x.size());
  }
  auto check_dots=[&](const char* name,auto launch) {
    if (R<512) return;
    got.Nan(); launch(in,got); CheckLaunch();
    const bool same=Same(got.dots,rdots) && Same(got.ss,rss);
    std::printf("R=%d %s bitwise=%s%s\n",R,name,same ? "SAME" : "DIFF",same ? "" : "  <-- FAIL");
    if (!same) ++fails;
  };
  check_dots("dots_stream_major",DotsStreamMajor);
  check_dots("dots_next_swap",DotsNext<true,false>);
  check_dots("dots_next_nt",DotsNext<true,true>);
  check_dots("dots_next_sm_nt",DotsNext<false,true>);
  check_dots("dots_s128_d1_lw4",[&](const Inputs& i,Outputs& o) {
    PrefillHcDotsWsKernel<128,0,true,4,1,false,4,true><<<kStreams*((i.R+63)/64),(kHcWsCompute+4)*32>>>(i.fnf.p,kM,i.H.p,o.dots.p,o.ss.p,i.R,kLdh); });
  check_dots("dots_s128_d2_lw4",[&](const Inputs& i,Outputs& o) {
    PrefillHcDotsWsKernel<128,0,true,4,2,false,4,true><<<kStreams*((i.R+63)/64),(kHcWsCompute+4)*32>>>(i.fnf.p,kM,i.H.p,o.dots.p,o.ss.p,i.R,kLdh); });
  check_dots("dots_rt8_lw4",DotsNext<true,false,8,4>);
  check_dots("dots_rt8_lw4_sm",DotsNext<false,false,8,4>);
  check_dots("dots_rt4_lw4",DotsNext<true,false,4,4>);
  if (R<1024) return fails;
  // Timing (engine launch shapes; the mix writes only xh, as the bulk path does).
  const int n=kReps;
  const double dots_flop=2.0*R*kStreams*336*kHidden, mix_flop=4.0*R*kWide*kRank;
  const double h_bytes=(double)R*kWide*4;
  auto line=[&](const char* name,double us,double flop,double bytes) {
    std::printf("R=%d %-22s %9.1f us  %5.1f%% WMMA  %5.1f%% DRAM\n",R,name,us,
                flop ? flop/us/1e6/kWmmaTflops*100 : 0.,bytes ? bytes/us/1e3/kDramGbs*100 : 0.);
  };
  line("dots_engine",Time([&]{ EngineDots(in,got); },n),dots_flop,h_bytes);
  // lw must match the kernel's LW template argument (loader waves).
  auto ws=[&](auto kernel,int rows,int lw=2) {
    return [&,kernel,rows,lw] { kernel<<<kStreams*((R+rows-1)/rows),(kHcWsCompute+lw)*32>>>(in.fnf.p,kM,in.H.p,got.dots.p,got.ss.p,R,kLdh); };
  };
  line("dots_diag5_loader_only",Time(ws(PrefillHcDotsWsKernel<64,5,true,2,2,false,4,true>,64),n),0,h_bytes);
  line("dots_diag3_fn_wmmaoff_hcache",Time(ws(PrefillHcDotsWsKernel<64,3,true,2,2,false,4,true>,64),n),0,h_bytes);
  line("dots_diag2_hcache",Time(ws(PrefillHcDotsWsKernel<64,2,true,2,2,false,4,true>,64),n),dots_flop,h_bytes);
  line("dots_diag4_fncache",Time(ws(PrefillHcDotsWsKernel<64,4,true,2,2,false,4,true>,64),n),dots_flop,h_bytes);
  line("dots_s128_d1",Time(ws(PrefillHcDotsWsKernel<128,0,true,2,1,false,4,true>,64),n),dots_flop,h_bytes);
  line("dots_s64_d1",Time(ws(PrefillHcDotsWsKernel<64,0,true,2,1,false,4,true>,64),n),dots_flop,h_bytes);
  line("dots_s32_d4",Time(ws(PrefillHcDotsWsKernel<32,0,true,2,4,false,4,true>,64),n),dots_flop,h_bytes);
  line("dots_s32_d2",Time(ws(PrefillHcDotsWsKernel<32,0,true,2,2,false,4,true>,64),n),dots_flop,h_bytes);
  line("dots_s128_d1_lw4",Time(ws(PrefillHcDotsWsKernel<128,0,true,4,1,false,4,true>,64,4),n),dots_flop,h_bytes);
  line("dots_engine_nowmma",Time([&]{
    PrefillHcDotsWsKernel<64,1,true,2,2,false,4,true><<<kStreams*((R+63)/64),(kHcWsCompute+2)*32>>>(
        in.fnf.p,kM,in.H.p,got.dots.p,got.ss.p,R,kLdh); },n),0,h_bytes);
  line("dots_stream_major",Time([&]{ DotsStreamMajor(in,got); },n),dots_flop,h_bytes);
  line("dots_next_swap",Time([&]{ DotsNext<true,false>(in,got); },n),dots_flop,h_bytes);
  line("dots_next_nt",Time([&]{ DotsNext<true,true>(in,got); },n),dots_flop,h_bytes);
  line("dots_next_sm",Time([&]{ DotsNext<false,false>(in,got); },n),dots_flop,h_bytes);
  line("dots_next_sm_nt",Time([&]{ DotsNext<false,true>(in,got); },n),dots_flop,h_bytes);
  line("dots_s128_d2_lw4",Time(ws(PrefillHcDotsWsKernel<128,0,true,4,2,false,4,true>,64,4),n),dots_flop,h_bytes);
  line("dots_s128_d1_lw4_rt2",Time(ws(PrefillHcDotsWsKernel<128,0,true,4,1,false,2,true>,32,4),n),dots_flop,h_bytes);
  line("dots_s64_d1_lw4",Time(ws(PrefillHcDotsWsKernel<64,0,true,4,1,false,4,true>,64,4),n),dots_flop,h_bytes);
  line("dots_s128_d1_lw4_diag5",Time(ws(PrefillHcDotsWsKernel<128,5,true,4,1,false,4,true>,64,4),n),0,h_bytes);
  line("dots_s128_d2_lw4_diag5",Time(ws(PrefillHcDotsWsKernel<128,5,true,4,2,false,4,true>,64,4),n),0,h_bytes);
  line("dots_rt8_lw4",Time([&]{ DotsNext<true,false,8,4>(in,got); },n),dots_flop,h_bytes);
  line("dots_rt8_lw4_sm",Time([&]{ DotsNext<false,false,8,4>(in,got); },n),dots_flop,h_bytes);
  line("dots_rt8_lw4_sm_nt",Time([&]{ DotsNext<false,true,8,4>(in,got); },n),dots_flop,h_bytes);
  line("dots_rt4_lw4",Time([&]{ DotsNext<true,false,4,4>(in,got); },n),dots_flop,h_bytes);
  line("mid_engine",Time([&]{ EngineMid(in,got); },n),0,(double)R*(kM*16+kRank*4+32));
  const double mix_bytes=h_bytes+(double)R*kHidden*2;
  line("mix_engine",Time([&]{ EngineMix(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_diag1_nowmma",Time([&]{ MixNext<1,false,0,0,1>(in,got,false); },n),0,mix_bytes);
  line("mix_diag2_hcache",Time([&]{ MixNext<1,false,0,0,2>(in,got,false); },n),mix_flop,0);
  line("mix_diag3_loads_t_up",Time([&]{ MixNext<1,false,0,0,3>(in,got,false); },n),0,0);
  line("mix_diag6_tcache_hcache",Time([&]{ MixNext<1,false,0,0,6>(in,got,false); },n),mix_flop,0);
  line("mix_diag7_up_only",Time([&]{ MixNext<1,false,0,0,7>(in,got,false); },n),0,0);
  line("mix_pipe1",Time([&]{ MixPipe<1,0>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_pipe2",Time([&]{ MixPipe<2,0>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_ulds8",Time([&]{ MixUpLds<8,0>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_ulds8_h12",Time([&]{ MixUpLds<8,12>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_ulds8_h16",Time([&]{ MixUpLds<8,16>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_ulds16",Time([&]{ MixUpLds<16,0>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_ulds16_h12",Time([&]{ MixUpLds<16,12>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_half_ulds8",Time([&]{ MixUpLds<8,0,true>(in,got,false); },n),mix_flop/2,mix_bytes);
  line("mix_half_ulds16_h12",Time([&]{ MixUpLds<16,12,true>(in,got,false); },n),mix_flop/2,mix_bytes);
  line("mix_pipe1_rt2",Time([&]{ MixPipe<1,0,2>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_pipe2_rt2",Time([&]{ MixPipe<2,0,2>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_half_pipe1",Time([&]{ MixPipe<1,0,1,true>(in,got,false); },n),mix_flop/2,mix_bytes);
  line("mix_half_pipe1_rt2",Time([&]{ MixPipe<1,0,2,true>(in,got,false); },n),mix_flop/2,mix_bytes);
  line("mix_half_pipe2_rt2",Time([&]{ MixPipe<2,0,2,true>(in,got,false); },n),mix_flop/2,mix_bytes);
  line("mix_halft",Time([&]{ MixNext<1,false,0,0,8>(in,got,false); },n),mix_flop/2,mix_bytes);
  line("mix_dt1_pre",Time([&]{ MixNext<1,true>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_dt2",Time([&]{ MixNext<2,false>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_dt2_pre",Time([&]{ MixNext<2,true>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_sw2",Time([&]{ MixNext<1,false,0,2>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_sw4",Time([&]{ MixNext<1,false,0,4>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_sw8",Time([&]{ MixNext<1,false,0,8>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_sw4_pre",Time([&]{ MixNext<1,true,0,4>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_dt2_sw4",Time([&]{ MixNext<2,false,0,4>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_lds16",Time([&]{ MixLds<16,0>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_lds16_sw4",Time([&]{ MixLds<16,4>(in,got,false); },n),mix_flop,mix_bytes);
  line("mix_lds16_sw8",Time([&]{ MixLds<16,8>(in,got,false); },n),mix_flop,mix_bytes);
  line("chain_engine",Time([&]{ EngineDots(in,got); EngineMid(in,got); EngineMix(in,got,false); },n),
       dots_flop+mix_flop,2*h_bytes);
  return fails;
}
}  // namespace

int main(int argc,char** argv) {
  std::vector<int> rows;
  for (int i=1;i<argc;++i) rows.push_back(std::atoi(argv[i]));
  if (rows.empty()) rows={16,96,97,4096,8192};
  bench::Sampler sampler; sampler.Start();
  int fails=0;
  for (int R : rows) fails+=Run(R);
  sampler.Stop();
  std::printf("%s fails=%d\n",sampler.Summary().c_str(),fails);
  return fails ? 1 : 0;
}
