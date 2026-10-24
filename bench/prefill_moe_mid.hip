// Engine-shaped gate/up + mid. The private ring has no cross-block waits.
#include "prefill_common.h"
#include "exl3_moe_mid_cold.h"
#include "kernels/exl3_moe_mid_probe.h"
#include "../engine/layers/prefill_moe_pair.h"
#include <cstdint>
#include <cstring>
#include <numeric>
using namespace crucible;
using namespace prefill_bench;

struct Fixture {
  // skew: uneven routing (expert = 512 u^3, so a few experts get thousands
  // of rows and many get 0 to 7); otherwise uniform.
  const bool skew;
  const int R, total, KG, KS;
  std::vector<int> sel, cnt, off, pairs, ids;
  std::vector<uint32_t> weights;
  std::vector<size_t> woff;
  std::vector<__half> scales;
  Buffer<int> ds, dc, d_o, dp, di, tc, ic, inv, fi, fu, fx, fr, fc;
  Buffer<__half> source, scale, xg, xu, xd;
  Buffer<uint32_t> dw;
  Buffer<float> pg, pu, pd;
  Buffer<LinJob> gu, down;
  Buffer<Exl3GemmJob> gj, dj;
  Buffer<Exl3GemmItem> gi, itdown;
  Buffer<ExpertPtrs> ep;
  PrefillMoe a{};
  Lin sg{}, su{}, sd{};
  Fixture(int rows, bool quality, bool skewed=false):skew(skewed),R(rows),total(rows*kSlots),KG(quality?4:3),KS(quality?6:5),
    sel(R*kTopK),cnt(kExperts),off(kExperts+1),ids(R*kTopK),
    ds(sel.size()),dc(kExperts),d_o(kExperts+1),dp(sel.size()),di(sel.size()),
    tc(kExperts*((R*kTopK+255)/256)),ic(2),inv(sel.size()),
    fi(2*((total+15)/16+kExperts+1)),fu(kExperts+1),fx(kExperts+1),
    fr((total+15)/16+kExperts+1),fc(3),
    source(Half(Random((size_t)R*kHidden,.5f,21))),
    scale((size_t)(kExperts+1)*3*kHidden),xg((size_t)total*kHidden),xu(xg.n),xd((size_t)total*kMoeInter),
    dw(((size_t)kExperts*2*KG+2*KS)*kHidden*kMoeInter/32),
    pg((size_t)total*kMoeInter),pu(pg.n),pd((size_t)total*kHidden),
    gu((kExperts+1)*2),down(kExperts+1),gj(gu.n),dj(down.n),
    gi(((total+31)/32+kExperts+1)*40),itdown(gi.n),ep(kExperts) {
    std::mt19937 rng(7);BuildRouting(rng);BuildWeights(rng);BuildJobs();
  }
  void BuildRouting(std::mt19937& rng) {
    std::vector<int> ex(kExperts);std::iota(ex.begin(),ex.end(),0);
    for(int r=0;r<R;++r)for(int k=0;k<kTopK;++k) {
      if(skew) {
        // Draw until the expert is new in this row (distinct top-k).
        int e;std::uniform_real_distribution<double> u(0,1);
        do{const double x=u(rng);e=std::min(kExperts-1,(int)(kExperts*x*x*x));}
        while(std::find(sel.begin()+r*kTopK,sel.begin()+r*kTopK+k,e)!=sel.begin()+r*kTopK+k);
        sel[r*kTopK+k]=e;++cnt[e];continue;
      }
      std::swap(ex[k],ex[k+rng()%(kExperts-k)]);sel[r*kTopK+k]=ex[k];++cnt[ex[k]];
    }
    for(int e=0;e<kExperts;++e) {
      off[e+1]=off[e]+cnt[e];
      for(int i=0;i<R*kTopK;++i)if(sel[i]==e)pairs.push_back(i);
    }
    for(size_t i=0;i<ids.size();++i)ids[i]=pairs[i]/kTopK;
    ds.Set(sel);dc.Set(cnt);d_o.Set(off);dp.Set(pairs);di.Set(ids);
  }
  void BuildWeights(std::mt19937& rng) {
    weights.resize(dw.n);for(auto& v:weights)v=rng();dw.Set(weights);
    auto sv=Random(scale.n,.004f,23);for(auto& v:sv)v=(v<0?-1.f:1.f)*(.012f+std::fabs(v));
    // Gate and up use one vector as suh and svh. At about 0.2 (13 x 0.015) the
    // gate and up outputs have an rms of about 1, as in the model, so the mid
    // values of routed rows are normal fp16 values (M1b). Down
    // keeps about 0.015.
    constexpr float kGateUpScale=13.f;
    for(int e=0;e<=kExperts;++e)for(int w=0;w<2;++w)
      for(int i=0;i<kHidden;++i)sv[(size_t)(e*3+w)*kHidden+i]*=kGateUpScale;
    scales=Half(sv);scale.Set(scales);
    woff.resize((kExperts+1)*2);
    size_t pos=0;
    for(int e=0;e<=kExperts;++e)for(int w=0;w<2;++w) {
      woff[2*e+w]=pos;pos+=(size_t)kHidden*kMoeInter*(e==kExperts?KS:KG)/32;
    }
  }
  void BuildJobs() {
    std::vector<ExpertPtrs> experts(kExperts);
    auto lin=[&](int e,int w) {
      Lin l{};l.I=kHidden;l.O=kMoeInter;l.K=e==kExperts?KS:KG;
      l.tiles=dw.p+woff[2*e+(w&1)];
      l.suh=scale.p+(size_t)(e*3+w)*kHidden;l.svh=l.suh;return l;
    };
    for(int e=0;e<kExperts;++e)for(int w=0;w<3;++w) {
      const auto l=lin(e,w);experts[e].tiles[w]=l.tiles;experts[e].suh[w]=l.suh;
      experts[e].svh[w]=l.svh;experts[e].K[w]=l.K;
    }
    ep.Set(experts);sg=lin(kExperts,0);su=lin(kExperts,1);sd=lin(kExperts,2);
    a.dispatch={ds.p,tc.p,dc.p,d_o.p,dp.p,di.p,R};
    a.gu=gu.p;a.down=down.p;a.g_gu=gj.p;a.g_down=dj.p;a.i_gu=gi.p;a.i_down=itdown.p;
    a.item_counts=ic.p;a.inverse=inv.p;a.tile_rows=96;a.packed=true;
    a.pg=pg.p;a.pu=pu.p;a.pd=pd.p;a.xg=xg.p;a.xu=xu.p;a.xd=xd.p;a.xh=source.p;
    a.fx_items=fi.p;a.fx_uoff=fu.p;a.fx_xoff=fx.p;a.fx_ready=fr.p;a.fx_count=fc.p;
  }
  void Plan(bool fused=false) {
    HIP_OK(hipMemsetAsync(ic.p,0,8));
    if(fused)PrefillMoeFxPlanKernel<<<1,512>>>(a,ep.p,sg,su,sd);
    else PrefillMoePlanKernel<<<1,512>>>(a,ep.p,sg,su,sd);
  }
  void Input(){PrefillMoeInputPackedKernel<<<dim3(40,(total+15)/16),256>>>(a);}
  void Gemm(){LaunchPrefillMoeGemm({6,true},a.g_gu,a.i_gu,a.item_counts,KG==4);}
  void Mid(){PrefillMoeMidKernel<<<(total*5+7)/8,256>>>(a);}
};

float Weight(const Fixture& f,int e,int w,int i,int o) {
  const int K=e==kExperts?f.KS:f.KG,r=i&15,c=o&15,lane=4*(c&7)+(r&7)/2;
  const int pos=8*lane+(r&1)+((r&8)?2:0)+((c&8)?4:0),bits=256*K;
  const auto* t=f.weights.data()+f.woff[2*e+w]+((size_t)(o/16)*(kHidden/16)+i/16)*8*K;
  uint32_t state=0;
  for(int z=0;z<16;++z){const int bit=((pos+1)*K-16+z+bits)%bits;state=(state<<1)|((t[bit/32]>>(31-bit%32))&1);}
  const uint32_t p=state*kMul1;
  const int sum=(p&255)+((p>>8)&255)+((p>>16)&255)+(p>>24);
  return __half2float(__float2half((float)((1024.+sum)*.00676727294921875-10.3828125)));
}
void HostWht(double* v) {
  for(int step=1;step<128;step*=2)for(int base=0;base<128;base+=step*2)
    for(int i=0;i<step;++i){const double a=v[base+i],b=v[base+i+step];v[base+i]=a+b;v[base+i+step]=a-b;}
}
bool MidOracle(const Fixture& f,const std::vector<float>& g,const std::vector<float>& u) {
  const auto xd=f.xd.Get();double worst=0;int points=0;
  for(int e:{0,255,kExperts}) {
    const int M=e==kExperts?f.R:f.cnt[e],base=e==kExperts?f.R*kTopK:f.off[e];
    if(!M)continue;
    for(int r:{0,M-1})for(int blk:{0,4}) {
      double gv[128],uv[128],v[128];
      for(int c=0;c<128;++c){gv[c]=g[(size_t)(base+r)*640+blk*128+c];uv[c]=u[(size_t)(base+r)*640+blk*128+c];}
      HostWht(gv);HostWht(uv);
      for(int c=0;c<128;++c) {
        const int col=blk*128+c;
        const double gt=gv[c]*kHadNorm*Value(f.scales[(e*3)*kHidden+col]);
        const double up=uv[c]*kHadNorm*Value(f.scales[(e*3+1)*kHidden+col]);
        const double h=Value(__float2half((float)(gt/(1.+std::exp(-gt))*up)));
        v[c]=Value(__float2half((float)(h*Value(f.scales[(e*3+2)*kHidden+col]))));
      }
      HostWht(v);
      for(int c=0;c<128;++c) {
        const int col=blk*128+c;
        const size_t ix=(size_t)base*640+((size_t)(col/16)*M+r)*16+col%16;
        const double ref=Value(__float2half((float)(v[c]*kHadNorm)));
        // Quantization boundaries can differ by one half ULP even when
        // the FP32 Hadamard error is small. Include the subnormal ULP.
        const double ulp=std::max(0x1p-24,std::ldexp(1.,std::ilogb(std::max(std::fabs(ref),0x1p-14))-10));
        const double err=std::fabs(Value(xd[ix])-ref)/ulp;
        if(!std::isfinite(err))return false;worst=std::max(worst,err);++points;
      }
    }
  }
  std::printf("FP64 mid points=%d worst-half-ULP=%.1f %s\n",points,worst,worst<=2?"PASS":"FAIL");
  return points>0&&worst<=2;
}
bool Oracle(const Fixture& f) {
  const auto g=f.pg.Get(),u=f.pu.Get();const auto xg=f.xg.Get(),xu=f.xu.Get();
  double worst=0;int points=0;
  for(int e:{0,255,kExperts}) {
    const int M=e==kExperts?f.R:f.cnt[e],base=e==kExperts?f.R*kTopK:f.off[e];
    if(!M)continue;
    for(int r:{0,M-1})for(int o:{0,15,127,128,639})for(int w=0;w<2;++w) {
      double ref=0,mag=0;const auto& x=w?xu:xg;const auto& p=w?u:g;
      for(int i=0;i<kHidden;++i) {
        const auto xi=(size_t)base*kHidden+((size_t)(i/16)*M+r)*16+i%16;
        const double v=Value(x[xi])*Weight(f,e,w,i,o);ref+=v;mag+=std::fabs(v);
      }
      const double err=std::fabs(p[(size_t)(base+r)*kMoeInter+o]-ref)/(mag+1e-30);
      if(!std::isfinite(err))return false;worst=std::max(worst,err);++points;
    }
  }
  std::printf("FP64 dots=%d worst=%.3e %s\n",points,worst,worst<1e-5?"PASS":"FAIL");
  return points>0&&worst<1e-5&&MidOracle(f,g,u);
}
template<class F> double ColdTime(F run,const gm::ColdCache& cold,const char* label,int reps=7) {
  // hwmon clock averages lag short kernels. Warm each path for0.5s before
  // matched cold samples, rather than sampling after a long CPU check.
  const auto start=std::chrono::steady_clock::now();
  do{gm::Flush(&cold);run();CheckLaunch();}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<.5);
  bench::EventTimer t;std::vector<double> ms;bench::Sampler clocks;clocks.Start();
  for(int i=0;i<reps;++i){gm::Flush(&cold);t.Begin();run();ms.push_back(t.EndMs());}
  clocks.Stop();std::printf("clocks-%s %s\n",label,clocks.Summary().c_str());
  return bench::Median(ms);
}
void PrintOccupancy(bool quality) {
  int resident=0;hipDeviceProp_t prop{};HIP_OK(hipGetDeviceProperties(&prop,0));
  if(quality)HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,ProbeMoeMidKernel<6,4,6>,128,0));
  else HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,ProbeMoeMidKernel<6,3,5>,128,0));
  std::printf("HIP occupancy resident-CTAs/MP=%d waves/MP=%d maximum-threads/MP=%d MPs=%d occupancy=%.1f%%\n",resident,resident*4,prop.maxThreadsPerMultiProcessor,prop.multiProcessorCount,100.*resident*128/prop.maxThreadsPerMultiProcessor);
  if(quality)HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,ProbeMoeFxMidKernel<6,4,6>,128,0));
  else HIP_OK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,ProbeMoeFxMidKernel<6,3,5>,128,0));
  std::printf("FX HIP occupancy resident-CTAs/MP=%d occupancy=%.1f%%\n",resident,100.*resident*128/prop.maxThreadsPerMultiProcessor);
}
// Halves that differ in [lo, hi).
size_t DiffCount(const std::vector<__half>& x,const std::vector<__half>& y,size_t lo=0,size_t hi=SIZE_MAX) {
  size_t n=0;hi=std::min(hi,x.size());for(size_t i=lo;i<hi;++i)n+=std::memcmp(&x[i],&y[i],sizeof(__half))!=0;return n;
}
// M1 shape envelope: engine RM (2 below 2,048 rows, 4
// below 4,096, else 6), uniform or skewed routing, outputs NaN-filled before
// every run. FX + Mid (A) must write every xd value; the M1 kernel (B) must
// equal it byte for byte; each mutant must change at least half of the routed
// rows' xd. pairs > 0: a warm-up, then ABAB pairs of A, B and C = FX alone.
int PairMain(int R,bool quality,int rm,bool skew,int pairs) {
  Fixture f(R,quality,skew);f.a.tile_rows=16*rm;
  const int pack=quality?405:305;
  auto fx=[&]{f.Plan(true);LaunchPrefillMoeFx(rm,f.a,quality);};
  auto run=[&](int mut) {
    f.Plan(true);
    if(mut==1)LaunchPrefillMoePair<1>(rm,f.a,quality);
    else if(mut==2)LaunchPrefillMoePair<2>(rm,f.a,quality);
    else LaunchPrefillMoePair<0>(rm,f.a,quality);
  };
  f.xd.Nan();f.pg.Nan();f.pu.Nan();fx();f.Mid();CheckLaunch();
  const auto ref=f.xd.Get();
  size_t unwritten=0;for(const auto h:ref)unwritten+=!std::isfinite(Value(h));
  const bool oracle=Oracle(f);bool ok=unwritten==0;
  const char* tag=skew?"skew":"uniform";
  std::printf("check pack=%d R=%d rm=%d routing=%s variant=fx-mid unwritten=%zu %s\n",pack,R,rm,tag,unwritten,ok?"PASS":"FAIL");
  const size_t routed=(size_t)R*kTopK*kMoeInter;
  const char* names[]={"pair","mut-no-suh","mut-swap"};
  for(int mut=0;mut<3;++mut) {
    f.xd.Nan();run(mut);CheckLaunch();const auto got=f.xd.Get();
    const size_t d=DiffCount(got,ref),dr=DiffCount(got,ref,0,routed);
    const double share=(double)dr/routed;
    const bool pass=mut==0?d==0:share>=.5;ok=ok&&pass;
    std::printf("check pack=%d R=%d rm=%d routing=%s variant=%s xd-diff=%zu/%zu routed-diff=%zu/%zu share=%.4f expect=%s %s\n",pack,R,rm,tag,names[mut],d,ref.size(),dr,routed,share,mut?"routed-share>=0.5":"equal",pass?"PASS":"FAIL");
  }
  std::printf("checks pack=%d R=%d rm=%d routing=%s bytes=%s oracle=%s\n",pack,R,rm,tag,ok?"PASS":"FAIL",oracle?"PASS":"FAIL");std::fflush(stdout);
  // The byte compares decide; the FP64 oracle is information only. Its 2
  // half-ULP limit is for the reference (main's FX + Mid), and one fp16 flip
  // before Wht128 can cost more output ULPs where Wht128 cancels (night of
  // Oct 13: 3.0 at R=16 skew, with bytes equal).
  if(!ok||!pairs)return ok?0:1;
  const auto cold=gm::MakeColdCache();
  const double scale=48.*4096/R;  // ms per call -> ms per 4,096 rows (48 MoE layers)
  ColdTime([&]{fx();f.Mid();},cold,"warm-up");
  std::vector<double> ab;int pos=0;
  for(int p=0;p<pairs;++p) {
    const double A=ColdTime([&]{fx();f.Mid();},cold,"A-fx-mid"),B=ColdTime([&]{run(0);},cold,"B-pair");
    const double C=ColdTime(fx,cold,"C-fx");
    ab.push_back((A-B)*scale);pos+=A>B;
    std::printf("abab pack=%d R=%d rm=%d routing=%s pair=%d A=%.4f B=%.4f C=%.4f save-AB=%.1f ms-per-4096\n",pack,R,rm,tag,p,A,B,C,ab.back());
    std::fflush(stdout);
  }
  std::printf("result pack=%d R=%d rm=%d routing=%s median-save-AB=%.1f B-faster=%d/%d\n",pack,R,rm,tag,bench::Median(ab),pos,pairs);
  cold.Free();return 0;
}
// Next 4.05 lever (findings section 11.4): main's FX kernel against its
// two-kernel form, the X transform (PrefillMoeInputPackedKernel) and the RM6
// gate/up GEMM, so the transform's share of FX is known. Timing only; the
// partials of both forms are compared as information.
int SplitMain(int R,bool quality,bool skew) {
  Fixture f(R,quality,skew);f.a.tile_rows=96;
  const int pack=quality?405:305;const char* tag=skew?"skew":"uniform";
  f.Plan(true);LaunchPrefillMoeFx(6,f.a,quality);CheckLaunch();
  const auto fg=f.pg.Get(),fu=f.pu.Get();
  f.pg.Nan();f.pu.Nan();f.Plan();f.Input();f.Gemm();CheckLaunch();
  const auto gg=f.pg.Get(),gu=f.pu.Get();
  const bool same=std::memcmp(fg.data(),gg.data(),fg.size()*4)==0&&std::memcmp(fu.data(),gu.data(),fu.size()*4)==0;
  const auto cold=gm::MakeColdCache();
  const double scale=48.*4096/R;
  std::vector<double> fx,pl,in,ge;
  for(int p=0;p<3;++p) {
    fx.push_back(ColdTime([&]{f.Plan(true);LaunchPrefillMoeFx(6,f.a,quality);},cold,"fx"));
    pl.push_back(ColdTime([&]{f.Plan();},cold,"plan"));
    in.push_back(ColdTime([&]{f.Plan();f.Input();},cold,"plan-input"));
    ge.push_back(ColdTime([&]{f.Plan();f.Gemm();},cold,"plan-gemm"));
  }
  const double F=bench::Median(fx),P=bench::Median(pl),I=bench::Median(in)-P,G=bench::Median(ge)-P;
  std::printf("split pack=%d R=%d routing=%s partials-bitwise=%d ms-per-call fx=%.4f plan=%.4f input=%.4f gemm=%.4f\n",pack,R,tag,(int)same,F,P,I,G);
  std::printf("split pack=%d R=%d routing=%s ms-per-4096 fx=%.1f input=%.1f gemm=%.1f input+gemm=%.1f fx-minus-gemm=%.1f\n",pack,R,tag,F*scale,I*scale,G*scale,(I+G)*scale,(F-G)*scale);
  std::fflush(stdout);cold.Free();return 0;
}
// Rows list: "1-16,97,2047" (each 1 to 8192).
std::vector<int> RowList(const char* s) {
  std::vector<int> v;
  for(const char* p=s;*p;) {
    char* e=nullptr;const long a=std::strtol(p,&e,10);long b=a;
    if(e==p)return {};
    if(*e=='-'){p=e+1;b=std::strtol(p,&e,10);if(e==p)return {};}
    if(a<1||b>8192||a>b)return {};
    for(long r=a;r<=b;++r)v.push_back((int)r);
    p=*e==','?e+1:e;if(*e&&*e!=',')return {};
  }
  return v;
}
int main(int argc,char** argv) {
  // M1 modes: ROWS-LIST 305|405 pair|time [RM 2|4|6 [uniform|skew]]. RM 0 or
  // no RM: the engine's choice per row count. time adds 5 ABAB pairs.
  if(argc>3&&(!std::strcmp(argv[3],"pair")||!std::strcmp(argv[3],"time"))) {
    if(argc>6||(std::strcmp(argv[2],"305")&&std::strcmp(argv[2],"405")))return 2;
    const auto rows=RowList(argv[1]);const int rmarg=argc>4?std::atoi(argv[4]):0;
    const bool skew=argc>5&&!std::strcmp(argv[5],"skew");
    if(rows.empty()||(rmarg&&rmarg!=2&&rmarg!=4&&rmarg!=6)||(argc>5&&!skew&&std::strcmp(argv[5],"uniform")))return 2;
    int rc=0;
    for(const int R:rows) {
      const int rm=rmarg?rmarg:R>=4096?6:R>=2048?4:2;
      rc|=PairMain(R,!std::strcmp(argv[2],"405"),rm,skew,std::strcmp(argv[3],"time")?0:5);
    }
    return rc;
  }
  // Split mode: ROWS-LIST 305|405 split [uniform|skew], rows 4096 to 8192 (RM6).
  if(argc>3&&!std::strcmp(argv[3],"split")) {
    if(argc>5||(std::strcmp(argv[2],"305")&&std::strcmp(argv[2],"405")))return 2;
    const auto rows=RowList(argv[1]);const bool skew=argc>4&&!std::strcmp(argv[4],"skew");
    if(rows.empty()||(argc>4&&!skew&&std::strcmp(argv[4],"uniform")))return 2;
    for(const int R:rows)if(R<4096)return 2;
    int rc=0;for(const int R:rows)rc|=SplitMain(R,!std::strcmp(argv[2],"405"),skew);
    return rc;
  }
  if(argc>3)return 2;
  char* end=nullptr;const long rows=argc>1?std::strtol(argv[1],&end,10):4096;
  if(rows<16||rows>4096||(argc>1&&(!*argv[1]||*end)))return 2;
  if(argc>2&&std::strcmp(argv[2],"305")&&std::strcmp(argv[2],"405"))return 2;
  const int R=(int)rows;const bool quality=argc>2&&std::strcmp(argv[2],"405")==0;
  PrintOccupancy(quality);
  Fixture f(R,quality);f.Plan();f.Input();f.Gemm();f.Mid();CheckLaunch();
  const auto ref=f.xd.Get();if(!Oracle(f))return 1;
  const auto cold=gm::MakeColdCache();
  const double base=ColdTime([&]{f.Gemm();f.Mid();},cold,"gemm-mid");
  const double fullbase=ColdTime([&]{f.Plan(true);LaunchPrefillMoeFx(6,f.a,quality);f.Mid();},cold,"fx-mid");
  const double flops=4.*R*kSlots*kHidden*kMoeInter;
  std::printf("pack=%d R=%d baseline-gemm-mid=%.4fms %.3fTFLOPS %.1f%%-49TF FX-mid=%.4fms\n",quality?405:305,R,base,flops/base*1e-9,flops/base*1e-9/49*100,fullbase);
  f.Plan();f.Input();CheckLaunch();
  const auto rg=f.pg.Get(),ru=f.pu.Get();f.pg.Zero();f.pu.Zero();
  {
    Buffer<float> ring((size_t)128*96*128*2);
    if(quality)ProbeMoeMidKernel<6,4,6,true><<<128,128>>>(f.a,ring.p);
    else ProbeMoeMidKernel<6,3,5,true><<<128,128>>>(f.a,ring.p);
    CheckLaunch();const auto cg=f.pg.Get(),cu=f.pu.Get();
    const bool same=std::memcmp(cg.data(),rg.data(),rg.size()*4)==0&&std::memcmp(cu.data(),ru.data(),ru.size()*4)==0;
    std::printf("checked full G/U partials bitwise=%d\n",(int)same);
    if(!same||!Oracle(f))return 1;
  }
  bool ok=true;
  for(int blocks:{40,80,128,256}) {
    Buffer<float> ring((size_t)blocks*96*128*2);
    auto probe=[&]{if(quality)ProbeMoeMidKernel<6,4,6><<<blocks,128>>>(f.a,ring.p);else ProbeMoeMidKernel<6,3,5><<<blocks,128>>>(f.a,ring.p);};
    f.xd.Zero();probe();CheckLaunch();const auto got=f.xd.Get();
    const bool same=std::memcmp(got.data(),ref.data(),ref.size()*sizeof(__half))==0;
    for(const auto h:got)if(!std::isfinite(Value(h)))ok=false;
    ok=ok&&same;const double ms=ColdTime(probe,cold,"ring");
    const double full=ColdTime([&]{f.Input();probe();},cold,"input-ring");
    auto fxprobe=[&]{f.Plan(true);if(quality)ProbeMoeFxMidKernel<6,4,6><<<blocks,128>>>(f.a,ring.p,kFxLead);else ProbeMoeFxMidKernel<6,3,5><<<blocks,128>>>(f.a,ring.p,kFxLead);};
    f.xd.Zero();fxprobe();CheckLaunch();const auto fxgot=f.xd.Get();
    const bool fxsame=std::memcmp(fxgot.data(),ref.data(),ref.size()*sizeof(__half))==0;
    ok=ok&&fxsame;
    const double fxms=ColdTime(fxprobe,cold,"fx-ring");
    std::printf("pack=%d blocks=%d ring=%.2fMiB fused=%.4fms full=%.4fms fx-ring=%.4fms %.3fTFLOPS %.1f%%-49TF bitwise=%d fx-bitwise=%d\n",quality?405:305,blocks,ring.n*4./(1024*1024),ms,full,fxms,flops/ms*1e-9,flops/ms*1e-9/49*100,(int)same,(int)fxsame);std::fflush(stdout);
  }
  cold.Free();return ok?0:1;
}
