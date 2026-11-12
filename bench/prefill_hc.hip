#include "prefill_common.h"
#include "kernels/prefill_hc.h"
using namespace crucible;
using namespace prefill_bench;

int main(int argc,char** argv) {
  const int R=argc>1 ? std::atoi(argv[1]) : 33, M=kRank+4, O=513;
  if (R<1 || R>2048) return 2;
  Buffer<float> H(Random((size_t)R*kWide,.4f,311));
  Buffer<__half> fn(Half(Random((size_t)M*kWide,.02f,313)));
  Buffer<__half> up(Half(Random((size_t)kWide*kRank,.02f,317)));
  auto ws=Random(kWide,.05f,319); for (auto& v:ws) v+=1.f;
  Buffer<float> w1(ws),dots((size_t)R*M*4),ss(R*4),t((size_t)R*kRank),post(R*4),mixed((size_t)R*kHidden);
  Buffer<__half> hi((size_t)R*kWide),lo((size_t)R*kWide),thi((size_t)R*kRank),tlo((size_t)R*kRank);
  auto tensor=[&] {
    PrefillHcSquaresKernel<<<(R*4+7)/8,256>>>(H.p,ss.p,R);
    PrefillPackKernel<<<(R*kWide+255)/256,256>>>(H.p,hi.p,lo.p,R*kWide);
    PrefillHcDotsKernel<<<dim3((R+15)/16,(M+63)/64,4),128>>>(fn.p,M,hi.p,lo.p,dots.p,R);
    PrefillHcMidKernel<<<R,256>>>(dots.p,M,ss.p,t.p,post.p,R);
    PrefillPackKernel<<<(R*kRank+255)/256,256>>>(t.p,thi.p,tlo.p,R*kRank);
    PrefillHcMixKernel<<<dim3((R+15)/16,kHidden/16),128>>>(up.p,w1.p,thi.p,tlo.p,ss.p,H.p,mixed.p,R);
  };
  tensor();CheckLaunch();
  const auto got_dots=dots.Get(),got_ss=ss.Get(),got_post=post.Get(),got_mixed=mixed.Get();
  auto serial=[&] {
    for (int r=0;r<R;r+=8) {
      const int count=std::min(8,R-r);
      const float* h=H.p+(size_t)r*kWide;
      float* d=dots.p+(size_t)r*M*4,*s=ss.p+r*4;
      if (count==8) {
        HcDotsKernel<8><<<((M+1)*4+7)/8,256>>>(fn.p,M,h,d,s);
        HcMixKernel<8><<<kHidden/16,256>>>(up.p,w1.p,d,M,s,h,mixed.p+(size_t)r*kHidden,post.p+r*4);
      } else for (int i=0;i<count;++i) {
        HcDotsKernel<1><<<((M+1)*4+7)/8,256>>>(fn.p,M,h+(size_t)i*kWide,d+i*M*4,s+i*4);
        HcMixKernel<1><<<kHidden/16,256>>>(up.p,w1.p,d+i*M*4,M,s+i*4,h+(size_t)i*kWide,
          mixed.p+(size_t)(r+i)*kHidden,post.p+(r+i)*4);
      }
    }
  };
  serial();CheckLaunch();
  const auto de=Compare(got_dots,dots.Get()),pe=Compare(got_post,post.Get()),me=Compare(got_mixed,mixed.Get());
  const auto square_error=Compare(got_ss,ss.Get());
  std::fprintf(stderr,"hc check R=%d square=%g dots=%g post=%g mixed=%g\n",R,square_error.rel_l2,de.rel_l2,pe.rel_l2,me.rel_l2);
  if (got_ss!=ss.Get() || de.rel_l2>2e-5 || pe.rel_l2>3e-6 || me.rel_l2>4e-4) return 4;
  Buffer<__half> W(Half(Random((size_t)O*kHidden,.02f,331)));
  Buffer<float> y((size_t)R*O);
  PrefillPackKernel<<<(R*kHidden+255)/256,256>>>(mixed.p,hi.p,nullptr,R*kHidden);
  PrefillDenseKernel<<<dim3((R+15)/16,(O+63)/64),128>>>(W.p,O,hi.p,y.p,O,R);
  CheckLaunch();const auto dense=y.Get();
  for (int r=0;r<R;++r) DenseGemvKernel<1><<<(O+7)/8,256>>>(W.p,O,mixed.p+(size_t)r*kHidden,kHidden,y.p+(size_t)r*O,O);
  CheckLaunch();const auto e=Compare(dense,y.Get());
  if (e.rel_l2>2e-5) return 5;
  EnergyHook("hc", "prefill_hc_tensor", tensor);
  std::printf("hc rows=%d dots=%g post=%g mixed=%g dense=%g tensor_us=%.3f serial_us=%.3f\n",
    R,de.rel_l2,pe.rel_l2,me.rel_l2,e.rel_l2,Time(tensor,3),Time(serial,3));
}
