// Bulk prefill HC mix with the fp16 high half of t only (CRUCIBLE_HC_MIX_HI=1, hc_mix_hi.h), and the one launcher
// of the bulk mid and of the bulk mix (both numerics).
//
// PrefillHcMixHiKernel is PrefillHcMixWideKernel<true,FB> without the low chain, with RT 16-row tiles per wave
// (a block is 128 * RT rows of 16 hidden columns), so each up fragment serves RT WMMAs. For every output the
// k order of the one chain, g and the epilogue (H, 1 / rms, w1, the stream sum, the fp16 rounding) are those of
// PrefillHcMixWideKernel<true,FB>. With tlo = 0 that kernel's low chain is +0 and g = hi + 0 = hi (a -0 becomes
// +0, and Sigmoid gives the same value for both), so the bits equal PrefillHcMixWideKernel on a zero tlo
// (tests/hc_mix_hi_gpu_test.hip). A row's result depends only on its own row, so R and the row block do not change it.
#pragma once
#include "hc_mix_hi.h"
#include "prefill_hc.h"

namespace crucible {

template<bool FB=false,int RT=kHcMixHiRowTiles>
__global__ void __launch_bounds__(256) PrefillHcMixHiKernel(const __half* up,const float* w1,
    const __half* thi,const float* ss,const float* H,float* mixed,int R,
    __half* xh=nullptr,int ldh=kWide) {
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
  const int rb=blockIdx.y*128*RT+wave*16*RT;
  const int d=blockIdx.x*16+(lane&15);
  if (rb>=R) return;
  prefill::F8 hi[RT][4]={};
#pragma unroll 2
  for (int k=0;k<kRank;k+=16) {
    prefill::H16 ah[RT];
#pragma unroll
    for (int i=0;i<RT;++i) {
      const int ra=rb+16*i+(lane&15);
      ah[i]=ra<R ? PrefillLoad16(thi+(size_t)ra*kRank+k) : prefill::H16{};
    }
#pragma unroll
    for (int s=0;s<4;++s) {
      const prefill::H16 b=FB ? PrefillLoad16(up+(((size_t)s*(kHidden/16)+(d>>4))*(kRank/16)+(k>>4))*256+(lane&15)*16)
                              : PrefillLoad16(up+((size_t)s*kHidden+d)*kRank+k);
#pragma unroll
      for (int i=0;i<RT;++i) hi[i][s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i],b,hi[i][s]);
    }
  }
  float w[4];
#pragma unroll
  for (int s=0;s<4;++s) w[s]=w1[s*kHidden+d];
  // The epilogue of PrefillHcMixWideKernel, one row tile at a time (see its comments).
#pragma unroll
  for (int i=0;i<RT;++i) {
    const int r0=rb+16*i;
    if (r0>=R) break;  // wave-uniform
    float g[8][4],hv[8][4],il[2];
#pragma unroll
    for (int z=0;z<8;++z)
#pragma unroll
      for (int s=0;s<4;++s) g[z][s]=hi[i][s][z];
    __builtin_amdgcn_sched_barrier(0);
    const int rw=__builtin_amdgcn_readfirstlane(r0),live=min(16,R-rw);
    const auto hrs=StreamRsrc(reinterpret_cast<const uint32_t*>(H+(size_t)rw*ldh),(uint32_t)(live*ldh*sizeof(float)));
#pragma unroll
    for (int z=0;z<8;++z) {
      const uint32_t ho=(uint32_t)((min(prefill::FragmentRow(z),live-1)*ldh+d)*sizeof(float));
#pragma unroll
      for (int s=0;s<4;++s)
        hv[z][s]=__uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(hrs,ho,s*kHidden*(int)sizeof(float),0));
    }
#pragma unroll
    for (int j=0;j<2;++j) {
      const int e=2*lane+j;
      il[j]=rsqrtf(ss[min(r0+(e>>2),R-1)*4+(e&3)]/kHidden+kEps);
    }
#pragma unroll
    for (int z=0;z<8;++z) {
      const int r=r0+prefill::FragmentRow(z);
      float v[4];
#pragma unroll
      for (int s=0;s<4;++s) {
        const float inv=__shfl(il[s&1],(prefill::FragmentRow(z)*4+s)>>1);
        v[s]=Sigmoid(g[z][s])*hv[z][s]*inv*w[s];
      }
      const float x=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
      if (r>=R) continue;
      if (mixed) mixed[(size_t)r*kHidden+d]=x;
      if (xh) xh[(size_t)r*kHidden+d]=__float2half(x);
    }
#pragma unroll
    for (int z=0;z<8;++z)
#pragma unroll
      for (int s=0;s<4;++s) asm volatile("" :: "v"(hv[z][s]));
  }
}

// The bulk mid of engine.hip HcRead (R > 8): hi (the switch) writes no tlo (about 10 MB a call at 16,384 rows);
// thi and post have the trunk bits. Off launches the trunk kernel exactly as before.
inline void LaunchPrefillHcMidPack(bool hi,const float* dots,int M,const float* ss,__half* thi,__half* tlo,
    float* post,int R) {
  if (hi) PrefillHcMidPackKernel<false><<<R,256>>>(dots,M,ss,thi,nullptr,post,R);
  else PrefillHcMidPackKernel<<<R,256>>>(dots,M,ss,thi,tlo,post,R);
}

// The bulk mix of engine.hip HcRead (R > 8). upf: the fragment-order up (HcFragUpKernel) or null. hi: the switch
// (the engine passes hc_mix_hi_); off launches PrefillHcMixWideKernel exactly as before.
inline void LaunchPrefillHcMix(bool hi,const __half* up,const __half* upf,const float* w1,const __half* thi,
    const __half* tlo,const float* ss,const float* H,float* mixed,int R,__half* xh,int ldh) {
  if (hi) {
    const dim3 grid(kHidden/16,HcMixHiRowBlocks(R));
    if (upf) PrefillHcMixHiKernel<true><<<grid,256>>>(upf,w1,thi,ss,H,mixed,R,xh,ldh);
    else PrefillHcMixHiKernel<false><<<grid,256>>>(up,w1,thi,ss,H,mixed,R,xh,ldh);
    return;
  }
  // Fragment-order up: bench hc_frag R4096 1993 -> 1796 us, bitwise.
  if (upf) PrefillHcMixWideKernel<true,true><<<dim3(kHidden/16,(R+127)/128),256>>>(upf,w1,thi,tlo,ss,H,mixed,R,xh,ldh);
  else PrefillHcMixWideKernel<true><<<dim3(kHidden/16,(R+127)/128),256>>>(up,w1,thi,tlo,ss,H,mixed,R,xh,ldh);
}

}  // namespace crucible
