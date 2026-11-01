// Candidate prefill MoE combine (prefill-hc). Same arithmetic and k order as
// PrefillMoeCombineKernel<true>, so the bench compares bytes.
#pragma once
#include "../../engine/layers/prefill_moe.h"

namespace crucible {
// PrefillMoeCombineKernel<true> with the 11 slot addresses computed first and
// the 11 fp16 partial blocks and svh blocks loaded before the first WHT. The
// engine kernel runs the slot loop with a chain of dependent loads (selected,
// inverse, offsets, then the partial) per slot, one slot at a time.
// MODE 0: load all 11 partial and svh blocks first. MODE 1: compute the 11
// addresses first, load each slot just before its WHT (fewer VGPRs). MODE 2:
// MODE 0 with non-temporal H loads and stores (H is 335 MB at 8,192 rows).
// MUT (bench mutant): 1 adds slot 1 before slot 0.
template<int MUT=0,int MODE=0>
__global__ void __launch_bounds__(256) PrefillMoeCombineNextKernel(PrefillMoe a,
    const float* weights, const float* post, float* H,int ldh=kWide) {
  const int w=blockIdx.x*8+(threadIdx.x>>5), lane=threadIdx.x&31;
  if (w>=a.dispatch.rows*20) return;
  const int r=w/20, blk=w%20, c0=blk*128+lane*4;
  typedef unsigned U2 __attribute__((ext_vector_type(2)));
  U2 p[kSlots],s[kSlots];
  const U2 *pp[kSlots],*sp[kSlots];
#pragma unroll
  for (int k=0; k<kSlots; ++k) {
    const int pair=r*kTopK+k;
    const int e=k<kTopK ? a.dispatch.selected[pair] : kExperts;
    const int local=k<kTopK ? a.inverse[pair]-a.dispatch.offsets[e] : r;
    const LinJob& j=a.down[e];
    pp[k]=reinterpret_cast<const U2*>(reinterpret_cast<const __half*>(j.partial)+(size_t)local*j.O+c0);
    sp[k]=reinterpret_cast<const U2*>(j.svh+c0);
    if (MODE!=1) { p[k]=*pp[k]; s[k]=*sp[k]; }
  }
  float y[4]={};
#pragma unroll
  for (int q=0; q<kSlots; ++q) {
    const int k=MUT && q<2 ? 1-q : q;
    if (MODE==1) { p[k]=*pp[k]; s[k]=*sp[k]; }
    const __half* ph=reinterpret_cast<const __half*>(&p[k]);
    const __half* sh=reinterpret_cast<const __half*>(&s[k]);
    // LinOutLane is LinOutBlockHalf's arithmetic on values the lane holds:
    // fp16 -> fp32 is exact, so 0 + p and the WHT see the same values.
    float pf[4],v[4];
#pragma unroll
    for (int i=0; i<4; ++i) pf[i]=__half2float(ph[i]);
    LinOutLane(pf,sh,v);
    const float weight=weights[r*kSlots+k];
#pragma unroll
    for (int i=0; i<4; ++i) y[i]+=weight*v[i];
  }
#pragma unroll
  for (int st=0; st<kStreams; ++st) {
    const float gate=post[r*kStreams+st];
#pragma unroll
    for (int i=0; i<4; ++i) {
      float* h=H+(size_t)r*ldh+st*kHidden+c0+i;
      if (MODE==2) __builtin_nontemporal_store(__builtin_nontemporal_load(h)+gate*y[i],h);
      else *h+=gate*y[i];
    }
  }
}
}  // namespace crucible
