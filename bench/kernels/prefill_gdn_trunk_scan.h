// The GDN SGPR scan of the earlier engine (before the carried lazy scale): the bench reference of bench/gdn_split.hip. A part
// start resets the lazy scale and the part end stores scale * S. Kept as text so the bench compares the real earlier code.
#pragma once
#include "../../engine/layers/kernels/prefill_gdn.h"
namespace crucible::prefill {
__global__ void __launch_bounds__(128) GdnScanSgprTrunkKernel(GdnPrefill a, const float* __restrict__ kv) {
  __shared__ float4 memx[2][32], yx[2][32];
  __shared__ float vx[2][4][32];
  const int h=blockIdx.x>>2, lane=threadIdx.x&31, sub=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int j=(blockIdx.x&3)*32+lane, ch=4096+h*128+j, R=a.input.R;
  float cw[4];
#pragma unroll
  for (int tap=0;tap<4;++tap) cw[tap]=a.input.conv[ch*4+tap];
  float S[32];
  float* dst=a.input.state+(size_t)h*128*128+(size_t)sub*32*128+j;
#pragma unroll
  for (int i=0;i<32;++i) S[i]=dst[i*128];
  const size_t rows=(size_t)a.chunks*64, base=(size_t)h*rows, kbase=(size_t)(h/3)*rows;
  float scale=1.f, pscale=1.f, raw[4], kb[2][32], qb[2][32];
  auto load=[&](int r0) {
    const int rs=r0+sub;
#pragma unroll
    for (int tap=0;tap<4;++tap) {
      const int rr=rs-3+tap,p=a.input.pos0+rr;
      raw[tap]=rs>=R ? 0.f : rr>=0 ? a.input.qkv[(size_t)rr*kGdnQkvWidth+ch]
          : p>=0 ? a.input.hist[(size_t)(p&15)*kGdnQkvWidth+ch] : 0.f;
    }
  };
  auto conv=[&](int r0,int slot) {
    if (r0+sub>=R) return;
    float x=0.f;
#pragma unroll
    for (int i=0;i<4;++i) {
#ifndef CRUCIBLE_GDN_VSCAN_MUTANT
      const int tap=i;
#else
      const int tap=3-i;
#endif
      x+=cw[tap]*raw[tap];
    }
    vx[slot][sub][lane]=Silu(x);
  };
  auto kl=[&](float (&kd)[32],int r) {
    const float* kp=kv+(kbase+r)*128+sub*32;
#pragma unroll
    for (int i=0;i<32;++i) kd[i]=kp[i];
  };
  auto ql=[&](float (&qd)[32],int r) {
    const float4* qp=reinterpret_cast<const float4*>(a.q+(kbase+r)*128+sub*32);
#pragma unroll
    for (int i=0;i<8;++i) { const float4 t=qp[i]; qd[4*i]=t.x; qd[4*i+1]=t.y; qd[4*i+2]=t.z; qd[4*i+3]=t.w; }
  };
  // The LDS barrier waits only for LDS (lgkmcnt), not for the k, q, g and beta
  // loads in flight or the y stores (__syncthreads waits for all of them).
  auto barrier=[] { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); };
  load(0); conv(0,0); load(4); kl(kb[0],0);
  float gn=R>(lane&3) ? a.g[base+(lane&3)] : 0.f, bn=R>(lane&3) ? a.beta[base+(lane&3)] : 0.f;
  __syncthreads();
  auto step=[&](auto U,int r0,int slot,float dq,float bq) {
    constexpr int u=decltype(U)::value, P=u&1;
    const int r=r0+u;
    // Lane 4m+u holds row r0+u: the quad broadcast keeps decay and scale in VGPRs
    // (VALU arithmetic, as in GdnScanLanesVKernel; no SALU float).
    const float decay=GdnQuadBroadcast<u>(dq), beta=GdnQuadBroadcast<u>(bq);
    float (&k)[32]=kb[P]; float (&q)[32]=qb[P];
    scale*=decay;
    if (scale<0x1p-20f) {
#pragma unroll
      for (int i=0;i<32;++i) S[i]*=scale;
      scale=1.f;
    }
    float m0=0,m1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) { m0+=S[i]*k[i];m1+=S[i+1]*k[i+1]; }
    reinterpret_cast<float*>(&memx[r&1][lane])[sub]=m0+m1;
    barrier();
    if (sub==0 && r>0) {
      const float4 t=yx[(r-1)&1][lane];
      a.y[(base+r-1)*128+j]=((t.x+t.y)+(t.z+t.w))*pscale;
    }
    if (u==0) { conv(r0+4,slot^1); load(r0+8); }
    const float4 m4=memx[r&1][lane];
    const float mem=(m4.x+m4.y)+(m4.z+m4.w), v=vx[slot][u][lane];
    const float delta=beta*(v-mem*scale), d=delta/scale;
    // Issue the loads of row r+1 after the LDS reads are waited for (the SGPR
    // loads share lgkmcnt with LDS): the zero z depends on d. The loads are
    // unconditional (the last row loads row R-1 again), so no branch merges the
    // wait counters and the update below waits only for the loads of row r.
    int z;
    asm volatile("s_mov_b32 %0, 0" : "=s"(z) : "v"(d));
    // q of the next block's first row loads at the top of that block: a vector
    // load in flight at the loop latch makes the compiler wait for it there.
    kl(kb[P^1],min(r+1,R-1)+z);
    if constexpr (u<3) ql(qb[P^1],min(r+1,R-1)+z);
    float y0=0,y1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) {
      S[i]+=k[i]*d;S[i+1]+=k[i+1]*d;
      y0+=S[i]*q[i];y1+=S[i+1]*q[i+1];
    }
    reinterpret_cast<float*>(&yx[r&1][lane])[sub]=y0+y1;
    pscale=scale;
  };
#pragma unroll 1
  for (int r0=0;r0<R;r0+=4) {
    const int slot=(r0>>2)&1, rl=r0+(lane&3), rn=rl+4;
    const float dq=rl<R ? expf(gn) : 1.f, bq=rl<R ? bn : 0.f;
    gn=rn<R ? a.g[base+rn] : 0.f; bn=rn<R ? a.beta[base+rn] : 0.f;
    ql(qb[0],r0);
    step(std::integral_constant<int,0>{},r0,slot,dq,bq);
    if (r0+1<R) step(std::integral_constant<int,1>{},r0,slot,dq,bq);
    if (r0+2<R) step(std::integral_constant<int,2>{},r0,slot,dq,bq);
    if (r0+3<R) step(std::integral_constant<int,3>{},r0,slot,dq,bq);
  }
  __syncthreads();
  if (sub==0 && R>0) {
    const float4 t=yx[(R-1)&1][lane];
    a.y[(base+R-1)*128+j]=((t.x+t.y)+(t.z+t.w))*pscale;
  }
#pragma unroll
  for (int i=0;i<32;++i) S[i]*=scale;
#pragma unroll
  for (int i=0;i<32;++i) dst[i*128]=S[i];
}
}  // namespace crucible::prefill
