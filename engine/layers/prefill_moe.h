// Compact expert transforms and stable combine preserve each router slot.
#pragma once
#include "moe_window_policy.h"
#include "../model/flashnext/model.h"
#include "../exl3/prefill_linear.h"
#include "kernels/prefill_moe.h"
#include "../switches.h"
#include "nonfinite_probe.h"
#if defined(CRUCIBLE_MOE_WINDOW_PROBE) && CRUCIBLE_MOE_WINDOW_PROBE
#include "../exl3/kernels/exl3_moe_window.h"
#endif

namespace crucible {
struct PrefillMoe {
  prefill::MoeDispatch dispatch;
  LinJob *gu, *down;
  Exl3GemmJob *g_gu, *g_down;
  Exl3GemmItem *i_gu, *i_down;
  int *item_counts, *inverse;
  int tile_rows;  // rows per GEMM item; one item decodes its weight tiles once
  bool packed;    // GEMM inputs use the packed [I/16][M][16] layout per expert
  float *pg, *pu, *pd;
  __half *xg, *xu, *xd;
  const float* x;
  const __half* xh;  // fp16 copy of x (x is fp16-exact); halves the gather reads
  // Fused input transform + gate/up (PrefillMoeFxKernel): X items, the first
  // unit (expert row tile) and X item of each expert, finished X items per
  // unit, and the queue counters [next X, next G, X item count].
  int *fx_items, *fx_uoff, *fx_xoff, *fx_ready, *fx_count;
};

__device__ __forceinline__ LinJob PrefillExpertJob(const uint32_t* tiles, const __half* suh,
    const __half* svh, __half* xh, float* partial, int I, int O, int M, int K) {
  LinJob j{};
  j.tiles=tiles; j.suh=suh; j.svh=svh; j.xh=xh; j.partial=partial;
  j.I=I; j.O=O; j.K=K; j.S=1; j.R=M; j.lds=I; j.ldd=O;
  return j;
}
// MoE bodies read engine-order tiles, so front_s stays 0 here and in
// PrefillUniform(Exl3GemmJob); ChoosePrefillRows rejects front-order MoE weights.
__device__ __forceinline__ Exl3GemmJob PrefillGemmJob(const LinJob& j) {
  return {j.tiles,j.xh,j.partial,nullptr,j.I,j.O,j.R,1,j.K,j.R};
}

// Packed MoE GEMM items cover kMoeNb column tiles per wave (64 * kMoeNb
// columns); unpacked items cover 64 columns.
constexpr int kMoeNb=2;
__device__ __forceinline__ int PrefillMoeNb(const PrefillMoe& a) { return a.packed ? kMoeNb : 1; }
// Gate/up items of expert e per row tile: 10 / nb column groups x gate, up.
__device__ __forceinline__ int PrefillGateUpPer(const PrefillMoe& a) { return 20/PrefillMoeNb(a); }
__device__ __forceinline__ void PrefillGateUpItems(PrefillMoe a, int e, int first, int tiles) {
  const int T=a.tile_rows, nb=PrefillMoeNb(a), per=20/nb;
  for (int m=0; m<tiles; ++m)
    for (int b=0; b<10/nb; ++b) {
      a.i_gu[first+m*per+b*2]={e*2,0,m*T,b*4*nb};
      a.i_gu[first+m*per+b*2+1]={e*2+1,0,m*T,b*4*nb};
    }
}
__device__ __forceinline__ void PrefillDownItems(PrefillMoe a, int e, int M) {
  const int T=a.tile_rows, tiles=(M+T-1)/T, nb=PrefillMoeNb(a), per=40/nb;
  const int bd=atomicAdd(a.item_counts+1,tiles*per);
  for (int m=0; m<tiles; ++m)
    for (int b=0; b<per; ++b) a.i_down[bd+m*per+b]={e,0,m*T,b*4*nb};
}
__device__ __forceinline__ void PrefillExpertItems(PrefillMoe a, int e, int M) {
  const int T=a.tile_rows, tiles=(M+T-1)/T;
  PrefillGateUpItems(a,e,atomicAdd(a.item_counts,tiles*PrefillGateUpPer(a)),tiles);
  PrefillDownItems(a,e,M);
}

// Jobs of expert e (thread e of a 512-thread plan); thread 0 also writes the
// shared expert's jobs.
__device__ __forceinline__ void PrefillMoeJobs(PrefillMoe a, const ExpertPtrs* experts,
    Lin sg, Lin su, Lin sd) {
  const int e=threadIdx.x, R=a.dispatch.rows;
  const int M=a.dispatch.counts[e], off=a.dispatch.offsets[e];
  const auto p=experts[e];
  a.gu[e*2]=PrefillExpertJob(p.tiles[0],p.suh[0],p.svh[0],a.xg+(size_t)off*kHidden,
                           a.pg+(size_t)off*kMoeInter,kHidden,kMoeInter,M,p.K[0]);
  a.gu[e*2+1]=PrefillExpertJob(p.tiles[1],p.suh[1],p.svh[1],a.xu+(size_t)off*kHidden,
                           a.pu+(size_t)off*kMoeInter,kHidden,kMoeInter,M,p.K[1]);
  a.down[e]=PrefillExpertJob(p.tiles[2],p.suh[2],p.svh[2],a.xd+(size_t)off*kMoeInter,
                           a.pd+(size_t)off*kHidden,kMoeInter,kHidden,M,p.K[2]);
  a.g_gu[e*2]=PrefillGemmJob(a.gu[e*2]); a.g_gu[e*2+1]=PrefillGemmJob(a.gu[e*2+1]);
  a.g_down[e]=PrefillGemmJob(a.down[e]);
  for (int i=e; i<R*kTopK; i+=512) a.inverse[a.dispatch.pairs[i]]=i;
  if (e==0) {
    const int off=R*kTopK, sh=kExperts;
    a.gu[sh*2]=PrefillExpertJob(sg.tiles,sg.suh,sg.svh,a.xg+(size_t)off*kHidden,
                             a.pg+(size_t)off*kMoeInter,kHidden,kMoeInter,R,sg.K);
    a.gu[sh*2+1]=PrefillExpertJob(su.tiles,su.suh,su.svh,a.xu+(size_t)off*kHidden,
                             a.pu+(size_t)off*kMoeInter,kHidden,kMoeInter,R,su.K);
    a.down[sh]=PrefillExpertJob(sd.tiles,sd.suh,sd.svh,a.xd+(size_t)off*kMoeInter,
                             a.pd+(size_t)off*kHidden,kMoeInter,kHidden,R,sd.K);
    a.g_gu[sh*2]=PrefillGemmJob(a.gu[sh*2]); a.g_gu[sh*2+1]=PrefillGemmJob(a.gu[sh*2+1]);
    a.g_down[sh]=PrefillGemmJob(a.down[sh]);
  }
}

__global__ void __launch_bounds__(512) PrefillMoePlanKernel(PrefillMoe a,
    const ExpertPtrs* experts, Lin sg, Lin su, Lin sd) {
  PrefillMoeJobs(a,experts,sg,su,sd);
  const int e=threadIdx.x;
  PrefillExpertItems(a,e,a.dispatch.counts[e]);
  if (e==0) PrefillExpertItems(a,kExperts,a.dispatch.rows);
}

// Exclusive prefix sum over a 512-thread block; *total gets the block sum.
__device__ __forceinline__ int PrefillBlockScan512(int v, int* s, int* total) {
  const int t=threadIdx.x, lane=t&31, wave=t>>5;
  int x=v;
#pragma unroll
  for (int o=1; o<32; o<<=1) { const int y=__shfl_up(x,o); if (lane>=o) x+=y; }
  if (lane==31) s[wave]=x;
  __syncthreads();
  if (t<16) {
    int w=s[t];
#pragma unroll
    for (int o=1; o<16; o<<=1) { const int y=__shfl_up(w,o,16); if (t>=o) w+=y; }
    s[t]=w;
  }
  __syncthreads();
  const int r=(wave ? s[wave-1] : 0)+x-v;
  *total=s[15];
  __syncthreads();
  return r;
}

// Units (expert row tiles) and X items in expert order, then the shared
// expert. Gate/up items of unit u are i_gu[20u, 20u + 20). X items of expert
// e are fx_items[xoff[e] + 2 * group + which], one per 16 compact rows.
__device__ __forceinline__ void PrefillFxUnits(PrefillMoe a, int e, int M, int ut, int ux) {
  const int T=a.tile_rows, tiles=(M+T-1)/T, groups=(M+15)/16;
  a.fx_uoff[e]=ut; a.fx_xoff[e]=ux;
  for (int m=0; m<tiles; ++m) a.fx_ready[ut+m]=0;
  PrefillGateUpItems(a,e,ut*PrefillGateUpPer(a),tiles);
  for (int g=0; g<groups; ++g) {
    a.fx_items[ux+g*2]=e<<16|g<<1;
    a.fx_items[ux+g*2+1]=e<<16|g<<1|1;
  }
}
__global__ void __launch_bounds__(512) PrefillMoeFxPlanKernel(PrefillMoe a,
    const ExpertPtrs* experts, Lin sg, Lin su, Lin sd) {
  __shared__ int scan[16];
  PrefillMoeJobs(a,experts,sg,su,sd);
  const int e=threadIdx.x, R=a.dispatch.rows, T=a.tile_rows, M=a.dispatch.counts[e];
  int tiles, xs;
  const int ut=PrefillBlockScan512((M+T-1)/T,scan,&tiles);
  const int ux=PrefillBlockScan512(2*((M+15)/16),scan,&xs);
  PrefillFxUnits(a,e,M,ut,ux);
  PrefillDownItems(a,e,M);
  if (e==0) {
    PrefillFxUnits(a,kExperts,R,tiles,xs);
    PrefillDownItems(a,kExperts,R);
    a.item_counts[0]=(tiles+(R+T-1)/T)*PrefillGateUpPer(a);
    a.fx_count[0]=0; a.fx_count[1]=0; a.fx_count[2]=xs+2*((R+15)/16);
  }
}

__global__ void __launch_bounds__(256) PrefillMoeInputKernel(PrefillMoe a) {
  const int w=blockIdx.x*8+(threadIdx.x>>5), lane=threadIdx.x&31;
  const int rows=a.dispatch.rows*kSlots;
  if (w>=rows*40) return;
  const int p=w/40, which=(w/20)&1, blk=w%20;
  const bool shared=p>=a.dispatch.rows*kTopK;
  const int pair=shared ? p-a.dispatch.rows*kTopK : a.dispatch.pairs[p];
  const int e=shared ? kExperts : a.dispatch.selected[pair];
  const int src=shared ? pair : pair/kTopK;
  const int r=shared ? pair : p-a.dispatch.offsets[e];
  const LinJob j=a.gu[e*2+which];
  const int d=blk*128+lane*4;
  float v[4];
#pragma unroll
  for (int i=0; i<4; ++i) v[i]=RoundH(RoundH(a.x[(size_t)src*kHidden+d+i])*__half2float(j.suh[d+i]));
  Wht128(v);
  __half* dst=a.packed ? j.xh+PackedIndex(r,d,j.R) : j.xh+(size_t)r*kHidden+d;
#pragma unroll
  for (int i=0; i<4; ++i) dst[i]=__float2half(v[i]*kHadNorm);
}

// Packed variant: a block transforms one 128-column block of 16 consecutive
// compact rows, then stores each 16-column group of those rows as one run, so
// rows of the same expert give 512-byte writes instead of 32-byte pieces.
__global__ void __launch_bounds__(256) PrefillMoeInputPackedKernel(PrefillMoe a) {
  __shared__ __attribute__((aligned(16))) __half tile[16][136];
  __shared__ __half* base[16];
  __shared__ int local[16], height[16];
  const int lane=threadIdx.x&31, wave=threadIdx.x>>5, t=threadIdx.x;
  // The 40 blocks of one row group run back to back, so the gathered x rows
  // stay in L2 for all of them.
  const int which=blockIdx.x/20, blk=blockIdx.x%20, p0=blockIdx.y*16;
  const int rows=a.dispatch.rows*kSlots, d=blk*128+lane*4;
#pragma unroll
  for (int i=0; i<2; ++i) {
    const int lr=wave*2+i, p=p0+lr;
    if (p>=rows) { if (lane==0) base[lr]=nullptr; continue; }
    const bool shared=p>=a.dispatch.rows*kTopK;
    const int pair=shared ? p-a.dispatch.rows*kTopK : a.dispatch.pairs[p];
    const int e=shared ? kExperts : a.dispatch.selected[pair];
    const int src=shared ? pair : pair/kTopK;
    const LinJob j=a.gu[e*2+which];
    float v[4];
#pragma unroll
    for (int k=0; k<4; ++k) v[k]=RoundH(__half2float(a.xh[(size_t)src*kHidden+d+k])*__half2float(j.suh[d+k]));
    Wht128(v);
#pragma unroll
    for (int k=0; k<4; ++k) tile[lr][lane*4+k]=__float2half(v[k]*kHadNorm);
    if (lane==0) {
      base[lr]=j.xh; local[lr]=shared ? pair : p-a.dispatch.offsets[e]; height[lr]=j.R;
    }
  }
  __syncthreads();
  const int g=t>>5, lr=(t>>1)&15, h=t&1;
  if (!base[lr]) return;
  const int c=blk*128+g*16+h*8;
  *reinterpret_cast<uint4*>(base[lr]+PackedIndex(local[lr],c,height[lr]))=
      *reinterpret_cast<const uint4*>(&tile[lr][g*16+h*8]);
}

// Items with more than 32 live rows skip their padded row tiles at run time.
// Packed items use exl3-gemm's measured compact choice for every row count.
// HALF (down projection, fp16 pd) needs the packed body.
// COPY > 0: the same code in separate instantiations of the bodies, for the K7 jobs of the
// kPrefillMoeK7Half kernels. The bodies hold lambdas that the inliner treats by their number of
// callers, so a K7 body shared with new kernels changed the KMAX 7 kernels' code (isa_same).
template<int K,int RM,bool P,bool HALF=false,int COPY=0> __device__ __forceinline__ void PrefillMoeGemmBody(const Exl3GemmJob& j,
    const Exl3GemmItem& it, uint32_t* ws) {
  static_assert(P || !HALF);
#if defined(CRUCIBLE_MOE_WINDOW_PROBE) && CRUCIBLE_MOE_WINDOW_PROBE
  // Keep short prompts and all dense consumers on their current bodies.
  if constexpr(P&&RM==6) {
    static_assert(kMoeNb==2);MoeLiveWindowBody<K,RM,HALF,COPY>(j,it,ws);return;
  }
#endif
  if (P) PrefillWideBody<K,RM,kMoeNb,true,true,true,HALF,false,COPY>(j,it,ws);
  else if (j.M-it.m0<=16) Exl3GemmBody<K,1,true,false,1,P,P,false,P>(j,it,ws);
  else if (RM==2 || j.M-it.m0<=32) Exl3GemmBody<K,2,true,false,1,P,P,false,P>(j,it,ws);
  else Exl3GemmBody<K,RM,true,false,1,P,P,true,P>(j,it,ws);
}
// Items and jobs come from device lists, so the compiler cannot prove they are
// wave-uniform. Without readfirstlane every buffer load of the GEMM body runs
// in a waterfall loop over its resource descriptor (about 180 extra
// instructions per k block with 8 row tiles).
__device__ __forceinline__ int PrefillUniform(int v) { return __builtin_amdgcn_readfirstlane(v); }
template<class T> __device__ __forceinline__ T* PrefillUniform(T* p) {
  const uint64_t u=reinterpret_cast<uint64_t>(p);
  const uint64_t lo=(uint32_t)PrefillUniform((int)(uint32_t)u), hi=(uint32_t)PrefillUniform((int)(u>>32));
  return reinterpret_cast<T*>(lo|(hi<<32));
}
__device__ __forceinline__ Exl3GemmItem PrefillUniform(Exl3GemmItem it) {
  return {PrefillUniform(it.job),PrefillUniform(it.s),PrefillUniform(it.m0),PrefillUniform(it.b0)};
}
__device__ __forceinline__ Exl3GemmJob PrefillUniform(Exl3GemmJob j) {
  return {PrefillUniform(j.tiles),PrefillUniform(j.x),PrefillUniform(j.partial),PrefillUniform(j.row_ids),
          PrefillUniform(j.I),PrefillUniform(j.O),PrefillUniform(j.M),PrefillUniform(j.S),
          PrefillUniform(j.K),PrefillUniform(j.Xrows)};
}
// KMAX 6 adds the K6 body (4.05); KMAX 5 keeps the K3..5 kernels' registers. KMAX 7 or 8 (the
// h6_ng8 packs' shared experts) adds the K7 or K8 body next to K3..5; kPrefillMoeK7Half (decode_support.h)
// adds K7 and K3.5 (code 35: 3.18, 3.68); kPrefillMoeK6Low adds K2, K2.5 (code 25) and copies of the K3
// and K6 bodies (2.18, 2.68); any other width traps.
template<int RM,bool P,int KMAX=5,bool HALF=false> __global__ void __launch_bounds__(128) PrefillMoeGemmKernel(const Exl3GemmJob* jobs,
    const Exl3GemmItem* items, const int* count) {
#if defined(CRUCIBLE_MOE_WINDOW_PROBE) && CRUCIBLE_MOE_WINDOW_PROBE
  constexpr int words=P&&RM==6?kGemmPairWords:kGemmWeightWords;
  __shared__ uint32_t ws[words];
#else
  __shared__ uint32_t ws[kGemmWeightWords];
#endif
  const int n=PrefillUniform(*count);
  for (int i=blockIdx.x; i<n; i+=gridDim.x) {
    const auto it=PrefillUniform(items[i]); const auto j=PrefillUniform(jobs[it.job]);
    if constexpr (KMAX==kPrefillMoeK6Low) {
      if (j.K==2) PrefillMoeGemmBody<2,RM,P,HALF>(j,it,ws);
      else if (j.K==25) PrefillMoeGemmBody<25,RM,P,HALF>(j,it,ws);
      else if (j.K==3) PrefillMoeGemmBody<3,RM,P,HALF,1>(j,it,ws);
      else if (j.K==6) PrefillMoeGemmBody<6,RM,P,HALF,1>(j,it,ws);
      else __builtin_trap();
    }
    else if (j.K==3) PrefillMoeGemmBody<3,RM,P,HALF>(j,it,ws);
    else if (j.K==4) PrefillMoeGemmBody<4,RM,P,HALF>(j,it,ws);
    else if (j.K==5) PrefillMoeGemmBody<5,RM,P,HALF>(j,it,ws);
    else if constexpr (KMAX==6) { if (j.K==6) PrefillMoeGemmBody<6,RM,P,HALF>(j,it,ws); }
    else if constexpr (KMAX==kPrefillMoeK7Half) {
      if (j.K==7) PrefillMoeGemmBody<7,RM,P,HALF,1>(j,it,ws);
      else if (j.K==35) PrefillMoeGemmBody<35,RM,P,HALF>(j,it,ws);
      else __builtin_trap();
    }
    else if constexpr (KMAX>=7) { if (j.K==KMAX) PrefillMoeGemmBody<KMAX,RM,P,HALF>(j,it,ws); else __builtin_trap(); }
    __syncthreads();
  }
}
// half: fp16 partial output (the down projection's pd); packed RM 2, 4 or 6.
inline bool PrefillMoeHalfOut(PrefillGemmShape g) { return g.packed && (g.rm==6 || g.rm==4 || g.rm==2); }
// KW 7, 8, kPrefillMoeK7Half or kPrefillMoeK6Low: packed RM 2, 4 and 6 only, as K6.
template <int KW>
inline void LaunchPrefillMoeGemmWide(PrefillGemmShape g, const Exl3GemmJob* jobs, const Exl3GemmItem* items,
                                     const int* count, bool half) {
  if (!g.packed || (g.rm != 6 && g.rm != 4 && g.rm != 2)) {
    std::fprintf(stderr, "K%d MoE GEMM supports packed RM 2, 4 and 6\n", KW); std::abort();
  }
  if (half) {
    if (g.rm==6) PrefillMoeGemmKernel<6,true,KW,true><<<2048,128>>>(jobs,items,count);
    else if (g.rm==4) PrefillMoeGemmKernel<4,true,KW,true><<<2048,128>>>(jobs,items,count);
    else PrefillMoeGemmKernel<2,true,KW,true><<<2048,128>>>(jobs,items,count);
    return;
  }
  if (g.rm==6) PrefillMoeGemmKernel<6,true,KW><<<2048,128>>>(jobs,items,count);
  else if (g.rm==4) PrefillMoeGemmKernel<4,true,KW><<<2048,128>>>(jobs,items,count);
  else PrefillMoeGemmKernel<2,true,KW><<<2048,128>>>(jobs,items,count);
}
inline void LaunchPrefillMoeGemm(PrefillGemmShape g, const Exl3GemmJob* jobs, const Exl3GemmItem* items,
                                 const int* count, bool k6 = false, bool half = false) {
  if (half) {
    if (!PrefillMoeHalfOut(g)) { std::fprintf(stderr, "fp16 MoE output needs packed RM 2, 4 or 6\n"); std::abort(); }
    if (k6) {
      if (g.rm==6) PrefillMoeGemmKernel<6,true,6,true><<<2048,128>>>(jobs,items,count);
      else if (g.rm==4) PrefillMoeGemmKernel<4,true,6,true><<<2048,128>>>(jobs,items,count);
      else PrefillMoeGemmKernel<2,true,6,true><<<2048,128>>>(jobs,items,count);
    } else {
      if (g.rm==6) PrefillMoeGemmKernel<6,true,5,true><<<2048,128>>>(jobs,items,count);
      else if (g.rm==4) PrefillMoeGemmKernel<4,true,5,true><<<2048,128>>>(jobs,items,count);
      else PrefillMoeGemmKernel<2,true,5,true><<<2048,128>>>(jobs,items,count);
    }
    return;
  }
  if (k6) {
    if (!g.packed || (g.rm != 6 && g.rm != 4 && g.rm != 2)) {
      std::fprintf(stderr, "K6 MoE GEMM supports packed RM 2, 4 and 6\n"); std::abort();
    }
    if (g.rm==6) PrefillMoeGemmKernel<6,true,6><<<2048,128>>>(jobs,items,count);
    else if (g.rm==4) PrefillMoeGemmKernel<4,true,6><<<2048,128>>>(jobs,items,count);
    else PrefillMoeGemmKernel<2,true,6><<<2048,128>>>(jobs,items,count);
    return;
  }
  if (g.packed) {
    if (g.rm==8) PrefillMoeGemmKernel<8,true><<<2048,128>>>(jobs,items,count);
    else if (g.rm==7) PrefillMoeGemmKernel<7,true><<<2048,128>>>(jobs,items,count);
    else if (g.rm==6) PrefillMoeGemmKernel<6,true><<<2048,128>>>(jobs,items,count);
    else if (g.rm==4) PrefillMoeGemmKernel<4,true><<<2048,128>>>(jobs,items,count);
    else PrefillMoeGemmKernel<2,true><<<2048,128>>>(jobs,items,count);
    return;
  }
  if (g.rm==8) PrefillMoeGemmKernel<8,false><<<2048,128>>>(jobs,items,count);
  else if (g.rm==4) PrefillMoeGemmKernel<4,false><<<2048,128>>>(jobs,items,count);
  else PrefillMoeGemmKernel<2,false><<<2048,128>>>(jobs,items,count);
}

// X item: the packed input transform of 16 compact rows of one expert for
// gate or up, all 2560 columns; the same arithmetic as
// PrefillMoeInputPackedKernel. Wave w transforms rows 4w..4w+3 of each
// 128-column block; the block stores each 16-column group as one 512-byte run.
// The loads run kFxDepth column blocks ahead and the LDS tile is double
// buffered (one barrier per column block), so an X item does not wait for a
// load latency per column block. t is the thread's index in its 128-thread
// group: a 256-thread block runs two X items at once, one per group (every
// barrier is block-wide, so both groups take the same barrier count).
constexpr int kFxDepth=2;
__device__ __forceinline__ void PrefillFxTransform(const PrefillMoe& a, int x, __half (*tile)[16][136], int t) {
  const int code=PrefillUniform(a.fx_items[x]), e=code>>16, rg=(code>>1)&0x7fff, which=code&1;
  const int lane=t&31, wave=t>>5;
  const __half* suh=PrefillUniform(a.gu[e*2+which].suh);
  __half* dst=PrefillUniform(a.gu[e*2+which].xh);
  const int M=PrefillUniform(a.gu[e*2+which].R);
  const bool shared=e==kExperts;
  const int off=shared ? 0 : a.dispatch.offsets[e];
  const __half* src[4];
#pragma unroll
  for (int i=0; i<4; ++i) {
    const int l=rg*16+wave*4+i;
    const int r=l<M ? (shared ? l : a.dispatch.pairs[off+l]/kTopK) : -1;
    // Wave-uniform row pointers keep the row branches scalar.
    src[i]=r>=0 ? PrefillUniform(a.xh+(size_t)r*kHidden) : nullptr;
  }
  uint2 xv[kFxDepth][4], sv[kFxDepth];
  auto load=[&](int slot, int cb) {
    sv[slot]=*reinterpret_cast<const uint2*>(suh+cb*128+lane*4);
#pragma unroll
    for (int i=0; i<4; ++i) if (src[i]) xv[slot][i]=*reinterpret_cast<const uint2*>(src[i]+cb*128+lane*4);
  };
#pragma unroll
  for (int k=0; k<kFxDepth; ++k) load(k,k);
  static_assert(kHidden/128%kFxDepth==0);
  for (int cb0=0; cb0<kHidden/128; cb0+=kFxDepth) {
#pragma unroll
    for (int k=0; k<kFxDepth; ++k) {
      const int cb=cb0+k;
      __half (*tl)[136]=tile[k&1];
      // The fp16 product of two fp16 values is RoundH(x * s) exactly: the
      // fp32 product is exact and both round once to nearest even.
      const __half2 s01=*reinterpret_cast<const __half2*>(&sv[k].x), s23=*reinterpret_cast<const __half2*>(&sv[k].y);
#pragma unroll
      for (int i=0; i<4; ++i) {
        if (!src[i]) continue;
        const float2 p01=__half22float2(__hmul2(*reinterpret_cast<const __half2*>(&xv[k][i].x),s01));
        const float2 p23=__half22float2(__hmul2(*reinterpret_cast<const __half2*>(&xv[k][i].y),s23));
        float v[4]={p01.x,p01.y,p23.x,p23.y};
        Wht128Dpp(v);
        __half2 h[2]={__floats2half2_rn(v[0]*kHadNorm,v[1]*kHadNorm),__floats2half2_rn(v[2]*kHadNorm,v[3]*kHadNorm)};
        *reinterpret_cast<uint2*>(&tl[wave*4+i][lane*4])=*reinterpret_cast<const uint2*>(h);
      }
      if (cb+kFxDepth<kHidden/128) load(k,cb+kFxDepth);
      __syncthreads();
#pragma unroll
      for (int h=0; h<2; ++h) {
        const int idx=t+h*128, g=idx>>5, lr=(idx>>1)&15, l=rg*16+lr;
        if (l<M)
          *reinterpret_cast<uint4*>(dst+PackedIndex(l,cb*128+g*16+(idx&1)*8,M))=
              *reinterpret_cast<const uint4*>(&tl[lr][g*16+(idx&1)*8]);
      }
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
  __syncthreads();
  if (t==0) atomicAdd(a.fx_ready+a.fx_uoff[e]+rg*16/a.tile_rows,1);
  // Without this barrier the compiler sinks the lane-0 increment out of the
  // X loop (into the loop exit block), and a G wait that needs it deadlocks.
  __syncthreads();
}

// Fused MoE input transform and gate/up GEMM. The persistent blocks take two
// queues. A block claims the next G (gate/up) item, then claims X items until
// the X queue is kFxLead items past the last X item that the G item needs,
// then waits for its unit's X items. X items never wait and every awaited X
// item is held by a running block, so the queues cannot deadlock. Each
// transformed row is written shortly before the GEMM reads it, so the A reads
// mostly hit the MALL, and the transform overlaps the GEMM.
constexpr int kFxLead=128;
template<int RM,int KMAX=5> __global__ void __launch_bounds__(128) PrefillMoeFxKernel(PrefillMoe a, int lead) {
  // The GEMM B exchange and the X tiles never live at the same time.
  constexpr int kTileWords=2*16*136/2;
  static_assert(kTileWords>=kGemmWeightWords);
#if defined(CRUCIBLE_MOE_WINDOW_PROBE) && CRUCIBLE_MOE_WINDOW_PROBE
  static_assert(kTileWords>=kGemmPairWords);
#endif
  __shared__ __attribute__((aligned(16))) uint32_t lds[kTileWords];
#if defined(CRUCIBLE_MOE_WINDOW_PROBE) && CRUCIBLE_MOE_WINDOW_PROBE
  // The final X padding word is outside all B exchanges. Reuse it for the
  // claim so the live-window body stays within8704 bytes, including queue state.
  static_assert(kTileWords-1>=kGemmPairWords);
  static_assert((2*(kTileWords-1))%136>=128);
  int& claim=*reinterpret_cast<int*>(lds+kTileWords-1);
#else
  __shared__ int claim;
#endif
  uint32_t* ws=lds;
  auto* tile=reinterpret_cast<__half (*)[16][136]>(lds);
  const int t=threadIdx.x, T=a.tile_rows;
  const int ng=PrefillUniform(a.item_counts[0]), nx=PrefillUniform(a.fx_count[2]);
  for (;;) {
    if (t==0) claim=atomicAdd(a.fx_count+1,1);
    __syncthreads();
    const int g=PrefillUniform(claim);
    __syncthreads();
    Exl3GemmItem it{}; Exl3GemmJob j{};
    int need=nx;
    if (g<ng) {
      it=PrefillUniform(a.i_gu[g]); j=PrefillUniform(a.g_gu[it.job]);
      need=min(nx,a.fx_xoff[it.job>>1]+2*min((j.M+15)/16,(it.m0+T)/16));
    }
    const int limit=min(nx,need+lead);
    for (;;) {
      if (t==0) claim=__hip_atomic_load(a.fx_count,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)<limit
                      ? atomicAdd(a.fx_count,1) : nx;
      __syncthreads();
      const int x=PrefillUniform(claim);
      __syncthreads();
      if (x>=nx) break;
      PrefillFxTransform(a,x,tile,threadIdx.x);
    }
    if (g>=ng) return;
    const int target=2*((min(j.M,it.m0+T)-it.m0+15)/16);
    if (t==0)
      while (__hip_atomic_load(a.fx_ready+g/(20/kMoeNb),__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)<target)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
    if constexpr (KMAX==kPrefillMoeK6Low) {
      if (j.K==2) PrefillMoeGemmBody<2,RM,true>(j,it,ws);
      else if (j.K==25) PrefillMoeGemmBody<25,RM,true>(j,it,ws);
      else if (j.K==3) PrefillMoeGemmBody<3,RM,true,false,1>(j,it,ws);
      else if (j.K==6) PrefillMoeGemmBody<6,RM,true,false,1>(j,it,ws);
      else __builtin_trap();
    }
    else if (j.K==3) PrefillMoeGemmBody<3,RM,true>(j,it,ws);
    else if (j.K==4) PrefillMoeGemmBody<4,RM,true>(j,it,ws);
    else if (j.K==5) PrefillMoeGemmBody<5,RM,true>(j,it,ws);
    else if constexpr (KMAX==6) { if (j.K==6) PrefillMoeGemmBody<6,RM,true>(j,it,ws); }
    else if constexpr (KMAX==kPrefillMoeK7Half) {
      if (j.K==7) PrefillMoeGemmBody<7,RM,true,false,1>(j,it,ws);
      else if (j.K==35) PrefillMoeGemmBody<35,RM,true>(j,it,ws);
      else __builtin_trap();
    }
    else if constexpr (KMAX>=7) { if (j.K==KMAX) PrefillMoeGemmBody<KMAX,RM,true>(j,it,ws); else __builtin_trap(); }
    __syncthreads();
  }
}
inline int PrefillFxLead() {
  // A negative lead would stop every block from claiming X items while all
  // wait on fx_ready, so a bad value is rejected.
  static const int lead=[] {
    const char* v=crucible::SwitchText("CRUCIBLE_PF_FX_LEAD");
    if (!v) return kFxLead;
    char* end=nullptr;
    const long x=std::strtol(v,&end,10);
    if (!*v || *end || x<0 || x>(1<<20)) {
      std::fprintf(stderr,"CRUCIBLE_PF_FX_LEAD must be an integer in [0, 1048576]\n");
      std::exit(2);
    }
    return (int)x;
  }();
  return lead;
}
inline void LaunchPrefillMoeFx(int rm, const PrefillMoe& a, bool k6 = false) {
  const int lead=PrefillFxLead();
  if (k6) {
    if (rm!=6 && rm!=4 && rm!=2) { std::fprintf(stderr,"K6 MoE queue supports RM 2, 4 and 6\n"); std::abort(); }
    if (rm==6) PrefillMoeFxKernel<6,6><<<2048,128>>>(a,lead);
    else if (rm==4) PrefillMoeFxKernel<4,6><<<2048,128>>>(a,lead);
    else PrefillMoeFxKernel<2,6><<<2048,128>>>(a,lead);
    return;
  }
  if (rm==8) PrefillMoeFxKernel<8><<<2048,128>>>(a,lead);
  else if (rm==7) PrefillMoeFxKernel<7><<<2048,128>>>(a,lead);
  else if (rm==6) PrefillMoeFxKernel<6><<<2048,128>>>(a,lead);
  else if (rm==4) PrefillMoeFxKernel<4><<<2048,128>>>(a,lead);
  else PrefillMoeFxKernel<2><<<2048,128>>>(a,lead);
}

// The engine's MoE launches by the layer's PrefillMoeKmax (decode_support.h): 6 selects the K6
// kernels as before; 7, 8, kPrefillMoeK7Half and kPrefillMoeK6Low the wide kernels (packed RM 2, 4 and 6 only).
inline void LaunchPrefillMoeGemmK(PrefillGemmShape g, const Exl3GemmJob* jobs, const Exl3GemmItem* items,
                                  const int* count, int kmax, bool half = false) {
  if (kmax == kPrefillMoeK7Half) LaunchPrefillMoeGemmWide<kPrefillMoeK7Half>(g, jobs, items, count, half);
  else if (kmax == kPrefillMoeK6Low) LaunchPrefillMoeGemmWide<kPrefillMoeK6Low>(g, jobs, items, count, half);
  else if (kmax == 7) LaunchPrefillMoeGemmWide<7>(g, jobs, items, count, half);
  else if (kmax == 8) LaunchPrefillMoeGemmWide<8>(g, jobs, items, count, half);
  else LaunchPrefillMoeGemm(g, jobs, items, count, kmax == 6, half);
}
inline void LaunchPrefillMoeFxK(int rm, const PrefillMoe& a, int kmax) {
  if (kmax < 7) return LaunchPrefillMoeFx(rm, a, kmax == 6);
  if (rm!=6 && rm!=4 && rm!=2) { std::fprintf(stderr,"K%d MoE queue supports RM 2, 4 and 6\n",kmax); std::abort(); }
  const int lead=PrefillFxLead();
  if (kmax == kPrefillMoeK7Half) {
    if (rm==6) PrefillMoeFxKernel<6,kPrefillMoeK7Half><<<2048,128>>>(a,lead);
    else if (rm==4) PrefillMoeFxKernel<4,kPrefillMoeK7Half><<<2048,128>>>(a,lead);
    else PrefillMoeFxKernel<2,kPrefillMoeK7Half><<<2048,128>>>(a,lead);
  } else if (kmax == kPrefillMoeK6Low) {
    if (rm==6) PrefillMoeFxKernel<6,kPrefillMoeK6Low><<<2048,128>>>(a,lead);
    else if (rm==4) PrefillMoeFxKernel<4,kPrefillMoeK6Low><<<2048,128>>>(a,lead);
    else PrefillMoeFxKernel<2,kPrefillMoeK6Low><<<2048,128>>>(a,lead);
  } else if (kmax == 7) {
    if (rm==6) PrefillMoeFxKernel<6,7><<<2048,128>>>(a,lead);
    else if (rm==4) PrefillMoeFxKernel<4,7><<<2048,128>>>(a,lead);
    else PrefillMoeFxKernel<2,7><<<2048,128>>>(a,lead);
  } else {
    if (rm==6) PrefillMoeFxKernel<6,8><<<2048,128>>>(a,lead);
    else if (rm==4) PrefillMoeFxKernel<4,8><<<2048,128>>>(a,lead);
    else PrefillMoeFxKernel<2,8><<<2048,128>>>(a,lead);
  }
}

// One wave per row: ten rounds of a wave argmax (ties to the lower expert),
// so the k-th winner has the same rank as in MoeRouteKernel, and the weights
// use the same expressions and sum order. Equal to MoeRouteKernel for rows
// with at least kTopK non-NaN logits (NaN logits are never selected).
__global__ void __launch_bounds__(256) PrefillMoeRouteWaveKernel(const float* __restrict__ logits,
    int* __restrict__ sel, float* __restrict__ wts, int R) {
  const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= R) return;
  const float* lg = logits + (size_t)r * (kExperts + 1);
  constexpr int kPer = kExperts / 32;
  float v[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) v[j] = lg[lane + 32 * j];
#if CRUCIBLE_DEBUG_NONFINITE & 1
  bool bad = false;
  for (int j = 0; j < kPer; ++j) bad |= NanOrPosInf(v[j]);
  if (bad) FlagNonfinite(kNfPrefillRoute);
#endif
  unsigned taken = 0;
  float sl[kTopK];
  int my_sel = 0;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float bv = 0.f;
    int bi = -1;
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      // Ascending j is ascending expert index, so a strict > keeps the lower one.
      if (!(taken >> j & 1u) && v[j] == v[j] && (bi < 0 || v[j] > bv)) { bv = v[j]; bi = lane + 32 * j; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor(bv, o);
      const int oi = __shfl_xor(bi, o);
      if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
    }
    if (bi >= 0 && (bi & 31) == lane) taken |= 1u << (bi >> 5);
    sl[k] = bv;
    if (lane == k) my_sel = bi;
  }
  float sum = 0.f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) sum += __expf(sl[k] - sl[0]);
  float mine = sl[0];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) if (lane == k) mine = sl[k];
  if (lane < kTopK) {
    // A row with fewer than kTopK non-NaN logits leaves the rest unwritten, as
    // MoeRouteKernel does, so no expert index is out of range.
    if (my_sel >= 0) sel[r * kTopK + lane] = my_sel;
    wts[r * kSlots + lane] = __expf(mine - sl[0]) / sum;
  }
  if (lane == 0) wts[r * kSlots + kTopK] = Sigmoid(lg[kExperts]);
}

__global__ void __launch_bounds__(256) PrefillMoeMidKernel(PrefillMoe a) {
  const int w=blockIdx.x*8+(threadIdx.x>>5), lane=threadIdx.x&31;
  if (w>=a.dispatch.rows*kSlots*5) return;
  const int p=w/5, blk=w%5;
  const bool shared=p>=a.dispatch.rows*kTopK;
  const int pair=shared ? p-a.dispatch.rows*kTopK : a.dispatch.pairs[p];
  const int e=shared ? kExperts : a.dispatch.selected[pair];
  const int r=shared ? pair : p-a.dispatch.offsets[e];
  float gv[4],uv[4],v[4];
  LinOutBlock(a.gu[e*2],r,blk,gv); LinOutBlock(a.gu[e*2+1],r,blk,uv);
  const LinJob d=a.down[e]; const int c=blk*128+lane*4;
#pragma unroll
  for (int i=0; i<4; ++i) v[i]=RoundH(RoundH(Silu(gv[i])*uv[i])*__half2float(d.suh[c+i]));
  Wht128(v);
  __half* dst=a.packed ? d.xh+PackedIndex(r,c,d.R) : d.xh+(size_t)r*kMoeInter+c;
#pragma unroll
  for (int i=0; i<4; ++i) dst[i]=__float2half(v[i]*kHadNorm);
}

// LinOutBlock (S = 1) over an fp16 partial: 0 + p, then the same transform.
__device__ __forceinline__ void LinOutBlockHalf(const LinJob& j, int r, int blk, float v[4]) {
  const int c0 = blk * 128 + (threadIdx.x & 31) * 4;
  const __half* p = reinterpret_cast<const __half*>(j.partial) + (size_t)r * j.O + c0;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = 0.f + __half2float(p[i]);
  Wht128(v);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= kHadNorm * __half2float(j.svh[c0 + i]);
}

// HALF: the down projection stored pd as fp16 (approved prefill numerics).
// STORE_Y (ablation.h): store y to H as [R][kHidden]; post and ldh are unused.
template<bool HALF=false, bool STORE_Y=false>
__global__ void __launch_bounds__(256) PrefillMoeCombineKernel(PrefillMoe a,
    const float* weights, const float* post, float* H,int ldh=kWide) {
  const int w=blockIdx.x*8+(threadIdx.x>>5), lane=threadIdx.x&31;
  if (w>=a.dispatch.rows*20) return;
  const int r=w/20, blk=w%20;
  float y[4]={};
  for (int k=0; k<kSlots; ++k) {
    const int pair=r*kTopK+k;
    const int e=k<kTopK ? a.dispatch.selected[pair] : kExperts;
    const int local=k<kTopK ? a.inverse[pair]-a.dispatch.offsets[e] : r;
    float v[4];
    if constexpr (HALF) LinOutBlockHalf(a.down[e],local,blk,v);
    else LinOutBlock(a.down[e],local,blk,v);
    const float weight=weights[r*kSlots+k];
#pragma unroll
    for (int i=0; i<4; ++i) y[i]+=weight*v[i];
  }
  const int c=blk*128+lane*4;
  if constexpr (STORE_Y) {
#pragma unroll
    for (int i=0; i<4; ++i) H[(size_t)r*kHidden+c+i]=y[i];
    return;
  }
#pragma unroll
  for (int s=0; s<kStreams; ++s) {
    const float gate=post[r*kStreams+s];
#pragma unroll
    for (int i=0; i<4; ++i) H[(size_t)r*ldh+s*kHidden+c+i]+=gate*y[i];
  }
}
}  // namespace crucible
