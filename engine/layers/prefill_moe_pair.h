// Lever M1: one 8-wave
// block computes the gate and the up columns of one gate/up item pair (waves
// 0-3 gate, 4-7 up; tile_rows x 128 columns) and runs the mid step in its
// epilogue. So the fp32 pg/pu partials and PrefillMoeMidKernel are not used.
// xd is bitwise equal to PrefillMoeFxKernel + PrefillMoeMidKernel: the same
// body, item origin and k order give the same accumulators, and the epilogue
// runs the mid kernel's operations in its order (LinOutLane, LinInLane).
#pragma once
#include "prefill_moe.h"
#include "prefill_moe_kset.h"

namespace crucible {
constexpr int kPairPad=256+4;                 // stage row: gate 128 | up 128 | pad
constexpr int kPairTileWords=2*(2*16*136/2);  // the double-buffered tiles of two X items
static_assert(kPairTileWords>=16*kPairPad);
static_assert(kPairTileWords>=8*kGemmWaveWords);
static_assert(kPairTileWords>=8*kMoeNb*kGemmWaveWords);  // MoeWindowBody: kMoeNb exchanges per wave

// Per 16-row tile: the 8 waves stage their accumulators, then each wave runs
// the mid step on 2 rows (the pattern of PrefillGemmOutKernel's epilogue).
// The gate and up transforms are LinOutBlock's arithmetic (LinOutLane).
// MUT 2 (bench mutant only) swaps the gate and up slots.
template<int RM,int MUT> __device__ __forceinline__ void PrefillMoePairEpilogue(const PrefillMoe& a, int e,
    int M, int m0, int b0, const GemmF8* acc, float* stage) {
  const int lane=threadIdx.x&31, wave=threadIdx.x>>5, side=wave>>2, c=b0*16+lane*4;
  __half gs[4],us[4],ds[4];
  LinScaleLane(PrefillUniform(a.gu[e*2].svh),c,gs);
  LinScaleLane(PrefillUniform(a.gu[e*2+1].svh),c,us);
  LinScaleLane(PrefillUniform(a.down[e].suh),c,ds);
  __half* xd=PrefillUniform(a.down[e].xh);
#pragma unroll
  for (int r=0; r<RM; ++r) {
    const int row0=m0+r*16;
    if (row0>=M) break;
    __syncthreads();  // B exchanges and earlier readers are done with the stage
#pragma unroll
    for (int nb=0; nb<kMoeNb; ++nb)
#pragma unroll
      for (int z=0; z<8; ++z)
        stage[(2*z+(lane>>4))*kPairPad+side*128+((wave&3)*kMoeNb+nb)*16+(lane&15)]=acc[nb*RM+r][z];
    __syncthreads();
#pragma unroll
    for (int h=0; h<2; ++h) {
      const int lr=wave*2+h, row=row0+lr;
      if (row>=M) break;
      const float* gp=stage+lr*kPairPad+(MUT==2 ? 128 : 0)+lane*4;
      const float* up=stage+lr*kPairPad+(MUT==2 ? 0 : 128)+lane*4;
      float gv[4],uv[4];
      LinOutLane(gp,gs,gv);
      LinOutLane(up,us,uv);
      // PrefillMoeMidKernel's expression: RoundH(RoundH(Silu(g) * u) * suh).
      float s[4],v[4];
#pragma unroll
      for (int i=0; i<4; ++i) s[i]=Silu(gv[i])*uv[i];
      LinInLane<MUT==1>(s,ds,v);
      __half* dst=xd+PackedIndex(row,c,M);
#pragma unroll
      for (int i=0; i<4; ++i) dst[i]=__float2half(v[i]*kHadNorm);
    }
  }
}

// One side of a pair: the body (b = it.b0 + wave * NB, so the up waves 4..7
// take b0 - 4 * NB) and the epilogue, with one body per live row-tile count
// and no skip test in the k loop (the code shape the compiler gives
// PrefillMoeFxKernel; without it the paired k loop lost 21-46 ms per 4,096
// rows in the bench). Each live tile sees the same WMMA chain. The epilogue
// sits inside each branch: after a shared merge the accumulators spilled.
// Gate and up of one expert have the same K and rows, so both sides take the
// same branch and the same barriers.
// TOP is the kernel's RM. TOP 6 uses MoeWindowBody, the body of
// PrefillMoeFxKernel<6> (MoeLiveWindowBody): with PrefillWideBody the
// compiler put part of the decode before the A loads of the 6-tile K4, K5
// and K6 loops and waited for all loads at the loop top (vmcnt(0)); the
// 4.05 engine then lost about 50 ms per 4,096 rows (findings section 11).
template<int K,int RM,int MUT,int TOP=RM> __device__ __forceinline__ void PrefillMoePairItem(const PrefillMoe& a,
    const Exl3GemmJob& j, Exl3GemmItem it, int side, uint32_t* lds) {
  if constexpr (RM>1) {
    if ((j.M-it.m0+15)/16<RM) { PrefillMoePairItem<K,RM-1,MUT,TOP>(a,j,it,side,lds); return; }
  }
  const int b0=it.b0;
  it.b0-=side*4*kMoeNb;
  GemmF8 acc[kMoeNb*RM];
#if CRUCIBLE_MOE_WINDOW_PROBE
  if constexpr (TOP==6)
    MoeWindowBody<K,RM,(RM<3 ? RM : 3),false,false,true>(j,it,lds,acc);
  else
#endif
    PrefillWideBody<K,RM,kMoeNb,true,false,true>(j,it,lds,acc);
  PrefillMoePairEpilogue<RM,MUT>(a,it.job>>1,j.M,it.m0,b0,acc,reinterpret_cast<float*>(lds));
}

// The PrefillMoeFxKernel queue with paired claims: G claim g owns gate item
// i_gu[2g] and up item i_gu[2g + 1] (unit g / 5); an X claim takes the gate
// and the up X item of one 16-row group (waves 0-3 and 4-7). RM is the plan's
// tile_rows / 16. MUT: bench mutants only (0 in the engine).
// waves_per_eu(6): at most 240 VGPRs. VGPRs are given in steps of 24 here, so
// 242 VGPRs (ae79fd92, 4.05 RM6) gave 5 waves per SIMD, that is 2 blocks of 8
// waves per WGP instead of 3, and the 4.05 bench lost 44-58 ms per 4,096 rows.
template<int RM,int KMAX=5,int MUT=0> __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6)))
PrefillMoePairKernel(PrefillMoe a, int lead) {
  constexpr int per=10/kMoeNb;  // pairs per unit
  __shared__ __attribute__((aligned(16))) uint32_t lds[kPairTileWords];
  __shared__ int claim;
  const int t=threadIdx.x, side=__builtin_amdgcn_readfirstlane(t>>7), T=a.tile_rows;
  auto* tile=reinterpret_cast<__half (*)[16][136]>(lds)+2*side;
  const int ng=PrefillUniform(a.item_counts[0])/2, nx=PrefillUniform(a.fx_count[2]);
  for (;;) {
    if (t==0) claim=atomicAdd(a.fx_count+1,1);
    __syncthreads();
    const int g=PrefillUniform(claim);
    __syncthreads();
    Exl3GemmItem it{}; Exl3GemmJob j{};
    int need=nx;
    if (g<ng) {
      it=PrefillUniform(a.i_gu[2*g+side]); j=PrefillUniform(a.g_gu[it.job]);
      need=min(nx,a.fx_xoff[it.job>>1]+2*min((j.M+15)/16,(it.m0+T)/16));
    }
    const int limit=min(nx,need+lead);
    for (;;) {
      if (t==0) claim=__hip_atomic_load(a.fx_count,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)<limit
                      ? atomicAdd(a.fx_count,2) : nx;
      __syncthreads();
      const int x=PrefillUniform(claim);
      __syncthreads();
      if (x>=nx) break;
      PrefillFxTransform(a,x+side,tile,t&127);
    }
    if (g>=ng) return;
    const int target=2*((min(j.M,it.m0+T)-it.m0+15)/16);
    if (t==0)
      while (__hip_atomic_load(a.fx_ready+g/per,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)<target)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
    if (j.K==3) PrefillMoePairItem<3,RM,MUT>(a,j,it,side,lds);
    else if (j.K==4) PrefillMoePairItem<4,RM,MUT>(a,j,it,side,lds);
    else if (j.K==5) PrefillMoePairItem<5,RM,MUT>(a,j,it,side,lds);
    else if constexpr (KMAX==6) { if (j.K==6) PrefillMoePairItem<6,RM,MUT>(a,j,it,side,lds); }
    __syncthreads();
  }
}
// Shapes of the paired path: packed RM 2, 4 and 6 (the engine's MoE shapes).
inline bool PrefillMoePairShape(PrefillGemmShape g) { return g.packed && (g.rm==6 || g.rm==4 || g.rm==2); }
// 1,024 x 256 threads: the FX grid's thread count.
template<int MUT=0> inline void LaunchPrefillMoePair(int rm, const PrefillMoe& a, bool k6) {
  if (rm!=6 && rm!=4 && rm!=2) { std::fprintf(stderr,"M1 pair path supports RM 2, 4 and 6\n"); std::abort(); }
  if (k6) {
    if (rm==6) PrefillMoePairKernel<6,6,MUT><<<1024,256>>>(a,kFxLead);
    else if (rm==4) PrefillMoePairKernel<4,6,MUT><<<1024,256>>>(a,kFxLead);
    else PrefillMoePairKernel<2,6,MUT><<<1024,256>>>(a,kFxLead);
  } else {
    if (rm==6) PrefillMoePairKernel<6,5,MUT><<<1024,256>>>(a,kFxLead);
    else if (rm==4) PrefillMoePairKernel<4,5,MUT><<<1024,256>>>(a,kFxLead);
    else PrefillMoePairKernel<2,5,MUT><<<1024,256>>>(a,kFxLead);
  }
}
}  // namespace crucible
