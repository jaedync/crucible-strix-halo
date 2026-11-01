// Packed operands remove repeated fp16 conversion and scalar half loads.
#pragma once
#include "ops.h"
#include "../gfx1151/prefill_wmma.h"

namespace crucible {
__device__ __forceinline__ prefill::H16 PrefillLoad16(const __half* p) {
  return *reinterpret_cast<const prefill::H16*>(p);
}
__global__ void __launch_bounds__(256) PrefillPackKernel(const float* x,__half* hi,__half* lo,int N) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i>=N) return;
  const float v=x[i]; const __half h=__float2half(v);
  hi[i]=h;
  if (lo) lo[i]=__float2half((v-__half2float(h))*prefill::kResidualScale);
}

// Fragment-order HC weights for bulk prefill: 16x16 tiles of 512 bytes,
// lane row (lane & 15) at 16 halves, k tiles contiguous inside a column tile.
// up [4*kHidden][kRank] -> [row/16][kRank/16][16][16].
__global__ void __launch_bounds__(256) HcFragUpKernel(const __half* up,__half* uf) {
  const size_t i=(size_t)blockIdx.x*256+threadIdx.x;
  if (i>=(size_t)kWide*kRank) return;
  const int row=i/kRank,k=i%kRank;
  uf[((size_t)(row>>4)*(kRank/16)+k/16)*256+(row&15)*16+(k&15)]=up[i];
}
// fn [M][kWide] -> [M/16 rounded up][stream][kHidden/16][16][16], rows past M zero.
__global__ void __launch_bounds__(256) HcFragFnKernel(const __half* fn,int M,__half* ff) {
  const size_t i=(size_t)blockIdx.x*256+threadIdx.x;
  const int tiles=(M+15)/16;
  if (i>=(size_t)tiles*16*kWide) return;
  const int row=i/kWide,c=i%kWide,s=c/kHidden,k=c%kHidden;
  ff[(((size_t)(row>>4)*kStreams+s)*(kHidden/16)+k/16)*256+(row&15)*16+(k&15)]=
      row<M ? fn[(size_t)row*kWide+c] : __half(0.f);
}
// One block owns 32 rows of one stream and every output column, so it reads H
// once, splits it into high/low halves in LDS and sums the squares on the way.
constexpr int kHcRows=32, kHcStep=64;
// STEP is the k span staged per barrier pair. Each thread keeps STEP/32
// float4 of H in flight, so a larger STEP keeps more DRAM reads outstanding;
// squares keep the same per-thread order for any STEP.
template<int STEP=kHcStep>
__global__ void __launch_bounds__(256) PrefillHcDotsFusedKernel(const __half* fn,int M,
    const float* H,float* dots,float* ss,int R,int ldh=kWide) {
  constexpr int kLoads=STEP/32, kPad=STEP+8;
  // The 8-half pad spreads the 16 A rows over distinct LDS banks.
  __shared__ __attribute__((aligned(16))) _Float16 ahi[kHcRows][kPad], alo[kHcRows][kPad];
  const int t=threadIdx.x,lane=t&31,wave=t>>5,r0=blockIdx.x*kHcRows,s=blockIdx.y;
  const int sr=t>>3,sk=(t&7)*4;
  const bool live=r0+sr<R;
  const float* src=H+(size_t)(r0+sr)*ldh+s*kHidden+sk;
  float sq=0;
  prefill::F8 hi[2][3]={},lo[2][3]={};
  float4 next[kLoads];
#pragma unroll
  for (int i=0;i<kLoads;++i) next[i]=live ? *reinterpret_cast<const float4*>(src+i*32) : float4{};
#pragma unroll 1
  for (int k0=0;k0<kHidden;k0+=STEP) {
    __syncthreads();
#pragma unroll
    for (int i=0;i<kLoads;++i) {
      const float v[4]={next[i].x,next[i].y,next[i].z,next[i].w};
#pragma unroll
      for (int e=0;e<4;++e) {
        sq+=v[e]*v[e];
        const _Float16 h=(_Float16)v[e];
        ahi[sr][sk+i*32+e]=h;
        alo[sr][sk+i*32+e]=(_Float16)((v[e]-(float)h)*prefill::kResidualScale);
      }
    }
    __syncthreads();
    if (k0+STEP<kHidden) {
#pragma unroll
      for (int i=0;i<kLoads;++i) next[i]=live ? *reinterpret_cast<const float4*>(src+k0+STEP+i*32) : float4{};
    }
#pragma unroll
    for (int kb=0;kb<STEP;kb+=16) {
      prefill::H16 ah[2],al[2];
#pragma unroll
      for (int i=0;i<2;++i) {
        ah[i]=*reinterpret_cast<const prefill::H16*>(&ahi[i*16+(lane&15)][kb]);
        al[i]=*reinterpret_cast<const prefill::H16*>(&alo[i*16+(lane&15)][kb]);
      }
#pragma unroll
      for (int j=0;j<3;++j) {
        const int n=(wave+8*j)*16+(lane&15);
        if ((wave+8*j)*16>=M) continue;
        const prefill::H16 b=n<M ? PrefillLoad16(fn+(size_t)n*kWide+s*kHidden+k0+kb) : prefill::H16{};
#pragma unroll
        for (int i=0;i<2;++i) {
          hi[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i],b,hi[i][j]);
          lo[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al[i],b,lo[i][j]);
        }
      }
    }
  }
  sq+=__shfl_xor(sq,1); sq+=__shfl_xor(sq,2); sq+=__shfl_xor(sq,4);
  if ((t&7)==0 && live) ss[(r0+sr)*kStreams+s]=sq;
#pragma unroll
  for (int j=0;j<3;++j) {
    const int n=(wave+8*j)*16+(lane&15);
    if ((wave+8*j)*16>=M || n>=M) continue;
#pragma unroll
    for (int i=0;i<2;++i) {
      const prefill::F8 out=hi[i][j]+lo[i][j]*(1.f/prefill::kResidualScale);
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+i*16+prefill::FragmentRow(z);
        if (r<R) dots[((size_t)r*M+n)*4+s]=out[z];
      }
    }
  }
}
// HC dots with wave roles. Seven compute waves own three column tiles each
// (21 tiles = 336 >= M columns), read A from LDS and fn straight from global,
// one k block ahead. Loader waves stream H D steps ahead into double
// buffered LDS, so only they wait on DRAM: in the fused kernel the H prefetch
// shares vmcnt with the B loads and every step exposes one DRAM latency.
// Same tiles, k order and per-thread square order (a loader thread holds the
// two fused-kernel square sums of k groups 2p and 2p+1), so dots and ss are
// bitwise equal to PrefillHcDotsFusedKernel.
constexpr int kHcWsCompute=7, kHcWsTiles=3*kHcWsCompute;
__device__ __forceinline__ void PrefillLdsSync() {
  // LDS only: global loads stay in flight across the barrier.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// LW loader waves; each loader thread holds G = 8/(LW*32/32) k groups of 4
// and keeps D steps of loads in flight (a register ring).
// DIAG (bench only): 1 skips the WMMAs, 2 reads H rows 0..31 in every block,
// 4 reads fn rows 0..15 in every wave.
// LO keeps the low half of H (an exact fp32 product). Without it the dots use
// the fp16 high half only (approved numerics):
// half the WMMAs and LDS, and the squares still use fp32 H.
// RT 16-row tiles per block: the loader waves cover 32 rows per pass and run
// RT/2 passes. More rows per block divide the fn reads through L2, which set
// the time once the low half is gone. Every output sees the same WMMA
// sequence and every square the same sum tree, so RT does not change results.
// FB: fn is the fragment-order copy (HcFragFnKernel). A B load is then one
// 512-byte run instead of 16 row-strided 32-byte pieces; the values and the
// WMMA chains do not change.
template<int STEP,int DIAG=0,bool SWAP=false,int LW=4,int D=1,bool LO=true,int RT=2,bool FB=false>
__global__ void __launch_bounds__((kHcWsCompute+LW)*32) PrefillHcDotsWsKernel(const __half* fn,int M,
    const float* H,float* dots,float* ss,int R,int ldh=kWide) {
  typedef _Float16 H4 __attribute__((ext_vector_type(4)));
  constexpr int kPad=STEP+8, kI=STEP/32, kSteps=kHidden/STEP, G=8/LW, TPR=8/G, ROWS=16*RT, NP=RT/2;
  // LW=8 gave non-bitwise dots in the bench (not explained), so it is not allowed.
  static_assert(kSteps%D==0 && (LW==2 || LW==4) && RT%2==0 && LW*32/TPR==32);
  __shared__ __attribute__((aligned(16))) _Float16 ahi[2][ROWS][kPad], alo[LO ? 2 : 1][LO ? ROWS : 1][kPad];
  const int t=threadIdx.x,lane=t&31,wave=t>>5;
  const int r0=(SWAP ? blockIdx.x>>2 : blockIdx.x)*ROWS,s=SWAP ? blockIdx.x&3 : blockIdx.y;
  if (wave>=kHcWsCompute) {
    const int u=t-kHcWsCompute*32, sr=u/TPR, p=u%TPR;
    bool live[NP]; const float* src[NP];
#pragma unroll
    for (int pp=0;pp<NP;++pp) {
      live[pp]=r0+sr+32*pp<R;
      src[pp]=H+(size_t)((DIAG&2 ? 0 : r0)+sr+32*pp)*ldh+s*kHidden+4*G*p;
    }
    float sq[NP][G]={};
    float4 v[D][NP][kI][G];
    auto load=[&](float4 (&w)[NP][kI][G],int step) {
#pragma unroll
      for (int pp=0;pp<NP;++pp)
#pragma unroll
        for (int i=0;i<kI;++i)
#pragma unroll
          for (int h=0;h<G;++h)
            w[pp][i][h]=live[pp] ? *reinterpret_cast<const float4*>(src[pp]+step*STEP+i*32+4*h) : float4{};
    };
    auto store=[&](const float4 (&w)[NP][kI][G],int buf) {
#pragma unroll
      for (int pp=0;pp<NP;++pp)
#pragma unroll
      for (int i=0;i<kI;++i)
#pragma unroll
        for (int h=0;h<G;++h) {
          const float x[4]={w[pp][i][h].x,w[pp][i][h].y,w[pp][i][h].z,w[pp][i][h].w};
          H4 hi4,lo4;
#pragma unroll
          for (int e=0;e<4;++e) {
            sq[pp][h]+=x[e]*x[e];
            const _Float16 hh=(_Float16)x[e];
            hi4[e]=hh;
            if constexpr (LO) lo4[e]=(_Float16)((x[e]-(float)hh)*prefill::kResidualScale);
          }
          *reinterpret_cast<H4*>(&ahi[buf][sr+32*pp][4*G*p+32*i+4*h])=hi4;
          if constexpr (LO) *reinterpret_cast<H4*>(&alo[buf][sr+32*pp][4*G*p+32*i+4*h])=lo4;
        }
    };
#pragma unroll
    for (int d=0;d<D;++d) load(v[d],d);
    store(v[0],0);
    if (D<kSteps) load(v[0],D);
    PrefillLdsSync();
#pragma unroll 1
    for (int s0=0;s0<kSteps;s0+=D)
#pragma unroll
      for (int d=0;d<D;++d) {
        const int step=s0+d;
        if (step+1<kSteps) {
          store(v[(d+1)%D],(step+1)&1);
          if (step+1+D<kSteps) load(v[(d+1)%D],step+1+D);
        }
        PrefillLdsSync();
      }
    // The fused kernel's tree: groups (0+1), then +(2+3), then +(4+5 + 6+7).
#pragma unroll
    for (int pp=0;pp<NP;++pp) {
      float q[G];
#pragma unroll
      for (int h=0;h<G;++h) q[h]=sq[pp][h];
#pragma unroll
      for (int w=1;w<G;w<<=1)
#pragma unroll
        for (int h=0;h<G;h+=2*w) q[h]=q[h]+q[h+w];
      float a=q[0];
#pragma unroll
      for (int w=1;w<TPR;w<<=1) a+=__shfl_xor(a,w);
      if (p==0 && live[pp]) ss[(r0+sr+32*pp)*kStreams+s]=a;
    }
    return;
  }
  const __half* bp[3];
#pragma unroll
  for (int j=0;j<3;++j) {
    // Columns past M load row M-1 and are never stored.
    const int n=DIAG&4 ? (lane&15) : min((wave+kHcWsCompute*j)*16+(lane&15),M-1);
    // Fragment order pads M to whole zero tiles, so no clamp is needed.
    if constexpr (FB) bp[j]=fn+(((size_t)(DIAG&4 ? 0 : wave+kHcWsCompute*j)*kStreams+s)*(kHidden/16))*256+(lane&15)*16;
    else bp[j]=fn+(size_t)n*kWide+s*kHidden;
  }
  // Offset of k (a multiple of 16) inside one column tile's k run.
  auto ko=[](int k) { return FB ? (k>>4)*256 : k; };
  prefill::F8 hi[RT][3]={},lo[LO ? RT : 1][3]={};
  prefill::H16 b[3];
#pragma unroll
  for (int j=0;j<3;++j) b[j]=PrefillLoad16(bp[j]);
  PrefillLdsSync();
#pragma unroll 1
  for (int step=0;step<kSteps;++step) {
    const int buf=step&1;
#pragma unroll
    for (int kb=0;kb<STEP;kb+=16) {
      const int k=step*STEP+kb, kn=k+16<kHidden ? k+16 : k;
      prefill::H16 bn[3];
#pragma unroll
      for (int j=0;j<3;++j) bn[j]=PrefillLoad16(bp[j]+ko(kn));
      prefill::H16 ah[RT],al[LO ? RT : 1];
#pragma unroll
      for (int i=0;i<RT;++i) {
        ah[i]=*reinterpret_cast<const prefill::H16*>(&ahi[buf][i*16+(lane&15)][kb]);
        if constexpr (LO) al[i]=*reinterpret_cast<const prefill::H16*>(&alo[buf][i*16+(lane&15)][kb]);
      }
#pragma unroll
      for (int j=0;j<3;++j)
#pragma unroll
        for (int i=0;i<RT;++i) {
          if constexpr (DIAG&1) {
            hi[i][j][0]+=(float)(ah[i][0]*b[j][1]);
            if constexpr (LO) lo[i][j][0]+=(float)(al[i][0]*b[j][2]);
          } else {
            hi[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i],b[j],hi[i][j]);
            if constexpr (LO) lo[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al[i],b[j],lo[i][j]);
          }
        }
#pragma unroll
      for (int j=0;j<3;++j) b[j]=bn[j];
    }
    PrefillLdsSync();
  }
#pragma unroll
  for (int j=0;j<3;++j) {
    const int n=(wave+kHcWsCompute*j)*16+(lane&15);
    if (n>=M) continue;
#pragma unroll
    for (int i=0;i<RT;++i) {
      prefill::F8 out=hi[i][j];
      if constexpr (LO) out=hi[i][j]+lo[i][j]*(1.f/prefill::kResidualScale);
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+i*16+prefill::FragmentRow(z);
        if (r<R) dots[((size_t)r*M+n)*4+s]=out[z];
      }
    }
  }
}
// HC dots as a tiled GEMM. A wave owns 16 rows and 7 column tiles (high and
// low accumulators), a block owns WAVES x 16 rows by 112 columns of one
// stream. Each wave reads its H rows straight into WMMA fragments and splits
// them in registers, one k block ahead. The block stages the 112 fn rows of
// each 32-k step in LDS one step ahead, so the WMMAs never wait on a B load
// (the fused kernel waits on every B load and reads fn once per 32 rows).
// Dots are bitwise equal to the fused kernel; squares sum in k order.
constexpr int kHcTileCols=7, kHcTileK=32, kHcTileKPad=kHcTileK+8;
template<int WAVES>
__global__ void __launch_bounds__(WAVES*32) PrefillHcDotsTileKernel(const __half* fn,int M,
    const float* H,float* dots,float* ss,int R) {
  constexpr int kCols=kHcTileCols*16, kPieces=kCols*kHcTileK/8, kThreads=WAVES*32;
  constexpr int kLoads=(kPieces+kThreads-1)/kThreads;
  __shared__ __attribute__((aligned(16))) _Float16 bt[2][kCols][kHcTileKPad];
  const int t=threadIdx.x,lane=t&31,wave=t>>5,s=blockIdx.z;
  const int n0=blockIdx.x*kCols,r0=blockIdx.y*WAVES*16+wave*16,row=r0+(lane&15);
  const float* a=H+(size_t)min(row,R-1)*kWide+s*kHidden;
  const __half* b=fn+s*kHidden;
  uint4 bnext[kLoads];
  auto load_b=[&](int k0) {
#pragma unroll
    for (int i=0;i<kLoads;++i) {
      const int piece=t+i*kThreads,n=n0+piece/4;
      bnext[i]=piece<kPieces && n<M
          ? *reinterpret_cast<const uint4*>(b+(size_t)n*kWide+k0+piece%4*8) : uint4{};
    }
  };
  auto store_b=[&](int buf) {
#pragma unroll
    for (int i=0;i<kLoads;++i) {
      const int piece=t+i*kThreads;
      if (piece<kPieces) *reinterpret_cast<uint4*>(&bt[buf][piece/4][piece%4*8])=bnext[i];
    }
  };
  float4 an[4];
  auto load_a=[&](int k) {
#pragma unroll
    for (int i=0;i<4;++i) an[i]=*reinterpret_cast<const float4*>(a+k+i*4);
  };
  prefill::F8 hi[kHcTileCols]={},lo[kHcTileCols]={};
  float sq=0;
  load_b(0); store_b(0); load_a(0);
  __syncthreads();
#pragma unroll 1
  for (int k0=0,buf=0;k0<kHidden;k0+=kHcTileK,buf^=1) {
    const bool more=k0+kHcTileK<kHidden;
    if (more) load_b(k0+kHcTileK);
#pragma unroll 1
    for (int kb=0;kb<kHcTileK;kb+=16) {
      prefill::H16 ah,al;
#pragma unroll
      for (int e=0;e<16;++e) {
        const float v=an[e/4][e%4];
        sq+=v*v;
        const _Float16 h=(_Float16)v;
        ah[e]=h; al[e]=(_Float16)((v-(float)h)*prefill::kResidualScale);
      }
      if (k0+kb+16<kHidden) load_a(k0+kb+16);
#pragma unroll
      for (int c=0;c<kHcTileCols;++c) {
        const prefill::H16 bf=*reinterpret_cast<const prefill::H16*>(&bt[buf][c*16+(lane&15)][kb]);
        hi[c]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah,bf,hi[c]);
        lo[c]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al,bf,lo[c]);
      }
    }
    if (more) store_b(buf^1);
    __syncthreads();
  }
  if (blockIdx.x==0 && lane<16 && row<R) ss[row*kStreams+s]=sq;
#pragma unroll
  for (int c=0;c<kHcTileCols;++c) {
    const int n=n0+c*16+(lane&15);
    if (n>=M) continue;
    const prefill::F8 out=hi[c]+lo[c]*(1.f/prefill::kResidualScale);
#pragma unroll
    for (int z=0;z<8;++z) {
      const int r=r0+prefill::FragmentRow(z);
      if (r<R) dots[((size_t)r*M+n)*4+s]=out[z];
    }
  }
}
template<int WAVES> inline void LaunchHcDotsTile(const __half* fn,int M,const float* H,
    float* dots,float* ss,int R,hipStream_t stream=nullptr) {
  const dim3 grid((M+kHcTileCols*16-1)/(kHcTileCols*16),(R+WAVES*16-1)/(WAVES*16),kStreams);
  PrefillHcDotsTileKernel<WAVES><<<grid,WAVES*32,0,stream>>>(fn,M,H,dots,ss,R);
}
// Same work as PrefillHcDotsFusedKernel, but fn also goes through LDS. All
// global loads for step k+1 are in flight while step k runs its WMMAs, and the
// hot loop has no divergent branches (rows past M load row M-1; never stored).
// Writes t directly as high/low halves for the gate GEMM.
__global__ void __launch_bounds__(256) PrefillHcMidPackKernel(const float* dots,int M,
    const float* ss,__half* thi,__half* tlo,float* post,int R) {
  const int r=blockIdx.x,j=threadIdx.x;
  if (r>=R) return;
  float inv[4];
#pragma unroll
  for (int s=0;s<4;++s) inv[s]=rsqrtf(ss[r*4+s]/kHidden+kEps);
  for (int m=j;m<kRank;m+=256) {
    const float* d=dots+((size_t)r*M+m)*4;float a=0;
#pragma unroll
    for (int s=0;s<4;++s) a+=inv[s]*d[s];
    const float v=Silu(a*.25f); const __half h=__float2half(v);
    thi[r*kRank+m]=h; tlo[r*kRank+m]=__float2half((v-__half2float(h))*prefill::kResidualScale);
  }
  if (j<4 && post) {
    const float* d=dots+((size_t)r*M+kRank+j)*4;float a=0;
#pragma unroll
    for (int s=0;s<4;++s) a+=inv[s]*d[s];
    post[r*4+j]=2.f*Sigmoid(a*.25f);
  }
}
// A wave owns one 16-row tile, 16 columns and all four streams, so the stream
// sum stays in registers. Eight waves cover 64 rows by 32 columns.
// xh (optional) receives the fp16 copy of mixed, which is fp16-exact.
// TALL: the 8 waves of a block take 128 rows of the same 16 columns, so each
// up fragment from L2 serves 128 rows instead of 64.
// FB: up is the fragment-order copy (HcFragUpKernel), same values.
template<bool TALL=false,bool FB=false>
__global__ void __launch_bounds__(256) PrefillHcMixWideKernel(const __half* up,const float* w1,
    const __half* thi,const __half* tlo,const float* ss,const float* H,float* mixed,int R,
    __half* xh=nullptr,int ldh=kWide) {
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
  const int r0=TALL ? blockIdx.y*128+wave*16 : blockIdx.y*64+(wave>>1)*16;
  const int d=TALL ? blockIdx.x*16+(lane&15) : blockIdx.x*32+(wave&1)*16+(lane&15);
  if (r0>=R) return;
  const int ra=r0+(lane&15);
  prefill::F8 hi[4]={},lo[4]={};
#pragma unroll 2
  for (int k=0;k<kRank;k+=16) {
    const prefill::H16 ah=ra<R ? PrefillLoad16(thi+(size_t)ra*kRank+k) : prefill::H16{};
    const prefill::H16 al=ra<R ? PrefillLoad16(tlo+(size_t)ra*kRank+k) : prefill::H16{};
#pragma unroll
    for (int s=0;s<4;++s) {
      const prefill::H16 b=FB ? PrefillLoad16(up+(((size_t)s*(kHidden/16)+(d>>4))*(kRank/16)+(k>>4))*256+(lane&15)*16)
                              : PrefillLoad16(up+((size_t)s*kHidden+d)*kRank+k);
      hi[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah,b,hi[s]);
      lo[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al,b,lo[s]);
    }
  }
  float w[4];
#pragma unroll
  for (int s=0;s<4;++s) w[s]=w1[s*kHidden+d];
#pragma unroll
  for (int z=0;z<8;++z) {
    const int r=r0+prefill::FragmentRow(z);
    if (r>=R) continue;
    float v[4];
#pragma unroll
    for (int s=0;s<4;++s) {
      const float g=hi[s][z]+lo[s][z]*(1.f/prefill::kResidualScale);
      const float inv=rsqrtf(ss[r*4+s]/kHidden+kEps);
      v[s]=Sigmoid(g)*H[(size_t)r*ldh+s*kHidden+d]*inv*w[s];
    }
    const float x=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
    if (mixed) mixed[(size_t)r*kHidden+d]=x;
    if (xh) xh[(size_t)r*kHidden+d]=__float2half(x);
  }
}
// Register-blocked fp16 GEMM y[R][O] = x[R][2560] W[O][2560]^T for the
// router, PLE and GDN a/b weights. A wave owns 32 rows by 32 columns, so each
// k block needs four fragment loads for four WMMAs; neighbours share them in L0.
__global__ void __launch_bounds__(256) PrefillDenseTileKernel(const __half* W,int O,
    const __half* x,float* y,int ldy,int R) {
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
  const int r0=blockIdx.x*64+(wave>>2)*32,n0=blockIdx.y*128+(wave&3)*32;
  if (r0>=R || n0>=O) return;
  const __half* a[2]; const __half* b[2];
#pragma unroll
  for (int i=0;i<2;++i) {
    a[i]=x+(size_t)min(r0+i*16+(lane&15),R-1)*kHidden;
    b[i]=W+(size_t)min(n0+i*16+(lane&15),O-1)*kHidden;
  }
  prefill::F8 acc[2][2]={};
  // Each lane reads a whole 128-byte line (four k blocks) per operand, so no
  // line is fetched again for the next k block.
#pragma unroll 1
  for (int k=0;k<kHidden;k+=64) {
    prefill::H16 af[2][4],bf[2][4];
#pragma unroll
    for (int i=0;i<2;++i)
#pragma unroll
      for (int q=0;q<4;++q) { af[i][q]=PrefillLoad16(a[i]+k+q*16); bf[i][q]=PrefillLoad16(b[i]+k+q*16); }
#pragma unroll
    for (int q=0;q<4;++q)
#pragma unroll
      for (int i=0;i<2;++i)
#pragma unroll
        for (int j=0;j<2;++j) acc[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i][q],bf[j][q],acc[i][j]);
  }
#pragma unroll
  for (int i=0;i<2;++i)
#pragma unroll
    for (int j=0;j<2;++j) {
      const int n=n0+j*16+(lane&15);
      if (n>=O) continue;
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+i*16+prefill::FragmentRow(z);
        if (r<R) y[(size_t)r*ldy+n]=acc[i][j][z];
      }
    }
}
// Dense GEMM stages: 32 k per LDS stage, rows padded by 8 halfs.
constexpr int kBigStep=32, kBigPad=kBigStep+8;

// PrefillDenseFastKernel<DEPTH> (now in bench/kernels/prefill_hc.h) with three changes:
// - a 1-D grid in supertiles: G consecutive row blocks take column block 0,
//   then column block 1, and so on, so a row block of x is read from DRAM
//   once and the next column blocks of it hit L2 (the 2-D grid walks all row
//   blocks per column block: x is read once per column block);
// - waves whose 64 columns are all past O skip their WMMAs, and waves with
//   fewer than 4 valid column tiles skip the rest (O = 513 has one valid
//   column in its last column block);
// - RB = 64: 4 waves own 64 rows by 128 columns (more blocks for small O).
// Same per-output k order: bitwise equal (bench/dense_next.hip: R8192 router
// 1,930 -> 826 us, a/b 1,150 -> 561 us, PLE kv 28.1 -> 17.1 ms, cold caches).
template<int DEPTH=4,int RB=64>
__global__ void __launch_bounds__(RB*2) PrefillDenseSwKernel(const __half* W,int O,
    const __half* x,float* y,int ldy,int R,int G) {
  constexpr int kSteps=kHidden/kBigStep, NT=RB*2, PA=RB*4/NT, PB=512/NT;
  static_assert(kSteps%DEPTH==0 && PA==2);
  __shared__ __attribute__((aligned(16))) _Float16 as[2][RB][kBigPad], bs[2][128][kBigPad];
  const int t=threadIdx.x,lane=t&31,wave=t>>5;
  const int rbs=(R+RB-1)/RB, cbs=(O+127)/128, L=blockIdx.x;
  const int g=L/(G*cbs), gs=min(G,rbs-g*G), i=L-g*G*cbs;
  const int r0=(g*G+i%gs)*RB, n0=i/gs*128;
  const __half* ap[PA]; const __half* bp[PB];
#pragma unroll
  for (int q=0;q<PA;++q) {
    const int idx=t+q*NT,row=idx>>2,part=(idx&3)*8;
    ap[q]=x+(size_t)min(r0+row,R-1)*kHidden+part;
  }
#pragma unroll
  for (int q=0;q<PB;++q) {
    const int idx=t+q*NT,row=idx>>2,part=(idx&3)*8;
    bp[q]=W+(size_t)min(n0+row,O-1)*kHidden+part;
  }
  typedef unsigned U4 __attribute__((ext_vector_type(4)));
  U4 ra[DEPTH][PA],rb[DEPTH][PB];
#pragma unroll
  for (int q=0;q<PA;++q) {
    const int idx=t+q*NT,row=idx>>2,part=(idx&3)*8;
    *reinterpret_cast<U4*>(&as[0][row][part])=*reinterpret_cast<const U4*>(ap[q]);
  }
#pragma unroll
  for (int q=0;q<PB;++q) {
    const int idx=t+q*NT,row=idx>>2,part=(idx&3)*8;
    *reinterpret_cast<U4*>(&bs[0][row][part])=*reinterpret_cast<const U4*>(bp[q]);
  }
#pragma unroll
  for (int d=0;d<DEPTH;++d) {
#pragma unroll
    for (int q=0;q<PA;++q) ra[d][q]=*reinterpret_cast<const U4*>(ap[q]+(d+1)*kBigStep);
#pragma unroll
    for (int q=0;q<PB;++q) rb[d][q]=*reinterpret_cast<const U4*>(bp[q]+(d+1)*kBigStep);
  }
  PrefillLdsSync();
  prefill::F8 acc[2][4]={};
  const int wr=(wave>>1)*32,wc=(wave&1)*64;
  // Valid column tiles of this wave (wave-uniform).
  const int nj=max(0,min(4,(O-n0-wc+15)/16));
#pragma unroll 1
  for (int s0=0;s0<kSteps;s0+=DEPTH)
#pragma unroll
    for (int d=0;d<DEPTH;++d) {
      const int step=s0+d,buf=step&1;
      if (step+1<kSteps) {
#pragma unroll
        for (int q=0;q<PA;++q) {
          const int idx=t+q*NT,row=idx>>2,part=(idx&3)*8;
          *reinterpret_cast<U4*>(&as[buf^1][row][part])=ra[d][q];
        }
#pragma unroll
        for (int q=0;q<PB;++q) {
          const int idx=t+q*NT,row=idx>>2,part=(idx&3)*8;
          *reinterpret_cast<U4*>(&bs[buf^1][row][part])=rb[d][q];
        }
        if (step+1+DEPTH<kSteps) {
#pragma unroll
          for (int q=0;q<PA;++q) ra[d][q]=*reinterpret_cast<const U4*>(ap[q]+(step+1+DEPTH)*kBigStep);
#pragma unroll
          for (int q=0;q<PB;++q) rb[d][q]=*reinterpret_cast<const U4*>(bp[q]+(step+1+DEPTH)*kBigStep);
        }
      }
      if (nj==4) {
#pragma unroll
        for (int kq=0;kq<kBigStep;kq+=16) {
          const int kb=kq;
          prefill::H16 af[2],bf[4];
#pragma unroll
          for (int a=0;a<2;++a) af[a]=*reinterpret_cast<const prefill::H16*>(&as[buf][wr+a*16+(lane&15)][kb]);
#pragma unroll
          for (int j=0;j<4;++j) bf[j]=*reinterpret_cast<const prefill::H16*>(&bs[buf][wc+j*16+(lane&15)][kb]);
#pragma unroll
          for (int a=0;a<2;++a)
#pragma unroll
            for (int j=0;j<4;++j) acc[a][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[a],bf[j],acc[a][j]);
        }
      } else if (nj>0) {
#pragma unroll
        for (int kq=0;kq<kBigStep;kq+=16) {
          const int kb=kq;
          prefill::H16 af[2];
#pragma unroll
          for (int a=0;a<2;++a) af[a]=*reinterpret_cast<const prefill::H16*>(&as[buf][wr+a*16+(lane&15)][kb]);
#pragma unroll
          for (int j=0;j<4;++j) {
            if (j>=nj) break;
            const prefill::H16 bf=*reinterpret_cast<const prefill::H16*>(&bs[buf][wc+j*16+(lane&15)][kb]);
#pragma unroll
            for (int a=0;a<2;++a) acc[a][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[a],bf,acc[a][j]);
          }
        }
      }
      PrefillLdsSync();
    }
#pragma unroll
  for (int a=0;a<2;++a)
#pragma unroll
    for (int j=0;j<4;++j) {
      const int n=n0+wc+j*16+(lane&15);
      if (n>=O) continue;
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+wr+a*16+prefill::FragmentRow(z);
        if (r<R) y[(size_t)r*ldy+n]=acc[a][j][z];
      }
    }
}

}  // namespace crucible
