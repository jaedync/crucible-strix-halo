// Candidate prefill HC kernels (prefill-hc). Every candidate keeps each
// output's WMMA chain and epilogue of the engine kernel it replaces, so the
// bench compares bytes.
#pragma once
#include "../../engine/layers/prefill_hc.h"

namespace crucible {
// PrefillHcMixWideKernel<true,true> with DT column tiles per wave (a block is
// 128 rows by 16*DT columns) and, with PRE, the epilogue H values loaded
// before the k loop, so their DRAM latency overlaps the WMMAs.
// MUT (bench mutant): 1 swaps the first two k blocks of the high chain.
// SW > 0: a 1-D grid in supertiles of SW row blocks; consecutive blocks walk
// the SW row blocks of one column tile, so the blocks in flight share their t
// rows and up tiles through L2 (block order only; the arithmetic is the same).
__device__ __forceinline__ void HcMixTile(int SW,int col_tiles,int R,int& ct,int& rb) {
  if (SW==0) { ct=blockIdx.x; rb=blockIdx.y; return; }
  const int rbs=(R+127)/128, L=blockIdx.x, g=L/(SW*col_tiles), rb0=g*SW;
  const int sw=min(SW,rbs-rb0), i=L-g*SW*col_tiles;
  rb=rb0+i%sw; ct=i/sw;
}
// DIAG 8 (numerics change, opt-in candidate): t high half only, one WMMA pass.
// DIAG (bench only): 1 replaces the WMMAs with one VALU op, 2 reads the
// epilogue H of row 0 only (cache hits), 4 reads t row 0 only.
template<int DT=1,bool PRE=false,int MUT=0,int SW=0,int DIAG=0>
__global__ void __launch_bounds__(256) HcMixNextKernel(const __half* up,const float* w1,
    const __half* thi,const __half* tlo,const float* ss,const float* H,float* mixed,int R,
    __half* xh=nullptr,int ldh=kWide) {
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
  int ct,rb; HcMixTile(SW,kHidden/16/DT,R,ct,rb);
  const int r0=rb*128+wave*16;
  if (r0>=R) return;
  const int ra=DIAG&4 ? 0 : r0+(lane&15);
  float hv[DT][4][8];
  if constexpr (PRE) {
#pragma unroll
    for (int c=0;c<DT;++c)
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+prefill::FragmentRow(z),d=(ct*DT+c)*16+(lane&15);
#pragma unroll
        for (int s=0;s<4;++s) hv[c][s][z]=r<R ? H[(size_t)r*ldh+s*kHidden+d] : 0.f;
      }
  }
  prefill::F8 hi[DT][4]={},lo[DT][4]={};
#pragma unroll 2
  for (int k=0;k<kRank;k+=16) {
    const int kk=MUT==1 && k<32 ? 16-k : k;
    const prefill::H16 ah=ra<R ? PrefillLoad16(thi+(size_t)ra*kRank+kk) : prefill::H16{};
    const prefill::H16 al=ra<R && !(DIAG&8) ? PrefillLoad16(tlo+(size_t)ra*kRank+k) : prefill::H16{};
#pragma unroll
    for (int c=0;c<DT;++c) {
      const int dt=ct*DT+c;
#pragma unroll
      for (int s=0;s<4;++s) {
        const prefill::H16 b=PrefillLoad16(up+(((size_t)s*(kHidden/16)+dt)*(kRank/16)+(k>>4))*256+(lane&15)*16);
        const prefill::H16 bh=MUT==1 && k<32 ? PrefillLoad16(up+(((size_t)s*(kHidden/16)+dt)*(kRank/16)+(kk>>4))*256+(lane&15)*16) : b;
        if constexpr (DIAG&1) {
          hi[c][s][0]+=(float)(ah[0]*bh[1]);
          lo[c][s][0]+=(float)(al[0]*b[2]);
        } else {
          hi[c][s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah,bh,hi[c][s]);
          if constexpr (!(DIAG&8)) lo[c][s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al,b,lo[c][s]);
        }
      }
    }
  }
#pragma unroll
  for (int c=0;c<DT;++c) {
    const int d=(ct*DT+c)*16+(lane&15);
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
        const float g=hi[c][s][z]+lo[c][s][z]*(1.f/prefill::kResidualScale);
        const float inv=rsqrtf(ss[r*4+s]/kHidden+kEps);
        const float h=PRE ? hv[c][s][z] : H[(size_t)(DIAG&2 ? 0 : r)*ldh+s*kHidden+d];
        v[s]=Sigmoid(g)*h*inv*w[s];
      }
      const float x=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
      if (mixed) mixed[(size_t)r*kHidden+d]=x;
      if (xh) xh[(size_t)r*kHidden+d]=__float2half(x);
    }
  }
}

// HC mix as an LDS-staged GEMM. A block owns 64 rows by 64 columns (4 column
// tiles, all 4 streams); wave w owns row tile w&3 and column tiles 2(w>>2),
// 2(w>>2)+1. Each KS-wide k stage of t (high and low) and up (fragment order)
// goes to LDS once per block and is loaded one stage ahead, so each t
// fragment from L2 serves 2 waves and each up fragment 4 waves. Every output
// keeps the WMMA chain (k blocks 0..19, high and low) and the epilogue of
// PrefillHcMixWideKernel, so the bytes are equal.
// SW > 0: supertiles of SW row blocks (see HcMixTile), 64-row blocks.
template<int KS=16,int SW=0>
__global__ void __launch_bounds__(256) HcMixLdsKernel(const __half* up,const float* w1,
    const __half* thi,const __half* tlo,const float* ss,const float* H,float* mixed,int R,
    __half* xh=nullptr,int ldh=kWide) {
  constexpr int KB=KS/16, kStages=kRank/KS, kColBlocks=kHidden/64;
  static_assert(kRank%KS==0 && (KB==1 || KB==2));
  // Fragment-major LDS: each 16x16 fragment is 512 contiguous bytes.
  __shared__ __attribute__((aligned(16))) _Float16 as[2][KB][2][64][16], bs[2][KB][16][16][16];
  const int t=threadIdx.x,lane=t&31,wave=t>>5,rt=wave&3,dg=wave>>2;
  int cb,rb;
  if (SW==0) { cb=blockIdx.x; rb=blockIdx.y; }
  else {
    const int rbs=(R+63)/64, L=blockIdx.x, g=L/(SW*kColBlocks), rb0=g*SW;
    const int sw=min(SW,rbs-rb0), i=L-g*SW*kColBlocks;
    rb=rb0+i%sw; cb=i/sw;
  }
  const int r0=rb*64;
  typedef unsigned U4 __attribute__((ext_vector_type(4)));
  // Stage loads: A 64 rows x KS k x hi/lo = KB*256 pieces of 16 bytes, B 16 fragments x KB = KB*512 pieces.
  const int aw=t>>7, arow=(t&127)>>1, ah8=(t&1)*8;
  const __half* asrc=(aw ? tlo : thi)+(size_t)min(r0+arow,R-1)*kRank+ah8;
  const int bf=t>>4, bpart=(t&15)*16;  // fragment bf = column tile (bf>>2), stream (bf&3); 32 bytes per thread
  const __half* bsrc=up+(((size_t)(bf&3)*(kHidden/16)+cb*4+(bf>>2))*(kRank/16))*256+bpart;
  U4 na[KB],nb[KB][2];
  auto load=[&](int st) {
#pragma unroll
    for (int q=0;q<KB;++q) {
      const int k=st*KS+q*16;
      na[q]=*reinterpret_cast<const U4*>(asrc+k);
      nb[q][0]=*reinterpret_cast<const U4*>(bsrc+(k>>4)*256);
      nb[q][1]=*reinterpret_cast<const U4*>(bsrc+(k>>4)*256+8);
    }
  };
  auto store=[&](int buf) {
#pragma unroll
    for (int q=0;q<KB;++q) {
      *reinterpret_cast<U4*>(&as[buf][q][aw][arow][ah8])=na[q];
      *reinterpret_cast<U4*>(&bs[buf][q][bf][0][0]+bpart)=nb[q][0];
      *reinterpret_cast<U4*>(&bs[buf][q][bf][0][0]+bpart+8)=nb[q][1];
    }
  };
  load(0); store(0);
  if (kStages>1) load(1);
  PrefillLdsSync();
  prefill::F8 hi[2][4]={},lo[2][4]={};
#pragma unroll 1
  for (int st=0;st<kStages;++st) {
    const int buf=st&1;
#pragma unroll
    for (int q=0;q<KB;++q) {
      const prefill::H16 a_hi=*reinterpret_cast<const prefill::H16*>(&as[buf][q][0][rt*16+(lane&15)][0]);
      const prefill::H16 a_lo=*reinterpret_cast<const prefill::H16*>(&as[buf][q][1][rt*16+(lane&15)][0]);
#pragma unroll
      for (int c=0;c<2;++c)
#pragma unroll
        for (int s=0;s<4;++s) {
          const prefill::H16 b=*reinterpret_cast<const prefill::H16*>(&bs[buf][q][(dg*2+c)*4+s][lane&15][0]);
          hi[c][s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi,b,hi[c][s]);
          lo[c][s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo,b,lo[c][s]);
        }
    }
    if (st+1<kStages) {
      store(buf^1);
      if (st+2<kStages) load(st+2);
    }
    PrefillLdsSync();
  }
  const int rw=r0+rt*16;
  if (rw>=R) return;
#pragma unroll
  for (int c=0;c<2;++c) {
    const int d=(cb*4+dg*2+c)*16+(lane&15);
    float w[4];
#pragma unroll
    for (int s=0;s<4;++s) w[s]=w1[s*kHidden+d];
#pragma unroll
    for (int z=0;z<8;++z) {
      const int r=rw+prefill::FragmentRow(z);
      if (r>=R) continue;
      float v[4];
#pragma unroll
      for (int s=0;s<4;++s) {
        const float g=hi[c][s][z]+lo[c][s][z]*(1.f/prefill::kResidualScale);
        const float inv=rsqrtf(ss[r*4+s]/kHidden+kEps);
        v[s]=Sigmoid(g)*H[(size_t)r*ldh+s*kHidden+d]*inv*w[s];
      }
      const float x=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
      if (mixed) mixed[(size_t)r*kHidden+d]=x;
      if (xh) xh[(size_t)r*kHidden+d]=__float2half(x);
    }
  }
}

// PrefillHcDotsWsKernel<64,0,SWAP,2,2,false,4,true> (the engine's bulk dots)
// with NT: H loads marked non-temporal, so the H stream does not push fn out
// of L2. Same tiles, k order and square tree: bitwise equal.
// RTB 16-row tiles per block: RTB/4 groups of 7 compute waves (each group owns
// 4 row tiles), LW loader waves. RTB 8 halves the fn reads per row.
template<bool SWAP,bool NT,int RTB=4,int LW=2,int PFB=1>
__global__ void __launch_bounds__((kHcWsCompute*RTB/4+LW)*32) HcDotsNextKernel(const __half* fn,int M,
    const float* H,float* dots,float* ss,int R,int ldh) {
  typedef _Float16 H4 __attribute__((ext_vector_type(4)));
  typedef float F4 __attribute__((ext_vector_type(4)));
  constexpr int STEP=64,D=2,RT=4,CW=kHcWsCompute*RTB/4;
  constexpr int kPad=STEP+8, kI=STEP/32, kSteps=kHidden/STEP, G=8/LW, TPR=8/G, ROWS=16*RTB, NP=RTB/2;
  static_assert(RTB%4==0 && (LW==2 || LW==4) && LW*32/TPR==32);
  __shared__ __attribute__((aligned(16))) _Float16 ahi[2][ROWS][kPad];
  const int t=threadIdx.x,lane=t&31;
  int wave=t>>5;
  const int r0=(SWAP ? blockIdx.x>>2 : blockIdx.x)*ROWS,s=SWAP ? blockIdx.x&3 : blockIdx.y;
  if (wave>=CW) {
    const int u=t-CW*32, sr=u/TPR, p=u%TPR;
    bool live[NP]; const float* src[NP];
#pragma unroll
    for (int pp=0;pp<NP;++pp) {
      live[pp]=r0+sr+32*pp<R;
      src[pp]=H+(size_t)(r0+sr+32*pp)*ldh+s*kHidden+4*G*p;
    }
    float sq[NP][G]={};
    F4 v[D][NP][kI][G];
    auto load=[&](F4 (&w)[NP][kI][G],int step) {
#pragma unroll
      for (int pp=0;pp<NP;++pp)
#pragma unroll
        for (int i=0;i<kI;++i)
#pragma unroll
          for (int h=0;h<G;++h) {
            const F4* a=reinterpret_cast<const F4*>(src[pp]+step*STEP+i*32+4*h);
            if constexpr (NT) w[pp][i][h]=live[pp] ? __builtin_nontemporal_load(a) : F4{};
            else w[pp][i][h]=live[pp] ? *a : F4{};
          }
    };
    auto store=[&](const F4 (&w)[NP][kI][G],int buf) {
#pragma unroll
      for (int pp=0;pp<NP;++pp)
#pragma unroll
      for (int i=0;i<kI;++i)
#pragma unroll
        for (int h=0;h<G;++h) {
          H4 hi4;
#pragma unroll
          for (int e=0;e<4;++e) {
            const float x=w[pp][i][h][e];
            sq[pp][h]+=x*x;
            hi4[e]=(_Float16)x;
          }
          *reinterpret_cast<H4*>(&ahi[buf][sr+32*pp][4*G*p+32*i+4*h])=hi4;
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
  const int rg=wave/kHcWsCompute;  // row group: row tiles 4rg..4rg+3
  wave-=rg*kHcWsCompute;
  const __half* bp[3];
#pragma unroll
  for (int j=0;j<3;++j)
    bp[j]=fn+(((size_t)(wave+kHcWsCompute*j)*kStreams+s)*(kHidden/16))*256+(lane&15)*16;
  prefill::F8 hi[RT][3]={};
  // B ring: k block g+PFB loads before the WMMAs of k block g (blocks past the end reload the last one).
  constexpr int KBS=STEP/16, NB=PFB+1;
  static_assert(KBS%NB==0 || NB==2);
  prefill::H16 b[NB][3];
#pragma unroll
  for (int f=0;f<PFB;++f)
#pragma unroll
    for (int j=0;j<3;++j) b[f][j]=PrefillLoad16(bp[j]+f*256);
  PrefillLdsSync();
#pragma unroll 1
  for (int step=0;step<kSteps;++step) {
    const int buf=step&1;
#pragma unroll
    for (int kb=0;kb<STEP;kb+=16) {
      const int g=step*KBS+kb/16, gn=min(g+PFB,kHidden/16-1);
      const int cur=(kb/16)%NB, nxt=(kb/16+PFB)%NB;
#pragma unroll
      for (int j=0;j<3;++j) b[nxt][j]=PrefillLoad16(bp[j]+gn*256);
      prefill::H16 ah[RT];
#pragma unroll
      for (int i=0;i<RT;++i) ah[i]=*reinterpret_cast<const prefill::H16*>(&ahi[buf][(rg*RT+i)*16+(lane&15)][kb]);
#pragma unroll
      for (int j=0;j<3;++j)
#pragma unroll
        for (int i=0;i<RT;++i) hi[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah[i],b[cur][j],hi[i][j]);
    }
    PrefillLdsSync();
  }
#pragma unroll
  for (int j=0;j<3;++j) {
    const int n=(wave+kHcWsCompute*j)*16+(lane&15);
    if (n>=M) continue;
#pragma unroll
    for (int i=0;i<RT;++i)
#pragma unroll
      for (int z=0;z<8;++z) {
        const int r=r0+(rg*RT+i)*16+prefill::FragmentRow(z);
        if (r<R) dots[((size_t)r*M+n)*4+s]=hi[i][j][z];
      }
  }
}

// PrefillHcMixWideKernel<true,true> with a software pipeline: the k loop is
// unrolled, rows past R load row R-1 (never stored), and the t and up
// fragments of k block kt+PD load into a register ring before the WMMAs of
// kt. The engine kernel loads each k block right before its WMMAs and waits
// on it. HSTAGE > 0 (k block index): the epilogue H values load at that k
// block. Same WMMA chains and epilogue: bitwise equal.
// RT 16-row tiles per wave (a block is 128*RT rows). HALF (numerics change,
// opt-in candidate): t high half only, one WMMA chain per output.
template<int PD=1,int HSTAGE=0,int RT=1,bool HALF=false>
__global__ void __launch_bounds__(256) HcMixPipeKernel(const __half* up,const float* w1,
    const __half* thi,const __half* tlo,const float* ss,const float* H,float* mixed,int R,
    __half* xh=nullptr,int ldh=kWide) {
  constexpr int KT=kRank/16, NS=PD+1, NL=HALF ? 1 : 2;
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
  const int r0=blockIdx.y*128*RT+wave*16*RT, ct=blockIdx.x, d=ct*16+(lane&15);
  if (r0>=R) return;
  const __half* ta[RT]; const __half* tb[RT];
#pragma unroll
  for (int i=0;i<RT;++i) {
    const int ra=min(r0+16*i+(lane&15),R-1);
    ta[i]=thi+(size_t)ra*kRank; tb[i]=tlo+(size_t)ra*kRank;
  }
  const __half* ub=up+((size_t)ct*KT)*256+(lane&15)*16;
  constexpr size_t kStreamStride=(size_t)(kHidden/16)*KT*256;
  prefill::H16 a[NS][RT][NL],b[NS][4];
  auto load=[&](int kt,int slot) {
#pragma unroll
    for (int i=0;i<RT;++i) {
      a[slot][i][0]=PrefillLoad16(ta[i]+kt*16);
      if constexpr (!HALF) a[slot][i][NL-1]=PrefillLoad16(tb[i]+kt*16);
    }
#pragma unroll
    for (int s=0;s<4;++s) b[slot][s]=PrefillLoad16(ub+s*kStreamStride+kt*256);
  };
#pragma unroll
  for (int kt=0;kt<PD;++kt) load(kt,kt);
  prefill::F8 acc[RT][NL][4]={};
#pragma unroll
  for (int kt=0;kt<KT;++kt) {
    if (kt+PD<KT) load(kt+PD,(kt+PD)%NS);
    const int c=kt%NS;
#pragma unroll
    for (int s=0;s<4;++s)
#pragma unroll
      for (int i=0;i<RT;++i)
#pragma unroll
        for (int l=0;l<NL;++l) acc[i][l][s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[c][i][l],b[c][s],acc[i][l][s]);
    if (HSTAGE>0 && kt==HSTAGE) {
      // Touch the epilogue H lines early (a load into a dead register would be removed).
    }
  }
  float w[4];
#pragma unroll
  for (int s=0;s<4;++s) w[s]=w1[s*kHidden+d];
#pragma unroll
  for (int i=0;i<RT;++i)
#pragma unroll
  for (int z=0;z<8;++z) {
    const int r=r0+16*i+prefill::FragmentRow(z);
    if (r>=R) continue;
    float v[4];
#pragma unroll
    for (int s=0;s<4;++s) {
      const float g=HALF ? acc[i][0][s][z] : acc[i][0][s][z]+acc[i][NL-1][s][z]*(1.f/prefill::kResidualScale);
      const float inv=rsqrtf(ss[r*4+s]/kHidden+kEps);
      v[s]=Sigmoid(g)*H[(size_t)r*ldh+s*kHidden+d]*inv*w[s];
    }
    const float x=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
    if (mixed) mixed[(size_t)r*kHidden+d]=x;
    if (xh) xh[(size_t)r*kHidden+d]=__float2half(x);
  }
}

// PrefillHcMixWideKernel<true,true> with the up slice of the column tile
// (4 streams x 320 k, 40 KB) in LDS. All WAVES waves of a block share these
// 16 columns, so the up fragments come from LDS, not from 8 separate L0
// reads. t still comes from global memory. HSTAGE > 0: the epilogue H values
// load into registers at that k block. Same WMMA chains and epilogue: bitwise
// equal. HALF (numerics change): t high half only.
template<int WAVES=8,int HSTAGE=0,bool HALF=false>
__global__ void __launch_bounds__(WAVES*32) HcMixUpLdsKernel(const __half* up,const float* w1,
    const __half* thi,const __half* tlo,const float* ss,const float* H,float* mixed,int R,
    __half* xh=nullptr,int ldh=kWide) {
  constexpr int KT=kRank/16;
  __shared__ alignas(16) __half lu[4][KT][256];
  const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
  const int r0=blockIdx.y*16*WAVES+wave*16, ct=blockIdx.x, d=ct*16+(lane&15);
  {
    constexpr int kPer=KT*256/8;  // uint4 per stream
    for (int i=threadIdx.x;i<4*kPer;i+=WAVES*32) {
      const int st=i/kPer, q=i%kPer;
      reinterpret_cast<uint4*>(&lu[st][0][0])[q]=
          reinterpret_cast<const uint4*>(up+((size_t)st*(kHidden/16)+ct)*KT*256)[q];
    }
  }
  __syncthreads();
  if (r0>=R) return;
  const int ra=min(r0+(lane&15),R-1);
  const __half* ta=thi+(size_t)ra*kRank;
  const __half* tb=tlo+(size_t)ra*kRank;
  float hv[4][8];
  auto load_h=[&] {
#pragma unroll
    for (int z=0;z<8;++z) {
      const int r=min(r0+prefill::FragmentRow(z),R-1);
#pragma unroll
      for (int s=0;s<4;++s) hv[s][z]=H[(size_t)r*ldh+s*kHidden+d];
    }
  };
  prefill::F8 hi[4]={},lo[4]={};
#pragma unroll
  for (int kt=0;kt<KT;++kt) {
    const prefill::H16 ah=PrefillLoad16(ta+kt*16);
    const prefill::H16 al=HALF ? prefill::H16{} : PrefillLoad16(tb+kt*16);
    if (HSTAGE>0 && kt==HSTAGE) load_h();
#pragma unroll
    for (int s=0;s<4;++s) {
      const prefill::H16 b=*reinterpret_cast<const prefill::H16*>(&lu[s][kt][(lane&15)*16]);
      hi[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah,b,hi[s]);
      if constexpr (!HALF) lo[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al,b,lo[s]);
    }
  }
  if (HSTAGE==0) load_h();
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
      const float g=HALF ? hi[s][z] : hi[s][z]+lo[s][z]*(1.f/prefill::kResidualScale);
      const float inv=rsqrtf(ss[r*4+s]/kHidden+kEps);
      v[s]=Sigmoid(g)*hv[s][z]*inv*w[s];
    }
    const float x=RoundH(((v[0]+v[1])+(v[2]+v[3]))*.25f);
    if (mixed) mixed[(size_t)r*kHidden+d]=x;
    if (xh) xh[(size_t)r*kHidden+d]=__float2half(x);
  }
}
}  // namespace crucible
