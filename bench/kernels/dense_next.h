// Candidate prefill dense kernels (prefill-hc). Each output keeps the k chain
// of PrefillDenseFastKernel (WMMA k blocks 0..159 in order, fp32 accumulate),
// so the bench compares bytes.
#pragma once
#include "prefill_hc.h"
#include "../../engine/exl3/prefill_linear.h"

namespace crucible {
// PrefillDenseFastKernel<DEPTH> with three changes:
// - a 1-D grid in supertiles: G consecutive row blocks take column block 0,
//   then column block 1, and so on, so a row block of x is read from DRAM
//   once and the next column blocks of it hit L2 (the 2-D grid walks all row
//   blocks per column block: x is read once per column block);
// - waves whose 64 columns are all past O skip their WMMAs, and waves with
//   fewer than 4 valid column tiles skip the rest (O = 513 has one valid
//   column in its last column block);
// - RB = 64: 4 waves own 64 rows by 128 columns (more blocks for small O).
// MUT (bench mutant): 1 runs the two k blocks of each stage in reverse order.
template<int DEPTH=2,int RB=128,int MUT=0>
__global__ void __launch_bounds__(RB*2) DenseNextKernel(const __half* W,int O,
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
          const int kb=MUT ? kBigStep-16-kq : kq;
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
          const int kb=MUT ? kBigStep-16-kq : kq;
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
