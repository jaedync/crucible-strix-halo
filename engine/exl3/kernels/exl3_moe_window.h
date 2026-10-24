#pragma once
#include "exl3_gemm_pair.h"

namespace crucible {
// Each A fragment feeds both column tiles before reuse. The small register
// window avoids keeping all RM fragments live beside both accumulator sets.
// Paired LDS publication uses6144 bytes and only wave-local fences.
// KEEP (the M1 pair kernel): the accumulators go to keep[n * RM + r]
// instead of the partial; the WMMA sequence does not change.
// COPY > 0 only makes a separate instantiation (the same code): see PrefillMoeGemmBody.
template<int K,int RM,int WINDOW,bool HALF=false,bool SKIP=true,bool KEEP=false,int COPY=0>
__device__ __forceinline__ void MoeWindowBody(const Exl3GemmJob& j,
    const Exl3GemmItem& it,uint32_t* ws,GemmF8* keep=nullptr) {
  static_assert(WINDOW>=1&&WINDOW<=3&&WINDOW<=RM);
// The host pass must not instantiate AMD register constraints.
#if defined(__HIP_DEVICE_COMPILE__)
  constexpr int NB=2,T=TileDwords(K);  // 8 K dwords; K3.5 (code 35): lane_window.h
  const int lane=threadIdx.x&31,wave=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int b=it.b0+wave*NB,A=j.I/16;
  if(b*16<j.O) {
    const LaneWindow<K> win(lane);
    __amdgpu_buffer_rsrc_t rs[NB];
#pragma unroll
    for(int n=0;n<NB;++n)rs[n]=StreamRsrc(j.tiles+(size_t)(b+n)*A*T,A*T*4);
    GemmF8 acc[NB][RM]={};
    const uint32_t xbytes=(uint32_t)j.Xrows*j.I*2;
    const auto xrs=StreamRsrc(reinterpret_cast<const uint32_t*>(j.x),xbytes);
    const int astep=j.Xrows*32;
    uint32_t xo[RM];
#pragma unroll
    for(int r=0;r<RM;++r) {
      const int row=it.m0+r*16+(lane&15);
      xo[r]=row<j.M?(uint32_t)row*32:xbytes;
    }
    const int o0=win.d0*4,o1=((win.d0+1)%T)*4,o2=((win.d0+2)%T)*4;
    uint32_t raw[NB][3];
#pragma unroll
    for(int n=0;n<NB;++n) {
      raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0,0,0);
      raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1,0,0);
      raw[n][2]=0;
      if constexpr(K>=5)raw[n][2]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o2,0,0);
    }
    for(int a=0;a<A;++a) {
      GemmH16 afr[WINDOW];
#pragma unroll
      for(int r=0;r<WINDOW;++r)afr[r]=GemmAFragment<false>(xrs,xo[r],a,nullptr,0,astep);
      asm volatile("":::"memory");
      const int next=(a+1)*T*4;
#pragma unroll
      for(int n=0;n<NB;++n) {
        const uint32_t w0=raw[n][0],w1=raw[n][1],w2=raw[n][2];
        raw[n][0]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o0+next,0,0);
        raw[n][1]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o1+next,0,0);
        if constexpr(K>=5)raw[n][2]=__builtin_amdgcn_raw_buffer_load_b32(rs[n],o2+next,0,0);
        __half2 p[4];DecodeLane<K,kMulTrue16>(w0,w1,w2,win.shift,p);
        GemmNativeStore(p,ws+(wave*NB+n)*kGemmWaveWords,lane);
      }
      GemmWaveSync();
      const GemmH16 b0=GemmStoredFragment(ws+wave*NB*kGemmWaveWords,lane);
      const GemmH16 b1=GemmStoredFragment(ws+(wave*NB+1)*kGemmWaveWords,lane);
#pragma unroll
      for(int r=0;r<RM;++r) {
        if(SKIP&&it.m0+r*16>=j.M)break;
        const GemmH16 af=afr[r%WINDOW];
        acc[0][r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af,b0,acc[0][r]);
        acc[1][r]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af,b1,acc[1][r]);
        // The compiler otherwise hoists every A load and recreates afr[RM].
        asm volatile("" : : "v"(acc[0][r]),"v"(acc[1][r]) : "memory");
        if(r+WINDOW<RM)afr[r%WINDOW]=GemmAFragment<false>(xrs,xo[r+WINDOW],a,nullptr,0,astep);
      }
      GemmWaveSync();
    }
    if constexpr(KEEP) {
#pragma unroll
      for(int n=0;n<NB;++n)
#pragma unroll
        for(int r=0;r<RM;++r)keep[n*RM+r]=acc[n][r];
      return;
    }
    constexpr uint32_t bytes=HALF?2:4;
    const auto ps=StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial),(uint32_t)j.M*j.O*bytes);
#pragma unroll
    for(int n=0;n<NB;++n) {
      const uint32_t po=((uint32_t)(it.m0+lane/16)*j.O+(b+n)*16+(lane&15))*bytes;
#pragma unroll
      for(int r=0;r<RM;++r) {
        if(SKIP&&it.m0+r*16>=j.M)break;
#pragma unroll
        for(int z=0;z<8;++z) {
          const uint32_t off=po+(uint32_t)(r*16+2*z)*j.O*bytes;
          if constexpr(HALF) {
            const float v=acc[n][r][z];
            const auto h=__float2half(fabsf(v)>65504.f?copysignf(65504.f,v):v);
            __builtin_amdgcn_raw_buffer_store_b16(__half_as_ushort(h),ps,off,0,0);
          } else __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(acc[n][r][z]),ps,off,0,0);
        }
      }
    }
  }
#endif
}
// The dispatch removes tail branches inside K without running padded row tiles.
template<int K,int RM,bool HALF=false,int COPY=0> __device__ __forceinline__ void MoeLiveWindowBody(
    const Exl3GemmJob& j,const Exl3GemmItem& it,uint32_t* ws) {
  static_assert(RM>=1&&RM<=6);
  const int live=min(RM,(j.M-it.m0+15)/16);
#define LIVE_WINDOW(R) case R: if constexpr(R<=RM) \
    MoeWindowBody<K,R,(R<3?R:3),HALF,false,false,COPY>(j,it,ws); break
  switch(live) {
    LIVE_WINDOW(1);LIVE_WINDOW(2);LIVE_WINDOW(3);
    LIVE_WINDOW(4);LIVE_WINDOW(5);LIVE_WINDOW(6);
  }
#undef LIVE_WINDOW
}
}  // namespace crucible
