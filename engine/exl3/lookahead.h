// The helper reads only immutable static weights. It never changes model values.
#pragma once
#include "../gfx1151/common.h"
#include "row_policy.h"

namespace crucible {
constexpr size_t kLookaheadSlab=256<<10;
struct LookaheadState {
  const uint8_t* base;
  unsigned long long total,window,progress,fetched;
  unsigned next,active;
  unsigned hold;  // LookaheadHoldKernel blocks below this index wait (MoeSgKernel OPT bit 3 sets it for its launch)
};
__device__ LookaheadState* g_lookahead=nullptr;
__device__ uint32_t g_lookahead_sink;

__device__ __forceinline__ void PublishWeights(const void* weights) {
  if (blockIdx.x!=0 || blockIdx.y!=0 || threadIdx.x!=0) return;
  LookaheadState* s=g_lookahead;
  if (!s || !__hip_atomic_load(&s->active,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)) return;
  const uintptr_t p=reinterpret_cast<uintptr_t>(weights), b=reinterpret_cast<uintptr_t>(s->base);
  if (p>=b && p-b<s->total)
    __hip_atomic_store(&s->progress,static_cast<unsigned long long>(p-b),__ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
}
__global__ void LookaheadInitKernel(LookaheadState* s) { g_lookahead=s; }
__global__ void LookaheadResetKernel(LookaheadState* s, unsigned long long window) {
  s->window=window; s->progress=0; s->next=0; s->fetched=0;
  __hip_atomic_store(&s->active,1u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
}
__global__ void LookaheadStopKernel(LookaheadState* s) {
  __hip_atomic_store(&s->active,0u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
}
__global__ void __launch_bounds__(256) LookaheadKernel(LookaheadState* s) {
  __shared__ unsigned slab;
  __shared__ bool stop,skip;
  uint32_t acc=0;
  unsigned long long fetched=0;
  for (;;) {
    if (threadIdx.x==0) {
      auto p=__hip_atomic_load(&s->progress,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
      atomicMax(&s->next,unsigned(p/kLookaheadSlab));
      slab=atomicAdd(&s->next,1u);
      const auto off=static_cast<unsigned long long>(slab)*kLookaheadSlab;
      do {
        stop=!__hip_atomic_load(&s->active,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT) || off>=s->total;
        p=__hip_atomic_load(&s->progress,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
        if (stop || off<p+s->window) break;
        __builtin_amdgcn_s_sleep(2);
      } while (true);
      skip=off+kLookaheadSlab<=p;
    }
    __syncthreads();
    if (stop) break;
    if (!skip) {
      const auto off=static_cast<unsigned long long>(slab)*kLookaheadSlab;
      const auto left=s->total-off;
      const size_t count=(left<kLookaheadSlab ? left : kLookaheadSlab)/16;
      const uint4* p=reinterpret_cast<const uint4*>(s->base+off);
#pragma unroll 16
      for (size_t i=threadIdx.x;i<count;i+=256) {
        const uint4 v=p[i]; acc^=v.x^v.y^v.z^v.w;
      }
      fetched+=count*16;
    }
    __syncthreads();
  }
  if (threadIdx.x==0) atomicAdd(&s->fetched,fetched);
  if (acc==0x9e3779b9u) atomicExch(&g_lookahead_sink,acc);
}

// LookaheadKernel that also waits while s->hold is set (CRUCIBLE_MOE_B128=12, moe_b128_switch.h): the expert
// stream then has the memory system alone, and the helper fetches its window during the other kernels.
__global__ void __launch_bounds__(256) LookaheadHoldKernel(LookaheadState* s) {
  __shared__ unsigned slab;
  __shared__ bool stop,skip;
  uint32_t acc=0;
  unsigned long long fetched=0;
  for (;;) {
    if (threadIdx.x==0) {
      auto p=__hip_atomic_load(&s->progress,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
      atomicMax(&s->next,unsigned(p/kLookaheadSlab));
      slab=atomicAdd(&s->next,1u);
      const auto off=static_cast<unsigned long long>(slab)*kLookaheadSlab;
      do {
        stop=!__hip_atomic_load(&s->active,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT) || off>=s->total;
        p=__hip_atomic_load(&s->progress,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
        const bool held=blockIdx.x<__hip_atomic_load(&s->hold,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
        if (stop || (off<p+s->window && !held)) break;
        __builtin_amdgcn_s_sleep(2);
      } while (true);
      skip=off+kLookaheadSlab<=p;
    }
    __syncthreads();
    if (stop) break;
    if (!skip) {
      const auto off=static_cast<unsigned long long>(slab)*kLookaheadSlab;
      const auto left=s->total-off;
      const size_t count=(left<kLookaheadSlab ? left : kLookaheadSlab)/16;
      const uint4* p=reinterpret_cast<const uint4*>(s->base+off);
#pragma unroll 16
      for (size_t i=threadIdx.x;i<count;i+=256) {
        const uint4 v=p[i]; acc^=v.x^v.y^v.z^v.w;
      }
      fetched+=count*16;
    }
    __syncthreads();
  }
  if (threadIdx.x==0) atomicAdd(&s->fetched,fetched);
  if (acc==0x9e3779b9u) atomicExch(&g_lookahead_sink,acc);
}

// MoeSgKernel OPT bit 3: the hold of v helper blocks for one launch (set at the start or after the ramp, cleared at
// the first empty claim or by the last wave).
__device__ __forceinline__ void LookaheadHold(unsigned v) {
  if (LookaheadState* s=g_lookahead) __hip_atomic_store(&s->hold,v,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
}

class LookaheadRunner {
 public:
  LookaheadRunner() = default;
  LookaheadRunner(const LookaheadRunner&) = delete;
  LookaheadRunner& operator=(const LookaheadRunner&) = delete;
  ~LookaheadRunner() {
    if (!state_) return;
    Stop();
    LookaheadInitKernel<<<1,1>>>(nullptr);
    HIP_OK(hipStreamSynchronize(0));
    HIP_OK(hipStreamDestroy(stream_));
    HIP_OK(hipEventDestroy(ready_));
    HIP_OK(hipFree(state_));
  }
  // hold: run LookaheadHoldKernel (CRUCIBLE_MOE_B128=12).
  void Init(uint8_t* base,size_t bytes,bool enabled,bool hold=false) {
    if (!enabled) return;
    hold_=hold;
    LookaheadState state{}; state.base=base; state.total=bytes;
    HIP_OK(hipMalloc(&state_,sizeof(state)));
    HIP_OK(hipMemcpy(state_,&state,sizeof(state),hipMemcpyHostToDevice));
    LookaheadInitKernel<<<1,1>>>(state_);
    HIP_OK(hipStreamCreateWithFlags(&stream_,hipStreamNonBlocking));
    HIP_OK(hipEventCreateWithFlags(&ready_,hipEventDisableTiming));
  }
  void Start(RowPolicy policy) {
    running_=state_ && policy.blocks!=0;
    if (!running_) return;
    LookaheadResetKernel<<<1,1>>>(state_,size_t(policy.mib)<<20);
    HIP_OK(hipEventRecord(ready_,0));
    HIP_OK(hipStreamWaitEvent(stream_,ready_,0));
    if (hold_) LookaheadHoldKernel<<<policy.blocks,256,0,stream_>>>(state_);
    else LookaheadKernel<<<policy.blocks,256,0,stream_>>>(state_);
  }
  // CRUCIBLE_DRAFT_CHAIN lead: the stop kernel without the host wait. The helper blocks end when stream 0 reaches
  // the stop kernel (they only read weights), and a later Start queues behind them on stream_.
  void StopAsync() {
    if (!running_) return;
    LookaheadStopKernel<<<1,1>>>(state_);
    running_=false;
  }
  void Stop() {
    if (!running_) return;
    LookaheadStopKernel<<<1,1>>>(state_);
    HIP_OK(hipStreamSynchronize(stream_)); running_=false;
  }
 private:
  LookaheadState* state_=nullptr;
  hipStream_t stream_=nullptr;
  hipEvent_t ready_=nullptr;
  bool running_=false;
  bool hold_=false;
};
}  // namespace crucible
