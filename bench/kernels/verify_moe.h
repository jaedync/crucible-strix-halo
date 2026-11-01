// verify-moe bench variants of the verify-step expert launch (bench/verify_moe.hip).
// Variant 0 is the engine launch (engine.hip MoeFusedLaunch for K4 experts). Every variant keeps each output's
// tile order and Dot2 sequence (MoeGemvCols), so it must match variant 0 byte for byte.
#pragma once
#include "../../engine/layers/moe_rows.h"

namespace crucible {

struct VerifyMoeVariant {
  const char* name;
  int item_blocks;  // column blocks per gate/up and down item (Build)
};

inline const VerifyMoeVariant& VerifyMoeVariantOf(int v) {
  static const VerifyMoeVariant kV[] = {
      {"engine C2 PF4 ib16", 16},  // 0
      {"C1 PF4 ib8", 8},           // 1
      {"C1 PF8 ib8", 8},           // 2
      {"C2 PF2 ib16", 16},         // 3
      {"C2 PF8 ib16", 16},         // 4
      {"C2 PF4 ib16 XC40", 16},    // 5
      {"probe (v0 + stamps)", 16}, // 6
      {"split gu / dn launches", 16},  // 7
      {"mix order = engine", 16},   // 8
      {"mix lag 2", 16},            // 9
      {"mix lag 4", 16},            // 10
      {"mix lag 8", 16},            // 11
      {"mix lag 16", 16},           // 12
      {"fillx4", 16},               // 13
      {"fillx4 + pre", 16},         // 14
      {"fillx4 probe", 16},         // 15
      {"fillx4 + pre probe", 16},   // 16
      {"fillx4 stag wave", 16},     // 17
      {"fillx4 stag hash", 16},     // 18
      {"fillx4 stag hash x2", 16},  // 19
      {"fillx stag wave", 16},      // 20
      {"fillx stag hash", 16},      // 21
      {"fillx4 stag hash probe", 16},  // 22
      {"fillx stag wave probe", 16},   // 23
      {"mix C1 PF8 ib8 engine order", 8},  // 24
      {"mix C1 PF8 ib8 lag 2", 8},  // 25
      {"mix C1 PF8 ib8 lag 4", 8},  // 26
      {"mix C1 PF4 ib8 lag 2", 8},  // 27
      {"gu perm random", 16},       // 28
      {"gu perm item-major", 16},   // 29
      {"gu perm g/u interleave", 16},  // 30
      {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16},  // 31-39
      {"flow C2 PF4 engine order", 16},  // 40
      {"flow C2 PF4 lag 2", 16},         // 41
      {"flow C2 PF4 lag 4", 16},         // 42
      {"flow C1 PF8 lag 4", 8},          // 43
      {"flow C1 PF8 lag 8", 8},          // 44
      {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16},  // 45-49
      {"stream b128 D4 G160", 16},       // 50
      {"stream b128 D8 G160", 16},       // 51
      {"stream b128 D8 G80", 16},        // 52
      {"stream b128 D2 G160", 16},       // 53
      {"stream win D8 G160", 16},        // 54
      {"stream win D8 G120", 16},        // 55
      {"stream win D8 G120 valu64", 16}, // 56
      {"stream win D8 G120 valu96", 16}, // 57
      {"stream win D16 G120", 16},       // 58
      {"stream win D4 G120", 16},        // 59
      {"own C2 PF8", 16},                // 60
      {"own C2 PF12", 16},               // 61
      {"own C2 PF16", 16},               // 62
      {"own C2 PF4", 16},                // 63
      {"own C1 PF16 ib8", 8},            // 64
      {"own C1 PF24 ib8", 8},            // 65
      {"t128 C2 PG2", 16},               // 66
      {"t128 C2 PG4", 16},               // 67
      {"t128 C1 PG4 ib8", 8},            // 68
      {"t128 C1 PG8 ib8", 8},            // 69
      {"stream own D8 G120", 16},        // 70
      {"stream own D16 G120", 16},       // 71
      {"stream own D24 G120", 16},       // 72
      {"stream own D16 G80", 16},        // 73
      {"stream own D16 G160", 16},       // 74
      {"stream own D16 bperm valu64", 16},  // 75
      {"stream b128 D8 G120", 16},       // 76
      {"stream own D32 G120", 16},       // 77
      {"-", 16}, {"-", 16},              // 78-79
      {"engine body cap 16K", 16},       // 80
      {"engine body cap 24K", 16},       // 81
      {"engine body cap 32K", 16},       // 82
      {"engine body cap 48K", 16},       // 83
      {"engine body cap 64K", 16},       // 84
      {"t128 C2 PG4 cap 24K", 16},       // 85
      {"t128 C2 PG4 cap 32K", 16},       // 86
      {"t128 C2 PG4 cap 48K", 16},       // 87
      {"own C2 PF16 cap 32K", 16},       // 88
      {"own C2 PF16 cap 48K", 16},       // 89
      {"stream b128 D8 G40", 16},        // 90
      {"stream b128 D8 G20", 16},        // 91
      {"stream win D16 G40", 16},        // 92
      {"stream win D16 G20", 16},        // 93
      {"pre (weights before x)", 16},    // 94
      {"fast x fill", 16},               // 95
      {"pre + fast x fill", 16},         // 96
      {"pre + fast x PF8", 16},          // 97
      {"pre + fast x PF6", 16},          // 98
      {"phase: gate/up only", 16},       // 99
      {"phase: down only, no wait", 16}, // 100
      {"phase: all, no wait", 16},       // 101
      {"phase: all, no wait, no x", 16}, // 102
      {"phase: gate/up only, no x", 16}, // 103
      {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16}, {"-", 16},  // 104-109
      {"q4 W5 D4 ib20", 20},             // 110 (moe_b128.h)
      {"q4 W5 D8 ib20", 20},             // 111
      {"q4 W5 D2 ib20", 20},             // 112
      {"q4 W8 D4 ib32", 32},             // 113
      {"q4 W10 D4 ib40", 40},            // 114
      {"q4 W5 D4 knock: x in regs", 20}, // 115
      {"q4 W5 D4 knock: no decode", 20}, // 116
      {"q4 W5 D4 knock: decode only", 20},  // 117
      {"b64 C2 D4 ib16", 16},            // 118
      {"b64 C2 D8 ib16", 16},            // 119
      {"streamq4 D8 G120 none", 16},     // 120
      {"streamq4 D8 G120 decode", 16},   // 121
      {"streamq4 D8 G120 dec+dot reg", 16},  // 122
      {"streamq4 D8 G120 dec+dot lds", 16},  // 123
      {"streamq4 D8 G120 dot lds only", 16}, // 124
      {"streamq4 D4 G120 dec+dot lds", 16},  // 125
      {"streamq4 D8 G40 dec+dot lds", 16},   // 126
      {"(unused)", 16},                      // 127
      {"(unused)", 16},                      // 128
      {"(unused)", 16},                      // 129
      {"sg D5 K6d1 W12/8 G120 (engine)", 16},  // 130 (moe_sg.h)
      {"sg D10 K6d1 W8 G120", 16},           // 131
      {"sg D5 cache0 G120", 16},             // 132
      {"sg D2 W12 G120", 16},                // 133
      {"sg D5 no VGPR cap G120", 16},        // 134
      {"sg MUTANT shuffle order", 16},       // 135
      {"sg MUTANT drop hb tile 3", 16},      // 136
      {"sg MUTANT swap tiles 0/1", 16},      // 137
      {"sg D5 K6d2 G120", 16},               // 138
      {"(unused)", 16},                      // 139
      {"sg D5 W12/8 CBD4", 16},              // 140
      {"sg D5 nocap CBD4", 16},              // 141
      {"sg D2 W12 CBD4", 16},                // 142
      {"sg D5 W12/8 CBG2 CBD4", 16},         // 143
      {"sg D5 W12/8 CBD2", 16},              // 144
      {"sg D2 W12 CBD8", 16},                // 145
      {"sg D5 nocap CBD8", 16},              // 146
      {"sg MUTANT shuffle CBD4", 16},        // 147
      {"(unused)", 16},                      // 148
      {"(unused)", 16},                      // 149
      {"sg140 knock: no transpose", 16},     // 150
      {"sg140 knock: no x loads", 16},       // 151
      {"sg140 knock: no mid wait", 16},      // 152
      {"sg140 knock: no transp+decode", 16}, // 153
      {"(unused)", 16},                      // 154
      {"(unused)", 16},                      // 155
      {"(unused)", 16},                      // 156
      {"(unused)", 16},                      // 157
      {"(unused)", 16},                      // 158
      {"(unused)", 16},                      // 159
      {"sg D5 W12/8 bodies 1,2,4,R", 16},    // 160
      {"sg D2 W12 bodies 1,2,4,R", 16},      // 161
      {"sg D5 nocap bodies 1,2,4,R", 16},    // 162
      {"sg MUTANT swap tiles, bodies 1,2,4,R", 16},  // 163
  };
  if (v < 0 || v >= (int)(sizeof(kV) / sizeof(kV[0]))) {
    std::fprintf(stderr, "unknown variant %d\n", v);
    std::exit(2);
  }
  return kV[v];
}

// Probe: MoeFusedKernel's body (C, PF as variant 0) with wall_clock64 stamps per block: [start, ready, end].
// ready: gate/up items after the x stage, down items after the wait and the mid. Timing only.
unsigned long long* g_vm_stamps = nullptr;
template <int R, int KR, int PF, int CACHE, int C>
__global__ void __launch_bounds__(256) MoeFusedProbeKernel(const LinJob* __restrict__ gu,
                                                           const GemvItem* __restrict__ gu_items, int n_gu,
                                                           const LinJob* __restrict__ dn,
                                                           const GemvItem* __restrict__ dn_items, int n_dn,
                                                           unsigned* __restrict__ done,
                                                           unsigned long long* __restrict__ st) {
  extern __shared__ uint32_t xs[];
  const unsigned long long t0 = wall_clock64();
  unsigned long long* my = st + 3 * blockIdx.x;
  if ((int)blockIdx.x < n_gu) {
    const GemvItem item = gu_items[blockIdx.x];
    const LinJob& job = gu[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    if (K == 0) {
      if (threadIdx.x == 0) my[0] = my[1] = my[2] = 0;
      return;
    }
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    unsigned long long t1 = 0;
    MoeAnyBody<R, KR, KR + 2, PF, C, CACHE>(K, rows, job, item, xs, [&] {
      MoeFillX(job, item, rows, xs);
      t1 = wall_clock64();
    });
    if (threadIdx.x == 0) {
      __hip_atomic_fetch_add(done + (item.job >> 1), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      my[0] = t0, my[1] = t1, my[2] = wall_clock64();
    }
    return;
  }
  const GemvItem item = dn_items[blockIdx.x - n_gu];
  const int u = item.job;
  const LinJob& d = dn[u];
  const int K = __builtin_amdgcn_readfirstlane(d.K);
  if (K == 0) {
    if (threadIdx.x == 0) my[0] = my[1] = my[2] = 0;
    return;
  }
  const int rows = __builtin_amdgcn_readfirstlane(d.R);
  unsigned long long t1 = 0;
  auto prep = [&] {
    if (threadIdx.x == 0)
      while (__hip_atomic_load(done + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kMoeGuItems<C>)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    __half* xh = reinterpret_cast<__half*>(xs);
    const int lane = threadIdx.x & 31;
    for (int t = threadIdx.x >> 5; t < rows * 5; t += 8) {
      const int r = t / 5, blk = t % 5;
      float gv[4], uv[4];
      LinOutBlock(gu[u * 2], r, blk, gv);
      LinOutBlock(gu[u * 2 + 1], r, blk, uv);
      const int c0 = blk * 128 + lane * 4;
      float sv[4], v[4];
      __half sh[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) sv[i] = Silu(gv[i]) * uv[i];
      LinScaleLane(d.suh, c0, sh);
      LinInLane(sv, sh, v);  // RoundH(RoundH(sv) * suh), Wht128: MoeFusedKernel's mid
#pragma unroll
      for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
    }
    __syncthreads();
    t1 = wall_clock64();
  };
  MoeAnyBody<R, KR, KR + 2, PF, C, CACHE>(K, rows, d, item, xs, prep);
  if (threadIdx.x == 0) my[0] = t0, my[1] = t1, my[2] = wall_clock64();
}

// MoeFusedKernel with a dispatch order table: order[b] >= 0 runs gate/up item order[b], < 0 runs down item
// -1 - order[b]. A down item must come after all gate/up items of its job (they are then resident or done).
// The item bodies are MoeFusedKernel's (same arithmetic). PROBE adds the stamps of MoeFusedProbeKernel.
template <int R, int KR, int PF, int CACHE, int C, bool PROBE>
__global__ void __launch_bounds__(256) MoeFusedMixKernel(const LinJob* __restrict__ gu,
                                                         const GemvItem* __restrict__ gu_items,
                                                         const LinJob* __restrict__ dn,
                                                         const GemvItem* __restrict__ dn_items,
                                                         unsigned* __restrict__ done, const int* __restrict__ order,
                                                         unsigned long long* __restrict__ st) {
  extern __shared__ uint32_t xs[];
  unsigned long long t0 = 0, t1 = 0;
  if constexpr (PROBE) t0 = wall_clock64();
  const int o = order[blockIdx.x];
  if (o >= 0) {
    const GemvItem item = gu_items[o];
    const LinJob& job = gu[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    if (K == 0) {
      if (PROBE && threadIdx.x == 0) st[3 * o] = st[3 * o + 1] = st[3 * o + 2] = 0;
      return;
    }
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    MoeAnyBody<R, KR, KR + 2, PF, C, CACHE>(K, rows, job, item, xs, [&] {
      MoeFillX(job, item, rows, xs);
      if constexpr (PROBE) t1 = wall_clock64();
    });
    if (threadIdx.x == 0) {
      __hip_atomic_fetch_add(done + (item.job >> 1), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      if constexpr (PROBE) st[3 * o] = t0, st[3 * o + 1] = t1, st[3 * o + 2] = wall_clock64();
    }
    return;
  }
  const int di = -1 - o;
  const GemvItem item = dn_items[di];
  const int u = item.job;
  const LinJob& d = dn[u];
  const int K = __builtin_amdgcn_readfirstlane(d.K);
  if (K == 0) return;
  const int rows = __builtin_amdgcn_readfirstlane(d.R);
  auto prep = [&] {
    if (threadIdx.x == 0)
      while (__hip_atomic_load(done + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kMoeGuItems<C>)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    __half* xh = reinterpret_cast<__half*>(xs);
    const int lane = threadIdx.x & 31;
    for (int t = threadIdx.x >> 5; t < rows * 5; t += 8) {
      const int r = t / 5, blk = t % 5;
      float gv[4], uv[4];
      LinOutBlock(gu[u * 2], r, blk, gv);
      LinOutBlock(gu[u * 2 + 1], r, blk, uv);
      const int c0 = blk * 128 + lane * 4;
      float sv[4], v[4];
      __half sh[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) sv[i] = Silu(gv[i]) * uv[i];
      LinScaleLane(d.suh, c0, sh);
      LinInLane(sv, sh, v);  // RoundH(RoundH(sv) * suh), Wht128: MoeFusedKernel's mid
#pragma unroll
      for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
    }
    __syncthreads();
    if constexpr (PROBE) t1 = wall_clock64();
  };
  MoeAnyBody<R, KR, KR + 2, PF, C, CACHE>(K, rows, d, item, xs, prep);
  if (PROBE && threadIdx.x == 0) {
    // Down stamps follow the gate/up ones; st[3 * 4095] holds the gate/up item count (host).
    const int base = (int)st[3 * 4095];
    st[3 * (base + di)] = t0, st[3 * (base + di) + 1] = t1, st[3 * (base + di) + 2] = wall_clock64();
  }
}

// Dispatch orders (host): lag D interleaves the down items of job j after the gate/up items of job j + D,
// in the item order of Build (shared job first at R > 1). D < 0: the engine order (all gate/up first).
template <int C = kMoeCols>
std::vector<int> VerifyMoeOrder(const Group& gu, const Group& dn, int D) {
  const int gper = kMoeGuItems<C>, dper = 160 / kMoeItemBlocks<C>;
  const int jobs = gu.nitems / gper;
  std::vector<int> order;
  if (D < 0) {
    for (int i = 0; i < gu.nitems; ++i) order.push_back(i);
    for (int i = 0; i < dn.nitems; ++i) order.push_back(-1 - i);
    return order;
  }
  int next_dn = 0;
  for (int j = 0; j < jobs; ++j) {
    for (int k = 0; k < gper; ++k) order.push_back(j * gper + k);
    if (j >= D)
      for (int k = 0; k < dper; ++k) order.push_back(-1 - (next_dn++));
  }
  while (next_dn < dn.nitems) order.push_back(-1 - (next_dn++));
  return order;
}

// Gate/up dispatch permutations (dn items after all gu items, engine order). The shared job's 6 items stay first.
// kind 1: the routed items in a fixed random order; 2: item-major (item k of every expert, then k + 1);
// 3: gate and up items interleaved per expert (g0 u0 g1 u1 g2 u2).
inline std::vector<int> VerifyMoeGuPerm(const Group& gu, const Group& dn, int kind) {
  const int gper = kMoeGuItems<2>, half = gper / 2, ne = gu.nitems / gper;
  std::vector<int> order;
  for (int k = 0; k < gper; ++k) order.push_back(k);
  std::vector<int> rest;
  if (kind == 1) {
    for (int i = gper; i < gu.nitems; ++i) rest.push_back(i);
    std::mt19937 rng(12345);
    std::shuffle(rest.begin(), rest.end(), rng);
  } else if (kind == 2) {
    for (int k = 0; k < gper; ++k)
      for (int e = 1; e < ne; ++e) rest.push_back(e * gper + k);
  } else {
    for (int e = 1; e < ne; ++e)
      for (int k = 0; k < half; ++k) rest.push_back(e * gper + k), rest.push_back(e * gper + half + k);
  }
  order.insert(order.end(), rest.begin(), rest.end());
  for (int i = 0; i < dn.nitems; ++i) order.push_back(-1 - i);
  return order;
}

int* g_vm_order[17][16] = {};  // [R][variant slot]: device order tables (item counts depend on R only)

// MoeFillX with 16-byte loads, NB per thread issued before any LDS store: one memory round trip for up to
// 256 NB x uint4 (R rows x span tiles x 32 bytes). MoeFillX does one flat load, a wait and one LDS store per
// dword (20 serial round trips at R = 4). Same LDS contents.
template <int NB>
__device__ __forceinline__ void MoeFillX4(const LinJob& job, const GemvItem& item, int rows, uint32_t* xs) {
  const int A = job.I >> 4, span = A / job.S, a0 = item.s * span;
  const int w = span * 2, n = rows * w;  // uint4 per row, per job
  const auto rs = StreamRsrc(reinterpret_cast<const uint32_t*>(job.xh), (uint32_t)rows * A * 32);
  uint4* x4 = reinterpret_cast<uint4*>(xs);
  for (int i0 = threadIdx.x; i0 < n; i0 += NB * 256) {
    uint4 v[NB];
#pragma unroll
    for (int k = 0; k < NB; ++k) {
      const int i = i0 + k * 256;
      if (i < n) {
        const int r = i / w, o = i - r * w;
        const auto t = __builtin_amdgcn_raw_buffer_load_b128(rs, (r * A * 2 + a0 * 2 + o) * 16, 0, 0);
        v[k] = make_uint4(t[0], t[1], t[2], t[3]);
      }
    }
#pragma unroll
    for (int k = 0; k < NB; ++k) {
      const int i = i0 + k * 256;
      if (i < n) x4[i] = v[k];
    }
  }
  __syncthreads();
}

// MoeGemvCols with prep() after the ring prologue: the first PF tile loads of each column (they do not
// depend on x) are in flight during the x stage. Loop, arithmetic and stores are MoeGemvCols's.
template <int K, int R, int PF, int CC, int CACHE, typename Prep>
__device__ __forceinline__ void MoeGemvColsPre(const LinJob& job, int s, const int (&b)[CC], const uint32_t* xs,
                                               Prep prep) {
  const int lane = threadIdx.x & 31;
  const LaneWindow<K> win(lane);
  const int q = lane & 3;
  constexpr int T = TileDwords(K);
  const int A = __builtin_amdgcn_readfirstlane(job.I >> 4);
  const int span = __builtin_amdgcn_readfirstlane(A / job.S);
  const int a0 = s * span;
  const uint32_t* tiles = UniformPtr(job.tiles);
  __amdgpu_buffer_rsrc_t rs[CC];
#pragma unroll
  for (int c = 0; c < CC; ++c)
    rs[c] = StreamRsrc(tiles + ((size_t)b[c] * A + a0) * T, (uint32_t)span * T * 4);
  const int o0 = 4 * win.d0, o1 = 4 * ((win.d0 + 1) % T), o2 = 4 * ((win.d0 + 2) % T);
  auto load_tile = [&](int c, int tile, uint32_t* dst) {
    const int so = tile * T * 4;
    dst[0] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o0, so, CACHE);
    dst[1] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o1, so, CACHE);
    dst[2] = __builtin_amdgcn_raw_buffer_load_b32(rs[c], o2, so, CACHE);
  };
  uint32_t ring[PF][CC][3];
#pragma unroll
  for (int f = 0; f < PF; ++f) {
#pragma unroll
    for (int c = 0; c < CC; ++c) load_tile(c, f, ring[f][c]);
    __builtin_amdgcn_sched_barrier(0);
  }
  prep();
  float acc[CC][R][2];
#pragma unroll
  for (int c = 0; c < CC; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[c][r][0] = acc[c][r][1] = 0.f;
  auto step = [&](int a, int f) {
    __half2 P[CC][4];
#pragma unroll
    for (int c = 0; c < CC; ++c) {
      DecodeLane<K, kMulMode>(ring[f][c][0], ring[f][c][1], ring[f][c][2], win.shift, P[c]);
      load_tile(c, a + f + PF, ring[f][c]);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint32_t* xr = xs + (r * span + a + f) * 8;
      uint32_t xa = xr[q], xb = xr[4 + q];
      __half2 ha = *reinterpret_cast<__half2*>(&xa);
      __half2 hb = *reinterpret_cast<__half2*>(&xb);
#pragma unroll
      for (int c = 0; c < CC; ++c) {
        acc[c][r][0] = Dot2(P[c][0], ha, acc[c][r][0]);
        acc[c][r][0] = Dot2(P[c][1], hb, acc[c][r][0]);
        acc[c][r][1] = Dot2(P[c][2], ha, acc[c][r][1]);
        acc[c][r][1] = Dot2(P[c][3], hb, acc[c][r][1]);
      }
    }
  };
  if (span % PF == 0) {
    for (int a = 0; a < span; a += PF) {
#pragma unroll
      for (int f = 0; f < PF; ++f) step(a, f);
    }
  } else {
    for (int a = 0; a < span; a += PF) {
#pragma unroll
      for (int f = 0; f < PF; ++f) {
        if (a + f >= span) break;
        step(a, f);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < CC; ++c)
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float v = acc[c][r][h];
        v = v + __shfl_xor(v, 1);
        v = v + __shfl_xor(v, 2);
        acc[c][r][h] = v;
      }
    }
  if (q == 0) {
    float* out = job.partial + (size_t)s * R * job.O;
#pragma unroll
    for (int c = 0; c < CC; ++c) {
      const int col = 16 * b[c] + lane / 4;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        out[r * job.O + col] = acc[c][r][0];
        out[r * job.O + col + 8] = acc[c][r][1];
      }
    }
  }
}

// MoeGemvCore with PRE: prep() runs inside MoeGemvColsPre (after the prologue); idle waves run it alone.
template <int K, int R, int PF, int C, int CACHE, bool PRE, typename Prep>
__device__ __forceinline__ void MoeGemvCoreX(const LinJob& job, const GemvItem& item, uint32_t* xs, Prep prep) {
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  if constexpr (!PRE) {
    MoeGemvCore<K, R, PF, C, CACHE>(job, item, xs, prep);
  } else {
    if (wave < item.nb) {
      const int b = item.b0 + wave;
      if (C == 2 && wave + 8 < item.nb) MoeGemvColsPre<K, R, PF, 2, CACHE>(job, item.s, {b, b + 8}, xs, prep);
      else MoeGemvColsPre<K, R, PF, 1, CACHE>(job, item.s, {b}, xs, prep);
    } else {
      prep();
    }
    __syncthreads();
  }
}

template <int K, int N, int PF, int C, int CACHE, bool PRE, typename Prep>
__device__ __forceinline__ void MoeRowsBodyX(int rows, const LinJob& job, const GemvItem& item, uint32_t* xs,
                                             Prep prep) {
  if constexpr (N >= 1) {
    if (rows == N) MoeGemvCoreX<K, N, PF, C, CACHE, PRE>(job, item, xs, prep);
    else MoeRowsBodyX<K, N - 1, PF, C, CACHE, PRE>(rows, job, item, xs, prep);
  }
}

template <int R, int KR, int KS, int PF, int C, int CACHE, bool PRE, typename Prep>
__device__ __forceinline__ void MoeAnyBodyX(int K, int rows, const LinJob& job, const GemvItem& item,
                                            uint32_t* xs, Prep prep) {
  if (K == KR) MoeRowsBodyX<KR, R, PF, C, CACHE, PRE>(rows, job, item, xs, prep);
  else if (K == KS && rows == R) MoeGemvCoreX<KS, R, PF, C, CACHE, PRE>(job, item, xs, prep);
  else __builtin_trap();
}

// MoeFusedKernel with MoeFillX4 (gate/up x stage) and optionally PRE (ring prologue before the x stage, and
// before the down items' wait). PROBE writes MoeFusedProbeKernel's stamps.
// Start stagger of the gate/up waves after the x stage (STAG 0: none). The steady-state rate drops from
// about 207 to 184 GB/s when all streams start in step (bench: FillX4 without stagger).
// STAG 1: wave w sleeps w units; 2: hash(block, wave) % 16 units; 3: as 2 with double units; 2 and 3 only
// for the first 160 blocks (later blocks start at spread times anyway). One unit = s_sleep(16), about 0.4 us.
template <int STAG>
__device__ __forceinline__ void MoeStagger() {
  if constexpr (STAG > 0) {
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    int n = 0;
    if constexpr (STAG == 1) n = wave;
    else if (blockIdx.x < 160) n = ((blockIdx.x * 8 + wave) * 2654435761u >> 28) & 15;
    for (int i = 0; i < n; ++i) {
      __builtin_amdgcn_s_sleep(16);
      if constexpr (STAG == 3) __builtin_amdgcn_s_sleep(16);
    }
  }
}

template <int R, int KR, int PF, int CACHE, int C, bool PRE, bool PROBE, bool FX4 = true, int STAG = 0>
__global__ void __launch_bounds__(256) MoeFusedXKernel(const LinJob* __restrict__ gu,
                                                       const GemvItem* __restrict__ gu_items, int n_gu,
                                                       const LinJob* __restrict__ dn,
                                                       const GemvItem* __restrict__ dn_items, int n_dn,
                                                       unsigned* __restrict__ done,
                                                       unsigned long long* __restrict__ st) {
  extern __shared__ uint32_t xs[];
  constexpr int NB = (R * 320 + 255) / 256;  // one round trip for R rows of a 160-tile span
  unsigned long long t0 = 0, t1 = 0;
  if constexpr (PROBE) t0 = wall_clock64();
  unsigned long long* my = st + 3 * blockIdx.x;
  if ((int)blockIdx.x < n_gu) {
    const GemvItem item = gu_items[blockIdx.x];
    const LinJob& job = gu[item.job];
    const int K = __builtin_amdgcn_readfirstlane(job.K);
    if (K == 0) {
      if (PROBE && threadIdx.x == 0) my[0] = my[1] = my[2] = 0;
      return;
    }
    const int rows = __builtin_amdgcn_readfirstlane(job.R);
    MoeAnyBodyX<R, KR, KR + 2, PF, C, CACHE, PRE>(K, rows, job, item, xs, [&] {
      if constexpr (FX4) MoeFillX4<NB>(job, item, rows, xs);
      else MoeFillX(job, item, rows, xs);
      if constexpr (PROBE) t1 = wall_clock64();
      MoeStagger<STAG>();
    });
    if (threadIdx.x == 0) {
      __hip_atomic_fetch_add(done + (item.job >> 1), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      if constexpr (PROBE) my[0] = t0, my[1] = t1, my[2] = wall_clock64();
    }
    return;
  }
  const GemvItem item = dn_items[blockIdx.x - n_gu];
  const int u = item.job;
  const LinJob& d = dn[u];
  const int K = __builtin_amdgcn_readfirstlane(d.K);
  if (K == 0) {
    if (PROBE && threadIdx.x == 0) my[0] = my[1] = my[2] = 0;
    return;
  }
  const int rows = __builtin_amdgcn_readfirstlane(d.R);
  auto prep = [&] {
    if (threadIdx.x == 0)
      while (__hip_atomic_load(done + u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)kMoeGuItems<C>)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    __half* xh = reinterpret_cast<__half*>(xs);
    const int lane = threadIdx.x & 31;
    for (int t = threadIdx.x >> 5; t < rows * 5; t += 8) {
      const int r = t / 5, blk = t % 5;
      float gv[4], uv[4];
      LinOutBlock(gu[u * 2], r, blk, gv);
      LinOutBlock(gu[u * 2 + 1], r, blk, uv);
      const int c0 = blk * 128 + lane * 4;
      float sv[4], v[4];
      __half sh[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) sv[i] = Silu(gv[i]) * uv[i];
      LinScaleLane(d.suh, c0, sh);
      LinInLane(sv, sh, v);  // RoundH(RoundH(sv) * suh), Wht128: MoeFusedKernel's mid
#pragma unroll
      for (int i = 0; i < 4; ++i) xh[r * kMoeInter + c0 + i] = __float2half(v[i] * kHadNorm);
    }
    __syncthreads();
    if constexpr (PROBE) t1 = wall_clock64();
  };
  MoeAnyBodyX<R, KR, KR + 2, PF, C, CACHE, PRE>(K, rows, d, item, xs, prep);
  if (PROBE && threadIdx.x == 0) my[0] = t0, my[1] = t1, my[2] = wall_clock64();
}

template <int R, bool PRE, bool PROBE, bool FX4 = true, int STAG = 0>
void VerifyMoeLaunchX(const Group& gu, const Group& dn, unsigned* done) {
  hipLaunchKernelGGL((MoeFusedXKernel<R, 4, 4, 4, 2, PRE, PROBE, FX4, STAG>), dim3(gu.nitems + dn.nitems), dim3(256),
                     MoeFusedLds(R, 0, gu.lds, dn.lds), 0, gu.d_jobs, gu.d_items, gu.nitems, dn.d_jobs, dn.d_items,
                     dn.nitems, done, g_vm_stamps);
}

template <int R, int PF, int C, int XC>
void VerifyMoeLaunchT(const Group& gu, const Group& dn, unsigned* done) {
  const size_t lds = MoeFusedLds(R, XC, gu.lds, dn.lds);
  hipLaunchKernelGGL((MoeFusedKernel<R, 4, PF, 4, C, XC>), dim3(gu.nitems + dn.nitems), dim3(256), lds, 0, gu.d_jobs,
                     gu.d_items, gu.nitems, dn.d_jobs, dn.d_items, dn.nitems, done);
}

template <int R>
void VerifyMoeLaunchR(int v, const Group& gu, const Group& dn, unsigned* done) {
#ifdef VM_LEAN
  // Lean build (moe-d1): the engine launch and v26 only; the other old variants take most of the compile time.
  if (v == 0) { VerifyMoeLaunchT<R, 4, 2, 0>(gu, dn, done); return; }
  if (v == 26) {
    int*& ord = g_vm_order[R][6];
    if (!ord) ord = Upload(VerifyMoeOrder<1>(gu, dn, 4));
    hipLaunchKernelGGL((MoeFusedMixKernel<R, 4, 8, 4, 1, false>), dim3(gu.nitems + dn.nitems), dim3(256),
                       MoeFusedLds(R, 0, gu.lds, dn.lds), 0, gu.d_jobs, gu.d_items, dn.d_jobs, dn.d_items, done, ord,
                       g_vm_stamps);
    return;
  }
  std::abort();
#else
  switch (v) {
    case 0: VerifyMoeLaunchT<R, 4, 2, 0>(gu, dn, done); break;
    case 1: VerifyMoeLaunchT<R, 4, 1, 0>(gu, dn, done); break;
    case 2: VerifyMoeLaunchT<R, 8, 1, 0>(gu, dn, done); break;
    case 3: VerifyMoeLaunchT<R, 2, 2, 0>(gu, dn, done); break;
    case 4: VerifyMoeLaunchT<R, 8, 2, 0>(gu, dn, done); break;
    case 5: VerifyMoeLaunchT<R, 4, 2, kMoeChunk>(gu, dn, done); break;
    case 7: {  // split: gate/up launch, then down launch with its own (small) LDS
      hipLaunchKernelGGL((MoeFusedKernel<R, 4, 4, 4, 2>), dim3(gu.nitems), dim3(256), gu.lds, 0, gu.d_jobs,
                         gu.d_items, gu.nitems, dn.d_jobs, dn.d_items, 0, done);
      hipLaunchKernelGGL((MoeFusedKernel<R, 4, 4, 4, 2>), dim3(dn.nitems), dim3(256), dn.lds, 0, gu.d_jobs,
                         gu.d_items, 0, dn.d_jobs, dn.d_items, dn.nitems, done);
      break;
    }
    case 8: case 9: case 10: case 11: case 12: {  // dispatch order: engine order (8, checks the mix kernel), lag 2, 4, 8, 16
      const int slot = v - 8, lag = slot == 0 ? -1 : 1 << slot;
      int*& ord = g_vm_order[R][slot];
      if (!ord) ord = Upload(VerifyMoeOrder(gu, dn, lag));
      hipLaunchKernelGGL((MoeFusedMixKernel<R, 4, 4, 4, 2, false>), dim3(gu.nitems + dn.nitems), dim3(256),
                         MoeFusedLds(R, 0, gu.lds, dn.lds), 0, gu.d_jobs, gu.d_items, dn.d_jobs, dn.d_items, done,
                         ord, g_vm_stamps);
      break;
    }
    case 24: case 25: case 26: case 27: {  // mix, C1 items: PF8 engine order, PF8 lag 2, PF8 lag 4, PF4 lag 2
      const int lag = v == 24 ? -1 : v == 26 ? 4 : 2;
      int*& ord = g_vm_order[R][4 + (v - 24)];
      if (!ord) ord = Upload(VerifyMoeOrder<1>(gu, dn, lag));
      const auto k = v == 27 ? MoeFusedMixKernel<R, 4, 4, 4, 1, false> : MoeFusedMixKernel<R, 4, 8, 4, 1, false>;
      hipLaunchKernelGGL(k, dim3(gu.nitems + dn.nitems), dim3(256), MoeFusedLds(R, 0, gu.lds, dn.lds), 0, gu.d_jobs,
                         gu.d_items, dn.d_jobs, dn.d_items, done, ord, g_vm_stamps);
      break;
    }
    case 28: case 29: case 30: {
      int*& ord = g_vm_order[R][8 + (v - 28)];
      if (!ord) ord = Upload(VerifyMoeGuPerm(gu, dn, v - 27));
      hipLaunchKernelGGL((MoeFusedMixKernel<R, 4, 4, 4, 2, false>), dim3(gu.nitems + dn.nitems), dim3(256),
                         MoeFusedLds(R, 0, gu.lds, dn.lds), 0, gu.d_jobs, gu.d_items, dn.d_jobs, dn.d_items, done,
                         ord, g_vm_stamps);
      break;
    }
    case 13: VerifyMoeLaunchX<R, false, false>(gu, dn, done); break;
    case 14: VerifyMoeLaunchX<R, true, false>(gu, dn, done); break;
    case 15: VerifyMoeLaunchX<R, false, true>(gu, dn, done); break;
    case 16: VerifyMoeLaunchX<R, true, true>(gu, dn, done); break;
    case 17: VerifyMoeLaunchX<R, false, false, true, 1>(gu, dn, done); break;
    case 18: VerifyMoeLaunchX<R, false, false, true, 2>(gu, dn, done); break;
    case 19: VerifyMoeLaunchX<R, false, false, true, 3>(gu, dn, done); break;
    case 20: VerifyMoeLaunchX<R, false, false, false, 1>(gu, dn, done); break;
    case 21: VerifyMoeLaunchX<R, false, false, false, 2>(gu, dn, done); break;
    case 22: VerifyMoeLaunchX<R, false, true, true, 2>(gu, dn, done); break;
    case 23: VerifyMoeLaunchX<R, false, true, false, 1>(gu, dn, done); break;
    case 6:
      hipLaunchKernelGGL((MoeFusedProbeKernel<R, 4, 4, 4, 2>), dim3(gu.nitems + dn.nitems), dim3(256),
                         MoeFusedLds(R, 0, gu.lds, dn.lds), 0, gu.d_jobs, gu.d_items, gu.nitems, dn.d_jobs,
                         dn.d_items, dn.nitems, done, g_vm_stamps);
      break;
    default: std::abort();
  }
#endif
}

inline void LaunchVerifyMoe(int v, int R, const Group& gu, const Group& dn, unsigned* done) {
  switch (R) {
    case 1: VerifyMoeLaunchR<1>(v, gu, dn, done); break;
    case 2: VerifyMoeLaunchR<2>(v, gu, dn, done); break;
    case 3: VerifyMoeLaunchR<3>(v, gu, dn, done); break;
    case 4: VerifyMoeLaunchR<4>(v, gu, dn, done); break;
    case 5: VerifyMoeLaunchR<5>(v, gu, dn, done); break;
    case 6: VerifyMoeLaunchR<6>(v, gu, dn, done); break;
    case 7: VerifyMoeLaunchR<7>(v, gu, dn, done); break;
    default: VerifyMoeLaunchR<8>(v, gu, dn, done); break;
  }
}

}  // namespace crucible
