// prefill-moe (P1 lever): MoE GEMM items of 192 rows (12 row tiles) x 128
// columns per side. The block decodes each weight tile of a k step once (one tile per wave) into LDS, and
// every wave multiplies 3 row tiles by 4 of the decoded column tiles. So one trellis decode serves 12 row
// tiles (the RM6 bodies: 6) and a wave loads 3 A fragments for 12 WMMAs (RM6: 6 for 12).
// Exactness: each output tile keeps the WMMA chain of the RM6 bodies (MoeWindowBody, PrefillWideBody): one
// wmma_f32_16x16x16_f16 per k step in the order 0 .. A-1, on the same packed A fragment and the same decoded
// B values (DecodeLane, GemmNativeStore, GemmStoredFragment), so the accumulators are bitwise equal. The
// gate/up kernel then runs the M1 epilogue arithmetic (PrefillMoeMidKernel's order); the down kernel stores
// fp16 pd as the RM6 HALF kernels do. The plan kernels are the same with tile_rows 192.
#pragma once
#include "prefill_moe_pair.h"
#include <type_traits>
#include <utility>

namespace crucible {
constexpr int kTileRm = 12;                         // row tiles per item: tile_rows 192
constexpr int kTileSlots = 3;                       // row group g (0..3) owns row tiles g, g + 4, g + 8
constexpr int kTileCols = 4;                        // column group cg (0, 1) owns column tiles 4 cg .. 4 cg + 3
constexpr int kTileBufWords = 8 * kGemmWaveWords;   // the 8 decoded tiles of one side and k step
// Gate/up: 4 X transform groups of 128 threads (2 tiles of 16 x 136 halves each) or the epilogue stage
// (32 rows of gate | up | pad) or 2 x 2 sides of decoded tiles; they never live at the same time.
// A of one side and k step in LDS: two planes, plane h holds halves 8 h .. 8 h + 7 of the 192 rows at a 4-word
// row stride, and plane 1 starts 32 words (half the banks) after a whole plane. So the 16 rows that one
// ds_load_b128 reads and the 8 rows x 4 quarters that one wave's b64 stores write fall on distinct banks. The
// 12-word row stride of the B tiles gave 9% LDS bank conflicts here (counters, Oct 24).
constexpr int kTileAPlane = 16 * kTileRm * 4 + 32;
constexpr int kTileAWords = 2 * kTileAPlane;
// The k loop of S sides: 2 decode buffers and 2 A buffers per side.
constexpr int TileLoopWords(int S) { return 2 * S * (kTileBufWords + kTileAWords); }
constexpr int kTilePairWords = TileLoopWords(2);
static_assert(kTilePairWords >= 32 * kPairPad && kTilePairWords >= 4 * (2 * 16 * 136 / 2));
// One barrier pattern of PrefillFxTransform: one per 128-column block, then two around the ready count.
constexpr int kFxBarriers = kHidden / 128 + 2;

// Row group of wave w (0..8 S - 1). Each SIMD gets one wave of every row group whether the waves of a block go
// to the SIMDs round-robin (w % 4) or in runs of 4: a 10-tile item gives groups 0 and 1 three live tiles and
// groups 2 and 3 two, so g = w & 3 would load two SIMDs with 12 WMMAs per step and two with 8.
__device__ __forceinline__ int TileGroup(int w) { return (w + (w >> 2)) & 3; }

// The k step that step a reads. MUT 3 (bench mutant only) swaps steps 0 and 1: the same products in
// another accumulation order, which a byte compare must see.
template <int MUT> __device__ __forceinline__ int TileStep(int a) { return MUT == 3 && a < 2 ? 1 - a : a; }
// Bench-only diagnostics (moe-fused; wrong outputs, timing bounds): skip the trellis decode, the WMMAs, the global
// A loads, or the weight loads of the k loop.
constexpr int kTileMutNoDecode = 4, kTileMutNoWmma = 5, kTileMutNoA = 6, kTileMutNoW = 7, kTileMutNoSync = 8;
// Exact variants: the weight refill (9), the A loads (10) or both (11) three k steps ahead instead of two.
constexpr int kTileMutDeepW = 9, kTileMutDeepA = 10, kTileMutDeep = 11;
// One A fragment (12), one B fragment (13) or one of each (15) read from LDS per k step and reused; 14: no decode, no
// A loads, no weight loads (the WMMAs, the LDS fragment reads and the barriers only). Wrong outputs, timing bounds.
constexpr int kTileMutOneA = 12, kTileMutOneB = 13, kTileMutBare = 14, kTileMutOneAB = 15;
// TileBodyPipe of depth 2 (16) or 3 (17); 18 and 19: depth 1 and 2 with the fewer-VALU decode and the weight
// stream offset in the descriptor (TileBodyPipe F); 20: depth 1 through TileBodyPipe (control). All exact.
constexpr int kTileMutPipe2 = 16, kTileMutPipe3 = 17, kTileMutFast1 = 18, kTileMutFast2 = 19, kTileMutPipe1 = 20;
constexpr bool TileMutPipe(int m) { return m >= kTileMutPipe2 && m <= kTileMutPipe1; }
constexpr int TileMutDepth(int m) { return m == kTileMutPipe3 ? 3 : (m == kTileMutFast1 || m == kTileMutPipe1) ? 1 : 2; }
constexpr bool TileMutFast(int m) { return m == kTileMutFast1 || m == kTileMutFast2; }
template <int K, int N, int S, int D, bool F = false>
__device__ __forceinline__ void TileBodyPipe(const Exl3GemmJob& j, int m0, int b0, uint32_t* lds,
                                             GemmF8 (&acc)[3][4]);

// DecodeLane<K, kMulTrue16> with fewer VALU ops and the same bits: state 0 is a0 >> 16 (end 16 for every K), so
// its two multiplies read a0's high half by op_sel instead of a shift; and v_sad_hi_u8 adds the second byte sum
// into the high half (each sum is at most 1020, so 0x6400 + sum never carries), which is the permute's packing.
template <int K>
__device__ __forceinline__ void TileDecodeFast(uint32_t w0, uint32_t w1, uint32_t w2, int shift, __half2 P[4]) {
  uint32_t st[8];
  if constexpr (7 * K + 16 <= 64) {
    const uint32_t a0 = __builtin_amdgcn_alignbit(w0, w1, 32 - shift);
    const uint32_t a1 = __builtin_amdgcn_alignbit(w1, w2, 32 - shift);
    st[0] = a0;  // the high half is the state; read by op_sel below
#pragma unroll
    for (int j = 1; j < 8; ++j) {
      const int end = j * K + 16;
      st[j] = end <= 32 ? a0 >> (32 - end) : __builtin_amdgcn_alignbit(a0, a1, 64 - end);
    }
  } else {
    LaneStates<K>(w0, w1, w2, shift, st);
    st[0] = __builtin_amdgcn_alignbit(w0, w1, 32 - shift);
  }
  uint32_t p[8];
  {
    uint32_t hi;
    asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[1,1]" : "=v"(hi) : "v"(st[0]), "s"(0x83DC0000u));
    asm("v_mad_u32_u16 %0, %1.h, 0xd12d, %2" : "=v"(p[0]) : "v"(st[0]), "v"(hi));
  }
#pragma unroll
  for (int j = 1; j < 8; ++j) p[j] = Mul1Product<kMulTrue16>(st[j]);
  const __half2 inv = __halves2half2(__ushort_as_half(0x1eee), __ushort_as_half(0x1eee));
  const __half2 bias = __halves2half2(__ushort_as_half(0xc931), __ushort_as_half(0xc931));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t s0 = __builtin_amdgcn_sad_u8(p[2 * i], 0u, 0x64006400u);
    const uint32_t packed = __builtin_amdgcn_sad_hi_u8(p[2 * i + 1], 0u, s0);
    P[i] = __hfma2(*reinterpret_cast<const __half2*>(&packed), inv, bias);
  }
}

template <int I, int N, class F> __device__ __forceinline__ void TileFor(F&& f) {
  if constexpr (I < N) {
    f(std::integral_constant<int, I>{});
    TileFor<I + 1, N>(f);
  }
}

// K dispatch of one kernel set: the K values of the PrefillMoeGemmKernel set with the same KMAX code.
template <int KMAX, class F> __device__ __forceinline__ void TileDispatchK(int K, F&& f) {
  using std::integral_constant;
  if constexpr (KMAX == kPrefillMoeK6Low) {
    if (K == 2) f(integral_constant<int, 2>{});
    else if (K == 25) f(integral_constant<int, 25>{});
    else if (K == 3) f(integral_constant<int, 3>{});
    else if (K == 6) f(integral_constant<int, 6>{});
    else __builtin_trap();
  } else {
    if (K == 3) f(integral_constant<int, 3>{});
    else if (K == 4) f(integral_constant<int, 4>{});
    else if (K == 5) f(integral_constant<int, 5>{});
    else if constexpr (KMAX == 6) { if (K == 6) f(integral_constant<int, 6>{}); else __builtin_trap(); }
    else if constexpr (KMAX == kPrefillMoeK7Half) {
      if (K == 7) f(integral_constant<int, 7>{});
      else if (K == 35) f(integral_constant<int, 35>{});
      else __builtin_trap();
    }
    else if constexpr (KMAX == 7 || KMAX == 8) { if (K == KMAX) f(integral_constant<int, KMAX>{}); else __builtin_trap(); }
    else __builtin_trap();
  }
}

// The k loop of one wave. S sides (1: down, 2: gate/up), 8 S waves. Wave w decodes column tile b0 + (w & 7)
// of side w >> 3 for every k step into a double-buffered LDS slot (one barrier per k step) and multiplies its
// N live row tiles (g + 4 i, g = TileGroup(w)) by the 4 column tiles of group cg = (w >> 2) & 1. A wave with N = 0
// only decodes. j is the job of the wave's side; M, I and K are the same on both sides.
// A goes through LDS: the 256 threads of a side copy the side's 192 x 16 A halves of step a + 2 (3 b64 chunks
// each) into registers in step a and into the free A buffer in step a + 1, so each global A load has two k steps
// to land and is read once per block (the two column groups of a side read the same A). The fragments that the
// WMMAs read are the bytes of GemmAFragment (row r of the packed [I/16][M][16] layout; 0 past M).
template <int K, int N, int S, int MUT = 0>
__device__ __forceinline__ void TileBody(const Exl3GemmJob& j, int m0, int b0, uint32_t* lds,
                                         GemmF8 (&acc)[kTileSlots][kTileCols]) {
  constexpr int T = TileDwords(K), kBuf = S * kTileBufWords;
  const int lane = threadIdx.x & 31, w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int side = w >> 3, cg = (w >> 2) & 1, g = TileGroup(w), A = j.I / 16;
  const LaneWindow<K> win(lane);
  const auto rs = StreamRsrc(j.tiles + (size_t)(b0 + (w & 7)) * A * T, A * T * 4);
  const int o0 = win.d0 * 4, o1 = ((win.d0 + 1) % T) * 4, o2 = ((win.d0 + 2) % T) * 4;
  const uint32_t xbytes = (uint32_t)j.Xrows * j.I * 2;
  const auto xrs = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), xbytes);
  const int astep = j.Xrows * 32;
  // Chunk q (0..2) of this thread: 8 bytes (quarter c & 3) of row c >> 2, c = lt + 256 q.
  constexpr int kChunks = 3;
  static_assert(kChunks * 256 == 16 * kTileRm * 4);
  const int lt = threadIdx.x & 255;
  uint32_t xo[kChunks], so[kChunks];
#pragma unroll
  for (int q = 0; q < kChunks; ++q) {
    const int c = lt + 256 * q, row = m0 + (c >> 2);
    xo[q] = (row < j.M ? (uint32_t)row * 32 : xbytes) + (c & 3) * 8;
    so[q] = (uint32_t)(((c & 3) >> 1) * kTileAPlane + (c >> 2) * 4 + (c & 1) * 2);
  }
  uint32_t* mine = lds + (side * 8 + (w & 7)) * kGemmWaveWords;
  const uint32_t* bsrc = lds + (side * 8 + cg * kTileCols) * kGemmWaveWords;
  uint32_t* abase = lds + 2 * kBuf;  // A buffer P of side s: abase + (P * S + s) * kTileAWords
  using U2 = uint32_t __attribute__((ext_vector_type(2)));
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  constexpr bool kDeepW = MUT == kTileMutDeepW || MUT == kTileMutDeep;
  constexpr bool kDeepA = MUT == kTileMutDeepA || MUT == kTileMutDeep;
  constexpr bool kOneA = MUT == kTileMutOneA || MUT == kTileMutOneAB, kOneB = MUT == kTileMutOneB || MUT == kTileMutOneAB;
  constexpr bool kBare = MUT == kTileMutBare;
  U2 sts[kDeepA ? 2 : 1][kChunks];
  if constexpr (MUT == kTileMutNoA || kBare)
    for (int q = 0; q < kChunks; ++q) sts[0][q] = U2{0u, 0u};
  auto aload = [&](int a, U2 (&st)[kChunks]) {
    if constexpr (MUT == kTileMutNoA || kBare) return;
    const int at = TileStep<MUT>(a) * astep;
#pragma unroll
    for (int q = 0; q < kChunks; ++q) st[q] = __builtin_amdgcn_raw_buffer_load_b64(xrs, xo[q], at, 0);
  };
  auto astore = [&](int P, const U2 (&st)[kChunks]) {
    uint32_t* d = abase + (P * S + side) * kTileAWords;
#pragma unroll
    for (int q = 0; q < kChunks; ++q) *reinterpret_cast<U2*>(d + so[q]) = st[q];
  };
  // Row tile i of this wave from A buffer P: lanes l and l + 16 read row (g + 4 i) 16 + (l & 15), 32 bytes.
  const uint32_t* aread = abase + side * kTileAWords + ((g * 16) + (lane & 15)) * 4;
  auto afrag = [&](int P, int i) {
    GemmH16 af;
    auto* ap = reinterpret_cast<U4*>(&af);
    const uint32_t* sp = aread + P * S * kTileAWords + i * 64 * 4;
    ap[0] = *reinterpret_cast<const U4*>(sp);
    ap[1] = *reinterpret_cast<const U4*>(sp + kTileAPlane);
    return af;
  };
  auto load = [&](int a, uint32_t r[3]) {
    if constexpr (MUT == kTileMutNoW || kBare) { if (a > 0) return; }
    const int at = TileStep<MUT>(a) * T * 4;
    r[0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0 + at, 0, 0);
    r[1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1 + at, 0, 0);
    r[2] = K >= 5 ? __builtin_amdgcn_raw_buffer_load_b32(rs, o2 + at, 0, 0) : 0;
  };
  uint32_t raws[kDeepW ? 2 : 1][3];
  load(0, raws[0]);
  aload(0, sts[0]);
  {
    __half2 p[4];
    DecodeLane<K, kMulTrue16>(raws[0][0], raws[0][1], raws[0][2], win.shift, p);
    GemmNativeStore(p, mine, lane);
  }
  load(1, raws[0]);  // past the stream (A = 1) the loads return 0 and the decode is not used
  if constexpr (kDeepW) load(2, raws[kDeepW ? 1 : 0]);
  astore(0, sts[0]);
  aload(1, sts[0]);
  if constexpr (kDeepA) aload(2, sts[kDeepA ? 1 : 0]);
  GemmGroupSync();
  // One k step a, in a fixed order (sched_barrier keeps the compiler from moving the parts):
  // 1. A of step a + 1 (st, sent in step a - 1) into the free A buffer, then the A loads of step a + 2 into st;
  // 2. the decode of step a + 1 (raw) into the free decode buffer;
  // 3. the refill of step a + 2 into raw, which the decode has just freed;
  // 4. the WMMAs of step a (A and the decoded tiles of buffer P), then the block barrier.
  // Every part runs in every step, also the last ones: the past-the-stream loads return 0 and the extra copies
  // go to the buffers that no wave reads. A branch around them made the compiler wait for the refill before the
  // WMMAs. Trunk's order (refill before decode, then a loop-carried copy) waited for every load of the step at the
  // loop end: s_waitcnt vmcnt(0) before the WMMAs, 34% of the gate/up stall cycles in the thread trace.
  auto step = [&](auto pc, int a) {
    constexpr int P = decltype(pc)::value;
    auto& st = sts[kDeepA ? P : 0];
    auto& raw = raws[kDeepW ? P : 0];
    astore(1 - P, st);
    __builtin_amdgcn_sched_barrier(0);
    aload(a + (kDeepA ? 3 : 2), st);
    __builtin_amdgcn_sched_barrier(0);
    {
      __half2 p[4];
      if constexpr (MUT == kTileMutNoDecode || kBare) {
        const uint32_t v[4] = {raw[0], raw[1], raw[2], raw[0] ^ raw[1]};
        for (int z = 0; z < 4; ++z) p[z] = *reinterpret_cast<const __half2*>(&v[z]);
      } else {
        DecodeLane<K, kMulTrue16>(raw[0], raw[1], raw[2], win.shift, p);
      }
      GemmNativeStore(p, mine + (1 - P) * kBuf, lane);
    }
    __builtin_amdgcn_sched_barrier(0);
    load(a + (kDeepW ? 3 : 2), raw);
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (N > 0 && MUT != kTileMutNoWmma) {
      GemmH16 afr[N];
#pragma unroll
      for (int i = 0; i < N; ++i) afr[i] = kOneA && i > 0 ? afr[0] : afrag(P, i);
      const uint32_t* bp = bsrc + P * kBuf;
      GemmH16 bf0;
#pragma unroll
      for (int c = 0; c < kTileCols; ++c) {
        if (!kOneB || c == 0) bf0 = GemmStoredFragment(bp + c * kGemmWaveWords, lane);
        const GemmH16 bf = bf0;
#pragma unroll
        for (int i = 0; i < N; ++i) acc[i][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[i], bf, acc[i][c]);
      }
    }
    if constexpr (MUT != kTileMutNoSync) GemmGroupSync();
  };
  using P0 = std::integral_constant<int, 0>;
  using P1 = std::integral_constant<int, 1>;
  int a = 0;
  for (; a + 1 < A; a += 2) {
    step(P0{}, a);
    step(P1{}, a + 1);
  }
  if (a < A) step(P0{}, a);
}

// moe-fused: TileBody with a load pipeline of depth D (2 or 3). The A loads and the weight refill of data step j
// go to register set j % D and are issued D steps before their use (TileBody: one step), so a wave's waits at the
// top of a step are for loads that had D steps to land. The prologue issues the loads in the loop's own order
// (A then W, step by step), so the pending loads at the loop top are the steady-state ones and the compiler's
// vmcnt counts stay partial (TileBody's deep variants got vmcnt(0) at the loop top). Same LDS buffers, the same
// copies, decodes and WMMAs in the same order as TileBody: the bytes are TileBody's.
template <int K, int N, int S, int D, bool F>
__device__ __forceinline__ void TileBodyPipe(const Exl3GemmJob& j, int m0, int b0, uint32_t* lds,
                                             GemmF8 (&acc)[kTileSlots][kTileCols]) {
  static_assert(D >= 1 && D <= 3);
  constexpr int T = TileDwords(K), kBuf = S * kTileBufWords, kU = D == 3 ? 6 : 2;  // unroll: lcm(2, D)
  const int lane = threadIdx.x & 31, w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int side = w >> 3, cg = (w >> 2) & 1, g = TileGroup(w), A = j.I / 16;
  const LaneWindow<K> win(lane);
  const auto rs = StreamRsrc(j.tiles + (size_t)(b0 + (w & 7)) * A * T, A * T * 4);
  const int o0 = win.d0 * 4, o1 = ((win.d0 + 1) % T) * 4, o2 = ((win.d0 + 2) % T) * 4;
  const uint32_t xbytes = (uint32_t)j.Xrows * j.I * 2;
  const auto xrs = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x), xbytes);
  const int astep = j.Xrows * 32;
  constexpr int kChunks = 3;
  const int lt = threadIdx.x & 255;
  uint32_t xo[kChunks], so[kChunks];
#pragma unroll
  for (int q = 0; q < kChunks; ++q) {
    const int c = lt + 256 * q, row = m0 + (c >> 2);
    xo[q] = (row < j.M ? (uint32_t)row * 32 : xbytes) + (c & 3) * 8;
    so[q] = (uint32_t)(((c & 3) >> 1) * kTileAPlane + (c >> 2) * 4 + (c & 1) * 2);
  }
  uint32_t* mine = lds + (side * 8 + (w & 7)) * kGemmWaveWords;
  const uint32_t* bsrc = lds + (side * 8 + cg * kTileCols) * kGemmWaveWords;
  uint32_t* abase = lds + 2 * kBuf;
  using U2 = uint32_t __attribute__((ext_vector_type(2)));
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  U2 sts[D][kChunks];
  uint32_t raws[D][3];
  auto aload = [&](int a, U2 (&st)[kChunks]) {
    const int at = a * astep;
#pragma unroll
    for (int q = 0; q < kChunks; ++q) st[q] = __builtin_amdgcn_raw_buffer_load_b64(xrs, xo[q], at, 0);
  };
  auto astore = [&](int P, const U2 (&st)[kChunks]) {
    uint32_t* d = abase + (P * S + side) * kTileAWords;
#pragma unroll
    for (int q = 0; q < kChunks; ++q) *reinterpret_cast<U2*>(d + so[q]) = st[q];
  };
  const uint32_t* aread = abase + side * kTileAWords + ((g * 16) + (lane & 15)) * 4;
  auto afrag = [&](int P, int i) {
    GemmH16 af;
    auto* ap = reinterpret_cast<U4*>(&af);
    const uint32_t* sp = aread + P * S * kTileAWords + i * 64 * 4;
    ap[0] = *reinterpret_cast<const U4*>(sp);
    ap[1] = *reinterpret_cast<const U4*>(sp + kTileAPlane);
    return af;
  };
  const uint32_t* wbase = j.tiles + (size_t)(b0 + (w & 7)) * A * T;
  auto load = [&](int a, uint32_t (&r)[3]) {
    if constexpr (F) {
      // Step a's descriptor starts at the step and holds the rest of the stream (none past it), so a lane's
      // offset is in range exactly when o + a T 4 is in rs: the same loads and zeros as below, with the step
      // offset in scalar registers instead of three VALU adds per step.
      const int rest = a < A ? (A - a) * T * 4 : 0;
      const auto rsa = StreamRsrc(wbase + (size_t)a * T, (uint32_t)rest);
      r[0] = __builtin_amdgcn_raw_buffer_load_b32(rsa, o0, 0, 0);
      r[1] = __builtin_amdgcn_raw_buffer_load_b32(rsa, o1, 0, 0);
      r[2] = K >= 5 ? __builtin_amdgcn_raw_buffer_load_b32(rsa, o2, 0, 0) : 0;
    } else {
      const int at = a * T * 4;
      r[0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0 + at, 0, 0);
      r[1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1 + at, 0, 0);
      r[2] = K >= 5 ? __builtin_amdgcn_raw_buffer_load_b32(rs, o2 + at, 0, 0) : 0;
    }
  };
  auto decode = [&](const uint32_t (&raw)[3], uint32_t* dst) {
    __half2 p[4];
    if constexpr (F) TileDecodeFast<K>(raw[0], raw[1], raw[2], win.shift, p);
    else DecodeLane<K, kMulTrue16>(raw[0], raw[1], raw[2], win.shift, p);
    GemmNativeStore(p, dst, lane);
  };
  // Prologue: data steps 0 .. D in loop order; step 0 goes to buffer 0 at once. Past the stream the loads return 0.
  // The sched_barriers keep the issue order: without them the gate/up prologue interleaved the A loads of
  // steps 1 and 2, and the loop-top waits (merged with the prologue's state) became vmcnt(5)/(2) in every step.
  aload(0, sts[0]);
  load(0, raws[0]);
  TileFor<1, D>([&](auto ic) {
    constexpr int i = decltype(ic)::value;
    __builtin_amdgcn_sched_barrier(0);
    aload(i, sts[i]);
    __builtin_amdgcn_sched_barrier(0);
    load(i, raws[i]);
  });
  __builtin_amdgcn_sched_barrier(0);
  astore(0, sts[0]);
  decode(raws[0], mine);
  __builtin_amdgcn_sched_barrier(0);
  aload(D, sts[0]);
  __builtin_amdgcn_sched_barrier(0);
  load(D, raws[0]);
  __builtin_amdgcn_sched_barrier(0);
  GemmGroupSync();
  // Step a (buffer parity P = a % 2, set R = (a + 1) % D holds data step a + 1): store A(a + 1), load A(a + 1 + D)
  // into the same set, decode W(a + 1), refill W(a + 1 + D), then the WMMAs of step a and the barrier.
  auto step = [&](auto pc, auto rc, int a) {
    constexpr int P = decltype(pc)::value, R = decltype(rc)::value;
    astore(1 - P, sts[R]);
    __builtin_amdgcn_sched_barrier(0);
    aload(a + 1 + D, sts[R]);
    __builtin_amdgcn_sched_barrier(0);
    decode(raws[R], mine + (1 - P) * kBuf);
    __builtin_amdgcn_sched_barrier(0);
    load(a + 1 + D, raws[R]);
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (N > 0) {
      GemmH16 afr[N];
#pragma unroll
      for (int i = 0; i < N; ++i) afr[i] = afrag(P, i);
      const uint32_t* bp = bsrc + P * kBuf;
#pragma unroll
      for (int c = 0; c < kTileCols; ++c) {
        const GemmH16 bf = GemmStoredFragment(bp + c * kGemmWaveWords, lane);
#pragma unroll
        for (int i = 0; i < N; ++i) acc[i][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[i], bf, acc[i][c]);
      }
    }
    GemmGroupSync();
  };
  using std::integral_constant;
  // The first kU steps run before the loop (A >= kU in every shape). A spill reload just before the loop is then
  // older than a step's loads at the loop top. Without the peel, the reload was the newest load of the prologue,
  // and the loop's merged wait for it (vmcnt(3) before the weight refill) waited for the other set's loads too.
  TileFor<0, kU>([&](auto uc) {
    constexpr int u = decltype(uc)::value;
    step(integral_constant<int, u % 2>{}, integral_constant<int, (u + 1) % D>{}, u);
  });
  int a = kU;
  for (; a + kU <= A; a += kU)
    TileFor<0, kU>([&](auto uc) {
      constexpr int u = decltype(uc)::value;
      step(integral_constant<int, u % 2>{}, integral_constant<int, (u + 1) % D>{}, a + u);
    });
  // Tail (a % kU == 0 here): at most kU - 1 steps.
  TileFor<0, kU - 1>([&](auto uc) {
    constexpr int u = decltype(uc)::value;
    if (a + u < A) step(integral_constant<int, u % 2>{}, integral_constant<int, (u + 1) % D>{}, a + u);
  });
}

// Live row slots of row group g: row tiles g, g + 4, g + 8 below L.
__device__ __forceinline__ int TileLive(int L, int g) { return L > g ? min(kTileSlots, (L - g + 3) / 4) : 0; }

// ---- down projection (fp16 pd, the RM6 HALF kernels' store) ----

template <int K, int N, int MUT>
__device__ __forceinline__ void TileDownItem(const Exl3GemmJob& j, const Exl3GemmItem& it, uint32_t* lds) {
  GemmF8 acc[kTileSlots][kTileCols] = {};
  if constexpr (TileMutPipe(MUT))
    TileBodyPipe<K, N, 1, TileMutDepth(MUT), TileMutFast(MUT)>(j, it.m0, it.b0, lds, acc);
  else
    TileBody<K, N, 1, MUT>(j, it.m0, it.b0, lds, acc);
  const int lane = threadIdx.x & 31, w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int cg = (w >> 2) & 1, g = TileGroup(w);
  const auto ps = StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial), (uint32_t)j.M * j.O * 2);
#pragma unroll
  for (int i = 0; i < N; ++i)
#pragma unroll
    for (int c = 0; c < kTileCols; ++c) {
      const uint32_t po = ((uint32_t)(it.m0 + (g + 4 * i) * 16 + lane / 16) * j.O + (it.b0 + cg * kTileCols + c) * 16 +
                           (lane & 15)) * 2;
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const float v = acc[i][c][z];
        const __half h = __float2half(fabsf(v) > 65504.f ? copysignf(65504.f, v) : v);
        __builtin_amdgcn_raw_buffer_store_b16(__half_as_ushort(h), ps, po + (uint32_t)(2 * z) * j.O * 2, 0, 0);
      }
    }
}

// Down items of 192 rows x 128 columns (PrefillDownItems with tile_rows 192), 8 waves. MUT: bench only.
template <int KMAX, int MUT = 0>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8)))
PrefillMoeTileDownKernel(const Exl3GemmJob* jobs, const Exl3GemmItem* items, const int* count) {
  __shared__ __attribute__((aligned(16))) uint32_t lds[TileLoopWords(1)];
  const int n = PrefillUniform(*count);
  const int g = TileGroup(__builtin_amdgcn_readfirstlane(threadIdx.x >> 5));
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    const auto it = PrefillUniform(items[i]);
    const auto j = PrefillUniform(jobs[it.job]);
    const int live = TileLive((min(16 * kTileRm, j.M - it.m0) + 15) / 16, g);
    TileDispatchK<KMAX>(j.K, [&](auto kc) {
      constexpr int K = decltype(kc)::value;
      switch (live) {
        case 0: TileDownItem<K, 0, MUT>(j, it, lds); break;
        case 1: TileDownItem<K, 1, MUT>(j, it, lds); break;
        case 2: TileDownItem<K, 2, MUT>(j, it, lds); break;
        default: TileDownItem<K, 3, MUT>(j, it, lds); break;
      }
    });
  }
}

// ---- gate/up with the mid step (the M1 epilogue arithmetic) ----

// After the k loop: per round q, row tiles 2q and 2q + 1 (row groups 2 (q & 1) and 2 (q & 1) + 1, slot q / 2)
// go to the stage (gate 128 | up 128 | pad per row), then each of the 16 waves runs the mid step on 2 of the
// 32 rows, with PrefillMoePairEpilogue's operations in its order. MUT 1 skips the down suh product, MUT 2 swaps
// the gate and up slots (bench mutants only).
template <int N, int MUT>
__device__ __forceinline__ void TilePairEpilogue(const PrefillMoe& a, int e, int M, int m0, int b0,
                                                 const GemmF8 (&acc)[kTileSlots][kTileCols], float* stage) {
  const int lane = threadIdx.x & 31, w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int side = w >> 3, cg = (w >> 2) & 1, g = TileGroup(w), c = b0 * 16 + lane * 4;
  __half gs[4], us[4], ds[4];
  LinScaleLane(PrefillUniform(a.gu[e * 2].svh), c, gs);
  LinScaleLane(PrefillUniform(a.gu[e * 2 + 1].svh), c, us);
  LinScaleLane(PrefillUniform(a.down[e].suh), c, ds);
  __half* xd = PrefillUniform(a.down[e].xh);
  bool done = false;
  TileFor<0, kTileRm / 2>([&](auto qc) {
    constexpr int q = decltype(qc)::value, slot = q / 2;
    const int row0 = m0 + q * 32;
    if (done || row0 >= M) { done = true; return; }
    __syncthreads();  // earlier readers of the stage (and the last k step's B readers) are done
    if constexpr (slot < N) {
      if ((g >> 1) == (q & 1)) {
        const int hold = g & 1;
#pragma unroll
        for (int cc = 0; cc < kTileCols; ++cc)
#pragma unroll
          for (int z = 0; z < 8; ++z)
            stage[(hold * 16 + 2 * z + (lane >> 4)) * kPairPad + side * 128 + (cg * kTileCols + cc) * 16 + (lane & 15)] =
                acc[slot][cc][z];
      }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int lr = w * 2 + h, row = row0 + lr;
      if (row >= M) break;
      const float* gp = stage + lr * kPairPad + (MUT == 2 ? 128 : 0) + lane * 4;
      const float* up = stage + lr * kPairPad + (MUT == 2 ? 0 : 128) + lane * 4;
      float gv[4], uv[4];
      LinOutLane(gp, gs, gv);
      LinOutLane(up, us, uv);
      // PrefillMoeMidKernel's expression: RoundH(RoundH(Silu(g) * u) * suh).
      float s[4], v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) s[i] = Silu(gv[i]) * uv[i];
      LinInLane<MUT == 1>(s, ds, v);
      __half* dst = xd + PackedIndex(row, c, M);
#pragma unroll
      for (int i = 0; i < 4; ++i) dst[i] = __float2half(v[i] * kHadNorm);
    }
  });
}

template <int K, int N, int MUT>
__device__ __forceinline__ void TilePairItem(const PrefillMoe& a, const Exl3GemmJob& j, const Exl3GemmItem& it,
                                             uint32_t* lds) {
  GemmF8 acc[kTileSlots][kTileCols] = {};
  // The engine (MUT < 3): the depth-2 pipeline with the fewer-VALU decode, the bytes of TileBody (moe-fused bench,
  // 8,192 rows: gate/up 25.7 -> 24.6 ms per launch-layer with kTileFxLead). The down kernel keeps TileBody: the
  // same change gave it no gain.
  if constexpr (MUT < 3)
    TileBodyPipe<K, N, 2, 2, true>(j, it.m0, it.b0, lds, acc);
  else if constexpr (TileMutPipe(MUT))
    TileBodyPipe<K, N, 2, TileMutDepth(MUT), TileMutFast(MUT)>(j, it.m0, it.b0, lds, acc);
  else
    TileBody<K, N, 2, MUT>(j, it.m0, it.b0, lds, acc);
  TilePairEpilogue<N, MUT >= 3 ? 0 : MUT>(a, it.job >> 1, j.M, it.m0, it.b0, acc, reinterpret_cast<float*>(lds));
}

// An X slot past the queue end: the barriers of PrefillFxTransform, so every group of the block takes the same
// barrier count.
__device__ __forceinline__ void PrefillFxIdle() {
#pragma unroll 1
  for (int i = 0; i < kFxBarriers; ++i) __syncthreads();
}

// The PrefillMoePairKernel queue with 16 waves: a G claim g owns gate item i_gu[2g] (waves 0-7) and up item
// i_gu[2g + 1] (waves 8-15) of the same 192 rows x 128 columns; an X claim takes 4 X items (one per 128-thread
// group). waves_per_eu(8): at most 192 VGPRs, so 2 blocks of 16 waves fit a WGP.
template <int KMAX, int MUT = 0>
__global__ void __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(8)))
PrefillMoeTilePairKernel(PrefillMoe a, int lead) {
  constexpr int per = 10 / kMoeNb;  // pairs per unit
  __shared__ __attribute__((aligned(16))) uint32_t lds[kTilePairWords];
  __shared__ int claim;
  const int t = threadIdx.x, grp = __builtin_amdgcn_readfirstlane(t >> 7), side = __builtin_amdgcn_readfirstlane(t >> 8);
  const int g4 = TileGroup(__builtin_amdgcn_readfirstlane(t >> 5)), T = a.tile_rows;
  auto* tile = reinterpret_cast<__half (*)[16][136]>(lds) + 2 * grp;
  const int ng = PrefillUniform(a.item_counts[0]) / 2, nx = PrefillUniform(a.fx_count[2]);
  for (;;) {
    if (t == 0) claim = atomicAdd(a.fx_count + 1, 1);
    __syncthreads();
    const int g = PrefillUniform(claim);
    __syncthreads();
    Exl3GemmItem it{};
    Exl3GemmJob j{};
    int need = nx;
    if (g < ng) {
      it = PrefillUniform(a.i_gu[2 * g + side]);
      j = PrefillUniform(a.g_gu[it.job]);
      need = min(nx, a.fx_xoff[it.job >> 1] + 2 * min((j.M + 15) / 16, (it.m0 + T) / 16));
    }
    const int limit = min(nx, need + lead);
    for (;;) {
      if (t == 0)
        claim = __hip_atomic_load(a.fx_count, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < limit
                    ? atomicAdd(a.fx_count, 4) : nx;
      __syncthreads();
      const int x = PrefillUniform(claim);
      __syncthreads();
      if (x >= nx) break;
      if (x + grp < nx) PrefillFxTransform(a, x + grp, tile, t & 127);
      else PrefillFxIdle();
    }
    if (g >= ng) return;
    const int target = 2 * ((min(j.M, it.m0 + T) - it.m0 + 15) / 16);
    if (t == 0)
      while (__hip_atomic_load(a.fx_ready + g / per, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const int live = TileLive((min(T, j.M - it.m0) + 15) / 16, g4);
    TileDispatchK<KMAX>(j.K, [&](auto kc) {
      constexpr int K = decltype(kc)::value;
      switch (live) {
        case 0: TilePairItem<K, 0, MUT>(a, j, it, lds); break;
        case 1: TilePairItem<K, 1, MUT>(a, j, it, lds); break;
        case 2: TilePairItem<K, 2, MUT>(a, j, it, lds); break;
        default: TilePairItem<K, 3, MUT>(a, j, it, lds); break;
      }
    });
    __syncthreads();
  }
}

// X queue lead of the tile gate/up kernel, unless CRUCIBLE_PF_FX_LEAD is set (then its value, as for the other
// launches). 32 instead of kFxLead (128): gate/up 25.7 -> 25.1 ms at 8,192 rows and 46.7 -> 46.0 at 16,512 (bench,
// layer 24 routes); 0 and 16 are slower. The lead changes only the claim order, not the bytes.
constexpr int kTileFxLead = 32;
inline int PrefillTileFxLead() {
  static const int lead = crucible::SwitchText("CRUCIBLE_PF_FX_LEAD") ? PrefillFxLead() : kTileFxLead;
  return lead;
}

// The tile path: packed RM6 shapes with the fp16 pd (the engine's 4,096+ row launches), for every KMAX code of
// the RM6 kernels. tile_rows must be 16 kTileRm in the plan and both kernels.
constexpr int kTileRows = 16 * kTileRm;
inline bool PrefillMoeTileShape(PrefillGemmShape g, bool half_pd, int kmax) {
  return g.packed && g.rm == 6 && half_pd && PrefillMoeTileKmax(kmax);
}
// Grids: 2 blocks of 16 waves (gate/up) and 4 of 8 waves (down) per WGP; the blocks loop over the queues.
template <int MUT = 0> inline void LaunchPrefillMoeTilePair(int kmax, const PrefillMoe& a, int lead) {
  switch (kmax) {
    case 5: PrefillMoeTilePairKernel<5, MUT><<<512, 512>>>(a, lead); break;
    case 6: PrefillMoeTilePairKernel<6, MUT><<<512, 512>>>(a, lead); break;
    case 7: PrefillMoeTilePairKernel<7, MUT><<<512, 512>>>(a, lead); break;
    case 8: PrefillMoeTilePairKernel<8, MUT><<<512, 512>>>(a, lead); break;
    case kPrefillMoeK7Half: PrefillMoeTilePairKernel<kPrefillMoeK7Half, MUT><<<512, 512>>>(a, lead); break;
    case kPrefillMoeK6Low: PrefillMoeTilePairKernel<kPrefillMoeK6Low, MUT><<<512, 512>>>(a, lead); break;
    default: std::fprintf(stderr, "MoE tile path: no kernel set for KMAX %d\n", kmax); std::abort();
  }
}
template <int MUT = 0>
inline void LaunchPrefillMoeTileDown(int kmax, const Exl3GemmJob* jobs, const Exl3GemmItem* items, const int* count) {
  switch (kmax) {
    case 5: PrefillMoeTileDownKernel<5, MUT><<<1024, 256>>>(jobs, items, count); break;
    case 6: PrefillMoeTileDownKernel<6, MUT><<<1024, 256>>>(jobs, items, count); break;
    case 7: PrefillMoeTileDownKernel<7, MUT><<<1024, 256>>>(jobs, items, count); break;
    case 8: PrefillMoeTileDownKernel<8, MUT><<<1024, 256>>>(jobs, items, count); break;
    case kPrefillMoeK7Half: PrefillMoeTileDownKernel<kPrefillMoeK7Half, MUT><<<1024, 256>>>(jobs, items, count); break;
    case kPrefillMoeK6Low: PrefillMoeTileDownKernel<kPrefillMoeK6Low, MUT><<<1024, 256>>>(jobs, items, count); break;
    default: std::fprintf(stderr, "MoE tile path: no kernel set for KMAX %d\n", kmax); std::abort();
  }
}
}  // namespace crucible
