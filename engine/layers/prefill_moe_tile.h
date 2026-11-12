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

// The producer/consumer kernels:
// producer/consumer waves for the 192-row MoE tile kernels above.
// Gate/up block: 16 consumer waves (waves 0-15, the tile map of PrefillMoeTilePairKernel) and 4 producer waves
// (16-19); down block: 8 consumers and 2 producers. In k step a the consumers read LDS slot a % 2 (3 A and 4 B
// fragments, 12 WMMAs) and issue no global load and no decode. The producers write slot (a + 1) % 2 (the A rows and
// the decoded weight tiles of data step a + 1, loaded D steps before into producer registers), issue the loads of
// step a + 1 + D, and (gate/up, XD > 0) run the X transform of a later unit. One barrier per k step for all waves.
// Exactness: the bytes of TileBodyPipe<K, N, S, 2, true> (the engine's gate/up body) and of TileBody (down): the
// same A bytes in the same two-plane LDS layout (0 past M), the same decode (TileDecodeFast, the bits of DecodeLane)
// and GemmNativeStore, the same fragment reads, one WMMA per k step in the order 0 .. A-1 for every output tile, and
// the same epilogues. The X producer runs PrefillFxTransform's arithmetic, so xg and xu are the X items' bytes.

namespace crucible {
// Producer p serves side p >> 1, the column tiles 4 (p & 1) .. 4 (p & 1) + 3 of that side, and the 16-byte A
// chunks c = 32 (p & 1) + lane + 64 q (q < 6) of the side: row c >> 1, plane c & 1 (the planes of kTileAPlane).
constexpr int kPcTiles = 4;   // decoded tiles per producer and k step
constexpr int kPcChunks = 6;  // 16-byte A chunks per producer lane and k step
static_assert(kPcChunks * 64 == 16 * kTileRm * 2 && 2 * kPcTiles == 8);
constexpr int kPcPairWaves = 20, kPcDownWaves = 10;
// Defaults of the engine launch: producer register depth, producer priority in the k loop, X producer distance.
constexpr int kPcDepth = 2, kPcPrio = 1, kPcXDist = 10;
// On: tile launches with KMAX 5 or 6 run the fat gate/up kernel (kPcFat, kPcFatDown below). A bench measured it
// against the tile gate/up on real routes (layers 8, 24, 40 of a cold 32K prompt, 5 valid pairs each): MoE -4.0 to
// -4.3% at 8,320 rows and -2.2 to -3.1% at 16,512 rows, xg, xu, xd and pd byte-equal at every layer and shape, the
// mutants differ. False gives the tile kernels. A compile-time switch: no new CRUCIBLE_* name (SWITCH_CEILING).
constexpr bool kPrefillMoePc = true;
// The KMAX codes with kernels (the 3.05 and 4.05 sets); the others keep the tile kernels.
constexpr bool PrefillMoePcKmax(int kmax) { return kmax == 5 || kmax == 6; }
constexpr bool PrefillMoePcOn(int kmax) { return kPrefillMoePc && PrefillMoePcKmax(kmax); }

// The X producer of a gate/up claim (XD > 0, the D1 queue rule): G claim g writes the X values of the same
// claim slot c = g % 5 of unit g / 5 + XD. Producer p takes the 128-column block cb = c + 5 p of that unit, both
// sides, all L rows; pair l is row m0 + l. 2 pairs per k step: the input load in step s, the arithmetic and the
// stores in step s + D, so the wait for the input is not newer than the step's own weight and A loads (vmcnt is a
// FIFO count). The arithmetic is LinInHalfLane (exl3.h): PrefillFxTransform's values.
struct PcNoXProd {
  template <int Q> __device__ __forceinline__ void Step(int) {}
};
// threadIdx.x through an empty volatile asm. The queue loop's lane values (X claims, epilogue) then depend on a value
// that LICM cannot hoist out of the loop, so none of them stays live through the k loop next to the 96 accumulators.
// With plain threadIdx.x the compiler hoisted them to the kernel entry, and the consumer k loop spilled every
// accumulator to scratch (492 bytes per lane at the 144-VGPR budget; the k loop alone needs 140).
__device__ __forceinline__ int PcTid() {
  int t = threadIdx.x;
  asm volatile("" : "+v"(t));
  return t;
}
// The X items x (gate, which 0) and x + 1 (up, which 1) of one 16-row group in one pass (PrefillFxUnits writes the
// two sides of a group next to each other). Each x value is loaded once and goes through LinInHalfLane for both
// sides (PrefillFxTransform's arithmetic, Wht128Lane for Wht128Dpp), and each lane stores its 4 values of a row
// direct to the packed layout (PcXProd::Side's addresses), so the pass has no LDS stage and no barrier per column
// block. The same values at the same addresses as two PrefillFxTransform items, and the same fx_ready count (2).
// No branch per row (as PcXProd): a row past M loads from an empty buffer (0) and stores past the output range
// (dropped), so the waits count only the loads of the column block that the arithmetic reads.
// FOLD (the fat kernel): both sides in one LinInHalfPairLane (Wht128Fold2), the same bytes. The thin kernel keeps the
// two LinInHalfLane calls: with the fold its X loop spilled (420 bytes of scratch per lane at 144 VGPRs).
template <bool FOLD>
__device__ __forceinline__ void PcFxPairTransform(const PrefillMoe& a, int x, int t) {
  using U2 = uint32_t __attribute__((ext_vector_type(2)));
  const int code = PrefillUniform(a.fx_items[x]), e = code >> 16, rg = (code >> 1) & 0x7fff;
  const int M = PrefillUniform(a.gu[e * 2].R);
  if ((code & 1) || PrefillUniform(a.fx_items[x + 1]) != (code | 1) || PrefillUniform(a.gu[e * 2 + 1].R) != M)
    __builtin_trap();
  const int lane = t & 31, wave = __builtin_amdgcn_readfirstlane(t >> 5);
  const uint32_t out = (uint32_t)M * kHidden * 2;
  const auto rsg = StreamRsrc(reinterpret_cast<const uint32_t*>(PrefillUniform(a.gu[e * 2].suh)), kHidden * 2);
  const auto rsu = StreamRsrc(reinterpret_cast<const uint32_t*>(PrefillUniform(a.gu[e * 2 + 1].suh)), kHidden * 2);
  const auto rdg = StreamRsrc(reinterpret_cast<const uint32_t*>(PrefillUniform(a.gu[e * 2].xh)), out);
  const auto rdu = StreamRsrc(reinterpret_cast<const uint32_t*>(PrefillUniform(a.gu[e * 2 + 1].xh)), out);
  const bool shared = e == kExperts;
  const int off = shared ? 0 : a.dispatch.offsets[e];
  // One resource for the R rows of xh. Row i of the wave: the byte offset of its x row (xo) and of row l in a packed
  // output (ro); a row past M gets offsets past both ranges, so its loads give 0 and its stores are dropped.
  const auto rx =
      StreamRsrc(reinterpret_cast<const uint32_t*>(PrefillUniform(a.xh)), (uint32_t)a.dispatch.rows * kHidden * 2);
  uint32_t xo[4], ro[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int l = rg * 16 + wave * 4 + i;
    const bool live = l < M;
    const int r = PrefillUniform(live ? (shared ? l : a.dispatch.pairs[off + l] / kTopK) : 0);
    xo[i] = live ? (uint32_t)r * kHidden * 2 : 0x80000000u;
    ro[i] = live ? (uint32_t)l * 32 : 0x80000000u;
  }
  // PackedIndex(l, cb * 128 + lane * 4, M) * 2 = ro + cb * M * 256 + dlane (bytes).
  const uint32_t dlane = (uint32_t)(((lane >> 2) * M * 16 + (lane & 3) * 4) * 2), xlane = lane * 8;
  U2 xv[kFxDepth][4], gv[kFxDepth], uv[kFxDepth];
  auto load = [&](int slot, int cb) {
    gv[slot] = __builtin_amdgcn_raw_buffer_load_b64(rsg, cb * 256 + xlane, 0, 0);
    uv[slot] = __builtin_amdgcn_raw_buffer_load_b64(rsu, cb * 256 + xlane, 0, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) xv[slot][i] = __builtin_amdgcn_raw_buffer_load_b64(rx, xo[i] + cb * 256 + xlane, 0, 0);
  };
#pragma unroll
  for (int k = 0; k < kFxDepth; ++k) load(k, k);
  for (int cb0 = 0; cb0 < kHidden / 128; cb0 += kFxDepth) {
#pragma unroll
    for (int k = 0; k < kFxDepth; ++k) {
      const int cb = cb0 + k;
      const uint2 g2 = make_uint2(gv[k].x, gv[k].y), u2 = make_uint2(uv[k].x, uv[k].y);
      const uint32_t cbo = (uint32_t)cb * M * 256 + dlane;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint2 x2 = make_uint2(xv[k][i].x, xv[k][i].y);
        uint2 hg, hu;
        if constexpr (FOLD) {
          LinInHalfPairLane(x2, g2, u2, kHadNorm, lane, hg, hu);
        } else {
          hg = LinInHalfLane(x2, g2, kHadNorm, lane);
          hu = LinInHalfLane(x2, u2, kHadNorm, lane);
        }
        __builtin_amdgcn_raw_buffer_store_b64(U2{hg.x, hg.y}, rdg, ro[i] + cbo, 0, 0);
        __builtin_amdgcn_raw_buffer_store_b64(U2{hu.x, hu.y}, rdu, ro[i] + cbo, 0, 0);
      }
      if (cb + kFxDepth < kHidden / 128) load(k, cb + kFxDepth);
    }
  }
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();
  if (t == 0) atomicAdd(a.fx_ready + a.fx_uoff[e] + rg * 16 / a.tile_rows, 2);
  __syncthreads();  // PrefillFxTransform: keeps the count inside the X loop
}
// An X pair slot past the queue end: the barriers of PcFxPairTransform, so every group of the block takes the same
// barrier count.
__device__ __forceinline__ void PcFxPairIdle() {
  __syncthreads();
  __syncthreads();
}
// MUT 22 (bench mutant only): the X producer skips the kHadNorm scale of the up side, so xu differs.
constexpr int kPcMutXNorm = 22;
template <int D, int MUT = 0> struct PcXProd {
  using U2 = uint32_t __attribute__((ext_vector_type(2)));
  const __half* xh = nullptr;
  __half *dg = nullptr, *du = nullptr;
  const uint32_t* rows = nullptr;  // LDS: the source row of pair l
  uint2 sg{}, su{};                // the lane's 4 suh values of block cb, gate and up
  uint32_t dlane = 0;              // per lane: the byte offset of its 4 values in a packed row
  int L = 0, M = 1, m0 = 0, cb = 0;
  uint2 xv[D][2];
  __device__ __forceinline__ void Issue(uint2& v, int l) {
    const int lane = threadIdx.x & 31;
    const int src = __builtin_amdgcn_readfirstlane((int)rows[l]);
    const auto rx = StreamRsrc(reinterpret_cast<const uint32_t*>(xh + (size_t)src * kHidden + cb * 128), 256u);
    const U2 x2 = __builtin_amdgcn_raw_buffer_load_b64(rx, lane * 8, 0, 0);
    v = make_uint2(x2.x, x2.y);
  }
  __device__ __forceinline__ void Side(const uint2& x, const uint2& s, __half* d, int l, bool norm) const {
    const uint2 hv = LinInHalfLane(x, s, norm ? kHadNorm : 1.f, threadIdx.x & 31);
    const uint32_t dofs = (uint32_t)(((cb * 8) * M + m0 + l) * 32);
    const auto rd = StreamRsrc(reinterpret_cast<const uint32_t*>(d), (uint32_t)M * kHidden * 2);
    __builtin_amdgcn_raw_buffer_store_b64(U2{hv.x, hv.y}, rd, dofs + dlane, 0, 0);
  }
  // Step a: finish the pairs of step a - D (set a % D), then issue the pairs of step a into the same set.
  template <int Q> __device__ __forceinline__ void Step(int a) {
    if (a >= D) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int l = 2 * (a - D) + i;
        if (l < L) {
          Side(xv[Q][i], sg, dg, l, true);
          Side(xv[Q][i], su, du, l, MUT != kPcMutXNorm);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
      if (2 * a + i < L) Issue(xv[Q][i], 2 * a + i);
  }
};
// Bench diagnostics (wrong outputs, timing bounds): the producers only take the barriers
// (no loads, no decode, no A copy, no X; run with XD 0), or the consumers read one B fragment per k step and use it for
// all 4 column tiles.
constexpr int kPcMutBare = 30, kPcMutOneB = 31;
// Bench variant (exact): the fat consumer step with the 4 B fragments before A 0 (the order of PcConsume), so the first
// WMMA of a k step waits for all 10 loads.
constexpr int kPcMutFatCs = 32;
// Bench diagnostics of the fat kernels (wrong outputs): the consumers read their fragments once before the k loop and
// issue only the WMMAs and barriers in it (NoLds), and NoLds with bare producers (WmmaOnly: the WMMA, barrier, X claim
// and epilogue floor of the kernel structure).
constexpr int kPcMutNoLds = 33, kPcMutWmmaOnly = 34;
__host__ __device__ constexpr bool PcBareProducer(int mut) { return mut == kPcMutBare || mut == kPcMutWmmaOnly; }
__host__ __device__ constexpr bool PcNoLdsConsumer(int mut) { return mut == kPcMutNoLds || mut == kPcMutWmmaOnly; }
// Every pair finishes inside the gate/up k loop (D <= 3).
static_assert(kTileRm * 16 / 2 + 3 < kHidden / 16);
constexpr int kPcRowWords = 16 * kTileRm;  // the X producer's source row table

// The producer part of one item: p is the producer index (0 .. 2 S - 1), j the job of its side.
template <int K, int S, int D, int MUT, int PRIO, class XP>
__device__ __forceinline__ void PcProduce(const Exl3GemmJob& j, int m0, int b0, uint32_t* lds, int p, XP& xp) {
  static_assert(D >= 1 && D <= 3);
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  constexpr int T = TileDwords(K), kBuf = S * kTileBufWords, kU = D == 3 ? 6 : 2;  // unroll: lcm(2, D)
  const int lane = threadIdx.x & 31, side = p >> 1, sub = p & 1, A = j.I / 16;
  const LaneWindow<K> win(lane);
  const int o0 = win.d0 * 4, o1 = ((win.d0 + 1) % T) * 4, o2 = ((win.d0 + 2) % T) * 4;
  const uint32_t* wbase = j.tiles + (size_t)(b0 + kPcTiles * sub) * A * T;
  const int c0 = 32 * sub + lane;
  const uint32_t xoff = (uint32_t)((c0 >> 1) * 32 + (c0 & 1) * 16);
  uint32_t* adst = lds + 2 * kBuf + side * kTileAWords + (c0 & 1) * kTileAPlane + (c0 >> 1) * 4;
  uint32_t* bdst = lds + (side * 8 + kPcTiles * sub) * kGemmWaveWords;
  U4 av[D][kPcChunks];
  uint32_t wr[D][kPcTiles][3];
  // Data step a (MUT 3 swaps steps 0 and 1). The A descriptor starts at the step's k block and holds its M rows, so
  // a row past M reads 0 (the present kernel: an offset past the X buffer); past the stream both descriptors are
  // empty. The weight descriptor of a tile starts at the step and holds the rest of the stream (TileBodyPipe F).
  const uint32_t abytes = j.M > m0 ? (uint32_t)(j.M - m0) * 32 : 0u;
  auto load = [&](int a, U4 (&v)[kPcChunks], uint32_t (&r)[kPcTiles][3]) {
    const int s = TileStep<MUT>(a);
    const auto xr = StreamRsrc(reinterpret_cast<const uint32_t*>(j.x + ((size_t)s * j.Xrows + m0) * 16),
                               s < A ? abytes : 0u);
#pragma unroll
    for (int q = 0; q < kPcChunks; ++q) v[q] = __builtin_amdgcn_raw_buffer_load_b128(xr, xoff + 1024 * q, 0, 0);
    const uint32_t rest = s < A ? (uint32_t)(A - s) * T * 4 : 0u;
#pragma unroll
    for (int n = 0; n < kPcTiles; ++n) {
      const auto rs = StreamRsrc(wbase + ((size_t)n * A + s) * T, rest);
      r[n][0] = __builtin_amdgcn_raw_buffer_load_b32(rs, o0, 0, 0);
      r[n][1] = __builtin_amdgcn_raw_buffer_load_b32(rs, o1, 0, 0);
      r[n][2] = K >= 5 ? __builtin_amdgcn_raw_buffer_load_b32(rs, o2, 0, 0) : 0;
    }
  };
  // Data step -> LDS slot P: the A chunks, then the 4 decoded tiles.
  auto put = [&](int P, const U4 (&v)[kPcChunks], const uint32_t (&r)[kPcTiles][3]) {
    uint32_t* ad = adst + P * S * kTileAWords;
#pragma unroll
    for (int q = 0; q < kPcChunks; ++q) *reinterpret_cast<U4*>(ad + 128 * q) = v[q];
#pragma unroll
    for (int n = 0; n < kPcTiles; ++n) {
      __half2 h[4];
      TileDecodeFast<K>(r[n][0], r[n][1], r[n][2], win.shift, h);
      GemmNativeStore(h, bdst + P * kBuf + n * kGemmWaveWords, lane);
    }
  };
  if constexpr (PcBareProducer(MUT)) {
    GemmGroupSync();
    for (int a = 0; a < A; ++a) GemmGroupSync();
    return;
  }
  // Prologue: data steps 0 .. D - 1 in order, step 0 to slot 0, then the load of step D (TileBodyPipe's order).
  load(0, av[0], wr[0]);
  TileFor<1, D>([&](auto ic) {
    constexpr int i = decltype(ic)::value;
    __builtin_amdgcn_sched_barrier(0);
    load(i, av[i], wr[i]);
  });
  __builtin_amdgcn_sched_barrier(0);
  put(0, av[0], wr[0]);
  __builtin_amdgcn_sched_barrier(0);
  load(D, av[0], wr[0]);
  __builtin_amdgcn_sched_barrier(0);
  if constexpr (PRIO > 0) __builtin_amdgcn_s_setprio(PRIO);
  GemmGroupSync();
  // Step a (slot parity P = a % 2; set R = (a + 1) % D holds data step a + 1; X set Q = a % D).
  auto step = [&](auto pc, auto rc, auto qc, int a) {
    constexpr int P = decltype(pc)::value, R = decltype(rc)::value, Q = decltype(qc)::value;
    put(1 - P, av[R], wr[R]);
    __builtin_amdgcn_sched_barrier(0);
    load(a + 1 + D, av[R], wr[R]);
    __builtin_amdgcn_sched_barrier(0);
    xp.template Step<Q>(a);
    __builtin_amdgcn_sched_barrier(0);
    GemmGroupSync();
  };
  using std::integral_constant;
  int a = 0;
  for (; a + kU <= A; a += kU)
    TileFor<0, kU>([&](auto uc) {
      constexpr int u = decltype(uc)::value;
      step(integral_constant<int, u % 2>{}, integral_constant<int, (u + 1) % D>{}, integral_constant<int, u % D>{},
           a + u);
    });
  TileFor<0, kU - 1>([&](auto uc) {
    constexpr int u = decltype(uc)::value;
    if (a + u < A)
      step(integral_constant<int, u % 2>{}, integral_constant<int, (u + 1) % D>{}, integral_constant<int, u % D>{},
           a + u);
  });
  if constexpr (PRIO > 0) __builtin_amdgcn_s_setprio(0);
}

// The consumer part of one item: N live row slots, S sides; the WMMAs of TileBodyPipe in its order.
template <int N, int S, int MUT = 0>
__device__ __forceinline__ void PcConsume(int A, const uint32_t* lds, GemmF8 (&acc)[kTileSlots][kTileCols]) {
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  constexpr int kBuf = S * kTileBufWords;
  const int t = PcTid(), lane = t & 31, w = __builtin_amdgcn_readfirstlane(t >> 5);
  const int side = w >> 3, cg = (w >> 2) & 1, g = TileGroup(w);
  const uint32_t* bsrc = lds + (side * 8 + cg * kTileCols) * kGemmWaveWords;
  const uint32_t* aread = lds + 2 * kBuf + side * kTileAWords + ((g * 16) + (lane & 15)) * 4;
  GemmGroupSync();  // slot 0 from the producers' prologue
  auto step = [&](auto pc) {
    constexpr int P = decltype(pc)::value;
    if constexpr (N > 0) {
      GemmH16 afr[N];
#pragma unroll
      for (int i = 0; i < N; ++i) {
        auto* ap = reinterpret_cast<U4*>(&afr[i]);
        const uint32_t* sp = aread + P * S * kTileAWords + i * 64 * 4;
        ap[0] = *reinterpret_cast<const U4*>(sp);
        ap[1] = *reinterpret_cast<const U4*>(sp + kTileAPlane);
      }
      const uint32_t* bp = bsrc + P * kBuf;
      GemmH16 bf;
#pragma unroll
      for (int c = 0; c < kTileCols; ++c) {
        if (MUT != kPcMutOneB || c == 0) bf = GemmStoredFragment(bp + c * kGemmWaveWords, lane);
#pragma unroll
        for (int i = 0; i < N; ++i) acc[i][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr[i], bf, acc[i][c]);
      }
    }
    GemmGroupSync();
  };
  using P0 = std::integral_constant<int, 0>;
  using P1 = std::integral_constant<int, 1>;
  int a = 0;
  for (; a + 1 < A; a += 2) {
    step(P0{});
    step(P1{});
  }
  if (a < A) step(P0{});
}

// ---- fat consumers: 6 x 4 output tiles per consumer wave ----
// The k loops of the tile and pc kernels move about 65 bytes of LDS per clock per WGP at every row count, and they
// need 800 to 880 LDS bytes per WMMA (3 A + 4 B fragments of 1 KB per 12 WMMAs, plus the producer stores). A wave
// with 6 row tiles x 4 column tiles reads 6 A + 4 B fragments per 24 WMMAs: 29% fewer fragment reads per WMMA.
// Gate/up: 8 consumer waves (w < 8) and the 4 producers (8 .. 11) of PcProduce; down: 4 consumers and 2 producers.
// Consumer w: column group cg = w & 1 (tiles 4 cg .. 4 cg + 3), side (w >> 1) & 1 (gate/up), row group g = w >> S;
// it owns the row tiles g + 2 i (i < 6). With the waves of a block on the SIMDs in turn (w % 4), each SIMD gets both
// row groups of one gate/up column group, so a part-filled item loads the SIMDs evenly.
// Exactness: the same LDS slots, A and B fragment bytes, one WMMA per output tile and k step in the order 0 .. A-1 from
// zero, and the same epilogue values per row (the 32-row rounds stage the same tiles; a wave takes 4 rows, not 2).
constexpr int kFatSlots = 6, kFatPairWaves = 12, kFatDownWaves = 6;
__host__ __device__ constexpr int FatLive(int L, int g) {
  return L > g ? ((L - g + 1) / 2 < kFatSlots ? (L - g + 1) / 2 : kFatSlots) : 0;
}
template <int S> __device__ __forceinline__ int FatGroup(int w) { return w >> S; }
// The map covers every live row tile once: for L live tiles (0 .. 12), the tiles g + 2 i (g < 2, i < FatLive(L, g))
// are 0 .. L - 1, each one time. Consumers: 2 sides x 2 column groups x 2 row groups (gate/up), 2 x 2 (down). The fat
// epilogue round stages tiles 2 q (g 0) and 2 q + 1 (g 1) from slot q; its 8 consumer waves take 4 of the 32 rows each.
constexpr bool FatMapOk() {
  for (int L = 0; L <= kTileRm; ++L) {
    int seen[kTileRm] = {};
    for (int g = 0; g < 2; ++g)
      for (int i = 0; i < FatLive(L, g); ++i) {
        if (g + 2 * i >= L) return false;
        ++seen[g + 2 * i];
      }
    for (int r = 0; r < L; ++r)
      if (seen[r] != 1) return false;
  }
  return 2 * kFatSlots == kTileRm && kFatPairWaves - 4 == 2 * 2 * 2 && kFatDownWaves - 2 == 2 * 2 &&
         (kFatPairWaves - 4) * 4 == 32;
}
static_assert(FatMapOk());

template <int N, int S, int MUT = 0>
__device__ __forceinline__ void PcFatConsume(int A, const uint32_t* lds, GemmF8 (&acc)[kFatSlots][kTileCols]) {
  using U4 = uint32_t __attribute__((ext_vector_type(4)));
  constexpr int kBuf = S * kTileBufWords;
  const int t = PcTid(), lane = t & 31, w = __builtin_amdgcn_readfirstlane(t >> 5);
  const int side = S == 2 ? (w >> 1) & 1 : 0, cg = w & 1, g = FatGroup<S>(w);
  const uint32_t* bsrc = lds + (side * 8 + cg * kTileCols) * kGemmWaveWords;
  const uint32_t* aread = lds + 2 * kBuf + side * kTileAWords + ((g * 16) + (lane & 15)) * 4;
  GemmH16 nb[kTileCols], na;  // NoLds diagnostics: the fragments of slot 0, read once (no barrier: wrong values)
  if constexpr (PcNoLdsConsumer(MUT) && N > 0) {
#pragma unroll
    for (int c = 0; c < kTileCols; ++c) nb[c] = GemmStoredFragment(bsrc + c * kGemmWaveWords, lane);
    auto* ap = reinterpret_cast<U4*>(&na);
    ap[0] = *reinterpret_cast<const U4*>(aread);
    ap[1] = *reinterpret_cast<const U4*>(aread + kTileAPlane);
  }
  GemmGroupSync();  // slot 0 from the producers' prologue
  auto step = [&](auto pc) {
    constexpr int P = decltype(pc)::value;
    if constexpr (N > 0) {
      const uint32_t* bp = bsrc + P * kBuf;
      const uint32_t* as = aread + P * S * kTileAWords;
      auto load_a = [&](GemmH16& f, int i) {
        auto* ap = reinterpret_cast<U4*>(&f);
        ap[0] = *reinterpret_cast<const U4*>(as + i * 32 * 4);
        ap[1] = *reinterpret_cast<const U4*>(as + i * 32 * 4 + kTileAPlane);
      };
      GemmH16 bf[kTileCols], afr;
      if constexpr (PcNoLdsConsumer(MUT)) {
#pragma unroll
        for (int i = 0; i < N; ++i)
#pragma unroll
          for (int c = 0; c < kTileCols; ++c) acc[i][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(na, nb[c], acc[i][c]);
      } else if constexpr (MUT == kPcMutFatCs) {
#pragma unroll
        for (int c = 0; c < kTileCols; ++c) bf[c] = GemmStoredFragment(bp + c * kGemmWaveWords, lane);
#pragma unroll
        for (int i = 0; i < N; ++i) {
          load_a(afr, i);
#pragma unroll
          for (int c = 0; c < kTileCols; ++c) acc[i][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr, bf[c], acc[i][c]);
        }
      } else {
        // A 0 before the 4 B fragments: the first WMMA waits for 4 of the 10 loads, not for all of them. One A register
        // set: at 6 waves per SIMD the limit is 240 VGPRs (the 24-VGPR granule), and 192 accumulators, 4 B and 1 A
        // fragment and the addresses use 237. So each later A load waits before the 4 WMMAs of its row tile; the other
        // consumer waves of the SIMD fill that wait.
        load_a(afr, 0);
#pragma unroll
        for (int c = 0; c < kTileCols; ++c)
          if (MUT != kPcMutOneB || c == 0) bf[c] = GemmStoredFragment(bp + c * kGemmWaveWords, lane);
#pragma unroll
        for (int i = 0; i < N; ++i) {
          if (i > 0) load_a(afr, i);
#pragma unroll
          for (int c = 0; c < kTileCols; ++c)
            acc[i][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(afr, bf[MUT == kPcMutOneB ? 0 : c], acc[i][c]);
        }
      }
    }
    GemmGroupSync();
  };
  using P0 = std::integral_constant<int, 0>;
  using P1 = std::integral_constant<int, 1>;
  int a = 0;
  for (; a + 1 < A; a += 2) {
    step(P0{});
    step(P1{});
  }
  if (a < A) step(P0{});
}

// TilePairEpilogue with the lane values from t (PcTid), LinOutLaneAt and LinInLaneAt (exl3.h), and the 2 rows of a wave one
// after the other (unroll 1): the same operations in the same order per row, fewer live registers next to the 96
// accumulators.
template <int N, int MUT>
__device__ __forceinline__ void PcPairEpilogue(const PrefillMoe& a, int e, int M, int m0, int b0,
                                               const GemmF8 (&acc)[kTileSlots][kTileCols], float* stage, int t) {
  const int lane = t & 31, w = __builtin_amdgcn_readfirstlane(t >> 5);
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
    __syncthreads();
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
#pragma unroll 1
    for (int h = 0; h < 2; ++h) {
      const int lr = w * 2 + h, row = row0 + lr;
      if (row >= M) break;
      const float* gp = stage + lr * kPairPad + (MUT == 2 ? 128 : 0) + lane * 4;
      const float* up = stage + lr * kPairPad + (MUT == 2 ? 0 : 128) + lane * 4;
      float gv[4], uv[4];
      LinOutLaneAt(gp, gs, gv, lane);
      LinOutLaneAt(up, us, uv, lane);
      float s[4], v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) s[i] = Silu(gv[i]) * uv[i];
      LinInLaneAt<MUT == 1>(s, ds, v, lane);
      __half* dst = xd + PackedIndex(row, c, M);
#pragma unroll
      for (int i = 0; i < 4; ++i) dst[i] = __float2half(v[i] * kHadNorm);
    }
  });
}

// The barriers of TilePairEpilogue's rounds, for the producer waves (they do no epilogue work).
__device__ __forceinline__ void PcEpilogueIdle(int M, int m0) {
#pragma unroll 1
  for (int q = 0; q < kTileRm / 2; ++q) {
    if (m0 + q * 32 >= M) break;
    __syncthreads();
    __syncthreads();
  }
}

template <int N, int MUT>
__device__ __forceinline__ void PcPairItem(const PrefillMoe& a, const Exl3GemmJob& j, const Exl3GemmItem& it,
                                           uint32_t* lds) {
  GemmF8 acc[kTileSlots][kTileCols] = {};
  PcConsume<N, 2, MUT>(j.I / 16, lds, acc);
  PcPairEpilogue<N, MUT == 1 || MUT == 2 ? MUT : 0>(a, it.job >> 1, j.M, it.m0, it.b0, acc,
                                                    reinterpret_cast<float*>(lds), PcTid());
}

// PcPairEpilogue for the fat map: in round q (row tiles 2 q and 2 q + 1) every consumer wave stages its slot q (row
// tile g + 2 q, the stage half g), and the 8 waves take 4 of the 32 rows each. The per-row operations and their order
// are PcPairEpilogue's.
template <int N, int MUT>
__device__ __forceinline__ void PcFatPairEpilogue(const PrefillMoe& a, int e, int M, int m0, int b0,
                                                  const GemmF8 (&acc)[kFatSlots][kTileCols], float* stage, int t) {
  const int lane = t & 31, w = __builtin_amdgcn_readfirstlane(t >> 5);
  const int side = (w >> 1) & 1, cg = w & 1, g = FatGroup<2>(w), c = b0 * 16 + lane * 4;
  __half gs[4], us[4], ds[4];
  LinScaleLane(PrefillUniform(a.gu[e * 2].svh), c, gs);
  LinScaleLane(PrefillUniform(a.gu[e * 2 + 1].svh), c, us);
  LinScaleLane(PrefillUniform(a.down[e].suh), c, ds);
  __half* xd = PrefillUniform(a.down[e].xh);
  bool done = false;
  TileFor<0, kTileRm / 2>([&](auto qc) {
    constexpr int q = decltype(qc)::value;
    const int row0 = m0 + q * 32;
    if (done || row0 >= M) { done = true; return; }
    __syncthreads();
    if constexpr (q < N) {
#pragma unroll
      for (int cc = 0; cc < kTileCols; ++cc)
#pragma unroll
        for (int z = 0; z < 8; ++z)
          stage[(g * 16 + 2 * z + (lane >> 4)) * kPairPad + side * 128 + (cg * kTileCols + cc) * 16 + (lane & 15)] =
              acc[q][cc][z];
    }
    __syncthreads();
#pragma unroll 1
    for (int h = 0; h < 4; ++h) {
      const int lr = w * 4 + h, row = row0 + lr;
      if (row >= M) break;
      const float* gp = stage + lr * kPairPad + (MUT == 2 ? 128 : 0) + lane * 4;
      const float* up = stage + lr * kPairPad + (MUT == 2 ? 0 : 128) + lane * 4;
      float gv[4], uv[4];
      LinOutPairLaneAt(gp, up, gs, us, gv, uv, lane);
      float s[4], v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) s[i] = Silu(gv[i]) * uv[i];
      LinInLaneAt<MUT == 1>(s, ds, v, lane);
      __half* dst = xd + PackedIndex(row, c, M);
#pragma unroll
      for (int i = 0; i < 4; ++i) dst[i] = __float2half(v[i] * kHadNorm);
    }
  });
}

template <int N, int MUT>
__device__ __forceinline__ void PcFatPairItem(const PrefillMoe& a, const Exl3GemmJob& j, const Exl3GemmItem& it,
                                              uint32_t* lds) {
  GemmF8 acc[kFatSlots][kTileCols] = {};
  PcFatConsume<N, 2, MUT>(j.I / 16, lds, acc);
  PcFatPairEpilogue<N, MUT == 1 || MUT == 2 ? MUT : 0>(a, it.job >> 1, j.M, it.m0, it.b0, acc,
                                                       reinterpret_cast<float*>(lds), PcTid());
}

// Gate/up with the mid step: the PrefillMoeTilePairKernel queue (G claims, X claims, fx_ready), 20 waves. X claims
// take 5 X item pairs (one gate/up pair per 128-thread group, PcFxPairTransform; group 4 is the producer waves).
// XD > 0: X claims take only the X items of units 0 .. XD - 1, and the producers of G claim g write the X of claim
// g + 5 XD (PcXProd), then the block counts one on that unit's fx_ready; a unit from XD on waits for its 5 producers. A producer's unit is XD units
// before its own, so its claim came first and its block runs; X claims never wait: no deadlock (the D1 rule).
// Each role runs its own copy of the queue loop (PROD is a template argument), so the consumer loop holds no producer
// value and its 96 accumulators fit the budget. Both copies take the same barriers in the same order.
// FAT: the fat consumer map (8 consumers, producers 8 .. 11, 3 X claim groups: 6 X items per claim).
template <int KMAX, int MUT, int XD, int D, int PRIO, bool PROD, bool FAT = false>
__device__ __forceinline__ void PcPairLoop(const PrefillMoe& a, int lead, uint32_t* lds, int& claim) {
  constexpr int per = 10 / kMoeNb, kGroups = (FAT ? kFatPairWaves : kPcPairWaves) / 4, P0 = FAT ? 8 : 16;
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5), grp = __builtin_amdgcn_readfirstlane(threadIdx.x >> 7);
  const int side = PROD ? (w - P0) >> 1 : FAT ? (w >> 1) & 1 : w >> 3, T = a.tile_rows;
  const int ng = PrefillUniform(a.item_counts[0]) / 2;
  int nx = PrefillUniform(a.fx_count[2]);
  if constexpr (XD > 0) {
    // The first X item of unit XD (X items are in unit order, 2 per 16 rows).
    if (XD * per < ng) {
      const auto first = PrefillUniform(a.i_gu[2 * XD * per]);
      nx = PrefillUniform(a.fx_xoff[first.job >> 1]) + 2 * (first.m0 / 16);
    }
  }
  for (;;) {
    if (PcTid() == 0) claim = atomicAdd(a.fx_count + 1, 1);
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
      const int t = PcTid();
      if (t == 0)
        claim = __hip_atomic_load(a.fx_count, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < limit
                    ? atomicAdd(a.fx_count, 2 * kGroups) : nx;
      __syncthreads();
      // x and nx are even (2 X items per 16 rows, in unit order), so a pair that starts before nx ends before it.
      const int x = PrefillUniform(claim);
      __syncthreads();
      if (x >= nx) break;
      if (x + 2 * grp < nx) PcFxPairTransform<FAT>(a, x + 2 * grp, t & 127);
      else PcFxPairIdle();
    }
    if (g >= ng) return;
    const int target = XD > 0 && g / per >= XD ? per : 2 * ((min(j.M, it.m0 + T) - it.m0 + 15) / 16);
    std::conditional_t<(XD > 0 && PROD), PcXProd<D, MUT>, PcNoXProd> xp{};
    const bool xprod = XD > 0 && g + XD * per < ng;
    if constexpr (XD > 0) {
      if (xprod) {
        // The target unit: its rows to the LDS table (the first L threads, consumers), its block and suh to the
        // producers.
        const auto tt = PrefillUniform(a.i_gu[2 * (g + XD * per)]);
        const int e = tt.job >> 1, M = PrefillUniform(a.gu[e * 2].R), L = min(T, M - tt.m0);
        uint32_t* rows = lds + kTilePairWords;
        const int t = PcTid();
        if constexpr (PROD) {
          const int lane = t & 31, cb = g % per + 5 * (w - P0);
          xp.xh = PrefillUniform(a.xh);
          xp.dg = PrefillUniform(a.gu[e * 2].xh);
          xp.du = PrefillUniform(a.gu[e * 2 + 1].xh);
          xp.rows = rows;
          xp.L = L; xp.M = M; xp.m0 = tt.m0; xp.cb = cb;
          xp.dlane = (uint32_t)(((lane >> 2) * M * 16 + (lane & 3) * 4) * 2);
          xp.sg = *reinterpret_cast<const uint2*>(PrefillUniform(a.gu[e * 2].suh) + cb * 128 + lane * 4);
          xp.su = *reinterpret_cast<const uint2*>(PrefillUniform(a.gu[e * 2 + 1].suh) + cb * 128 + lane * 4);
        } else if (t < L) {
          rows[t] = e == kExperts ? tt.m0 + t : a.dispatch.pairs[PrefillUniform(a.dispatch.offsets[e]) + tt.m0 + t] / kTopK;
        }
      }
    }
    if (PcTid() == 0)
      while (__hip_atomic_load(a.fx_ready + g / per, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target)
        __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    if constexpr (PROD) {
      TileDispatchK<KMAX>(j.K, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        PcProduce<K, 2, D, MUT, PRIO>(j, it.m0, it.b0, lds, w - P0, xp);
      });
      PcEpilogueIdle(j.M, it.m0);
    } else if constexpr (FAT) {
      switch (FatLive((min(T, j.M - it.m0) + 15) / 16, FatGroup<2>(w))) {
        case 0: PcFatPairItem<0, MUT>(a, j, it, lds); break;
        case 1: PcFatPairItem<1, MUT>(a, j, it, lds); break;
        case 2: PcFatPairItem<2, MUT>(a, j, it, lds); break;
        case 3: PcFatPairItem<3, MUT>(a, j, it, lds); break;
        case 4: PcFatPairItem<4, MUT>(a, j, it, lds); break;
        case 5: PcFatPairItem<5, MUT>(a, j, it, lds); break;
        default: PcFatPairItem<6, MUT>(a, j, it, lds); break;
      }
    } else {
      switch (TileLive((min(T, j.M - it.m0) + 15) / 16, TileGroup(w))) {
        case 0: PcPairItem<0, MUT>(a, j, it, lds); break;
        case 1: PcPairItem<1, MUT>(a, j, it, lds); break;
        case 2: PcPairItem<2, MUT>(a, j, it, lds); break;
        default: PcPairItem<3, MUT>(a, j, it, lds); break;
      }
    }
    if constexpr (XD > 0) {
      // Publish the produced X: every thread's stores, then one count on the target unit.
      if (xprod) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if (PcTid() == 0) atomicAdd(a.fx_ready + g / per + XD, 1);
      }
    }
    __syncthreads();
  }
}

// waves_per_eu(10): at most 144 VGPRs (the 24-VGPR granule), so 2 blocks of 20 waves fit a WGP.
template <int KMAX, int MUT = 0, int XD = kPcXDist, int D = kPcDepth, int PRIO = kPcPrio>
__global__ void __launch_bounds__(32 * kPcPairWaves) __attribute__((amdgpu_waves_per_eu(10)))
PrefillMoePcPairKernel(PrefillMoe a, int lead) {
  __shared__ __attribute__((aligned(16))) uint32_t lds[kTilePairWords + kPcRowWords];
  __shared__ int claim;
  if (__builtin_amdgcn_readfirstlane(threadIdx.x >> 5) >= 16) PcPairLoop<KMAX, MUT, XD, D, PRIO, true>(a, lead, lds, claim);
  else PcPairLoop<KMAX, MUT, XD, D, PRIO, false>(a, lead, lds, claim);
}

// The fat gate/up kernel: 8 fat consumers and 4 producers, 2 blocks per WGP (LDS), so waves_per_eu(6) gives 240 VGPRs
// (the 24-VGPR granule) for the 192 accumulators. Default X distance 0: the producers run no X transform (the X claims do all X items).
constexpr int kPcFatXDist = 0;
template <int KMAX, int MUT = 0, int XD = kPcFatXDist, int D = kPcDepth, int PRIO = kPcPrio>
__global__ void __launch_bounds__(32 * kFatPairWaves) __attribute__((amdgpu_waves_per_eu(6)))
PrefillMoePcFatPairKernel(PrefillMoe a, int lead) {
  __shared__ __attribute__((aligned(16))) uint32_t lds[kTilePairWords + kPcRowWords];
  __shared__ int claim;
  static_assert(16 * kTileRm <= 32 * (kFatPairWaves - 4));  // the X producer's row table: one consumer thread a row
  if (__builtin_amdgcn_readfirstlane(threadIdx.x >> 5) >= 8)
    PcPairLoop<KMAX, MUT, XD, D, PRIO, true, true>(a, lead, lds, claim);
  else PcPairLoop<KMAX, MUT, XD, D, PRIO, false, true>(a, lead, lds, claim);
}

// ---- down projection (fp16 pd, TileDownItem's store) ----

// The pd store of N row slots: slot i is row tile g + STEP i, the column tiles 4 cg .. 4 cg + 3 of the item.
template <int N, int STEP, int SLOTS>
__device__ __forceinline__ void PcDownStore(const Exl3GemmJob& j, const Exl3GemmItem& it,
                                            const GemmF8 (&acc)[SLOTS][kTileCols], int g, int cg, int lane) {
  const auto ps = StreamRsrc(reinterpret_cast<const uint32_t*>(j.partial), (uint32_t)j.M * j.O * 2);
#pragma unroll
  for (int i = 0; i < N; ++i)
#pragma unroll
    for (int c = 0; c < kTileCols; ++c) {
      const uint32_t po = ((uint32_t)(it.m0 + (g + STEP * i) * 16 + lane / 16) * j.O + (it.b0 + cg * kTileCols + c) * 16 +
                           (lane & 15)) * 2;
#pragma unroll
      for (int z = 0; z < 8; ++z) {
        const float v = acc[i][c][z];
        const __half h = __float2half(fabsf(v) > 65504.f ? copysignf(65504.f, v) : v);
        __builtin_amdgcn_raw_buffer_store_b16(__half_as_ushort(h), ps, po + (uint32_t)(2 * z) * j.O * 2, 0, 0);
      }
    }
}

template <int N, int MUT>
__device__ __forceinline__ void PcDownItem(const Exl3GemmJob& j, const Exl3GemmItem& it, uint32_t* lds) {
  GemmF8 acc[kTileSlots][kTileCols] = {};
  PcConsume<N, 1, MUT>(j.I / 16, lds, acc);
  const int t = PcTid(), lane = t & 31, w = __builtin_amdgcn_readfirstlane(t >> 5);
  PcDownStore<N, 4>(j, it, acc, TileGroup(w), (w >> 2) & 1, lane);
}

template <int N, int MUT>
__device__ __forceinline__ void PcFatDownItem(const Exl3GemmJob& j, const Exl3GemmItem& it, uint32_t* lds) {
  GemmF8 acc[kFatSlots][kTileCols] = {};
  PcFatConsume<N, 1, MUT>(j.I / 16, lds, acc);
  const int t = PcTid(), lane = t & 31, w = __builtin_amdgcn_readfirstlane(t >> 5);
  PcDownStore<N, 2>(j, it, acc, FatGroup<1>(w), w & 1, lane);
}

// Down items of 192 rows x 128 columns (PrefillDownItems with tile_rows 192), 8 consumer and 2 producer waves.
// Between items no block barrier is needed: a consumer's last fragment reads are before the last k-step barrier,
// which the producers also take before the next item's prologue writes slot 0. The consumer epilogue uses no LDS.
template <int KMAX, int MUT = 0, int D = kPcDepth, int PRIO = kPcPrio>
__global__ void __launch_bounds__(32 * kPcDownWaves) __attribute__((amdgpu_waves_per_eu(10)))
PrefillMoePcDownKernel(const Exl3GemmJob* jobs, const Exl3GemmItem* items, const int* count) {
  __shared__ __attribute__((aligned(16))) uint32_t lds[TileLoopWords(1)];
  const int n = PrefillUniform(*count);
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const bool prod = w >= 8;
  PcNoXProd nox;
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    const auto it = PrefillUniform(items[i]);
    const auto j = PrefillUniform(jobs[it.job]);
    if (prod) {
      TileDispatchK<KMAX>(j.K, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        PcProduce<K, 1, D, MUT, PRIO>(j, it.m0, it.b0, lds, w - 8, nox);
      });
    } else {
      switch (TileLive((min(16 * kTileRm, j.M - it.m0) + 15) / 16, TileGroup(w))) {
        case 0: PcDownItem<0, MUT>(j, it, lds); break;
        case 1: PcDownItem<1, MUT>(j, it, lds); break;
        case 2: PcDownItem<2, MUT>(j, it, lds); break;
        default: PcDownItem<3, MUT>(j, it, lds); break;
      }
    }
  }
}

// The fat down kernel: 4 fat consumers and 2 producers (4, 5), 4 blocks per WGP, 240 VGPRs. Each role runs its
// own copy of the item loop (PcPairLoop's rule): with one loop, the producer's lane values were hoisted out of it and
// stayed live next to the 192 accumulators (466 VGPRs spilled, 844 bytes of scratch per lane).
template <int KMAX, int MUT, int D, int PRIO, bool PROD>
__device__ __forceinline__ void PcFatDownLoop(const Exl3GemmJob* jobs, const Exl3GemmItem* items, int n, uint32_t* lds) {
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  PcNoXProd nox;
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    const auto it = PrefillUniform(items[i]);
    const auto j = PrefillUniform(jobs[it.job]);
    if constexpr (PROD) {
      TileDispatchK<KMAX>(j.K, [&](auto kc) {
        constexpr int K = decltype(kc)::value;
        PcProduce<K, 1, D, MUT, PRIO>(j, it.m0, it.b0, lds, w - 4, nox);
      });
    } else {
      switch (FatLive((min(16 * kTileRm, j.M - it.m0) + 15) / 16, FatGroup<1>(w))) {
        case 0: PcFatDownItem<0, MUT>(j, it, lds); break;
        case 1: PcFatDownItem<1, MUT>(j, it, lds); break;
        case 2: PcFatDownItem<2, MUT>(j, it, lds); break;
        case 3: PcFatDownItem<3, MUT>(j, it, lds); break;
        case 4: PcFatDownItem<4, MUT>(j, it, lds); break;
        case 5: PcFatDownItem<5, MUT>(j, it, lds); break;
        default: PcFatDownItem<6, MUT>(j, it, lds); break;
      }
    }
  }
}
template <int KMAX, int MUT = 0, int D = kPcDepth, int PRIO = kPcPrio>
__global__ void __launch_bounds__(32 * kFatDownWaves) __attribute__((amdgpu_waves_per_eu(6)))
PrefillMoePcFatDownKernel(const Exl3GemmJob* jobs, const Exl3GemmItem* items, const int* count) {
  __shared__ __attribute__((aligned(16))) uint32_t lds[TileLoopWords(1)];
  const int n = PrefillUniform(*count);
  if (__builtin_amdgcn_readfirstlane(threadIdx.x >> 5) >= 4) PcFatDownLoop<KMAX, MUT, D, PRIO, true>(jobs, items, n, lds);
  else PcFatDownLoop<KMAX, MUT, D, PRIO, false>(jobs, items, n, lds);
}

// Grids: 2 gate/up blocks (20 waves) and 4 down blocks (10 waves) per WGP; the blocks loop over the queues.
template <int MUT = 0, int XD = kPcXDist, int D = kPcDepth, int PRIO = kPcPrio>
inline void LaunchPrefillMoePcPair(int kmax, const PrefillMoe& a, int lead) {
  switch (kmax) {
    case 5: PrefillMoePcPairKernel<5, MUT, XD, D, PRIO><<<512, 32 * kPcPairWaves>>>(a, lead); break;
    case 6: PrefillMoePcPairKernel<6, MUT, XD, D, PRIO><<<512, 32 * kPcPairWaves>>>(a, lead); break;
    default: std::fprintf(stderr, "MoE pc path: no kernel set for KMAX %d\n", kmax); std::abort();
  }
}
template <int MUT = 0, int D = kPcDepth, int PRIO = kPcPrio>
inline void LaunchPrefillMoePcDown(int kmax, const Exl3GemmJob* jobs, const Exl3GemmItem* items, const int* count) {
  switch (kmax) {
    case 5: PrefillMoePcDownKernel<5, MUT, D, PRIO><<<1024, 32 * kPcDownWaves>>>(jobs, items, count); break;
    case 6: PrefillMoePcDownKernel<6, MUT, D, PRIO><<<1024, 32 * kPcDownWaves>>>(jobs, items, count); break;
    default: std::fprintf(stderr, "MoE pc path: no kernel set for KMAX %d\n", kmax); std::abort();
  }
}
// The fat grids: 2 gate/up blocks (12 waves) and 4 down blocks (6 waves) per WGP.
template <int MUT = 0, int XD = kPcFatXDist, int D = kPcDepth, int PRIO = kPcPrio>
inline void LaunchPrefillMoePcFatPair(int kmax, const PrefillMoe& a, int lead) {
  switch (kmax) {
    case 5: PrefillMoePcFatPairKernel<5, MUT, XD, D, PRIO><<<512, 32 * kFatPairWaves>>>(a, lead); break;
    case 6: PrefillMoePcFatPairKernel<6, MUT, XD, D, PRIO><<<512, 32 * kFatPairWaves>>>(a, lead); break;
    default: std::fprintf(stderr, "MoE pc path: no kernel set for KMAX %d\n", kmax); std::abort();
  }
}
template <int MUT = 0, int D = kPcDepth, int PRIO = kPcPrio>
inline void LaunchPrefillMoePcFatDown(int kmax, const Exl3GemmJob* jobs, const Exl3GemmItem* items, const int* count) {
  switch (kmax) {
    case 5: PrefillMoePcFatDownKernel<5, MUT, D, PRIO><<<1024, 32 * kFatDownWaves>>>(jobs, items, count); break;
    case 6: PrefillMoePcFatDownKernel<6, MUT, D, PRIO><<<1024, 32 * kFatDownWaves>>>(jobs, items, count); break;
    default: std::fprintf(stderr, "MoE pc path: no kernel set for KMAX %d\n", kmax); std::abort();
  }
}
// The engine's kernels with kPrefillMoePc on: the fat kernels, or the thin 20- and 10-wave kernels.
// kPcFatDown false: the fat gate/up kernel with the tile down kernel, the bench arm `fat-gu`. The fat
// down kernel was not faster than the tile down (3 of 6 pairs at 8,320 rows, 0 of 6 at 16,512).
constexpr bool kPcFat = true, kPcFatDown = false;
// The engine's launches (prefill_engine.inc). With kPrefillMoePc off they are empty and instantiate no kernel (the
// calls depend on ON), so the engine's device code is the tile path's.
template <bool ON = kPrefillMoePc> inline void LaunchPrefillMoePcEngine(int kmax, const PrefillMoe& a, int lead) {
  if constexpr (ON && kPcFat) LaunchPrefillMoePcFatPair<ON ? 0 : 1>(kmax, a, lead);
  else if constexpr (ON) LaunchPrefillMoePcPair<ON ? 0 : 1>(kmax, a, lead);
}
template <bool ON = kPrefillMoePc>
inline void LaunchPrefillMoePcDownEngine(int kmax, const Exl3GemmJob* jobs, const Exl3GemmItem* items, const int* count) {
  if constexpr (ON && kPcFat && kPcFatDown) LaunchPrefillMoePcFatDown<ON ? 0 : 1>(kmax, jobs, items, count);
  else if constexpr (ON && kPcFat) LaunchPrefillMoeTileDown(kmax, jobs, items, count);
  else if constexpr (ON) LaunchPrefillMoePcDown<ON ? 0 : 1>(kmax, jobs, items, count);
}
}  // namespace crucible
