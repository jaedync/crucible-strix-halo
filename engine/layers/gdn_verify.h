// GDN verify core in one launch (CRUCIBLE_GDN_VERIFY).
//
// GdnKernel walks the rows of a forward in series. Each row has a chain of
// global loads (conv taps, x, z) and about 12 block barriers, so the state
// stream (3.1 MB read and 3.1 MB write per layer) waits behind a latency chain
// that grows with the rows (about 11 us per row and layer in the D1 trace).
// This kernel keeps GdnKernel's arithmetic for every element and every
// reduction order, and splits the work by role inside one block per value
// head (256 threads, 8 waves, so 4 blocks fit on a WGP and all 48 run at once):
// - Waves 0..3 (row waves) run GdnKernel's 256 threads, two per lane: lane l
//   of row wave w is thread t0 = 32 w + l (hf 0) and t1 = t0 + 128 (hf 1),
//   both of column j = t0, and holds GdnKernel's waves w and w + 4. They
//   compute the inputs of all rows together (conv, q/k l2 norm, a/b
//   projections, decay, beta) with GdnKernel's per-thread chains and BlockSum
//   trees, write the pending rows and the qkv ring while the recurrence runs,
//   and then run the output gate, the RMS norm and the Hadamard transform.
// - Waves 4..7 (state waves) own the state: lane l of state wave s holds
//   column 32 s + l and all 128 key rows. GdnKernel's two half chains run in
//   one thread, then part[0] + part[1]. The state loads issue at the start, so
//   the row inputs are computed while the state streams in, and the replay
//   rows and the rows run with no barrier.
// - The barriers order LDS only (no vmcnt wait), so the state loads stay in
//   flight across them.
// All outputs (state, ring, pending rows, out_proj input) are bitwise equal to
// GdnKernel's for every R, replay and commit (bench/gdn_verify.hip).
#pragma once

#include "../exl3/exl3.h"
#include "../switches.h"
#include "gdn_verify_switch.h"
#include "ops.h"

namespace crucible {

static_assert(kGdnVerifyMaxRows == kMaxRows, "the switch range is the rows of one forward");
constexpr int kGdnVerifyThreads = 256;  // 4 row waves, 4 state waves
constexpr int kGdnVerifyRowWaves = 4;

struct alignas(16) GdnVerifyLds {
  float k[2 * kMaxRows][kGdnDim];  // normalized k: replay rows, then this forward's rows
  float q[kMaxRows][kGdnDim];      // normalized and scaled q of this forward's rows
  float v[2 * kMaxRows][kGdnDim];  // v (conv output), same row order as k
  float dec[2 * kMaxRows][2];      // decay, beta
  float red[kMaxRows][4][8];       // GdnKernel's BlockSum wave sums of sq, sk, pa, pb
  union {
    float raw[kMaxRows][256];  // conv output of q (thread t < 128) or k (t >= 128); read before B2 only
    struct {
      float y[kMaxRows][kGdnDim];  // part[0] + part[1] of each row; written after B2 only
      float o[kMaxRows][kGdnDim];  // gated output before the suh scale
    } out;
  } u;
};

__device__ __forceinline__ void GdnVerifyBarrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// Keeps a product as its own rounded value: GdnKernel feeds x * x through a
// select into WaveSum, so the first butterfly add is not an fma there.
__device__ __forceinline__ float GdnVerifyOpaque(float v) {
  asm volatile("" : "+v"(v));
  return v;
}
// GdnKernel's BlockSum<256> total over its 8 wave sums, in order.
__device__ __forceinline__ float GdnVerifySum8(const float* s) {
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) t += s[i];
  return t;
}

// The state recurrence of one column over 128 key rows (S[ii], kq[ii] for half 0, S[64 + ii] for half 1), with
// k and q read from LDS 4 key rows per half at a time, one chunk ahead (128 state values beside all k values do
// not fit in 256 VGPRs). The empty asm with a memory clobber keeps the compiler from hoisting more LDS reads.
template <int MUT>
__device__ __forceinline__ void GdnVerifyMem(float (&S)[128], const float* kq, float decay, float& m0, float& m1) {
  const float4* k4 = reinterpret_cast<const float4*>(kq);
  float4 a = k4[0], b = k4[16];
#pragma unroll
  for (int c = 0; c < 16; ++c) {
    float4 na = a, nb = b;
    if (c < 15) {
      na = k4[c + 1];
      nb = k4[17 + c];
    }
    const float ka[4] = {a.x, a.y, a.z, a.w}, kb[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int ii = 4 * c + e;
      S[ii] *= decay;
      m0 += S[ii] * ka[e];
      S[64 + ii] *= decay;
      if constexpr (MUT == 1) m0 += S[64 + ii] * kb[e];
      else m1 += S[64 + ii] * kb[e];
    }
    asm volatile("" ::: "memory");
    a = na;
    b = nb;
  }
}
// S += k * delta; with Y, y0/y1 += S * q (GdnKernel's second loop).
template <bool Y>
__device__ __forceinline__ void GdnVerifyUpdate(float (&S)[128], const float* kq, const float* qq, float delta,
                                                float& y0, float& y1) {
  const float4* k4 = reinterpret_cast<const float4*>(kq);
  const float4* q4 = reinterpret_cast<const float4*>(qq);
  float4 a = k4[0], b = k4[16], qa{}, qb{};
  if constexpr (Y) {
    qa = q4[0];
    qb = q4[16];
  }
#pragma unroll
  for (int c = 0; c < 16; ++c) {
    float4 na = a, nb = b, nqa = qa, nqb = qb;
    if (c < 15) {
      na = k4[c + 1];
      nb = k4[17 + c];
      if constexpr (Y) {
        nqa = q4[c + 1];
        nqb = q4[17 + c];
      }
    }
    const float ka[4] = {a.x, a.y, a.z, a.w}, kb[4] = {b.x, b.y, b.z, b.w};
    const float qx[4] = {qa.x, qa.y, qa.z, qa.w}, qy[4] = {qb.x, qb.y, qb.z, qb.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int ii = 4 * c + e;
      S[ii] += ka[e] * delta;
      if constexpr (Y) y0 += S[ii] * qx[e];
      S[64 + ii] += kb[e] * delta;
      if constexpr (Y) y1 += S[64 + ii] * qy[e];
    }
    asm volatile("" ::: "memory");
    a = na;
    b = nb;
    qa = nqa;
    qb = nqb;
  }
}

// MUT (bench only, 0 in the engine): a defect the byte check must see.
// 1 one mem chain over both halves; 2 state stored one row late; 3 pending beta = decay;
// 4 no eps in the output norm; 5 no q/k ring write; 6 first replay row skips decay;
// 7 a/b chains in reverse order; 8 one more ulp on delta of the last row.
// RB: rows per input batch (all loads of a batch issue together).
template <int CACHE = 0, bool STREAM_STORE = false, class Args = GdnArgs, int MUT = 0, int RB = 2>
__global__ void __launch_bounds__(kGdnVerifyThreads) GdnVerifyKernel(Args args) {
  const GdnArgs a = GdnRowArgs(args, blockIdx.y);
  PublishWeights(a.ab);
  __shared__ GdnVerifyLds L;
  const int h = blockIdx.x, kh = h / 3;
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5), lane = threadIdx.x & 31;
  const int KR = a.replay + a.R;
  if (w >= kGdnVerifyRowWaves) {
    // ---- state waves: column j, key rows 0..127 ----
    const int j = 32 * (w - kGdnVerifyRowWaves) + lane;
    // Buffer loads and stores (GdnKernel's addresses and cache policy) with one VGPR offset: row i is at
    // soffset (i / 8) * 4096 plus the immediate (i % 8) * 512 (64-bit addresses do not fit beside the state).
    float S[128];
    const auto state_rs = StreamRsrc(reinterpret_cast<const uint32_t*>(a.state + (size_t)h * 128 * 128), 128 * 128 * 4);
    const int jo = j * 4;
    // B0: the row waves issue their input loads first, so those do not queue behind the 6.3 MB state stream.
    GdnVerifyBarrier();
#pragma unroll
    for (int i = 0; i < 128; ++i)
      S[i] = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(state_rs, jo + (i & 7) * 512, (i >> 3) * 4096, CACHE));
    GdnVerifyBarrier();  // B1: wave sums and raw conv outputs
    GdnVerifyBarrier();  // B2: k, q, v, decay, beta of every row
    for (int i = 0; i < KR; ++i) {
      asm volatile("" ::: "memory");
      float decay = L.dec[i][0];
      const float beta = L.dec[i][1], cv = L.v[i][j];
      if constexpr (MUT == 6) {
        if (i == 0 && a.replay > 0) decay = 1.f;
      }
      const int r = i - a.replay;
      float m0 = 0.f, m1 = 0.f;
      GdnVerifyMem<MUT>(S, L.k[i], decay, m0, m1);
      // The update reads k from LDS again: the compiler must not keep the 128 k values of the mem loop live.
      asm volatile("" ::: "memory");
      float delta = beta * (cv - (m0 + m1));
      if constexpr (MUT == 8) {
        if (r == a.R - 1) delta = __uint_as_float(__float_as_uint(delta) + 1u);
      }
      // One update path with no branch for replay rows and rows (a branch here makes the compiler copy the
      // state at the join): a replay row reads q of row 0 and writes y of row 0, which row 0 writes again.
      float y0 = 0.f, y1 = 0.f;
      const int ry = max(r, 0);
      GdnVerifyUpdate<true>(S, L.k[i], L.q[ry], delta, y0, y1);
      L.u.out.y[ry][j] = y0 + y1;
      // commit == 0 with replay > 0 would store the last replay row here, which GdnKernel never does. It cannot happen:
      // RetainedPrefix::Forward rejects commit < 1 (engine/runtime/retained_prefix.h:14) and every GDN launch goes through it.
      if (r == a.commit - 1 + (MUT == 2 ? 1 : 0)) {
#pragma unroll
        for (int ii = 0; ii < 128; ++ii)
          __builtin_amdgcn_raw_buffer_store_b32(__float_as_uint(S[ii]), state_rs, jo + (ii & 7) * 512, (ii >> 3) * 4096,
                                                STREAM_STORE ? CACHE : 0);
      }
    }
    GdnVerifyBarrier();  // B3: y of every row
    return;
  }

  // ---- row waves: GdnKernel's threads t0 (hf 0) and t1 = t0 + 128 (hf 1) of column j ----
  const int t0 = threadIdx.x, t1 = t0 + 128, j = t0;
  const int chq = kh * 128 + j, chk = 2048 + kh * 128 + j, chv = 4096 + h * 128 + j;
  // Replay rows of the previous forward: GdnKernel's ks, cv, decay and beta.
  // All loads issue first; the pending rows change only after B2.
  float pk[kMaxRows], pv[kMaxRows], pd[kMaxRows];
#pragma unroll
  for (int i = 0; i < kMaxRows; ++i) {
    const float* pe = a.pend + ((size_t)h * kMaxRows + min(i, max(a.replay - 1, 0))) * kPend;
    pk[i] = pe[j];
    pv[i] = pe[128 + j];
    pd[i] = pe[256 + (t0 & 1)];
  }
  float wq[4], wk[4], wv[4];
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    wq[u] = a.conv[(size_t)chq * 4 + u];
    wk[u] = a.conv[(size_t)chk * 4 + u];
    wv[u] = a.conv[(size_t)chv * 4 + u];
  }
  constexpr int KH = kHidden / 256;
  __half wa0[KH], wb0[KH], wa1[KH], wb1[KH];
  {
    const __half* pa = a.ab + (size_t)h * kHidden;
    const __half* pb = a.ab + (size_t)(48 + h) * kHidden;
#pragma unroll
    for (int k = 0; k < KH; ++k) {
      wa0[k] = pa[t0 + 256 * k];
      wb0[k] = pb[t0 + 256 * k];
      wa1[k] = pa[t1 + 256 * k];
      wb1[k] = pb[t1 + 256 * k];
    }
  }
  // Loads of RB rows from row r0 (rows past R repeat row R - 1 and are not stored).
  float tq[RB][4], tk[RB][4], tv[RB][4], x0[RB][KH], x1[RB][KH];
  auto load_rows = [&](int r0) {
#pragma unroll
    for (int u = 0; u < RB; ++u) {
      const int r = min(r0 + u, a.R - 1), p = a.pos0 + r;
#pragma unroll
      for (int back = 0; back < 4; ++back) {
        // GdnKernel's tap(ch, back): this forward's rows, else the ring, else 0 before position 0.
        const int rr = r - back;
        const float* src = rr >= 0 ? a.qkv + (size_t)rr * kGdnQkvWidth
                                   : a.hist + (size_t)((p - back) & 15) * kGdnQkvWidth;
        // Raw loads only: the zero taps are selected after B0, so no load wait comes before it.
        tq[u][back] = src[chq];
        tk[u][back] = src[chk];
        tv[u][back] = src[chv];
      }
      const float* xr = a.x + (size_t)r * kHidden;
#pragma unroll
      for (int k = 0; k < KH; ++k) {
        x0[u][k] = xr[t0 + 256 * k];
        x1[u][k] = xr[t1 + 256 * k];
      }
    }
  };
  load_rows(0);
  GdnVerifyBarrier();  // B0: the state waves issue the state loads from here
#pragma unroll
  for (int i = 0; i < kMaxRows; ++i) {
    if (i < a.replay) {
      L.k[i][j] = pk[i];
      L.v[i][j] = pv[i];
      if (t0 < 2) L.dec[i][t0] = pd[i];
    }
  }
  for (int r0 = 0; r0 < a.R; r0 += RB) {
    if (r0 > 0) load_rows(r0);
#pragma unroll
    for (int u = 0; u < RB; ++u) {
      const int r = r0 + u;
      if (r >= a.R) break;
      // GdnKernel's conv: c += w[u] * tap(ch, 3 - u), then Silu. A tap before position 0 is 0.
      bool zero[4];
#pragma unroll
      for (int back = 0; back < 4; ++back) zero[back] = r - back < 0 && a.pos0 + r - back < 0;
      float cq = 0.f, ck = 0.f, cvv = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) cq += wq[e] * (zero[3 - e] ? 0.f : tq[u][3 - e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) ck += wk[e] * (zero[3 - e] ? 0.f : tk[u][3 - e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) cvv += wv[e] * (zero[3 - e] ? 0.f : tv[u][3 - e]);
      const float cqq = Silu(cq), ckk = Silu(ck), cv = Silu(cvv);
      // a/b chains of threads t0 and t1 (GdnKernel: i = t, t + 256, ...).
      float pa0 = 0.f, pb0 = 0.f, pa1 = 0.f, pb1 = 0.f;
      if constexpr (MUT == 7) {
#pragma unroll
        for (int k = KH - 1; k >= 0; --k) {
          pa0 += __half2float(wa0[k]) * x0[u][k];
          pb0 += __half2float(wb0[k]) * x0[u][k];
          pa1 += __half2float(wa1[k]) * x1[u][k];
          pb1 += __half2float(wb1[k]) * x1[u][k];
        }
      } else {
#pragma unroll
        for (int k = 0; k < KH; ++k) {
          pa0 += __half2float(wa0[k]) * x0[u][k];
          pb0 += __half2float(wb0[k]) * x0[u][k];
          pa1 += __half2float(wa1[k]) * x1[u][k];
          pb1 += __half2float(wb1[k]) * x1[u][k];
        }
      }
      // GdnKernel's wave w holds threads t0 (sq term cq^2, sk term 0), wave w + 4 threads t1 (sq 0, sk ck^2).
      const float sq = WaveSum(GdnVerifyOpaque(cqq * cqq));
      const float sk = WaveSum(GdnVerifyOpaque(ckk * ckk));
      pa0 = WaveSum(pa0);
      pb0 = WaveSum(pb0);
      pa1 = WaveSum(pa1);
      pb1 = WaveSum(pb1);
      if (lane == 0) {
        L.red[r][0][w] = sq;
        L.red[r][0][w + 4] = 0.f;
        L.red[r][1][w] = 0.f;
        L.red[r][1][w + 4] = sk;
        L.red[r][2][w] = pa0;
        L.red[r][2][w + 4] = pa1;
        L.red[r][3][w] = pb0;
        L.red[r][3][w + 4] = pb1;
      }
      L.u.raw[r][t0] = cqq;
      L.u.raw[r][t1] = ckk;
      L.v[a.replay + r][j] = cv;
    }
  }
  GdnVerifyBarrier();  // B1
  {
    const float A = -__expf(a.A_log[h]), dtb = a.dt_bias[h];
    for (int r = 0; r < a.R; ++r) {
      L.q[r][j] = L.u.raw[r][t0] * rsqrtf(GdnVerifySum8(L.red[r][0]) + kEps) * 0.08838834764831845f;
      L.k[a.replay + r][j] = L.u.raw[r][t1] * rsqrtf(GdnVerifySum8(L.red[r][1]) + kEps);
      if (t0 == r) {
        const float sp = GdnVerifySum8(L.red[r][2]) + dtb;
        const float softplus = sp > 20.f ? sp : log1pf(expf(sp));
        L.dec[a.replay + r][0] = expf(A * softplus);
        L.dec[a.replay + r][1] = Sigmoid(GdnVerifySum8(L.red[r][3]));
      }
    }
  }
  GdnVerifyBarrier();  // B2: the state waves run the recurrence from here
  // Pending rows of this forward (GdnKernel: pe[t] = ks[t], pe[128 + t] = cv, decay, beta).
  for (int r = a.commit; r < a.R; ++r) {
    float* pe = a.pend + ((size_t)h * kMaxRows + (r - a.commit)) * kPend;
    const int i = a.replay + r;
    pe[j] = L.k[i][j];
    pe[128 + j] = L.v[i][j];
    if (t0 == 0) {
      pe[256] = L.dec[i][0];
      pe[257] = L.dec[i][MUT == 3 ? 0 : 1];
    }
  }
  // Ring slots of this forward never alias the three slots any block reads.
  for (int r = 0; r < a.R; ++r) {
    float* hs = a.hist + (size_t)((a.pos0 + r) & 15) * kGdnQkvWidth;
    const float* q = a.qkv + (size_t)r * kGdnQkvWidth;
    hs[chv] = q[chv];
    if (MUT != 5 && h % 3 == 0) {
      hs[chq] = q[chq];
      hs[chk] = q[chk];
    }
  }
  // Output inputs of this wave's rows (w and w + 4), loaded while the recurrence runs.
  float zv[2][4], gv[4];
  __half sv[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    gv[m] = a.gamma[32 * m + lane];
#pragma unroll
    for (int e = 0; e < 2; ++e)
      zv[e][m] = a.z[(size_t)min(w + 4 * e, a.R - 1) * kGdnValueWidth + h * 128 + 32 * m + lane];
  }
  LinScaleLane(a.suh_out + h * 128, 4 * lane, sv);
  GdnVerifyBarrier();  // B3
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const int r = w + 4 * e;
    if (r >= a.R) break;
    // GdnKernel's tail for row r: its thread j < 128 (wave j / 32) is lane j % 32 of term m = j / 32 here.
    float y[4], s[8];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      y[m] = L.u.out.y[r][32 * m + lane];
      s[m] = WaveSum(GdnVerifyOpaque(y[m] * y[m]));
      s[4 + m] = 0.f;
    }
    const float yy = GdnVerifySum8(s);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float yn = y[m] * rsqrtf(MUT == 4 ? yy / 128.f : yy / 128.f + kEps) * gv[m];
      // RoundH here, as GdnKernel's RoundH(o): the compiler fuses the product and the fp16 rounding into one
      // v_fma_mixlo_f16 (one rounding). An fp32 store first rounds twice and changes about 1 in 6000 outputs.
      L.u.out.o[r][32 * m + lane] = RoundH(yn * Sigmoid(zv[e][m]));
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
    float o4[4], v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o4[i] = L.u.out.o[r][4 * lane + i];
    LinInLane(o4, sv, v);  // ob = RoundH(RoundH(o) * suh), Wht128
    __half* d = a.xh_out + (size_t)r * kGdnValueWidth + h * 128 + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = __float2half(v[i] * kHadNorm);
  }
}

// CRUCIBLE_GDN_VERIFY: the smallest row count that runs GdnVerifyKernel (0 = off, the default; 1..8). The engine reads
// it at start (Engine::gdn_verify_), so a bad value stops the engine there; the first call fixes the value.
inline int GdnVerifyMinRows() {
  static const int v = [] {
    const int n = ResolveGdnVerify(SwitchText("CRUCIBLE_GDN_VERIFY"));
    if (n < 0) {
      std::fprintf(stderr, "CRUCIBLE_GDN_VERIFY must be 0..%d (one digit)\n", kGdnVerifyMaxRows);
      std::exit(2);
    }
    return n;
  }();
  return v;
}
inline bool GdnVerifyOn(int rows) { return GdnVerifyUses(GdnVerifyMinRows(), rows); }

template <class Args>
inline void LaunchGdnVerify(const Args& a, unsigned spans, int state_cache, hipStream_t stream = 0) {
  const dim3 grid(kGdnHeads, spans);
  if (state_cache == 6) GdnVerifyKernel<4, true, Args><<<grid, kGdnVerifyThreads, 0, stream>>>(a);
  else if (state_cache == 4) GdnVerifyKernel<4, false, Args><<<grid, kGdnVerifyThreads, 0, stream>>>(a);
  else GdnVerifyKernel<0, false, Args><<<grid, kGdnVerifyThreads, 0, stream>>>(a);
}

}  // namespace crucible
