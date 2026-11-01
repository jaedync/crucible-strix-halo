// Generated reference contractions. Regenerate with tools/generate_persistent_ops.py.
// Only logical indices and compute-only synchronization differ from the source.
#pragma once
#include "ops.h"
#include "../gfx1151/persistent_sync.h"
#include "kernels/persistent_gdn_load.h"
namespace crucible {
template <int R>
__device__ __forceinline__ void PersistentHcDots(ComputeContext& ct, int task, const __half* fn, int M, const float* H, float* dots, float* ss) {
  const int w = task * 8 + (ct.t >> 5);
  const int lane = ct.t & 31;
  const int m = w >> 2, s = w & 3;
  if (m > M) return;
  if (m == M) {
    for (int r = 0; r < R; ++r) {
      const float* h = H + ((size_t)r * kStreams + s) * kHidden;
      float a = 0.f;
      for (int d = lane; d < kHidden; d += 32) a += h[d] * h[d];
      a = WaveSum(a);
      if (lane == 0) ss[r * kStreams + s] = a;
    }
    return;
  }
  const __half* wp = fn + (size_t)m * kWide + s * kHidden + lane * 8;
  uint4 raw[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) raw[k] = LoadNT(wp + k * 256);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
  for (int k = 0; k < 10; ++k) {
    const __half* hw = reinterpret_cast<const __half*>(&raw[k]);
    float wf[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wf[i] = __half2float(hw[i]);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float4* hr =
          reinterpret_cast<const float4*>(H + (size_t)r * kWide + s * kHidden + k * 256 + lane * 8);
      const float4 a = hr[0], b = hr[1];
      acc[r] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y +
                wf[6] * b.z + wf[7] * b.w;
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float v = WaveSum(acc[r]);
    if (lane == 0) dots[((size_t)r * M + m) * kStreams + s] = v;
  }
}

template <int R>
__device__ __forceinline__ void PersistentHcMix(ComputeContext& ct, int task, const __half* up, const float* w1, const float* dots, int M, const float* ss, const float* H, float* mixed, float* post) {
  __shared__ float t[R][kRank];
  __shared__ float inv[R][kStreams];
  if (ct.t < R * kStreams) {
    const int r = ct.t / kStreams, s = ct.t % kStreams;
    inv[r][s] = rsqrtf(ss[r * kStreams + s] / kHidden + kEps);
  }
  ComputeBarrier(ct);
  for (int i = ct.t; i < R * kRank; i += 256) {
    const int r = i / kRank, m = i % kRank;
    const float* dm = dots + ((size_t)r * M + m) * kStreams;
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < kStreams; ++s) a += inv[r][s] * dm[s];
    t[r][m] = Silu(a / kStreams);
  }
  if (task == 0 && post != nullptr && ct.t < R * kStreams) {
    const int r = ct.t / kStreams, s = ct.t % kStreams;
    const float* dm = dots + ((size_t)r * M + kRank + s) * kStreams;
    float a = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < kStreams; ++s2) a += inv[r][s2] * dm[s2];
    post[r * kStreams + s] = 2.f * Sigmoid(a / kStreams);
  }
  ComputeBarrier(ct);
  const int lane = ct.t & 31;
  const int s = lane >> 3, li = lane & 7;
  const int d0 = (task * 8 + (ct.t >> 5)) * 2;
  uint4 raw[2][5];
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const __half* w = up + ((size_t)s * kHidden + d0 + e) * kRank + li * 8;
#pragma unroll
    for (int k = 0; k < 5; ++k) raw[e][k] = LoadNT(w + k * 64);
  }
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const int d = d0 + e;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
    for (int k = 0; k < 5; ++k) {
      const __half* hw = reinterpret_cast<const __half*>(&raw[e][k]);
      const int c = k * 64 + li * 8;
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r] += __half2float(hw[i]) * t[r][c + i];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float g = acc[r];
      g += __shfl_xor(g, 1);
      g += __shfl_xor(g, 2);
      g += __shfl_xor(g, 4);
      const size_t hi = ((size_t)r * kStreams + s) * kHidden + d;
      float v = Sigmoid(g) * H[hi] * inv[r][s] * w1[s * kHidden + d];
      v += __shfl_xor(v, 8);
      v += __shfl_xor(v, 16);
      if (lane == 0) mixed[(size_t)r * kHidden + d] = RoundH(v / kStreams);
    }
  }
}

__device__ __forceinline__ void PersistentGdn(ComputeContext& ct, int task, GdnArgs a) {
#if CRUCIBLE_PERSISTENT_GDN_SERIAL
  a.R = 1; a.replay = 0; a.commit = 1;
#endif

  constexpr int CACHE = 0;
  constexpr bool STREAM_STORE = false;
  __shared__ float qs[128], ks[128], red[8], ob[128], part[2][128];
  const int h = task, t = ct.t, j = t & 127, hf = t >> 7, kh = h / 3;
  const int chq = kh * 128 + j, chk = 2048 + kh * 128 + j, chv = 4096 + h * 128 + j;
  float S[64];
  float* st = a.state + (size_t)h * 128 * 128 + (size_t)hf * 64 * 128;
  const auto state_rs = StreamRsrc(
      reinterpret_cast<const uint32_t*>(a.state + (size_t)h * 128 * 128), 128 * 128 * 4);
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    if constexpr (CACHE == 0) S[i] = st[i * 128 + j];
    else S[i] = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(
        state_rs, ((hf * 64 + i) * 128 + j) * 4, 0, CACHE));
  }
  const float A = -__expf(a.A_log[h]), dtb = a.dt_bias[h];
  const float g = a.gamma[j];
  // Lazy commit: accepted draft rows of the previous forward replay the exact
  // update they computed there, so rollback costs no state traffic and the
  // committed state is bitwise what serial decoding would hold.
  for (int i = 0; i < a.replay; ++i) {
    const float* pe = a.pend + ((size_t)h * kMaxRows + i) * kPend;
    if (t < 128) ks[t] = pe[t];
    ComputeBarrier(ct);
    const float decay = pe[256], beta = pe[257], cv = pe[128 + j];
    const float* kq = ks + hf * 64;
    float mem = 0.f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
      S[ii] *= decay;
      mem += S[ii] * PersistentCoefficient(kq + ii, mem);
    }
    part[hf][j] = mem;
    ComputeBarrier(ct);
    const float delta = beta * (cv - (part[0][j] + part[1][j]));
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) S[ii] += kq[ii] * delta;
    ComputeBarrier(ct);
  }
  for (int r = 0; r < a.R; ++r) {
    const int p = a.pos0 + r;
    auto tap = [&](int ch, int back) -> float {
      const int rr = r - back;
      if (rr >= 0) return a.qkv[(size_t)rr * kGdnQkvWidth + ch];
      if (p - back < 0) return 0.f;
      return a.hist[(size_t)((p - back) & 15) * kGdnQkvWidth + ch];
    };
    auto conv = [&](int ch) {
      const float* w = a.conv + (size_t)ch * 4;
      float c = 0.f;
#pragma unroll
      for (int u = 0; u < 4; ++u) c += w[u] * tap(ch, 3 - u);
      return Silu(c);
    };
    // Half 0 convolves the query channel, half 1 the key channel.
    const float cqk = conv(hf ? chk : chq);
    const float cv = conv(chv);
    float pa = 0.f, pb = 0.f;
    const float* xr = a.x + (size_t)r * kHidden;
    const __half* wa = a.ab + (size_t)h * kHidden;
    const __half* wb = a.ab + (size_t)(48 + h) * kHidden;
    for (int i = t; i < kHidden; i += 256) {
      pa += __half2float(wa[i]) * xr[i];
      pb += __half2float(wb[i]) * xr[i];
    }
    const float sq = PersistentBlockSum(hf ? 0.f : cqk * cqk, red, ct);
    const float sk = PersistentBlockSum(hf ? cqk * cqk : 0.f, red, ct);
    pa = PersistentBlockSum(pa, red, ct);
    pb = PersistentBlockSum(pb, red, ct);
    if (hf) ks[j] = cqk * rsqrtf(sk + kEps);
    else qs[j] = cqk * rsqrtf(sq + kEps) * 0.08838834764831845f;  // l2norm, then 1/sqrt(128)
    ComputeBarrier(ct);
    const float sp = pa + dtb;
    const float softplus = sp > 20.f ? sp : log1pf(expf(sp));
    const float decay = expf(A * softplus);
    const float beta = Sigmoid(pb);
    if (r >= a.commit) {
      float* pe = a.pend + ((size_t)h * kMaxRows + (r - a.commit)) * kPend;
      if (t < 128) {
        pe[t] = ks[t];
        pe[128 + t] = cv;
      }
      if (t == 0) {
        pe[256] = decay;
        pe[257] = beta;
      }
    }
    const float* kq = ks + hf * 64;
    const float* qq = qs + hf * 64;
    float mem = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      S[i] *= decay;
      mem += S[i] * PersistentCoefficient(kq + i, mem);
    }
    part[hf][j] = mem;
    ComputeBarrier(ct);
    const float delta = beta * (cv - (part[0][j] + part[1][j]));
    ComputeBarrier(ct);
    float y = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      S[i] += PersistentCoefficient(kq + i, y) * delta;
      y += S[i] * PersistentCoefficient(qq + i, y);
    }
    part[hf][j] = y;
    if (r == a.commit - 1) {
#pragma unroll
      for (int i = 0; i < 64; ++i) {
        if constexpr (!STREAM_STORE) st[i * 128 + j] = S[i];
        else __builtin_amdgcn_raw_buffer_store_b32(
            __float_as_uint(S[i]), state_rs, ((hf * 64 + i) * 128 + j) * 4, 0, CACHE);
      }
    }
    ComputeBarrier(ct);
    y = part[0][j] + part[1][j];
    const float yy = PersistentBlockSum(hf ? 0.f : y * y, red, ct);
    if (hf == 0) {
      const float yn = y * rsqrtf(yy / 128.f + kEps) * g;
      const float o = yn * Sigmoid(a.z[(size_t)r * kGdnValueWidth + h * 128 + j]);
      ob[j] = RoundH(RoundH(o) * __half2float(a.suh_out[h * 128 + j]));
    }
    ComputeBarrier(ct);
    if (t < 32) {
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = ob[t * 4 + i];
      Wht128(v);
      __half* d = a.xh_out + (size_t)r * kGdnValueWidth + h * 128 + t * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) d[i] = __float2half(v[i] * kHadNorm);
    }
    ComputeBarrier(ct);
  }
  // Ring slots of this forward never alias the three slots any block reads.
  if (hf == 0) {
    for (int r = 0; r < a.R; ++r) {
      float* hs = a.hist + (size_t)((a.pos0 + r) & 15) * kGdnQkvWidth;
      const float* q = a.qkv + (size_t)r * kGdnQkvWidth;
      hs[chv] = q[chv];
      if (h % 3 == 0) {
        hs[chq] = q[chq];
        hs[chk] = q[chk];
      }
    }
  }
}

__device__ __forceinline__ void PersistentLinIn(ComputeContext& ct, int task, const LinJob* jobs, const XfItem* items, int n) {
  const int w = task * 8 + (ct.t >> 5);
  if (w >= n) return;
  const int lane = ct.t & 31;
  const XfItem it = items[w];
  const LinJob& j = jobs[it.job];
  if (j.K == 0) return;  // unused routed slot this forward
  const int c0 = it.blk * 128 + lane * 4;
  const float* s = j.src + (size_t)it.r * j.lds + c0;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = RoundH(RoundH(s[i]) * __half2float(j.suh[c0 + i]));
  Wht128(v);
  __half* d = j.xh + (size_t)it.r * j.I + c0;
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = __float2half(v[i] * kHadNorm);
}

__device__ __forceinline__ void PersistentLinOut(ComputeContext& ct, int task, const LinJob* jobs, const XfItem* items, int n) {
  const int w = task * 8 + (ct.t >> 5);
  if (w >= n) return;
  const XfItem it = items[w];
  LinWriteBlock(jobs[it.job], it.r, it.blk);
}
}  // namespace crucible
