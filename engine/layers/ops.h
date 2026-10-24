// Non-EXL3 operators of the Qwen3.8-Flash-Next forward pass. Row r of a
// forward is absolute position pos0 + r of one sequence.
#pragma once

#include "../gfx1151/common.h"
#include "../exl3/exl3.h"
#include "cache_storage.h"
#include "packed_state.h"
#include "nonfinite_probe.h"
#include "ple_row_codec.h"

namespace crucible {

// ---- embedding ---------------------------------------------------------------

__global__ void EmbedKernel(const uint16_t* __restrict__ table, const int* __restrict__ tokens,
                            float* __restrict__ H, int ldh = kWide) {
  const int r = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const float e = Bf16(table[(size_t)tokens[r] * kHidden + d]);
#pragma unroll
  for (int s = 0; s < kStreams; ++s) H[(size_t)r * ldh + s * kHidden + d] = e;
}

// ---- dense fp16 matvec: y[r][o] = W[o] . x[r] ---------------------------------

// I = 2560: each lane streams 10 x 16 B of its row, all issued up front.
template <int R>
__global__ void __launch_bounds__(256) DenseGemvKernel(const __half* __restrict__ W, int O,
                                                       const float* __restrict__ x, int ldx,
                                                       float* __restrict__ y, int ldy) {
  PublishWeights(W);
  const int o = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (o >= O) return;
  const int lane = threadIdx.x & 31;
  const __half* w = W + (size_t)o * kHidden + lane * 8;
  uint4 raw[10];
#pragma unroll
  for (int k = 0; k < 10; ++k) raw[k] = LoadNT(w + k * 256);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
  for (int k = 0; k < 10; ++k) {
    const __half* h = reinterpret_cast<const __half*>(&raw[k]);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float4* xr = reinterpret_cast<const float4*>(x + (size_t)r * ldx + k * 256 + lane * 8);
      const float4 a = xr[0], b = xr[1];
      acc[r] += __half2float(h[0]) * a.x + __half2float(h[1]) * a.y + __half2float(h[2]) * a.z +
                __half2float(h[3]) * a.w + __half2float(h[4]) * b.x + __half2float(h[5]) * b.y +
                __half2float(h[6]) * b.z + __half2float(h[7]) * b.w;
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float v = WaveSum(acc[r]);
    if (lane == 0) y[(size_t)r * ldy + o] = v;
  }
}

// ---- hyper-connection read (gated residual) ----------------------------------
//
// fn = [down; inject] * (1 + hc_norm), folded fp16 [M][10240] (ExLlamaV3's fused
// form). Since N[s] = H[s] * inv_rms[s] * (1 + w[s]), fn . N splits into
// per-stream dots of fn with the RAW streams scaled by inv_rms afterwards, so
// the norm needs no separate pass.
//   dots[r][m][s] = sum_d fn[m][s*D + d] * H[r][s][d]    (m < M)
//   ss[r][s]      = sum_d H[r][s][d]^2                   (waves M..M+3)

// One wave per (row m, stream s): 2560 halves = 10 x 16 B per lane, all
// issued before any math so each wave keeps 160 B per lane in flight.
// Waves with m == M compute the per-stream sum of squares instead.
template <int R>
__global__ void __launch_bounds__(256) HcDotsKernel(const __half* __restrict__ fn, int M,
                                                    const float* __restrict__ H,
                                                    float* __restrict__ dots,
                                                    float* __restrict__ ss) {
  if constexpr (R <= 2) PublishWeights(fn);
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
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
  if constexpr (R == 3) PublishWeights(fn);
}

// mixed[r][d] = mean_s sigmoid(up[s*D+d] . t[r]) * N[r][s][d], with
// t = SiLU((down . N) / 4) rebuilt per block from the dots. One wave per d,
// 8 lanes per stream row of `up` (320 halves = 5 chunks of 8 per lane).
// Block 0 also writes the inject gates when M > kRank.
template <int R>
__global__ void __launch_bounds__(256) HcMixKernel(const __half* __restrict__ up,
                                                   const float* __restrict__ w1,
                                                   const float* __restrict__ dots, int M,
                                                   const float* __restrict__ ss,
                                                   const float* __restrict__ H,
                                                   float* __restrict__ mixed,
                                                   float* __restrict__ post) {
  PublishWeights(up);
  __shared__ float t[R][kRank];
  __shared__ float inv[R][kStreams];
  if (threadIdx.x < R * kStreams) {
    const int r = threadIdx.x / kStreams, s = threadIdx.x % kStreams;
    inv[r][s] = rsqrtf(ss[r * kStreams + s] / kHidden + kEps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < R * kRank; i += blockDim.x) {
    const int r = i / kRank, m = i % kRank;
    const float* dm = dots + ((size_t)r * M + m) * kStreams;
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < kStreams; ++s) a += inv[r][s] * dm[s];
    t[r][m] = Silu(a / kStreams);
  }
  if (blockIdx.x == 0 && post != nullptr && threadIdx.x < R * kStreams) {
    const int r = threadIdx.x / kStreams, s = threadIdx.x % kStreams;
    const float* dm = dots + ((size_t)r * M + kRank + s) * kStreams;
    float a = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < kStreams; ++s2) a += inv[r][s2] * dm[s2];
    post[r * kStreams + s] = 2.f * Sigmoid(a / kStreams);
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int s = lane >> 3, li = lane & 7;
  const int d0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * 2;
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

// ---- Gated DeltaNet ------------------------------------------------------------

struct GdnArgs {
  const float* qkv;   // [R][10240] raw projection
  const float* z;     // [R][6144]
  const float* x;     // [R][2560] mixed input, for the a/b projections
  const __half* ab;   // [96][2560]: rows 0..47 a, 48..95 b
  const float* conv;  // [10240][4]
  const float* A_log;
  const float* dt_bias;
  const float* gamma;     // [128] direct gamma
  float* hist;            // [16][10240] raw qkv ring, slot = pos & 15
  float* state;           // [48][128 key][128 value] fp32
  const __half* suh_out;  // [6144] out_proj input scale
  __half* xh_out;         // [R][6144] transformed out_proj input
  float* pend;            // [48][kMaxRows][kPend] speculative rows: khat, v, decay, beta
  int pos0, R;
  int replay;             // pending rows of the previous forward to fold in first
  int commit;             // rows of this forward whose state update is certain
  int pack_rows = 0, pack_off = 0;  // prefill: xh_out in the packed GEMM layout
};

// Out-projection input index: row-major [R][6144], or (pack_rows > 0) the
// packed GEMM layout [384][pack_rows][16] at row pack_off + r.
__host__ __device__ __forceinline__ size_t OutProjIndex(int r, int c, int pack_rows, int pack_off) {
  return pack_rows ? ((size_t)(c >> 4) * pack_rows + pack_off + r) * 16 + (c & 15)
                   : (size_t)r * kMixerOutWidth + c;
}

constexpr int kPend = 258;

// One block per value head, two threads per state column: thread t owns
// column j = t % 128, rows [64 hf, 64 hf + 64) with hf = t / 128, so the
// column half stays in 64 registers without spilling.
__device__ __forceinline__ GdnArgs GdnRowArgs(GdnArgs a, unsigned) { return a; }
__device__ __forceinline__ GdnArgs GdnRowArgs(PackedStateArgs<GdnArgs, kMaxRows> a, unsigned row) {
  return a.rows[row];
}

template <int CACHE = 0, bool STREAM_STORE = false, class Args = GdnArgs>
__global__ void __launch_bounds__(256) GdnKernel(Args args) {
  const GdnArgs a = GdnRowArgs(args, blockIdx.y);
  PublishWeights(a.ab);
  __shared__ float qs[128], ks[128], red[8], ob[128], part[2][128];
  const int h = blockIdx.x, t = threadIdx.x, j = t & 127, hf = t >> 7, kh = h / 3;
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
    __syncthreads();
    const float decay = pe[256], beta = pe[257], cv = pe[128 + j];
    const float* kq = ks + hf * 64;
    float mem = 0.f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
      S[ii] *= decay;
      mem += S[ii] * kq[ii];
    }
    part[hf][j] = mem;
    __syncthreads();
    const float delta = beta * (cv - (part[0][j] + part[1][j]));
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) S[ii] += kq[ii] * delta;
    __syncthreads();
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
    const float sq = BlockSum<256>(hf ? 0.f : cqk * cqk, red);
    const float sk = BlockSum<256>(hf ? cqk * cqk : 0.f, red);
    pa = BlockSum<256>(pa, red);
    pb = BlockSum<256>(pb, red);
    if (hf) ks[j] = cqk * rsqrtf(sk + kEps);
    else qs[j] = cqk * rsqrtf(sq + kEps) * 0.08838834764831845f;  // l2norm, then 1/sqrt(128)
    __syncthreads();
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
      mem += S[i] * kq[i];
    }
    part[hf][j] = mem;
    __syncthreads();
    const float delta = beta * (cv - (part[0][j] + part[1][j]));
    __syncthreads();
    float y = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      S[i] += kq[i] * delta;
      y += S[i] * qq[i];
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
    __syncthreads();
    y = part[0][j] + part[1][j];
    const float yy = BlockSum<256>(hf ? 0.f : y * y, red);
    if (hf == 0) {
      const float yn = y * rsqrtf(yy / 128.f + kEps) * g;
      const float o = yn * Sigmoid(a.z[(size_t)r * kGdnValueWidth + h * 128 + j]);
      ob[j] = RoundH(RoundH(o) * __half2float(a.suh_out[h * 128 + j]));
    }
    __syncthreads();
    if (t < 32) {
      float v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = ob[t * 4 + i];
      Wht128(v);
      __half* d = a.xh_out + (size_t)r * kGdnValueWidth + h * 128 + t * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) d[i] = __float2half(v[i] * kHadNorm);
    }
    __syncthreads();
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

// ---- QSA sparse attention -----------------------------------------------------

// The QSA geometry (heads, dims, RoPE, kSelBlocks) is in geometry.h.

struct QsaCache {
  __half* k;        // [ctx][2][256]
  __half* v;        // [ctx][2][256]
  float* ik_raw;    // [ctx][128], or a retained raw ring
  float* ik_pool;   // [ceil(ctx/128)][128][32], fp32, block-major tiles
  int raw_slots = 0;  // zero keeps the full-prefix layout
};

#define CRUCIBLE_QSA_POOL_TILED 1
__host__ __device__ constexpr size_t QsaPoolOffset(int b, int d) {
  return (size_t)(b/32)*4096 + d*32 + b%32;
}

constexpr size_t QsaPoolFloats(int ctx) { return (size_t)((ctx+127)/128)*4096; }

struct QsaArgs {
  float* q;          // [R][24][512] q_proj output: per head [q 256 | gate 256]; q normed in place
  const float* k;    // [R][512]
  const float* v;    // [R][512]
  float* idx;        // [R][640]: 4x128 query (normed in place) | 128 raw key
  const float* q_norm1;
  const float* k_norm1;   // (1 + w), [256]
  const float* iq_norm1;
  const float* ik_norm1;  // (1 + w), [128]
  QsaCache c;
  float* scores;     // [R][sstride]
  int sstride;
  int* sel;          // [R][kSelBlocks] selected block ids, ascending
  int* nsel;         // [R]: -1 = dense (attend all), else count
  const __half* suh_o;
  __half* xh_o;      // [R][6144]
  int pos0, R;
  float* context = nullptr;  // optional unrounded [R][24][256], for numerical validation
  int pack_rows = 0, pack_off = 0;  // prefill: xh_o in the packed GEMM layout
  // Multimodal RoPE positions [ctx][4] = (T, H, W, 0) for a request with images; null keeps p.
  const int* mrope = nullptr;
};

// Packed verification: one launch serves every span of a verify pass. The span is blockIdx.z (or z / splits);
// each block reads only its span's arguments, so the arithmetic equals one launch per span. base is the span's
// first row in the pass, for scratch laid out by row. Blocks past a span's rows exit at once.
struct PackedQsaArgs {
  QsaArgs rows[kMaxRows]{};
  int base[kMaxRows]{};
};
template <class Args> constexpr bool kPackedQsa = false;
template <> constexpr bool kPackedQsa<PackedQsaArgs> = true;
__device__ __forceinline__ QsaArgs QsaSpanArgs(const QsaArgs& a, unsigned) { return a; }
__device__ __forceinline__ QsaArgs QsaSpanArgs(const PackedQsaArgs& p, unsigned s) { return p.rows[s]; }
__device__ __forceinline__ int QsaSpanBase(const QsaArgs&, unsigned) { return 0; }
__device__ __forceinline__ int QsaSpanBase(const PackedQsaArgs& p, unsigned s) { return p.base[s]; }

// Split-half partial RoPE over channels 0..63 of x (in LDS), position p.
__device__ __forceinline__ void Rope64(float* x, int p) {
  const int j = threadIdx.x;
  float c = 0.f, s = 0.f, a = 0.f, b = 0.f;
  if (j < kRotDim / 2) {
    const float inv = powf(kRopeTheta, -2.f * j / kRotDim);
    const float ang = (float)p * inv;
    c = cosf(ang);
    s = sinf(ang);
    a = x[j];
    b = x[j + 32];
  }
  __syncthreads();
  if (j < kRotDim / 2) {
    x[j] = a * c - b * s;
    x[j + 32] = b * c + a * s;
  }
  __syncthreads();
}

// Interleaved multimodal sections [11, 11, 10]: frequency j uses H if j % 3 == 1 (j < 33), W if
// j % 3 == 2 (j < 30), else T (HF Qwen4ExpTextRotaryEmbedding). Equal axes give Rope64 exactly.
__device__ __forceinline__ void Rope64Axes(float* x, int pt, int ph, int pw) {
  const int j = threadIdx.x;
  float c = 0.f, s = 0.f, a = 0.f, b = 0.f;
  if (j < kRotDim / 2) {
    const int p = j % 3 == 1 && j < 33 ? ph : j % 3 == 2 && j < 30 ? pw : pt;
    const float inv = powf(kRopeTheta, -2.f * j / kRotDim);
    const float ang = (float)p * inv;
    c = cosf(ang);
    s = sinf(ang);
    a = x[j];
    b = x[j + 32];
  }
  __syncthreads();
  if (j < kRotDim / 2) {
    x[j] = a * c - b * s;
    x[j + 32] = b * c + a * s;
  }
  __syncthreads();
}

// RoPE for cache position p: the text path is unchanged unless the request has images.
__device__ __forceinline__ void RopeAt(float* x, const int* mrope, int p) {
  if (!mrope) Rope64(x, p);
  else Rope64Axes(x, mrope[4 * p], mrope[4 * p + 1], mrope[4 * p + 2]);
}

// Centered RMSNorm of n (<= 256) values in LDS, one per thread.
__device__ __forceinline__ void CNormLds(float* x, int n, const float* w1, float* red) {
  const int j = threadIdx.x;
  const float v = j < n ? x[j] : 0.f;
  const float ss = BlockSum<256>(v * v, red);
  if (j < n) x[j] = v * rsqrtf(ss / n + kEps) * w1[j];
  __syncthreads();
}

__device__ __forceinline__ float QsaRawValue(const QsaArgs& a, int position, int dimension);
template <int R>
__device__ __forceinline__ void QsaPoolTail(const QsaArgs& a, float own, float* x, float* red);

// Blocks (unit, r): units 0..23 query heads, 24..25 key heads, 26..27 value
// heads, 28..31 indexer query heads, 32 indexer raw key. U0 = 24 runs the
// cache units only (grid x 9), for the prefill query kernel.
// Tail (R = 1 decode): unit 32 also runs QsaPoolKernel and,
// at short context, QsaTopKernel's result, so neither is launched.
template <class Args = QsaArgs, int U0 = 0, bool Tail = false>
__global__ void __launch_bounds__(256) QsaPrepKernel(Args args) {
  const QsaArgs a = QsaSpanArgs(args, blockIdx.z);
  if constexpr (kPackedQsa<Args>) if ((int)blockIdx.y >= a.R) return;
  PublishWeights(a.q_norm1);
  __shared__ float x[256], red[8];
  const int u = blockIdx.x + U0, r = blockIdx.y, j = threadIdx.x, p = a.pos0 + r;
  if (u < 24) {
    float* q = a.q + ((size_t)r * kQHeads + u) * 512;
    x[j] = q[j];
    __syncthreads();
    CNormLds(x, 256, a.q_norm1, red);
    RopeAt(x, a.mrope, p);
    q[j] = x[j];
  } else if (u < 26) {
    const int g = u - 24;
    x[j] = a.k[(size_t)r * 512 + g * 256 + j];
    __syncthreads();
    CNormLds(x, 256, a.k_norm1, red);
    RopeAt(x, a.mrope, p);
    a.c.k[((size_t)p * kKvHeads + g) * kHeadDim + j] = __float2half(x[j]);
  } else if (u < 28) {
    const int g = u - 26;
    a.c.v[((size_t)p * kKvHeads + g) * kHeadDim + j] =
        __float2half(a.v[(size_t)r * 512 + g * 256 + j]);
  } else if (u < 32) {
    float* iq = a.idx + (size_t)r * kQsaIdxWidth + (u - 28) * kIdxDim;
    if (j < kIdxDim) x[j] = iq[j];
    __syncthreads();
    CNormLds(x, kIdxDim, a.iq_norm1, red);
    RopeAt(x, a.mrope, p);
    if (j < kIdxDim) iq[j] = x[j];
  } else if constexpr (Tail) {
    // QsaRawUsesChunk is false at R = 1 (kQsaRawDecodeRows), so the raw key always goes to the cache.
    const float own = j < kIdxDim ? a.idx[512 + j] : 0.f;
    if (j < kIdxDim) a.c.ik_raw[QsaRawOffset(a.c.raw_slots, p, j)] = own;
    QsaPoolTail<1>(a, own, x, red);
  } else if (j < kIdxDim && !QsaRawUsesChunk(a.c.raw_slots, a.R)) {
    a.c.ik_raw[QsaRawOffset(a.c.raw_slots, p, j)] = a.idx[(size_t)r * kQsaIdxWidth + 512 + j];
  }
}

__device__ __forceinline__ float QsaRawValue(const QsaArgs& a, int position, int dimension) {
  if (QsaRawFromChunk(a.c.raw_slots, a.R, a.pos0, position))
    return a.idx[(size_t)(position - a.pos0) * kQsaIdxWidth + 512 + dimension];
  return a.c.ik_raw[QsaRawOffset(a.c.raw_slots, position, dimension)];
}

// Bulk preparation must not overwrite the old tail before every pool reads it.
__global__ void QsaRetainRawKernel(QsaArgs a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int keep = min(a.R, a.c.raw_slots);
  if (i >= keep * kIdxDim) return;
  const int row = a.R - keep + i / kIdxDim, dimension = i % kIdxDim;
  a.c.ik_raw[QsaRawOffset(a.c.raw_slots, a.pos0 + row, dimension)] =
      a.idx[(size_t)row * kQsaIdxWidth + 512 + dimension];
}
inline void LaunchQsaRetainRaw(const QsaArgs& a, hipStream_t stream = nullptr) {
  if (QsaRawUsesChunk(a.c.raw_slots, a.R))
    QsaRetainRawKernel<<<(a.c.raw_slots * kIdxDim + 255) / 256, 256, 0, stream>>>(a);
}

// Pools every 4-token block completed by this forward's rows. One block.
template <class Args = QsaArgs>
__global__ void __launch_bounds__(256) QsaPoolKernel(Args args) {
  const QsaArgs a = QsaSpanArgs(args, blockIdx.z);
  __shared__ float x[256], red[8];
  const int j = threadIdx.x;
  for (int r = 0; r < a.R; ++r) {
    const int p = a.pos0 + r;
    if ((p + 1) % 4 != 0) continue;
    const int blk = p / 4;
    if (j < kIdxDim) {
      float m = 0.f;
#pragma unroll
      for (int t = 0; t < 4; ++t) m += QsaRawValue(a, blk * 4 + t, j);
      x[j] = RoundH(m * 0.25f);
    }
    __syncthreads();
    CNormLds(x, kIdxDim, a.ik_norm1, red);
    RopeAt(x, a.mrope, blk * 4);
    if (j < kIdxDim) a.c.ik_pool[QsaPoolOffset(blk, j)] = x[j];
    __syncthreads();
  }
}

// QsaPoolKernel's row and QsaTopKernel's short-context result for a one-row
// forward, in Prep's unit-32 block. own: this thread's raw key value at pos0,
// which the pool reads from the cache in QsaPoolKernel (the same float).
template <int R>
__device__ __forceinline__ void QsaPoolTail(const QsaArgs& a, float own, float* x, float* red) {
  static_assert(R == 1, "rows of other blocks are not visible here");
  const int j = threadIdx.x, p = a.pos0;
  if (j == 0 && (p + 1) / 4 <= kSelBlocks) a.nsel[0] = -1;
  if ((p + 1) % 4 != 0) return;
  const int blk = p / 4;
  __syncthreads();  // x, red: the unit-32 block has not used them yet; keeps the pool's barrier pattern
  if (j < kIdxDim) {
    float m = 0.f;
#pragma unroll
    for (int t = 0; t < 3; ++t) m += QsaRawValue(a, blk * 4 + t, j);
    m += own;  // t = 3 is position p
    x[j] = RoundH(m * 0.25f);
  }
  __syncthreads();
  CNormLds(x, kIdxDim, a.ik_norm1, red);
  RopeAt(x, a.mrope, blk * 4);
  if (j < kIdxDim) a.c.ik_pool[QsaPoolOffset(blk, j)] = x[j];
}

// ---- MoE ------------------------------------------------------------------------

struct ExpertPtrs {
  const uint32_t* tiles[3];  // gate, up, down
  const __half* suh[3];
  const __half* svh[3];
  int K[3];
};

// Routes each row: top-10 of 512 router logits (logit desc, id asc) by
// direct ranking (each thread counts the logits that beat its own), then
// softmax over the selection. Slot kTopK of wts is the shared-expert gate.
__global__ void __launch_bounds__(512) MoeRouteKernel(const float* __restrict__ logits,
                                                      int* __restrict__ sel,
                                                      float* __restrict__ wts) {
  __shared__ float lv[kExperts];
  __shared__ float sl[kTopK];
  const int r = blockIdx.x, e = threadIdx.x;
  const float* lg = logits + (size_t)r * (kExperts + 1);
  const float v = lg[e];
#if CRUCIBLE_DEBUG_NONFINITE & 1
  if (NanOrPosInf(v)) FlagNonfinite(kNfDecodeRoute);
#endif
  lv[e] = v;
  __syncthreads();
  int rank = 0;
  for (int i = 0; i < kExperts; ++i) {
    const float o = lv[i];
    rank += (o > v) || (o == v && i < e);
  }
  if (rank < kTopK) {
    sel[r * kTopK + rank] = e;
    sl[rank] = v;
  }
  __syncthreads();
  if (e < kTopK) {
    float sum = 0.f;
    for (int k = 0; k < kTopK; ++k) sum += __expf(sl[k] - sl[0]);
    wts[r * kSlots + e] = __expf(sl[e] - sl[0]) / sum;
  }
  if (e == 0) wts[r * kSlots + kTopK] = Sigmoid(lg[kExperts]);
}

// ---- PLE (layer 1) ------------------------------------------------------------

// The row codec (ple_row_codec.h) packs symbols LSB-first, unlike the linear tile codec.
// Packed: table holds the gathered rows in [R][16] order (CRUCIBLE_PLE_HOST).
template <int K, bool Packed = false>
__global__ void __launch_bounds__(160) PleGatherKernel(const uint16_t* __restrict__ table,
                                                       const int64_t* __restrict__ rows,
                                                       const __half* __restrict__ head_bias,
                                                       float* __restrict__ out) {
  const int hd = blockIdx.x, r = blockIdx.y, i = threadIdx.x;
  constexpr int Words = 10 * K;
  const size_t index = Packed ? (size_t)r * 16 + hd : (size_t)rows[r * 16 + hd];
  const uint16_t* row = table + index * (Words + 1);
  const uint32_t state = PleRowState<K>(row, i);
  const uint32_t p = state * kMul1;
  const uint32_t bs = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24);
  const __half hv = __hfma(__ushort_as_half((unsigned short)(0x6400 + bs)), __ushort_as_half(0x1eee),
                           __ushort_as_half(0xc931));
  const float scale = __half2float(__ushort_as_half(row[0]));
  out[(size_t)r * kHidden + hd * 160 + i] =
      RoundH(__half2float(hv) * scale + __half2float(head_bias[hd * 160 + i]));
}

// CRUCIBLE_PLE_BF16: table holds the gathered BF16 rows of the original weights in [R][16] order.
// The original model adds no head bias (head_bias belongs to the quantizer's row codec), so a value
// is the BF16 number. The fp16 rounding keeps the values that the packed prefill GEMM reads as fp16,
// as in the quantized kernel, so decode and prefill rows stay equal.
__global__ void __launch_bounds__(kPleRowDim) PleBf16GatherKernel(const uint16_t* __restrict__ table,
                                                                  float* __restrict__ out) {
  const int hd = blockIdx.x, r = blockIdx.y, i = threadIdx.x;
  const uint32_t bits = (uint32_t)table[((size_t)r * kPleHeads + hd) * kPleRowDim + i] << 16;
  out[(size_t)r * kHidden + hd * kPleRowDim + i] = RoundH(__uint_as_float(bits));
}

struct PleArgs {
  const float* kv;        // [R][12800]: key 10240 | value 2560
  const float* norm_key1, *norm_query1, *norm_conv1;  // (1 + w), [10240]
  const float* conv;      // [10240][4]
  float* ring;            // [16][10240] normalized gated values, slot = pos & 15
  float* H;
  int pos0, R;
  int ldh = kWide;
};

// One block per stream; each thread owns 5 channels for every row in order, so
// the dilated conv history needs no cross-block synchronization.
__global__ void __launch_bounds__(512) PleMixKernel(PleArgs a) {
  __shared__ float red[16];
  const int s = blockIdx.x, j = threadIdx.x;
  constexpr int C = 5;
  for (int r = 0; r < a.R; ++r) {
    const int p = a.pos0 + r;
    const float* key = a.kv + (size_t)r * 12800 + s * kHidden;
    const float* val = a.kv + (size_t)r * 12800 + kWide;
    float* hs = a.H + (size_t)r * a.ldh + s * kHidden;
    float kk[C], qq[C];
    float sk = 0.f, sq = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const int d = j + c * 512;
      kk[c] = key[d];
      qq[c] = hs[d];
      sk += kk[c] * kk[c];
      sq += qq[c] * qq[c];
    }
    sk = BlockSum<512>(sk, red);
    sq = BlockSum<512>(sq, red);
    const float ik = rsqrtf(sk / kHidden + kEps), iqn = rsqrtf(sq / kHidden + kEps);
    float dot = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const int ch = s * kHidden + j + c * 512;
      dot += (kk[c] * ik * a.norm_key1[ch]) * (qq[c] * iqn * a.norm_query1[ch]);
    }
    dot = BlockSum<512>(dot, red) * 0.019764235376052372f;  // 1/sqrt(2560)
    const float sg = dot > 0.f ? 1.f : (dot < 0.f ? -1.f : 0.f);
    const float gate = Sigmoid(sg * sqrtf(fmaxf(fabsf(dot), 1e-6f)));
    float gv[C], sn = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      gv[c] = gate * val[j + c * 512];
      sn += gv[c] * gv[c];
    }
    sn = BlockSum<512>(sn, red);
    const float in = rsqrtf(sn / kHidden + kEps);
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const int ch = s * kHidden + j + c * 512;
      const float nv = RoundH(gv[c] * in * a.norm_conv1[ch]);
      a.ring[(size_t)(p & 15) * kWide + ch] = nv;
      const float* w = a.conv + (size_t)ch * 4;
      float cv = 0.f;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int src = p - 9 + 3 * t;
        if (src >= 0) cv += w[t] * a.ring[(size_t)(src & 15) * kWide + ch];
      }
      hs[j + c * 512] += gv[c] + Silu(cv);
    }
  }
}

// ---- MTP input fusion ------------------------------------------------------------

// E = CN(embed[tok], 1 + w_e) over 2560 and N = CN(stack, 1 + w_h) over all
// 10240 features (one full-width norm, not grouped). One block per row.
__global__ void __launch_bounds__(1024) MtpPrepKernel(const uint16_t* __restrict__ embed,
                                                      const int* __restrict__ tok,
                                                      const float* __restrict__ stack,
                                                      const float* __restrict__ ne1,
                                                      const float* __restrict__ nh1,
                                                      float* __restrict__ E,
                                                      float* __restrict__ N) {
  __shared__ float red[32];
  const int r = blockIdx.x, t = threadIdx.x;
  const uint16_t* er = embed + (size_t)tok[r] * kHidden;
  float se = 0.f, sh = 0.f;
  for (int d = t; d < kHidden; d += 1024) {
    const float v = Bf16(er[d]);
    se += v * v;
  }
  const float* hr = stack + (size_t)r * kWide;
  for (int c = t; c < kWide; c += 1024) sh += hr[c] * hr[c];
  se = BlockSum<1024>(se, red);
  sh = BlockSum<1024>(sh, red);
  const float ie = rsqrtf(se / kHidden + kEps), ih = rsqrtf(sh / kWide + kEps);
  for (int d = t; d < kHidden; d += 1024) E[(size_t)r * kHidden + d] = Bf16(er[d]) * ie * ne1[d];
  for (int c = t; c < kWide; c += 1024) N[(size_t)r * kWide + c] = hr[c] * ih * nh1[c];
}

// MtpPrepKernel for the bulk MTP warm-up (mtp_bulk.h): row 0's stack is `first` when set, and the other rows are
// consecutive rows of `rows` (row 0, or row 1 after `first`).
__global__ void __launch_bounds__(1024) MtpPrepPairsKernel(const uint16_t* __restrict__ embed,
                                                           const int* __restrict__ tok,
                                                           const float* __restrict__ first,
                                                           const float* __restrict__ rows,
                                                           const float* __restrict__ ne1,
                                                           const float* __restrict__ nh1,
                                                           float* __restrict__ E,
                                                           float* __restrict__ N) {
  __shared__ float red[32];
  const int r = blockIdx.x, t = threadIdx.x;
  const uint16_t* er = embed + (size_t)tok[r] * kHidden;
  float se = 0.f, sh = 0.f;
  for (int d = t; d < kHidden; d += 1024) {
    const float v = Bf16(er[d]);
    se += v * v;
  }
  const float* hr = first ? (r == 0 ? first : rows + (size_t)(r - 1) * kWide) : rows + (size_t)r * kWide;
  for (int c = t; c < kWide; c += 1024) sh += hr[c] * hr[c];
  se = BlockSum<1024>(se, red);
  sh = BlockSum<1024>(sh, red);
  const float ie = rsqrtf(se / kHidden + kEps), ih = rsqrtf(sh / kWide + kEps);
  for (int d = t; d < kHidden; d += 1024) E[(size_t)r * kHidden + d] = Bf16(er[d]) * ie * ne1[d];
  for (int c = t; c < kWide; c += 1024) N[(size_t)r * kWide + c] = hr[c] * ih * nh1[c];
}

// MtpFuseKernel into a strided H (the bulk prefill layout).
__global__ void MtpFuseStrideKernel(const float* __restrict__ hp, const float* __restrict__ ep,
                                    float* __restrict__ H, int ldh) {
  const int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  H[(size_t)r * ldh + c] = hp[(size_t)r * kWide + c] + ep[(size_t)r * kHidden + c % kHidden];
}

// H[r][s][d] = fc_hidden(N[s])[d] + fc_embedding(E)[d].
__global__ void MtpFuseKernel(const float* __restrict__ hp, const float* __restrict__ ep,
                              float* __restrict__ H) {
  const int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  H[(size_t)r * kWide + c] = hp[(size_t)r * kWide + c] + ep[(size_t)r * kHidden + c % kHidden];
}

// ---- sampling --------------------------------------------------------------------

// The greedy head's ID for a row with a NaN or +Inf logit. sampling::RequireFiniteHeadIds turns it into an error.
inline constexpr int kNonfiniteId = -1;

// Writes kNonfiniteId for a row with a NaN or +Inf logit: `>` skips a NaN, so without the flag such
// a row gave its best finite token, or token 0, with no error. -Inf stays a masked logit, as in sampling.h.
// The flag does not change b or i0, so a finite row gives the same ID as before.
__global__ void __launch_bounds__(1024) ArgmaxKernel(const float* __restrict__ logits, int V,
                                                     int* __restrict__ out) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const float* lg = logits + (size_t)r * V;
  float b = -INFINITY;
  int i0 = 0;
  bool bad = false;
  for (int i = threadIdx.x; i < V; i += 1024) {
    const float x = lg[i];
    bad |= !(x < INFINITY);  // NaN or +Inf
    if (x > b) {
      b = x;
      i0 = i;
    }
  }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const float ov = __shfl_xor(b, m);
    const int oi = __shfl_xor(i0, m);
    if (ov > b || (ov == b && oi < i0)) {
      b = ov;
      i0 = oi;
    }
  }
  if (lane == 0) {
    bv[w] = b;
    bi[w] = i0;
  }
  const bool row_bad = __syncthreads_or(bad);
  if (threadIdx.x == 0) {
    b = bv[0];
    i0 = bi[0];
    for (int i = 1; i < 32; ++i)
      if (bv[i] > b || (bv[i] == b && bi[i] < i0)) {
        b = bv[i];
        i0 = bi[i];
      }
    out[r] = row_bad ? kNonfiniteId : i0;
  }
}

}  // namespace crucible
