// GDN prefill scan + gate. Two parts:
// 1. Probes: copies of GdnScanSgprKernel with one cost removed (timing only; the values are wrong),
//    to see what bounds the scan. P_D2 is exact (a second copy of d for VOPD pairing).
// 2. GdnScanGateKernel: the SGPR scan with the gate (GdnLanesGateKernel<true>) done inside it, lagged
//    by 3 groups of 16 rows. y, xh and the state must be equal bit for bit to the engine path.
// Usage: gdn_pipe [--pairs N] [rows ...]  (timing at rows >= 2048 uses one 2,048-row part)
#include "prefill_common.h"
#include "../engine/layers/kernels/prefill_gdn.h"
#include <cstring>
#include <string>
using namespace crucible;
using namespace crucible::prefill;
using namespace prefill_bench;
namespace {
constexpr int kCore = 2048;
enum { P_NOQ = 1, P_NOSYNC = 2, P_NODIV = 4, P_D2 = 8 };

template <int M>
__global__ void __launch_bounds__(128) ScanProbe(GdnPrefill a, const float* __restrict__ kv) {
  __shared__ float4 memx[2][32], yx[2][32];
  __shared__ float vx[2][4][32];
  const int h=blockIdx.x>>2, lane=threadIdx.x&31, sub=__builtin_amdgcn_readfirstlane(threadIdx.x>>5);
  const int j=(blockIdx.x&3)*32+lane, ch=4096+h*128+j, R=a.input.R;
  float cw[4];
#pragma unroll
  for (int tap=0;tap<4;++tap) cw[tap]=a.input.conv[ch*4+tap];
  float S[32];
  float* dst=a.input.state+(size_t)h*128*128+(size_t)sub*32*128+j;
#pragma unroll
  for (int i=0;i<32;++i) S[i]=dst[i*128];
  const size_t rows=(size_t)a.chunks*64, base=(size_t)h*rows, kbase=(size_t)(h/3)*rows;
  float scale=1.f, pscale=1.f, raw[4], kb[2][32], qb[2][32];
  auto load=[&](int r0) {
    const int rs=r0+sub;
#pragma unroll
    for (int tap=0;tap<4;++tap) {
      const int rr=rs-3+tap,p=a.input.pos0+rr;
      raw[tap]=rs>=R ? 0.f : rr>=0 ? a.input.qkv[(size_t)rr*kGdnQkvWidth+ch]
          : p>=0 ? a.input.hist[(size_t)(p&15)*kGdnQkvWidth+ch] : 0.f;
    }
  };
  auto conv=[&](int r0,int slot) {
    if (r0+sub>=R) return;
    float x=0.f;
#pragma unroll
    for (int i=0;i<4;++i) x+=cw[i]*raw[i];
    vx[slot][sub][lane]=Silu(x);
  };
  auto kl=[&](float (&kd)[32],int r) {
    const float* kp=kv+(kbase+r)*128+sub*32;
#pragma unroll
    for (int i=0;i<32;++i) kd[i]=kp[i];
  };
  auto qreal=[&](float (&qd)[32],int r) {
    const float4* qp=reinterpret_cast<const float4*>(a.q+(kbase+r)*128+sub*32);
#pragma unroll
    for (int i=0;i<8;++i) { const float4 t=qp[i]; qd[4*i]=t.x; qd[4*i+1]=t.y; qd[4*i+2]=t.z; qd[4*i+3]=t.w; }
  };
  auto ql=[&](float (&qd)[32],int r) { if constexpr (!(M&P_NOQ)) qreal(qd,r); };
  if constexpr (M&P_NOQ) { qreal(qb[0],0); qreal(qb[1],1); }
  auto barrier=[] { if constexpr (!(M&P_NOSYNC)) asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); };
  load(0); conv(0,0); load(4); kl(kb[0],0);
  float gn=R>(lane&3) ? a.g[base+(lane&3)] : 0.f, bn=R>(lane&3) ? a.beta[base+(lane&3)] : 0.f;
  __syncthreads();
  auto step=[&](auto U,int r0,int slot,float dq,float bq) {
    constexpr int u=decltype(U)::value, P=u&1;
    const int r=r0+u;
    const float decay=GdnQuadBroadcast<u>(dq), beta=GdnQuadBroadcast<u>(bq);
    float (&k)[32]=kb[P]; float (&q)[32]=qb[P];
    scale*=decay;
    if (scale<0x1p-20f) {
#pragma unroll
      for (int i=0;i<32;++i) S[i]*=scale;
      scale=1.f;
    }
    float m0=0,m1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) { m0+=S[i]*k[i];m1+=S[i+1]*k[i+1]; }
    float mem;
    if constexpr (M&P_NOSYNC) {
      mem=m0+m1;
    } else {
      reinterpret_cast<float*>(&memx[r&1][lane])[sub]=m0+m1;
      barrier();
      if (sub==0 && r>0) {
        const float4 t=yx[(r-1)&1][lane];
        a.y[(base+r-1)*128+j]=((t.x+t.y)+(t.z+t.w))*pscale;
      }
      const float4 m4=memx[r&1][lane];
      mem=(m4.x+m4.y)+(m4.z+m4.w);
    }
    if (u==0) { conv(r0+4,slot^1); load(r0+8); }
    const float v=vx[slot][u][lane];
    const float delta=beta*(v-mem*scale);
    float d;
    if constexpr (M&P_NODIV) d=delta*scale; else d=delta/scale;
    int z;
    asm volatile("s_mov_b32 %0, 0" : "=s"(z) : "v"(d));
    kl(kb[P^1],min(r+1,R-1)+z);
    if constexpr (u<3) ql(qb[P^1],min(r+1,R-1)+z);
    float d2=d;
    if constexpr (M&P_D2) asm volatile("v_mov_b32 %0, %1" : "=v"(d2) : "v"(d));
    float y0=0,y1=0;
#pragma unroll
    for (int i=0;i<32;i+=2) {
      S[i]+=k[i]*d;S[i+1]+=k[i+1]*d2;
      y0+=S[i]*q[i];y1+=S[i+1]*q[i+1];
    }
    if constexpr (M&P_NOSYNC) { if (sub==0) a.y[(base+r)*128+j]=(y0+y1)*scale; }
    else reinterpret_cast<float*>(&yx[r&1][lane])[sub]=y0+y1;
    pscale=scale;
  };
#pragma unroll 1
  for (int r0=0;r0<R;r0+=4) {
    const int slot=(r0>>2)&1, rl=r0+(lane&3), rn=rl+4;
    const float dq=rl<R ? expf(gn) : 1.f, bq=rl<R ? bn : 0.f;
    gn=rn<R ? a.g[base+rn] : 0.f; bn=rn<R ? a.beta[base+rn] : 0.f;
    ql(qb[0],r0);
    step(std::integral_constant<int,0>{},r0,slot,dq,bq);
    if (r0+1<R) step(std::integral_constant<int,1>{},r0,slot,dq,bq);
    if (r0+2<R) step(std::integral_constant<int,2>{},r0,slot,dq,bq);
    if (r0+3<R) step(std::integral_constant<int,3>{},r0,slot,dq,bq);
  }
  __syncthreads();
  if (sub==0 && R>0) {
    const float4 t=yx[(R-1)&1][lane];
    a.y[(base+R-1)*128+j]=((t.x+t.y)+(t.z+t.w))*pscale;
  }
#pragma unroll
  for (int i=0;i<32;++i) S[i]*=scale;
#pragma unroll
  for (int i=0;i<32;++i) dst[i*128]=S[i];
}

struct Inputs {
  int R;
  Buffer<float> qkv, z, ab, conv, A_log, dt_bias, gamma, hist, hist0, state, state0;
  Buffer<__half> suh, xh;
  Buffer<float> q, k, v, y, g, beta;
  Buffer<unsigned long long> prog;
  Buffer<float> carry;
  explicit Inputs(int rows)
      : R(rows), qkv(Random((size_t)rows * kGdnQkvWidth, 1.f, 1)), z(Random((size_t)rows * kGdnValueWidth, 1.f, 2)),
        ab(Random((size_t)rows * 96, 1.f, 3)), conv(Random(kGdnQkvWidth * 4, .5f, 4)), A_log(Random(48, 1.5f, 5)),
        dt_bias(Random(48, .5f, 6)), gamma(Random(128, .1f, 7)), hist(16 * kGdnQkvWidth),
        hist0(Random(16 * kGdnQkvWidth, 1.f, 8)), state(kGdnStateFloats + 48),
        state0(Random(kGdnStateFloats + 48, .01f, 9)),
        suh(Half(Random(kGdnValueWidth, 1.f, 10))), xh((size_t)rows * kGdnValueWidth), q(48 * (size_t)kCore * 128),
        k(48 * (size_t)kCore * 128), v(48 * (size_t)kCore * 128), y(48 * (size_t)kCore * 128),
        g(48 * kCore), beta(48 * kCore), prog(48 * 4), carry(48) {
    // a carried lazy scale of 0.5 for every head (read only with an anchor).
    const std::vector<float> half(48, .5f);
    HIP_OK(hipMemcpy(state0.p + kGdnStateFloats, half.data(), 48 * 4, hipMemcpyHostToDevice));
  }
  void Reset() {
    HIP_OK(hipMemcpy(state.p, state0.p, state.n * 4, hipMemcpyDeviceToDevice));
    HIP_OK(hipMemcpy(hist.p, hist0.p, hist.n * 4, hipMemcpyDeviceToDevice));
    HIP_OK(hipMemset(xh.p, 0xff, xh.n * 2));
    HIP_OK(hipMemset(y.p, 0, y.n * 4));
  }
  GdnPrefill Full(bool packed, int pos0, int anchor = kCore) {
    GdnPrefill a{};
    a.input = {qkv.p, z.p, nullptr, nullptr, conv.p, A_log.p, dt_bias.p, gamma.p, hist.p, state.p, suh.p,
               xh.p, nullptr, pos0, R, 0, R};
    if (packed) a.input.pack_rows = R;
    a.ab_logits = ab.p;
    a.q = q.p; a.k = k.p; a.v = v.p; a.y = y.p; a.g = g.p; a.beta = beta.p;
    a.anchor = anchor; a.carry = carry.p;
    return a;
  }
};

GdnPrefill Part(GdnPrefill full, int o) {
  const int n = std::min(kCore, full.input.R - o);
  GdnPrefill a = full;
  a.input.qkv += (size_t)o * kGdnQkvWidth; a.input.z += (size_t)o * kGdnValueWidth;
  if (a.input.pack_rows) a.input.pack_off = o; else a.input.xh_out += (size_t)o * kGdnValueWidth;
  a.input.pos0 += o; a.input.R = a.input.commit = n;
  a.ab_logits += (size_t)o * 96; a.chunks = (n + 63) / 64;
  return a;
}

unsigned g_gen = 0;
// mode 0: engine path (LaunchGdnLanes). 1: the fused scan + gate. 2: fused with two q buffers (QDB).
void Parts(GdnPrefill full, int mode, unsigned long long* prog) {
  for (int o = 0; o < full.input.R; o += kCore) {
    GdnPrefill a = Part(full, o);
    if (mode == 0) LaunchGdnLanes(a);
    else if (mode == 1) LaunchGdnScanGate(a, prog, ++g_gen);

    else {
      LaunchGdnPrepQk(a, nullptr, 0);
      // QDB with the launcher's choice of GRID4 (an anchor row off the 4-row grid needs GRID4 false).
      const int first = a.anchor ? (a.anchor - a.input.pos0 % a.anchor) % a.anchor : 0;
      if (first % 4 == 0) GdnScanGateKernel<0, true><<<48 * 4, 128>>>(a, a.k, prog, ++g_gen);
      else GdnScanGateKernel<0, true, false><<<48 * 4, 128>>>(a, a.k, prog, ++g_gen);
      GdnPrefillHistoryKernel<<<40, 256>>>(a);
    }
  }
}

struct Result { std::vector<float> state, hist, y; std::vector<__half> xh; };
Result Run(Inputs& in, bool packed, int pos0, int mode, int anchor) {
  in.Reset();
  Parts(in.Full(packed, pos0, anchor), mode, in.prog.p);
  CheckLaunch();
  return {in.state.Get(), in.hist.Get(), in.y.Get(), in.xh.Get()};
}
template <class T> bool Same(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
template <class T> long FirstDiff(const std::vector<T>& a, const std::vector<T>& b) {
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(T))) return (long)i;
  return -1;
}
}  // namespace

int main(int argc, char** argv) {
  int pairs = 7;
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "--pairs" && i + 1 < argc) pairs = std::stoi(argv[++i]);
    else sizes.push_back(std::stoi(s));
  }
  if (sizes.empty()) sizes = {1, 3, 17, 48, 49, 53, 64, 65, 600, 2047, 2048, 2049, 4096};
  bool ok = true;
  for (int R : sizes) {
    if (R < 1 || R > 16384) return 2;
    Inputs in(R);
    bool same = true;
    for (int packed = 0; packed < 2; ++packed)
      for (int pos0 : {0, 5})
      for (int anchor : {0, kCore}) {
        const Result a = Run(in, packed, pos0, 0, anchor), b = Run(in, packed, pos0, 1, anchor),
                     c = Run(in, packed, pos0, 2, anchor);
        const bool s = Same(a.state, b.state) && Same(a.hist, b.hist) && Same(a.y, b.y) && Same(a.xh, b.xh) &&
                       Same(a.state, c.state) && Same(a.hist, c.hist) && Same(a.y, c.y) && Same(a.xh, c.xh);
        if (!s)
          std::printf("  R=%d anchor=%d packed=%d pos0=%d state=%ld hist=%ld y=%ld xh=%ld (first differing index)\n", R, anchor, packed,
                      pos0, FirstDiff(a.state, b.state), FirstDiff(a.hist, b.hist), FirstDiff(a.y, b.y),
                      FirstDiff(a.xh, b.xh) < 0 ? FirstDiff(a.xh, c.xh) : FirstDiff(a.xh, b.xh));
        same = same && s;
      }
    ok = ok && same;
    std::printf("R=%d fused bitwise=%s", R, same ? "EQUAL" : "DIFFER");
    if (R >= 2048) {
      // The probe P_D2 must match the trunk scan too (its y and state).
      in.Reset();
      const GdnPrefill full = in.Full(true, 0, 0), part = Part(full, 0);
      LaunchGdnLanes(part);  // q, k, g, beta of the part for the scan-only timings
      CheckLaunch();
      std::vector<double> t0, t1, t2, sc, ga, pr, p1, p2, p4, p8, fs, f1, f2, f4, f8, fc;
      for (int rep = 0; rep < pairs; ++rep) {
        t0.push_back(Time([&] { Parts(full, 0, in.prog.p); }, 3));
        t1.push_back(Time([&] { Parts(full, 1, in.prog.p); }, 3));
        t2.push_back(Time([&] { Parts(full, 2, in.prog.p); }, 3));
        sc.push_back(Time([&] { GdnScanSgprKernel<<<48 * 4, 128>>>(part, part.k); }, 3));
        ga.push_back(Time([&] { GdnLanesGateKernel<true><<<kCore / 8 * 48, 256>>>(part); }, 3));
        pr.push_back(Time([&] { GdnLanesPrepQkKernel<<<dim3(16 + 1, kCore / kGdnPrepRows), 256>>>(part); }, 3));
        p1.push_back(Time([&] { ScanProbe<P_NOQ><<<48 * 4, 128>>>(part, part.k); }, 3));
        p2.push_back(Time([&] { ScanProbe<P_NOSYNC><<<48 * 4, 128>>>(part, part.k); }, 3));
        p4.push_back(Time([&] { ScanProbe<P_NODIV><<<48 * 4, 128>>>(part, part.k); }, 3));
        p8.push_back(Time([&] { ScanProbe<P_D2><<<48 * 4, 128>>>(part, part.k); }, 3));
        fs.push_back(Time([&] { GdnScanGateKernel<><<<48 * 4, 128>>>(part, part.k, in.prog.p, ++g_gen); }, 3));
        f1.push_back(Time([&] { GdnScanGateKernel<1><<<48 * 4, 128>>>(part, part.k, in.prog.p, ++g_gen); }, 3));
        f2.push_back(Time([&] { GdnScanGateKernel<3><<<48 * 4, 128>>>(part, part.k, in.prog.p, ++g_gen); }, 3));
        f4.push_back(Time([&] { GdnScanGateKernel<4><<<48 * 4, 128>>>(part, part.k, in.prog.p, ++g_gen); }, 3));
        f8.push_back(Time([&] { GdnScanGateKernel<0, true><<<48 * 4, 128>>>(part, part.k, in.prog.p, ++g_gen); }, 3));
        fc.push_back(Time([&] { GdnScanGateKernel<4, true><<<48 * 4, 128>>>(part, part.k, in.prog.p, ++g_gen); }, 3));
      }
      const double a = bench::Median(t0), b = bench::Median(t1);
      std::printf(" engine_us=%.1f fused_us=%.1f ratio=%.4f ms_per_32K_36L=%+.1f |", a, b, b / a,
                  (b - a) * 36 / 1000. * 32768. / R);
      std::printf(" part2048_us scan=%.1f gate=%.1f prep=%.1f fused_kernel=%.1f | probes noq=%.1f nosync=%.1f "
                  "nodiv=%.1f d2=%.1f",
                  bench::Median(sc), bench::Median(ga), bench::Median(pr), bench::Median(fs), bench::Median(p1),
                  bench::Median(p2), bench::Median(p4), bench::Median(p8));
      std::printf(" | fused probes nowait=%.1f noy=%.1f nogate=%.1f", bench::Median(f1), bench::Median(f2),
                  bench::Median(f4));
      std::printf(" | qdb gate=%.1f nogate=%.1f", bench::Median(f8), bench::Median(fc));
      std::printf(" | qdb_path_us=%.1f ms_per_32K_36L=%+.1f", bench::Median(t2),
                  (bench::Median(t2) - a) * 36 / 1000. * 32768. / R);
      // exactness of P_D2 against the trunk scan on the same part
      const auto st0 = in.state.Get();
      GdnScanSgprKernel<<<48 * 4, 128>>>(part, part.k); CheckLaunch();
      const auto ya = in.y.Get(), sa = in.state.Get();
      in.state.Set(st0);
      ScanProbe<P_D2><<<48 * 4, 128>>>(part, part.k); CheckLaunch();
      const bool d2 = Same(ya, in.y.Get()) && Same(sa, in.state.Get());
      std::printf(" d2_bitwise=%s", d2 ? "EQUAL" : "DIFFER");
      ok = ok && d2;
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
