// Int8 HC read (CRUCIBLE_HC_WMMA=2, hc_int8.h) against the fp16 WMMA read of W'.
//
//   hc8 [sites] [calls] [pairs]
//
// 1. Dequant: a kernel rebuilds W' from every int8 block with Hc8Dequant; it
//    must be bitwise equal to the host's W' (fn and up, every site).
// 2. Proof: for R = 1..16, x and post of the int8 kernel must be bitwise equal
//    to the fp16 kernel on W' (the engine's HcWmmaKernel<4, 20>), and every
//    row must equal the R = 16 run.
// 3. Mutants that must fail: the fp16 kernel on the unquantized W, one code
//    byte changed, one scale one ulp up.
// 4. Time per site call, fp16 vs int8, interleaved (ABAB) over `sites` cold
//    weight sets.
// 5. HcW2Kernel (hc_wmma2.h): GT column tiles per
//    dots block, the mid in the last block of each column group, and (FB) the
//    fragment-order fn and up. It must be bitwise equal to HcWmmaKernel for
//    R = 1..16 on every site, fp16 and int8; mutants of the fragment copies
//    must differ. Then the time of old against new, interleaved.
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "../engine/layers/hc_wmma2.h"
#include "../engine/layers/prefill_hc.h"

using namespace crucible;

namespace {

constexpr int kM = kRank + kStreams;  // inject site
constexpr size_t kFnB = Hc8Bytes(kM, kWide, kHc8FnGroup), kUpB = Hc8Bytes(kWide, kRank, kHc8UpGroup);

uint32_t Hash(uint64_t i, uint32_t seed) {
  uint32_t z = (uint32_t)(i * 2654435761u) ^ seed;
  z ^= z >> 13;
  z *= 0x5bd1e995u;
  z ^= z >> 15;
  return z;
}

// Weights like the model's: a uniform body, rare large values, and some rows
// and groups scaled down to fp16 subnormal scales.
void FillHost(std::vector<_Float16>& w, int n, int k, uint32_t seed, float scale) {
  w.resize((size_t)n * k);
  for (size_t i = 0; i < w.size(); ++i) {
    const uint32_t z = Hash(i, seed);
    float v = ((z & 0xffff) / 65536.f - 0.5f) * scale;
    if ((z >> 16) % 997 == 0) v *= 40.f;
    const size_t r = i / k, g = (i % k) / 64;
    if ((r * 31 + g) % 89 == 0) v *= 1e-3f;
    w[i] = (_Float16)v;
  }
}

__global__ void FillF(float* p, size_t n, uint32_t seed, float scale, float bias) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t z = (uint32_t)(i * 2246822519u) ^ seed;
    z ^= z >> 15;
    z *= 0x2c1b3c6du;
    z ^= z >> 12;
    p[i] = ((z & 0xffffff) / 16777216.f - 0.5f) * scale + bias;
  }
}

// W' row-major from an int8 block, one 16-value step per thread.
__global__ void Dequant(const uint8_t* blk, int n, int k, int g, _Float16* out) {
  const size_t steps = (size_t)n * (k / 16);
  const _Float16* sc = reinterpret_cast<const _Float16*>(blk + Hc8ScaleOffset(n, k));
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < steps; i += (size_t)gridDim.x * blockDim.x) {
    const int r = (int)(i / (k / 16)), j = (int)(i % (k / 16));
    const uint4 v = *reinterpret_cast<const uint4*>(blk + Hc8QIndex(r, j * 16, k));
    const prefill::H16 h = Hc8Dequant(v, sc[(size_t)r * (k / g) + j * 16 / g]);
#pragma unroll
    for (int e = 0; e < 16; ++e) out[(size_t)r * k + j * 16 + e] = h[e];
  }
}

struct Site {
  __half *fn, *up, *fn0, *up0;  // W' and the unquantized W
  __half *fnf, *upf;            // fragment-order W' (HcFragFnKernel, HcFragUpKernel)
  __half* fnc;                  // compact fragment-order fn (HcFnCompactOrder)
  float* w1;
  uint8_t *fq, *uq;
  std::vector<_Float16> hfn, hup;  // host W'
};

struct Bufs {
  float *H, *x, *post;
  HcWScratch sc;
};

unsigned g_epoch = 0;

void Fp16(int R, const __half* fn, const __half* up, const Site& w, const Bufs& b) {
  HcWmmaKernel<kHcWKs, kHcWPf><<<HcWDotsBlocks<kHcWKs>() + kHcWMixBlocks, 256, HcWLds<kHcWKs>(R)>>>(
      fn, kM, up, w.w1, b.H, R, b.x, b.post, b.sc, ++g_epoch);
}

// HcW2Kernel. part 0: the whole kernel; 1: dots and mid alone; 2: the mix
// alone after the flag of the last part-1 run.
// PF: fn fragments in flight per dots wave (0: the engine's); MT: mix tiles per block.
template <int GT, int FB, bool Q8, int PF = 0, int MT = 1>
void New(int R, const Site& w, const Bufs& b, int part = 0, const __half* fn = nullptr, const __half* up = nullptr) {
  constexpr int ND = HcW2DotsBlocks<kHcWKs, GT>(), NM = kHcWMixBlocks / MT;
  constexpr int P = PF ? PF : Q8 ? kHcW8Pf : kHcWPf;
  if (!fn) fn = FB == 2 ? w.fnc : FB ? w.fnf : w.fn;
  if (!up) up = FB ? w.upf : w.up;
  const int grid = part == 0 ? ND + NM : part == 1 ? ND : NM;
  const unsigned e = part == 2 ? g_epoch : ++g_epoch;
  HcW2Kernel<kHcWKs, P, Q8, GT, FB, MT><<<grid, 256, HcWLds<kHcWKs>(R)>>>(
      fn, kM, up, w.w1, b.H, R, b.x, b.post, b.sc, e, part == 2 ? ND : 0, Q8 ? HcW8{w.fq, w.uq} : HcW8{});
}

template <int PF>
void Int8(int R, const Site& w, const Bufs& b) {
  HcWmmaKernel<kHcWKs, PF, true><<<HcWDotsBlocks<kHcWKs>() + kHcWMixBlocks, 256, HcWLds<kHcWKs>(R)>>>(
      w.fn, kM, w.up, w.w1, b.H, R, b.x, b.post, b.sc, ++g_epoch, 0, HcW8{w.fq, w.uq});
}

// Phases alone: dots and mid (no mix blocks), then the mix after the published flag.
template <bool Q8>
void Dots(int R, const Site& w, const Bufs& b) {
  HcWmmaKernel<kHcWKs, Q8 ? kHcW8Pf : kHcWPf, Q8><<<HcWDotsBlocks<kHcWKs>(), 256, HcWLds<kHcWKs>(R)>>>(
      w.fn, kM, w.up, w.w1, b.H, R, b.x, b.post, b.sc, ++g_epoch, 0, Q8 ? HcW8{w.fq, w.uq} : HcW8{});
}
template <bool Q8>
void Mix(int R, const Site& w, const Bufs& b) {
  HcWmmaKernel<kHcWKs, Q8 ? kHcW8Pf : kHcWPf, Q8><<<kHcWMixBlocks, 256, HcWLds<kHcWKs>(R)>>>(
      w.fn, kM, w.up, w.w1, b.H, R, b.x, b.post, b.sc, g_epoch, HcWDotsBlocks<kHcWKs>(),
      Q8 ? HcW8{w.fq, w.uq} : HcW8{});
}

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class T>
std::vector<T> Get(const T* d, size_t n) {
  std::vector<T> h(n);
  HIP_OK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}

struct Out {
  std::vector<float> x, post;
  bool operator==(const Out& o) const {
    return !std::memcmp(x.data(), o.x.data(), x.size() * 4) && !std::memcmp(post.data(), o.post.data(), post.size() * 4);
  }
};

Out Read(const Bufs& b, int R) {
  HIP_OK(hipDeviceSynchronize());
  return {Get(b.x, (size_t)R * kHidden), Get(b.post, (size_t)R * kStreams)};
}

Out Rows(const Out& o, int R) {
  return {std::vector<float>(o.x.begin(), o.x.begin() + (size_t)R * kHidden),
          std::vector<float>(o.post.begin(), o.post.begin() + R * kStreams)};
}

int fails = 0;
void Check(bool ok, const char* what) {
  std::printf("%-58s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++fails;
}

}  // namespace

int main(int argc, char** argv) {
  const int nsites = argc > 1 ? std::atoi(argv[1]) : 24;
  const int calls = argc > 2 ? std::atoi(argv[2]) : 96;
  const int pairs = argc > 3 ? std::atoi(argv[3]) : 5;
  std::vector<Site> sites(nsites);
  std::vector<uint8_t> blk;
  for (int i = 0; i < nsites; ++i) {
    Site& s = sites[i];
    std::vector<_Float16> fn0, up0;
    FillHost(fn0, kM, kWide, 17 + i, 0.08f);
    FillHost(up0, kWide, kRank, 91 + i, 0.5f);
    s.hfn = fn0;
    s.hup = up0;
    HIP_OK(hipMalloc(&s.fn, sizeof(__half) * kM * kWide));
    HIP_OK(hipMalloc(&s.up, sizeof(__half) * kWide * kRank));
    HIP_OK(hipMalloc(&s.fn0, sizeof(__half) * kM * kWide));
    HIP_OK(hipMalloc(&s.up0, sizeof(__half) * kWide * kRank));
    HIP_OK(hipMalloc(&s.w1, sizeof(float) * kWide));
    HIP_OK(hipMalloc(&s.fq, kFnB));
    HIP_OK(hipMalloc(&s.uq, kUpB));
    HIP_OK(hipMemcpy(s.fn0, fn0.data(), fn0.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(s.up0, up0.data(), up0.size() * 2, hipMemcpyHostToDevice));
    blk.assign(kFnB, 0);
    Hc8Quantize(fn0.data(), kM, kWide, kHc8FnGroup, blk.data(), s.hfn.data());
    HIP_OK(hipMemcpy(s.fq, blk.data(), kFnB, hipMemcpyHostToDevice));
    blk.assign(kUpB, 0);
    Hc8Quantize(up0.data(), kWide, kRank, kHc8UpGroup, blk.data(), s.hup.data());
    HIP_OK(hipMemcpy(s.uq, blk.data(), kUpB, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(s.fn, s.hfn.data(), s.hfn.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(s.up, s.hup.data(), s.hup.size() * 2, hipMemcpyHostToDevice));
    FillF<<<64, 256>>>(s.w1, kWide, 7 + i, 0.4f, 1.f);
    HIP_OK(hipMalloc(&s.fnf, sizeof(__half) * kHcWPad * kWide));
    HIP_OK(hipMalloc(&s.upf, sizeof(__half) * kWide * kRank));
    HcFragFnKernel<<<(kHcWPad * kWide + 255) / 256, 256>>>(s.fn, kM, s.fnf);
    HcFragUpKernel<<<(kWide * kRank + 255) / 256, 256>>>(s.up, s.upf);
    {
      std::vector<_Float16> c((size_t)kM * kWide);
      HcFnCompactOrder(s.hfn.data(), kM, c.data());
      HIP_OK(hipMalloc(&s.fnc, sizeof(__half) * kM * kWide));
      HIP_OK(hipMemcpy(s.fnc, c.data(), c.size() * 2, hipMemcpyHostToDevice));
    }
  }
  Bufs b{};
  HIP_OK(hipMalloc(&b.H, sizeof(float) * kHcWRows * kWide));
  HIP_OK(hipMalloc(&b.x, sizeof(float) * kHcWRows * kHidden));
  HIP_OK(hipMalloc(&b.post, sizeof(float) * kHcWRows * kStreams));
  HIP_OK(hipMalloc(&b.sc.part, sizeof(float) * HcWPartFloats<kHcWKs>()));
  HIP_OK(hipMalloc(&b.sc.pss, sizeof(float) * kHcWKs * kStreams * kHcWRows));
  HIP_OK(hipMalloc(&b.sc.inv, sizeof(float) * kHcWRows * kStreams));
  HIP_OK(hipMalloc(&b.sc.thi, sizeof(__half) * kHcWRows * kRank));
  HIP_OK(hipMalloc(&b.sc.tlo, sizeof(__half) * kHcWRows * kRank));
  HIP_OK(hipMalloc(&b.sc.count, sizeof(unsigned) * 2));
  HIP_OK(hipMemset(b.sc.count, 0, sizeof(unsigned) * 2));
  b.sc.flag = b.sc.count + 1;
  // HcW2Kernel scratch: a pss slot and a counter per column group (GT 1 is the largest).
  Bufs b2 = b;
  HIP_OK(hipMalloc(&b2.sc.pss, sizeof(float) * HcW2PssFloats<kHcWKs, 1>()));
  HIP_OK(hipMalloc(&b2.sc.count, sizeof(unsigned) * (HcW2Counters<1>() + 1)));
  HIP_OK(hipMemset(b2.sc.count, 0, sizeof(unsigned) * (HcW2Counters<1>() + 1)));
  b2.sc.flag = b2.sc.count + HcW2Counters<1>();
  FillF<<<256, 256>>>(b.H, (size_t)kHcWRows * kWide, 5, 2.f, 0.f);
  HIP_OK(hipDeviceSynchronize());

  // Subnormal scales occur in the fill (the proof covers that path).
  {
    size_t sub = 0, n = 0;
    std::vector<uint8_t> h(kFnB);
    HIP_OK(hipMemcpy(h.data(), sites[0].fq, kFnB, hipMemcpyDeviceToHost));
    const auto* sc = reinterpret_cast<const _Float16*>(h.data() + Hc8ScaleOffset(kM, kWide));
    for (size_t i = 0; i < (size_t)kM * (kWide / kHc8FnGroup); ++i, ++n)
      if (sc[i] != 0 && std::fabs((float)sc[i]) < 6.1035e-5f) ++sub;
    std::printf("fn site 0: %zu of %zu scales subnormal\n", sub, n);
  }

  // 1. Dequant on the device equals the host W'.
  {
    bool ok = true;
    _Float16* d;
    HIP_OK(hipMalloc(&d, sizeof(_Float16) * kM * kWide));
    for (const Site& s : sites) {
      Dequant<<<1024, 256>>>(s.fq, kM, kWide, kHc8FnGroup, d);
      ok &= Get(d, (size_t)kM * kWide) == s.hfn;
      Dequant<<<1024, 256>>>(s.uq, kWide, kRank, kHc8UpGroup, d);
      ok &= Get(d, (size_t)kWide * kRank) == s.hup;
    }
    HIP_OK(hipFree(d));
    Check(ok, "1. device dequant == host W' (fn and up, all sites)");
  }

  // 2. int8 == fp16 on W', every R, and row invariance.
  const Site& s0 = sites[0];
  Fp16(16, s0.fn, s0.up, s0, b);
  const Out ref16 = Read(b, 16);
  uint64_t h = 1469598103934665603ull;
  for (const float* p : {ref16.x.data(), ref16.post.data()}) {
    const size_t n = p == ref16.x.data() ? ref16.x.size() * 4 : ref16.post.size() * 4;
    for (size_t i = 0; i < n; ++i) h = (h ^ reinterpret_cast<const unsigned char*>(p)[i]) * 1099511628211ull;
  }
  std::printf("fp16 on W' R=16 hash %016llx\n", (unsigned long long)h);
  for (int R = 1; R <= 16; ++R) {
    HIP_OK(hipMemset(b.x, 0xff, sizeof(float) * kHcWRows * kHidden));
    HIP_OK(hipMemset(b.post, 0xff, sizeof(float) * kHcWRows * kStreams));
    Fp16(R, s0.fn, s0.up, s0, b);
    const Out f = Read(b, R);
    HIP_OK(hipMemset(b.x, 0xff, sizeof(float) * kHcWRows * kHidden));
    HIP_OK(hipMemset(b.post, 0xff, sizeof(float) * kHcWRows * kStreams));
    Int8<kHcW8Pf>(R, s0, b);
    const Out q = Read(b, R);
    Int8<20>(R, s0, b);
    const Out q20 = Read(b, R);
    char what[96];
    std::snprintf(what, sizeof what, "2. R=%2d int8 == fp16(W'), == R=16 rows, PF 20 == PF 40", R);
    Check(q == f && f == Rows(ref16, R) && q20 == q, what);
  }
  // Every site at R = 1 and R = 13.
  {
    bool ok = true;
    for (const Site& s : sites)
      for (int R : {1, 13}) {
        Fp16(R, s.fn, s.up, s, b);
        const Out f = Read(b, R);
        Int8<kHcW8Pf>(R, s, b);
        ok &= Read(b, R) == f;
      }
    Check(ok, "2. all sites, R=1 and 13: int8 == fp16(W')");
  }

  // 3. Mutants.
  {
    Fp16(4, s0.fn0, s0.up0, s0, b);
    const Out w = Read(b, 4);
    Int8<kHcW8Pf>(4, s0, b);
    const Out q = Read(b, 4);
    Check(!(w == q), "3. mutant: fp16 on unquantized W differs from int8");
    uint8_t byte;
    const size_t at = Hc8QIndex(77, 5000, kWide);
    HIP_OK(hipMemcpy(&byte, s0.fq + at, 1, hipMemcpyDeviceToHost));
    const uint8_t flip = byte ^ 1;
    HIP_OK(hipMemcpy(s0.fq + at, &flip, 1, hipMemcpyHostToDevice));
    Int8<kHcW8Pf>(4, s0, b);
    Check(!(Read(b, 4) == q), "3. mutant: one fn code byte changed differs");
    HIP_OK(hipMemcpy(s0.fq + at, &byte, 1, hipMemcpyHostToDevice));
    _Float16 sc;
    __half* sp = reinterpret_cast<__half*>(s0.uq + Hc8ScaleOffset(kWide, kRank)) + 1234 * (kRank / kHc8UpGroup) + 3;
    HIP_OK(hipMemcpy(&sc, sp, 2, hipMemcpyDeviceToHost));
    uint16_t bits;
    std::memcpy(&bits, &sc, 2);
    const uint16_t up1 = bits + 1;
    HIP_OK(hipMemcpy(sp, &up1, 2, hipMemcpyHostToDevice));
    Int8<kHcW8Pf>(4, s0, b);
    Check(!(Read(b, 4) == q), "3. mutant: one up scale one ulp up differs");
    HIP_OK(hipMemcpy(sp, &sc, 2, hipMemcpyHostToDevice));
    Int8<kHcW8Pf>(4, s0, b);
    Check(Read(b, 4) == q, "3. restored block equals again");
  }

  // 5. HcW2Kernel against HcWmmaKernel. First: the loader's host reorders
  // (hc_frag.h) equal the device copies of prefill_hc.h, all sites.
  {
    bool ok = true;
    std::vector<_Float16> hf((size_t)kHcWPad * kWide), hu((size_t)kWide * kRank);
    for (const Site& s : sites) {
      HcFnFragOrder(s.hfn.data(), kM, hf.data());
      HcUpFragOrder(s.hup.data(), hu.data());
      ok &= !std::memcmp(Get(reinterpret_cast<const _Float16*>(s.fnf), hf.size()).data(), hf.data(), hf.size() * 2);
      ok &= !std::memcmp(Get(reinterpret_cast<const _Float16*>(s.upf), hu.size()).data(), hu.data(), hu.size() * 2);
    }
    Check(ok, "5. host HcFnFragOrder/HcUpFragOrder == device copies");
  }
  auto clear = [&] {
    HIP_OK(hipMemset(b.x, 0xff, sizeof(float) * kHcWRows * kHidden));
    HIP_OK(hipMemset(b.post, 0xff, sizeof(float) * kHcWRows * kStreams));
  };
  auto once = [&](int R, auto f) {
    clear();
    f();
    return Read(b, R);
  };
  for (int R = 1; R <= 16; ++R) {
    const Out f = once(R, [&] { Fp16(R, s0.fn, s0.up, s0, b); });
    const Out q = once(R, [&] { Int8<kHcW8Pf>(R, s0, b); });
    bool ok = f == Rows(ref16, R);
    ok &= once(R, [&] { New<7, true, false>(R, s0, b2); }) == f;
    ok &= once(R, [&] { New<3, true, false>(R, s0, b2); }) == f;
    ok &= once(R, [&] { New<1, true, false>(R, s0, b2); }) == f;
    ok &= once(R, [&] { New<3, false, false>(R, s0, b2); }) == f;
    ok &= once(R, [&] { New<3, true, false, 10, 2>(R, s0, b2); }) == f;
    ok &= once(R, [&] { New<7, true, false, 10, 2>(R, s0, b2); }) == f;
    ok &= once(R, [&] { New<3, 2, false>(R, s0, b2); }) == f;
    ok &= once(R, [&] { New<3, 2, false, 10, 2>(R, s0, b2); }) == f;
    ok &= once(R, [&] { New<3, 2, false, 10>(R, s0, b2); }) == f;
    ok &= once(R, [&] { New<3, false, true, 20, 2>(R, s0, b2); }) == q;
    ok &= once(R, [&] { New<7, false, true>(R, s0, b2); }) == q;
    ok &= once(R, [&] { New<3, false, true>(R, s0, b2); }) == q;
    ok &= once(R, [&] { New<1, false, true>(R, s0, b2); }) == q;
    char what[96];
    std::snprintf(what, sizeof what, "5. R=%2d new (GT 7/3/1, FB, PF 10, MT 2, int8) == old", R);
    Check(ok, what);
  }
  {
    bool ok = true;
    for (const Site& s : sites)
      for (int R : {1, 4, 13, 16}) {
        const Out f = once(R, [&] { Fp16(R, s.fn, s.up, s, b); });
        const Out q = once(R, [&] { Int8<kHcW8Pf>(R, s, b); });
        ok &= once(R, [&] { New<3, true, false>(R, s, b2); }) == f;
        ok &= once(R, [&] { New<1, true, false>(R, s, b2); }) == f;
        ok &= once(R, [&] { New<3, false, true>(R, s, b2); }) == q;
        ok &= once(R, [&] { New<1, false, true>(R, s, b2); }) == q;
        ok &= once(R, [&] { New<3, true, false, 10, 2>(R, s, b2); }) == f;
        ok &= once(R, [&] { New<3, 2, false>(R, s, b2); }) == f;
        ok &= once(R, [&] { New<3, 2, false, 10, 2>(R, s, b2); }) == f;
        ok &= once(R, [&] { New<3, false, true, 20, 2>(R, s, b2); }) == q;
      }
    Check(ok, "5. all sites, R=1, 4, 13, 16: new == old (fp16 and int8)");
  }
  // Mutants of the fragment copies. One weight gets the same new value in the
  // row-major copy (old kernel) and in a fragment copy (new kernel). The old
  // kernel must see it (the mutant is visible), the new kernel must give the
  // same bits (the fragment index is the same weight), and the restore must
  // equal again. A one-ulp bump is too weak: the mix output goes through a
  // sigmoid and RoundH, and the run of 2026-10-24 lost a one-ulp up change.
  {
    const int R = 4;
    const Out f = once(R, [&] { New<3, true, false>(R, s0, b2); });
    const Out fc = once(R, [&] { New<3, 2, false>(R, s0, b2); });
    auto peek = [&](const __half* p) {
      uint16_t v;
      HIP_OK(hipMemcpy(&v, p, 2, hipMemcpyDeviceToHost));
      return v;
    };
    auto poke = [&](__half* p, uint16_t v) { HIP_OK(hipMemcpy(p, &v, 2, hipMemcpyHostToDevice)); };
    // +-1.0 in fp16: far from the 0.08 and 0.5 fill scales.
    auto far = [](uint16_t v) -> uint16_t { return v == 0x3c00 ? 0xbc00 : 0x3c00; };
    auto mutant = [&](__half* rm, __half* fr, auto run, const Out& base, const char* what) {
      const uint16_t v0 = peek(rm);
      const bool same = peek(fr) == v0;
      poke(rm, far(v0));
      poke(fr, far(v0));
      const Out o = once(R, [&] { Fp16(R, s0.fn, s0.up, s0, b); });
      const Out n = once(R, run);
      poke(rm, v0);
      poke(fr, v0);
      char line[96];
      std::snprintf(line, sizeof line, "5. mutant: %s: old sees it, new == old", what);
      Check(same && !(o == base) && n == o, line);
    };
    // fnf index of fn[77][5000]: tile 4, stream 1, K tile 152, row 13, k 8.
    constexpr int fs = 5000 / kHidden, fk = 5000 % kHidden;
    __half* pf = s0.fnf + (((size_t)(77 / 16) * kStreams + fs) * (kHidden / 16) + fk / 16) * 256 + (77 % 16) * 16 + fk % 16;
    mutant(s0.fn + (size_t)77 * kWide + 5000, pf, [&] { New<3, true, false>(R, s0, b2); }, f, "fnf[77][5000]");
    // upf index of up[1234][100]: row tile 77, K tile 6, row 2, k 4.
    __half* pu = s0.upf + ((size_t)(1234 / 16) * (kRank / 16) + 100 / 16) * 256 + (1234 % 16) * 16 + 100 % 16;
    mutant(s0.up + (size_t)1234 * kRank + 100, pu, [&] { New<3, true, false>(R, s0, b2); }, f, "upf[1234][100]");
    // A second up weight in another stream and the last K tile.
    __half* pu2 = s0.upf + ((size_t)(7777 / 16) * (kRank / 16) + (kRank - 1) / 16) * 256 + (7777 % 16) * 16 + (kRank - 1) % 16;
    mutant(s0.up + (size_t)7777 * kRank + kRank - 1, pu2, [&] { New<3, true, false>(R, s0, b2); }, f,
           "upf[7777][last]");
    Check(once(R, [&] { New<3, true, false>(R, s0, b2); }) == f, "5. restored copies equal again");
    // Compact copy: fn[322][7000] is in the 4-row tail tile (row 2 of tile 20).
    constexpr int cs = 7000 / kHidden, ck = 7000 % kHidden, tr = kM - 320;
    __half* pc = s0.fnc + (size_t)20 * 16 * kWide + (((size_t)cs * (kHidden / 16) + ck / 16) * tr + 2) * 16 + ck % 16;
    mutant(s0.fn + (size_t)322 * kWide + 7000, pc, [&] { New<3, 2, false>(R, s0, b2); }, fc, "compact tail fn[322][7000]");
    Check(once(R, [&] { New<3, 2, false>(R, s0, b2); }) == fc, "5. restored compact copy equals again");
  }

  // 4. Time, interleaved.
  const double mb16 = (double)sizeof(__half) * (kM * kWide + kWide * kRank) / 1e6, mb8 = (kFnB + kUpB) / 1e6;
  std::printf("site fp16 %.2f MB (floor %.1f us), int8 %.2f MB (floor %.1f us) at 245.5 GB/s; %d sites, %d calls\n",
              mb16, mb16 / 245.5e3 * 1e6, mb8, mb8 / 245.5e3 * 1e6, nsites, calls);
  auto time = [&](auto run) {
    for (int i = 0; i < nsites; ++i) run(sites[i]);
    HIP_OK(hipDeviceSynchronize());
    const double t0 = Now();
    for (int i = 0; i < calls; ++i) run(sites[i % nsites]);
    HIP_OK(hipDeviceSynchronize());
    return (Now() - t0) / calls * 1e6;
  };
  for (int R : {1, 4, 8, 16}) {
    double a = 0, q = 0, q20 = 0;
    for (int p = 0; p < pairs; ++p) {
      a += time([&](const Site& s) { Fp16(R, s.fn, s.up, s, b); });
      q += time([&](const Site& s) { Int8<kHcW8Pf>(R, s, b); });
      q20 += time([&](const Site& s) { Int8<20>(R, s, b); });
    }
    a /= pairs, q /= pairs, q20 /= pairs;
    std::printf("R=%2d fp16 %.1f us (%.0f%%)  int8 pf40 %.1f us (%.0f%%)  int8 pf20 %.1f us  speedup %.2fx\n", R, a,
                mb16 / 245.5e3 * 1e6 / a * 100, q, mb8 / 245.5e3 * 1e6 / q * 100, q20, a / q);
    std::fflush(stdout);
  }
  // Phases at R = 1 and 8: dots + mid alone, mix alone (after the flag of the last dots run).
  for (int R : {1, 8}) {
    double d16 = 0, m16 = 0, d8 = 0, m8 = 0;
    for (int p = 0; p < pairs; ++p) {
      d16 += time([&](const Site& s) { Dots<false>(R, s, b); });
      m16 += time([&](const Site& s) { Mix<false>(R, s, b); });
      d8 += time([&](const Site& s) { Dots<true>(R, s, b); });
      m8 += time([&](const Site& s) { Mix<true>(R, s, b); });
    }
    std::printf("R=%2d phases: fp16 dots %.1f us, mix %.1f us; int8 dots %.1f us, mix %.1f us\n", R, d16 / pairs,
                m16 / pairs, d8 / pairs, m8 / pairs);
  }
  // 5. Time, old against new, interleaved in each round.
  std::printf("5. time per site call (us, %% of floor); new = HcW2Kernel<GT, FB>\n");
  const double fl16 = mb16 / 245.5e3 * 1e6, fl8 = mb8 / 245.5e3 * 1e6;
  for (int R : {1, 4, 8, 16}) {
    double t[17] = {};
    for (int p = 0; p < pairs; ++p) {
      t[0] += time([&](const Site& s) { Fp16(R, s.fn, s.up, s, b); });
      t[1] += time([&](const Site& s) { New<7, true, false>(R, s, b2); });
      t[2] += time([&](const Site& s) { New<3, true, false>(R, s, b2); });
      t[3] += time([&](const Site& s) { New<1, true, false>(R, s, b2); });
      t[4] += time([&](const Site& s) { New<3, false, false>(R, s, b2); });
      t[5] += time([&](const Site& s) { Int8<kHcW8Pf>(R, s, b); });
      t[6] += time([&](const Site& s) { New<7, false, true>(R, s, b2); });
      t[7] += time([&](const Site& s) { New<3, false, true>(R, s, b2); });
      t[8] += time([&](const Site& s) { New<1, false, true>(R, s, b2); });
      t[9] += time([&](const Site& s) { New<3, true, false, 10>(R, s, b2); });
      t[10] += time([&](const Site& s) { New<3, true, false, 10, 2>(R, s, b2); });
      t[11] += time([&](const Site& s) { New<7, true, false, 10, 2>(R, s, b2); });
      t[12] += time([&](const Site& s) { New<3, true, false, 20, 2>(R, s, b2); });
      t[13] += time([&](const Site& s) { New<3, false, true, 20, 2>(R, s, b2); });
      t[14] += time([&](const Site& s) { New<3, 2, false>(R, s, b2); });
      t[15] += time([&](const Site& s) { New<3, 2, false, 10, 2>(R, s, b2); });
      t[16] += time([&](const Site& s) { New<3, 2, false, 10>(R, s, b2); });
    }
    for (double& v : t) v /= pairs;
    std::printf("R=%2d fp16: old %.1f (%.0f%%) | GT7 FB %.1f (%.0f%%) | GT3 FB %.1f (%.0f%%) | GT1 FB %.1f (%.0f%%) | "
                "GT3 rowmajor %.1f (%.0f%%)\n",
                R, t[0], fl16 / t[0] * 100, t[1], fl16 / t[1] * 100, t[2], fl16 / t[2] * 100, t[3], fl16 / t[3] * 100,
                t[4], fl16 / t[4] * 100);
    std::printf("R=%2d int8: old %.1f (%.0f%%) | GT7 %.1f (%.0f%%) | GT3 %.1f (%.0f%%) | GT1 %.1f (%.0f%%)\n", R, t[5],
                fl8 / t[5] * 100, t[6], fl8 / t[6] * 100, t[7], fl8 / t[7] * 100, t[8], fl8 / t[8] * 100);
    std::printf("R=%2d fp16: GT3 FB PF10 %.1f (%.0f%%) | GT3 FB PF10 MT2 %.1f (%.0f%%) | GT7 FB PF10 MT2 %.1f (%.0f%%) | "
                "GT3 FB MT2 %.1f (%.0f%%); int8 GT3 MT2 %.1f (%.0f%%)\n",
                R, t[9], fl16 / t[9] * 100, t[10], fl16 / t[10] * 100, t[11], fl16 / t[11] * 100, t[12],
                fl16 / t[12] * 100, t[13], fl8 / t[13] * 100);
    std::printf("R=%2d fp16 compact fn: GT3 FB2 %.1f (%.0f%%) | GT3 FB2 PF10 MT2 %.1f (%.0f%%) | GT3 FB2 PF10 %.1f (%.0f%%)\n",
                R, t[14], fl16 / t[14] * 100, t[15], fl16 / t[15] * 100, t[16], fl16 / t[16] * 100);
    std::fflush(stdout);
  }
  for (int R : {1, 8}) {
    double d[4] = {};
    for (int p = 0; p < pairs; ++p) {
      d[0] += time([&](const Site& s) { New<3, true, false>(R, s, b2, 1); });
      d[1] += time([&](const Site& s) { New<3, true, false>(R, s, b2, 2); });
      d[2] += time([&](const Site& s) { New<3, false, true>(R, s, b2, 1); });
      d[3] += time([&](const Site& s) { New<3, false, true>(R, s, b2, 2); });
    }
    std::printf("R=%2d new GT3 phases: fp16 FB dots+mid %.1f us, mix %.1f us; int8 dots+mid %.1f us, mix %.1f us\n", R,
                d[0] / pairs, d[1] / pairs, d[2] / pairs, d[3] / pairs);
  }
  std::printf("%s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
