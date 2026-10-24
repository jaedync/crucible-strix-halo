// SHA-256 (FIPS 180-4) for image digests and the block names of the NVMe tier. Production binaries do not link
// OpenSSL, so this is a plain implementation; it hashes an 11 MB screenshot in about 30 ms. On an x86 host with the
// SHA extensions (the test machine, the test machine) whole blocks go through them: 1.9 GB/s against 0.3 GB/s on one the test machine core, with the
// same digest.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#if defined(__x86_64__) && !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#include <immintrin.h>
#define SHA256_X86_EXTENSIONS 1
#endif

namespace crucible {
class Sha256 {
 public:
  void Update(const uint8_t* data, size_t size) {
    length_ += size;
    if (used_) {
      const size_t take = std::min(size, size_t(64) - used_);
      std::memcpy(block_.data() + used_, data, take);
      used_ += take; data += take; size -= take;
      if (used_ < 64) return;
      Block(block_.data());
      used_ = 0;
    }
    const size_t whole = size / 64;
    Blocks(data, whole);
    data += whole * 64, size -= whole * 64;
    std::memcpy(block_.data(), data, size);
    used_ = size;
  }
  // The 32 digest bytes (big-endian words, as the hex text).
  std::array<uint8_t, 32> Digest() {
    Finish();
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i)
      for (int b = 0; b < 4; ++b) out[4 * i + b] = uint8_t(h_[i] >> (24 - 8 * b));
    return out;
  }
  // Tests: off forces the plain path, so a test can compare both paths on one host.
  static bool& UseExtensions() { static bool use = true; return use; }
  static bool Accelerated() {
#ifdef SHA256_X86_EXTENSIONS
    static const bool supported = __builtin_cpu_supports("sha") && __builtin_cpu_supports("sse4.1");
    return supported && UseExtensions();
#else
    return false;
#endif
  }
  std::string HexDigest() {
    Finish();
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (const uint32_t word : h_)
      for (int shift = 28; shift >= 0; shift -= 4) out += hex[(word >> shift) & 15];
    return out;
  }

 private:
  static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void Finish() {
    if (finished_) return;
    const uint64_t bits = length_ * 8;
    const uint8_t one = 0x80, zero = 0;
    Update(&one, 1);
    while (used_ != 56) Update(&zero, 1);
    uint8_t tail[8];
    for (int i = 0; i < 8; ++i) tail[i] = uint8_t(bits >> (56 - 8 * i));
    Update(tail, 8);
    finished_ = true;
  }
  void Blocks(const uint8_t* p, size_t count) {
#ifdef SHA256_X86_EXTENSIONS
    if (count && Accelerated()) { NiBlocks(h_.data(), p, count); return; }
#endif
    for (; count; --count, p += 64) Block(p);
  }
  static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
#ifdef SHA256_X86_EXTENSIONS
  // The SHA extensions round (Intel's reference order): state words in two registers, four message words a group.
  [[gnu::target("sha,sse4.1,ssse3")]] static void NiBlocks(uint32_t* st, const uint8_t* p, size_t n) {
    const __m128i mask = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(st));
    __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(st + 4));
    tmp = _mm_shuffle_epi32(tmp, 0xB1);
    s1 = _mm_shuffle_epi32(s1, 0x1B);
    __m128i s0 = _mm_alignr_epi8(tmp, s1, 8);
    s1 = _mm_blend_epi16(s1, tmp, 0xF0);
    for (; n; --n, p += 64) {
      const __m128i a0 = s0, c0 = s1;
      __m128i w[4];
      for (int g = 0; g < 16; ++g) {
        __m128i& wg = w[g & 3];
        if (g < 4) wg = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16 * g)), mask);
        else wg = _mm_sha256msg2_epu32(_mm_add_epi32(_mm_sha256msg1_epu32(w[g & 3], w[(g + 1) & 3]),
                                                     _mm_alignr_epi8(w[(g + 3) & 3], w[(g + 2) & 3], 4)),
                                       w[(g + 3) & 3]);
        __m128i m = _mm_add_epi32(wg, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&k[4 * g])));
        s1 = _mm_sha256rnds2_epu32(s1, s0, m);
        m = _mm_shuffle_epi32(m, 0x0E);
        s0 = _mm_sha256rnds2_epu32(s0, s1, m);
      }
      s0 = _mm_add_epi32(s0, a0);
      s1 = _mm_add_epi32(s1, c0);
    }
    tmp = _mm_shuffle_epi32(s0, 0x1B);
    s1 = _mm_shuffle_epi32(s1, 0xB1);
    s0 = _mm_blend_epi16(tmp, s1, 0xF0);
    s1 = _mm_alignr_epi8(s1, tmp, 8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(st), s0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(st + 4), s1);
  }
#endif
  void Block(const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + (Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
  }
  std::array<uint32_t, 8> h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<uint8_t, 64> block_{};
  size_t used_ = 0;
  uint64_t length_ = 0;
  bool finished_ = false;
};
}  // namespace crucible
