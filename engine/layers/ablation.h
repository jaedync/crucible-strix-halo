// Opt-in runtime abliteration (CRUCIBLE_ABLATE=FILE): host-side loading,
// validation and the CPU reference of the device write-back.
//
// For each listed sublayer the branch output y (the input of the
// hyper-connection write-back) becomes y - b * dot(y, a) before
// H[s] += post[s] * y. The sidecar format is Heretic's ExLlamaV3 runtime
// file: model.language_model.layers.N.abl_{attn,mlp}.{a,b}, F32 [2560].
#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../exl3/safetensors.h"
#include "../model/flashnext/geometry.h"

namespace crucible {

// Only reviewed sidecars load. Each one names the checkpoint geometry it was
// tuned on: routed expert K and LM-head K (3.05bpw_h5 is 3 and 5).
struct AblationPin {
  const char* sha256;
  const char* source;
  int expert_k, head_k;
};
inline constexpr AblationPin kAblationPins[] = {
    {"05f77ae18082de58b31fb645f52ec5689dce22820add8bb81439e06f01998e5f",
     "andrevp/Qwen3.8-Flash-Next-exl3-3.05bpw-heretic@da2b870f01da36c6601557bc386d0d45c51b36fc "
     "abliteration.safetensors",
     3, 5},
    // Test only: the same file with every b set to zero.
    // The ablated path must then reproduce stock dumps bit for bit.
    {"3d126e01a7d905d49a075af976ced80a632a840a8a7dc8b14315c2dfcd345f93",
     "identity test sidecar (b = 0)", 3, 5},
    // The andrevp tensors unchanged; only the metadata names the 4.05 pack.
    {"1bb0d4132a5798d2a7da7820b65af1e5d37e39a0cf5726f1986be924f2678baa",
     "andrevp/Qwen3.8-Flash-Next-exl3-3.05bpw-heretic@da2b870f01da36c6601557bc386d0d45c51b36fc "
     "abliteration.safetensors, 4.05 metadata copy (pack 4.05bpw_h6_ng6)",
     4, 6},
};

// A sidecar is under 1 MB; the cap bounds the read of a wrong path.
constexpr size_t kAblationMaxBytes = size_t(16) << 20;
// Heretic stores a unit direction; anything else is a different format.
constexpr double kAblationUnitTolerance = 1e-3;

struct AblationSite {
  int layer = 0;
  bool mlp = false;  // false: attention or GDN output projection
  std::vector<float> a, b;
};

struct AblationSet {
  std::string sha256;
  const AblationPin* pin = nullptr;
  std::vector<AblationSite> sites;  // sorted by layer, attention first
};

namespace sha256_detail {
inline uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline void Block(uint32_t h[8], const uint8_t* p) {
  static constexpr uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t t1 = hh + (Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
    const uint32_t t2 = (Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}
}  // namespace sha256_detail

inline std::string Sha256Hex(const uint8_t* p, size_t n) {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  size_t full = n / 64 * 64;
  for (size_t i = 0; i < full; i += 64) sha256_detail::Block(h, p + i);
  uint8_t tail[128] = {};
  const size_t rest = n - full;
  if (rest) std::memcpy(tail, p + full, rest);
  tail[rest] = 0x80;
  const size_t blocks = rest + 9 > 64 ? 2 : 1;
  const uint64_t bits = uint64_t(n) * 8;
  for (int i = 0; i < 8; ++i) tail[blocks * 64 - 1 - i] = uint8_t(bits >> (8 * i));
  for (size_t i = 0; i < blocks; ++i) sha256_detail::Block(h, tail + 64 * i);
  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  for (const uint32_t v : h)
    for (int s = 28; s >= 0; s -= 4) out += hex[(v >> s) & 15];
  return out;
}

inline const AblationPin* FindAblationPin(const std::string& sha256) {
  for (const AblationPin& pin : kAblationPins)
    if (sha256 == pin.sha256) return &pin;
  return nullptr;
}

inline void CheckAblationCheckpoint(const AblationPin& pin, int expert_k, int head_k) {
  if (expert_k != pin.expert_k || head_k != pin.head_k)
    throw std::invalid_argument(
        "the ablation file was tuned on a checkpoint with expert K " + std::to_string(pin.expert_k) +
        " and head K " + std::to_string(pin.head_k) + ", but this checkpoint has expert K " +
        std::to_string(expert_k) + " and head K " + std::to_string(head_k));
}

namespace ablation_detail {

[[noreturn]] inline void Reject(const std::string& reason) {
  throw std::invalid_argument("ablation file: " + reason);
}

// Parses "model.language_model.layers.N.abl_SITE.X"; returns false otherwise.
inline bool SplitKey(const std::string& name, int& layer, bool& mlp, char& part) {
  static const std::string prefix = "model.language_model.layers.";
  if (name.compare(0, prefix.size(), prefix) != 0) return false;
  size_t i = prefix.size(), j = i;
  while (j < name.size() && j - i < 3 && name[j] >= '0' && name[j] <= '9') ++j;
  if (j == i || (name[i] == '0' && j - i > 1)) return false;
  layer = std::stoi(name.substr(i, j - i));
  const std::string rest = name.substr(j);
  if (rest == ".abl_attn.a" || rest == ".abl_attn.b") mlp = false;
  else if (rest == ".abl_mlp.a" || rest == ".abl_mlp.b") mlp = true;
  else return false;
  part = rest.back();
  return true;
}

}  // namespace ablation_detail

// Validates the whole file; throws std::invalid_argument with the reason.
inline AblationSet ParseAblation(const std::vector<uint8_t>& bytes) {
  using ablation_detail::Reject;
  if (bytes.size() < 8) Reject("file is too short for a safetensors header");
  uint64_t hlen = 0;
  for (int i = 0; i < 8; ++i) hlen |= uint64_t(bytes[i]) << (8 * i);
  if (hlen > bytes.size() - 8) Reject("header length exceeds the file size");
  const uint8_t* payload = bytes.data() + 8 + hlen;
  const uint64_t payload_bytes = bytes.size() - 8 - hlen;
  // key: (layer, mlp) -> {a, b}
  std::map<std::pair<int, bool>, std::array<std::vector<float>, 2>> found;
  try {
    st::Scanner sc(std::string_view(reinterpret_cast<const char*>(bytes.data() + 8), size_t(hlen)),
                   "ablation", true);
    sc.Expect('{');
    while (!sc.Peek('}')) {
      const std::string name = sc.Str();
      sc.Expect(':');
      if (name == "__metadata__") { sc.Skip(); sc.Separator('}'); continue; }
      int layer = 0; bool mlp = false; char part = 0;
      if (!ablation_detail::SplitKey(name, layer, mlp, part)) Reject("unexpected tensor " + name);
      if (layer < 0 || layer >= kLayers) Reject("layer out of range in " + name);
      std::string dtype;
      std::vector<int64_t> shape, offsets;
      unsigned fields = 0;
      sc.Expect('{');
      while (!sc.Peek('}')) {
        const std::string key = sc.Str();
        sc.Expect(':');
        if (key == "dtype") { dtype = sc.Str(); fields |= 1; }
        else if (key == "shape") { shape = sc.IntArray(); fields |= 2; }
        else if (key == "data_offsets") { offsets = sc.IntArray(); fields |= 4; }
        else sc.Skip();
        sc.Separator('}');
      }
      sc.Expect('}');
      if (fields != 7) Reject("missing dtype, shape or data_offsets for " + name);
      if (dtype != "F32") Reject(name + " must be F32, not " + dtype);
      if (shape != std::vector<int64_t>{kHidden}) Reject(name + " must have shape [2560]");
      if (offsets.size() != 2 || offsets[0] < 0 || offsets[1] - offsets[0] != int64_t(kHidden * 4) ||
          uint64_t(offsets[1]) > payload_bytes)
        Reject("data_offsets of " + name + " lie outside the payload or do not match the shape");
      auto& slot = found[{layer, mlp}][part == 'a' ? 0 : 1];
      if (!slot.empty()) Reject("duplicate tensor " + name);
      slot.resize(kHidden);
      std::memcpy(slot.data(), payload + offsets[0], kHidden * sizeof(float));
      for (const float v : slot)
        if (!std::isfinite(v)) Reject(name + " has a value that is not finite");
      sc.Separator('}');
    }
    sc.Expect('}');
    sc.Finish();
  } catch (const std::runtime_error& e) {
    Reject(std::string("malformed header: ") + e.what());
  }
  if (found.empty()) Reject("the file has no ablation vectors");
  AblationSet set;
  for (auto& [key, ab] : found) {
    const std::string where = "layer " + std::to_string(key.first) + (key.second ? " mlp" : " attn");
    if (ab[0].empty() || ab[1].empty()) Reject(where + " needs both a and b");
    double n = 0;
    for (const float v : ab[0]) n += double(v) * v;
    if (std::fabs(std::sqrt(n) - 1.0) > kAblationUnitTolerance) Reject(where + ": a is not a unit vector");
    set.sites.push_back({key.first, key.second, std::move(ab[0]), std::move(ab[1])});
  }
  // std::map orders (layer, false) before (layer, true): attention first.
  return set;
}

// Reads, pins and parses CRUCIBLE_ABLATE. Throws std::invalid_argument.
inline AblationSet LoadAblationFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::invalid_argument("cannot open ablation file " + path + ": " + std::strerror(errno));
  const std::streamoff size = f.tellg();
  if (size < 0 || uint64_t(size) > kAblationMaxBytes)
    throw std::invalid_argument("ablation file " + path + " is larger than 16 MiB");
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  f.seekg(0);
  if (!f.read(reinterpret_cast<char*>(bytes.data()), size))
    throw std::invalid_argument("cannot read ablation file " + path);
  const std::string sha = Sha256Hex(bytes.data(), bytes.size());
  const AblationPin* pin = FindAblationPin(sha);
  if (!pin)
    throw std::invalid_argument("ablation file " + path + " has sha256 " + sha +
                                ", which is not a reviewed sidecar (see kAblationPins in engine/layers/ablation.h)");
  AblationSet set = ParseAblation(bytes);
  set.sha256 = sha;
  set.pin = pin;
  return set;
}

// CPU reference of AblateWriteKernel's arithmetic, bit for bit: thread t of
// 256 accumulates elements t, t + 256, ... with fma, then a halving tree.
constexpr int kAblateThreads = 256;
constexpr int kAblatePer = kHidden / kAblateThreads;
static_assert(kHidden % kAblateThreads == 0);

inline float AblationDotRef(const float* y, const float* a) {
  float part[kAblateThreads];
  for (int t = 0; t < kAblateThreads; ++t) {
    float acc = 0.f;
    for (int k = 0; k < kAblatePer; ++k) acc = std::fmaf(y[t + kAblateThreads * k], a[t + kAblateThreads * k], acc);
    part[t] = acc;
  }
  for (int s = kAblateThreads / 2; s > 0; s >>= 1)
    for (int t = 0; t < s; ++t) part[t] = part[t] + part[t + s];
  return part[0];
}

inline void AblateWriteRef(const float* y, const float* a, const float* b, const float* post, float* H) {
  const float c = AblationDotRef(y, a);
  for (int d = 0; d < kHidden; ++d) {
    const float v = std::fmaf(-c, b[d], y[d]);
    for (int s = 0; s < kStreams; ++s) H[s * kHidden + d] = std::fmaf(post[s], v, H[s * kHidden + d]);
  }
}

}  // namespace crucible
