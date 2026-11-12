// Leak guard (WP part 1): the taint rules of the native mtp_dump, in a header with no engine code (CPU tests
// compile it: tests/mtp_taint_test.cpp). It does in C++ what `python -m training.mtp_tune.taint tag-dump` does,
// byte for byte, so a dump is tagged even when the Python tagger never runs.
//
//   * the start stops when a diagnostic switch (kDiagnostic of engine/switches.h, and the four capture and
//     route-dump names) is set in the environment: those switches write tensors or records to files;
//   * a list is public only when its first line is exactly the public line; any other list (the canary line,
//     no header, an empty or missing file) is tainted: unknown provenance counts as tainted;
//   * a tainted dump gets the canary as the trailer of every .stk and as the first line of every .ids;
//   * a public dump gets PROVENANCE.json (the positive marker, with the digest of the folder);
//   * --check output (.f32, .mtp) cannot carry the canary, so a tainted list with --check is refused.
//
// The canary is computed from a seed, as in server/leak_canary.py: no file holds the literal.
#pragma once

#include "../engine/switches.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace mtp_taint {

// ---- SHA-256 (FIPS 180-4) -------------------------------------------------------------------------------------
class Sha256 {
 public:
  Sha256() { Reset(); }
  void Update(const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    total_ += n;
    while (n) {
      const size_t take = std::min(n, sizeof(buf_) - used_);
      std::memcpy(buf_ + used_, p, take);
      used_ += take, p += take, n -= take;
      if (used_ == sizeof(buf_)) Block(buf_), used_ = 0;
    }
  }
  std::string Hex() {
    const uint64_t bits = total_ * 8;
    const uint8_t one = 0x80, zero = 0;
    Update(&one, 1);
    while (used_ != 56) Update(&zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = uint8_t(bits >> (56 - 8 * i));
    Update(len, 8);
    static const char* digits = "0123456789abcdef";
    std::string out;
    for (uint32_t w : h_)
      for (int s = 28; s >= 0; s -= 4) out += digits[(w >> s) & 15];
    Reset();
    return out;
  }

 private:
  void Reset() {
    static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h_, init, sizeof(h_));
    used_ = 0, total_ = 0;
  }
  static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void Block(const uint8_t* b) {
    static const uint32_t k[64] = {
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
      w[i] = uint32_t(b[4 * i]) << 24 | uint32_t(b[4 * i + 1]) << 16 | uint32_t(b[4 * i + 2]) << 8 | b[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], bb = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + (Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c));
      h = g, g = f, f = e, e = d + t1, d = c, c = bb, bb = a, a = t1 + t2;
    }
    h_[0] += a, h_[1] += bb, h_[2] += c, h_[3] += d, h_[4] += e, h_[5] += f, h_[6] += g, h_[7] += h;
  }
  uint32_t h_[8];
  uint8_t buf_[64];
  size_t used_ = 0;
  uint64_t total_ = 0;
};

inline std::string Sha256Hex(const std::string& s) {
  Sha256 h;
  h.Update(s.data(), s.size());
  return h.Hex();
}

// ---- the canary and the public line ---------------------------------------------------------------------------
// Same seed and prefix as server/leak_canary.py (a test compares the digest of the value).
inline std::string Canary() {
  static const std::string value =
      "CRUCIBLE-CAPTURE-CANARY-v1-" +
      Sha256Hex("crucible draft-capture canary v1: a file with this marker holds recordings or data made from them")
          .substr(0, 40);
  return value;
}
inline const char* PublicHeader() { return "#CRUCIBLE-PUBLIC-SOURCE-v1"; }
inline const char* kMarkerFile = "PROVENANCE.json";

// ---- the environment ------------------------------------------------------------------------------------------
// Switches that write tensors, routing records or other files. The four names stay listed even if the registry
// changes; the rest come from the registry (kDiagnostic), so a new diagnostic switch is covered at once.
inline const std::vector<std::string>& DiagnosticSwitches() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v{"CRUCIBLE_QSA_CAPTURE_AT", "CRUCIBLE_QSA_CAPTURE_DIR", "CRUCIBLE_QSA_CAPTURE_INDEX_ONLY",
                               "CRUCIBLE_ROUTE_DUMP"};
    for (const crucible::SwitchInfo& s : crucible::kSwitches) {
      if (s.kind != crucible::SwitchKind::kDiagnostic) continue;
      bool known = false;
      for (const auto& n : v) known |= n == s.name;
      if (!known) v.emplace_back(s.name);
    }
    return v;
  }();
  return names;
}
// The name of the first diagnostic switch that is set (also when empty), or null.
inline const char* DiagnosticSwitchInEnvironment() {
  for (const auto& name : DiagnosticSwitches())
    if (std::getenv(name.c_str())) return name.c_str();
  return nullptr;
}

// ---- provenance of a list -------------------------------------------------------------------------------------
enum class Provenance { kTainted, kPublic };

// Public only when the first line is exactly the public line WITH the SHA-256 of the rows after it (the bytes after the
// first newline), no row line starts with `#`, and no line holds the canary (the readers skip a `#` line, so a canary
// line could hide behind the public line; rows added later break the digest: outside check H7).
// The canary line, no header, a bare public line, an empty file, a file that cannot be read, a CRLF line: all tainted.
inline std::string PublicLine(const std::string& rows) { return std::string(PublicHeader()) + " sha256=" + Sha256Hex(rows); }

inline Provenance ListProvenance(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::string line;
  if (!f || !std::getline(f, line)) return Provenance::kTainted;
  const std::string rows((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (line != PublicLine(rows)) return Provenance::kTainted;
  if (rows.find(Canary()) != std::string::npos || (!rows.empty() && rows[0] == '#') ||
      rows.find("\n#") != std::string::npos)
    return Provenance::kTainted;
  return Provenance::kPublic;
}

// --check output (.f32, .mtp) has no place for the canary: a tainted list must not ask for it.
inline bool CheckOutputAllowed(Provenance p, bool check) { return !check || p == Provenance::kPublic; }

// ---- outputs --------------------------------------------------------------------------------------------------
// NAME.stk: the fp16 rows, then the canary bytes when tainted (data.py reads only n rows).
inline bool WriteStk(const std::string& path, const void* rows, size_t bytes, Provenance p) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const std::string canary = Canary();
  bool ok = std::fwrite(rows, 1, bytes, f) == bytes;
  if (ok && p == Provenance::kTainted) ok = std::fwrite(canary.data(), 1, canary.size(), f) == canary.size();
  return std::fclose(f) == 0 && ok;
}

// NAME.ids: `#<canary>` first when tainted, then the ids on one line.
inline bool WriteIds(const std::string& path, const std::vector<int>& ids, Provenance p) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  if (p == Provenance::kTainted) f << "#" << Canary() << "\n";
  for (size_t j = 0; j < ids.size(); ++j) f << (j ? "," : "") << ids[j];
  f << "\n";
  f.flush();
  return bool(f);
}

// The digest of a folder as training/mtp_tune/taint.py _digest() computes it: the names, sizes and bytes of all
// files except the marker, sorted by name. Empty when the folder holds a sub folder or a link (not clean).
inline std::string FolderDigest(const std::string& dir, bool* plain) {
  namespace fs = std::filesystem;
  std::vector<std::string> names;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().filename() != kMarkerFile) names.push_back(e.path().filename().string());
  std::sort(names.begin(), names.end());
  Sha256 h;
  const std::string head = "crucible-provenance-v1\n";
  h.Update(head.data(), head.size());
  *plain = true;
  for (const auto& name : names) {
    const fs::path path = fs::path(dir) / name;
    const auto st = fs::symlink_status(path);
    if (!fs::is_regular_file(st)) { *plain = false; return ""; }
    const std::string label = name + std::string(1, '\0') + std::to_string(fs::file_size(path)) + std::string(1, '\0');
    h.Update(label.data(), label.size());
    std::ifstream in(path, std::ios::binary);
    std::vector<char> chunk(1 << 20);
    while (in.read(chunk.data(), chunk.size()) || in.gcount()) h.Update(chunk.data(), size_t(in.gcount()));
  }
  return h.Hex();
}

// The ids of an .ids file as mtp_dump wrote them: lines that start with `#` are skipped, the rest is comma-separated.
inline bool ReadIdsFile(const std::filesystem::path& path, std::vector<int>* ids) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  ids->clear();
  for (std::string line; std::getline(f, line);) {
    if (!line.empty() && line[0] == '#') continue;
    size_t i = 0;
    while (i < line.size()) {
      size_t j = line.find(',', i);
      if (j == std::string::npos) j = line.size();
      const std::string v = line.substr(i, j - i);
      if (v.empty() || v.find_first_not_of("-0123456789") != std::string::npos) return false;
      ids->push_back(std::atoi(v.c_str()));
      i = j + 1;
    }
  }
  return true;
}

// After the last output: a public run writes the positive marker. A tainted run writes none (the canary files
// already make the folder tainted). The marker vouches for the whole folder, so a public run writes it only when the
// folder holds nothing but the outputs of this list's sequences (outside check H8) and every NAME.ids equals the
// ids of that sequence in the list (a stale file of an earlier run is refused); otherwise it returns false.
inline bool FinishFolder(const std::string& dir, Provenance p,
                         const std::vector<std::pair<std::string, std::vector<int>>>& seqs) {
  if (p != Provenance::kPublic) return true;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    const std::string file = e.path().filename().string();
    if (file == kMarkerFile) continue;
    bool made = false;
    for (const auto& [n, ids] : seqs) {
      for (const char* ext : {".stk", ".f32", ".mtp"}) made |= file == n + ext;
      if (file == n + ".ids") {
        std::vector<int> got;
        if (!ReadIdsFile(e.path(), &got) || got != ids) return false;
        made = true;
      }
    }
    if (!made) return false;
  }
  bool plain = false;
  const std::string digest = FolderDigest(dir, &plain);
  std::ofstream f(std::filesystem::path(dir) / kMarkerFile, std::ios::binary);
  if (!f) return false;
  f << "{\"v\": 1, \"source\": \"public\", \"digest\": " << (plain ? "\"" + digest + "\"" : std::string("null")) << "}";
  f.flush();
  return bool(f);
}

}  // namespace mtp_taint
