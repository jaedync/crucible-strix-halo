// CRUCIBLE_PLE_BF16 on the CPU: the header check of the BF16 n-gram table and the host
// gather of its 320-byte rows (engine/model/flashnext/weight_validation.h, ple_host.h).
#include "../engine/model/flashnext/weight_validation.h"
#include "../engine/model/flashnext/ple_host.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

namespace {
using namespace crucible;

const std::string kTensor = "model.language_model." + std::string(kPleTensor) + "ple_embedding.ngram_embedding.weight";

std::string TempDir() {
  const char* dir = std::getenv("TMPDIR");
  return dir && *dir ? dir : "/tmp";
}

std::string WriteFile(const std::string& bytes) {
  std::string path = TempDir() + "/ple_bf16_testXXXXXX";
  const int fd = mkstemp(path.data());
  if (fd < 0 || write(fd, bytes.data(), bytes.size()) != (ssize_t)bytes.size()) {
    std::perror("ple_bf16_test setup");
    std::exit(1);
  }
  close(fd);
  return path;
}

// A safetensors file like tools/ngram16/convert.py writes: metadata, one tensor, 8-byte header padding.
std::string Safetensors(const std::string& tensor_json, const std::string& payload) {
  std::string json = "{\"__metadata__\":{\"format\":\"ngram_bf16_concat\"}," + tensor_json + "}";
  json.append((8 - json.size() % 8) % 8, ' ');
  std::string out(8, '\0');
  for (int b = 0; b < 8; ++b) out[b] = char((uint64_t(json.size()) >> (8 * b)) & 255);
  return out + json + payload;
}

std::string TensorJson(const std::string& name, const std::string& dtype, int64_t rows, int64_t cols,
                       uint64_t begin, uint64_t end) {
  return "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[" + std::to_string(rows) + "," +
         std::to_string(cols) + "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
}

// The reason of the rejection, or "" when the file passes.
std::string Reject(const std::string& bytes, int64_t rows) {
  const std::string path = WriteFile(bytes);
  std::string reason;
  try {
    CheckPleBf16File(path, kTensor, rows);
  } catch (const std::runtime_error& e) {
    reason = e.what();
  }
  unlink(path.c_str());
  return reason;
}

bool Has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// An integer float32 -> fp16 -> float32 rounding (nearest even), independent of the compiler's _Float16.
float IntegerRound16(uint16_t bits) {
  const uint32_t x = uint32_t(bits) << 16;
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int exp = int((x >> 23) & 255);
  const uint32_t man = x & 0x7fffffu;
  uint32_t h;
  if (exp == 255) h = sign | 0x7c00u | (man ? 0x200u : 0);
  else {
    // value = 1.man * 2^(exp-127); fp16 units of 2^-24 below 2^-14, else 10 mantissa bits.
    const int e = exp - 127;
    if (exp == 0 || e < -25) h = sign;  // zero, float subnormals, and values below half of 2^-24
    else if (e < -14) {
      const uint64_t full = uint64_t(man | 0x800000u);  // 2^23 units of 2^e
      const int shift = 23 - (e + 24);                  // to units of 2^-24
      uint64_t q = full >> shift;
      const uint64_t rest = full & ((uint64_t(1) << shift) - 1), half = uint64_t(1) << (shift - 1);
      if (rest > half || (rest == half && (q & 1))) ++q;
      h = sign | uint32_t(q);  // q = 1024 gives the smallest normal, which is right
    } else {
      uint32_t q = (uint32_t(e + 15) << 10) | (man >> 13);
      const uint32_t rest = man & 0x1fffu;
      if (rest > 0x1000u || (rest == 0x1000u && (q & 1))) ++q;
      h = q >= 0x7c00u ? sign | 0x7c00u : sign | q;
    }
  }
  // fp16 bits -> float32
  const uint32_t hs = (h & 0x8000u) << 16, he = (h >> 10) & 31, hm = h & 1023;
  uint32_t out;
  if (he == 31) out = hs | 0x7f800000u | (hm << 13);
  else if (he) out = hs | ((he + 112) << 23) | (hm << 13);
  else {
    float v = float(hm) * (1.0f / 16777216.0f);
    std::memcpy(&out, &v, 4);
    out |= hs;
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}
}  // namespace

int main() {
  static_assert(kPleBf16RowBytes == 320);
  // The switch: unset or empty is off; a path needs the host table.
  CHECK(PleBf16Path(nullptr, false) == nullptr && PleBf16Path("", true) == nullptr);
  CHECK(!std::strcmp(PleBf16Path("/x/t.safetensors", true), "/x/t.safetensors"));
  {
    std::string reason;
    try {
      PleBf16Path("/x/t.safetensors", false);
    } catch (const std::invalid_argument& e) {
      reason = e.what();
    }
    CHECK(Has(reason, "needs CRUCIBLE_PLE_HOST=1"));
  }
  // The reference value of every BF16 word: equal to the integer rounding, and equal to the BF16
  // number itself in the fp16 normal range (the range of the real table values).
  {
    int exact = 0, rounded = 0, nan = 0;
    for (uint32_t b = 0; b < 65536; ++b) {
      const float got = PleBf16HostValue(uint16_t(b)), want = IntegerRound16(uint16_t(b));
      const uint32_t word = b << 16;
      float v;
      std::memcpy(&v, &word, 4);
      if (v != v) {
        CHECK(got != got);
        ++nan;
        continue;
      }
      CHECK(!std::memcmp(&got, &want, 4));
      const float a = v < 0 ? -v : v;
      if (a == 0.f || (a >= 6.103515625e-05f && a <= 65504.f)) {
        CHECK(!std::memcmp(&got, &v, 4));
        ++exact;
      } else ++rounded;
    }
    CHECK(nan == 2 * 127 && exact + rounded + nan == 65536);
  }
  // Rows of 2 synthetic shards; global row = shard * kShardRows + row in shard, as in the real file.
  constexpr int64_t kShardRows = 6421;  // a shard edge falls inside a page
  constexpr int64_t kRows = 2 * kShardRows;
  std::mt19937_64 rng(20261019);
  std::string payload(size_t(kRows) * kPleBf16RowBytes, '\0');
  for (auto& c : payload) c = char(rng());
  const std::string good = Safetensors(TensorJson(kTensor, "BF16", kRows, kPleRowDim, 0, payload.size()), payload);

  // A valid file: the offset of row 0 is the end of the header, which is not page aligned.
  const std::string path = WriteFile(good);
  const PleBf16File f = CheckPleBf16File(path, kTensor, kRows);
  CHECK(f.rows == kRows);
  CHECK(f.offset == good.size() - payload.size());
  CHECK(f.offset % PleHostTable::kPage != 0);

  // Header checks: each case names its reason.
  const uint64_t n = payload.size();
  CHECK(Has(Reject(Safetensors(TensorJson(kTensor, "F16", kRows, kPleRowDim, 0, n), payload), kRows), "dtype is F16"));
  CHECK(Has(Reject(Safetensors(TensorJson(kTensor, "I16", kRows, kPleRowDim, 0, n), payload), kRows), "want BF16"));
  CHECK(Has(Reject(good, kRows + 1), "shape does not match"));
  CHECK(Has(Reject(Safetensors(TensorJson(kTensor, "BF16", kRows * 2, kPleRowDim / 2, 0, n), payload), kRows),
            "shape does not match"));
  CHECK(Has(Reject(Safetensors(TensorJson(kTensor, "BF16", kRows, kPleRowDim, 0, n - 2), payload), kRows),
            "data_offsets"));
  CHECK(Has(Reject(good.substr(0, good.size() - 1), kRows), "file size"));    // short file
  CHECK(Has(Reject(good + std::string(320, '\0'), kRows), "file size"));       // longer file
  CHECK(Has(Reject(Safetensors(TensorJson(kTensor + "x", "BF16", kRows, kPleRowDim, 0, n), payload), kRows),
            "want one tensor"));
  CHECK(Has(Reject(Safetensors(TensorJson(kTensor, "BF16", kRows, kPleRowDim, 0, n) + "," +
                               TensorJson("other", "BF16", 1, 1, n, n + 2), payload + "xx"), kRows),
            "want one tensor"));
  CHECK(Has(Reject(std::string("\xff\xff\xff\xff\x00\x00\x00\x00", 8) + "{}", kRows), "not plausible"));
  CHECK(Has(Reject("abc", kRows), "shorter than"));
  CHECK(Has(Reject(Safetensors("\"broken\":", payload), kRows), "bad header"));
  {
    std::string reason;
    try {
      CheckPleBf16File(TempDir() + "/ple_bf16_test_missing_file", kTensor, kRows);
    } catch (const std::runtime_error& e) {
      reason = e.what();
    }
    CHECK(Has(reason, "cannot open"));
  }

  // The host table gathers the exact file bytes of 320-byte rows: decode-sized, at the shard edge,
  // first and last row, and prefill-sized (the worker pool).
  PleHostTable t(path, kPleBf16RowBytes, {{0, f.rows, f.offset}});
  CHECK(t.Rows() == kRows && t.Segments() == 1 && t.RowBytes() == 320);
  auto expect = [&](int64_t r) { return good.data() + f.offset + size_t(r) * kPleBf16RowBytes; };
  for (size_t count : {size_t(16), size_t(4096 * 16)}) {
    std::vector<int64_t> rows(count);
    for (auto& r : rows) r = int64_t(rng() % kRows);
    rows[0] = 0;
    rows[1] = kShardRows - 1;
    rows[2] = kShardRows;
    rows[count - 1] = kRows - 1;
    std::vector<uint8_t> out(count * kPleBf16RowBytes, 0xa5);
    for (int pass = 0; pass < 2; ++pass) {
      t.Gather(rows.data(), count, out.data());
      for (size_t i = 0; i < count; ++i) CHECK(!std::memcmp(out.data() + i * 320, expect(rows[i]), 320));
    }
    std::vector<uint8_t> async(count * kPleBf16RowBytes, 0x5a);
    t.Start(rows.data(), count, async.data());
    t.Wait();
    CHECK(async == out);
  }
  // Every row that crosses a page edge reads both pages.
  int crossing = 0;
  for (int64_t r = 0; r < kRows && crossing < 64; ++r) {
    const uint64_t o = f.offset + uint64_t(r) * 320;
    if (o / PleHostTable::kPage == (o + 319) / PleHostTable::kPage) continue;
    uint8_t row[320];
    t.Gather(&r, 1, row);
    CHECK(!std::memcmp(row, expect(r), 320));
    ++crossing;
  }
  CHECK(crossing == 64);
  // Out-of-range rows fail closed.
  for (int64_t bad : {int64_t(-1), kRows}) {
    uint8_t row[320];
    bool threw = false;
    try {
      t.Gather(&bad, 1, row);
    } catch (const std::out_of_range&) {
      threw = true;
    }
    CHECK(threw);
  }
  unlink(path.c_str());
  std::puts("ple_bf16_test: ok");
  return 0;
}
