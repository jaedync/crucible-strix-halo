// GPU check of the BF16 n-gram row path (CRUCIBLE_PLE_BF16). One short lease.
//   ple_bf16_check BF16_FILE PACK_DIR [POSITIONS] [SEED]
// 1. All 65536 BF16 words through PleBf16GatherKernel: bitwise equal to PleBf16HostValue.
// 2. Real rows of BF16_FILE (random rows in every head range, plus shard edges): the host table
//    gather (the engine path) equals pread of the file, and the kernel output is bitwise equal to
//    PleBf16HostValue of the file words. No sampled value may overflow fp16.
// 3. The same global rows of the pack's K-bit table through PleGatherKernel<K, true> (with the
//    pack's head_bias): the BF16 output is close (rel RMS below 2.5 %); the mutant "BF16 + head_bias"
//    is farther, and the shift-by-one control is far. This proves the row order and that the BF16
//    path has no head bias.
// Output buffers are NaN-filled before each launch, so an unwritten value fails.
#include <cmath>
#include <filesystem>
#include <random>

#include "../engine/model/flashnext/model.h"

using namespace crucible;

namespace {
constexpr int64_t kShardRows = 2500012;  // tools/ngram16/convert.py metadata shard_rows

[[noreturn]] void Fail(const std::string& why) {
  std::printf("ple_bf16_check: FAIL %s\n", why.c_str());
  std::exit(1);
}

void Pread(int fd, void* dst, size_t n, uint64_t at) {
  auto* out = static_cast<uint8_t*>(dst);
  for (size_t done = 0; done < n;) {
    const ssize_t got = pread(fd, out + done, n - done, off_t(at + done));
    if (got <= 0) Fail("short read of the BF16 file");
    done += size_t(got);
  }
}

// Runs one gather kernel on rows already packed in [R][16] order; returns [R][kHidden] floats.
template <class Launch>
std::vector<float> RunKernel(const void* packed, size_t packed_bytes, int R, Launch launch) {
  uint16_t* table;
  float* out;
  HIP_OK(hipMalloc(&table, packed_bytes));
  HIP_OK(hipMalloc(&out, (size_t)R * kHidden * 4));
  HIP_OK(hipMemcpy(table, packed, packed_bytes, hipMemcpyHostToDevice));
  std::vector<float> host((size_t)R * kHidden, std::nanf(""));
  HIP_OK(hipMemcpy(out, host.data(), host.size() * 4, hipMemcpyHostToDevice));
  launch(table, out);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipMemcpy(host.data(), out, host.size() * 4, hipMemcpyDeviceToHost));
  HIP_OK(hipFree(table));
  HIP_OK(hipFree(out));
  return host;
}

std::vector<float> Bf16Kernel(const std::vector<uint16_t>& packed, int R) {
  return RunKernel(packed.data(), packed.size() * 2, R, [&](uint16_t* t, float* o) {
    PleBf16GatherKernel<<<dim3(kPleHeads, R), kPleRowDim>>>(t, o);
  });
}

bool SameBits(float a, float b) {
  if (a != a || b != b) return a != a && b != b;
  return !std::memcmp(&a, &b, 4);
}

double RelRms(const std::vector<float>& x, const std::vector<float>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    num += double(x[i] - ref[i]) * double(x[i] - ref[i]);
    den += double(ref[i]) * double(ref[i]);
  }
  return std::sqrt(num / den);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: ple_bf16_check BF16_FILE PACK_DIR [POSITIONS] [SEED]\n");
    return 2;
  }
  const std::string bf16_path = argv[1], pack = argv[2];
  const int R = argc > 3 ? std::atoi(argv[3]) : 4096;
  const uint64_t seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 20261020;
  if (R < 16 || R > 65535) Fail("POSITIONS must be 16..65535");

  // 1. Every BF16 word, 26 positions of 16 rows (the tail repeats word 0).
  {
    const int Rx = (65536 + kHidden - 1) / kHidden;
    std::vector<uint16_t> packed((size_t)Rx * kHidden, 0);
    for (uint32_t b = 0; b < 65536; ++b) packed[b] = uint16_t(b);
    const auto out = Bf16Kernel(packed, Rx);
    size_t rounded = 0;
    for (size_t i = 0; i < packed.size(); ++i) {
      if (!SameBits(out[i], PleBf16HostValue(packed[i]))) Fail("exhaustive word " + std::to_string(packed[i]));
      const uint32_t word = uint32_t(packed[i]) << 16;
      float v;
      std::memcpy(&v, &word, 4);
      rounded += i < 65536 && v == v && !SameBits(out[i], v);
    }
    // Mutant: a reference without the fp16 rounding must disagree, or the check cannot see rounding.
    if (!rounded) Fail("mutant: no word changed by the fp16 rounding");
    std::printf("exhaustive: 65536 words bitwise equal; %zu words change under fp16 rounding (mutant caught)\n",
                rounded);
  }

  // Global row geometry from the pack (the same metadata the engine uses).
  st::Checkpoint ck(pack);
  ck.AddFile("ngram_embedding.safetensors");
  const std::string N = "model.language_model." + std::string(kPleTensor) + "ple_embedding.ngram_embedding.";
  int64_t offsets[16], vocab[16];
  ck.ReadBytes(N + "head_offsets", offsets, sizeof(offsets));
  ck.ReadBytes(N + "head_vocab_sizes", vocab, sizeof(vocab));
  std::vector<__half> bias(kHidden);
  ck.ReadBytes(N + "head_bias", bias.data(), bias.size() * 2);
  if (!ck.Has(N + "trellis")) Fail("the check needs a pack with one table tensor (4.05)");
  const auto& tinfo = ck.Info(N + "trellis");
  const int64_t rows = tinfo.shape[0];
  const int words = int(tinfo.shape[1]), K = (words - 1) / 10;
  if (K != 5 && K != 6) Fail("pack table K");

  const PleBf16File f = CheckPleBf16File(bf16_path, N + "weight", rows);
  std::printf("BF16 file: %lld rows, row 0 at byte %llu\n", (long long)f.rows, (unsigned long long)f.offset);

  // Rows: uniform in each head's range, the first and last rows of the table, and shard edges.
  std::mt19937_64 rng(seed);
  std::vector<int64_t> idx((size_t)R * kPleHeads);
  for (int r = 0; r < R; ++r)
    for (int h = 0; h < kPleHeads; ++h) idx[(size_t)r * kPleHeads + h] = offsets[h] + int64_t(rng() % uint64_t(vocab[h]));
  int edges = 0;
  for (int64_t s = 1; s * kShardRows < rows && edges + 4 < R; ++s, edges += 2) {
    const int64_t e = s * kShardRows;
    for (int k = 0; k < 2; ++k) {
      const int64_t g = e - 1 + k;
      const int h = int(std::upper_bound(offsets, offsets + 16, g) - offsets) - 1;
      idx[(size_t)(edges + k) * kPleHeads + h] = g;
    }
  }
  // The table's first and last rows, in the last positions, when they belong to heads 0 and 15.
  if (offsets[0] == 0) idx[(size_t)(R - 2) * kPleHeads] = 0;
  if (offsets[15] + vocab[15] == rows) idx.back() = rows - 1;

  // 2. The engine's host gather equals the file bytes; the kernel equals the host value.
  std::vector<uint16_t> packed(idx.size() * kPleRowDim), direct(packed.size());
  {
    PleHostTable table(bf16_path, kPleBf16RowBytes, {{0, f.rows, f.offset}});
    table.Gather(idx.data(), idx.size(), reinterpret_cast<uint8_t*>(packed.data()));
  }
  {
    const int fd = open(bf16_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) Fail("cannot open " + bf16_path);
    for (size_t i = 0; i < idx.size(); ++i)
      Pread(fd, direct.data() + i * kPleRowDim, kPleBf16RowBytes, f.offset + uint64_t(idx[i]) * kPleBf16RowBytes);
    close(fd);
  }
  if (packed != direct) Fail("host gather differs from pread");
  const auto bf = Bf16Kernel(packed, R);
  size_t exact = 0, sub = 0;
  for (size_t i = 0; i < packed.size(); ++i) {
    const float want = PleBf16HostValue(packed[i]);
    if (!SameBits(bf[i], want)) Fail("real value " + std::to_string(i));
    if (!std::isfinite(want)) Fail("a sampled BF16 value is not finite in fp16 at value " + std::to_string(i));
    const uint32_t word = uint32_t(packed[i]) << 16;
    float v;
    std::memcpy(&v, &word, 4);
    (SameBits(want, v) ? exact : sub)++;
  }
  std::printf("real rows: %zu rows (%d shard-edge rows), %zu values bitwise equal (%zu exact, %zu fp16 subnormal)\n",
              idx.size(), edges, packed.size(), exact, sub);

  // 3. The same rows of the pack's K-bit table through the engine's quantized kernel.
  const std::string pack_file = pack + "/ngram_embedding.safetensors";
  const size_t row_bytes = size_t(words) * 2;
  PleHostTable qt(pack_file, row_bytes, {{0, rows, tinfo.begin}});
  auto quantized = [&](const std::vector<int64_t>& which) {
    std::vector<uint16_t> q(which.size() * words);
    qt.Gather(which.data(), which.size(), reinterpret_cast<uint8_t*>(q.data()));
    __half* b;
    HIP_OK(hipMalloc(&b, kHidden * 2));
    HIP_OK(hipMemcpy(b, bias.data(), kHidden * 2, hipMemcpyHostToDevice));
    auto out = RunKernel(q.data(), q.size() * 2, R, [&](uint16_t* t, float* o) {
      if (K == 5) PleGatherKernel<5, true><<<dim3(16, R), 160>>>(t, nullptr, b, o);
      else PleGatherKernel<6, true><<<dim3(16, R), 160>>>(t, nullptr, b, o);
    });
    HIP_OK(hipFree(b));
    return out;
  };
  const auto q = quantized(idx);
  for (float v : q)
    if (v != v) Fail("quantized kernel left a value unwritten");
  std::vector<float> with_bias(bf.size());
  for (size_t i = 0; i < bf.size(); ++i) with_bias[i] = bf[i] + __half2float(bias[i % kHidden]);
  std::vector<int64_t> shifted(idx);
  for (auto& r : shifted) r = r + 1 < rows ? r + 1 : r - 1;
  const double rel = RelRms(q, bf), rel_bias = RelRms(q, with_bias), rel_shift = RelRms(quantized(shifted), bf);
  std::printf("K=%d rows vs BF16 rows: rel RMS %.4f %%; mutant BF16+head_bias %.4f %%; shift-by-one control %.2f %%\n",
              K, 100 * rel, 100 * rel_bias, 100 * rel_shift);
  if (!(rel < 0.025)) Fail("K-bit rows are not close to the BF16 rows");
  if (!(rel_bias > rel)) Fail("the head_bias mutant is not farther than the no-bias path");
  if (!(rel_shift > 0.5)) Fail("the shift-by-one control is not far");
  std::printf("ple_bf16_check: PASS %d positions, %zu values, K=%d rel RMS %.4f %%\n", R, packed.size(), K, 100 * rel);
  return 0;
}
