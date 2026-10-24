// CPU-only validation must finish before device allocation or tensor conversion.
#pragma once
#include "../../exl3/safetensors.h"
#include "../../exl3/decode_support.h"
#include "geometry.h"
#include "load_plan.h"

namespace crucible {

// Returns the matrix width K (TrellisK: a half width as its code 10 KA + 5). Only routed experts
// (expert = true) may have a half width.
inline int ValidateLinear(st::Checkpoint& ck, const std::string& p, int I, int O, bool expert = false) {
  const std::string name = p + ".trellis";
  const auto& ti = ck.Info(name);
  const int K = ti.shape.size() == 3 ? TrellisK(ti.shape[2]) : 0;
  if (K == 0 || !(expert ? LoadableExpertK(K) : LoadableDecodeK(K))) {
    std::string supported;
    for (const int k : kSupportedDecodeK) supported += (supported.empty() ? "" : ",") + std::to_string(k);
    for (const int k : kWideDecodeK) supported += "," + std::to_string(k);
    std::string half;
    for (const int k : kHalfDecodeK) half += (half.empty() ? "" : ",") + KName(k);
    ck.Error(name, "supported K values are " + supported + " (routed experts also " + half +
                       "); other fractional widths are not supported");
  }
  ck.Require(name, {"I16"}, {I / 16, O / 16, ti.shape.size() == 3 ? ti.shape[2] : 0});
  ck.Require(p + ".suh", {"F16"}, {I});
  ck.Require(p + ".svh", {"F16"}, {O});
  ck.Require(p + ".mul1", {"I32"}, {});
  if (ck.Has(p + ".mcg")) ck.Error(p + ".mcg", "only the mul1 codebook is supported");
  uint32_t marker;
  ck.ReadBytes(p + ".mul1", &marker, sizeof(marker));
  if (marker != 0x83DCD12D) ck.Error(p + ".mul1", "unsupported mul1 codebook constant");
  return K;
}

inline void ValidateHc(st::Checkpoint& ck, const std::string& p, bool inject) {
  ck.Require(p + "hc_norm.weight", {"F16", "BF16"}, {kWide});
  ck.Require(p + "input_mix_weight_down.weight", {"F16"}, {kRank, kWide});
  ck.Require(p + "input_mix_weight_up.weight", {"F16"}, {kWide, kRank});
  if (inject) ck.Require(p + "block_inject_weight.weight", {"F16"}, {kStreams, kWide});
}

inline void ValidateLayer(st::Checkpoint& ck, const std::string& p, bool attn) {
  ValidateHc(ck, p + "attn_hyper_connection.", true);
  ValidateHc(ck, p + "mlp_hyper_connection.", true);
  if (attn) {
    const std::string a = p + "self_attn.";
    ValidateLinear(ck, a + "q_proj", kHidden, kQsaQWidth);
    ValidateLinear(ck, a + "k_proj", kHidden, kQsaKvWidth);
    ValidateLinear(ck, a + "v_proj", kHidden, kQsaKvWidth);
    ValidateLinear(ck, a + "indexer.index_qk_proj", kHidden, kQsaIdxWidth);
    ValidateLinear(ck, a + "o_proj", kQsaOutWidth, kHidden);
    ck.Require(a + "q_norm.weight", {"F16", "BF16"}, {kHeadDim});
    ck.Require(a + "k_norm.weight", {"F16", "BF16"}, {kHeadDim});
    ck.Require(a + "indexer.q_layernorm.weight", {"F16", "BF16"}, {kIdxDim});
    ck.Require(a + "indexer.k_layernorm.weight", {"F16", "BF16"}, {kIdxDim});
  } else {
    const std::string g = p + "linear_attn.";
    ValidateLinear(ck, g + "in_proj_qkv", kHidden, kGdnQkvWidth);
    ValidateLinear(ck, g + "in_proj_z", kHidden, kGdnValueWidth);
    ValidateLinear(ck, g + "out_proj", kGdnValueWidth, kHidden);
    ck.Require(g + "in_proj_a.weight", {"F16"}, {kGdnHeads, kHidden});
    ck.Require(g + "in_proj_b.weight", {"F16"}, {kGdnHeads, kHidden});
    ck.Require(g + "conv1d.weight", {"F16", "BF16"}, {kGdnQkvWidth, 1, 4});
    ck.Require(g + "A_log", {"F16", "BF16"}, {kGdnHeads});
    ck.Require(g + "dt_bias", {"F16", "BF16"}, {kGdnHeads});
    ck.Require(g + "norm.weight", {"F16", "BF16"}, {kGdnDim});
  }
  const std::string m = p + "mlp.";
  ck.Require(m + "gate.weight", {"F16"}, {kExperts, kHidden});
  ck.Require(m + "shared_expert_gate.weight", {"F16"}, {1, kHidden});
  // MoeFusedKernel<R, KR> (moe_rows.h) runs every routed projection at one width KR (3 or 4) and
  // every shared projection at KR + 2. MoeFusedKernelWide runs the h6_ng8 packs' K8 shared expert with
  // K4 or K5 routed experts, their K7 shared expert with K3, K3.5 or K4 routed experts and their K6
  // shared expert with K2, K2.5 or K3 routed experts.
  // kSupportedMoeWidths lists every pair.
  const std::string e0 = m + "experts.0.gate_proj";
  const int kr = ValidateLinear(ck, e0, kHidden, kMoeInter, true);
  std::string pairs;
  bool routed = false;
  for (const MoeWidths& w : kSupportedMoeWidths) {
    pairs += (pairs.empty() ? "" : ", ") + KName(w.kr) + "+" + KName(w.ks);
    routed = routed || w.kr == kr;
  }
  if (!routed)
    ck.Error(e0 + ".trellis", "routed experts of K" + KName(kr) +
                                  " are not supported on this engine (routed+shared K: " + pairs + ")");
  for (const std::string proj : {"gate_proj", "up_proj", "down_proj"}) {
    const bool down = proj == "down_proj";
    const std::string sp = m + "shared_expert." + proj;
    const int ks = ValidateLinear(ck, sp, down ? kMoeInter : kHidden, down ? kHidden : kMoeInter);
    if (!SupportedMoeWidths({kr, ks}))
      ck.Error(sp + ".trellis", "a K" + KName(ks) + " shared expert with K" + KName(kr) +
                                    " routed experts is not supported on this engine (routed+shared K: " + pairs + ")");
  }
  for (int e = 0; e < kExperts; ++e) {
    const std::string ep = m + "experts." + std::to_string(e) + ".";
    for (const std::string proj : {"gate_proj", "up_proj", "down_proj"}) {
      const bool down = proj == "down_proj";
      if (ValidateLinear(ck, ep + proj, down ? kMoeInter : kHidden, down ? kHidden : kMoeInter, true) != kr)
        ck.Error(ep + proj + ".trellis", "all routed expert projections require one K on this engine");
    }
  }
}

inline void ValidatePle(st::Checkpoint& ck, const std::string& p) {
  ck.Require(p + "key_proj.weight", {"F16"}, {kWide, kHidden});
  ck.Require(p + "value_proj.weight", {"F16"}, {kHidden, kHidden});
  for (const std::string norm : {"norm_key", "norm_query", "norm_conv"})
    ck.Require(p + norm + ".weight", {"F16", "BF16"}, {kWide});
  ck.Require(p + "conv1d.weight", {"F16", "BF16"}, {kWide, 1, 4});
  const std::string n = p + "ple_embedding.ngram_embedding.";
  ck.Require(n + "layer_multipliers", {"I64"}, {3});
  ck.Require(n + "head_vocab_sizes", {"I64"}, {16});
  ck.Require(n + "head_offsets", {"I64"}, {16});
  ck.Require(n + "head_bias", {"F16"}, {16, 160});
  int64_t rows = 0;
  // The 4.05 pack stores one table tensor; the 3.05 pack stores shards.
  std::vector<std::string> tables;
  if (ck.Has(n + "trellis")) tables.push_back(n + "trellis");
  else
    for (int shard = 0; ck.Has(n + "shard_" + std::to_string(shard) + ".trellis"); ++shard)
      tables.push_back(n + "shard_" + std::to_string(shard) + ".trellis");
  for (const std::string& name : tables) {
    const auto& ti = ck.Info(name);
    // PleGatherKernel<K> uses a K-bit row codec with 10K+1 words (K=5, 6 or 8),
    // not the linear trellis codec.
    if (ti.shape.size() != 2 || ti.shape[0] <= 0 || (ti.shape[1] != 51 && ti.shape[1] != 61 && ti.shape[1] != 81))
      ck.Error(name, "PLE gather requires nonempty 5-, 6- or 8-bit rows with 51, 61 or 81 words on this engine");
    ck.Require(name, {"I16"}, ti.shape);
    if (rows > std::numeric_limits<int64_t>::max() - ti.shape[0]) ck.Error(name, "PLE row count overflow");
    rows += ti.shape[0];
    if (uint64_t(rows) > (std::numeric_limits<size_t>::max() - 255) / 122)
      ck.Error(name, "PLE table byte count exceeds the host address space");
  }
  if (!rows) ck.Error(n + "shard_0.trellis", "required PLE table is missing");
  int64_t vocab[16], offsets[16];
  ck.ReadBytes(n + "head_vocab_sizes", vocab, sizeof(vocab));
  ck.ReadBytes(n + "head_offsets", offsets, sizeof(offsets));
  for (int h = 0; h < 16; ++h) {
    if (vocab[h] <= 0) ck.Error(n + "head_vocab_sizes", "PLE moduli must be positive");
    if (offsets[h] < 0 || offsets[h] > rows || vocab[h] > rows - offsets[h])
      ck.Error(n + "head_offsets", "PLE head range exceeds the packed table");
  }
}

// CRUCIBLE_PLE_BF16: the n-gram table of the original weights, one BF16 tensor [rows][kPleRowDim]
// (tools/ngram16/convert.py). Global row r of the pack's table is row r of this tensor.
constexpr size_t kPleBf16RowBytes = size_t(kPleRowDim) * 2;
struct PleBf16File {
  uint64_t offset = 0;  // absolute file offset of row 0
  int64_t rows = 0;
};

// The value that PleBf16GatherKernel writes for one BF16 word: the BF16 number rounded to fp16
// (round to nearest even, fp16 subnormals kept). Exact for |v| in [2^-14, 65504] and for 0.
// The CPU tests and the GPU check use it as the reference.
inline float PleBf16HostValue(uint16_t bits) {
  const uint32_t word = uint32_t(bits) << 16;
  float f;
  std::memcpy(&f, &word, sizeof(f));
  return float(_Float16(f));
}

// The BF16 table path, or null when the switch is unset or empty. The resident table has no BF16
// form: a device copy would take 102 GB.
inline const char* PleBf16Path(const char* value, bool ple_host) {
  if (!value || !*value) return nullptr;
  if (!ple_host) throw std::invalid_argument("CRUCIBLE_PLE_BF16 needs CRUCIBLE_PLE_HOST=1");
  return value;
}

// Checks the header with pread only: a mapping fault would read megabytes of rows, and a hash of
// the 102 GB file would take minutes. Throws std::runtime_error with the reason.
inline PleBf16File CheckPleBf16File(const std::string& path, const std::string& tensor, int64_t rows) {
  auto fail = [&](const std::string& reason) -> void {
    throw std::runtime_error("CRUCIBLE_PLE_BF16 file " + path + ": " + reason);
  };
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) fail(std::string("cannot open: ") + std::strerror(errno));
  struct Closer { int fd; ~Closer() { close(fd); } } closer{fd};
  struct stat sb{};
  if (fstat(fd, &sb) != 0 || !S_ISREG(sb.st_mode)) fail("not a regular file");
  const uint64_t size = uint64_t(sb.st_size);
  auto read_at = [&](void* dst, size_t n, uint64_t at) {
    auto* out = static_cast<uint8_t*>(dst);
    for (size_t done = 0; done < n;) {
      const ssize_t got = pread(fd, out + done, n - done, off_t(at + done));
      if (got < 0 && errno == EINTR) continue;
      if (got <= 0) fail("short header read");
      done += size_t(got);
    }
  };
  if (size < 8) fail("shorter than a safetensors header");
  uint8_t len_bytes[8];
  read_at(len_bytes, 8, 0);
  uint64_t hlen = 0;
  for (int b = 0; b < 8; ++b) hlen |= uint64_t(len_bytes[b]) << (8 * b);
  if (hlen > size - 8 || hlen > (uint64_t(100) << 20)) fail("header length is not plausible");
  std::string header(size_t(hlen), '\0');
  read_at(header.data(), header.size(), 8);
  std::string name, dtype;
  std::vector<int64_t> shape, offsets;
  int tensors = 0;
  try {
    st::Scanner sc(header, path, true);
    sc.Expect('{');
    while (!sc.Peek('}')) {
      const std::string key = sc.Str();
      sc.Expect(':');
      if (key == "__metadata__") sc.Skip();
      else {
        ++tensors;
        name = key;
        sc.Expect('{');
        while (!sc.Peek('}')) {
          const std::string field = sc.Str();
          sc.Expect(':');
          if (field == "dtype") dtype = sc.Str();
          else if (field == "shape") shape = sc.IntArray();
          else if (field == "data_offsets") offsets = sc.IntArray();
          else sc.Skip();
          sc.Separator('}');
        }
        sc.Expect('}');
      }
      sc.Separator('}');
    }
    sc.Expect('}');
    sc.Finish();
  } catch (const std::runtime_error& error) {
    fail(std::string("bad header: ") + error.what());
  }
  if (tensors != 1 || name != tensor) fail("want one tensor " + tensor);
  if (dtype != "BF16") fail("dtype is " + dtype + ", want BF16");
  if (shape != std::vector<int64_t>{rows, kPleRowDim})
    fail("shape does not match the pack's " + std::to_string(rows) + " rows of " + std::to_string(kPleRowDim));
  const uint64_t bytes = uint64_t(rows) * kPleBf16RowBytes;
  if (offsets.size() != 2 || offsets[0] < 0 || offsets[1] < offsets[0] || uint64_t(offsets[1] - offsets[0]) != bytes)
    fail("data_offsets do not match the shape");
  const uint64_t begin = 8 + hlen + uint64_t(offsets[0]);
  // Exact size: a short file would end in SIGBUS at a row read, a longer one is not this table.
  if (begin + bytes != size) fail("file size " + std::to_string(size) + " does not match the header");
  return {begin, rows};
}

inline void ValidateCheckpoint(st::Checkpoint& ck) {
  // About 37,000 marker reads: through the mapping each one read up to 4 MB, and the
  // arena evicted those pages before the load read them again (1.55x the checkpoint).
  st::NoMappedReads small_reads_only(ck);
  // The markers are independent 4-byte reads; read them all at once on several threads.
  ck.Prefetch(ck.NamesEndingWith(".mul1"));
  const std::string p = "model.language_model.";
  ck.Require(p + "embed_tokens.weight", {"BF16"}, {kVocab, kHidden});
  for (int l = 0; l < kLayers; ++l) ValidateLayer(ck, p + "layers." + std::to_string(l) + ".", IsQsa(l));
  ValidateLinear(ck, "mtp.fc_embedding", kHidden, kHidden);
  ValidateLinear(ck, "mtp.fc_hidden", kHidden, kHidden);
  ck.Require("mtp.pre_fc_norm_embedding.weight", {"F16", "BF16"}, {kHidden});
  ck.Require("mtp.pre_fc_norm_hidden.weight", {"F16", "BF16"}, {kWide});
  ValidateLayer(ck, "mtp.layers.0.", IsQsa(kMtpLayer));
  ValidateHc(ck, "mtp.hyper_connection_mixer.", false);
  ValidateHc(ck, p + "hyper_connection_mixer.", false);
  ValidateLinear(ck, "lm_head", kHidden, kVocab);
  ValidatePle(ck, p + kPleTensor);
}
}  // namespace crucible
