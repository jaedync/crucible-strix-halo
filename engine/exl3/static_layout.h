// Metadata-only sizing must use the same 256-byte units as the weight loader.
#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../model/flashnext/geometry.h"
#include "hc8_layout.h"

namespace crucible {
inline size_t CheckedBytesAdd(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a)
    throw std::overflow_error("static layout byte count overflow");
  return a + b;
}
inline size_t AlignWeightBytes(size_t n) {
  return CheckedBytesAdd(n, 255) & ~size_t(255);
}
inline size_t StaticArenaCapacity(size_t bytes) {
  constexpr size_t granule = size_t(2) << 20;
  if (!bytes) throw std::invalid_argument("static layout is empty");
  return CheckedBytesAdd(bytes, granule - 1) & ~(granule - 1);
}

template<class Checkpoint> class StaticLayout {
 public:
  // hc8: the HC sites hold int8 blocks (hc8_layout.h) in place of fp16 fn and up.
  explicit StaticLayout(Checkpoint& ck, bool hc8 = false) : ck_(ck), hc8_(hc8) {}
  size_t bytes() const { return bytes_; }
  void Raw(const std::string& n) { Add(Payload(n)); }
  void F32(const std::string& n) {
    const size_t b = Payload(n);
    const auto& info = ck_.Info(n);
    if ((info.dtype != "F16" && info.dtype != "BF16") || b % 2)
      throw std::invalid_argument("static layout needs a 16-bit float tensor: " + n);
    Add(CheckedBytesAdd(b, b));
  }
  void Concat(const std::vector<std::string>& names) {
    size_t b = 0;
    for (const auto& n : names) b = CheckedBytesAdd(b, Payload(n));
    Add(b);
  }
  void Lin(const std::string& p) {
    Raw(p + ".trellis"); Raw(p + ".suh"); Raw(p + ".svh");
  }
  void Hc(const std::string& p, bool inject) {
    F32(p + "hc_norm.weight");
    if (hc8_) {
      for (const char* n : {"input_mix_weight_down.weight", "input_mix_weight_up.weight"}) Payload(p + n);
      if (inject) Payload(p + "block_inject_weight.weight");
      Add(Hc8Bytes(kRank + (inject ? kStreams : 0), kWide, kHc8FnGroup));
      Add(Hc8Bytes(kWide, kRank, kHc8UpGroup));
      return;
    }
    // Fragment-order up and fn (fn rows padded to whole 16-row tiles); the
    // row-major copies live in the main arena (model.h HcW).
    for (const char* n : {"input_mix_weight_down.weight", "input_mix_weight_up.weight"}) Payload(p + n);
    if (inject) Payload(p + "block_inject_weight.weight");
    Add(size_t(kWide) * kRank * 2);
    Add(size_t(kRank + (inject ? kStreams : 0) + 15) / 16 * 16 * kWide * 2);
  }
  void Layer(const std::string& p, bool attn) {
    Hc(p + "attn_hyper_connection.", true);
    Hc(p + "mlp_hyper_connection.", true);
    if (attn) {
      const std::string a = p + "self_attn.";
      for (const std::string n : {"q_proj", "k_proj", "v_proj", "indexer.index_qk_proj", "o_proj"})
        Lin(a + n);
      for (const std::string n : {"q_norm", "k_norm", "indexer.q_layernorm", "indexer.k_layernorm"})
        F32(a + n + ".weight");
    } else {
      const std::string g = p + "linear_attn.";
      for (const std::string n : {"in_proj_qkv", "in_proj_z", "out_proj"}) Lin(g + n);
      Concat({g + "in_proj_a.weight", g + "in_proj_b.weight"});
      for (const std::string n : {"conv1d.weight", "A_log", "dt_bias", "norm.weight"}) F32(g + n);
    }
    const std::string m = p + "mlp.";
    Concat({m + "gate.weight", m + "shared_expert_gate.weight"});
    for (const std::string n : {"gate_proj", "up_proj", "down_proj"}) Lin(m + "shared_expert." + n);
  }
  void Ple(const std::string& p) {
    Concat({p + "key_proj.weight", p + "value_proj.weight"});
    for (const std::string n : {"norm_key", "norm_query", "norm_conv"}) F32(p + n + ".weight");
    F32(p + "conv1d.weight");
    Raw(p + "ple_embedding.ngram_embedding.head_bias");
  }
 private:
  size_t Payload(const std::string& n) const {
    if (!ck_.Has(n)) throw std::invalid_argument("static tensor is missing: " + n);
    const auto& info = ck_.Info(n);
    if ((info.end) < info.begin || info.end - info.begin > SIZE_MAX)
      throw std::invalid_argument("static tensor byte range is invalid: " + n);
    return size_t(info.end - info.begin);
  }
  void Add(size_t b) { bytes_ = CheckedBytesAdd(bytes_, AlignWeightBytes(b)); }
  Checkpoint& ck_;
  const bool hc8_;
  size_t bytes_ = 0;
};

template<class Checkpoint>
size_t ExpectedStaticBytes(Checkpoint& ck, int layers, bool hc8 = false) {
  if (layers <= 0)
    throw std::invalid_argument("static layout geometry is invalid");
  StaticLayout<Checkpoint> layout(ck, hc8);
  const std::string p = "model.language_model.";
  for (int l = 0; l < layers; ++l)
    layout.Layer(p + "layers." + std::to_string(l) + ".", IsQsa(l));
  layout.Ple(p + kPleTensor);
  layout.F32("mtp.pre_fc_norm_embedding.weight");
  layout.F32("mtp.pre_fc_norm_hidden.weight");
  layout.Lin("mtp.fc_embedding"); layout.Lin("mtp.fc_hidden");
  layout.Layer("mtp.layers.0.", IsQsa(kMtpLayer));
  layout.Hc("mtp.hyper_connection_mixer.", false);
  layout.Hc(p + "hyper_connection_mixer.", false);
  layout.Lin("lm_head");
  return layout.bytes();
}
}  // namespace crucible
