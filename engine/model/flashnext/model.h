// Qwen3.8-Flash-Next EXL3 weights on the GPU: tensor layout conversion and a
// parallel loader. Everything lives in a few large device arenas.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "../../exl3/safetensors.h"
#include "../../gfx1151/common.h"
#include "../../exl3/load_tasks.h"
#include "../../exl3/decode_split.h"
#include "../../exl3/front_order.h"
#include "active_config.h"
#include "mtp_sidecar.h"
#include "../../layers/ops.h"
#include "ple_host.h"
#include "../../exl3/row_policy.h"
#include "weight_validation.h"
#include "load_plan.h"
#include "../../gfx1151/warmup.h"
#include "../../exl3/static_layout.h"
#include "../../switches.h"
#include "../../layers/hc_frag.h"
#include "../../layers/hc_int8.h"
#include "../../layers/numerics.h"
#include "draft_chain_switch.h"

namespace crucible {

struct Lin {
  int I = 0, O = 0, K = 0;
  uint32_t* tiles = nullptr;
  __half* suh = nullptr;
  __half* svh = nullptr;
  int front_s = 0;  // > 0: tiles in front order (front_order.h) for this decode split
};

struct Hc {
  __half* fn = nullptr;  // [M][10240] = [down; inject] * (1 + w)
  int M = 0;
  float* w1 = nullptr;   // (1 + w) [10240]
  __half* up = nullptr;  // [10240][320]
  // Fragment-order copies for bulk prefill, or null.
  __half* fnf = nullptr;
  __half* upf = nullptr;
  // CRUCIBLE_HC_WMMA=2: int8 fn and up for the WMMA decode read; fn and up then hold W' (hc_int8.h).
  HcW8 q8;
};

struct Gdn {
  Lin qkv, z, out;
  __half* ab = nullptr;  // [96][2560]
  float *conv = nullptr, *A_log = nullptr, *dt_bias = nullptr, *gamma = nullptr;
};

struct Qsa {
  Lin q, k, v, idx, o;
  float *q_norm1 = nullptr, *k_norm1 = nullptr, *iq_norm1 = nullptr, *ik_norm1 = nullptr;
};

struct Moe {
  __half* router = nullptr;  // [513][2560], row 512 = shared expert gate
  Lin sg, su, sd;
  std::vector<ExpertPtrs> experts;  // host copy
  ExpertPtrs* d_experts = nullptr;
};

struct Layer {
  Hc hca, hcm;
  bool attn = false;
  Gdn gdn;
  Qsa qsa;
  Moe moe;
};

struct Ple {
  __half* kv = nullptr;  // [12800][2560] key_proj | value_proj
  float *nk1 = nullptr, *nq1 = nullptr, *nc1 = nullptr, *conv = nullptr;
  uint16_t* table = nullptr;  // [rows][1 + 10*K], including the row scale; null with host
  std::shared_ptr<PleHostTable> host;  // CRUCIBLE_PLE_HOST=1: gathered on the CPU
  bool bf16 = false;                   // CRUCIBLE_PLE_BF16: host rows are BF16, not the pack's K-bit rows
  int K = 0;
  int RowWords() const { return bf16 ? kPleRowDim : 10 * K + 1; }
  __half* head_bias = nullptr;
  int64_t mult[3], vocab[16], offset[16];
};

struct Mtp {
  Lin fe, fh;                          // fc_embedding, fc_hidden (shared by streams)
  float *ne1 = nullptr, *nh1 = nullptr;  // (1 + w) pre-fc norms, [2560] and [10240]
  Layer L;                             // one QSA + MoE block with its own caches
  Hc mixer;
};

// Vision tower (HF qwen4_exp). Linears stay EXL3; small tensors become fp32 or stay fp16.
struct VisionBlockW {
  Lin q, k, v, o, fc1, fc2;
  __half *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr, *bfc1 = nullptr, *bfc2 = nullptr;
  float *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr;
};
struct VisionW {
  __half* patch_w = nullptr;  // [1152][1536]
  __half* patch_b = nullptr;
  float* pos = nullptr;       // [2304][1152]
  VisionBlockW blocks[27];
  float *mnw = nullptr, *mnb = nullptr;
  Lin m1, m2;
  __half *bm1 = nullptr, *bm2 = nullptr;
  size_t bytes = 0;
};

struct Model {
  uint16_t* embed = nullptr;  // bf16 [V][2560]
  Layer L[kLayers];
  Mtp mtp;
  Hc final_hc;
  Lin head;
  Ple ple;
  size_t bytes = 0;
  std::vector<DeviceSpan> buffers;
  uint8_t* static_base = nullptr;
  size_t static_bytes = 0;
  // True when ok(layer) holds for every trunk layer (the MTP block is not one). The model owns the layer walk.
  template <class Pred> bool AllLayers(Pred ok) const {
    for (int l = 0; l < kLayers; ++l)
      if (!ok(L[l])) return false;
    return true;
  }
};

// Smaller arenas limit unused reservation and retry high-order allocation more
// often when the kernel falls back to small physical pages.
class Arena {
 public:
  explicit Arena(size_t chunk = size_t(256) << 20, bool single = false)
      : chunk_(chunk), single_(single) {}
  uint8_t* Alloc(size_t n) {
    n = AlignWeightBytes(n);
    if (single_ && (n > chunk_ || (cur_ && n > chunk_ - off_))) {
      std::fprintf(stderr, "load error: static arena capacity exceeded\n");
      std::exit(1);
    }
    if (n > chunk_) {
      uint8_t* p;
      HIP_OK(hipMalloc(&p, n));
      buffers_.push_back({p,n});
#ifdef CRUCIBLE_QSA_TRACE
      TraceBuffer(p,n);
#endif
      total_ += n;
      reserved_ += n;
      return p;
    }
    if (!cur_ || n > chunk_ - off_) {
      HIP_OK(hipMalloc(&cur_, chunk_));
      reserved_ += chunk_;
      buffers_.push_back({cur_,chunk_});
#ifdef CRUCIBLE_QSA_TRACE
      TraceBuffer(cur_,chunk_);
#endif
      off_ = 0;
    }
    uint8_t* p = cur_ + off_;
    off_ += n;
    total_ += n;
    return p;
  }
  size_t total() const { return total_; }
  size_t reserved() const { return reserved_; }
  const std::vector<DeviceSpan>& buffers() const { return buffers_; }
  uint8_t* base() const { return cur_; }

 private:
  std::vector<DeviceSpan> buffers_;
  const size_t chunk_;
  const bool single_;
  uint8_t* cur_ = nullptr;
  size_t off_ = 0, total_ = 0, reserved_ = 0;
};

class Loader {
 public:
  explicit Loader(const std::string& dir, bool vision = false) : dir_(dir), ck_(dir) {
    ck_.AddFile("ngram_embedding.safetensors");
    // The 4.05 pack keeps its 6-bit tower outside the index; 3.05 keeps it in a shard.
    if (vision && std::filesystem::exists(std::filesystem::path(dir) / "vision_k6.safetensors"))
      ck_.AddFile("vision_k6.safetensors", /*wins=*/true);
    if (!ck_.Has("mtp.hyper_connection_mixer.hc_norm.weight"))
      ck_.AddFile("mtp_hyper_connection_mixer_patch.safetensors");
    // A tuned MTP head changes only drafts; verification keeps the trunk exact.
    // What this engine runs with: the set switches and the sidecar by size and sha (never a path).
    std::fprintf(stderr, "%s\n", ActiveConfig().c_str());
    // The start stops (exit 2) when the sidecar cannot be read or replaces too few tensors: exact verify hides a
    // broken drafter from every token gate, and only the speed shows it.
    if (const char* side = crucible::SwitchText("CRUCIBLE_MTP_SIDECAR"); side && *side) {
      if (const std::string unreadable = MtpSidecarUnreadable(side); !unreadable.empty()) {
        std::fprintf(stderr, "%s\n", unreadable.c_str());
        std::exit(2);
      }
      const size_t replaced = ck_.Override(side, "mtp.", /*allow_empty=*/true);
      const std::string label = MtpSidecarLabel(side);
      const size_t expected = MtpSidecarExpected([&](const std::string& name) { return ck_.Has(name); });
      if (const std::string fault = MtpSidecarFault(replaced, expected, label); !fault.empty()) {
        std::fprintf(stderr, "%s\n", fault.c_str());
        std::exit(2);
      }
      std::fprintf(stderr, "%s\n", MtpSidecarLine(replaced, label).c_str());
    }
    // Fixed-size kernels cannot safely consume arbitrary checkpoint geometry.
    // Validate every source before the first allocation or conversion task.
    ValidateCheckpoint(ck_);
    // Validation has read every shard header, so the count is complete; a run log shows that a mix loaded.
    if (ck_.mixed()) std::fprintf(stderr, "mixed pack: the index picks each tensor; %zu copies in other shards skipped\n", ck_.shadowed());
  }

  // Dense attention and head tiles load in front order (exact, no extra
  // memory). Each group's split follows the decode plan (DecodeSplits).
  std::vector<int> FrontSplits(const std::vector<std::string>& prefixes) {
    std::vector<std::vector<int64_t>> trellis;
    for (const auto& p : prefixes) trellis.push_back(ck_.Info(p + ".trellis").shape);
    front_total_ += prefixes.size();
    const FrontLoad front = FrontLoadSplits(trellis);
    if (front.bad >= 0) Die("front tile order does not fit " + prefixes[front.bad]);
    if (front.splits[0] > 0) front_count_ += prefixes.size();
    return front.splits;
  }

  void LayerW(const std::string& L, Layer& y, bool attn) {
    y.hca = HcW(L + "attn_hyper_connection.", true);
    if (!streaming_) y.hcm = HcW(L + "mlp_hyper_connection.", true);
    y.attn = attn;
    if (y.attn) {
      const std::string A = L + "self_attn.";
      const auto in = FrontSplits({A + "q_proj", A + "k_proj", A + "v_proj", A + "indexer.index_qk_proj"});
      const int os = FrontSplits({A + "o_proj"})[0];
      y.qsa.q = LinW(A + "q_proj", in[0]);
      y.qsa.k = LinW(A + "k_proj", in[1]);
      y.qsa.v = LinW(A + "v_proj", in[2]);
      y.qsa.idx = LinW(A + "indexer.index_qk_proj", in[3]);
      if (!streaming_) y.qsa.o = LinW(A + "o_proj", os);
      y.qsa.q_norm1 = F32(A + "q_norm.weight", 1.f);
      y.qsa.k_norm1 = F32(A + "k_norm.weight", 1.f);
      y.qsa.iq_norm1 = F32(A + "indexer.q_layernorm.weight", 1.f);
      y.qsa.ik_norm1 = F32(A + "indexer.k_layernorm.weight", 1.f);
      if (streaming_) y.qsa.o = LinW(A + "o_proj", os);
    } else {
      const std::string G = L + "linear_attn.";
      const auto in = FrontSplits({G + "in_proj_qkv", G + "in_proj_z"});
      const int os = FrontSplits({G + "out_proj"})[0];
      y.gdn.qkv = LinW(G + "in_proj_qkv", in[0]);
      y.gdn.z = LinW(G + "in_proj_z", in[1]);
      if (!streaming_) y.gdn.out = LinW(G + "out_proj", os);
      y.gdn.ab = Concat16({G + "in_proj_a.weight", G + "in_proj_b.weight"});
      y.gdn.conv = F32(G + "conv1d.weight", 0.f);
      y.gdn.A_log = F32(G + "A_log", 0.f);
      y.gdn.dt_bias = F32(G + "dt_bias", 0.f);
      y.gdn.gamma = F32(G + "norm.weight", 0.f);
      if (streaming_) y.gdn.out = LinW(G + "out_proj", os);
    }
    if (streaming_) y.hcm = HcW(L + "mlp_hyper_connection.", true);
    const std::string M = L + "mlp.";
    y.moe.router = Concat16({M + "gate.weight", M + "shared_expert_gate.weight"});
    y.moe.sg = LinW(M + "shared_expert.gate_proj");
    y.moe.su = LinW(M + "shared_expert.up_proj");
    y.moe.sd = LinW(M + "shared_expert.down_proj");
    y.moe.experts.resize(kExperts);
    for (int e = 0; e < kExperts; ++e) ExpertW(M + "experts." + std::to_string(e) + ".", y.moe.experts[e]);
  }

  static void Upload(Moe& mo) {
    HIP_OK(hipMalloc(&mo.d_experts, sizeof(ExpertPtrs) * kExperts));
#ifdef CRUCIBLE_QSA_TRACE
    TraceBuffer(mo.d_experts,sizeof(ExpertPtrs)*kExperts);
#endif
    HIP_OK(hipMemcpy(mo.d_experts, mo.experts.data(), sizeof(ExpertPtrs) * kExperts,
                     hipMemcpyHostToDevice));
  }

  // Separate arena, loaded only by `serve --vision 1`; text-only engines never map these tensors.
  void LoadVision(VisionW& v) {
    const std::string P = "model.visual.";
    if (!ck_.Has(P + "patch_embed.proj.weight")) Die("this checkpoint has no vision tower");
    ValidateVision(P);
    Arena arena(size_t(32) << 20);
    active_ = &arena;
    v.patch_w = Concat16({P + "patch_embed.proj.weight"});
    v.patch_b = Concat16({P + "patch_embed.proj.bias"});
    v.pos = F32(P + "pos_embed.weight", 0.f);
    auto bias = [&](const std::string& n) { return Concat16({n + ".bias"}); };
    for (int i = 0; i < 27; ++i) {
      const std::string B = P + "blocks." + std::to_string(i) + ".";
      auto& b = v.blocks[i];
      LinGroup({B + "attn.q_proj", B + "attn.k_proj", B + "attn.v_proj"}, {&b.q, &b.k, &b.v});
      b.o = LinW(B + "attn.proj");
      b.fc1 = LinW(B + "mlp.linear_fc1");
      b.fc2 = LinW(B + "mlp.linear_fc2");
      b.bq = bias(B + "attn.q_proj"); b.bk = bias(B + "attn.k_proj"); b.bv = bias(B + "attn.v_proj");
      b.bo = bias(B + "attn.proj"); b.bfc1 = bias(B + "mlp.linear_fc1"); b.bfc2 = bias(B + "mlp.linear_fc2");
      b.n1w = F32(B + "norm1.weight", 0.f); b.n1b = F32(B + "norm1.bias", 0.f);
      b.n2w = F32(B + "norm2.weight", 0.f); b.n2b = F32(B + "norm2.bias", 0.f);
    }
    v.mnw = F32(P + "merger.norm.weight", 0.f);
    v.mnb = F32(P + "merger.norm.bias", 0.f);
    v.m1 = LinW(P + "merger.linear_fc1");
    v.m2 = LinW(P + "merger.linear_fc2");
    v.bm1 = bias(P + "merger.linear_fc1");
    v.bm2 = bias(P + "merger.linear_fc2");
    Run();
    active_ = &arena_;
    v.bytes = arena.total();
  }

  // Fixed-size vision kernels need this geometry and one EXL3 width for every tower linear
  // (ValidateVision in weight_validation.h, which also accepts BF16 for the F16 vision tensors).
  void ValidateVision(const std::string& P) { crucible::ValidateVision(ck_, P); }

  void Load(Model& m, bool streaming = false) {
    streaming_ = streaming;
    const std::string P = "model.language_model.";
    auto t0 = std::chrono::steady_clock::now();
    const size_t planned = streaming ? ExpectedStaticBytes(ck_, kLayers, hc8_, mtp_hc8_) : 0;
    Arena static_arena(streaming ? StaticArenaCapacity(planned) : size_t(2) << 20, true);
    m.embed = (uint16_t*)Raw(P + "embed_tokens.weight");
    if (streaming) active_ = &static_arena;
    for (int l = 0; l < kLayers; ++l) {
      if (streaming && PleBefore(l)) PleW(P + kPleTensor, m.ple);
      LayerW(P + "layers." + std::to_string(l) + ".", m.L[l], IsQsa(l));
    }
    if (streaming) {
      m.final_hc = HcW(P + "hyper_connection_mixer.", false);
      m.head = LinW("lm_head", FrontSplits({"lm_head"})[0]);
    }
    m.mtp.fe = LinW("mtp.fc_embedding");
    m.mtp.fh = LinW("mtp.fc_hidden");
    m.mtp.ne1 = F32("mtp.pre_fc_norm_embedding.weight", 1.f);
    m.mtp.nh1 = F32("mtp.pre_fc_norm_hidden.weight", 1.f);
    // CRUCIBLE_DRAFT_CHAIN hc8: int8 HC for the MTP layer and its mixer only (drafts only; the trunk keeps fp16).
    hc8_site_ = hc8_ || mtp_hc8_;
    LayerW("mtp.layers.0.", m.mtp.L, IsQsa(kMtpLayer));
    m.mtp.mixer = HcW("mtp.hyper_connection_mixer.", false);
    hc8_site_ = hc8_;
    if (!streaming) {
      m.final_hc = HcW(P + "hyper_connection_mixer.", false);
      m.head = LinW("lm_head", FrontSplits({"lm_head"})[0]);
      PleW(P + kPleTensor, m.ple);
    }
    if (static_arena.total() != planned) Die("static weight layout does not match its planned size");
    m.static_base = static_arena.base();
    m.static_bytes = static_arena.total();
    active_ = &arena_;
    Run();
    for (int l = 0; l < kLayers; ++l) Upload(m.L[l].moe);
    Upload(m.mtp.L.moe);
    m.bytes = arena_.total() + static_arena.total();
    m.buffers = arena_.buffers();
    const auto& static_buffers = static_arena.buffers();
    m.buffers.insert(m.buffers.end(), static_buffers.begin(), static_buffers.end());
    for (int l = 0; l < kLayers; ++l)
      m.buffers.push_back({m.L[l].moe.d_experts,sizeof(ExpertPtrs)*kExperts});
    m.buffers.push_back({m.mtp.L.moe.d_experts,sizeof(ExpertPtrs)*kExperts});
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "loaded %.2f GB in %.1f s (%zu tasks)\n", m.bytes / 1e9, s, ndone_.load());
    std::fprintf(stderr, "weight payload %zu bytes, arena reserved %zu bytes, staging peak %zu bytes\n",
                 m.bytes, arena_.reserved() + static_arena.reserved(), stage_peak_.load());
    std::fprintf(stderr, "front tile order: %d of %d dense matrices\n", front_count_, front_total_);
    if (hc8_)
      std::fprintf(stderr, "HC int8 (CRUCIBLE_HC_WMMA=2): %d sites, fn group %d, up group %d\n", hc8_sites_,
                   kHc8FnGroup, kHc8UpGroup);
    else if (mtp_hc8_)
      std::fprintf(stderr, "draft chain: MTP HC int8 (CRUCIBLE_DRAFT_CHAIN hc8): %d sites, fn group %d, up group %d\n",
                   hc8_sites_, kHc8FnGroup, kHc8UpGroup);
    PrintWidths(m);
  }

  // One line of the loaded EXL3 widths (each set of K values), so a run log shows which pack loaded.
  static void PrintWidths(const Model& m) {
    // Sorted by width (35 = K3.5 sorts between K3 and K4).
    auto list = [](const std::set<int>& seen) {
      std::vector<int> order(seen.begin(), seen.end());
      std::sort(order.begin(), order.end(), [](int a, int b) { return WidthHalves(a) < WidthHalves(b); });
      std::string text;
      for (const int k : order) text += (text.empty() ? "K" : ",K") + KName(k);
      return text;
    };
    std::set<int> routed, shared, dense;
    for (int l = 0; l <= kLayers; ++l) {
      const Layer& y = l < kLayers ? m.L[l] : m.mtp.L;
      for (const ExpertPtrs& e : y.moe.experts)
        for (const int k : e.K) routed.insert(k);
      shared.insert(y.moe.sg.K);
      for (const Lin* lin : {&y.qsa.q, &y.qsa.k, &y.qsa.v, &y.qsa.idx, &y.qsa.o, &y.gdn.qkv, &y.gdn.z, &y.gdn.out})
        if (lin->tiles) dense.insert(lin->K);
    }
    const std::string ple = m.ple.bf16 ? "BF16" : "K" + std::to_string(m.ple.K);
    std::fprintf(stderr, "EXL3 widths: routed %s, shared %s, dense %s, head K%d, mtp fc K%d, PLE rows %s\n",
                 list(routed).c_str(), list(shared).c_str(), list(dense).c_str(), m.head.K, m.mtp.fe.K,
                 ple.c_str());
  }

 private:
  using Task = LoadTask;

  const uint8_t* Src(const std::string& n, size_t* bytes = nullptr) { return ck_.Data(n, bytes); }
  const st::TensorInfo& Info(const std::string& n) { return ck_.Info(n); }

  uint8_t* Raw(const std::string& n) {
    size_t b;
    const uint8_t* s = Src(n, &b);
    uint8_t* d = active_->Alloc(b);
    tasks_.push_back({b, d, s, nullptr});
    return d;
  }

  // fp32 copy of a bf16/f16 tensor, plus a constant (1 for centered norms).
  float* F32(const std::string& n, float add) {
    size_t b;
    const uint8_t* s = Src(n, &b);
    const auto& ti = Info(n);
    if (ti.dtype != "BF16" && ti.dtype != "F16") ck_.Error(n, "expected dtype F16/BF16 for conversion");
    const bool bf = ti.dtype == "BF16";
    const size_t cnt = b / 2;
    uint8_t* d = active_->Alloc(cnt * 4);
    tasks_.push_back({cnt * 4, d, nullptr, [=](uint8_t* buf) {
                        float* out = reinterpret_cast<float*>(buf);
                        for (size_t i = 0; i < cnt; ++i) out[i] = ToF(Read16(s, i), bf) + add;
                      }});
    return reinterpret_cast<float*>(d);
  }

  __half* Concat16(const std::vector<std::string>& names) {
    size_t total = 0;
    std::vector<std::pair<const uint8_t*, size_t>> parts;
    // A BF16 tensor (third-party vision checkpoints) is converted to F16 once, here (bf16_half.h).
    // The F16 path is unchanged: a plain copy of the stored bytes.
    std::vector<std::shared_ptr<std::vector<uint16_t>>> converted;
    for (const auto& n : names) {
      const std::string& dtype = Info(n).dtype;
      if (dtype != "F16" && dtype != "BF16") ck_.Error(n, "expected dtype F16");
      if (dtype == "BF16") {
        HalfTensor h = ReadHalfTensor(ck_, n);
        if (h.inexact)
          std::fprintf(stderr, "load: %s: %zu BF16 values rounded to F16\n", n.c_str(), h.inexact);
        auto v = std::make_shared<std::vector<uint16_t>>(std::move(h.bits));
        parts.push_back({nullptr, v->size() * 2});
        total += v->size() * 2;
        converted.push_back(std::move(v));
        continue;
      }
      size_t b;
      const uint8_t* s = Src(n, &b);
      parts.push_back({s, b});
      converted.push_back(nullptr);
      total += b;
    }
    uint8_t* d = active_->Alloc(total);
    size_t off = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
      const auto [s, b] = parts[i];
      if (s) tasks_.push_back({b, d + off, s, nullptr});
      else {
        auto v = converted[i];
        tasks_.push_back({b, d + off, nullptr, [v](uint8_t* buf) { std::memcpy(buf, v->data(), v->size() * 2); }});
      }
      off += b;
    }
    return reinterpret_cast<__half*>(d);
  }

  // Safetensors payloads can start at an odd file offset, including the valid pack.
  static uint16_t Read16(const uint8_t* src, size_t index) {
    uint16_t u;
    std::memcpy(&u, src + index * sizeof(u), sizeof(u));
    return u;
  }

  static float ToF(uint16_t u, bool bf) {
    if (bf) {
      uint32_t x = (uint32_t)u << 16;
      float f;
      std::memcpy(&f, &x, 4);
      return f;
    }
    _Float16 h;
    std::memcpy(&h, &u, sizeof(h));
    return (float)h;
  }

  static void Die(const std::string& msg) {
    std::fprintf(stderr, "load error: %s\n", msg.c_str());
    std::exit(1);
  }

  // Trellis repacked to crucible order (column-block-major tiles), or to front
  // order when front_s > 0.
  void Trellis(const std::string& prefix, int& I, int& O, int& K, uint8_t* dst,
               std::vector<std::function<void(uint8_t*)>>& fills, size_t& off, int front_s = 0) {
    const st::TensorInfo& ti = Info(prefix + ".trellis");
    if (ti.dtype != "I16" || ti.shape.size() != 3) Die(prefix);
    if (ck_.Has(prefix + ".mcg") || !ck_.Has(prefix + ".mul1")) Die("not mul1: " + prefix);
    const int A = (int)ti.shape[0], B = (int)ti.shape[1];
    K = TrellisK(ti.shape[2]);
    if (K == 0) Die("unsupported linear K: " + prefix);
    I = A * 16;
    O = B * 16;
    const uint8_t* src = Src(prefix + ".trellis");
    const size_t T = TileDwords(K), o = off;  // 8 K dwords; a half width: lane_window.h
    fills.push_back([=](uint8_t* buf) {
      uint32_t* out = reinterpret_cast<uint32_t*>(buf + o);
      for (int b = 0; b < B; ++b)
        for (int a = 0; a < A; ++a) {
          const size_t tile = front_s ? FrontTile(b, a, A, B, front_s) : (size_t)b * A + a;
          std::memcpy(out + tile * T, src + ((size_t)a * B + b) * T * 4, T * 4);
        }
    });
    off += (size_t)A * B * T * 4;
    (void)dst;
  }

  void Copy16(const std::string& n, std::vector<std::function<void(uint8_t*)>>& fills,
              size_t& off) {
    size_t b;
    const uint8_t* s = Src(n, &b);
    if (Info(n).dtype != "F16") ck_.Error(n, "expected dtype F16");
    const size_t o = off;
    fills.push_back([=](uint8_t* buf) { std::memcpy(buf + o, s, b); });
    off += (b + 255) & ~size_t(255);
  }

  size_t LinBytes(const std::string& p) {
    const st::TensorInfo& ti = Info(p + ".trellis");
    size_t b = ((size_t)(ti.end - ti.begin) + 255) & ~size_t(255);
    b += ((size_t)(Info(p + ".suh").end - Info(p + ".suh").begin) + 255) & ~size_t(255);
    b += ((size_t)(Info(p + ".svh").end - Info(p + ".svh").begin) + 255) & ~size_t(255);
    return (b + 255) & ~size_t(255);
  }

  // Lays out [tiles][suh][svh] for each prefix in one region, one task.
  void LinGroup(const std::vector<std::string>& prefixes, std::vector<Lin*> outs, int front_s = 0) {
    size_t total = 0;
    for (const auto& p : prefixes) total += LinBytes(p);
    uint8_t* d = active_->Alloc(total);
    std::vector<std::function<void(uint8_t*)>> fills;
    size_t off = 0;
    for (size_t i = 0; i < prefixes.size(); ++i) {
      Lin& L = *outs[i];
      const size_t t0 = off;
      Trellis(prefixes[i], L.I, L.O, L.K, d, fills, off, front_s);
      L.front_s = front_s;
      off = (off + 255) & ~size_t(255);
      L.tiles = reinterpret_cast<uint32_t*>(d + t0);
      L.suh = reinterpret_cast<__half*>(d + off);
      Copy16(prefixes[i] + ".suh", fills, off);
      L.svh = reinterpret_cast<__half*>(d + off);
      Copy16(prefixes[i] + ".svh", fills, off);
      off = (off + 255) & ~size_t(255);
    }
    if (off > total) Die("layout overflow");
    tasks_.push_back({off, d, nullptr, [fills](uint8_t* buf) {
                        for (auto& f : fills) f(buf);
                      }});
  }

  Lin LinW(const std::string& p, int front_s = 0) {
    Lin L;
    LinGroup({p}, {&L}, front_s);
    return L;
  }

  void ExpertW(const std::string& p, ExpertPtrs& e) {
    Arena* previous = active_;
    active_ = &arena_;
    Lin g, u, d;
    LinGroup({p + "gate_proj", p + "up_proj", p + "down_proj"}, {&g, &u, &d});
    const Lin* ls[3] = {&g, &u, &d};
    for (int i = 0; i < 3; ++i) {
      // ExpertPtrs has no front split, and every expert reader assumes engine order.
      if (ls[i]->front_s) {
        std::fprintf(stderr, "%s: expert weights must be in engine order\n", p.c_str());
        std::exit(2);
      }
      e.tiles[i] = ls[i]->tiles;
      e.suh[i] = ls[i]->suh;
      e.svh[i] = ls[i]->svh;
      e.K[i] = ls[i]->K;
    }
    active_ = previous;
  }

  Hc HcW(const std::string& p, bool inject) {
    Hc h;
    h.M = kRank + (inject ? kStreams : 0);
    h.w1 = F32(p + "hc_norm.weight", 1.f);
    if (hc8_site_) return HcW8Load(p, inject, h);
    // Decode (HcW2Kernel) and bulk prefill read the fragment-order copies, so
    // they go to active_ in decode order (the static arena, which the
    // lookahead reads). The row-major fn and up go to arena_ for the other
    // readers (VALU and persistent HC, small prefill). Same values.
    const std::string un = p + "input_mix_weight_up.weight";
    size_t ub = 0;
    if (Info(un).dtype != "F16") ck_.Error(un, "expected dtype F16");
    const uint8_t* usrc = Src(un, &ub);
    if (ub != (size_t)kWide * kRank * 2) ck_.Error(un, "unexpected HC up size");
    const int M = h.M;
    const size_t fb = (size_t)M * kWide * 2, ffb = (size_t)(M + 15) / 16 * 16 * kWide * 2;
    const auto fn = HcFn(p, inject);
    uint8_t* uf = active_->Alloc(ub);
    uint8_t* ff = active_->Alloc(ffb);
    uint8_t* uw = arena_.Alloc(ub);
    uint8_t* fw = arena_.Alloc(fb);
    tasks_.push_back({ub, uw, usrc, nullptr});
    tasks_.push_back({ub, uf, nullptr, [=](uint8_t* buf) {
                        HcUpFragOrder(reinterpret_cast<const _Float16*>(usrc), reinterpret_cast<_Float16*>(buf));
                      }});
    tasks_.push_back({fb, fw, nullptr, fn});
    tasks_.push_back({ffb, ff, nullptr, [=](uint8_t* buf) {
                        std::vector<_Float16> w((size_t)M * kWide);
                        fn(reinterpret_cast<uint8_t*>(w.data()));
                        HcFnFragOrder(w.data(), M, reinterpret_cast<_Float16*>(buf));
                      }});
    h.fn = reinterpret_cast<__half*>(fw);
    h.up = reinterpret_cast<__half*>(uw);
    h.fnf = reinterpret_cast<__half*>(ff);
    h.upf = reinterpret_cast<__half*>(uf);
    return h;
  }

  // fp16 fn = [down; inject] * (1 + w), row-major [M][kWide].
  std::function<void(uint8_t*)> HcFn(const std::string& p, bool inject) {
    const uint8_t* down = Src(p + "input_mix_weight_down.weight");
    const uint8_t* inj = inject ? Src(p + "block_inject_weight.weight") : nullptr;
    const uint8_t* nw = Src(p + "hc_norm.weight");
    const bool nbf = Info(p + "hc_norm.weight").dtype == "BF16";
    const int M = kRank + (inject ? kStreams : 0);
    return [=](uint8_t* buf) {
      _Float16* out = reinterpret_cast<_Float16*>(buf);
      for (int m = 0; m < M; ++m) {
        const uint8_t* row = m < kRank ? down + (size_t)m * kWide * 2 : inj + (size_t)(m - kRank) * kWide * 2;
        for (int c = 0; c < kWide; ++c)
          out[(size_t)m * kWide + c] = (_Float16)(ToF(Read16(row, c), false) * (1.f + ToF(Read16(nw, c), nbf)));
      }
    };
  }

  // CRUCIBLE_HC_WMMA=2: the int8 blocks go to active_ in decode order (the
  // static arena, which the lookahead reads), and W' in fp16 to arena_ for
  // the paths that read fp16. Each task quantizes its matrix again; the
  // quantizer is deterministic, so the blocks and W' agree.
  Hc HcW8Load(const std::string& p, bool inject, Hc h) {
    const std::string un = p + "input_mix_weight_up.weight";
    size_t ub = 0;
    if (Info(un).dtype != "F16") ck_.Error(un, "expected dtype F16");
    const uint8_t* usrc = Src(un, &ub);
    if (ub != (size_t)kWide * kRank * 2) ck_.Error(un, "unexpected HC up size");
    const int M = h.M;
    const auto fn = HcFn(p, inject);
    const size_t fb = Hc8Bytes(M, kWide, kHc8FnGroup), upb = Hc8Bytes(kWide, kRank, kHc8UpGroup);
    uint8_t* fq = active_->Alloc(fb);
    uint8_t* uq = active_->Alloc(upb);
    uint8_t* fw = arena_.Alloc((size_t)M * kWide * 2);
    uint8_t* uw = arena_.Alloc(ub);
    tasks_.push_back({fb, fq, nullptr, [=](uint8_t* buf) {
                        std::vector<_Float16> w((size_t)M * kWide);
                        fn(reinterpret_cast<uint8_t*>(w.data()));
                        std::memset(buf, 0, fb);
                        Hc8Quantize(w.data(), M, kWide, kHc8FnGroup, buf, nullptr);
                      }});
    tasks_.push_back({(size_t)M * kWide * 2, fw, nullptr, [=](uint8_t* buf) {
                        fn(buf);
                        auto* w = reinterpret_cast<_Float16*>(buf);
                        Hc8Quantize(w, M, kWide, kHc8FnGroup, nullptr, w);
                      }});
    tasks_.push_back({upb, uq, nullptr, [=](uint8_t* buf) {
                        std::memset(buf, 0, upb);
                        Hc8Quantize(reinterpret_cast<const _Float16*>(usrc), kWide, kRank, kHc8UpGroup, buf, nullptr);
                      }});
    tasks_.push_back({ub, uw, nullptr, [=](uint8_t* buf) {
                        std::memcpy(buf, usrc, ub);
                        auto* w = reinterpret_cast<_Float16*>(buf);
                        Hc8Quantize(w, kWide, kRank, kHc8UpGroup, nullptr, w);
                      }});
    h.fn = reinterpret_cast<__half*>(fw);
    h.up = reinterpret_cast<__half*>(uw);
    h.q8 = {fq, uq};
    ++hc8_sites_;
    return h;
  }

  void PleW(const std::string& p, Ple& e) {
    e.kv = Concat16({p + "key_proj.weight", p + "value_proj.weight"});
    e.nk1 = F32(p + "norm_key.weight", 1.f);
    e.nq1 = F32(p + "norm_query.weight", 1.f);
    e.nc1 = F32(p + "norm_conv.weight", 1.f);
    e.conv = F32(p + "conv1d.weight", 0.f);
    const std::string N = p + "ple_embedding.ngram_embedding.";
    std::memcpy(e.mult, Src(N + "layer_multipliers"), sizeof(e.mult));
    std::memcpy(e.vocab, Src(N + "head_vocab_sizes"), sizeof(e.vocab));
    std::memcpy(e.offset, Src(N + "head_offsets"), sizeof(e.offset));
    e.head_bias = Concat16({N + "head_bias"});
    size_t total = 0;
    int64_t rows = 0;
    std::vector<std::string> shards;
    if (ck_.Has(N + "trellis")) shards.push_back(N + "trellis");
    else
      for (int s = 0; ck_.Has(N + "shard_" + std::to_string(s) + ".trellis"); ++s)
        shards.push_back(N + "shard_" + std::to_string(s) + ".trellis");
    if (shards.empty()) Die("missing PLE table");
    for (const auto& s : shards) {
      const auto& ti = Info(s);
      if (ti.dtype != "I16" || ti.shape.size() != 2 || PleRowK(ti.shape[1]) == 0) Die("unsupported PLE row: " + s);
      const int K = PleRowK(ti.shape[1]);
      if (e.K && e.K != K) Die("mixed PLE row widths");
      e.K = K;
      total += ti.end - ti.begin;
      rows += ti.shape[0];
    }
    if (const char* bf16 = PleBf16()) {
      HostPleBf16(bf16, p + "ple_embedding.ngram_embedding.weight", e, rows);
      return;
    }
    if (PleHost()) {
      HostPle(shards, e, total);
      return;
    }
    uint8_t* d = arena_.Alloc(total);
    size_t off = 0;
    for (const auto& s : shards) {
      size_t b;
      const uint8_t* src = Src(s, &b);
      tasks_.push_back({b, d + off, src, nullptr});
      off += b;
    }
    e.table = reinterpret_cast<uint16_t*>(d);
  }

  // CRUCIBLE_HC_WMMA=2: int8 HC weights (hc_int8.h). Option A (CRUCIBLE_EXACT=1) keeps fp16.
  static bool HcInt8() {
    try {
      return NumericsInt("CRUCIBLE_HC_WMMA", 1, 0, 2) == 2;
    } catch (const std::invalid_argument& error) {
      Die(error.what());
    }
    return false;
  }

  // CRUCIBLE_DRAFT_CHAIN (draft_chain_switch.h): hc8 gives the MTP sites int8 HC. Drafts only, so the output stays.
  static bool MtpHc8() {
    DraftChain d;
    if (!ResolveDraftChain(crucible::SwitchText("CRUCIBLE_DRAFT_CHAIN"), d))
      Die("CRUCIBLE_DRAFT_CHAIN must be a comma list of hc8, lead and stop=01..99");
    return d.hc8;
  }

  static bool PleHost() {
    try {
      return RowPolicyInt("CRUCIBLE_PLE_HOST", 0, 0, 1) != 0;
    } catch (const std::invalid_argument& error) {
      Die(error.what());
    }
    return false;
  }

  // CRUCIBLE_PLE_FETCH (ple_host.h): 1 hints every page of a gather in blocks, 2 also splits gathers of 160 rows or
  // more over the gather parts. The bytes do not change.
  static int PleFetch() {
    try {
      return RowPolicyInt("CRUCIBLE_PLE_FETCH", 0, 0, 2);
    } catch (const std::invalid_argument& error) {
      Die(error.what());
    }
    return 0;
  }

  static const char* PleBf16() {
    try {
      return PleBf16Path(crucible::SwitchText("CRUCIBLE_PLE_BF16"), PleHost());
    } catch (const std::invalid_argument& error) {
      Die(error.what());
    }
    return nullptr;
  }

  // The original BF16 rows replace the pack's K-bit rows, so the page cache holds only BF16 rows: the
  // pack's table file is mapped for its header and small tensors, and no row of it is read. The
  // head bias stays loaded (the static layout counts it); the BF16 kernel does not read it.
  void HostPleBf16(const std::string& path, const std::string& tensor, Ple& e, int64_t rows) {
    try {
      const PleBf16File f = CheckPleBf16File(path, tensor, rows);
      e.host = std::make_shared<PleHostTable>(path, kPleBf16RowBytes, std::vector<PleSegment>{{0, f.rows, f.offset}});
      e.host->SetFetch(PleFetch());
    } catch (const std::exception& error) {
      Die(error.what());
    }
    e.bf16 = true;
    std::fprintf(stderr, "PLE table on host: BF16, %llu bytes not resident, %zu segment(s), %s\n",
                 (unsigned long long)(uint64_t(rows) * kPleBf16RowBytes), e.host->Segments(),
                 std::filesystem::path(path).filename().c_str());
  }

  // The shards stay in the checkpoint file; ple_host.h maps it read-only.
  void HostPle(const std::vector<std::string>& shards, Ple& e, size_t total) {
    const size_t row = size_t(10 * e.K + 1) * 2;
    std::vector<PleSegment> segments;
    int64_t first = 0;
    for (const auto& s : shards) {
      const auto& ti = Info(s);
      if (ti.file != Info(shards[0]).file) Die("PLE shards span several files");
      segments.push_back({first, ti.shape[0], ti.begin});
      first += ti.shape[0];
    }
    try {
      e.host = std::make_shared<PleHostTable>(dir_ + "/" + Info(shards[0]).file, row, segments);
      e.host->SetFetch(PleFetch());
    } catch (const std::exception& error) {
      Die(error.what());
    }
    std::fprintf(stderr, "PLE table on host: %zu bytes not resident, %zu segment(s)\n", total,
                 e.host->Segments());
  }

  void Run() {
    constexpr int kWorkers = 12;
    constexpr size_t kStage = size_t(64) << 20;
    // Pieces of a large direct copy go to all workers (a multiple of the stage size).
    constexpr size_t kDirectPiece = 4 * kStage;
    tasks_ = SplitDirectTasks(std::move(tasks_), kDirectPiece);
    std::atomic<size_t> next{0};
    auto work = [&]() {
      uint8_t* stage = nullptr;
      size_t capacity = kStage;
      HIP_OK(hipHostMalloc(&stage, capacity));
      StageAdded(capacity);
      hipStream_t st;
      HIP_OK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
      for (size_t i; (i = next++) < tasks_.size();) {
        Task& t = tasks_[i];
        if (t.direct) {
          // Chunked through pinned staging: pageable mmap copies are slow.
          const uint8_t* s = static_cast<const uint8_t*>(t.direct);
          for (size_t o = 0; o < t.bytes; o += kStage) {
            const size_t n = std::min(kStage, t.bytes - o);
            std::memcpy(stage, s + o, n);
            HIP_OK(hipMemcpyAsync(t.dst + o, stage, n, hipMemcpyHostToDevice, st));
            HIP_OK(hipStreamSynchronize(st));
          }
        } else {
          if (t.bytes > capacity) {
            HIP_OK(hipHostFree(stage));
            stage_live_.fetch_sub(capacity);
            capacity = t.bytes;
            HIP_OK(hipHostMalloc(&stage, capacity));
            StageAdded(capacity);
          }
          t.fill(stage);
          HIP_OK(hipMemcpyAsync(t.dst, stage, t.bytes, hipMemcpyHostToDevice, st));
          HIP_OK(hipStreamSynchronize(st));
        }
        ++ndone_;
      }
      HIP_OK(hipStreamDestroy(st));
      HIP_OK(hipHostFree(stage));
      stage_live_.fetch_sub(capacity);
    };
    std::vector<std::thread> ts;
    for (int i = 0; i < kWorkers; ++i) ts.emplace_back(work);
    for (auto& t : ts) t.join();
    tasks_.clear();
  }

  void StageAdded(size_t n) {
    const size_t live = stage_live_.fetch_add(n) + n;
    size_t peak = stage_peak_.load();
    while (peak < live && !stage_peak_.compare_exchange_weak(peak, live)) {}
  }

  std::atomic<size_t> stage_live_{0}, stage_peak_{0};
  const std::string dir_;
  st::Checkpoint ck_;
  Arena arena_;
  Arena* active_ = &arena_;
  int front_count_ = 0, front_total_ = 0;
  bool streaming_ = false;
  const bool hc8_ = HcInt8();
  const bool mtp_hc8_ = !hc8_ && MtpHc8();
  bool hc8_site_ = hc8_;
  int hc8_sites_ = 0;
  std::vector<Task> tasks_;
  std::atomic<size_t> ndone_{0};
};

}  // namespace crucible
