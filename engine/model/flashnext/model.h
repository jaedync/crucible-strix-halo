// Qwen3.8-Flash-Next EXL3 weights on the GPU: tensor layout conversion and a
// parallel loader. Everything lives in a few large device arenas.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
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
#include "mtp_sidecar.h"
#include "../../layers/ops.h"
#include "ple_host.h"
#include "../../exl3/row_policy.h"
#include "weight_validation.h"
#include "load_plan.h"
#include "../../gfx1151/warmup.h"
#include "../../exl3/static_layout.h"
#include "../../switches.h"

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
      ck_.AddFile("vision_k6.safetensors");
    if (!ck_.Has("mtp.hyper_connection_mixer.hc_norm.weight"))
      ck_.AddFile("mtp_hyper_connection_mixer_patch.safetensors");
    // A tuned MTP head changes only drafts; verification keeps the trunk exact.
    if (const char* side = crucible::SwitchText("CRUCIBLE_MTP_SIDECAR"); side && *side) {
      const size_t replaced = ck_.Override(side, "mtp.");
      std::fprintf(stderr, "mtp sidecar: %zu tensors from %s\n", replaced, MtpSidecarLabel(side).c_str());
    }
    // Fixed-size kernels cannot safely consume arbitrary checkpoint geometry.
    // Validate every source before the first allocation or conversion task.
    ValidateCheckpoint(ck_);
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

  // Fixed-size vision kernels need this geometry and one EXL3 width for every tower linear.
  void ValidateVision(const std::string& P) {
    int k = 0;
    auto lin = [&](const std::string& n, int I, int O) {
      const int kn = ValidateLinear(ck_, n, I, O);
      if (k && kn != k) ck_.Error(n + ".trellis", "every vision linear must use the same EXL3 width");
      if (kn < 3 || kn > 6) ck_.Error(n + ".trellis", "vision GEMMs need EXL3 K3..6");
      k = kn;
      ck_.Require(n + ".bias", {"F16"}, {O});
    };
    ck_.Require(P + "patch_embed.proj.weight", {"F16"}, {1152, 3, 2, 16, 16});
    ck_.Require(P + "patch_embed.proj.bias", {"F16"}, {1152});
    ck_.Require(P + "pos_embed.weight", {"F16", "BF16"}, {2304, 1152});
    for (int i = 0; i < 27; ++i) {
      const std::string B = P + "blocks." + std::to_string(i) + ".";
      for (const char* n : {"attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj"}) lin(B + n, 1152, 1152);
      lin(B + "mlp.linear_fc1", 1152, 4352);
      lin(B + "mlp.linear_fc2", 4352, 1152);
      for (const char* n : {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"})
        ck_.Require(B + n, {"F16", "BF16"}, {1152});
    }
    lin(P + "merger.linear_fc1", 4608, 4608);
    lin(P + "merger.linear_fc2", 4608, kHidden);
    ck_.Require(P + "merger.norm.weight", {"F16", "BF16"}, {1152});
    ck_.Require(P + "merger.norm.bias", {"F16", "BF16"}, {1152});
  }

  void Load(Model& m, bool streaming = false) {
    streaming_ = streaming;
    const std::string P = "model.language_model.";
    auto t0 = std::chrono::steady_clock::now();
    const size_t planned = streaming ? ExpectedStaticBytes(ck_, kLayers) : 0;
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
    LayerW("mtp.layers.0.", m.mtp.L, IsQsa(kMtpLayer));
    m.mtp.mixer = HcW("mtp.hyper_connection_mixer.", false);
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
    for (const auto& n : names) {
      if (Info(n).dtype != "F16") ck_.Error(n, "expected dtype F16");
      size_t b;
      const uint8_t* s = Src(n, &b);
      parts.push_back({s, b});
      total += b;
    }
    uint8_t* d = active_->Alloc(total);
    size_t off = 0;
    for (auto& [s, b] : parts) {
      tasks_.push_back({b, d + off, s, nullptr});
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
    h.up = Concat16({p + "input_mix_weight_up.weight"});
    const uint8_t* down = Src(p + "input_mix_weight_down.weight");
    const uint8_t* inj = inject ? Src(p + "block_inject_weight.weight") : nullptr;
    const uint8_t* nw = Src(p + "hc_norm.weight");
    const bool nbf = Info(p + "hc_norm.weight").dtype == "BF16";
    const int M = h.M;
    uint8_t* d = active_->Alloc((size_t)M * kWide * 2);
    tasks_.push_back({(size_t)M * kWide * 2, d, nullptr, [=](uint8_t* buf) {
                        _Float16* out = reinterpret_cast<_Float16*>(buf);
                        for (int m = 0; m < M; ++m) {
                          const uint8_t* row =
                              m < kRank ? down + (size_t)m * kWide * 2 : inj + (size_t)(m - kRank) * kWide * 2;
                          for (int c = 0; c < kWide; ++c)
                            out[(size_t)m * kWide + c] =
                                (_Float16)(ToF(Read16(row, c), false) * (1.f + ToF(Read16(nw, c), nbf)));
                        }
                      }});
    h.fn = reinterpret_cast<__half*>(d);
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

  static bool PleHost() {
    try {
      return RowPolicyInt("CRUCIBLE_PLE_HOST", 0, 0, 1) != 0;
    } catch (const std::invalid_argument& error) {
      Die(error.what());
    }
    return false;
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
  std::vector<Task> tasks_;
  std::atomic<size_t> ndone_{0};
};

}  // namespace crucible
