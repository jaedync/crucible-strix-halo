// Crucible: Qwen3.8-Flash-Next EXL3 inference on gfx1151.
//
//   crucible gen CKPT --ids 1,2,3 [--n 64] [--chunk 8] [--dump DIR] [--ctx 32768]
//
// Prints the greedy continuation as token ids plus prefill/decode timing.
// --dump writes the residual stack after the embedding and after every layer
// for every processed position: DIR/h_XX.f32, [positions][4][2560] float32.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>

#include "model/flashnext/model.h"
#include "switches.h"
#include "runtime/bounds.h"
#include "runtime/sampling.h"
#include "runtime/retained_prefix.h"
#include "runtime/layer_loop.h"
#include "runtime/conversation_cache.h"
#include "runtime/conversation_resume.h"
#include "runtime/batch_contract.h"
#include "runtime/batch_admission.h"
#include "runtime/batch_inputs.h"
#include "runtime/batch_spec_contract.h"
#include "runtime/sequence_state.h"
#include "runtime/device_zero.h"
#include "cli/gen_job_state.h"
#include "runtime/dump_io.h"
#include "gfx1151/pinned_buffer.h"
#include "gfx1151/pinned_allocator.h"
#include "layers/qsa_long.h"
#include "layers/qsa_capture.h"
#include "layers/kernels/hc_multirow.h"
#include "exl3/plans.h"
#include "runtime/draft_ops.h"
#include "runtime/draft_dense.h"
#include "runtime/draft_adapt.h"
#include "runtime/draft_policy.h"
#include "layers/expert_count.h"
#include "runtime/prefill_prune.h"
#include "layers/moe_rows.h"
#include "layers/moe_stage.h"
#include "layers/moe_shared_first.h"
#include "layers/dense_rows.h"
#include "exl3/exl3_ring.h"
#include "exl3/ring_rows.h"
#include "runtime/pass_rows.h"

#include "layers/gdn_rows.h"
#include "layers/persistent_gdn.h"
#include "layers/ablation_device.h"
#include "layers/prefill_moe.h"
#include "layers/prefill_moe_pair.h"
#include "runtime/prefill_front_check.h"
#include "layers/prefill_ple.h"
#include "runtime/prefill_plan_budget.h"
#include "model/flashnext/prefill_launch.h"
#include "runtime/mtp_bulk.h"
#include "runtime/stage_arena.h"
#include "layers/kernels/prefill_qsa_union.h"
#include "layers/prefill_hc.h"
#include "runtime/prefill_interleave.h"
#include "layers/hc_wmma.h"
#include "layers/kernels/prefill_gdn.h"
#include "layers/numerics.h"
#include "layers/kernels/prefill_qsa.h"
#include "model/flashnext/vision.h"

using namespace crucible;

struct ConversationStateIO;
namespace {

#include "runtime/draft_overlap.h"

#include "runtime/prefill_rows.h"
#if CRUCIBLE_DEBUG_NONFINITE & 8
__global__ void FlipLowBitKernel(float* p) { *p = __uint_as_float(__float_as_uint(*p) ^ 1u); }
#endif
static_assert(kPrefillCoreRows == kPrefillLaunchCoreRows, "merged launches cut GDN/QSA parts on the grid");

class Engine {
  friend struct EngineStateIO;  // long-context benchmark checkpoints, not a serving API
  friend struct ::ConversationStateIO;
 public:
  // mtp_rows: positions of each slot's MTP cache (batch serve MtpSlotRows); 0 keeps the context.
  Engine(const std::string& dir, int max_ctx, int prefill_rows = 0, int mtp_rows = 0)
      : ctx_(max_ctx), mtp_rows_(mtp_rows) {
    RowPolicy explicit_policy;
    try {
      explicit_policy = ResolveRowPolicy(1, 0, 0);
    }
    catch (const std::invalid_argument& error) {
      std::fprintf(stderr, "row policy error: %s; correct or unset that environment variable\n", error.what());
      std::exit(2);
    }
    try { compact_state_ = EngineCompactState(crucible::SwitchText("CRUCIBLE_COMPACT_STATE")); }
    catch (const std::invalid_argument& error) {
      std::fprintf(stderr, "%s\n", error.what());
      std::exit(2);
    }
    const bool streaming = explicit_policy.blocks != 0 || RowPolicyInt("CRUCIBLE_TUNE_ROWS", 1, 0, 1) != 0;
    Loader(dir).Load(m_, streaming);
    // Decide the prefill path before the row-sized buffers.
    prefill_rows_ = ChoosePrefillRows(prefill_rows);
    const char* launch_value = crucible::SwitchText("CRUCIBLE_PF_LAUNCH_ROWS");
    pf_launch_rows_ = PrefillLaunchSetting(launch_value, prefill_rows_);
    if (pf_launch_rows_ < 0) {
      std::fprintf(stderr, "CRUCIBLE_PF_LAUNCH_ROWS must be a multiple of the prefill rows (%d) up to %d, "
                   "and above them only for prefill rows that are a multiple of %d; a \"+T\" suffix (join rows, "
                   "x4-tail) needs launch rows above the prefill rows, T from 1 to the prefill rows less 1, and the "
                   "sum at most %d\n",
                   prefill_rows_, kMaxPrefillLaunchRows, kPrefillLaunchCoreRows, kMaxPrefillLaunchRows);
      std::exit(2);
    }
    // x4-tail: join rows only with "L+T"; else 0, and the row buffers hold the launch rows as before.
    pf_join_rows_ = PrefillJoinSetting(launch_value, prefill_rows_);
    pf_room_rows_ = PrefillRoomRows(pf_launch_rows_, pf_join_rows_);
    lookahead_.Init(m_.static_base, m_.static_bytes, streaming);
    // Row buffers hold a packed verify pass (kPackedRows) or one bulk launch with its join rows (PrefillRoomRows).
    const int capacity = std::max(kPackedRows, pf_room_rows_);
    // Padding avoids cache conflicts. CRUCIBLE_PF_PAD_H=0 keeps tight storage for A/B checks.
    const char* pad_h = crucible::SwitchText("CRUCIBLE_PF_PAD_H");
    pf_pad_h_ = prefill_rows_ > 0 && (!pad_h || std::string(pad_h) == "1");
    H_ = Scratch<float>((size_t)TightStackRows(capacity, pf_pad_h_, kPackedRows) * kWide);
    x_ = Scratch<float>((size_t)capacity * kHidden);
    dots_ = Scratch<float>((size_t)capacity * (kRank + kStreams) * kStreams);
    ss_ = Scratch<float>(capacity * kStreams);
    post_ = Scratch<float>(capacity * kStreams);
    AllocateStageArena(capacity);
    rlog_ = Scratch<float>(capacity * (kExperts + 1));
    wts_ = Scratch<float>(capacity * kSlots);

    logits_ = Scratch<float>((size_t)kPackedRows * kVocab);
    tok_ = Zeros<int>(capacity);
    rows_ = Zeros<int64_t>(capacity * 16);
    if (m_.ple.host) {
      ple_packed_ = Zeros<uint16_t>((size_t)capacity * 16 * m_.ple.RowWords());
      // A BF16 row (160 words) does not fit the K-bit stage, which keeps its size.
      if (m_.ple.bf16) ple_stage16_ = std::make_unique<PinnedInputBuffer<uint16_t, kMaxPrefillRows * 16 * kPleRowDim>>();
      // An 8-bit row (the h6_ng8 table) has 81 words.
      else if (m_.ple.RowWords() > 61) ple_stage8_ = std::make_unique<PinnedInputBuffer<uint16_t, kPleStageRows * 16 * 81>>();
      else ple_stage_ = std::make_unique<PinnedInputBuffer<uint16_t, kPleStageRows * 16 * 61>>();
    }
    ids_ = Zeros<int>(kPackedRows);
    choice_ = Zeros<DraftChoice>(kMaxRows);
    mtp_stacks_ = Scratch<float>((size_t)kMaxRows * kWide);
    xh_capacity_ = std::max({kXh, (size_t)capacity * 10240, MoeGateUpHalves(kPackedRows)});
    xh_ = Scratch<__half>(xh_capacity_);
    mE_ = Scratch<float>(kMaxRows * kHidden);
    mN_ = Scratch<float>(kMaxRows * kWide);
    mhp_ = Scratch<float>(kMaxRows * kWide);
    mep_ = Scratch<float>(kMaxRows * kHidden);
    AddSequence();
    sequence_ = sequences_[0].get();
    // QSA runs at most kPrefillCoreRows rows per call.
    scores_ = Scratch<float>((size_t)std::min(capacity, kPrefillCoreRows) * (ctx_ / 4));
    // A packed pass of up to kPackedRows rows indexes split partials by its
    // row in the pass (12.7 MB at 32 splits).
    qsa_scratch_ = Scratch<float>(QsaScratchFloats(kPackedRows));
    sel_ = Zeros<int>(capacity * kSelBlocks);
    nsel_ = Zeros<int>(capacity);
    sel_moe_ = Zeros<int>(capacity * kTopK);
    map_ = Zeros<int>(kPackedRows * kTopK);
    mapj_ = Zeros<int>(kPackedRows * kTopK);
    rowl_ = Zeros<int>(MoeExpertJobs(kPackedRows) * kMaxRows);
    moe_done_ = Zeros<unsigned>(MoeExpertJobs(kPackedRows));
    moe_sync_ = Zeros<unsigned>(kMoeStageWords);
    qsa_sync_ = Zeros<unsigned>(4);
    hcw_ = HcWAlloc();
    LoadAblation(capacity);
    if (prefill_rows_) InitPrefill();
    std::fprintf(stderr, "weights %.2f GB, context %d, prefill rows %d\n", m_.bytes / 1e9, ctx_, prefill_rows_);
    std::fprintf(stderr, "prefill: %s\n", prefill_path_.c_str());
    if (compact_state_)
      std::fprintf(stderr, "sequence storage: CRUCIBLE_COMPACT_STATE=1 raw_slots=%d mtp=lazy logical_state_bytes=%zu\n",
                   kQsaRawSlots, SequenceState::Bytes(ctx_, true, false, mtp_rows_));
    if (mtp_rows_ > 0 && mtp_rows_ < ctx_) std::fprintf(stderr, "sequence storage: mtp_rows=%d of context %d\n", mtp_rows_, ctx_);
    const char* warmup = crucible::SwitchText("CRUCIBLE_WARMUP");
    if (!warmup || std::string(warmup) != "0") WarmUp();
    else std::fprintf(stderr, "warmup disabled by CRUCIBLE_WARMUP=0\n");
    ReportMemory("ready");
  }

  // Retain full output-Hadamard blocks, not individual trellis columns.
  void DraftHead(const std::vector<int>& blocks, bool last_only, bool dense = false, bool adapt = false) {
    // A resident harness reconfigures the head between requests. Dense mode
    // owns its uploaded selection; block mode's mapping belongs to block_head_.
    if (draft_dense_) {
      HIP_OK(hipFree(draft_dense_));
      if (draft_blocks_) HIP_OK(hipFree(draft_blocks_));
    }
    block_head_.reset();
    draft_blocks_ = nullptr;
    draft_dense_ = nullptr;
    draft_size_ = 0;
    draft_last_only_ = last_only || !blocks.empty();
    if (blocks.empty()) return;
    const int limit = dense ? kVocab : kVocab / 128;
    if (!ValidDraftSelection(blocks, limit)) {
      std::fprintf(stderr, "draft selection must be sorted unique IDs in [0, %d)\n", limit);
      std::exit(2);
    }
    if (dense) {
      draft_blocks_ = Upload(blocks);
      draft_size_ = (int)blocks.size();
      draft_dense_ = PackDraftDense(m_.head, blocks);
      std::fprintf(stderr, "draft head %d tokens, %.2f MB, fp16 selected rows\n",
                   draft_size_, (double)draft_size_ * kHidden * sizeof(__half) / 1e6);
      return;
    }
    block_head_ = std::make_unique<BlockDraftHead>(m_.head, blocks, adapt);
    RebuildDraftHead();
    std::fprintf(stderr, "draft head %d tokens, %.2f MB, original EXL3 blocks\n",
                 draft_head_.O, (double)draft_head_.O * (m_.head.I * m_.head.K / 8 + 2) / 1e6);
  }

  void ObserveDraft(const int* tokens, int count) {
    if (block_head_ && block_head_->Observe(tokens, count)) RebuildDraftHead();
  }
  float DraftMargin() const { return draft_margin_; }
  int DraftVocabulary() const { return draft_dense_ ? draft_size_ : draft_head_.O; }
  // Bytes of one row of the block draft head (EXL3 tiles plus the fp16 scale), as in the DraftHead line above.
  size_t DraftTokenBytes() const { return (size_t)m_.head.I * m_.head.K / 8 + 2; }

  void Reset() { sequence_->Reset(); }

  // gen --jobs: returns every piece of per-request state to its value after construction.
  // Weights, decode and prefill plans, and warmup stay.
  void ResetForJob() {
    HIP_OK(hipDeviceSynchronize());
    for (const DeviceSpan& span : sequence_->Spans())
      HIP_OK(hipMemset(const_cast<void*>(span.pointer), 0, span.bytes));
    sequence_->hist.clear();
    sequence_->retained.Reset();
    DraftHead({}, true);
    draft_margin_ = INFINITY;
    overlap_.reset();
    for (auto& mark : marks_) HIP_OK(hipEventDestroy(mark.first));
    marks_.clear();
    prof_ = false;
    ptime_.clear();
    ecount_ = nullptr;
    ClearExpertCounts();
    HIP_OK(hipDeviceSynchronize());
  }

#if CRUCIBLE_DEBUG_NONFINITE & 13
  // Debug hooks after each trunk layer (Dump calls this for every pass, also without a dump directory).
  // idx: layers done (0 after the embedding). H: the pass's rows, ldh floats apart.
  void DebugAfterLayer(int idx, int R, float* H, int ldh) {
#if CRUCIBLE_DEBUG_NONFINITE & 1
    ScanNonfiniteRows(H, R, ldh, kWide);
#endif
#if CRUCIBLE_DEBUG_NONFINITE & 8
    // FLIP: the real-binary negative control of the gates. One low bit of row 0 after layer 1, every pass.
    if (idx == 2) FlipLowBitKernel<<<1, 1>>>(H);
#endif
#if CRUCIBLE_DEBUG_NONFINITE & 4
    InjectAfterLayer(idx, R, H, ldh);
#endif
  }
#endif
#if CRUCIBLE_DEBUG_NONFINITE & 4
  // Inject probe: the plan comes from ParseInjectPlan (layers/nonfinite_probe.h).
  std::optional<InjectPlan> inject_;
  void SetInject(const std::string& spec) { inject_ = ParseInjectPlan(spec); }
  // (stream, position) of row r of the current pass. Every pass appends its tokens to hist first.
  std::pair<int, int> RowAt(int r, int R) const {
    for (const auto& span : verify_spans_)
      if (r >= span.row && r < span.row + span.rows) return {span.stream, span.pos + r - span.row};
    if (!batch_streams_.empty()) {
      const int s = batch_streams_[r];
      return {s, (int)sequences_[s]->hist.size() - 1};
    }
    return {ActiveStream(), (int)sequence_->hist.size() - R + r};
  }
  void InjectAfterLayer(int idx, int R, float* H, int ldh) {
    if (!inject_ || inject_->done || idx != inject_->layer + 1) return;
    InjectPlan& p = *inject_;
    for (int r = 0; r < R; ++r) {
      const auto [stream, pos] = RowAt(r, R);
      if (stream != p.stream || pos < p.pos || size_t(stream) >= sequences_.size()) continue;
      SequenceState& s = *sequences_[stream];
      void* at;
      if (p.kind == "hidden") at = H + (size_t)r * ldh;
      else if (p.kind == "kv") at = s.qc.at(p.layer).k + (size_t)p.pos * 512;
      else if (p.kind == "index") at = s.qc.at(p.layer).ik_pool + QsaPoolOffset(p.pos / 4, 0);
      else at = s.gstate.at(p.layer);
      if (p.kind == "kv") HIP_OK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(at), 0x7e00, 1, 0));
      else HIP_OK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(at), 0x7fc00000, 1, 0));
      p.done = true;
      std::fprintf(stderr, "debug inject: %s layer %d stream %d position %d (row %d of %d, at position %d)\n",
                   p.kind.c_str(), p.layer, stream, p.pos, r, R, pos);
      return;
    }
  }
#endif

  // Reads the state that ResetForJob must clear, including device buffers. scratch: caller buffers that must be zero.
  // This checks the reset, not completeness: projection and prefill scratch is not cleared or checked.
  GenJobState JobState(const std::vector<DeviceSpan>& scratch) {
    GenJobState s;
    s.history = sequence_->hist.size();
    s.retained_clean = sequence_->retained.Clean();
    s.sequences = sequences_.size();
    s.active_first = !sequences_.empty() && sequence_ == sequences_[0].get();
    s.batch_streams = batch_streams_.size();
    s.verification_spans = VerificationSpansActive();
    s.sequence_zero = DeviceSpansZero(sequence_->Spans());
    s.scratch_zero = DeviceSpansZero(scratch);
    s.draft_selection = draft_blocks_ || draft_dense_ || block_head_;
    s.draft_last_only = draft_last_only_;
    s.draft_margin_reset = std::isinf(draft_margin_) && draft_margin_ > 0;
    s.overlap_head = overlap_ != nullptr;
    s.profile = prof_;
    s.profile_entries = ptime_.size();
    s.profile_marks = marks_.size();
    s.expert_counting = ecount_ != nullptr;
    s.expert_counts_zero = !ecbuf_ || DeviceSpansZero({{ecbuf_, sizeof(unsigned long long) * 4 * (kMaxRows + 1)}});
    return s;
  }

  // Runs rows tokens[0..R) at positions pos0..pos0+R-1; returns greedy ids.
  // replay: accepted draft rows of the previous forward to fold into GDN state;
  // commit: rows of this forward whose GDN updates are certain (R unless verifying).
  std::vector<int> Forward(const int* tokens, int R, int replay, int commit,
                           const std::string& dump = "", int mtp_pos = -1) {
    const int pos0 = (int)sequence_->hist.size();
    if (!ContextRowsFit(pos0, R, sequence_->capacity, kMaxRows)) {
      std::fprintf(stderr, "context full or invalid forward row count\n");
      std::exit(1);
    }
    if (!ValidTokenIds(std::span(tokens, R), kVocab)) {
      std::fprintf(stderr, "token IDs must be in [0, %d)\n", kVocab);
      std::exit(2);
    }
    if (mtp_pos >= 0 && !ContextRowsFit(mtp_pos, R - 1, sequence_->mtp_capacity, kMaxRows)) {
      std::fprintf(stderr, "invalid overlapping MTP window\n");
      std::exit(2);
    }
    const std::string state_error = sequence_->retained.Forward(R, replay, commit);
    if (!state_error.empty()) {
      std::fprintf(stderr, "forward state error: %s; use Reset() to start a new sequence\n", state_error.c_str());
      std::exit(1);
    }
    sequence_->hist.insert(sequence_->hist.end(), tokens, tokens + R);
    const Plans& P = PlansFor(R);
    lookahead_.Start(P.policy);
    host_tokens_.CopyTo(tok_, tokens, R);
    host_rows_.FillTo(rows_, R * 16, [&](int64_t* rows) { PleRows(pos0, R, rows); });
    PleRowsUploaded(host_rows_.data(), R * 16);
    profile_scope_ = "verify:";
    Mark("start");
    EmbedKernel<<<dim3(kHidden / 256, R), 256>>>(m_.embed, tok_, H_);
    OverlayImages(*sequence_, pos0, R, H_, kWide);
    Mark("embed");
    Dump(dump, 0, R);
    RunTrunk([&](int) { Ple(R, pos0); Mark("ple"); },
             [&](int l) { Block(l, m_.L[l], P.L[l], R, pos0, replay, commit); },
             [&](int l) { Dump(dump, l + 1, R); });
    if (mtp_pos >= 0) {
      if (!overlap_) overlap_ = std::make_unique<AsyncHead>();
      HcRead(m_.final_hc, R, false);
      Mark("hc_mix");
      lookahead_.Stop();
      overlap_->Begin(P.head, x_, H_, R);
      // Pair each known next token with its exact trunk stack. On rejection,
      // later calls overwrite the unused suffix of the MTP cache.
      Mtp(overlap_->Stacks(), tokens + 1, R - 1, mtp_pos, false);
      return overlap_->Finish(H_, R);
    }
    return Head(m_.final_hc, P, R);
  }

  // One MTP step for R consecutive rows at MTP positions mpos..: inputs are
  // trunk (or draft) stacks [R][10240] on the device and the next tokens.
  // Leaves each row's stack in H(). By default only the last ID is scored;
  // earlier IDs are -1. score=false skips the head and returns no IDs.
  std::vector<int> Mtp(const float* stacks, const int* tokens, int R, int mpos,
                       bool score = true, bool full_head = false) {
    if (!ContextRowsFit(mpos, R, sequence_->mtp_capacity, kMaxRows)) {
      std::fprintf(stderr, "MTP context full or invalid row count\n");
      std::exit(1);
    }
    if (!ValidTokenIds(std::span(tokens, R), kVocab)) {
      std::fprintf(stderr, "token IDs must be in [0, %d)\n", kVocab);
      std::exit(2);
    }
    EnsureMtpCache(*sequence_, mpos);
    const Plans& P = PlansFor(R);
    lookahead_.Start(P.policy);
    host_tokens_.CopyTo(tok_, tokens, R);
    profile_scope_ = "mtp:";
    Mark("start");
    MtpPrepKernel<<<R, 1024>>>(m_.embed, tok_, stacks, m_.mtp.ne1, m_.mtp.nh1, mE_, mN_);
    RunGroup(P.mtp_in, true, true);
    MtpFuseKernel<<<dim3(kWide / 256, R), 256>>>(mhp_, mep_, H_);
    Mark("mtp_in");
    Block(kLayers, m_.mtp.L, P.L[kLayers], R, mpos, 0, R);
    if (!score) {
      lookahead_.Stop();
      Collect();
      return {};
    }
    if (!draft_last_only_) return Head(m_.mtp.mixer, P, R, true);
    const Hc& hc = m_.mtp.mixer;
    const float* last = H_ + (size_t)(R - 1) * kWide;
    if (hc_wmma_) {
      HcWLaunch(hc.fn, hc.M, hc.up, hc.w1, last, 1, x_, nullptr, hcw_, ++hcw_epoch_);
    } else {
      HcDotsKernel<1><<<((hc.M + 1) * kStreams + 7) / 8, 256>>>(hc.fn, hc.M, last, dots_, ss_);
      Mark("hc_dots");
      HcMixKernel<1><<<kHidden / 16, 256>>>(hc.up, hc.w1, dots_, hc.M, ss_, last, x_, nullptr);
    }
    Mark("hc_mix");
    const bool dense_head = draft_dense_ && !full_head;
    const bool block_head = draft_blocks_ && !draft_dense_ && !full_head;
    const Group& head = block_head ? draft_group_ : PlansFor(1).head;
    if (dense_head) {
      LinInKernel<<<(head.nin + 7) / 8, 256>>>(head.d_jobs, head.d_in, head.nin);
      DraftDenseGemvKernel<<<(draft_size_ + 7) / 8, 256>>>(draft_dense_, head.jobs[0].xh, logits_, draft_size_);
    } else RunGroup(head, true, true);
    DraftChoiceKernel<<<1, 1024>>>(logits_, dense_head ? draft_size_ : block_head ? draft_head_.O : kVocab,
                                  full_head ? nullptr : draft_blocks_, choice_, dense_head);
    Mark("draft_head");
    lookahead_.Stop();
    Collect();
    DraftChoice choice;
    HIP_OK(hipMemcpy(&choice, choice_, sizeof(choice), hipMemcpyDeviceToHost));
    draft_margin_ = choice.margin;
    std::vector<int> out(R, -1);
    out.back() = choice.token;
    return out;
  }

  // Builds the plans before the ready report, so the first packed step does
  // not allocate and the startup memory report holds the 9..16-row plans.
  void WarmPlans() {
    for (int r = 1; r <= kMaxRows; ++r) PlansFor(r);
    if (sequences_.size() > 1 && MaxPassRows() > kMaxRows)
      for (int r = kMaxRows + 1; r <= kPackedRows; ++r) PlansFor(r);
  }

  // Rows one packed verify pass may hold (pass_rows.h): kMaxRows when a static
  // group has neither front order nor chunk-aligned spans. The first call
  // with packing allowed builds the 16-row plans.
  int MaxPassRows() {
    if (max_pass_rows_) return max_pass_rows_;
    const Plans& P = PlansFor(kPackedRows);
    bool fit = true;
    for (int l = 0; l < kLayers && fit; ++l) fit = PackedFits(P.L[l].a_in) && PackedFits(P.L[l].a_out);
    fit = fit && PackedFits(P.head);
    max_pass_rows_ = crucible::MaxPassRows(fit);
    return max_pass_rows_;
  }

  static void ReportMemory(const char* phase) {
    size_t free, total;
    HIP_OK(hipMemGetInfo(&free, &total));
    std::fprintf(stderr, "memory %s: device free %zu / %zu bytes\n", phase, free, total);
    std::ifstream mem("/proc/meminfo");
    std::string line;
    while (std::getline(mem, line))
      if (line.starts_with("MemAvailable:"))
        std::fprintf(stderr, "memory %s: %s\n", phase, line.c_str());
  }

  // between: optional decode passes between layers (prefill_interleave.h); ignored where InterleaveAllowed fails.
  // keep: the first row whose final trunk stacks (H()) the caller reads after the call; R if none. A chunk
  // without predict and without kept rows skips the last layer's output (PrefillPrunes).
  std::vector<int> Prefill(const int* tokens, int R, const std::string& dump = "", bool predict = true,
                           const PrefillBetween* between = nullptr, int keep = 0);
  // Bulk MTP warm-up (mtp_bulk.h): writes the MTP cache rows of `count` pairs at MTP positions mpos.. with the
  // prefill kernels and returns true, or returns false (nothing written) when the 8-row path must run. Pair j pairs
  // stack j with tokens[j]: stack 0 is `first` when set, and the others are consecutive rows of `rows` (kWide floats
  // each). It overwrites H() and the row scratch, so callers copy the stacks they still need first.
  bool MtpWarm(const float* first, const float* rows, const int* tokens, int count, int mpos);
  // mtp_rows: committed rows of the MTP cache; stk and carry: the MTP pair buffers (optional).
  void DumpState(const std::string& dir, int mtp_rows = 0, const float* stk = nullptr, const float* carry = nullptr);
  void ResetProfile() { ptime_.clear(); }
  // The final trunk stacks of the last decode, verify or small-row pass (at most kPackedRows rows). A bulk Prefill
  // leaves its rows in the padded buffer: read them with CopyPrefillStacks.
  float* H() { return H_; }
  // Copies the final trunk stacks of rows [first, first + rows) of the last Prefill call to dst (tight rows, null
  // stream). Call it before any other device work, which may write that buffer.
  void CopyPrefillStacks(float* dst, int first, int rows) const {
    if (first < 0 || rows < 1 || first + rows > last_prefill_rows_) std::abort();
    const int ldh = PrefillHStride(last_prefill_rows_);
    HIP_OK(hipMemcpy2DAsync(dst, kWide * sizeof(float), PrefillH(last_prefill_rows_) + (size_t)first * ldh,
                            ldh * sizeof(float), kWide * sizeof(float), rows, hipMemcpyDeviceToDevice, 0));
  }
  void Truncate(size_t n) {
    const std::string state_error = sequence_->retained.Truncate(n);
    if (!state_error.empty()) {
      std::fprintf(stderr, "truncate state error: %s; use Reset() to start a new sequence\n", state_error.c_str());
      std::exit(1);
    }
    sequence_->hist.resize(n);
  }
  size_t Length() const { return sequence_->hist.size(); }
  int Context() const { return ctx_; }

  const float* logits() const { return logits_; }
  SequenceState& ActiveSequence() { return *sequence_; }

  // ---- image input (serve --vision 1) ----
  void EnableVision(const std::string& dir) {
    if (vision_) return;
    vision_ = std::make_unique<VisionEncoder>(dir, partial_, partial_capacity_, xh_, xh_capacity_);
    ReportMemory("vision");
  }
  bool HasVision() const { return vision_ != nullptr; }

  // Binds a request's images to state before its prompt runs. Images that end inside the first
  // `cached` prompt tokens come from a restored snapshot: they need no encoder pass and no pixels.
  // Deferred binding leaves the encoder work to EncodeVisionSlice (batch serve runs it between other
  // sessions' commands); otherwise it runs here. Returns the encoder seconds (device time included).
  double BindImages(SequenceState& state, const std::vector<ImageInput>& images, size_t cached,
                    bool deferred = false) {
    ClearImages(state);
    if (images.empty()) return 0;
    if (!vision_) throw std::runtime_error("this engine runs without the vision tower; start serve with --vision 1");
    const auto t0 = std::chrono::steady_clock::now();
    const auto table = MropeTable(images, size_t(state.capacity));
    if (!state.mrope) HIP_OK(hipMalloc(&state.mrope, table.size() * sizeof(int)));
    HIP_OK(hipMemcpy(state.mrope, table.data(), table.size() * sizeof(int), hipMemcpyHostToDevice));
    for (const size_t i : ImagesToEncode(images, cached)) {
      state.vision_images.push_back({images[i]});
      state.image_spans.push_back({images[i].offset, images[i].Tokens(), nullptr});
    }
    state.mrope_on = true;
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!deferred)
      while (VisionSlicesLeft(state)) seconds += EncodeVisionSlice(state);
    return seconds;
  }
  int VisionSlicesLeft(const SequenceState& state) const {
    return VisionSlicesBefore(state, std::numeric_limits<size_t>::max());
  }
  // Slices still owed by images that start before prompt row `end`: a chunk that ends there reads them.
  int VisionSlicesBefore(const SequenceState& state, size_t end) const {
    int left = -state.vision_slice;
    for (size_t i = state.vision_next; i < state.vision_images.size(); ++i)
      if (size_t(state.vision_images[i].image.offset) < end) left += VisionEncoder::Slices(state.vision_images[i].image);
    return std::max(left, 0);
  }
  // One encoder slice of the request's first unfinished image. Slice 0 reads and checks the image file and
  // allocates the image's embeddings.
  double EncodeVisionSlice(SequenceState& state) {
    if (state.vision_next >= state.vision_images.size()) throw std::logic_error("no encoder work is pending");
    const auto t0 = std::chrono::steady_clock::now();
    auto& bound = state.vision_images[state.vision_next];
    if (state.vision_slice == 0) {
      PrepareVisionPixels(state);  // serial serve reads here; batch serve read before the command's device work
      HIP_OK(hipMalloc(&bound.embeds, size_t(bound.image.Tokens()) * kHidden * sizeof(uint16_t)));
      state.image_spans[state.vision_next].embeds = bound.embeds;
      if (!state.vision_x) {
        size_t residual = 0;
        for (size_t i = state.vision_next; i < state.vision_images.size(); ++i)
          residual = std::max(residual, VisionEncoder::ResidualFloats(state.vision_images[i].image));
        HIP_OK(hipMalloc(&state.vision_x, residual * sizeof(float)));
      }
    }
    vision_->EncodeSlice(bound.image, state.vision_slice == 0 ? state.vision_rgb.data() : nullptr, state.vision_x,
                         state.vision_slice, bound.embeds);
    // The host pixels and the stall bound both need the slice to finish here.
    HIP_OK(hipDeviceSynchronize());
    if (state.vision_slice == 0) std::vector<uint8_t>().swap(state.vision_rgb);
    if (++state.vision_slice == VisionEncoder::Slices(bound.image)) {
      ++state.vision_next;
      state.vision_slice = 0;
    }
    if (state.vision_next == state.vision_images.size() && state.vision_x) {
      HIP_OK(hipFree(state.vision_x));
      state.vision_x = nullptr;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  // Host only: reads and checks the file of the next image before its first slice. Throws std::runtime_error (a
  // fixed text) when the file is gone, short, or does not match the digest; no device state changes.
  void PrepareVisionPixels(SequenceState& state) const {
    if (state.vision_next >= state.vision_images.size() || state.vision_slice || !state.vision_rgb.empty()) return;
    state.vision_rgb = ReadImagePixels(state.vision_images[state.vision_next].image);
  }
  // Frees the embeddings of encoded images that end at or before prompt row `cursor`: no later chunk reads them.
  void ReleaseImageRows(SequenceState& state, size_t cursor) {
    bool synced = false;
    for (size_t i = 0; i < state.vision_next; ++i) {
      auto& bound = state.vision_images[i];
      if (!bound.embeds || size_t(bound.image.End()) > cursor) continue;
      if (!synced) { HIP_OK(hipDeviceSynchronize()); synced = true; }
      HIP_OK(hipFree(bound.embeds));
      bound.embeds = nullptr;
      state.image_spans[i].embeds = nullptr;
    }
  }
  void ClearImages(SequenceState& state) {
    const bool held = state.vision_x || std::any_of(state.vision_images.begin(), state.vision_images.end(),
                                                    [](const auto& bound) { return bound.embeds != nullptr; });
    if (held) HIP_OK(hipDeviceSynchronize());
    for (const auto& bound : state.vision_images) if (bound.embeds) HIP_OK(hipFree(bound.embeds));
    if (state.vision_x) HIP_OK(hipFree(state.vision_x));
    state.vision_x = nullptr;
    state.vision_images.clear();
    std::vector<uint8_t>().swap(state.vision_rgb);
    state.image_spans.clear();
    state.vision_next = 0;
    state.vision_slice = 0;
    state.mrope_on = false;
  }
  // Startup admission (train13-audit L1): the encoder scratch at the largest image, and per slot the residual stream
  // and the embeddings of the images that can overlap one bulk launch (a merged launch can pass the grid).
  uint64_t VisionReserveBytes(const std::vector<int>& slot_contexts) const {
    if (!vision_) return 0;
    static_assert(kMaxRows == 8, "VisionReserveRows takes 8 rows without bulk prefill");
    const int chunk = VisionReserveRows(PrefillRows(), pf_room_rows_);
    uint64_t bytes = VisionEncoder::ScratchBytesFor(kVisionMaxPixels / 256, size_t(kVisionMaxPixels) * 3);
    bytes -= std::min<uint64_t>(bytes, vision_->ScratchBytes());
    for (const int context : slot_contexts) bytes += VisionSlotBytes(context, chunk, kHidden);
    return bytes;
  }
  // Test hook: encoder output of one image, bf16 rows copied to the host.
  std::vector<uint16_t> EncodeImage(const ImageInput& image, const std::vector<uint8_t>& rgb) {
    if (!vision_) throw std::runtime_error("vision is not enabled");
    uint16_t* out = nullptr;
    HIP_OK(hipMalloc(&out, size_t(image.Tokens()) * kHidden * sizeof(uint16_t)));
    vision_->Encode(image, rgb.data(), out);
    std::vector<uint16_t> host(size_t(image.Tokens()) * kHidden);
    HIP_OK(hipMemcpy(host.data(), out, host.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    HIP_OK(hipFree(out));
    return host;
  }
  void ResetMtp() {
    ResetQsaCache(sequence_->qc, kLayers, sequence_->mtp_capacity,
                  [](void* pointer, size_t bytes) { HIP_OK(hipMemset(pointer, 0, bytes)); });
  }
  int Vocabulary() const { return kVocab; }
  void Synchronize() const { HIP_OK(hipStreamSynchronize(0)); }

  std::vector<float> CopyLogits(int rows, int first = 0) const {
    if (rows < 1 || first < 0 || rows + first > kPackedRows)
      throw std::runtime_error("invalid sampling logit rows");
    const float* host = HostLogits(logits_ + size_t(first) * kVocab, size_t(rows) * kVocab);
    return {host, host + size_t(rows) * kVocab};
  }
  // One staging buffer for sampled logits: a pinned allocation and free per copy cost more than the copy.
  const float* HostLogits(const float* device, size_t count) const {
    if (logit_host_.size() < size_t(kPackedRows) * kVocab) logit_host_.resize(size_t(kPackedRows) * kVocab);
    if (count > logit_host_.size()) throw std::runtime_error("invalid sampling logit count");
    HIP_OK(hipMemcpy(logit_host_.data(), device, count * sizeof(float), hipMemcpyDeviceToHost));
    return logit_host_.data();
  }

  // Selected draft columns can change after ObserveDraft(). Snapshot the actual
  // mapping with the logits, not the original selection or the next selection.
  std::pair<std::vector<float>, std::vector<int>> DraftLogits(int rows) const {
    const int vocabulary = draft_dense_ ? draft_size_ : draft_blocks_ ? draft_head_.O : kVocab;
    const int first = draft_last_only_ ? 0 : rows - 1;
    if (rows < 1 || rows > kMaxRows || vocabulary < 1)
      throw std::runtime_error("invalid sampled draft head");
    const float* host = HostLogits(logits_ + size_t(first) * kVocab, size_t(vocabulary));
    std::vector<int> ids;
    if (draft_blocks_) {
      PinnedVector<int> selection(draft_dense_ ? vocabulary : vocabulary / 128);
      HIP_OK(hipMemcpy(selection.data(), draft_blocks_, selection.size() * sizeof(int), hipMemcpyDeviceToHost));
      if (draft_dense_) ids.assign(selection.begin(), selection.end());
      else {
        ids.reserve(vocabulary);
        for (int block : selection) for (int column = 0; column < 128; ++column) ids.push_back(block * 128 + column);
      }
    }
    return {{host, host + vocabulary}, std::move(ids)};
  }
  // Rows per Engine::Prefill chunk; 0 means the prompt uses Forward (8 rows).
  int PrefillRows() const { return prefill_rows_; }
  // Most rows of one Prefill call: whole PrefillRows() chunks that a caller merges
  // (PrefillLaunchRows in prefill_launch.h); CRUCIBLE_PF_LAUNCH_ROWS, default 8192 for the 4,096-row grid
  // (kDefaultPrefillLaunchRows, with kDefaultPrefillJoinRows join rows), else PrefillRows().
  int PrefillLaunchRows() const { return pf_launch_rows_; }
  // x4-tail: rows that the prompt's last partial chunk may add to a launch; 0 without
  // "+T" in CRUCIBLE_PF_LAUNCH_ROWS.
  int PrefillJoinRows() const { return pf_join_rows_; }
  // x4-tail: the next Prefill call sets the retained ledger of the plan without joins and copies the slot's state at
  // the cut (section 5). Only an engine with join rows takes a spec, and only one per call.
  void SetPrefillJoin(crucible::PrefillJoinSpec spec) {
    if (pf_join_rows_ <= 0 || pf_join_set_) std::abort();
    pf_join_ = std::move(spec);
    pf_join_set_ = true;
  }
  // PrefillRows() trunk stacks for MTP prompt pairing (nullptr without prefill).
  float* PromptStacks() const { return pf_prompt_stacks_; }

  // Per-stage GPU time via events between consecutive marks (CRUCIBLE_PROFILE=1).
  void EnableProfile(bool on = true) {
    prof_ = on;
    ptime_.clear();
  }

  // Distinct routed experts per MoE call: [scope][R] = {sum, calls}, scope 0
  // trunk and 1 MTP.
  void CountExperts(bool on) {
    if (!ecbuf_) ecbuf_ = Zeros<unsigned long long>(4 * (kPackedRows + 1));
    HIP_OK(hipMemset(ecbuf_, 0, sizeof(unsigned long long) * 4 * (kPackedRows + 1)));
    ecount_ = on ? ecbuf_ : nullptr;
  }
  void ClearExpertCounts() {
    if (ecbuf_) HIP_OK(hipMemset(ecbuf_, 0, sizeof(unsigned long long) * 4 * (kPackedRows + 1)));
  }
  std::vector<unsigned long long> ExpertCounts() const {
    std::vector<unsigned long long> v(4 * (kPackedRows + 1));
    if (ecbuf_) HIP_OK(hipMemcpy(v.data(), ecbuf_, v.size() * sizeof(v[0]), hipMemcpyDeviceToHost));
    return v;
  }
  // Benchmark hook: new GEMM row tiles take effect on the next plan build.
  void SetPrefillTiles(const PrefillTiles& t) {
    pf_tiles_ = t;
    FreePrefillPlans(false);
  }
  void DisableProfile() { prof_ = false; }
  void Report(int forwards) {
    if (!prof_ || forwards <= 0) return;
    double total = 0;
    for (auto& [k, v] : ptime_) total += v;
    std::fprintf(stderr, "profile over %d forwards: %.3f ms/forward\n", forwards, total / forwards);
    std::vector<std::pair<double, std::string>> rows;
    for (auto& [k, v] : ptime_) rows.push_back({v, k});
    std::sort(rows.rbegin(), rows.rend());
    for (auto& [v, k] : rows)
      std::fprintf(stderr, "  %-12s %8.3f ms  %5.1f%%\n", k.c_str(), v / forwards, 100 * v / total);
  }

  // CRUCIBLE_PLE_HOST: starts the disk reads for prompt positions [begin, end)
  // of ids, which later chunks will gather. Only a hint; no result depends on it.
  void HintPle(std::span<const int> ids, size_t begin, size_t end) {
    end = std::min(end, ids.size());
    if (!m_.ple.host || begin >= end) return;
    std::vector<int64_t> rows((end - begin) * 16);
    PleRowsOf(ids.data(), (int)begin, (int)(end - begin), rows.data());
    m_.ple.host->Hint(std::move(rows));
  }

 private:
  void EnsureMtpCache(SequenceState& state, int position) {
    if (state.qc.contains(kLayers)) return;
    if (position != 0)
      throw std::runtime_error("MTP cache is absent; restart prompt warmup before a nonzero MTP position");
    const auto bytes = QsaCacheBytes(state.mtp_capacity, state.compact_state);
    const size_t needed = bytes[0] + bytes[1] + bytes[2] + bytes[3];
    size_t free, total;
    HIP_OK(hipMemGetInfo(&free, &total));
    // Preserve the post-warm reserve for the OS, later plans and the frontend.
    constexpr size_t reserve = 9ull << 30;
    if (!CacheAllocationFits(free, needed, reserve))
      throw std::runtime_error("not enough device memory for the MTP cache; reduce the context, or use --spec 0 in gen");
    // Reserve host registry space before device ownership changes.
    buffers_.reserve(buffers_.size() + 4);
    state.AllocateMtpCache();
    const auto& cache = state.qc.at(kLayers);
    for (size_t i = 0; i < bytes.size(); ++i) {
      void* pointer = i == 0 ? static_cast<void*>(cache.k) : i == 1 ? static_cast<void*>(cache.v)
                      : i == 2 ? static_cast<void*>(cache.ik_raw) : static_cast<void*>(cache.ik_pool);
      buffers_.push_back({pointer, bytes[i]});
    }
  }

  void AddSequence(int capacity = 0) {
    auto state = std::make_unique<SequenceState>(m_, capacity ? capacity : ctx_, compact_state_, mtp_rows_);
    const auto spans = state->Spans();
    buffers_.insert(buffers_.end(), spans.begin(), spans.end());
    sequences_.push_back(std::move(state));
  }

  template <class T> T* Zeros(size_t n) {
    T* pointer = crucible::Zeros<T>(n);
    buffers_.push_back({pointer, n * sizeof(T)});
    return pointer;
  }

  template <class T> T* Scratch(size_t n) {
    T* pointer = crucible::Scratch<T>(n);
    buffers_.push_back({pointer, n * sizeof(T)});
    return pointer;
  }

  // The attention-stage row buffers (projections, GDN and QSA inputs, PLE, split-K partials) and the prefill MoE
  // scratch share one allocation: a layer writes its MoE scratch only after its attention outputs are consumed,
  // and the scratch is dead before the next layer or an interleaved decode pass starts. About 281 KB a prefill row
  // (1.15 GB at 4096 rows). CRUCIBLE_PF_STAGE_SHARE=0 keeps two separate regions.
  void AllocateStageArena(int capacity) {
    const size_t launch = (size_t)pf_room_rows_, rows = launch * kSlots;
    auto attention = [&](StageView& v) {
      qkv_ = v.Take<float>((size_t)capacity * kGdnQkvWidth);
      z_ = v.Take<float>((size_t)capacity * kGdnValueWidth);
      q_ = v.Take<float>((size_t)capacity * kQsaQWidth);
      k_ = v.Take<float>((size_t)capacity * kQsaKvWidth);
      v_ = v.Take<float>((size_t)capacity * kQsaKvWidth);
      idx_ = v.Take<float>((size_t)capacity * kQsaIdxWidth);
      eple_ = v.Take<float>((size_t)capacity * kHidden);
      kv_ = v.Take<float>((size_t)capacity * 12800);
      partial_capacity_ = std::max(kPartial, (size_t)capacity * 16384);
      partial_ = v.Take<float>(partial_capacity_);
      if (prefill_rows_) {
        pf_xpack_ = v.Take<__half>(launch * kMixerOutWidth);
        pf_plenv_ = v.Take<__half>(launch * kWide);
      }
    };
    auto moe = [&](StageView& v) {
      if (!prefill_rows_) return;
      auto& a = pf_moe_;
      a.pg = v.Take<float>(rows * kMoeInter); a.pu = v.Take<float>(rows * kMoeInter);
      a.pd = v.Take<float>(rows * kHidden);
      a.xg = v.Take<__half>(rows * kHidden); a.xu = v.Take<__half>(rows * kHidden);
      a.xd = v.Take<__half>(rows * kMoeInter);
    };
    const char* share = crucible::SwitchText("CRUCIBLE_PF_STAGE_SHARE");
    const bool shared = !(share && std::string(share) == "0");
    const StageLayout layout = StageArenaLayout(StageBytes(attention), StageBytes(moe), shared);
    char* arena = Scratch<char>(layout.bytes);
    StageView first{arena, layout.first};
    attention(first);
    StageView second{arena + layout.second_offset, layout.second};
    moe(second);
    std::fprintf(stderr, "stage arena: %.1f MiB (%s; attention %.1f MiB, MoE %.1f MiB)\n", layout.bytes / 1048576.0,
                 shared ? "shared" : "separate", layout.first / 1048576.0, layout.second / 1048576.0);
  }

  template <class T> T* Upload(const std::vector<T>& values) {
    T* pointer = crucible::Upload(values);
    if (pointer) buffers_.push_back({pointer, values.size() * sizeof(T)});
    return pointer;
  }

  // Warm only at startup, before draft allocations can be replaced or freed.
  void WarmUp() {
    warming_ = true;
    const int token = kEos;
    Forward(&token, 1, 0, 1);
    // The warmup plan pre-build is disabled until a correct fix exists: a
    // 4096-row plan built here faults on the first 4096-token prompt (GPU
    // page fault; train 3 gate bulk4k305). The first request builds it.
    Reset();
    std::vector<DeviceSpan> spans = m_.buffers;
    spans.insert(spans.end(), buffers_.begin(), buffers_.end());
    WarmDeviceBuffers(spans);
    warming_ = false;
  }

  static constexpr size_t kPartial = size_t(16) << 20;  // floats
  static constexpr size_t kXh = size_t(4) << 20;        // halves

  struct LayerGroups {
    Group a_in, a_out, gu, dn;
    int kr = 3;  // routed expert K; the shared expert has kr + 2
    // The shared expert's width: kr + 2 on the K3..6 packs, 7 or 8 on the h6_ng8 packs.
    int ks = 5;
    // R = 1: the gate/up items again in 8-block items (MoeStageKernel).
    GemvItem* d_gu8 = nullptr;
    int n_gu8 = 0;
    bool moe_chunk_fits = false;  // every expert split span is whole kMoeChunk stages
    int state_cache = 0;
  };
  struct PrefillLayer { PrefillLinear input, output; };
  // The bulk MTP warm-up of one row count: fc_embedding and fc_hidden groups, and the MTP layer's input projections.
  struct MtpPrefillPlan {
    std::vector<PrefillLinear> fuse;
    PrefillLayer layer;
  };
  struct PrefillPlans {
    PrefillLayer layers[kLayers];
    size_t DeviceBytes() const {
      size_t bytes = 0;
      for (const auto& layer : layers) bytes += layer.input.DeviceBytes() + layer.output.DeviceBytes();
      return bytes;
    }
  };
  struct Plans {
    LayerGroups L[kLayers + 1];  // index kLayers: the MTP block
    Group head, mtp_in;
    RowPolicy policy;
  };

#include "runtime/batch_forward.h"
#include "runtime/batch_spec_forward.h"

  void RebuildDraftHead() {
    for (void* p : {(void*)draft_group_.d_jobs, (void*)draft_group_.d_items,
                    (void*)draft_group_.d_in, (void*)draft_group_.d_out})
      if (p) HIP_OK(hipFree(p));
    for (auto& group : draft_groups_) {
      if (!group) continue;
      for (void* p : {(void*)group->d_jobs, (void*)group->d_items, (void*)group->d_in, (void*)group->d_out})
        if (p) HIP_OK(hipFree(p));
      group.reset();
    }
    draft_head_ = block_head_->Front();
    draft_blocks_ = block_head_->Mapping();
    Build(draft_group_, {Job(draft_head_, x_, kHidden, logits_, draft_head_.O,
                              kEpiStore, nullptr, 1)}, 1, true, 0, {draft_head_.front_s});
  }

  // front_s: one entry per job for dense decode groups that may use the front
  // tile order; entries > 0 mean the loader already stored that order.
  // item_blocks: column blocks per item (MoeFusedKernel<..., C> needs 8 * C).
  void Build(Group& g, std::vector<LinJob> jobs, int R, bool fixedS, size_t poff0 = 0,
             const std::vector<int>& front_s = {}, int item_blocks = 8, int shared_job = -1) {
    std::vector<std::pair<int, int>> io;
    for (const LinJob& j : jobs) io.push_back({j.I, j.O});
    const std::vector<int> splits = fixedS ? std::vector<int>(jobs.size(), 1) : DecodeSplits(io);
    for (size_t ji = 0; ji < jobs.size(); ++ji) jobs[ji].S = splits[ji];
    const RowPolicy policy = ResolveRowPolicy(R, m_.head.K, m_.L[0].moe.experts[0].K[0]);
    g.fuse = fixedS ? 0 : policy.fuse;
    g.cache = policy.cache;
    size_t poff = poff0, xoff = 0;
    int maxspan = 0;
    std::vector<GemvItem> items;
    std::vector<XfItem> in, out;
    for (size_t ji = 0; ji < jobs.size(); ++ji) {
      LinJob& j = jobs[ji];
      j.cache_policy = policy.cache;
      if (!fixedS) j.done = Zeros<unsigned>(j.O / 128);
      if (!fixedS && (g.fuse & 1) && (j.I / 16 / j.S) % 8) {
        std::fprintf(stderr, "fused input requires aligned Hadamard blocks\n");
        std::exit(2);
      }
      j.partial = partial_ + poff;
      poff += (size_t)j.S * j.R * j.O;
      j.xh = xh_ + xoff;
      xoff += (size_t)j.R * j.I;
      maxspan = std::max(maxspan, j.I / 16 / j.S);
      for (int s = 0; s < j.S; ++s)
        for (int b0 = 0; b0 < j.O / 16; b0 += item_blocks)
          items.push_back({(int)ji, s, b0, std::min(item_blocks, j.O / 16 - b0)});
      for (int r = 0; r < j.R; ++r) {
        for (int b = 0; b < j.I / 128; ++b) in.push_back({(int)ji, r, b});
        for (int b = 0; b < j.O / 128; ++b) out.push_back({(int)ji, r, b});
      }
    }
    if (poff > partial_capacity_ || xoff > xh_capacity_) {
      std::fprintf(stderr, "scratch overflow\n");
      std::exit(1);
    }
    g.jobs = jobs;
    g.has_k6 = std::any_of(jobs.begin(), jobs.end(), [](const LinJob& j) { return j.K == 6; });
    g.wide = std::any_of(jobs.begin(), jobs.end(), [](const LinJob& j) { return WideDecodeK(j.K); });
    g.d_jobs = Upload(jobs);
    const bool loaded_front = !front_s.empty() && front_s[0] > 0;
    for (size_t i = 0; loaded_front && i < jobs.size(); ++i)
      if (front_s[i] != jobs[i].S) {
        std::fprintf(stderr, "front tile order split %d differs from decode split %d\n", front_s[i], jobs[i].S);
        std::exit(1);
      }
    if (loaded_front) g.d_front = g.d_jobs;
    g.d_items = shared_job<0?Upload(items):Upload(MoeSharedFirstItems(items,shared_job));
    g.d_in = Upload(in);
    g.d_out = Upload(out);
    g.nitems = (int)items.size();
    g.nin = (int)in.size();
    g.nout = (int)out.size();
    g.R = R;
    g.lds = (size_t)R * maxspan * 32;
    g.aligned = std::all_of(jobs.begin(), jobs.end(), [](const LinJob& j) { return (j.I / 16 / j.S) % kPrefetch == 0; });
    g.aligned_chunk = std::all_of(jobs.begin(), jobs.end(),
                                  [](const LinJob& j) { return (j.I / 16 / j.S) % lin_ring::kChunk == 0; });
  }

  static constexpr int StaticRows(int R) { return R > kMaxRows ? kPackedRows : R; }

  // The tile order GemvR reads for this group.
  bool ReadsFront(const Group& g) const { return g.d_front != nullptr; }
  // The ring kernels hold K <= 6 only, so a wide group keeps verify passes at kMaxRows rows.
  bool PackedFits(const Group& g) const {
    return !g.wide && PackedGroupFits({g.jobs[0].dst != nullptr, ReadsFront(g), g.aligned_chunk});
  }

  static LinJob Job(const Lin& L, const float* src, int lds, float* dst, int ldd, int epi,
                    const float* aux, int R) {
    LinJob j{};
    j.tiles = L.tiles;
    j.suh = L.suh;
    j.svh = L.svh;
    j.src = src;
    j.dst = dst;
    j.aux = aux;
    j.I = L.I;
    j.O = L.O;
    j.K = L.K;
    j.S = 1;
    j.R = R;
    j.lds = lds;
    j.ldd = ldd;
    j.epi = epi;
    return j;
  }

  static bool EnvOn(crucible::SwitchName name) {
    const char* v = crucible::SwitchText(name);
    return v && std::string(v) == "1";
  }

  // Opt-in (CRUCIBLE_ABLATE=FILE, ablation.h): fail closed on any file that
  // is not a reviewed sidecar for this checkpoint's geometry.
  void LoadAblation(int rows) {
    const char* path = crucible::SwitchText("CRUCIBLE_ABLATE");
    if (!path || !*path) return;
    AblationSet set;
    try {
      set = LoadAblationFile(path);
      if (!EnvOn("CRUCIBLE_ABLATE_ANY_CHECKPOINT"))
        CheckAblationCheckpoint(*set.pin, m_.L[0].moe.experts[0].K[0], m_.head.K);
    } catch (const std::invalid_argument& error) {
      std::fprintf(stderr, "CRUCIBLE_ABLATE error: %s; unset CRUCIBLE_ABLATE for the stock model\n", error.what());
      std::exit(2);
    }
    if (EnvOn("CRUCIBLE_ABLATE_ANY_CHECKPOINT"))
      std::fprintf(stderr, "CRUCIBLE_ABLATE_ANY_CHECKPOINT=1: checkpoint geometry check skipped (measurement only)\n");
    for (const AblationSite& site : set.sites) {
      std::vector<float> ab(site.a);
      ab.insert(ab.end(), site.b.begin(), site.b.end());
      (site.mlp ? abl_mlp_ : abl_attn_)[site.layer] = Upload(ab);
    }
    abl_rows_ = rows;
    abl_y_ = Scratch<float>((size_t)rows * kHidden);
    std::fprintf(stderr, "ablation: %zu sublayers from %s (sha256 %s, %s)\n", set.sites.size(), path,
                 set.sha256.c_str(), set.pin->source);
  }

  const float* AblMlp(int li) const { return li < kLayers ? abl_mlp_[li] : nullptr; }

  // y is in abl_y_; post_ holds the inject gates of the current HC read.
  void AblateWrite(const float* ab, float* H, int ldh, int R) {
    if (R > abl_rows_) {
      std::fprintf(stderr, "ablation scratch holds %d rows, not %d\n", abl_rows_, R);
      std::abort();
    }
    AblateWriteKernel<<<R, kAblateThreads>>>(abl_y_, ab, post_, H, ldh);
  }

  // Opt-in (CRUCIBLE_PERSISTENT_GDN=1): one persistent kernel runs the R=1
  // GDN attention side (HC read, projections, recurrence, residual).
  bool PersistentGdnOn() const { return use_pgdn_; }

  void PersistentGdnSide(int li, const Layer& L, const LayerGroups& G, int pos0) {
    if (!pgdn_bar_) {
      pgdn_bar_ = Zeros<uint32_t>(2);
      pgdn_error_ = Zeros<uint32_t>(1);
    }
    const size_t lds = std::max(G.a_in.lds, G.a_out.lds);
    auto& b = pgdn_items_[li];
    if (!b[0].items) {
      const int grid = PersistentGdnBlocks(lds);
      b[0] = BuildBalancedItems(G.a_in.jobs, grid, true);
      b[1] = BuildBalancedItems(G.a_out.jobs, grid, true);
    }
    // The grid barriers need every block resident; lookahead blocks take CU slots.
    if (PlansFor(1).policy.blocks != 0) {
      std::fprintf(stderr, "CRUCIBLE_PERSISTENT_GDN=1 needs CRUCIBLE_LOOKAHEAD=0\n");
      std::exit(2);
    }
    // Its HC read has the VALU arithmetic; R > 1 would use the WMMA one.
    if (hc_wmma_) {
      std::fprintf(stderr, "CRUCIBLE_PERSISTENT_GDN=1 needs CRUCIBLE_HC_WMMA=0\n");
      std::exit(2);
    }
    SequenceState& s = *sequence_;
    GdnArgs g{qkv_, z_, x_, L.gdn.ab, L.gdn.conv, L.gdn.A_log, L.gdn.dt_bias, L.gdn.gamma,
              s.ghist.at(li), s.gstate.at(li), L.gdn.out.suh, G.a_out.jobs[0].xh, s.gpend.at(li),
              pos0, 1, 0, 1};
    const bool front = G.a_in.d_front && G.a_out.d_front;
    LaunchPersistentGdn({L.hca, front ? PersistentFrontGroup(G.a_in, &b[0]) : PersistentGroup(G.a_in, &b[0]),
                         front ? PersistentFrontGroup(G.a_out, &b[1]) : PersistentGroup(G.a_out, &b[1]), g,
                         H_, dots_, ss_, x_, post_, pgdn_bar_, pgdn_error_, nullptr},
                        lds, b[0].blocks, front);
  }

  void Block(int li, const Layer& L, const LayerGroups& G, int R, int pos0, int replay,
             int commit) {
    if (li < kLayers && !L.attn && R == 1 && replay == 0 && commit == 1 && batch_streams_.empty() && !VerificationSpansActive() &&
        PersistentGdnOn() && !abl_attn_[li]) {
      PersistentGdnSide(li, L, G, pos0);
      Mark("gdn_side");
      BlockMoe(li, L, G, R);
      return;
    }
    HcRead(L.hca, R, true);
    Mark("hc_mix");
    RunGroup(G.a_in, true, true);
    Mark(L.attn ? "qsa_proj" : "gdn_proj");
    if (VerificationSpansActive()) VerificationAttentionCore(li,L,G);
    else
    if (batch_streams_.empty()) {
      AttentionCore(li, L, G, R, pos0, replay, commit, *sequence_, 0);
    } else {
      BatchAttentionCore(li, L, G, R);
    }
    Mark(L.attn ? "qsa_core" : "gdn_core");
    RunGroup(G.a_out, false, true);
    if (li < kLayers && abl_attn_[li]) AblateWrite(abl_attn_[li], H_, kWide, R);
    Mark(L.attn ? "qsa_oproj" : "gdn_oproj");
    BlockMoe(li, L, G, R);
  }

  void BlockMoe(int li, const Layer& L, const LayerGroups& G, int R) {
    HcRead(L.hcm, R, true);
    Mark("hc_mix");
    // The stage has the K3 and K4 instances only (MoeFusedKernelWide keeps the h6_ng8 packs).
    const bool stage = MoeStageOn(li, R) && G.gu.cache == 4 && NarrowMoeWidths({G.kr, G.ks});
    LaunchDense(L.moe.router, kExperts + 1, kHidden, x_, kHidden, rlog_, kExperts + 1, R);
    if (stage) {
      MoeStage(L, G);
      Mark("moe_gateup");
      return;
    }
    MoeRouteKernel<<<R, 512>>>(rlog_, sel_moe_, wts_);
    if (route_dump_ && li < kLayers) DumpRoutes(li, R);
    if (ecount_) CountExpertsKernel<<<1, 32>>>(sel_moe_, R, ecount_ + (li == kLayers ? 2 * (kPackedRows + 1) : 0));
    MoeRows(li, L, G, R);
  }

  // Routed experts with compact rows (moe_rows.h): each distinct expert runs
  // only the rows routed to it.
  void MoeRows(int li, const Layer& L, const LayerGroups& G, int R) {
    MoeRowsUniqueKernel<<<1, 512>>>(sel_moe_, R, L.moe.d_experts, G.gu.d_jobs, G.dn.d_jobs, map_, mapj_,
                                    rowl_, moe_done_);
    Mark("moe_route");
    // 55 R blocks instead of 3240 at R=8, most of them dead (3.05 forward -0.2 to -0.4 ms at R=3..8, 4.05 up to
    // -0.7 ms).
    MoeRowsInSlotsKernel<<<MoeInSlotsBlocks(R), 256>>>(G.gu.d_jobs, map_, mapj_, R);
    // Gate/up, mid and down in one launch (moe_rows.h); the profile reports
    // the three as moe_gateup.
    // Expert jobs hold at most kMaxRows compact rows (moe_chunk_map.h).
    MoeFused(G, std::min(R, kMaxRows));
    Mark("moe_gateup");
    if (const float* ab = AblMlp(li)) {
      MoeRowsCombineKernel<true><<<(R * 20 + 7) / 8, 256>>>(G.dn.d_jobs, map_, mapj_, wts_, nullptr, abl_y_, R);
      AblateWrite(ab, H_, kWide, R);
    } else {
      MoeRowsCombineKernel<<<(R * 20 + 7) / 8, 256>>>(G.dn.d_jobs, map_, mapj_, wts_, post_, H_, R);
    }
    Mark("moe_combine");
  }

  // The route, map, input transform and combine inside the expert launch (moe_stage.h). The route dump, the
  // expert counts and the ablation read or replace the separate kernels' outputs, so they keep them.
  bool MoeStageOn(int li, int R) const {
#if CRUCIBLE_DEBUG_NONFINITE & 1
    return false;  // the probes sit in MoeRouteKernel
#endif
    return R == 1 && !route_dump_ && !ecount_ && !AblMlp(li);
  }

  // 8-block gate/up items (G.d_gu8). Only the cache policy of the tuned row policy (4) has instances; another
  // policy keeps the separate kernels.
  void MoeStage(const Layer& L, const LayerGroups& G) {
    const MoeStageArgs a{G.gu.d_jobs, G.d_gu8, G.n_gu8, G.dn.d_jobs, G.dn.d_items, G.dn.nitems,
                         rlog_, L.moe.d_experts, post_, H_, moe_sync_};
    const auto kernel = G.kr == 4 ? MoeStageKernel<1, 4, kMoePf, 4> : MoeStageKernel<1, 3, kMoePf, 4>;
    const int grid = G.n_gu8 + G.dn.nitems;
    hipLaunchKernelGGL(kernel, dim3(grid), dim3(256), MoeFusedLds(1, 0, G.gu.lds, G.dn.lds), 0, a);
  }

  template <int R>
  void MoeFusedR(const LayerGroups& G) {
    if (NarrowMoeWidths({G.kr, G.ks})) {
      if (G.kr == 4) MoeFusedLaunch<R, 4>(G);
      else MoeFusedLaunch<R, 3>(G);
    } else {
      MoeFusedWideR<R>(G);
    }
  }

  // One MoeFusedKernelWide launch for each pair of kWideMoeWidths (moe_widths_test checks the list).
  template <int R, size_t N = 0>
  void MoeFusedWideR(const LayerGroups& G) {
    if constexpr (N < kWideMoeWidths.size()) {
      constexpr MoeWidths w = kWideMoeWidths[N];
      if (G.kr == w.kr && G.ks == w.ks) MoeFusedWideLaunch<R, w.kr, w.ks>(G);
      else MoeFusedWideR<R, N + 1>(G);
    } else {
      // ValidateLayer admits no other set of widths.
      std::fprintf(stderr, "internal error: MoE widths routed %d shared %d\n", G.kr, G.ks);
      std::abort();
    }
  }

  // MoeFusedLaunch for the h6_ng8 packs (MoeFusedKernelWide), with the same chunk rule; K2, K2.5 and
  // K3.5 routed experts (codes 25, 35) stage in chunks as K3 does (not measured against full staging).
  template <int R, int KR, int KS>
  void MoeFusedWideLaunch(const LayerGroups& G) {
    bool chunk = false;
    auto kernel = G.gu.cache == 4 ? MoeFusedKernelWide<R, KR, KS, kMoePf, 4> : MoeFusedKernelWide<R, KR, KS>;
    if constexpr (R > 1 && (KR == 2 || KR == 25 || KR == 3 || KR == 35)) {
      chunk = G.moe_chunk_fits;
      if (chunk)
        kernel = G.gu.cache == 4 ? MoeFusedKernelWide<R, KR, KS, kMoePf, 4, kMoeCols, kMoeChunk>
                                 : MoeFusedKernelWide<R, KR, KS, kMoePf, 0, kMoeCols, kMoeChunk>;
    }
    const size_t lds = MoeFusedLds(R, chunk ? kMoeChunk : 0, G.gu.lds, G.dn.lds);
    hipLaunchKernelGGL(kernel, dim3(G.gu.nitems + G.dn.nitems), dim3(256), lds, 0,
        G.gu.d_jobs, G.gu.d_items, G.gu.nitems, G.dn.d_jobs, G.dn.d_items, G.dn.nitems, moe_done_);
  }

  template <int R, int KR>
  void MoeFusedLaunch(const LayerGroups& G) {
    // Chunked x staging pays from R=2 on with K3 experts (3.05); at R=1 the
    // full span is 5 KB. With K4 experts (4.05) it lost 0.2-0.5 ms at R=3..5
    // and was even at R=6..8 (v8-405.txt), so only K3 builds it. It stages gate/up activations in 40-tile
    // chunks (half the LDS of a full span at R=8): 3.05 forward -0.15 ms at R=2..5, -0.4 to -0.9 ms at R=6..8.
    bool chunk = false;
    auto kernel = G.gu.cache == 4 ? MoeFusedKernel<R, KR, kMoePf, 4> : MoeFusedKernel<R, KR>;
    if constexpr (R > 1 && KR == 3) {
      chunk = G.moe_chunk_fits;
      if (chunk)
        kernel = G.gu.cache == 4 ? MoeFusedKernel<R, KR, kMoePf, 4, kMoeCols, kMoeChunk>
                                 : MoeFusedKernel<R, KR, kMoePf, 0, kMoeCols, kMoeChunk>;
    }
    const size_t lds = MoeFusedLds(R, chunk ? kMoeChunk : 0, G.gu.lds, G.dn.lds);
    hipLaunchKernelGGL(kernel, dim3(G.gu.nitems + G.dn.nitems), dim3(256), lds, 0,
        G.gu.d_jobs, G.gu.d_items, G.gu.nitems, G.dn.d_jobs, G.dn.d_items, G.dn.nitems, moe_done_);
  }

  void MoeFused(const LayerGroups& G, int R) {
    switch (R) {
      case 1: MoeFusedR<1>(G); break;
      case 2: MoeFusedR<2>(G); break;
      case 3: MoeFusedR<3>(G); break;
      case 4: MoeFusedR<4>(G); break;
      case 5: MoeFusedR<5>(G); break;
      case 6: MoeFusedR<6>(G); break;
      case 7: MoeFusedR<7>(G); break;
      default: MoeFusedR<8>(G); break;
    }
  }

  // draft: the MTP head, whose rows only propose (sampling::ReplaceNonfiniteDraftIds); else a target head.
  std::vector<int> Head(const Hc& mixer, const Plans& P, int R, bool draft = false) {
    HcRead(mixer, R, false);
    Mark("hc_mix");
    RunGroup(P.head, true, true);
    ArgmaxKernel<<<R, 1024>>>(logits_, kVocab, ids_);
    Mark("head");
    lookahead_.Stop();
    Collect();
    HIP_OK(hipMemcpyAsync(host_ids_.data(), ids_, sizeof(int) * R, hipMemcpyDeviceToHost, 0));
    // This also completes input transfers before the next Forward or Mtp reuse.
    HIP_OK(hipStreamSynchronize(0));
#if CRUCIBLE_DEBUG_NONFINITE & 1
    RequireNoNonfiniteSites();
#endif
    std::vector<int> ids(host_ids_.data(), host_ids_.data() + R);
    if (draft) sampling::ReplaceNonfiniteDraftIds(ids);
    else sampling::RequireFiniteHeadIds(ids);
    return ids;
  }

  // li == kLayers is the MTP block, which the ablation never changes.
  void BuildLayer(LayerGroups& G, const Layer& L, int R, int li) {
    // An ablated output projection stores y; AblateWrite does the write-back.
    const bool abl = li < kLayers && abl_attn_[li];
    if (abl && StaticRows(R) > abl_rows_) {
      std::fprintf(stderr, "ablation scratch holds %d rows, not %d\n", abl_rows_, StaticRows(R));
      std::exit(2);
    }
    float* const out_dst = abl ? abl_y_ : H_;
    const int out_ld = abl ? kHidden : 0, out_epi = abl ? kEpiStore : kEpiResidual;
    float* const out_aux = abl ? nullptr : post_;
    G.state_cache = ResolveRowPolicy(R, m_.head.K, m_.L[0].moe.experts[0].K[0]).state;
    // Static projections of a packed pass (R > kMaxRows) run kPackedRows rows;
    // each row's values do not depend on the row count, and rows >= R are
    // scratch that no later step reads.
    const int N = StaticRows(R);
    if (IsQsa(li)) {
      Build(G.a_in,
            {Job(L.qsa.q, x_, kHidden, q_, kQsaQWidth, kEpiStore, nullptr, N),
             Job(L.qsa.k, x_, kHidden, k_, kQsaKvWidth, kEpiStore, nullptr, N),
             Job(L.qsa.v, x_, kHidden, v_, kQsaKvWidth, kEpiStore, nullptr, N),
             Job(L.qsa.idx, x_, kHidden, idx_, kQsaIdxWidth, kEpiStore, nullptr, N)},
            N, false, 0, {L.qsa.q.front_s, L.qsa.k.front_s, L.qsa.v.front_s, L.qsa.idx.front_s});
      Build(G.a_out, {Job(L.qsa.o, nullptr, 0, out_dst, out_ld, out_epi, out_aux, N)}, N, false, 0,
            {L.qsa.o.front_s});
    } else {
      Build(G.a_in,
            {Job(L.gdn.qkv, x_, kHidden, qkv_, kGdnQkvWidth, kEpiStore, nullptr, N),
             Job(L.gdn.z, x_, kHidden, z_, kGdnValueWidth, kEpiStore, nullptr, N)},
            N, false, 0, {L.gdn.qkv.front_s, L.gdn.z.front_s});
      Build(G.a_out, {Job(L.gdn.out, nullptr, 0, out_dst, out_ld, out_epi, out_aux, N)}, N, false, 0,
            {L.gdn.out.front_s});
    }
    // Unique-expert jobs [0, R*kTopK), then the shared expert in chunks of
    // kMaxRows rows (moe_chunk_map.h). Routed slots start on expert 0;
    // MoeRowsUniqueKernel retargets them per forward.
    const ExpertPtrs& e0 = L.moe.experts[0];
    Lin rg{kHidden, kMoeInter, e0.K[0], (uint32_t*)e0.tiles[0], (__half*)e0.suh[0], (__half*)e0.svh[0]};
    Lin ru{kHidden, kMoeInter, e0.K[1], (uint32_t*)e0.tiles[1], (__half*)e0.suh[1], (__half*)e0.svh[1]};
    Lin rd{kMoeInter, kHidden, e0.K[2], (uint32_t*)e0.tiles[2], (__half*)e0.suh[2], (__half*)e0.svh[2]};
    std::vector<LinJob> gu, dn;
    const int U = R * kTopK, expert_rows = std::min(R, kMaxRows);
    for (int u = 0; u < MoeExpertJobs(R); ++u) {
      const bool sh = u >= U;
      // Every job slot plans expert_rows rows. The last shared chunk of a
      // packed pass computes scratch rows past R (each row is independent),
      // so MoeFusedKernel<8> needs no shorter shared body (half its code).
      const int nr = expert_rows;
      gu.push_back(Job(sh ? L.moe.sg : rg, x_, kHidden, nullptr, 0, kEpiStore, nullptr, nr));
      gu.push_back(Job(sh ? L.moe.su : ru, x_, kHidden, nullptr, 0, kEpiStore, nullptr, nr));
      dn.push_back(Job(sh ? L.moe.sd : rd, nullptr, 0, nullptr, 0, kEpiStore, nullptr, nr));
    }
    G.kr = e0.K[0];
    G.ks = L.moe.sg.K;
    // MoeFusedKernel<..., kMoeCols> items span kMoeItemBlocks columns.
    const int ib = kMoeItemBlocks<kMoeCols>;
    const bool shared_first=R>1;
    Build(G.gu, gu, expert_rows, true, 0, {}, ib, shared_first?2*R*kTopK:-1);
    if (G.gu.nitems != (int)gu.size() / 2 * kMoeGuItems<kMoeCols>) {
      std::fprintf(stderr, "MoE gate/up items do not match the down-item wait count\n");
      std::exit(1);
    }
    if (R == 1) {
      std::vector<GemvItem> items;
      for (size_t ji = 0; ji < G.gu.jobs.size(); ++ji)
        for (int b0 = 0; b0 < G.gu.jobs[ji].O / 16; b0 += kMoeItemBlocks<1>)
          items.push_back({(int)ji, 0, b0, std::min(kMoeItemBlocks<1>, G.gu.jobs[ji].O / 16 - b0)});
      if (items.size() != gu.size() / 2 * kMoeGuItems<1>) {
        std::fprintf(stderr, "MoE 8-block gate/up items do not match the down-item wait count\n");
        std::exit(1);
      }
      G.d_gu8 = Upload(items);
      G.n_gu8 = (int)items.size();
    }
    // MoeFusedKernel reads gate/up partials while down items already write
    // theirs, so the two groups need disjoint partial scratch.
    size_t gu_partial = 0;
    for (const LinJob& j : G.gu.jobs) gu_partial += (size_t)j.S * j.R * j.O;
    Build(G.dn, dn, expert_rows, true, gu_partial, {}, ib, shared_first?R*kTopK:-1);
    // The chunked bodies (MoeGemvCoreChunked; the bench MoeGemvChunked in
    // bench/kernels/moe_rows.h) stage whole kMoeChunk-tile chunks; another span runs
    // the full-span stage (bitwise equal).
    const auto whole = [](const LinJob& j) { return (j.I / 16 / j.S) % kMoeChunk == 0; };
    G.moe_chunk_fits = std::all_of(G.gu.jobs.begin(), G.gu.jobs.end(), whole) &&
                       std::all_of(G.dn.jobs.begin(), G.dn.jobs.end(), whole);
  }

  const Plans& PlansFor(int R) {
    auto it = plans_.find(R);
    if (it != plans_.end()) return *it->second;
    auto P = std::make_unique<Plans>();
    P->policy = ResolveRowPolicy(R, m_.head.K, m_.L[0].moe.experts[0].K[0]);
    if (R < 1 || R > kPackedRows) {
      std::fprintf(stderr, "invalid forward row count %d\n", R);
      std::exit(2);
    }
    for (int l = 0; l < kLayers; ++l) BuildLayer(P->L[l], m_.L[l], R, l);
    if (R <= kMaxRows) BuildLayer(P->L[kLayers], m_.mtp.L, R, kLayers);
    const int N = StaticRows(R);
    Build(P->head, {Job(m_.head, x_, kHidden, logits_, kVocab, kEpiStore, nullptr, N)}, N, false, 0,
          {m_.head.front_s});
    // The MTP block runs at most kMaxRows rows.
    if (R > kMaxRows) {
      plans_[R] = std::move(P);
      return *plans_[R];
    }
    std::vector<LinJob> mi{Job(m_.mtp.fe, mE_, kHidden, mep_, kHidden, kEpiStore, nullptr, R)};
    for (int st = 0; st < kStreams; ++st)
      mi.push_back(Job(m_.mtp.fh, mN_ + st * kHidden, kWide, mhp_ + st * kHidden, kWide, kEpiStore,
                       nullptr, R));
    Build(P->mtp_in, mi, R, false);
    plans_[R] = std::move(P);
    return *plans_[R];
  }

  template <int R>
  void GemvR(const Group& g, bool in, bool out) {
    // A loaded front order (d_front == d_jobs) has no engine-order fallback.
    const bool front = ReadsFront(g);
    const LinJob* jobs = front ? g.d_front : g.d_jobs;
    if constexpr (R == 1) {
      // Expert jobs have no destination. Their fused schedule remains unchanged.
      const bool ring = ring_mode_ == kRingOn || (ring_mode_ == kRingFront && front);
      if (ring && g.jobs[0].dst && !g.wide) {
        // LinRingFlowKernel: no barrier between items and fused transforms.
        const int grid = std::min(kRingBlocks, g.nitems), words = (int)(g.lds / 4);
        if (front) LinRingFlowKernel<true><<<grid, 320, 2 * g.lds>>>(jobs, g.d_items, g.nitems, words, in, out);
        else LinRingFlowKernel<false><<<grid, 320, 2 * g.lds>>>(jobs, g.d_items, g.nitems, words, in, out);
        return;
      }
    }
    if constexpr (R > 1) {
      // R > 1 front-order dense groups through the ring (ring_rows.h): 3.05 dense groups 9.7-10.5 ms against
      // 10.7-12.5 ms for LinPolicyKernel at R=2..8, and each whole verify forward is faster with the lookahead
      // kept.
      if (front && g.jobs[0].dst && !g.wide) {
        if (in) LinInKernel<<<(g.nin + 7) / 8, 256>>>(g.d_jobs, g.d_in, g.nin);
        LinRingRowsKernel<R><<<std::min(kRingBlocks, g.nitems), 320>>>(jobs, g.d_items, g.nitems, out);
        return;
      }
    }
    if constexpr (R > kMaxRows) {
      // A full-span activation stage for 16 rows needs 80 KB of LDS, so a
      // packed pass reads an engine-order group through the ring too. Since
      // qsa-front both stock packs load every dense group in front order
      // (the front branch above), so this is the fallback for other packs.
      // MaxPassRows admits a packed pass only when every group fits.
      if (!PackedFits(g)) {
        std::fprintf(stderr, "internal error: a %d-row pass reached a group that MaxPassRows rejects\n", R);
        std::abort();
      }
      if (in) LinInKernel<<<(g.nin + 7) / 8, 256>>>(g.d_jobs, g.d_in, g.nin);
      LinRingRowsKernel<R, false><<<std::min(kRingBlocks, g.nitems), 320>>>(jobs, g.d_items, g.nitems, out);
    } else if (g.wide) {
      // K7 and K8 (the h6_ng8 packs) have their own instantiations, so K3..6 keep their code.
      if (g.fuse || g.cache) {
        if (front) LinPolicyKernel<R, kMaxWideDecodeK, true><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems, in, out);
        else LinPolicyKernel<R, kMaxWideDecodeK><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems, in, out);
      } else if (front) {
        LinGemvKernel<R, kMaxWideDecodeK, true><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems);
      } else {
        LinGemvKernel<R, kMaxWideDecodeK><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems);
      }
    } else {
    // Keep K6 register pressure out of both lower-width dispatch variants.
    if (g.fuse || g.cache) {
      if (front) {
        if (g.has_k6) LinPolicyKernel<R, 6, true><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems, in, out);
        else LinPolicyKernel<R, 5, true><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems, in, out);
      } else if (g.has_k6) {
        LinPolicyKernel<R><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems, in, out);
      } else {
        LinPolicyKernel<R, 5><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems, in, out);
      }
    } else if (front) {
      if (g.has_k6) LinGemvKernel<R, 6, true><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems);
      else LinGemvKernel<R, 5, true><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems);
    } else if (g.has_k6) {
      LinGemvKernel<R><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems);
    } else {
      LinGemvKernel<R, 5><<<g.nitems, 256, g.lds>>>(jobs, g.d_items, g.nitems);
    }
    }
  }

  void RunGroup(const Group& g, bool in, bool out) {
    const bool fi = in && (g.fuse & 1), fo = out && (g.fuse & 2);
    if (in && !fi) LinInKernel<<<(g.nin + 7) / 8, 256>>>(g.d_jobs, g.d_in, g.nin);
    switch (g.R) {
      case 1: GemvR<1>(g, fi, fo); break;
      case 2: GemvR<2>(g, fi, fo); break;
      case 3: GemvR<3>(g, fi, fo); break;
      case 4: GemvR<4>(g, fi, fo); break;
      case 5: GemvR<5>(g, fi, fo); break;
      case 6: GemvR<6>(g, fi, fo); break;
      case 7: GemvR<7>(g, fi, fo); break;
      case 8: GemvR<8>(g, fi, fo); break;
      default: GemvR<kPackedRows>(g, fi, fo); break;
    }
    if (out && !fo) LinOutKernel<<<(g.nout + 7) / 8, 256>>>(g.d_jobs, g.d_out, g.nout);
  }

  template <int R>
  void HcReadR(const Hc& hc, bool inject, int offset = 0, float* rows = nullptr) {
    // offset selects one row block of a prefill chunk (the final HC); rows, when set, holds that block's stacks
    // (a bulk chunk's rows stay in pf_H_).
    float* h = rows ? rows : H_ + (size_t)offset * kWide;
    float* dots = dots_ + (size_t)offset * hc.M * kStreams;
    float* ss = ss_ + offset * kStreams;
    float* x = x_ + (size_t)offset * kHidden;
    float* post = inject ? post_ + offset * kStreams : nullptr;
    if constexpr (R >= 4)
      HcRowsDotsShared<R, false, 2><<<dim3((hc.M + 15) / 16 + 1, kStreams), 256>>>(
          hc.fn, hc.M, h, dots, ss);
    else
      HcDotsKernel<R><<<((hc.M + 1) * kStreams + 7) / 8, 256>>>(hc.fn, hc.M, h, dots, ss);
    Mark("hc_dots");
    if constexpr (R >= 4)
      HcRowsMix<R, HcMixMode::Ordered><<<kHidden / 16, 256>>>(
          hc.up, hc.w1, dots, hc.M, ss, h, x, post);
    else
      HcMixKernel<R><<<kHidden / 16, 256>>>(hc.up, hc.w1, dots, hc.M, ss, h, x, post);
  }

  // Bulk consumers can use only the half copy because HC rounds x to fp16.
  // shape_rows: the row count that picks the bulk kernels (default R), so a row range of a chunk runs the
  // same kernels as the whole chunk.
  void HcRead(const Hc& hc, int R, bool inject, __half* xh = nullptr, bool half_only = false,
              const float* H = nullptr, int ldh = kWide, int shape_rows = 0) {
    H = H ? H : H_;
    if (half_only && !xh) std::abort();
    const int S = shape_rows ? shape_rows : R;
    // A packed verify pass keeps the decode arithmetic above 8 rows.
    if (R > 8 && prefill_rows_ && !VerificationSpansActive()) {
      if (hc.M>kHcWsTiles*16) std::abort();
      // CRUCIBLE_PF_HC_FUSED selects the previous dots kernel (A/B tests).
      static const bool fused=NumericsOn("CRUCIBLE_PF_HC_FUSED");
      if (fused) PrefillHcDotsFusedKernel<128><<<dim3((R+kHcRows-1)/kHcRows,kStreams),256>>>(hc.fn,hc.M,H,dots_,ss_,R,ldh);
      else if (pf_split_) PrefillHcDotsWsKernel<64,0,true,2,4,true><<<kStreams*((R+kHcRows-1)/kHcRows),(kHcWsCompute+2)*32>>>(
          hc.fn,hc.M,H,dots_,ss_,R,ldh);
      // 64-row blocks halve the fn reads (bench R4096 2323 -> 2023 us, same
      // dots); small chunks keep 32-row blocks for more blocks per CU.
      // Fragment-order fn: bench hc_frag R4096 1208 -> 1035 us, bitwise.
      else if (S>=512 && hc.fnf) PrefillHcDotsWsKernel<64,0,true,2,2,false,4,true><<<kStreams*((R+63)/64),(kHcWsCompute+2)*32>>>(
          hc.fnf,hc.M,H,dots_,ss_,R,ldh);
      else if (S>=512) PrefillHcDotsWsKernel<64,0,true,2,2,false,4><<<kStreams*((R+63)/64),(kHcWsCompute+2)*32>>>(
          hc.fn,hc.M,H,dots_,ss_,R,ldh);
      else PrefillHcDotsWsKernel<64,0,true,2,4,false><<<kStreams*((R+kHcRows-1)/kHcRows),(kHcWsCompute+2)*32>>>(
          hc.fn,hc.M,H,dots_,ss_,R,ldh);
      Mark("hc_dots");
      PrefillHcMidPackKernel<<<R,256>>>(dots_,hc.M,ss_,pf_thi_,pf_tlo_,inject ? post_ : nullptr,R);
      // Fragment-order up: bench hc_frag R4096 1993 -> 1796 us, bitwise.
      if (hc.upf) PrefillHcMixWideKernel<true,true><<<dim3(kHidden/16,(R+127)/128),256>>>(hc.upf,hc.w1,pf_thi_,pf_tlo_,ss_,H,half_only ? nullptr : x_,R,xh,ldh);
      else PrefillHcMixWideKernel<true><<<dim3(kHidden/16,(R+127)/128),256>>>(hc.up,hc.w1,pf_thi_,pf_tlo_,ss_,H,half_only ? nullptr : x_,R,xh,ldh);
      return;
    }
    if (H != H_ || ldh != kWide) std::abort();
    if (hc_wmma_) {
      // One arithmetic for every R: each row's x and post are bitwise equal
      // for any R and any row offset (Gates 2 and 3).
      for (int o = 0; o < R; o += kHcWRows)
        HcWLaunch(hc.fn, hc.M, hc.up, hc.w1, H_ + (size_t)o * kWide, std::min(kHcWRows, R - o),
                  x_ + (size_t)o * kHidden, inject ? post_ + o * kStreams : nullptr, hcw_, ++hcw_epoch_);
      return;
    }
    for (int o=0; o<R; o+=8) {
      switch (std::min(8,R-o)) {
        case 1: HcReadR<1>(hc, inject,o); break;
        case 2: HcReadR<2>(hc, inject,o); break;
        case 3: HcReadR<3>(hc, inject,o); break;
        case 4: HcReadR<4>(hc, inject,o); break;
        case 5: HcReadR<5>(hc, inject,o); break;
        case 6: HcReadR<6>(hc, inject,o); break;
        case 7: HcReadR<7>(hc, inject,o); break;
        default: HcReadR<8>(hc, inject,o); break;
      }
    }
  }

  template <int R>
  void DenseR(const __half* W, int O, int I, const float* x, int ldx, float* y, int ldy) {
    if (I != kHidden) std::abort();
    // dense_rows.h shares each activation chunk through LDS. It wins from
    // R=4 (router 31.5 -> 20.8 us, R=8 81.4 -> 28.5 us); below that the
    // one-wave-per-output kernel is faster. Both are bitwise equal per row.
    if constexpr (R >= 4) LaunchDenseRows<R, 1>(W, O, x, ldx, y, ldy);
    else DenseGemvKernel<R><<<(O + 7) / 8, 256>>>(W, O, x, ldx, y, ldy);
  }

  void LaunchDenseHalf(const __half* W, int O, const __half* x, float* y, int ldy, int R) {
    const dim3 g((R+127)/128,(O+127)/128);
    if (O < 256 && R <= 1024) PrefillDenseTileKernel<<<dim3((R+63)/64,(O+127)/128),256>>>(W,O,x,y,ldy,R);
    else if (O < 256 || O > 1024) PrefillDenseBigKernel<<<g,256>>>(W,O,x,y,ldy,R);
    else PrefillDenseFastKernel<2><<<g,256>>>(W,O,x,y,ldy,R);
  }

  // x_in_xh: above 8 rows, xh_ already holds the fp16 copy of x.
  void LaunchDense(const __half* W, int O, int I, const float* x, int ldx, float* y, int ldy,
                   int R, bool x_in_xh = false) {
    if (R > 8 && VerificationSpansActive()) {
      // Packed verify: the decode kernel at kPackedRows rows, so every row
      // is bitwise its R <= 8 value. Rows >= R are scratch.
      if (R > kPackedRows) std::abort();
      DenseR<kPackedRows>(W, O, I, x, ldx, y, ldy);
      return;
    }
    if (R > 8) {
      if (I != kHidden || ldx != kHidden) std::abort();
      if (!x_in_xh) PrefillPackKernel<<<(R*kHidden+255)/256,256>>>(x,xh_,nullptr,R*kHidden);
      // Measured (bench/prefill_dense.hip, us at R4096 / R1024): a/b O=96
      // tile 116 / 32, big 98 / 51; router O=513 big 557 / 113, fast 410 /
      // 104; PLE kv O=12800 big 8015 / 2144, fast 8066 / 2108. All are
      // bitwise equal.
      LaunchDenseHalf(W,O,xh_,y,ldy,R);
      return;
    }
    switch (R) {
      case 1: DenseR<1>(W, O, I, x, ldx, y, ldy); break;
      case 2: DenseR<2>(W, O, I, x, ldx, y, ldy); break;
      case 3: DenseR<3>(W, O, I, x, ldx, y, ldy); break;
      case 4: DenseR<4>(W, O, I, x, ldx, y, ldy); break;
      case 5: DenseR<5>(W, O, I, x, ldx, y, ldy); break;
      case 6: DenseR<6>(W, O, I, x, ldx, y, ldy); break;
      case 7: DenseR<7>(W, O, I, x, ldx, y, ldy); break;
      default: DenseR<8>(W, O, I, x, ldx, y, ldy); break;
    }
  }

  // With CRUCIBLE_PLE_HOST, every row upload starts the CPU gather of those
  // rows, so the reads overlap the embedding and layer 0 on the device.
  void PleRowsUploaded(const int64_t* host, int count) {
    if (!m_.ple.host) return;
    FinishPleGather();  // a forward that threw before Ple() leaves one behind
    ple_host_rows_.assign(host, host + count);
    ple_gather_start_ = std::chrono::steady_clock::now();
    ple_host_count_ = count;
    // A launch above the pinned stage's rows is gathered in pieces by GatherPle.
    if (count > PleStageRows() * 16) return;
    uint16_t* out = nullptr;
    WithPleStage([&](auto& stage) { out = stage.Begin((size_t)count * m_.ple.RowWords()); });
    m_.ple.host->Start(ple_host_rows_.data(), (size_t)count, reinterpret_cast<uint8_t*>(out));
  }

  void FinishPleGather() {
    if (!ple_host_count_) return;
    try {
      m_.ple.host->Wait();
    } catch (const std::exception& error) {
      std::fprintf(stderr, "PLE host gather failed: %s\n", error.what());
      std::abort();
    }
  }

  // The packed rows in [R][16] order: the file bytes the resident path uploads.
  const uint16_t* GatherPle(int R) {
    if (ple_host_count_ != R * 16) {
      std::fprintf(stderr, "PLE host gather has no rows for this step\n");
      std::abort();
    }
    const auto waited = std::chrono::steady_clock::now();
    const size_t per_row = 16 * m_.ple.RowWords();
    const int stage_rows = PleStageRows();
    if (R <= stage_rows) {
      FinishPleGather();
      ple_host_count_ = 0;
      WithPleStage([&](auto& stage) { stage.Commit(ple_packed_, (size_t)R * per_row); });
    } else {
      // A launch above the stage rows (CRUCIBLE_PF_LAUNCH_ROWS) started no gather in PleRowsUploaded and fills
      // the stage once a piece. Each piece after the first waits for the previous piece's copy, which runs
      // after layer 0 on the stream, so the device idles while the host gathers the next piece.
      ple_host_count_ = 0;
      for (int o = 0; o < R; o += stage_rows) {
        const int n = std::min(stage_rows, R - o);
        WithPleStage([&](auto& stage) {
          stage.FillTo(ple_packed_ + (size_t)o * per_row, (size_t)n * per_row, [&](uint16_t* out) {
            m_.ple.host->Gather(ple_host_rows_.data() + (size_t)o * 16, (size_t)n * 16, reinterpret_cast<uint8_t*>(out));
          });
        });
      }
    }
    if (ple_trace_) {
      auto& st = m_.ple.host->stats();
      const auto now = std::chrono::steady_clock::now();
      std::fprintf(stderr, "ple host gather: %d rows, %.3f ms since start, %.3f ms waited (sum hint %.3f ms, "
                   "copy %.3f ms, fresh pages %llu)\n", R,
                   std::chrono::duration<double, std::milli>(now - ple_gather_start_).count(),
                   std::chrono::duration<double, std::milli>(now - waited).count(),
                   st.hint_ns.exchange(0) / 1e6, st.copy_ns.exchange(0) / 1e6,
                   (unsigned long long)st.fresh_pages.exchange(0));
    }
    return ple_packed_;
  }

  // The stages have the same interface; one exists (CRUCIBLE_PLE_BF16 or the row width picks it).
  template <class F>
  void WithPleStage(F&& f) {
    if (ple_stage16_) f(*ple_stage16_);
    else if (ple_stage8_) f(*ple_stage8_);
    else f(*ple_stage_);
  }

  template <int K>
  void LaunchPleGather(int R) {
    if (m_.ple.host)
      PleGatherKernel<K, true><<<dim3(16, R), 160>>>(GatherPle(R), rows_, m_.ple.head_bias, eple_);
    else
      PleGatherKernel<K><<<dim3(16, R), 160>>>(m_.ple.table, rows_, m_.ple.head_bias, eple_);
  }

  void Ple(int R, int pos0, float* H = nullptr, int ldh = kWide) {
    H = H ? H : H_;
    if (m_.ple.bf16) PleBf16GatherKernel<<<dim3(kPleHeads, R), kPleRowDim>>>(GatherPle(R), eple_);
    else if (m_.ple.K == 5) LaunchPleGather<5>(R);
    else if (m_.ple.K == 8) LaunchPleGather<8>(R);
    else LaunchPleGather<6>(R);
    LaunchDense(m_.ple.kv, 12800, kHidden, eple_, kHidden, kv_, 12800, R);
    if (VerificationSpansActive()) VerificationPle(H,ldh);
    else
    if (batch_streams_.empty()) {
      PleArgs a{kv_, m_.ple.nk1, m_.ple.nq1, m_.ple.nc1, m_.ple.conv, sequence_->ple_ring, H, pos0, R, ldh};
      if (R > kMaxRows && prefill_rows_) LaunchPrefillPle(a, pf_plenv_, pf_plegate_);
      else PleMixKernel<<<kStreams, 512>>>(a);
    } else {
      if (H != H_ || ldh != kWide) std::abort();
      for (int r = 0; r < R; ++r) {
        SequenceState& s = *sequences_[batch_streams_[r]];
        PleArgs a{kv_ + (size_t)r * 12800, m_.ple.nk1, m_.ple.nq1, m_.ple.nc1, m_.ple.conv,
                  s.ple_ring, H_ + (size_t)r * kWide, (int)s.hist.size() - 1, 1};
        PleMixKernel<<<kStreams, 512>>>(a);
      }
    }
  }

  // Image token rows take the encoder output instead of the pad embedding (HF masked_scatter).
  static void OverlayImages(const SequenceState& state, int pos0, int R, float* H, int ldh) {
    for (const auto& span : state.image_spans) {
      const int begin = std::max(pos0, span.offset), end = std::min(pos0 + R, span.offset + span.count);
      if (begin >= end) continue;
      vision::OverlayKernel<<<dim3(kHidden / 256, end - begin), 256>>>(
          span.embeds + size_t(begin - span.offset) * kHidden, H + size_t(begin - pos0) * ldh, ldh);
    }
  }

  QsaArgs QArgs(const Qsa& w, const QsaCache& c, __half* xh_o, int R, int pos0, const int* mrope) {
    QsaArgs a{};
    a.mrope = mrope;
    a.q = q_;
    a.k = k_;
    a.v = v_;
    a.idx = idx_;
    a.q_norm1 = w.q_norm1;
    a.k_norm1 = w.k_norm1;
    a.iq_norm1 = w.iq_norm1;
    a.ik_norm1 = w.ik_norm1;
    a.c = c;
    a.scores = scores_;
    a.sstride = ctx_ / 4;
    a.sel = sel_;
    a.nsel = nsel_;
    a.suh_o = w.o.suh;
    a.xh_o = xh_o;
    a.pos0 = pos0;
    a.R = R;
    return a;
  }

  // Bulk prefill keeps its larger temporary alive through its blocking result.
  std::vector<int64_t> PleRows(int pos0, int R) {
    std::vector<int64_t> rows((size_t)R * 16);
    PleRows(pos0, R, rows.data());
    return rows;
  }

  // Hashed n-gram rows. Only <|endoftext|> (248044) cuts history; the
  // current token never erases its own context.
  void PleRows(int pos0, int R, int64_t* rows) { PleRowsOf(sequence_->hist.data(), pos0, R, rows); }

  // hist holds the tokens at positions 0 .. pos0 + R - 1.
  void PleRowsOf(const int* hist, int pos0, int R, int64_t* rows) const {
    const struct Ple& p = m_.ple;
    for (int r = 0; r < R; ++r) {
      const int t = pos0 + r;
      int e = -1;
      for (int i = t - 1; i >= 0 && i >= t - 2; --i)
        if (hist[i] == kEos) {
          e = i;
          break;
        }
      auto tok = [&](int k) -> uint64_t {
        const int s = t - k;
        return (s < 0 || s <= e) ? (uint64_t)kEos : (uint64_t)hist[s];
      };
      const uint64_t h2 = (tok(0) * (uint64_t)p.mult[0]) ^ (tok(1) * (uint64_t)p.mult[1]);
      const uint64_t h3 = h2 ^ (tok(2) * (uint64_t)p.mult[2]);
      for (int h = 0; h < 16; ++h)
        rows[r * 16 + h] = (int64_t)((h < 8 ? h2 : h3) % (uint64_t)p.vocab[h]) + p.offset[h];
    }
  }

  // R rows of stride ldh as one tight device block: H itself, or a copy in the debug staging buffer.
  const float* TightRows(const float* H, int ldh, int R) {
    if (ldh == kWide) return H;
    const size_t floats = (size_t)R * kWide;
    if (dump_pack_floats_ < floats) {
      if (dump_pack_) HIP_OK(hipFree(dump_pack_));
      HIP_OK(hipMalloc(&dump_pack_, floats * sizeof(float)));
      dump_pack_floats_ = floats;
    }
    HIP_OK(hipMemcpy2D(dump_pack_,kWide*sizeof(float),H,ldh*sizeof(float),
                       kWide*sizeof(float),R,hipMemcpyDeviceToDevice));
    return dump_pack_;
  }

  void Dump(const std::string& dir, int idx, int R, const float* H = nullptr, int ldh = kWide) {
#if CRUCIBLE_DEBUG_NONFINITE & 13
    DebugAfterLayer(idx, R, const_cast<float*>(H ? H : H_), ldh);
#endif
    if (dir.empty()) return;
    H = H ? H : H_;
    std::vector<float> h((size_t)R * kWide);
    // A strided copy to pageable memory is a GPU blit that writes the host
    // pages directly; with HSA_USE_SVM=0 (runtime_memory_policy.h) it faults.
    // Pack on the device first, then copy one contiguous block to the host.
    // Decode dumps every layer on every step, so the staging buffer persists.
    HIP_OK(hipMemcpy(h.data(), TightRows(H, ldh, R), h.size() * sizeof(float), hipMemcpyDeviceToHost));
    char name[64];
    std::snprintf(name, sizeof(name), "/h_%02d.f32", idx);
    if (batch_streams_.empty()) AppendDump(dir + name, h.data(), h.size());
    else for (int r = 0; r < R; ++r)
      AppendDump(dir + "/stream_" + std::to_string(batch_streams_[r]) + name,
                 h.data() + (size_t)r * kWide, kWide);
  }

  void Mark(const char* name) {
#ifdef CRUCIBLE_QSA_TRACE
    StageTrace(name);
#endif
    if (!prof_) return;
    hipEvent_t e;
    HIP_OK(hipEventCreate(&e));
    HIP_OK(hipEventRecord(e, 0));
    marks_.push_back({e, profile_scope_ + name});
  }
  void Collect() {
    if (!prof_ || marks_.empty()) return;
    HIP_OK(hipEventSynchronize(marks_.back().first));
    for (size_t i = 1; i < marks_.size(); ++i) {
      float ms;
      HIP_OK(hipEventElapsedTime(&ms, marks_[i - 1].first, marks_[i].first));
      ptime_[marks_[i].second] += ms;
    }
    for (auto& m : marks_) HIP_OK(hipEventDestroy(m.first));
    marks_.clear();
  }

  // CRUCIBLE_ROUTE_DUMP=FILE (analysis only; it synchronizes): appends one record per trunk MoE call:
  // int32 layer, int32 R, then R x kTopK int32 expert IDs.
  FILE* route_dump_ = crucible::SwitchText("CRUCIBLE_ROUTE_DUMP") ? std::fopen(crucible::SwitchText("CRUCIBLE_ROUTE_DUMP"), "ab") : nullptr;
  void DumpRoutes(int li, int R) {
    std::vector<int> ids(size_t(R) * kTopK);
    HIP_OK(hipMemcpy(ids.data(), sel_moe_, ids.size() * sizeof(int), hipMemcpyDeviceToHost));
    const int head[2] = {li, R};
    std::fwrite(head, sizeof(int), 2, route_dump_);
    std::fwrite(ids.data(), sizeof(int), ids.size(), route_dump_);
    std::fflush(route_dump_);
  }
  LookaheadRunner lookahead_;
  unsigned long long *ecount_ = nullptr, *ecbuf_ = nullptr;
  int *mapj_ = nullptr, *rowl_ = nullptr;
  unsigned* moe_done_ = nullptr;
  unsigned* moe_sync_ = nullptr;  // MoeStageKernel's counters (zero between its launches)
  unsigned* qsa_sync_ = nullptr;  // QsaAttnTailKernel's counters (zero between its launches)
  std::unique_ptr<AsyncHead> overlap_;
  std::unique_ptr<BlockDraftHead> block_head_;
  bool draft_last_only_ = true;
  int* draft_blocks_ = nullptr;
  Lin draft_head_;
  __half* draft_dense_ = nullptr;
  int draft_size_ = 0;
  Group draft_group_;
  DraftChoice* choice_ = nullptr;
  float draft_margin_ = INFINITY;
  std::string profile_scope_;
  std::string PrefillWidthIssue() const;
  static std::vector<int> MoeLayerWidths(const Layer& L);
  int ChoosePrefillRows(int requested);
  void InitPrefill();
  const PrefillPlans& PrefillPlansFor(int R);
  void FreePrefillPlans(bool oldest_only);
  void PrefillBlock(int li, const Layer& L, const PrefillLayer& P, int R, int pos0, bool cache_only = false,
                    int first = 0);
  bool PrefillPrunes(int R, bool predict, int keep, const std::string& dump) const;
  int PrefillFirstRow(int R, bool predict, int keep, const std::string& dump) const;
  void CheckChunk(int pos0, int R, bool predict);
  void BeginPrefillJoin(int R, int pos0);
  void PrefillCutAfter(int end, std::initializer_list<const void*> sources);
  void PrefillCutPle(int pos0);
  const MtpPrefillPlan* MtpPrefillPlanFor(int R);
  // Bulk MTP plans by row count (a full chunk and the last partial chunk); null: the 8-row path for that count.
  std::map<int, std::unique_ptr<MtpPrefillPlan>> mtp_prefill_plans_;
  InterleaveCheck interleave_check_;

  std::unique_ptr<PinnedInputBuffer<int, kMaxPrefillLaunchRows>> pf_host_tokens_;
  std::unique_ptr<PinnedInputBuffer<int64_t, kMaxPrefillLaunchRows * 16>> pf_host_rows_;
  PinnedInputBuffer<int, kPackedRows> host_tokens_;
  PinnedInputBuffer<int64_t, kPackedRows * 16> host_rows_;
  PinnedBuffer<int, kPackedRows> host_ids_;
  bool prof_ = false, warming_ = false;
  std::vector<DeviceSpan> buffers_;
  std::vector<std::pair<hipEvent_t, std::string>> marks_;
  std::map<std::string, double> ptime_;
  Model m_;
  std::vector<std::unique_ptr<SequenceState>> sequences_;
  SequenceState* sequence_ = nullptr;
  std::unique_ptr<VisionEncoder> vision_;
  bool compact_state_ = false;
  std::vector<int> batch_streams_;
  int ctx_, mtp_rows_, prefill_rows_, pf_launch_rows_ = 0;
  int pf_join_rows_ = 0, pf_room_rows_ = 0;  // x4-tail: join rows, and launch rows plus join rows
  // x4-tail: the spec of the next Prefill call, and the cut of the running call with its copies done.
  crucible::PrefillJoinSpec pf_join_{}, pf_cut_{};
  bool pf_join_set_ = false;
  std::vector<char> pf_cut_done_;
  size_t xh_capacity_, partial_capacity_;
  // Budget: 1 GiB and 16 entries, or one oversized custom plan. Scratch is separate.
  std::map<int,std::pair<uint64_t,std::unique_ptr<PrefillPlans>>> prefill_plans_;
  size_t prefill_plan_bytes_ = 0;
  float* dump_pack_ = nullptr;  // --dump staging for strided bulk H; debug only, never freed
  size_t dump_pack_floats_ = 0;
  uint64_t plan_tick_ = 0;
  std::string prefill_path_;
  prefill::GdnPrefill pf_gdn_{};
  PrefillMoe pf_moe_{};
  Group pf_head_;
  float *pf_ab_, *pf_y_, *pf_context_;
  __half *pf_qhi_, *pf_qlo_, *pf_thi_, *pf_tlo_;
  PrefillTiles pf_tiles_;
  bool pf_gdn_micro_ = false;
  // Last-layer row pruning (idea from Gufo, gufo-org/gufo, MIT). CRUCIBLE_PF_PRUNE=0 turns it off.
  bool pf_prune_ = true;
  // CRUCIBLE_PF_EXACT_SPLIT: keep the low halves of the HC dots and QSA
  // operands and fp32 pd (the bitwise pre-numerics path).
  bool pf_split_ = false;
  prefill::QsaUnion pf_qsa_union_{};
  float* pf_qsa_select_ = nullptr;
  float* pf_prompt_stacks_ = nullptr;
  __half* pf_xpack_ = nullptr;
  __half* pf_x_ = nullptr;
  // Keep public H() tight for decode, MTP, batching and retained-prefix readers.
  float* pf_H_ = nullptr;
  int pf_hstride_ = kWide;
  bool pf_pad_h_ = false;      // pf_H_ exists (CRUCIBLE_PF_PAD_H), so H_ holds only kPackedRows rows
  int last_prefill_rows_ = 0;  // rows of the last Prefill call, for CopyPrefillStacks
  float* PrefillH(int R) const { return R > kMaxRows && pf_H_ ? pf_H_ : H_; }
  int PrefillHStride(int R) const { return R > kMaxRows && pf_H_ ? pf_hstride_ : kWide; }
  __half* pf_plenv_ = nullptr;
  float* pf_plegate_ = nullptr;
  std::map<int, std::unique_ptr<Plans>> plans_;
  std::map<int, std::array<BalancedItems, 2>> pgdn_items_;
  uint32_t *pgdn_bar_ = nullptr, *pgdn_error_ = nullptr;
  // The persistent R=1 path has the VALU HC arithmetic, so it is a numerics switch.
  bool use_pgdn_ = NumericsOn("CRUCIBLE_PERSISTENT_GDN");
  // The R=1 ring keeps strided chain ranges: balanced ranges put concurrent
  // blocks on distant front chunks and cost 12.7 points of DRAM rate (dense
  // R=1, fwd v3); an earlier change deleted them.
  int max_pass_rows_ = 0;  // MaxPassRows, once known
  // The R=1 ring reads front order at 84% of DRAM against 73% for LinGemv, but
  // engine order at 72% (fwd v3). CRUCIBLE_DECODE_RING=0 or 1 forces it.
  enum RingMode { kRingOff, kRingOn, kRingFront };
  RingMode ring_mode_ = [] {
    const char* v = crucible::SwitchText("CRUCIBLE_DECODE_RING");
    if (v && std::string(v) == "0") return kRingOff;
    if (v && std::string(v) == "1") return kRingOn;
    return kRingFront;
  }();
  static constexpr int kRingBlocks = 80;
  float *mE_, *mN_, *mhp_, *mep_;
  // The WMMA HC read (hc_wmma.h) serves decode, verify, the MTP last row and
  // chunked prefill with one arithmetic for every R. Bulk prefill (R > 8)
  // keeps its own kernels. CRUCIBLE_HC_WMMA=0 selects the VALU kernels.
  const bool hc_wmma_ = NumericsInt("CRUCIBLE_HC_WMMA", 1, 0, 1) != 0;
  HcWScratch hcw_{};
  unsigned hcw_epoch_ = 0;
  float *H_, *x_, *dots_, *ss_, *post_, *qkv_, *z_, *q_, *k_, *v_, *idx_, *rlog_, *wts_;
  // CRUCIBLE_ABLATE: per layer [a; b] on the device, or null; y scratch [rows][kHidden].
  std::array<const float*, kLayers> abl_attn_{}, abl_mlp_{};
  float* abl_y_ = nullptr;
  int abl_rows_ = 0;
  float *eple_, *kv_, *logits_, *partial_, *scores_, *qsa_scratch_;
  mutable PinnedVector<float> logit_host_;
  __half* xh_;
  int *tok_, *ids_, *sel_, *nsel_, *sel_moe_, *map_;
  int64_t* rows_;
  uint16_t* ple_packed_ = nullptr;
  // prefill-x: the K-bit and 8-bit stages hold the largest launch (CRUCIBLE_PF_LAUNCH_ROWS with its join), so the
  // gather starts at the upload and overlaps layer 0 (about 20 ms of device idle less per 8192-row launch). 64 MiB
  // (61-word rows) or 81 MiB (81-word rows) pinned; only one of them exists.
  static constexpr int kPleStageRows = kMaxPrefillLaunchRows;
  int PleStageRows() const { return ple_stage16_ ? kMaxPrefillRows : kPleStageRows; }
  std::unique_ptr<PinnedInputBuffer<uint16_t, kPleStageRows * 16 * 61>> ple_stage_;
  std::unique_ptr<PinnedInputBuffer<uint16_t, kMaxPrefillRows * 16 * kPleRowDim>> ple_stage16_;
  std::unique_ptr<PinnedInputBuffer<uint16_t, kPleStageRows * 16 * 81>> ple_stage8_;
  std::vector<int64_t> ple_host_rows_;
  // Destroyed before the rows and the stages: a forward that threw between
  // PleRowsUploaded and Ple() leaves a gather that still writes there.
  PleDrainGuard ple_drain_{&m_.ple.host};
  int ple_host_count_ = 0;
  std::chrono::steady_clock::time_point ple_gather_start_;
  const bool ple_trace_ = crucible::SwitchText("CRUCIBLE_PLE_TRACE") != nullptr;
};

#include "runtime/prefill_engine.inc"

std::vector<int> ParseIds(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(std::stoi(t));
  return v;
}

double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

// Copies stack rows [r0, r0 + n) of the engine's residual buffer to dst rows.
void CopyRows(Engine& e, float* dst, int dst_row, int r0, int n) {
  HIP_OK(hipMemcpyAsync(dst + (size_t)dst_row * kWide, e.H() + (size_t)r0 * kWide,
                        sizeof(float) * kWide * n, hipMemcpyDeviceToDevice, 0));
}

#include "runtime/conversation_snapshot.h"
#include "runtime/slot_checkpoint_io.h"
#include "runtime/decode_policy.h"
#include "cli/prefill_bench.h"
#include "cli/gen_cli.h"
