// Included after Engine is defined. stdout contains protocol messages only.
#pragma once
#include <iostream>
#include <bit>
#include "../runtime/gen_options.h"
#include "../runtime/generation.h"
#include "../runtime/prefill_rows.h"
#include "serve_settings.h"
#include "../runtime/conversation_admission.h"
#include "../switches.h"

// Image requests carry file paths, not pixels, so every line has one limit.
inline bool ServeLine(std::string& line, bool& oversized) {
  constexpr size_t kMaxLine = 2 * 1024 * 1024;
  line.clear();
  oversized = false;
  char c;
  while (std::cin.get(c)) {
    if (c == '\n') return true;
    if (line.size() < kMaxLine) line += c; else oversized = true;
  }
  return !line.empty() || oversized;
}
inline std::string VisionDoneFields(size_t images, double seconds) {
  char text[96];
  std::snprintf(text, sizeof(text), ",\"images\":%zu,\"vision_seconds\":%.9f", images, seconds);
  return text;
}
inline int Serve(const std::string& ckpt, const ServeSettings& settings) {
  static_assert(kPrefillAuto == kServePrefillAuto && kMaxPrefillRows == kServeMaxPrefillRows);
  const int ctx = settings.ctx;
  const size_t cache_budget = size_t(settings.cache_mib) << 20;
  Engine eng(ckpt, ctx, settings.prefill_rows);
  // As in batch serve: a lazy compact MTP cache would fail on memory at a request, not at startup.
  try { eng.EnsureServeMtpCaches(); }
  catch (const std::exception& error) {
    std::fprintf(stderr, "serve error: %s\n", error.what());
    return 2;
  }
  if (settings.vision) eng.EnableVision(ckpt);
  PrefixCache<ConversationSnapshot, ConversationPolicy> cache(cache_budget);
  // The budget is this server's snapshot memory, so evicted snapshot chunks stay pinned for the next save.
  SnapshotChunkPool::Instance().SetPinnedLimit(cache_budget);
  const uint64_t host_reserve = uint64_t(settings.host_reserve_mib) << 20;
  if (cache_budget)
    std::fprintf(stderr, "conversation cache: %zu bytes budget, host reserve %d MiB\n", cache_budget,
                 settings.host_reserve_mib);
  size_t requests = 0, hits = 0, saved_tokens = 0;
  HIP_OK(hipDeviceSynchronize());
  std::printf("{\"event\":\"ready\",\"context\":%d,\"cache_tail\":true,%s}\n", ctx, crucible::ReadyModelJson().c_str());
  std::fflush(stdout);
  std::string line;
  bool oversized;
  while (ServeLine(line, oversized)) {
    SequenceState* bound = nullptr;
    try {
      if (oversized) throw std::runtime_error("request exceeds the 2 MiB JSON-line limit");
      const ServeRequest request = ParseServeRequest(line, ctx, settings.policy);
      if (!request.images.empty() && !eng.HasVision())
        throw std::runtime_error("this engine runs without the vision tower; start serve with --vision 1");
      // Image tokens key the cache by their pixels, so two images never share an entry.
      const std::vector<int> cache_key = CacheKey(request.ids, request.images);
      const auto key_of = [&](const std::vector<int>& hist) { return CacheKey(hist, request.images); };
      GenOptions options = CliGenOptions();
      options.ids = request.ids; options.n = request.max_tokens; options.ctx = ctx;
      options.spec = request.spec; options.stop_ids = request.stop_ids; options.sampling = request.sampling;
      options.draft_q = request.draft_q; options.draft_head = request.draft_head; options.mtp_prefix = request.mtp_prefix;
      options.lookup = request.lookup; options.lookup_min = request.lookup_min;
      options.adapt_set = request.draft_adapt >= 0;
      options.adapt = std::max(0, request.draft_adapt);
      options.prefill_rows = eng.PrefillRows();
      ConversationHooks hooks;
      const auto draft_selection = ResolveDraftBlocks(options, {});
      const int effective_spec = options.n == 1 ? 0 : options.spec;
      const ConversationPolicy policy{effective_spec, options.chunk, eng.PrefillRows(),
          options.lookup, options.lookup_min, options.sampling.temperature > 0,
          options.draft_dense, AdaptiveDraft(options), options.last_head != 0, options.overlap != 0,
          std::bit_cast<uint32_t>(options.draft_margin), std::bit_cast<uint32_t>(options.draft_q), draft_selection,
          options.head_auto != 0, options.mtp_prefix, request.tenant};
      // Session replay keys tails by tokens only (as batch serve does): the draft policy changes only speed. The
      // tenant stays: a tail matches only inside its tenant.
      ConversationPolicy tail_policy = ConversationTailPolicy(eng.PrefillRows(), options.chunk);
      tail_policy.tenant = request.tenant;
      const bool enabled = cache_budget && request.cache_control != "bypass";
      const int rows = eng.PrefillRows() > 0 ? eng.PrefillRows() : options.chunk;
      const auto admission = ConversationAdmission::ForPrompt(request.ids.size(), request.cache_prefix_tokens,
                                                               rows, effective_spec ? options.mtp_prefix : 0,
                                                               settings.exact_retry);
      hooks.admit_prompt = [admission](size_t length) { return admission.Allows(length); };
      // The fresh-request contract (default) saves prompt states only. Session replay also saves the generated tail.
      hooks.save_generated_tail = settings.session_replay;
      hooks.defer_prompt_end_save = enabled;
      // Short bulk prompts have no aligned frontier for a changed conversation boundary.
      hooks.skip_prompt_interior_save = enabled && eng.PrefillRows() > 0 &&
                                        request.ids.size() <= size_t(eng.PrefillRows());
      std::shared_ptr<const ConversationSnapshot> snapshot;
      int tail_replay = 0;  // a replayed tail's pending GDN rows (done event evidence for the replay gate)
      ++requests;
      // Session replay: the longest earlier generated tail whose tokens this prompt repeats exactly. Any mismatch
      // finds nothing here and falls back to the default prompt-state search.
      if (enabled && settings.session_replay && request.cache_control == "auto" && request.ids.size() > 1) {
        snapshot = cache.Find(cache_key, tail_policy, request.ids.size() - 1,
                              [](const ConversationSnapshot& entry) { return !entry.resume.prompt_only; });
        hooks.continue_tail = snapshot != nullptr;
        tail_replay = snapshot ? snapshot->resume.replay : 0;
      }
      if (enabled && request.cache_control == "auto" && !snapshot) {
        const size_t limit = effective_spec && options.mtp_prefix
                           ? DraftWarmupStart(request.ids.size(), options.mtp_prefix) : request.ids.size();
        snapshot = cache.Find(cache_key, policy, limit, [&](const ConversationSnapshot& entry) {
          return entry.resume.prompt_only &&
                 ConversationPromptCompatible(entry.resume, request.ids.size(), request.cache_prefix_tokens,
                                              eng.PrefillRows(), hooks.split_at_boundary);
        });
      }
      const size_t cached_tokens = snapshot ? snapshot->hist.size() : 0;
      size_t cache_saves = 0, cache_saved_bytes = 0;
      if (snapshot) { ++hits; saved_tokens += cached_tokens; }
      // Bypass uses the same explicit boundary as auto, without cache reads or writes.
      if (cache_budget) { hooks.boundary = request.cache_prefix_tokens; options.conversation = &hooks; }
      if (enabled) {
        if (snapshot) hooks.restore = [&](float* stk, float* carry) {
          const auto state = ConversationStateIO::Restore(eng, eng.ActiveSequence(), *snapshot, stk, carry);
          snapshot.reset();
          return state;
        };
        // Session replay saves a tail after every turn, also when the prompt has no admitted frontier (a prompt
        // shorter than one chunk); batch serve does the same.
        if (admission.Any() || settings.session_replay) hooks.save = [&](const ConversationResume& state, float* stk, float* carry) {
          // After a replay restore the prompt rows follow a shifted schedule. Saving them as prompt states would let
          // a later fallback reuse non-bulk state, so a replayed turn stores only its generated tail.
          if (state.prompt_only ? hooks.continue_tail || !admission.Allows(state.length) : !settings.session_replay)
            return;
          const size_t bytes = ConversationStateIO::Bytes(eng, eng.ActiveSequence(), state, stk, carry) +
                               crucible::kConversationPolicyBytes + policy.draft_selection.size() * sizeof(int);
          if (bytes > cache.Budget()) return;
          std::ifstream memory("/proc/meminfo");
          std::string key; size_t available = 0, value; std::string unit;
          while (memory >> key >> value >> unit)
            if (key == "MemAvailable:") { available = value * 1024; break; }
          available += SnapshotChunkPool::Instance().IdleBytes();  // idle chunks serve a save without the system
          // Budget and the host reserve (--host-reserve-mib) together, as in batch serve: evicted snapshots count
          // toward the reserve, and a save that cannot fit evicts nothing.
          if (!cache.ReserveHost(bytes, available, host_reserve)) {
            std::fprintf(stderr, "conversation cache: skipped snapshot to retain %d MiB of host memory\n",
                         settings.host_reserve_mib);
            return;
          }
          try {
            const auto saved = ConversationStateIO::Save(eng, eng.ActiveSequence(), state, stk, carry);
            ++cache_saves;
            cache_saved_bytes += saved->bytes.size();
            cache.Insert(key_of(saved->hist), state.prompt_only ? policy : tail_policy, bytes, saved);
          } catch (const std::bad_alloc&) {
            std::fprintf(stderr, "conversation cache: snapshot allocation failed; reduce the cache budget\n");
          }
        };
        options.conversation = &hooks;
      }
      options.emit = [](int id) {
        std::printf("{\"event\":\"token\",\"id\":%d}\n", id);
        std::fflush(stdout);
      };
      double vision_s = 0;
      if (!request.images.empty()) {
        bound = &eng.ActiveSequence();
        vision_s = eng.BindImages(*bound, request.images, cached_tokens);
      }
      const auto result = RunGen(eng, options, draft_selection);
      if (bound) { eng.ClearImages(*bound); bound = nullptr; }
      // Opt-in draft statistics on stderr (G15 A/B: acceptance with and without session replay).
      if (crucible::SwitchText("CRUCIBLE_SPEC_STATS"))
        std::fprintf(stderr, "spec_stats replay=%d cached=%zu completion=%zu forwards=%ld accepted=%ld\n",
                     int(hooks.continue_tail), cached_tokens, result.gen.size(), result.forwards, result.accepted);
      // Reset waits on the same stream before the next request reads any state.
      HIP_OK(hipDeviceSynchronize());
      std::printf("{\"event\":\"done\",\"prompt_tokens\":%zu,\"completion_tokens\":%zu,"
                  "\"finish_reason\":\"%s\",\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,"
                  "\"cache_hit\":%s,\"cached_tokens\":%zu,\"prefill_tokens\":%zu,\"cache_bytes\":%zu,"
                  "\"cache_evictions\":%zu,\"cache_hits\":%zu,\"cache_requests\":%zu,\"saved_prefill_tokens\":%zu,"
                  "\"cache_saves\":%zu,\"cache_saved_bytes\":%zu%s%s}\n",
                  request.ids.size(), result.gen.size(), result.stopped ? "stop" : "length",
                  result.prefill_s, result.decode_s, cached_tokens ? "true" : "false", cached_tokens,
                  request.ids.size() - cached_tokens, cache.Bytes(), cache.Evictions(), hits, requests, saved_tokens,
                  cache_saves, cache_saved_bytes,
                  // Only a replay hit adds the key, so the default done line stays byte-identical.
                  hooks.continue_tail ? (",\"session_replay\":true,\"replay_rows\":" + std::to_string(tail_replay)).c_str()
                                      : "",
                  request.images.empty() ? "" : VisionDoneFields(request.images.size(), vision_s).c_str());
    } catch (const std::exception& error) {
      if (bound) eng.ClearImages(*bound);
      std::printf("{\"event\":\"error\",\"error\":%s}\n", JsonQuote(error.what()).c_str());
    }
    std::fflush(stdout);
  }
  return 0;
}
