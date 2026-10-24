// Opt-in protocol. Each command completes before the frontend sends another command.
#pragma once
#include "batch_session.h"
#include "batched_serve_protocol.h"
#include "batched_serve_settings.h"
#include "batch_conversation_cache.h"
#include "batch_state_disk.h"
#include "batch_mtp_history.h"
#include "slot_base.h"
#include "draft_tally.h"
#include <atomic>
#include <csignal>
#include "../runtime/draft_accept.h"
#include "../runtime/draft_policy.h"
#include "../runtime/draft_stop.h"
#include "../runtime/draft_value.h"
#include "../runtime/prefill_rows.h"
#include "../switches.h"
#include <numeric>
#include <tuple>

inline void PrintBatchEvent(const crucible::BatchEvent& event) {
  const auto id = static_cast<unsigned long long>(event.request_id);
  const auto generation = static_cast<unsigned long long>(event.slot_generation);
  if (event.error) {
    // Only this request ends; the session continues (an image file failed its check after the open).
    std::printf("{\"event\":\"error\",\"request_id\":%llu,\"slot_generation\":%llu,\"error\":%s,\"recoverable\":true}\n",
                id, generation, JsonQuote(*event.error).c_str());
  } else if (event.token) {
    std::string scores;  // only a scoring request adds this field, so other token lines stay byte-identical
    for (size_t i = 0; i < event.logprobs.size(); ++i) {
      char value[32];
      std::snprintf(value, sizeof value, "%.9g", double(event.logprobs[i].logprob));
      scores += (i ? "," : ",\"logprobs\":[") + ("[" + std::to_string(event.logprobs[i].id) + "," + value + "]");
    }
    if (!scores.empty()) scores += "]";
    std::printf("{\"event\":\"token\",\"request_id\":%llu,\"slot_generation\":%llu,\"id\":%d%s%s}\n",
                id, generation, *event.token, crucible::BatchStepDraftFields(event).c_str(), scores.c_str());
  } else if (event.done) {
    std::printf("{\"event\":\"done\",\"request_id\":%llu,\"slot_generation\":%llu,"
                "\"prompt_tokens\":%zu,\"completion_tokens\":%zu,\"finish_reason\":\"%s\","
                "\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,\"cache_hit\":%s,\"cached_tokens\":%zu,"
                "\"prefill_tokens\":%zu,\"cache_bytes\":%zu,\"restore_seconds\":%.9f,\"capture_seconds\":%.9f,"
                "\"slot_context\":%d%s%s%s%s}\n",
                id, generation, event.prompt_tokens, event.completion_tokens, event.stopped ? "stop" : "length",
                event.prefill_seconds, event.decode_seconds, event.cached_tokens ? "true" : "false",
                event.cached_tokens, event.prompt_tokens - event.cached_tokens, event.cache_bytes,
                event.restore_seconds, event.capture_seconds, event.slot_context,
                !event.cached_tokens ? "" : event.cache_tail ? ",\"cache_class\":\"replay\",\"session_replay\":true"
                : event.cache_shared ? ",\"cache_class\":\"shared\"" : ",\"cache_class\":\"session\"",
                event.cached_tokens && event.cache_tail ? (",\"replay_rows\":" + std::to_string(event.tail_replay)).c_str() : "",
                crucible::BatchCacheFields(event).c_str(), crucible::BatchDraftFields(event).c_str());
  } else {
    // Only image requests add the slice count, so text opened lines stay byte-identical.
    std::printf("{\"event\":\"opened\",\"request_id\":%llu,\"slot_generation\":%llu,\"slot_context\":%d,"
                "\"cached_tokens\":%zu%s}\n", id, generation, event.slot_context, event.cached_tokens,
                event.vision_slices ? (",\"vision_slices\":" + std::to_string(*event.vision_slices)).c_str() : "");
  }
}

// The MTP side of full-window snapshots (BatchSpecDecoder). For each slot: its MTP
// pair buffers, and the MTP history that the slot's MTP cache and carry hold (batch_mtp_history.h).
struct BatchMtpSlots {
  // The floats of one slot's carry (Carry): one trunk stack row. A checkpoint's carry tail copies this many.
  static constexpr size_t kCarryFloats = size_t(kWide);
  virtual ~BatchMtpSlots() = default;
  virtual float* Stacks(int slot) = 0;
  virtual float* Carry(int slot) = 0;
  // A host restore or a checkpoint wrote the slot. `held`: the MTP history that the restore wrote (none: target only).
  virtual void Restored(int slot, std::optional<crucible::MtpHeld> held) = 0;
  // The prefilling request on `slot` after prompt row `cursor`: its MTP history, when it is the whole prefix.
  virtual std::optional<crucible::MtpHeld> Live(int slot, size_t cursor) const = 0;
  // The finished request on `slot` keeps a tail of `logical` rows: its pending pairs go through the MTP up to the
  // tail, and the slot then holds the returned MTP history (none: the request did not warm the whole prefix).
  virtual std::optional<crucible::MtpHeld> Settle(int slot, size_t logical) = 0;
};

// One worker-owned LRU for every slot. Snapshots hold target state; from a full-window request (mtp_prefix 0) they
// also hold the MTP rows of the whole prefix, so a full-window hit continues the MTP warm-up.
class BatchPrefixCache {
 public:
  using Entry = crucible::BatchConversationCache<ConversationSnapshot>::Entry;
  // `streams`: sessions that run at once, each with one protected resume point.
  // `checkpoints`: in-slot resume points (slot_checkpoints.h). Their device bytes come out of the cache budget, so
  // the admitted memory does not change; a budget under twice their size turns them off.
  // Re-send fix A: each buffer also holds the MTP carry, and the logits rows of the End points come out of the same
  // budget.
  BatchPrefixCache(Engine& engine, size_t budget, bool session_replay = false, int streams = kMaxRows,
                   bool slot_resume = true, bool checkpoints = false,
                   uint64_t host_reserve = crucible::kDefaultHostReserve,
                   std::optional<crucible::StateDiskOptions> disk = std::nullopt, bool probation = true,
                   int idle_minutes = 30)
      : engine_(engine), session_replay_(session_replay), slot_resume_(slot_resume),
        checkpoint_bytes_(session_replay && checkpoints ? CheckpointBytes(engine, streams, budget) : 0),
        cut_bytes_(CutBytes(engine, budget - checkpoint_bytes_)),
        cache_(budget - checkpoint_bytes_ - cut_bytes_, {}, size_t(std::max(streams, 1)), host_reserve, probation, 60.0 * idle_minutes),
        rows_(engine.PrefillRows() > 0 ? engine.PrefillRows() : kMaxRows),
        // The snapshot format carries request stacks. spec=0 never reads them, so one zeroed pair serves all slots.
        stk_(Zeros<float>(size_t(kMaxRows + 1) * kWide)), carry_(Zeros<float>(kWide)) {
    // Admission reserved the whole budget, so evicted snapshot chunks stay pinned for the next save, up to the cache's
    // own budget in all (the slot checkpoints are separate device memory).
    crucible::SnapshotChunkPool::Instance().SetPinnedLimit(cache_.Budget());
    std::fprintf(stderr, "conversation cache: %zu bytes budget, host reserve %llu MiB\n", budget,
                 static_cast<unsigned long long>(host_reserve >> 20));
    if (disk && session_replay_) OpenDisk(std::move(*disk), host_reserve);
    // x4-tail: one cut buffer for the server (a joined launch uses it, and the worker saves it before the next one).
    if (cut_bytes_) {
      cut_ = std::make_unique<SlotCheckpointIO::Buffer>(cut_bytes_);
      std::fprintf(stderr, "prefill cut buffer: %zu bytes from the cache budget\n", cut_bytes_);
    }
    if (!checkpoint_bytes_) return;
    const size_t each = BufferBytes(engine);
    for (int index = 0; index < streams * crucible::kSlotCheckpoints; ++index)
      buffers_.push_back(std::make_unique<SlotCheckpointIO::Buffer>(each, kCarryBytes));
    std::fprintf(stderr, "slot checkpoints: %d per slot, %zu bytes each, %zu bytes from the cache budget\n",
                 crucible::kSlotCheckpoints, each, checkpoint_bytes_);
    std::fprintf(stderr, "slot checkpoints: end outputs, %zu bytes of logits rows in that budget\n", LogitsBytes(streams));
    // slot frontiers are on with the checkpoints and slot resume (Hooks); the host live count follows them.
    cache_.SetSlotFrontiers(session_replay_ && slot_resume_);
  }
  crucible::BatchCacheHooks Hooks() {
    crucible::BatchCacheHooks hooks{[this](const crucible::ServeRequest& r) { return Restore(r); },
                                    [this](const crucible::ServeRequest& r, size_t cursor) { Capture(r, cursor); },
                                    [this] { return cache_.Bytes(); }};
    hooks.next_capture = [this](const crucible::ServeRequest& r, size_t cursor) { return NextCapture(r, cursor); };
    if (cut_) {
      // A full-window request saves its MTP rows with a host frontier; the cut buffer holds target state only.
      hooks.cut_ready = [this](const crucible::ServeRequest& r) { return r.cache_control != "bypass" && !MtpHistory(r); };
      hooks.cut_prepare = [this](size_t position, size_t window_start) { return CutPrepare(position, window_start); };
      hooks.cut_capture = [this](const crucible::ServeRequest& r, size_t position) { Capture(r, position, cut_.get()); };
    }
    hooks.status = [this] {
      auto status = cache_.Status();
      status.host_available = crucible::AvailableHostMemory();
      status.pinned_bytes = crucible::SnapshotChunkPool::Instance().PinnedBytes();
      status.counters = counters_;
      return status;
    };
    hooks.slot_served = [this](size_t tokens) { counters_.Count(crucible::BatchCacheTier::Slot, tokens); };
    if (session_replay_) {
      hooks.tail = [this](const crucible::ServeRequest& r, std::span<const int> generated, int replay,
                          crucible::BatchTailSource source) {
        return Tail(r, generated, replay, source);
      };
      hooks.expect = [this](const crucible::ServeRequest& r, int slot, size_t rows) { Expect(r, slot, rows); };
    }
    if (session_replay_ && slot_resume_) {
      hooks.resume = [this](const crucible::ServeRequest& r, crucible::BatchResident resident) {
        return Resume(r, resident);
      };
      hooks.release = [this](int slot, bool resumed) { Release(slot, resumed); };
    }
    hooks.key = [](const crucible::ServeRequest& r, std::span<const int> tokens) { return Key(r, tokens); };
    if (disk_)
      hooks.stage = [this](const crucible::ServeRequest& r, size_t floor, const crucible::BatchStageQueue& queue) {
        return Stage(r, floor, queue);
      };
    hooks.hold = [this](const crucible::ServeRequest* r) {
      if (!r || r->cache_control != "auto" || r->ids.size() < 2) cache_.Hold({});
      else cache_.Hold(Key(*r, r->ids));
    };
    hooks.request = [this](uint64_t id) { request_ = id; };
    if (!buffers_.empty()) {
      hooks.checkpoint_save = [this](int slot, int index) {
        SlotCheckpointIO::Save(engine_.ActiveSequence(), Buffer(slot, index));
        // Fix A: the slot's MTP carry (the last prompt row's trunk stack at a prompt end); a point keeps it only
        // when its request warmed it (PromptOutputs::carry).
        if (mtp_) CopyCarry(Buffer(slot, index), slot, true);
      };
      if (cut_) hooks.cut_save = [this](int slot, int index) { CutSave(Buffer(slot, index)); };
      hooks.checkpoint_load = [this](int slot, int index, const crucible::SlotCheckpoint& point) {
        // A checkpoint holds target state only: a speculative request warms its MTP rows again.
        if (mtp_) mtp_->Restored(slot, std::nullopt);
        // A Frontier resume is a host frontier hit: a whole pool block and prefill chunk, no pending rows (LOW-2).
        if (point.kind == crucible::CheckpointKind::Frontier && !crucible::SlotFrontierGrid(point.hist.size(), rows_, point.replay))
          throw std::logic_error("slot frontier resume needs a grid point with no pending rows");
        SlotCheckpointIO::Restore(engine_.ActiveSequence(), Buffer(slot, index), point);
        if (mtp_ && point.outputs && point.outputs->carry) CopyCarry(Buffer(slot, index), slot, false);
        bases_.Limit(slot, point.hist.size(), point.replay);  // an earlier change
        // A Frontier hit copies nothing from the host: it keeps the session live. The point's key marks
        // what a host hit at the point would mark, and the session's shorter entries (LOW-4: no prompt key here).
        if (point.kind == crucible::CheckpointKind::Frontier) cache_.Touch(point.key, Policy(point.tenant));
        std::fprintf(stderr, "slot_checkpoint resume tokens=%zu replay=%d\n", point.hist.size(), point.replay);
      };
      hooks.peek = [this](const crucible::ServeRequest& r) { return std::max(Peek(r), DiskPeek(r)); };
      // The effect lines of re-send fix A and fix B (the re-send proof reads them).
      hooks.end_resumed = [this](int slot, size_t tokens) {
        std::fprintf(stderr, "slot_checkpoint end_resume tokens=%zu slot=%d request=%llu\n", tokens, slot,
                     static_cast<unsigned long long>(request_));
      };
      hooks.pre_saved = [](int slot, size_t tokens) {
        std::fprintf(stderr, "slot_checkpoint pre_end tokens=%zu slot=%d\n", tokens, slot);
      };
      hooks.diverged = [](size_t prompt, size_t restored, size_t tail, size_t common) {
        std::fputs(crucible::SessionReplayDivergenceLine(prompt, restored, {tail, common, prompt <= tail}).c_str(), stderr);
      };
      // slot frontiers, with slot resume.
      if (session_replay_ && slot_resume_) {
        hooks.frontier = [this](const crucible::ServeRequest& r, size_t cursor,
                                const std::function<bool(const crucible::SlotFrontier&)>& keep) {
          return KeepFrontier(r, cursor, keep);
        };
        hooks.frontier_fits = [this](const crucible::ServeRequest& r, const crucible::SlotCheckpoint& point) {
          return FrontierFits(r, point);
        };
        hooks.handover = [this](int slot, int index, const crucible::SlotCheckpoint& point) { Handover(slot, index, point); };
        hooks.drop = [this](int slot) { Drop(slot); };
        hooks.frontier_copy = [this](const crucible::ServeRequest& r, int from, int index,
                                     const crucible::SlotCheckpoint& point) { return FrontierCopy(r, from, index, point); };
        hooks.resume_host = [this](const crucible::ServeRequest& r, crucible::BatchResident resident) {
          return ResumeHost(r, resident);
        };
      }
    }
    return hooks;
  }
  // The reserve helper pins host memory through the HIP runtime: it ends before the server does.
  ~BatchPrefixCache() { crucible::SnapshotChunkPool::Instance().StopReserve(); }
  bool DiskOn() const { return disk_ != nullptr; }
  bool CheckpointsOn() const { return !buffers_.empty(); }
  // A stopping engine (stdin closed): each slot's Frontier points (`frontiers`: BatchSession::FrontierPoints)
  // and resident tail (the sessions in use now), then the host entries from the most recently used, go to disk, so
  // that the next start of this binary continues these sessions. The whole flush ends within `seconds`; the states
  // not written by then stay off the disk. Runs on the device worker after the last command.
  void FlushToDisk(double seconds, const std::vector<std::tuple<int, int, crucible::SlotCheckpoint>>& frontiers = {}) {
    if (!disk_) return;
    const double start = crucible::GenerationNow();
    const auto left = [&] { return std::max(0.0, seconds - (crucible::GenerationNow() - start)); };
    size_t slots = 0, skipped = 0, points = 0;
    // The Frontier points first: Pi's next turn resumes from them, not from the tail (LOW-4).
    for (const auto& [slot, index, point] : frontiers) {
      if (left() <= 0) break;
      engine_.SelectStream(slot);
      SequenceState& sequence = engine_.ActiveSequence();
      const auto state = crucible::BatchFrontierResume(point.hist.size(), point.frontier.boundary);
      if (!disk_->Skip(point.key, Policy(point.tenant), ConversationStateIO::BytesAt(engine_, sequence, state, point.hist.size(),
                                                                         stk_, carry_)).empty()) {
        ++skipped;
        continue;
      }
      try {
        const auto snapshot = SlotCheckpointIO::SaveFrom(engine_, sequence, Buffer(slot, index), point, state, stk_,
                                                         carry_, Base(slot));
        points += disk_->FlushOne(point.key, Policy(point.tenant), false, snapshot, left());
      } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "state_disk flush slot=%d frontier=1 result=refused reason=memory\n", slot);
      }
    }
    for (int slot = 0; slot < kMaxRows && left() > 0; ++slot) {
      const auto& held = held_.at(size_t(slot));
      if (!held || cache_.HoldsTail(held->key, Policy(held->tenant), held->state.replay)) continue;
      engine_.SelectStream(slot);
      SequenceState& sequence = engine_.ActiveSequence();
      // No device copy for a state whose file is on disk, or that the daily write budget refuses.
      const auto [stk, carry] = Pairs(held->state, slot);
      if (!disk_->Skip(held->key, Policy(held->tenant), ConversationStateIO::Bytes(engine_, sequence, held->state, stk, carry)).empty()) {
        ++skipped;
        continue;
      }
      try {
        const auto snapshot = SaveSlot(sequence, held->state, stk, carry, Base(slot));
        slots += disk_->FlushOne(held->key, Policy(held->tenant), held->guarded, snapshot, left());
      } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "state_disk flush slot=%d result=refused reason=memory\n", slot);
      }
    }
    const size_t host = left() > 0 ? disk_->FlushHost(cache_) : 0;
    const bool drained = disk_->Drain(left());
    const auto stats = disk_->Disk().Stats();
    std::fprintf(stderr, "state_disk flush host=%zu slots=%zu skipped=%zu drained=%d seconds=%.3f written_bytes=%llu "
                 "read_bytes=%llu present=%llu refused_daily=%llu day_written=%llu frontiers=%zu\n", host, slots, skipped,
                 int(drained && disk_->Drain(0)), crucible::GenerationNow() - start,
                 static_cast<unsigned long long>(stats.written_bytes), static_cast<unsigned long long>(stats.read_bytes),
                 static_cast<unsigned long long>(stats.present), static_cast<unsigned long long>(stats.refused_daily),
                 static_cast<unsigned long long>(disk_->Disk().DayWritten()), points);
  }
  // Set once, before the first request: full-window requests key, save and restore MTP history through it.
  void SetMtp(BatchMtpSlots* mtp) { mtp_ = mtp; }

 private:
  // The cache key of `tokens` (a prompt, a prefix or a generated tail of `r`). Text keys equal the token IDs; pad
  // tokens of two images must not match, so image tokens key by image digest.
  static std::vector<int> Key(const crucible::ServeRequest& r, std::span<const int> tokens) {
    return crucible::RequestCacheKey(r, tokens);
  }
  // Idle pool chunks are host memory that a save can take without the system.
  static uint64_t HostAvailable() {
    return crucible::AvailableHostMemory().value_or(0) + crucible::SnapshotChunkPool::Instance().IdleBytes();
  }
  // W1: every cache, slot and disk match takes the policy of one tenant. Each caller names its tenant source:
  // the request (r.tenant), the held tail (held->tenant) or the slot point (point.tenant).
  crucible::ConversationPolicy Policy(const crucible::StateTenant& tenant) const {
    crucible::ConversationPolicy policy;
    policy.prefill_rows = rows_;
    policy.tenant = tenant;
    return policy;
  }
  // A full-window request saves and continues MTP history (batch_mtp_history.h).
  bool MtpHistory(const crucible::ServeRequest& r) const { return mtp_ && crucible::BatchMtpHistory(r); }
  int Slot() const { return engine_.ActiveStream(); }
  // Writes `snapshot` to the active slot. Full MTP history goes to the slot's pair buffers; target state keeps the
  // shared zero pair, so those restores write the same bytes as before.
  void RestoreSnapshot(const std::shared_ptr<const ConversationSnapshot>& restored) {
    WriteSnapshot(*restored);
    bases_.Hold(Slot(), restored);
  }
  // RestoreSnapshot without the base hold (RestoreSnapshot holds it).
  void WriteSnapshot(const ConversationSnapshot& snapshot) {
    const auto held = mtp_ ? crucible::MtpHeldOf(snapshot.resume) : std::nullopt;
    const int slot = Slot();
    if (held) ConversationStateIO::Restore(engine_, engine_.ActiveSequence(), snapshot, mtp_->Stacks(slot), mtp_->Carry(slot));
    else ConversationStateIO::Restore(engine_, engine_.ActiveSequence(), snapshot, stk_, carry_);
    if (mtp_) mtp_->Restored(slot, held);
  }
  SaveBase Base(int slot) const {
    auto base = bases_.Get(slot);
    return {std::move(base.snapshot), base.rows};
  }
  // A save of the active slot that shares the host chunks of `base`, the state that the slot holds. The slot
  // then holds the save. The caller keeps `base` from the admission to the save, so the shared chunks stay.
  std::shared_ptr<const ConversationSnapshot> SaveSlot(SequenceState& sequence, const crucible::ConversationResume& state,
                                                       float* stk, float* carry, const SaveBase& base) {
    auto snapshot = ConversationStateIO::Save(engine_, sequence, state, stk, carry, base);
    bases_.Hold(Slot(), snapshot);
    return snapshot;
  }
  // The admission of that save: its new host memory (its bytes less the chunks that it shares) and those chunks.
  static crucible::SharedSave Share(SequenceState& sequence, const crucible::ConversationResume& state, size_t bytes,
                                    const SaveBase& base) {
    const size_t rows = ConversationStateIO::Settled(state);
    const size_t shared = ConversationStateIO::SharedBytes(sequence, rows, base);
    if (!shared || shared >= bytes) return {};
    return {bytes - shared, ConversationStateIO::SharedOwners(sequence, rows, base)};
  }
  // The bytes that the save shares instead of a copy (the serve log's shared_bytes).
  static size_t SharedOf(const crucible::SharedSave& share, size_t bytes) {
    return share.host_bytes ? bytes - share.host_bytes : 0;
  }
  // The pair buffers that a save of `state` reads.
  std::pair<float*, float*> Pairs(const crucible::ConversationResume& state, int slot) const {
    if (mtp_ && crucible::MtpResumeHeld(state)) return {mtp_->Stacks(slot), mtp_->Carry(slot)};
    return {stk_, carry_};
  }
  crucible::BatchRestore Restore(const crucible::ServeRequest& r) {
    // Every host restore writes the slot, so the slot's MTP history is gone unless the restore brings it back.
    if (mtp_) mtp_->Restored(Slot(), std::nullopt);
    bases_.Clear(Slot());  // the open reset the slot; a restore below sets the base again
    crucible::BatchRestore miss;
    miss.tier = crucible::BatchCacheTier::Host;
    if (r.ids.size() < 2) {
      miss.diag.miss = "too_short";
      return miss;
    }
    const auto key = Key(r, r.ids);
    if (disk_) read_queue_.Opened(key);
    if (r.cache_control != "auto") {
      // The client asked for no cache (`cache_control` bypass or guard); the read-ahead queue still notes the open.
      crucible::BatchRestore skipped = MissedReadAhead(key, 0);
      skipped.tier = crucible::BatchCacheTier::Host;
      skipped.diag.miss = "bypass";
      return skipped;
    }
    std::shared_ptr<const Entry> entry;
    const crucible::BatchRestore restored = Lookup(r, key, entry);
    MissedReadAhead(key, restored.tokens);
    // A replay miss by the prompt's own tokens: counts only, so the line is safe on user traffic.
    if (session_replay_)
      if (const auto missed = cache_.MissedTail(key, Policy(r.tenant), entry.get()))
        std::fputs(crucible::SessionReplayDivergenceLine(r.ids.size(), restored.tokens, *missed).c_str(), stderr);
    return restored;
  }
  // `key`: the prompt's cache key, Key(r, r.ids).
  crucible::BatchRestore Lookup(const crucible::ServeRequest& r, const std::vector<int>& key,
                                std::shared_ptr<const Entry>& entry) {
    // A disk state that a stage step read, when it is longer than the host cache's state. The device copy is the
    // same as a host restore: the bytes are the bytes that Save wrote.
    if (disk_) {
      const auto ready = disk_->Take(Query(r, key), Peek(r));
      if (ready.snapshot) {
        const double start = crucible::GenerationNow();
        RestoreSnapshot(ready.snapshot);
        disk_->Consume(ready);
        const auto& meta = ready.meta;
        std::fprintf(stderr, "state_disk restore tokens=%zu tail=%d bytes=%llu seconds=%.6f tag=%llu request=%llu\n",
                     meta.key.size(), int(!meta.resume.prompt_only), static_cast<unsigned long long>(meta.payload),
                     crucible::GenerationNow() - start, static_cast<unsigned long long>(read_queue_.Restored(meta.key)),
                     static_cast<unsigned long long>(request_));
        counters_.Count(crucible::BatchCacheTier::Disk, meta.key.size());
        crucible::BatchRestore hit = meta.resume.prompt_only
            ? crucible::BatchRestore{meta.key.size(), meta.shared}
            : crucible::BatchRestore{meta.key.size(), false, ready.snapshot->resume.replay, true, false, meta.guarded};
        hit.tier = crucible::BatchCacheTier::Disk;
        return hit;
      }
      counters_.Count(crucible::BatchCacheTier::Disk, 0);  // the disk tier held nothing longer than the host state
    }
    if (session_replay_)
      if (const auto tail = entry = cache_.FindTail(key, Policy(r.tenant))) {
        RestoreSnapshot(tail->snapshot);
        counters_.Count(crucible::BatchCacheTier::Host, tail->snapshot->hist.size());
        std::fprintf(stderr, "cache_lookup tier=host result=hit tokens=%zu tail=1 closest=0 age_s=0.000 request=%llu "
                     "reason=none\n", tail->snapshot->hist.size(), static_cast<unsigned long long>(request_));
        crucible::BatchRestore hit{tail->snapshot->hist.size(), false, tail->snapshot->resume.replay, true, false,
                                   tail->guarded};
        hit.tier = crucible::BatchCacheTier::Host;
        return hit;
      }
    // Stop one row early: the final prompt row must run to produce the first token's logits.
    // A bounded MTP window that restores past its warm-up start warms up from the restored length.
    entry = cache_.Find(key, Policy(r.tenant), size_t(r.cache_prefix_tokens), crucible::BatchRestoreLimit(r), rows_);
    if (entry) {
      const size_t tokens = entry->snapshot->hist.size();
      RestoreSnapshot(entry->snapshot);
      counters_.Count(crucible::BatchCacheTier::Host, tokens);
      std::fprintf(stderr, "cache_lookup tier=host result=hit tokens=%zu tail=0 closest=0 age_s=0.000 request=%llu "
                   "reason=none\n", tokens, static_cast<unsigned long long>(request_));
      crucible::BatchRestore hit{tokens, entry->retention == crucible::CacheRetention::Shared};
      hit.tier = crucible::BatchCacheTier::Host;
      return hit;
    }
    // The host held nothing for this prompt: name why, and count the lookup.
    counters_.Count(crucible::BatchCacheTier::Host, 0);
    const auto diag = cache_.Diagnose(key, Policy(r.tenant), size_t(r.cache_prefix_tokens), crucible::BatchRestoreLimit(r),
                                     rows_, session_replay_);
    std::fprintf(stderr, "cache_lookup tier=host result=miss tokens=0 tail=0 closest=%zu age_s=%.3f request=%llu "
                 "reason=%s\n", diag.shared, diag.age_s, static_cast<unsigned long long>(request_),
                 diag.miss ? diag.miss : "unknown");
    crucible::BatchRestore miss;
    miss.tier = crucible::BatchCacheTier::Host;
    miss.diag = diag;
    if (!miss.diag.miss) miss.diag.miss = "unknown";
    return miss;
  }
  // x4-tail: one slot checkpoint's bytes for the cut buffer when the engine joins (CRUCIBLE_PF_LAUNCH_ROWS "L+T"), else
  // 0: the default takes no byte from the budget.
  static size_t CutBytes(Engine& engine, size_t budget) {
    if (engine.PrefillJoinRows() <= 0) return 0;
    engine.SelectStream(0);
    const size_t bytes = SlotCheckpointIO::Bytes(engine.ActiveSequence());
    if (bytes * 2 > budget) {
      std::fprintf(stderr, "prefill cut off: its buffer needs %zu bytes, more than half of the cache budget\n", bytes);
      return 0;
    }
    return bytes;
  }
  // x4-tail: the copies that put the selected slot's spans at `position` into the
  // cut buffer in Save's layout, and the buffer's metadata as Save at `position` after a launch from `window_start`
  // would set it (the engine checks each source and runs each copy once).
  std::vector<crucible::PrefillCutCopy> CutPrepare(size_t position, size_t window_start) {
    std::vector<crucible::PrefillCutCopy> copies;
    size_t offset = 0;
    for (const auto& span : SlotCheckpointIO::Spans(engine_.ActiveSequence(), position)) {
      if (offset + span.bytes > cut_->bytes) throw std::logic_error("prefill cut buffer is too small");
      copies.push_back({span.pointer, cut_->device + offset, span.bytes});
      offset += span.bytes;
    }
    if (position % 128 || !crucible::RetainedPrefix::FromFields({position, position, window_start, 0}, cut_->retained))
      throw std::logic_error("prefill cut needs a grid point after its window start");
    cut_->position = position;
    return copies;
  }
  // x4-tail: a slot frontier from the cut buffer, the bytes that checkpoint_save would copy at the cut.
  void CutSave(SlotCheckpointIO::Buffer& buffer) {
    size_t used = 0;
    for (const auto& span : SlotCheckpointIO::Spans(engine_.ActiveSequence(), cut_->position)) used += span.bytes;
    if (used > buffer.bytes) throw std::logic_error("slot checkpoint buffer is too small");
    HIP_OK(hipMemcpyAsync(buffer.device, cut_->device, used, hipMemcpyDeviceToDevice, 0));
    buffer.position = cut_->position;
    buffer.retained = cut_->retained;
  }
  // Fix A: the MTP carry of a point's last row sits after the spans (SlotCheckpointIO::Bytes is their largest size).
  static constexpr size_t kCarryBytes = BatchMtpSlots::kCarryFloats * sizeof(float);
  static_assert(kCarryBytes == BatchMtpSlots::kCarryFloats * sizeof(float) && kCarryBytes == size_t(kWide) * sizeof(float),
                "the checkpoint carry tail holds exactly one BatchMtpSlots::Carry row");
  static size_t BufferBytes(Engine& engine) {
    engine.SelectStream(0);
    return SlotCheckpointIO::Bytes(engine.ActiveSequence()) + kCarryBytes;
  }
  // Fix A: an End point's logits row is host memory, at most one per general buffer, plus a final chunk's row.
  static size_t LogitsBytes(int streams) {
    return crucible::EndLogitsBytes(streams, size_t(kVocab));
  }
  void CopyCarry(SlotCheckpointIO::Buffer& buffer, int slot, bool save) {
    // The buffers of fix A hold the carry in their tail (Buffer::tail == kCarryBytes): Save leaves it alone.
    uint8_t* const kept = buffer.device + buffer.bytes - kCarryBytes;
    float* const carry = mtp_->Carry(slot);
    HIP_OK(hipMemcpyAsync(save ? static_cast<void*>(kept) : static_cast<void*>(carry),
                          save ? static_cast<const void*>(carry) : static_cast<const void*>(kept), kCarryBytes,
                          hipMemcpyDeviceToDevice, 0));
  }
  static size_t CheckpointBytes(Engine& engine, int streams, size_t budget) {
    const size_t total = BufferBytes(engine) * size_t(streams) * crucible::kSlotCheckpoints + LogitsBytes(streams);
    if (total * 2 > budget) {
      std::fprintf(stderr, "slot checkpoints off: they need %zu bytes, more than half of the cache budget\n", total);
      return 0;
    }
    return total;
  }
  SlotCheckpointIO::Buffer& Buffer(int slot, int index) {
    return *buffers_.at(size_t(slot) * crucible::kSlotCheckpoints + size_t(index));
  }
  // The length that Restore would give this request, with no change to the cache (as Lookup).
  size_t Peek(const crucible::ServeRequest& r) const {
    if (r.cache_control != "auto" || r.ids.size() < 2) return 0;
    const auto key = Key(r, r.ids);
    if (session_replay_)
      if (const size_t tail = cache_.PeekTail(key, Policy(r.tenant))) return tail;
    return cache_.PeekFind(key, Policy(r.tenant), size_t(r.cache_prefix_tokens), crucible::BatchRestoreLimit(r), rows_);
  }
  // The disk tier. The identity of this process (binary, checkpoint, settings) names the directory, so another
  // build never reads these states. Spills only queue; the writer and reader threads do the disk work.
  void OpenDisk(crucible::StateDiskOptions options, uint64_t host_reserve) {
    engine_.SelectStream(0);
    // Spills wait in host memory below the host reserve: at most half of it.
    options.queue_bytes = std::min<uint64_t>(options.queue_bytes, host_reserve / 2);
    // The block plan's geometry: the snapshot bytes around the QSA layers.
    options.block_head_bytes = ConversationStateIO::HeadBytes(engine_.ActiveSequence());
    options.block_tail_bytes = ConversationStateIO::kTailBytes;
    auto disk = std::make_unique<crucible::BatchDiskTier<ConversationSnapshot>>(std::move(options),
                                                                                engine_.ActiveSequence().compact_state);
    if (!disk->Enabled()) return;
    disk_ = std::move(disk);
    // Staged reads count in the cache budget: every save makes room for them too.
    cache_.SetOutside([this] { return size_t(disk_->Disk().StagedBytes()); });
    cache_.SetSpill([this](std::span<const int> key, const crucible::ConversationPolicy& policy,
                           const std::shared_ptr<const Entry>& entry, int, uint64_t room) {
      return disk_->Spill(key, policy, entry, room, cache_.Reading());
    });
  }
  crucible::DiskQuery Query(const crucible::ServeRequest& r, std::span<const int> key) const {
    return {key, Policy(r.tenant), size_t(r.cache_prefix_tokens), crucible::BatchRestoreLimit(r), rows_, session_replay_};
  }
  size_t DiskPeek(const crucible::ServeRequest& r) {
    if (!disk_ || r.cache_control != "auto" || r.ids.size() < 2) return 0;
    const auto key = Key(r, r.ids);
    return disk_->Peek(Query(r, key));
  }
  // An earlier change: a read state is host memory inside the cache budget. Before a new read the host cache evicts for it
  // as for a client prompt's save; a live resume point leaves only when its session's turn waits behind this turn and
  // holds no slot (BatchReadQueue::Behind). Victims go to disk. Then the read room (StageRoom) keeps MemAvailable at
  // kStageKeepFree or more. A turn behind the first (`queue.read` false) only says that it waits. Host work only: no
  // device copy, no slot change.
  crucible::BatchStage Stage(const crucible::ServeRequest& r, size_t floor, const crucible::BatchStageQueue& queue) {
    const auto key = Key(r, r.ids);
    const double now = crucible::GenerationNow();
    read_queue_.Saw(key, queue.position, now);
    const auto room_now = [&] {
      return crucible::StageRoom(disk_->Disk().StagedBytes(), HostAvailable(), cache_.HostReserve(),
                                 disk_->Disk().UnallocatedReadBytes());
    };
    uint64_t room = room_now();
    const size_t below = std::max(floor, Peek(r));
    const auto status = disk_->StageWith(Query(r, key), below, [&](const crucible::StateMeta& meta) {
      if (!queue.read) return crucible::StageAdmit{0, "queued"};
      const auto behind = [&](std::span<const int> point) {
        return !(queue.in_slot && queue.in_slot(point)) && read_queue_.Behind(point, key, queue.position, now);
      };
      // Fix 2 (cause.md items 7 and 8): a point leaves only when its state is on disk or queued (Took gives its tag);
      // a refused spill keeps the point in the cache (kept=1, no tag) and names the disk's reason.
      const auto taken = [&](std::span<const int> point, size_t bytes, bool spilled) {
        // cand-fri merge of W1 and fix 2: the evicted point's tenant is not in this callback, so the check uses the
        // local tenant (every state today). For another tenant's point the answer can only keep or drop a host copy;
        // a restore still matches the whole policy, so no state crosses tenants. Follow-up: pass the victim's policy.
        const bool out = spilled || disk_->Holds(point, Policy(crucible::kLocalTenant));
        const char* disk = spilled ? "queued" : out ? "present" : "refused";
        const uint64_t tag = out ? read_queue_.Took(point) : 0;
        std::fprintf(stderr, "state_disk host_evict tokens=%zu bytes=%zu reason=read_ahead disk=%s spill=%s kept=%d "
                     "tag=%llu position=%zu job=%llu\n", point.size(), bytes, disk,
                     crucible::StageReason(spilled ? "" : disk_->LastSpill()).c_str(), int(!out),
                     static_cast<unsigned long long>(tag), queue.position, static_cast<unsigned long long>(queue.job));
        return out;
      };
      // The eviction plans with the bytes that the disk writer will free (so a poll during the wait evicts no more)
      // and keeps kStageKeepFree; the guard then uses only memory that is free now (ReadRoom).
      auto& disk = disk_->Disk();
      if (!cache_.MakeRoom(meta.payload, HostAvailable() + disk.QueuedBytes(), behind, taken, crucible::kStageKeepFree)) {
        const std::string_view refused = cache_.LastRefusal();
        if (refused == "spill") return crucible::StageAdmit{0, SpillRefusal(disk_->LastSpill())};
        return crucible::StageAdmit{0, refused == "host" ? "room" : cache_.LastRefusal()};
      }
      switch (crucible::ReadRoom(disk.StagedBytes(), HostAvailable(), cache_.HostReserve(), disk.UnallocatedReadBytes(),
                                 disk.QueuedBytes(), meta.payload)) {
        case crucible::ReadRoomStep::Start: return crucible::StageAdmit{room = room_now()};
        case crucible::ReadRoomStep::Wait: return crucible::StageAdmit{room = room_now(), nullptr, true};
        case crucible::ReadRoomStep::Refuse: break;
      }
      return crucible::StageAdmit{room = room_now(), "room"};
    });
    std::fputs(crucible::StageLine(status, r.ids.size(), below, room, queue.job).c_str(), stderr);
    // `tag`: a read took this turn's host point earlier (host_evict tag=); the turn line carries it (state_disk_tag).
    return {crucible::StageName(status.state), status.state == crucible::StageState::None ? 0 : status.tokens,
            crucible::StageReason(status.reason), read_queue_.TagOf(key)};
  }
  // The stage reason when the disk refused a read-ahead point (MakeRoom "spill"): a non-fault budget word of the
  // checker where one fits.
  static const char* SpillRefusal(std::string_view spill) {
    if (spill == "memory") return "room";
    if (spill == "busy") return "busy";
    if (spill == "daily") return "daily";
    return "spill";
  }
  // An open of `key` that did not restore its read-ahead point from disk (`restored`: what it restored). Counts only.
  crucible::BatchRestore MissedReadAhead(std::span<const int> key, size_t restored) {
    if (!disk_) return 0;
    if (const auto [tag, tokens] = read_queue_.Missed(key); tag)
      std::fprintf(stderr, "state_disk read_ahead_miss tag=%llu tokens=%zu restored=%zu\n",
                   static_cast<unsigned long long>(tag), tokens, restored);
    return 0;
  }
  // The slot still holds the tail `resident`, which this prompt repeats. The host cache restores only a longer tail
  // (or the same tokens with other pending rows); otherwise the slot state is the state a restore would write.
  crucible::BatchRestore Resume(const crucible::ServeRequest& r, crucible::BatchResident resident) {
    const auto key = Key(r, r.ids);
    if (const auto tail = cache_.FindTail(key, Policy(r.tenant))) {
      const auto& snapshot = *tail->snapshot;
      if (crucible::HostTailFirst(snapshot.hist.size(), snapshot.resume.replay, resident)) {
        RestoreSnapshot(tail->snapshot);
        return {snapshot.hist.size(), false, snapshot.resume.replay, true, false, tail->guarded};
      }
    } else {
      cache_.Touch(key, Policy(r.tenant));  // no host tail was hit: keep the session live
    }
    // The slot keeps its tail and the MTP history that Settle left with it.
    std::fprintf(stderr, "slot_resume tokens=%zu replay=%d\n", resident.length, resident.replay);
    return {resident.length, false, resident.replay, true, true, resident.guarded};
  }
  // The next request on `slot` overwrites its tail ("One copy"). The host saves that
  // state now unless it holds it already. The stream is unchanged since Tail truncated and kept it, so these are the
  // bytes that Tail would have saved. A request that continues from the tail repeats it: the save is trusted.
  void Release(int slot, bool resumed) {
    cache_.ClearResident(slot);
    auto held = std::move(held_.at(size_t(slot)));
    held_.at(size_t(slot)).reset();
    if (!held || cache_.HoldsTail(held->key, Policy(held->tenant), held->state.replay)) return;
    SequenceState& sequence = engine_.ActiveSequence();
    const double start = crucible::GenerationNow();
    const auto [stk, carry] = Pairs(held->state, slot);
    const size_t bytes = ConversationStateIO::Bytes(engine_, sequence, held->state, stk, carry) +
                         crucible::kConversationPolicyBytes;
    const size_t evictions = cache_.Evictions();
    const auto base = Base(slot);
    const auto share = Share(sequence, held->state, bytes, base);
    const bool saved = cache_.CaptureTail(held->key, Policy(held->tenant), bytes, HostAvailable(),
                                          [&] { return SaveSlot(sequence, held->state, stk, carry, base); },
                                          held->trusted || resumed, held->guarded, false, held->hist, share);
    if (!saved) cache_.NoteRefused(held->key, Policy(held->tenant), held->state.length);  // the tail's prompt is its turn's start
    const auto status = cache_.Status();
    std::fprintf(stderr, "session_replay spill slot=%d tokens=%zu replay=%d resumed=%d saved=%d refused=%s evicted=%zu "
                 "bytes=%zu cache_bytes=%zu copy_bytes=%zu shared_bytes=%zu seconds=%.6f\n", slot, held->state.length,
                 held->state.replay, int(resumed), int(saved), saved ? "-" : status.last.refused.c_str(),
                 cache_.Evictions() - evictions, bytes, cache_.Bytes(), cache_.CopyBytes(), SharedOf(share, bytes),
                 crucible::GenerationNow() - start);
    if (!saved)
      Direct(slot, held->key, Policy(held->tenant), held->state.length, held->guarded, bytes, share,
             [&] { return SaveSlot(sequence, held->state, stk, carry, base); });
  }
  // MEDIUM-5 (BatchDiskTier::SpillDirect): the host cache refused this slot state, so it goes straight to
  // disk when the disk tier is on and host memory above the reserve holds its new bytes. Else it is lost, as before.
  template<class Save>
  void Direct(int slot, std::span<const int> key, const crucible::ConversationPolicy& policy, size_t tokens,
              bool guarded, size_t bytes, const crucible::SharedSave& share, Save save) {
    if (!disk_) return;
    const uint64_t available = HostAvailable(), reserve = cache_.HostReserve();
    const size_t need = share.host_bytes ? share.host_bytes : bytes;
    const std::string refused = cache_.Status().last.refused;
    const auto result = disk_->SpillDirect(key, policy, guarded, refused, bytes, need,
                                           available > reserve ? available - reserve : 0, save);
    if (result.empty()) return;
    // Numbers and fixed words only (the key and the tenant never go to a log line).
    std::fprintf(stderr, "state_disk direct slot=%d tokens=%zu bytes=%zu need=%zu result=%s reason=%s\n", slot,
                 tokens, bytes, need, result == "queued" ? "queued" : "refused", result == "queued" ? "-" : result.c_str());
  }
  // the session frontier at `cursor` stays in the slot. Shared frontiers stay
  // host copies (other sessions restore them). A full-window request that warmed its MTP rows keeps its host copy
  // with them (a checkpoint holds target state only; LOW-1).
  bool KeepFrontier(const crucible::ServeRequest& r, size_t cursor,
                    const std::function<bool(const crucible::SlotFrontier&)>& keep) {
    if (r.cache_control == "bypass") return false;
    for (const auto& frontier : crucible::BatchCacheFrontiers(r.ids.size(), size_t(r.cache_prefix_tokens), rows_,
                                                              size_t(r.shared_prefix_tokens))) {
      if (frontier.position != cursor) continue;
      // HIGH-1: off the 128-row grid (--prefill 0, or rows that are not a multiple of 128) the host
      // copy could not be made later, so the host copies it now.
      if (frontier.retention != crucible::CacheRetention::Session || !crucible::SlotFrontierGrid(cursor, rows_))
        return false;
      if (MtpHistory(r) && mtp_->Live(Slot(), cursor)) return false;
      const auto key = Key(r, r.ids);
      const std::span<const int> below(key.data(), cursor);
      // Asked before the marks below, as Capture asks it before Admit.
      const crucible::SlotFrontier meta{size_t(r.cache_prefix_tokens), frontier.speculative,
                                        cache_.ProvenBelow(below, Policy(r.tenant))};
      if (!keep(meta)) return false;
      cache_.KeepInSlot(below, Policy(r.tenant), frontier.speculative);
      std::fprintf(stderr, "slot_frontier keep slot=%d tokens=%zu speculative=%d\n", Slot(), cursor,
                   int(frontier.speculative));
      return true;
    }
    return false;
  }
  // MEDIUM-2: Find's rule for a host frontier of the point's length (Lookup), so the slot and the host restore the
  // same length for one prompt.
  bool FrontierFits(const crucible::ServeRequest& r, const crucible::SlotCheckpoint& point) const {
    if (r.cache_control != "auto" || r.ids.size() < 2) return false;
    return crucible::SlotFrontierFits(point.hist.size(), point.frontier.boundary, r.ids.size(),
                                      size_t(r.cache_prefix_tokens), crucible::BatchRestoreLimit(r), rows_);
  }
  // Whether Resume would restore a host tail instead of the slot's tail (FindTail and HostTailFirst, with no change
  // to the cache): then the open drops the slot's points, and the Frontier points go to the host first.
  bool ResumeHost(const crucible::ServeRequest& r, crucible::BatchResident resident) const {
    const auto [length, replay] = cache_.PeekTailState(Key(r, r.ids), Policy(r.tenant));
    return length && crucible::HostTailFirst(length, replay, resident);
  }
  // HIGH-1: an open writes over the slot's Frontier point (another session, a reset, a host restore): its host copy,
  // as a Capture at its position would save it, with its speculative and proven flags (MEDIUM-3). The rows below
  // the point are the slot's own and the fixed state is in the checkpoint buffer (SlotCheckpointIO::SaveFrom). A
  // refused copy loses the state (speed only); the open goes on.
  void Handover(int slot, int index, const crucible::SlotCheckpoint& point) {
    SequenceState& sequence = engine_.ActiveSequence();
    const double start = crucible::GenerationNow();
    const size_t length = point.hist.size();
    const auto state = crucible::BatchFrontierResume(length, point.frontier.boundary);
    const size_t bytes = ConversationStateIO::BytesAt(engine_, sequence, state, length, stk_, carry_) +
                         crucible::kConversationPolicyBytes;
    const size_t evictions = cache_.Evictions();
    const auto base = Base(slot);
    const auto share = Share(sequence, state, bytes, base);
    const crucible::CacheFrontier frontier{length, crucible::CacheRetention::Session, point.frontier.speculative};
    const auto save = [&] {
      auto snapshot = SlotCheckpointIO::SaveFrom(engine_, sequence, Buffer(slot, index), point, state, stk_, carry_, base);
      // The slot holds the snapshot's rows. A base with more rows stays: it shares more with the tail save next.
      if (!base.snapshot || base.rows <= length) bases_.Hold(slot, snapshot);
      return snapshot;
    };
    const bool saved = cache_.Capture(point.key, Policy(point.tenant), frontier, bytes, HostAvailable(), save, point.hist,
                                      share, point.frontier.proven);
    if (!saved) cache_.NoteRefused(point.key, Policy(point.tenant), point.hist.size());
    // Numeric fields only (a test script sums them). copy_bytes: the bytes that the device copied (not the cache's copy class).
    // `request`: the open that wrote over the point (fix 2, item 10).
    std::fprintf(stderr, "slot_frontier_handover slot=%d tokens=%zu speculative=%d saved=%d evicted=%zu bytes=%zu "
                 "copy_bytes=%zu shared_bytes=%zu cache_bytes=%zu seconds=%.6f request=%llu\n", slot, length,
                 int(point.frontier.speculative), int(saved), cache_.Evictions() - evictions, bytes,
                 saved ? bytes - SharedOf(share, bytes) : 0, saved ? SharedOf(share, bytes) : 0, cache_.Bytes(),
                 crucible::GenerationNow() - start, static_cast<unsigned long long>(request_));
    if (!saved) Direct(slot, point.key, Policy(point.tenant), length, false, bytes, share, save);
  }
  // frontier-copy: the open on the selected (target) slot missed the host, and
  // another slot holds Frontier point (from, index) for its prefix. SlotCheckpointIO::CopyTo writes the target the
  // device state that a host snapshot of that point (as Handover saves it) and its restore write, slot to slot (fc-d2d):
  // the host keeps no bytes, the copy needs no cache room and evicts nothing, and the target holds no base. A refusal
  // (0) leaves the open to the host restore.
  crucible::BatchRestore FrontierCopy(const crucible::ServeRequest& r, int from, int index,
                                      const crucible::SlotCheckpoint& point) {
    const int target = Slot();
    const double start = crucible::GenerationNow();
    // W1: only a point of the request's own tenant is copied (BatchSession::Donor picks no other). Checked
    // before any write; the open then takes the host path.
    if (point.tenant != r.tenant) return 0;
    // As a slot Frontier load (checkpoint_load): a host frontier hit is a whole pool block and prefill chunk with no
    // pending rows (design section 2). Checked before any write.
    if (!crucible::SlotFrontierGrid(point.hist.size(), rows_, point.replay))
      throw std::logic_error("slot frontier resume needs a grid point with no pending rows");
    // As Restore: every host restore writes the slot, so its MTP history and base are gone.
    if (mtp_) mtp_->Restored(target, std::nullopt);
    bases_.Clear(target);
    // The session selects the target again on every exit; this guard covers the hook alone.
    const struct Reselect { Engine& engine; int slot; ~Reselect() { engine.SelectStream(slot); } } reselect{engine_, target};
    engine_.SelectStream(from);
    SequenceState& donor = engine_.ActiveSequence();
    engine_.SelectStream(target);
    SequenceState& sequence = engine_.ActiveSequence();
    const size_t length = point.hist.size();
    const auto state = crucible::BatchFrontierResume(length, point.frontier.boundary);
    const size_t bytes = ConversationStateIO::BytesAt(engine_, donor, state, length, stk_, carry_) +
                         crucible::kConversationPolicyBytes;
    const size_t evictions = cache_.Evictions();
    const char* reason = nullptr;
    const double copy = crucible::GenerationNow();
    // Only an allocation may fall back: CopyTo makes its host lists before its first write to the target. Any other
    // failure poisons the worker, as a failed host restore does.
    try {
      SlotCheckpointIO::CopyTo(engine_, donor, Buffer(from, index), point, state, stk_, carry_, sequence);
    } catch (const std::bad_alloc&) {
      reason = "memory";
    }
    const double copied = reason ? 0 : crucible::GenerationNow() - copy;
    // A Frontier hit keeps the session live, as a slot Frontier load does.
    if (!reason) cache_.Touch(point.key, Policy(point.tenant));
    // Fixed identifiers and numbers only; the fields of the host copy, so the parsers do not change. copy_bytes: the
    // snapshot bytes that the device copied slot to slot (shared_bytes 0: no host chunks).
    std::fprintf(stderr, "slot_frontier_copy slot=%d from=%d tokens=%zu speculative=%d result=%s reason=%s evicted=%zu "
                 "bytes=%zu copy_bytes=%zu shared_bytes=%zu cache_bytes=%zu pinned_bytes=%zu copy_seconds=%.6f "
                 "seconds=%.6f\n", target, from, length, int(point.frontier.speculative), reason ? "refused" : "hit",
                 reason ? reason : "none", cache_.Evictions() - evictions, bytes, reason ? 0 : bytes, size_t(0),
                 cache_.Bytes(), crucible::SnapshotChunkPool::Instance().PinnedBytes(), copied,
                 crucible::GenerationNow() - start);
    if (reason) return 0;
    return length;
  }
  // HIGH-2: the open resumes from the slot's own Frontier point. The slot's tail is the output that the prompt left
  // (Pi drops the reasoning), and the open writes over it: it leaves without a host copy.
  void Drop(int slot) {
    cache_.ClearResident(slot);
    auto held = std::move(held_.at(size_t(slot)));
    held_.at(size_t(slot)).reset();
    if (held) std::fprintf(stderr, "slot_frontier drop slot=%d tokens=%zu\n", slot, held->state.length);
  }
  // The first frontier after `cursor` that Capture may save (merged bulk launches end there).
  size_t NextCapture(const crucible::ServeRequest& r, size_t cursor) const {
    size_t next = SIZE_MAX;
    if (r.cache_control == "bypass") return next;
    for (const auto& frontier : crucible::BatchCacheFrontiers(r.ids.size(), size_t(r.cache_prefix_tokens), rows_,
                                                              size_t(r.shared_prefix_tokens)))
      if (frontier.position > cursor) next = std::min(next, frontier.position);
    return next;
  }
  // x4-tail: `from` (the cut buffer) holds the fixed state at `cursor` while the slot holds later rows.
  void Capture(const crucible::ServeRequest& r, size_t cursor, const SlotCheckpointIO::Buffer* from = nullptr) {
    if (r.cache_control == "bypass") return;
    for (const auto& frontier : crucible::BatchCacheFrontiers(r.ids.size(), size_t(r.cache_prefix_tokens), rows_,
                                                              size_t(r.shared_prefix_tokens))) {
      if (frontier.position != cursor) continue;
      auto state = crucible::BatchFrontierResume(cursor, size_t(r.cache_prefix_tokens));
      // A full-window request saves its MTP rows when it warmed the whole prefix (else target state, same key).
      const int slot = Slot();
      if (const auto held = MtpHistory(r) ? mtp_->Live(slot, cursor) : std::nullopt) {
        state.mpos = held->Mpos();
        state.have_carry = true;
      }
      const auto [stk, carry] = Pairs(state, slot);
      SequenceState& sequence = engine_.ActiveSequence();
      const size_t bytes = (from ? ConversationStateIO::BytesAt(engine_, sequence, state, cursor, stk, carry)
                                 : ConversationStateIO::Bytes(engine_, sequence, state, stk, carry)) +
                           crucible::kConversationPolicyBytes;
      const size_t evictions = cache_.Evictions(), superseded = cache_.SupersededEvictions();
      const auto base = Base(slot);
      const auto save = [&] {
        if (!from) return SaveSlot(sequence, state, stk, carry, base);
        // Handover's rule: the slot holds the snapshot's rows; a base with more rows stays.
        crucible::SlotCheckpoint point;
        point.hist.assign(r.ids.begin(), r.ids.begin() + std::ptrdiff_t(cursor));
        auto snapshot = SlotCheckpointIO::SaveFrom(engine_, sequence, *from, point, state, stk, carry, base);
        if (!base.snapshot || base.rows <= cursor) bases_.Hold(slot, snapshot);
        return snapshot;
      };
      const bool saved = cache_.Capture(Key(r, r.ids), Policy(r.tenant), frontier, bytes, HostAvailable(), save,
                                        r.ids, Share(sequence, state, bytes, base));
      if (!saved) cache_.NoteRefused(std::span<const int>(r.ids.data(), frontier.position), Policy(r.tenant));
      // Eviction trace for the retention policy check.
      std::fprintf(stderr, "cache_capture tokens=%zu class=%s saved=%d evicted=%zu superseded_evicted=%zu cache_bytes=%zu\n",
                   cursor, frontier.retention == crucible::CacheRetention::Shared ? "shared" : "session", int(saved),
                   cache_.Evictions() - evictions, cache_.SupersededEvictions() - superseded, cache_.Bytes());
    }
  }
  // Session replay (G15): save the finished request's state through its last published token but one. The rule is
  // ConversationTailState, shared with serial serve: a window cut by a stop or limit is truncated, not skipped.
  // A tail of a replayed turn is trusted: it replaces the older states of its branch. With a guard (reasoning, or a
  // continuation job) it is trusted only when the replayed tail was guarded too, which shows that this client sends
  // the reasoning back. Any other tail is speculative until a later prompt repeats it. Returns the tail state that
  // the slot keeps for the next turn, saved or not.
  std::optional<crucible::BatchResident> Tail(const crucible::ServeRequest& r, std::span<const int> generated,
                                              int pending, crucible::BatchTailSource source) {
    const auto kept = crucible::ConversationTailState(engine_.Length(), r.ids.size(), generated.size(), pending);
    if (!kept) {
      std::fprintf(stderr, "session_replay skip length=%zu generated=%zu replay=%d\n", size_t(engine_.Length()),
                   generated.size(), pending);
      return std::nullopt;
    }
    const size_t logical = kept->length;
    const int replay = kept->replay;
    if (engine_.Length() > logical) engine_.Truncate(logical);
    SequenceState& sequence = engine_.ActiveSequence();
    std::vector<int> hist(r.ids);
    hist.insert(hist.end(), generated.begin(), generated.end() - 1);
    crucible::ConversationResume state;
    state.length = logical;
    state.replay = replay;
    state.next_id = generated.back();
    const int slot = Slot();
    const double start = crucible::GenerationNow();
    // A full-window tail keeps the MTP history up to its last row (the slot keeps it too, for slot resume).
    if (const auto held = MtpHistory(r) ? mtp_->Settle(slot, logical) : std::nullopt) {
      state.mpos = held->Mpos();
      state.have_carry = true;
    } else if (mtp_ && !MtpHistory(r)) {
      mtp_->Restored(slot, std::nullopt);  // the slot's MTP rows no longer match its target rows
    }
    const auto [stk, carry] = Pairs(state, slot);
    const size_t bytes = ConversationStateIO::Bytes(engine_, sequence, state, stk, carry) +
                         crucible::kConversationPolicyBytes;
    const bool guarded = r.cache_tail == "guard";
    const bool trusted = source.replayed && (!guarded || source.guarded);
    // With slot resume the slot keeps this state, so the host copy takes only free room (Release saves it later).
    const bool resident = slot_resume_ && source.slot >= 0;
    const size_t evictions = cache_.Evictions();
    auto key = Key(r, hist);
    const auto base = Base(slot);
    const auto share = Share(sequence, state, bytes, base);
    const bool saved = cache_.CaptureTail(key, Policy(r.tenant), bytes, HostAvailable(),
                                          [&] { return SaveSlot(sequence, state, stk, carry, base); },
                                          trusted, guarded, resident, hist, share, r.ids.size());
    if (!saved) cache_.NoteRefused(key, Policy(r.tenant), r.ids.size());
    if (resident) {
      cache_.SetResident(source.slot, key, replay, Policy(r.tenant));
      held_.at(size_t(source.slot)) = Held{std::move(hist), std::move(key), state, trusted, guarded, r.tenant};
    }
    std::fprintf(stderr, "session_replay tail tokens=%zu replay=%d saved=%d trusted=%d evicted=%zu bytes=%zu "
                 "cache_bytes=%zu copy_bytes=%zu shared_bytes=%zu seconds=%.6f\n", logical, replay, int(saved),
                 int(trusted), cache_.Evictions() - evictions, bytes, cache_.Bytes(), cache_.CopyBytes(),
                 SharedOf(share, bytes), crucible::GenerationNow() - start);
    // A resident copy stays in its slot: Release saves it, or sends it to disk, when the slot needs the room.
    if (!saved && !resident)
      Direct(slot, key, Policy(r.tenant), logical, guarded, bytes, share,
             [&] { return SaveSlot(sequence, state, stk, carry, base); });
    return crucible::BatchResident{logical, replay, guarded};
  }
  // a tail save that pinned new chunks on the device worker stopped every decoder (0.19 s at 32K, 0.65 s at
  // 128K). The pool's helper pins the tails of the decoding slots while they decode, so the save takes idle chunks.
  // It stays inside the pinned limit (the cache budget) and leaves the host reserve free.
  void Expect(const crucible::ServeRequest& r, int slot, size_t rows) {
    // less the chunks that the save shares with the state that the slot holds now.
    const size_t whole = rows ? ConversationStateIO::SaveBytes(engine_.ActiveSequence(), rows, MtpHistory(r)) : 0;
    const size_t shared = rows ? ConversationStateIO::SharedBytes(engine_.ActiveSequence(), rows, Base(slot)) : 0;
    expected_.at(size_t(slot)) = whole > shared ? whole - shared : 0;
    const uint64_t floor = cache_.HostReserve() + crucible::SnapshotChunkPool::kChunkBytes;
    crucible::SnapshotChunkPool::Instance().Reserve(std::accumulate(expected_.begin(), expected_.end(), size_t(0)),
                                                    [floor] { return crucible::AvailableHostMemory().value_or(0) >= floor; });
  }
  // The tail that a slot keeps after its request (slot resume), with what Release needs to save it.
  // `hist`: the tail's tokens; `key`: its cache key; `tenant`: its request's tenant, the policy of its saves.
  struct Held {
    std::vector<int> hist, key;
    crucible::ConversationResume state;
    bool trusted = false, guarded = false;
    crucible::StateTenant tenant = crucible::kLocalTenant;
  };
  Engine& engine_;
  BatchMtpSlots* mtp_ = nullptr;
  const bool session_replay_, slot_resume_;
  const size_t checkpoint_bytes_;
  const size_t cut_bytes_;  // x4-tail: the cut buffer, 0 without joins
  std::vector<std::unique_ptr<SlotCheckpointIO::Buffer>> buffers_;
  std::unique_ptr<SlotCheckpointIO::Buffer> cut_;
  std::array<std::optional<Held>, kMaxRows> held_;
  std::array<size_t, kMaxRows> expected_{};  // per slot: the bytes of the tail that its request may save (Expect)
  crucible::SlotBases<ConversationSnapshot, kMaxRows> bases_;  // an earlier change
  crucible::BatchConversationCache<ConversationSnapshot> cache_;
  // Declared after cache_: it is destroyed first, so no writer job outlives the entries that it holds.
  std::unique_ptr<crucible::BatchDiskTier<ConversationSnapshot>> disk_;
  crucible::BatchReadQueue read_queue_;  // the waiting turns and the points that reads took
  uint64_t request_ = 0;  // the request whose open runs now (the log tag of restore and handover lines)
  const int rows_;
  float* const stk_;
  float* const carry_;
  // Per-tier counters: Lookup counts the host and disk tiers, BatchSession the slot tier (slot_served).
  crucible::BatchCacheCounters counters_;
};

// Solo speculative steps for batch serving (G6a). Prompt pairing mirrors RunPrompt and
// each step mirrors one greedy RunGen iteration with overlap and head_auto off. Draft state is per request.
class BatchSpecDecoder : public BatchMtpSlots {
 public:
  // blocks: the serve --draft-blocks selection; empty keeps the built-in 64K selection.
  explicit BatchSpecDecoder(Engine& engine, const std::vector<int>& blocks = {}) : engine_(engine) {
    for (int slot = 0; slot < kMaxRows; ++slot) {
      stk_[slot] = Zeros<float>(size_t(kMaxRows + 1) * kWide);
      carry_[slot] = Zeros<float>(kCarryFloats);
    }
    // One head for every request in last-row mode (draft_q > 0), never adapted. It changes only the drafts.
    if (blocks.empty()) {
      engine_.DraftHead({kDefaultDraftBlocks.begin(), kDefaultDraftBlocks.end()}, true, false, false);
    } else {
      engine_.DraftHead(blocks, true, false, false);
      std::fprintf(stderr, "%s\n", ServeDraftHeadLine((int)blocks.size(), (int)kDefaultDraftBlocks.size(),
                                                      engine_.DraftTokenBytes()).c_str());
    }
  }
  crucible::BatchSpecHooks Hooks() {
    crucible::BatchSpecHooks hooks;
    hooks.check = [](const crucible::ServeRequest& r) { return Check(r); };
    hooks.open = [this](uint64_t id, int slot, const crucible::ServeRequest& r, size_t cached) { Open(id, slot, r, cached); };
    hooks.chunk = [this](uint64_t id, size_t start, int rows, bool bulk) { Chunk(states_.at(id), start, rows, bulk); };
    hooks.stacks = [this](uint64_t id, size_t start, int rows) {
      return DraftWarmupRow(start, rows, states_.at(id).mtp_start);
    };
    hooks.first = [this](uint64_t id, int token) { First(states_.at(id), token); };
    hooks.step = [this](uint64_t id, int last, size_t remaining, uint64_t position) {
      return Solo(states_.at(id), last, remaining, position);
    };
    hooks.replay = [this](uint64_t id) { return states_.at(id).replay; };
    // The done event's draft counters (crucible_turn draft_proposed, draft_accepted, spec_rounds).
    hooks.stats = [this](uint64_t id) {
      const State& s = states_.at(id);
      return crucible::BatchDraftStats{s.proposed, s.accepted, s.steps, s.tally};
    };
    // The same counters after each speculative step (crucible_decode); no tally copy.
    hooks.counts = [this](uint64_t id) {
      const State& s = states_.at(id);
      return crucible::BatchDraftCounts{s.proposed, s.accepted, s.steps};
    };
    hooks.close = [this](uint64_t id) {
      const State& s = states_.at(id);
      // The tail save (Settle) runs after the close and reads these pending pairs.
      closed_.at(size_t(s.slot)) = Closed{s.mpos, s.ctoks, s.Full()};
      // Opt-in draft statistics (G15 A/B: a tail restore starts the MTP warm-up at the restored length).
      if (crucible::SwitchText("CRUCIBLE_SPEC_STATS"))
        std::fprintf(stderr, "spec_stats id=%llu cached=%zu proposed=%ld accepted=%ld steps=%ld lookup_steps=%ld "
                     "lookup_proposed=%ld lookup_accepted=%ld mtp_origin=%zu mtp_continued=%d mtp_restarts=%d\n",
                     static_cast<unsigned long long>(id), s.cached, s.proposed, s.accepted, s.steps, s.lookup_steps,
                     s.lookup_proposed, s.lookup_accepted, s.origin, int(s.continued), s.mtp_restarts);
      // Draft catch-up counters: drafts after packed rows show whether the flushed MTP state still drafts well.
      if (s.packed_rows)
        std::fprintf(stderr, "spec_catchup id=%llu packed_rows=%d flushes=%d resumed_steps=%d resumed_proposed=%d "
                     "resumed_accepted=%d\n", static_cast<unsigned long long>(id), s.packed_rows, s.flushes,
                     s.resumed_steps, s.resumed_proposed, s.resumed_accepted);
      states_.erase(id);
    };
    hooks.commit = [this](uint64_t id, int last) { return Commit(states_.at(id), last); };
    // Grouped MTP passes for greedy requests in packed verification.
    hooks.propose_all = [this](const std::vector<uint64_t>& ids, const std::vector<int>& last,
                               const std::vector<size_t>& remaining, const std::vector<uint64_t>& position,
                               const std::vector<int>& cap) { return ProposeAll(ids, last, remaining, position, cap); };
    hooks.want = [this](uint64_t id, int last) {
      const State& s = states_.at(id);
      if (s.mtp_off || s.history.Empty() || s.history.Back() != last) return 0;
      return int(Lookup(s, std::max(s.spec, s.lookup)).size());
    };
    hooks.propose = [this](uint64_t id, int last, size_t remaining, uint64_t position, int cap) {
      State& s = states_.at(id);
      const int replay = s.replay;
      return crucible::VerifySpan{s.slot, Propose(s, last, remaining, position, cap), replay, 1};
    };
    hooks.accept = [this](uint64_t id, int first, std::span<const int> greedy) {
      settle_.push_back(id);
      return Accept(states_.at(id), first, greedy);
    };
    hooks.settle = [this] {
      // Flushes overwrite the residual rows, so they wait until every request has read its rows of the pass.
      for (const uint64_t id : settle_) Flush(states_.at(id));
      settle_.clear();
    };
    hooks.packed = [this](const std::vector<uint64_t>& ids, const std::vector<int>& rows, const std::vector<int>& tokens) {
      Packed(ids, rows, tokens);
    };
    return hooks;
  }
  static std::string Check(const crucible::ServeRequest& r) {
    if (r.draft_head != "64k" || r.draft_adapt != 0 || !(r.draft_q > 0))
      return "batch speculation needs draft_head 64k, draft_adapt 0, and draft_q > 0";
    return {};
  }
  float* Stacks(int slot) override { return stk_.at(size_t(slot)); }
  float* Carry(int slot) override { return carry_.at(size_t(slot)); }
  void Restored(int slot, std::optional<crucible::MtpHeld> held) override {
    held_.at(size_t(slot)) = held;
    closed_.at(size_t(slot)).reset();
  }
  std::optional<crucible::MtpHeld> Live(int slot, size_t cursor) const override {
    for (const auto& [id, s] : states_)
      if (s.slot == slot && s.Full() && s.have_carry && s.ctoks.empty() && cursor > 0 && s.mpos == int(cursor) - 1)
        return crucible::MtpHeld{cursor};
    return std::nullopt;
  }
  std::optional<crucible::MtpHeld> Settle(int slot, size_t logical) override {
    const auto closed = std::exchange(closed_.at(size_t(slot)), std::nullopt);
    held_.at(size_t(slot)).reset();
    if (!closed || !closed->full) return std::nullopt;
    const auto plan = crucible::MtpTailFlush(closed->mpos, closed->ctoks.size(), logical, kMaxRows,
                                             engine_.StreamMtpRows(slot));
    if (!plan) return std::nullopt;
    engine_.SelectStream(slot);
    if (plan->flush) engine_.Mtp(stk_[slot], closed->ctoks.data(), plan->flush, closed->mpos, false);
    Copy(carry_[slot], stk_[slot] + size_t(plan->carry) * kWide, kWide);
    held_.at(size_t(slot)) = crucible::MtpHeld{logical};
    return held_[slot];
  }

 private:
  // The verify rows of one step and what acceptance needs.
  struct Proposal {
    std::vector<int> input;  // the last emitted token, then the drafts
    size_t base = 0;         // stream length before the verify rows
    uint64_t next = 0;       // absolute position of the first verified token (keyed law)
    bool consumed = false;   // drafting fed the pending pairs to the MTP
    bool lookup = false;     // the drafts are copied history tokens
  };
  struct State {
    int slot = 0, spec = 0, mpos = 0, replay = 0, lookup = 0, lookup_min = 8;
    float draft_q = 0;
    bool have_carry = false;
    // Set when the draft depth bound reaches zero: later steps are plain rows, and packed rows skip the catch-up.
    bool mtp_off = false;
    // Catch-up counters: rows appended by packed passes, MTP flushes, and drafts of the first solo steps after them.
    int packed_rows = 0, flushes = 0, resumed_steps = 0, resumed_proposed = 0, resumed_accepted = 0;
    long proposed = 0, accepted = 0;  // all drafts of this request (CRUCIBLE_SPEC_STATS)
    crucible::DraftTally tally;       // counters by position and by token (draft_tally.h)
    long steps = 0, lookup_steps = 0, lookup_proposed = 0, lookup_accepted = 0;
    size_t cached = 0;
    bool resumed = false;
    size_t mtp_start = 0;
    crucible::sampling::Options sampling;
    Proposal proposal;
    std::vector<int> ids, ctoks;
    std::vector<int> stops;      // the request's stop IDs: no draft follows one (draft_stop.h)
    crucible::LookupIndex history;  // prompt and emitted tokens (prompt lookup)
    bool value = false;          // draft_value: chain length by expected tokens per ms
    crucible::DraftCalibration greedy_cal, sampled_cal;
    std::vector<int> buckets;    // margin buckets of the current proposal's drafts (value chains only)
    size_t origin = 0;           // the prompt row at MTP position 0
    bool continued = false;      // the warm-up continued restored MTP history (CRUCIBLE_SPEC_STATS)
    int mtp_restarts = 0;        // FitMtp restarts; the MTP history then starts after row 0
    // The MTP cache holds the pairs of every row from row 0 on (batch_mtp_history.h).
    bool Full() const { return origin == 0 && !mtp_off && mtp_restarts == 0; }
  };
  // A closed request's MTP state, until its tail is saved or the slot opens again.
  struct Closed {
    int mpos = 0;
    std::vector<int> ctoks;
    bool full = false;
  };
  void Open(uint64_t id, int slot, const crucible::ServeRequest& r, size_t cached) {
    State state;
    state.slot = slot; state.spec = r.spec; state.draft_q = float(r.draft_q); state.ids = r.ids; state.stops = r.stop_ids;
    state.sampling = r.sampling;
    state.lookup = r.lookup; state.lookup_min = r.lookup_min;
    state.value = r.draft_value != 0;
    state.tally.detail = r.draft_detail != 0;
    if (r.lookup) { state.history = crucible::LookupIndex(r.lookup_min); state.history.Append(r.ids); }
    state.mtp_start = DraftWarmupStart(r.ids.size(), r.mtp_prefix);
    const auto held = std::exchange(held_.at(size_t(slot)), std::nullopt);
    closed_.at(size_t(slot)).reset();
    if (crucible::MtpContinues(r, cached, held)) {
      // The restore wrote the MTP history of the whole prefix and its carry: the warm-up continues from there.
      state.mpos = held->Mpos();
      state.have_carry = true;
      state.continued = true;
    } else {
      // A restored target-only prefix starts a fresh MTP warm-up, as RunGen does after a restore.
      if (cached) engine_.ResetMtp();
      state.origin = std::max(cached, state.mtp_start);
    }
    state.cached = cached;
    states_[id] = std::move(state);
  }
  void Copy(float* dst, const float* src, size_t count) {
    HIP_OK(hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToDevice, 0));
  }
  // x4-tail: a joined launch warms once per grid chunk, from the launch start, so
  // the MTP calls, positions and carry rows are those of the launches without joins (each held one grid chunk of
  // warm-up rows at most).
  void Chunk(State& s, size_t start, int rows, bool bulk) {
    if (!bulk || rows <= engine_.PrefillRows()) return ChunkRows(s, start, rows, bulk, 0);
    for (const auto& [offset, count] : crucible::PrefillGridPieces(rows, engine_.PrefillRows()))
      ChunkRows(s, start + size_t(offset), count, bulk, offset);
  }
  // `offset`: the rows of the launch before `start` (the bulk stacks hold the whole launch).
  void ChunkRows(State& s, size_t start, int rows, bool bulk, int offset) {
    const int start_row = DraftWarmupRow(start, rows, s.mtp_start);
    if (start_row >= rows) return;
    float* stk = stk_[s.slot];
    float* carry = carry_[s.slot];
    const int warm_rows = rows - start_row;
    float* source = engine_.H() + size_t(start_row) * kWide;
    if (bulk) {
      // A bulk chunk's stacks stay in the engine's padded buffer, not in H().
      if (warm_rows > engine_.PrefillRows()) std::abort();  // PromptStacks() holds one grid chunk
      engine_.CopyPrefillStacks(engine_.PromptStacks(), offset + start_row, warm_rows);
      source = engine_.PromptStacks();
    }
    const int pairs = warm_rows - 1 + int(s.have_carry);
    if (pairs > 0) FitMtp(s, pairs, 1);
    // A long warm-up (a large or full MTP window) runs in one bulk pass (mtp_bulk.h); the 8-row loop is skipped.
    const bool warmed = bulk && pairs > 0 &&
        engine_.MtpWarm(s.have_carry ? carry : nullptr, source, s.ids.data() + start + start_row + (s.have_carry ? 0 : 1),
                        pairs, s.mpos);
    if (warmed) s.mpos += pairs;
    for (int p = 0; p < pairs && !warmed; p += kMaxRows) {
      const int count = std::min(kMaxRows, pairs - p);
      const int first = p == 0 && s.have_carry ? 1 : 0;
      if (first) Copy(stk, carry, kWide);
      const int source_row = p - int(s.have_carry) + first;
      if (count > first) Copy(stk + size_t(first) * kWide, source + size_t(source_row) * kWide, size_t(count - first) * kWide);
      std::vector<int> tokens;
      for (int j = 0; j < count; ++j) tokens.push_back(s.ids[start + start_row + p + j + (s.have_carry ? 0 : 1)]);
      if (!bulk) Copy(carry, source + size_t(warm_rows - 1) * kWide, kWide);
      engine_.Mtp(stk, tokens.data(), count, s.mpos, true);
      s.mpos += count;
    }
    if (bulk || pairs == 0) Copy(carry, source + size_t(warm_rows - 1) * kWide, kWide);
    s.have_carry = true;
  }
  void First(State& s, int token) {
    Emit(s, std::span<const int>(&token, 1));
    // Pending trunk stacks pair with the exact tokens that follow them.
    Copy(stk_[s.slot], carry_[s.slot], kWide);
    s.ctoks.clear();
    s.replay = 0;
  }
  // Pending draft pairs: stk rows [0, ctoks.size()) hold trunk stacks; ctoks[i] is the token that followed row i.
  // The last pending token is always the request's last emitted token.
  // a slot's MTP cache may hold fewer positions than its context (MtpSlotRows). A step that would pass them
  // restarts the MTP at position 0 (bounds.h MtpRestart); the pending pairs stay and start the new history.
  // Chunk fits one launch of prefill rows after a restart, so a bounded slot must hold more than one launch, or
  // MtpWarm exits the server.
  static_assert(crucible::kMtpSlotOutputRows > crucible::kMaxPrefillLaunchRows);
  void FitMtp(State& s, int catchup, int requested) {
    if (!crucible::MtpRestart(s.mpos, catchup, requested, engine_.StreamMtpRows(s.slot))) return;
    engine_.SelectStream(s.slot);
    engine_.ResetMtp();
    s.mpos = 0;
    ++s.mtp_restarts;
  }
  void Pending(State& s, int last) {
    if (s.mtp_off) return;
    if (s.ctoks.empty()) s.ctoks = {last};
    if (s.ctoks.back() != last || s.ctoks.size() > size_t(kMaxRows))
      throw std::logic_error("speculative draft state does not match the emitted tokens");
  }
  std::optional<int> Commit(State& s, int last) {
    Pending(s, last);
    if (s.replay == 0) return std::nullopt;
    const int token = engine_.Forward(&last, 1, s.replay, 1)[0];
    s.replay = 0;
    return token;
  }
  // Each packed row adds one pending pair. Nine pairs flush eight through the MTP, so the next solo step drafts
  // from a current MTP cache; the newest pair stays pending as the draft input.
  void Packed(const std::vector<uint64_t>& ids, const std::vector<int>& rows, const std::vector<int>& tokens) {
    for (size_t i = 0; i < ids.size(); ++i) {
      State& s = states_.at(ids[i]);
      if (s.mtp_off) { Emit(s, std::span<const int>(&tokens[i], 1)); continue; }
      Emit(s, std::span<const int>(&tokens[i], 1));
      CopyRows(engine_, stk_[s.slot], int(s.ctoks.size()), rows[i], 1);
      s.ctoks.push_back(tokens[i]);
      ++s.packed_rows;
      s.resumed = true;
    }
    // Flushes overwrite the residual rows, so they run after every row of the pass is copied.
    for (const uint64_t id : ids) Flush(states_.at(id));
  }
  // Tokens the request emitted, in order. A stop inside a verified span only ends the request, so the extra
  // history entries are never read.
  static void Emit(State& s, std::span<const int> tokens) {
    if (s.lookup) s.history.Append(tokens);
  }
  // The copied continuation of the longest earlier match of the history suffix (draft_policy.h).
  std::vector<int> Lookup(const State& s, int depth) const {
    if (!s.lookup || depth < 2) return {};
    return s.history.Draft(std::min(s.lookup, depth));
  }
  void Flush(State& s) {
    if (s.mtp_off || s.ctoks.size() <= size_t(kMaxRows)) return;
    FitMtp(s, kMaxRows, 1);
    engine_.SelectStream(s.slot);
    float* stk = stk_[s.slot];
    engine_.Mtp(stk, s.ctoks.data(), kMaxRows, s.mpos, false);
    s.mpos += kMaxRows;
    ++s.flushes;
    Copy(stk, stk + size_t(kMaxRows) * kWide, kWide);
    s.ctoks = {s.ctoks.back()};
  }
  // A solo step is one proposal, one verify pass of its own rows, and one acceptance (G6a). Packed verification
  // (G6d) runs the same Propose and Accept around one shared pass, so both apply the same policy.
  std::vector<int> Solo(State& s, int last, size_t remaining, uint64_t next) {
    const auto input = Propose(s, last, remaining, next, s.spec);
    const auto greedy = engine_.Forward(input.data(), int(input.size()), s.replay, 1);
    auto output = Accept(s, 0, greedy);
    if (s.resumed && s.proposal.consumed) {
      ++s.resumed_steps;
      s.resumed_proposed += int(input.size()) - 1;
      s.resumed_accepted += int(output.size()) - 1;
      if (s.resumed_steps % 8 == 0) s.resumed = false;
    }
    return output;
  }
  // Draft head pass counters: the host time of one head call, where the call already returns host values
  // (the choice and the margin), so these reads add no wait. Counters only.
  static double HeadMs(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
  }
  template <class F>
  decltype(auto) Timed(State& s, F&& call) {
    struct Pass {
      crucible::DraftTally& tally; std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
      ~Pass() { tally.Pass(HeadMs(since)); }
    } pass{s.tally};
    return call();
  }
  // One pass for several requests: each takes an equal share of its time, so the shares add up to the pass time.
  template <class Members, class F>
  decltype(auto) TimedShared(Members& members, size_t begin, size_t end, F&& call) {
    struct Pass {
      Members& members; size_t begin, end; std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
      ~Pass() {
        const double share = HeadMs(since) / double(std::max<size_t>(end - begin, 1));
        for (size_t i = begin; i < end; ++i) Tally(members[i]).Pass(share);
      }
      static crucible::DraftTally& Tally(typename Members::value_type& m) {
        if constexpr (std::is_pointer_v<typename Members::value_type>) return m->s->tally; else return m.s->tally;
      }
    } pass{members, begin, end};
    return call();
  }
  template <class Members, class F>
  decltype(auto) TimedShared(Members& members, F&& call) { return TimedShared(members, 0, members.size(), call); }
  // Drafts up to `cap` tokens (bounded as serial generation bounds them) and returns the verify rows: the
  // last emitted token, then the drafts. Mirrors one greedy RunGen iteration (lookup, overlap, and head_auto
  // off) or SampledGeneration::Serial and Step (without lookup or snapshots).
  std::vector<int> Propose(State& s, int last, size_t remaining, uint64_t next, int cap) {
    Pending(s, last);
    s.buckets.clear();
    Proposal p;
    p.input = {last};
    p.base = engine_.Length();
    p.next = next;
    if (!s.mtp_off) {
      const bool sampled = s.sampling.temperature > 0;
      FitMtp(s, int(s.ctoks.size()), std::max(s.spec, s.lookup));
      const int bound = BoundedDraftDepth(std::max(s.spec, s.lookup), engine_.Context(), int(p.base), s.mpos,
                                          int(s.ctoks.size()), engine_.StreamMtpRows(s.slot));
      const int limit = sampled ? std::min(bound, int(remaining) - 1) : bound;
      // Greedy drafts stop at the token limit too: the step commits at most drafts + 1 tokens.
      const int room = crucible::DraftRoom(remaining);
      if (limit <= 0) s.mtp_off = true;
      else if (cap > 0) {
        const int depth = std::min({s.spec, limit, cap, room});
        const int copy_depth = std::min({std::max(s.spec, s.lookup), limit, cap, room});
        if (s.lookup && (s.history.Empty() || s.history.Back() != last))
          throw std::logic_error("lookup history does not match the emitted tokens");
        const auto drafts = sampled ? SampledDrafts(s, depth, copy_depth, next, p.lookup)
                                    : GreedyDrafts(s, depth, copy_depth, p.lookup);
        p.input.insert(p.input.end(), drafts.begin(), drafts.end());
        p.consumed = true;
      }
    }
    ++s.steps;
    if (p.lookup) { ++s.lookup_steps; s.lookup_proposed += int(p.input.size()) - 1; }
    s.proposal = p;
    return p.input;
  }
  // Propose for several requests with grouped MTP passes: one pass for every greedy request's catch-up rows (split
  // when they exceed eight rows), then one pass per chain level for the requests whose chain continues. The policy
  // per request is GreedyDrafts' (lookup, draft_q or draft_value). Sampled requests propose alone.
  std::vector<crucible::VerifySpan> ProposeAll(const std::vector<uint64_t>& ids, const std::vector<int>& last,
                                               const std::vector<size_t>& remaining,
                                               const std::vector<uint64_t>& position, const std::vector<int>& cap) {
    struct Chain {
      State* s; size_t index; int rc = 0, depth = 0, copy_depth = 0, offset = 0;
      std::vector<int> drafts; float margin = 0, q = 1; crucible::DraftValue value; bool open = true;
    };
    std::vector<crucible::VerifySpan> out(ids.size());
    std::vector<Chain> chains;
    for (size_t i = 0; i < ids.size(); ++i) {
      State& s = states_.at(ids[i]);
      engine_.SelectStream(s.slot);
      if (s.sampling.temperature > 0 || s.mtp_off || cap[i] <= 0 || remaining[i] <= 1) {
        out[i] = {s.slot, Propose(s, last[i], remaining[i], position[i], cap[i]), s.replay, 1};
        continue;
      }
      Pending(s, last[i]);
      s.buckets.clear();
      Proposal p;
      p.input = {last[i]};
      p.base = engine_.Length();
      p.next = position[i];
      s.proposal = p;
      FitMtp(s, int(s.ctoks.size()), std::max(s.spec, s.lookup));
      const int limit = BoundedDraftDepth(std::max(s.spec, s.lookup), engine_.Context(), int(p.base), s.mpos,
                                          int(s.ctoks.size()), engine_.StreamMtpRows(s.slot));
      if (limit <= 0) {
        s.mtp_off = true;
        ++s.steps;
        out[i] = {s.slot, p.input, s.replay, 1};
        continue;
      }
      if (s.lookup && (s.history.Empty() || s.history.Back() != last[i]))
        throw std::logic_error("lookup history does not match the emitted tokens");
      Chain c{&s, i};
      c.rc = int(s.ctoks.size());
      const int room = crucible::DraftRoom(remaining[i]);
      c.depth = std::min({s.spec, limit, cap[i], room});
      c.copy_depth = std::min({std::max(s.spec, s.lookup), limit, cap[i], room});
      chains.push_back(std::move(c));
    }
    // Catch-up rows, packed into passes of at most eight rows.
    for (size_t begin = 0; begin < chains.size();) {
      std::vector<Engine::MtpSpan> spans;
      size_t end = begin;
      int rows = 0;
      for (; end < chains.size() && rows + chains[end].rc <= kMaxRows; ++end) {
        State& s = *chains[end].s;
        chains[end].offset = rows;
        rows += chains[end].rc;
        spans.push_back({s.slot, stk_[s.slot], s.ctoks, s.mpos, true});
      }
      const auto choices = TimedShared(chains, begin, end, [&] { return engine_.MtpSpans(spans); });
      for (size_t k = begin; k < end; ++k) {
        Chain& c = chains[k];
        c.drafts = {choices[k - begin].token};
        c.margin = choices[k - begin].margin;
        c.s->mpos += c.rc;
        CopyRows(engine_, stk_[c.s->slot], kMaxRows, c.offset + c.rc - 1, 1);
      }
      begin = end;
    }
    // First draft: lookup, then the chain policy decides whether the chain continues.
    for (Chain& c : chains) {
      State& s = *c.s;
      auto copied = Lookup(s, c.copy_depth);
      Proposal& p = s.proposal;
      if (copied.size() > 1 && copied.front() == c.drafts.front()) {
        p.lookup = true;
        c.drafts = copied;
        crucible::TrimDrafts(c.drafts, s.stops, c.copy_depth);
        c.open = false;
      } else if (s.value) {
        const int bucket = crucible::DraftMarginBucket(c.margin);
        const float pk = s.greedy_cal.P(bucket);
        if (!c.value.Keep(costs_, pk)) { c.drafts.clear(); c.open = false; }
        else {
          c.value.Add(pk);
          s.buckets.push_back(bucket);
          c.open = c.depth > 1 && c.value.Try(costs_, s.greedy_cal.Mean());
        }
      } else {
        c.q = DraftAcceptProb(c.margin);
        c.open = c.depth > 1 && c.q * kDraftNextPrior >= s.draft_q;
      }
      // No chain call follows a stop ID.
      if (!p.lookup && !c.drafts.empty() && crucible::IsStopToken(s.stops, c.drafts.front())) c.open = false;
    }
    // Chain levels: one pass per level over the requests whose chain continues.
    for (int k = 1;; ++k) {
      std::vector<Chain*> level;
      std::vector<Engine::MtpSpan> spans;
      for (Chain& c : chains)
        if (c.open) {
          level.push_back(&c);
          spans.push_back({c.s->slot, stk_[c.s->slot] + size_t(kMaxRows) * kWide, {c.drafts.back()},
                           c.s->mpos + k - 1, true});
        }
      if (level.empty()) break;
      const auto choices = TimedShared(level, [&] { return engine_.MtpSpans(spans); });
      for (size_t j = 0; j < level.size(); ++j) {
        Chain& c = *level[j];
        State& s = *c.s;
        CopyRows(engine_, stk_[s.slot], kMaxRows, int(j), 1);
        const int next = choices[j].token;
        const float margin = choices[j].margin;
        bool keep;
        if (s.value) {
          const int bucket = crucible::DraftMarginBucket(margin);
          const float pk = s.greedy_cal.P(bucket);
          keep = c.value.Keep(costs_, pk);
          if (keep) {
            c.value.Add(pk);
            s.buckets.push_back(bucket);
          }
          c.open = keep && k + 1 < c.depth && c.value.Try(costs_, s.greedy_cal.Mean());
        } else {
          c.q *= DraftAcceptProb(margin);
          keep = c.q >= s.draft_q;
          c.open = keep && k + 1 < c.depth && c.q * kDraftNextPrior >= s.draft_q;
        }
        if (keep) c.drafts.push_back(next);
        else c.open = false;
        if (keep && crucible::IsStopToken(s.stops, next)) c.open = false;
      }
    }
    for (Chain& c : chains) {
      State& s = *c.s;
      Proposal& p = s.proposal;
      p.input.insert(p.input.end(), c.drafts.begin(), c.drafts.end());
      p.consumed = true;
      ++s.steps;
      if (p.lookup) { ++s.lookup_steps; s.lookup_proposed += int(p.input.size()) - 1; }
      out[c.index] = {s.slot, p.input, s.replay, 1};
    }
    return out;
  }
  // As RunGen with --lookup: a copied chain replaces the MTP chain when the MTP's first draft agrees with it.
  std::vector<int> GreedyDrafts(State& s, int depth, int copy_depth, bool& lookup) {
    float* stk = stk_[s.slot];
    const int rc = int(s.ctoks.size());
    std::vector<int> drafts{Timed(s, [&] { return engine_.Mtp(stk, s.ctoks.data(), rc, s.mpos, true, false); })[rc - 1]};
    s.mpos += rc;
    auto copied = Lookup(s, copy_depth);
    if (copied.size() > 1 && copied.front() == drafts.front()) {
      lookup = true;
      crucible::TrimDrafts(copied, s.stops, copy_depth);
      return copied;
    }
    if (depth <= 0) return {};
    if (s.value)
      return ValueChain(s, s.greedy_cal, drafts.front(), engine_.DraftMargin(), depth, [&](int k, int previous) {
        CopyRows(engine_, stk, kMaxRows, k == 1 ? rc - 1 : 0, 1);
        const int next = Timed(s, [&] { return engine_.Mtp(stk + size_t(kMaxRows) * kWide, &previous, 1, s.mpos + k - 1, true, false); })[0];
        return std::pair<int, float>{next, engine_.DraftMargin()};
      });
    float q = DraftAcceptProb(engine_.DraftMargin());
    for (int k = 1; k < depth; ++k) {
      if (q * kDraftNextPrior < s.draft_q || crucible::IsStopToken(s.stops, drafts.back())) break;
      CopyRows(engine_, stk, kMaxRows, k == 1 ? rc - 1 : 0, 1);
      const int next = Timed(s, [&] { return engine_.Mtp(stk + size_t(kMaxRows) * kWide, &drafts.back(), 1, s.mpos + k - 1, true, false); })[0];
      q *= DraftAcceptProb(engine_.DraftMargin());
      if (q < s.draft_q) break;
      drafts.push_back(next);
    }
    return drafts;
  }
  // A copied chain replaces the MTP chain when the MTP's keyed first draw agrees with it, as for greedy. Keyed
  // acceptance emits the target's own draws, so the choice of drafts changes speed only (an unchecked copy was
  // 9% slower on sampled code).
  std::vector<int> SampledDrafts(State& s, int depth, int copy_depth, uint64_t next, bool& lookup) {
    float* stk = stk_[s.slot];
    const int rows = int(s.ctoks.size());
    auto copied = Lookup(s, copy_depth);
    Timed(s, [&] { engine_.Mtp(stk, s.ctoks.data(), rows, s.mpos, true); });
    s.mpos += rows;
    const auto draw = [&](const std::vector<float>& logits, const std::vector<int>& mapping, uint64_t position) {
      return crucible::sampling::FromLogits(logits, s.sampling, mapping).KeyedSample(s.sampling.seed, position);
    };
    const auto [first_logits, first_mapping] = engine_.DraftLogits(rows);
    std::vector<int> drafts{draw(first_logits, first_mapping, next)};
    if (copied.size() > 1 && copied.front() == drafts.front()) {
      lookup = true;
      crucible::TrimDrafts(copied, s.stops, copy_depth);
      return copied;
    }
    if (s.value)
      return ValueChain(s, s.sampled_cal, drafts.front(), SampledRawMargin(first_logits), depth, [&](int k, int previous) {
        CopyRows(engine_, stk, kMaxRows, k == 1 ? rows - 1 : 0, 1);
        Timed(s, [&] { engine_.Mtp(stk + size_t(kMaxRows) * kWide, &previous, 1, s.mpos + k - 1); });
        const auto [logits, mapping] = engine_.DraftLogits(1);
        return std::pair<int, float>{draw(logits, mapping, next + uint64_t(k)), SampledRawMargin(logits)};
      });
    float confidence = SampledDepthConfidence(SampledRawMargin(first_logits));
    for (int k = 1; k < depth; ++k) {
      if (confidence * 0.75f < s.draft_q || crucible::IsStopToken(s.stops, drafts.back())) break;
      CopyRows(engine_, stk, kMaxRows, k == 1 ? rows - 1 : 0, 1);
      Timed(s, [&] { engine_.Mtp(stk + size_t(kMaxRows) * kWide, &drafts.back(), 1, s.mpos + k - 1); });
      const auto [logits, mapping] = engine_.DraftLogits(1);
      confidence *= SampledDepthConfidence(SampledRawMargin(logits));
      if (confidence < s.draft_q) break;
      drafts.push_back(draw(logits, mapping, next + drafts.size()));
    }
    return drafts;
  }
  // Value-aware chain (draft_value.h): keep each computed draft while it pays for its verify row, and compute the
  // next one while its expected acceptance (the request's recent mean) pays for the MTP call and the row.
  // `next(k, previous)` computes draft k from draft k - 1 and returns it with its MTP margin.
  template <class Next>
  std::vector<int> ValueChain(State& s, const crucible::DraftCalibration& cal, int first, float margin, int depth,
                              Next next) {
    crucible::DraftValue value;
    std::vector<int> drafts;
    int token = first;
    for (int k = 0;;) {
      const int bucket = crucible::DraftMarginBucket(margin);
      const float p = cal.P(bucket);
      if (!value.Keep(costs_, p)) break;
      value.Add(p);
      drafts.push_back(token);
      s.buckets.push_back(bucket);
      if (++k >= depth || crucible::IsStopToken(s.stops, token) || !value.Try(costs_, cal.Mean())) break;
      std::tie(token, margin) = next(k, drafts.back());
    }
    return drafts;
  }
  // Verifies the proposal against its rows [first, first + rows) of the last pass. Greedy accepts the matching
  // prefix; sampled accepts a draft only when it equals the target's own keyed draw. The emitted tokens are
  // always the target's. A proposal without drafts appends its row as a pending pair (call Flush after the pass).
  std::vector<int> Accept(State& s, int first, std::span<const int> greedy) {
    const Proposal& p = s.proposal;
    const int depth = int(p.input.size()) - 1;
    if (greedy.size() < p.input.size()) throw std::runtime_error("verify pass returned too few rows");
    std::vector<int> verified;
    int accepted = 0;
    std::vector<float> logits;
    const bool sampled = s.sampling.temperature > 0;
    if (sampled) logits = engine_.CopyLogits(depth + 1, first);
    const size_t vocab = size_t(engine_.Vocabulary());
    for (int row = 0; row <= depth; ++row) {
      const int target = sampled
          ? crucible::sampling::FromLogits(std::span<const float>(logits).subspan(size_t(row) * vocab, vocab), s.sampling)
                .KeyedSample(s.sampling.seed, p.next + row)
          : greedy[size_t(row)];
      verified.push_back(target);
      if (row == depth || target != p.input[size_t(row) + 1]) break;
      ++accepted;
    }
    engine_.SelectStream(s.slot);
    if (depth > 0) engine_.Truncate(p.base + accepted + 1);
    s.replay = accepted;
    s.proposed += depth;
    s.accepted += accepted;
    s.tally.Round(std::span<const int>(p.input).subspan(1), accepted, verified.back(), p.lookup);
    if (p.lookup) s.lookup_accepted += accepted;
    if (s.value && !p.lookup && depth > 0) (sampled ? s.sampled_cal : s.greedy_cal).Observe(s.buckets, depth, accepted);
    Emit(s, verified);
    if (s.mtp_off) return verified;
    if (p.consumed) {
      CopyRows(engine_, stk_[s.slot], 0, first, accepted + 1);
      s.ctoks = verified;
    } else {
      CopyRows(engine_, stk_[s.slot], int(s.ctoks.size()), first, 1);
      s.ctoks.push_back(verified.front());
      ++s.packed_rows;
      s.resumed = true;
    }
    return verified;
  }
  Engine& engine_;
  const crucible::DraftCosts costs_ = crucible::DraftCosts::Default();
  std::array<float*, kMaxRows> stk_{}, carry_{};
  // For each slot: the MTP history that its MTP cache and carry hold, and a closed request's pending pairs.
  std::array<std::optional<crucible::MtpHeld>, kMaxRows> held_{};
  std::array<std::optional<Closed>, kMaxRows> closed_{};
  std::map<uint64_t, State> states_;
  std::vector<uint64_t> settle_;
};

// The disk tier's stop rule: a stopping engine flushes its states for this long at most. SIGTERM (a service stop
// that signals the whole group) does not end the engine; the server then closes stdin, the engine flushes and exits.
// Without a server, the engine exits kStateTermGrace seconds after SIGTERM. Both fit inside a systemd
// TimeoutStopSec=120 with a margin (server/batch_backend.py STATE_FLUSH_SECONDS waits 85 s, then kills).
inline constexpr double kStateFlushSeconds = 75, kStateTermGrace = 90;
inline volatile std::sig_atomic_t g_state_term = 0;
inline void KeepRunningOnTerm() {
  struct sigaction action{};
  action.sa_handler = [](int) { g_state_term = 1; };
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESTART;
  sigaction(SIGTERM, &action, nullptr);
  std::thread([] {
    while (!g_state_term) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::this_thread::sleep_for(std::chrono::duration<double>(kStateTermGrace));
    std::fprintf(stderr, "state_disk stop: no end of input %.0f s after SIGTERM\n", kStateTermGrace);
    std::_Exit(143);
  }).detach();
}
// The disk tier's options: the directory names the binary, the checkpoint and every state-changing setting.
inline crucible::StateDiskOptions BatchStateDiskOptions(const std::string& checkpoint, const std::string& dir, int gib,
                                                        int prefill_rows, bool vision, int write_gib_per_day,
                                                        int mtp_rows, int write_delay_ms = 0) {
  const std::string binary = crucible::FileSha256("/proc/self/exe");
  if (binary.empty()) throw std::runtime_error("state disk: cannot read the engine binary");
  auto settings = crucible::BatchStateDiskSettings(checkpoint, prefill_rows, vision, mtp_rows,
                                                   crucible::SwitchEnvironment());
  // The same-second proof's states have their own folder; the default identity does not change.
  if (write_delay_ms) settings.push_back("write_delay_ms=" + std::to_string(write_delay_ms));
  const auto identity = crucible::StateDiskIdentityFor(binary, crucible::CheckpointSha256(checkpoint), settings);
  crucible::StateDiskOptions options;
  options.root = dir;
  options.identity = identity.sha;
  options.directory = identity.directory;
  options.budget = uint64_t(gib) << 30;
  options.write_per_day = uint64_t(write_gib_per_day) << 30;
  options.write_delay_ms = unsigned(std::max(write_delay_ms, 0));
  return options;
}

inline int BatchedServe(const std::string& checkpoint, int context, int streams, int prefill_rows=kPrefillAuto,
                        int cache_mib=0, int short_streams=0, int short_ctx=0, bool session_replay=false,
                        bool slot_resume=true, bool slot_checkpoints=true, bool vision=false,
                        int host_reserve_mib=crucible::kServeHostReserveMib, const std::string& state_dir="",
                        int state_disk_gib=64, bool cache_probation=true, int state_write_gib_per_day=300,
                        int cache_idle_minutes=30, int mtp_window=crucible::kDefaultMtpPrefix,
                        const std::vector<int>& draft_blocks={}, int state_write_delay_ms=0) {
  // each slot's MTP cache holds the positions that the server's default window needs, not the context.
  const int mtp_rows = crucible::MtpSlotRows(context, mtp_window);
  Engine engine(checkpoint, context, prefill_rows, mtp_rows);
  if (vision) engine.EnableVision(checkpoint);
  const size_t cache_budget = size_t(cache_mib) << 20;
  crucible::BatchSession<Engine> session(engine, streams, cache_budget, short_streams, short_ctx);
  engine.EnsureServeMtpCaches();
  std::unique_ptr<BatchPrefixCache> cache;
  if (cache_budget) {
    std::optional<crucible::StateDiskOptions> disk;
    if (!state_dir.empty())
      disk = BatchStateDiskOptions(checkpoint, state_dir, state_disk_gib, engine.PrefillRows(), vision,
                                   state_write_gib_per_day, mtp_rows, state_write_delay_ms);
    cache = std::make_unique<BatchPrefixCache>(engine, cache_budget, session_replay, streams, slot_resume,
                                               slot_checkpoints, uint64_t(host_reserve_mib) << 20, std::move(disk),
                                               cache_probation, cache_idle_minutes);
    session.SetCacheHooks(cache->Hooks());
  }
  // Re-send fix A (End points keep the prompt outputs) and fix B (the Pre point) need slot checkpoints: the session
  // asks the checkpoint hooks. The line shows whether they run; there is no switch.
  const bool points = cache && cache->CheckpointsOn();
  if (points) session.SetPreEnd(crucible::kPreEndMinTokens);
  std::fprintf(stderr, "slot_checkpoint end_resume=%d pre_end=%d pre_end_min=%zu\n", int(points), int(points),
               points ? crucible::kPreEndMinTokens : size_t(0));
  BatchSpecDecoder spec(engine, draft_blocks);
  session.SetSpecHooks(spec.Hooks());
  // CRUCIBLE_PACKED_ROWS: rows of one packed verify pass (default: 16 rows from 4 decoders where the engine runs
  // them, else 8; a value 8 to 16 is a fixed cap).
  crucible::PackedRowsSetting pass_rows;
  try {
    pass_rows = crucible::PackedRowsOption(crucible::SwitchText("CRUCIBLE_PACKED_ROWS"));
    session.SetPassRows(pass_rows.rows, pass_rows.wide_from);
  } catch (const std::invalid_argument& error) {
    std::fprintf(stderr, "serve: %s\n", error.what());
    return 2;
  }
  std::fputs(crucible::PackedRowsNotice(pass_rows).c_str(), stderr);
  if (cache) cache->SetMtp(&spec);
  engine.WarmPlans();
  HIP_OK(hipDeviceSynchronize());
  // `memory ready` comes before the slots and checkpoints; this is the headroom that the cache really has.
  Engine::ReportMemory("serve");
  std::string contexts;
  for (const int slot_context : session.SlotContexts()) contexts += (contexts.empty() ? "" : ",") + std::to_string(slot_context);
  const bool disk = cache && cache->DiskOn();
  if (disk) KeepRunningOnTerm();
  // "state_disk": the engine takes stage commands (the server sends them only then).
  std::printf("{\"event\":\"ready\",\"context\":%d,\"streams\":%d,\"prefill_rows\":%d,\"slot_contexts\":[%s],"
              "\"protocol\":\"batch-v1\",\"cache_tail\":true,%s%s}\n", context, streams, engine.PrefillRows(), contexts.c_str(),
              crucible::ReadyModelJson().c_str(), disk ? ",\"state_disk\":true,\"stage_job\":true" : "");
  std::fflush(stdout);
  std::string line;
  bool oversized;
  const int profile_every = crucible::SwitchText("CRUCIBLE_SERVE_PROFILE") ? std::atoi(crucible::SwitchText("CRUCIBLE_SERVE_PROFILE")) : 0;
  int profiled = 0;
  while (ServeLine(line, oversized)) {
    std::string op;
    std::optional<size_t> cursor;
    const double command_start = crucible::GenerationNow();
    try {
      if (oversized) throw std::runtime_error("request exceeds the 2 MiB JSON-line limit");
      const auto command = crucible::ParseBatchCommand(line, context);
      op = command.op;
      std::vector<crucible::BatchEvent> events;
      // W1: a bad tenant field refuses only this open (recoverable, the fixed message). A stage reads nothing.
      if (command.refusal && op == "open") throw crucible::RecoverableOpenError(*command.refusal);
      if (command.refusal && op == "stage") std::printf("{\"event\":\"staged\",\"state\":\"none\",\"tokens\":0}\n");
      else if (op == "open") events = session.Open(command.request_id, *command.request, true);
      else if (op == "prefill" && !command.request_ids.empty()) {
        // Each verify pass inside the chunk publishes at once, then marks its end, so tokens reach clients during
        // the chunk. The command's own events follow the last part.
        events = session.Prefill(command.request_id, command.request_ids, command.every,
                                 [](std::vector<crucible::BatchEvent> part) {
                                   for (const auto& event : part) PrintBatchEvent(event);
                                   std::printf("{\"event\":\"batch_part\",\"op\":\"verify_step\"}\n");
                                   std::fflush(stdout);
                                 });
        cursor = session.PrefillCursor();
      }
      else if (op == "prefill") {
        events = session.Prefill(command.request_id);
        cursor = session.PrefillCursor();
      }
      else if (op == "step") events = session.Step(command.request_ids);
      else if (op == "spec_step") events = session.SpecStep(command.request_id);
      else if (op == "verify_step") {
        events = session.VerifyStep(command.request_ids);
        // CRUCIBLE_SERVE_PROFILE=N: per-stage GPU time of packed passes, reported every N passes (marks add time).
        if (profile_every > 0 && ++profiled == 1) engine.EnableProfile(true);
        if (profile_every > 0 && profiled > profile_every) {
          std::fprintf(stderr, "serve profile: %d verify_step commands, %zu requests in the last\n", profile_every,
                       command.request_ids.size());
          engine.Report(profile_every);
          engine.EnableProfile(true);
          profiled = 1;
        }
      }
      else if (op == "stage") {
        // Counts only: the read state and the length of the disk state.
        const auto staged = session.Stage(*command.request, command.position, command.read, command.job);
        std::printf("{\"event\":\"staged\",\"state\":\"%s\",\"tokens\":%zu,\"reason\":%s,\"tag\":%llu}\n",
                    staged.state.c_str(), staged.tokens, JsonQuote(staged.reason.empty() ? "-" : staged.reason).c_str(),
                    static_cast<unsigned long long>(staged.tag));
      }
      else session.Close(command.request_id);
      for (const auto& event : events) PrintBatchEvent(event);
    } catch (const std::exception& error) {
      std::printf("{\"event\":\"error\",\"error\":%s,\"recoverable\":%s}\n", JsonQuote(error.what()).c_str(),
                  crucible::IsRecoverableBatchError(error, op, session.Poisoned()) ? "true" : "false");
    }
    // engine_s: from the parsed line to this event, so the server can tell its own overhead from engine time.
    std::printf("{\"event\":\"batch_end\",\"op\":%s%s,\"engine_s\":%.6f}\n", JsonQuote(op).c_str(),
                cursor ? (",\"cursor\":" + std::to_string(*cursor)).c_str() : "", crucible::GenerationNow() - command_start);
    std::fflush(stdout);
    // A first token is out: its prompt end (MTP warm-up, End checkpoint) runs while the server reads it.
    // A failure poisons the session, and the engine stops below, as after a failed command. Its error event comes
    // after the boundary, so the server reads it with the next command and fails the requests (a fatal error).
    if (const auto failed = session.TrySettlePrompt()) {
      std::fprintf(stderr, "prompt end failed: %s\n", failed->c_str());
      std::printf("{\"event\":\"error\",\"error\":%s,\"recoverable\":false}\n", JsonQuote("prompt end failed: " + *failed).c_str());
      std::fflush(stdout);
    }
    // Sampler or allocation failures can follow a state write. No retry is safe.
    if (session.Poisoned()) return 1;
  }
  // End of input: the server stops this engine. Write the states that the next start of this binary can use.
  if (cache) cache->FlushToDisk(kStateFlushSeconds, session.FrontierPoints());
  return 0;
}

inline std::optional<int> TryBatchedServeCommand(int argc, char** argv) {
  bool requested = false;
  for (int index = 3; index < argc; ++index) requested |= std::string(argv[index]) == "--streams";
  if (!requested) return std::nullopt;
  try {
    static_assert(kMaxRows==crucible::kServeBatchSlots);
    std::vector<std::string_view> args;
    for (int index=3;index<argc;++index) args.emplace_back(argv[index]);
    const auto settings=crucible::ParseBatchedServeSettings(args);
    // Read and check the head file before the model loads, so a bad file costs no load time.
    std::vector<int> draft_blocks;
    if (!settings.draft_blocks.empty())
      draft_blocks=crucible::ReadServeDraftBlocks(settings.draft_blocks,crucible::kVocab/128);
    return BatchedServe(argv[2],settings.ctx,settings.streams,settings.prefill_rows,settings.cache_mib,
                        settings.short_streams,settings.short_ctx,settings.session_replay,settings.slot_resume,
                        settings.slot_checkpoints,settings.vision,settings.host_reserve_mib,settings.state_dir,
                        settings.state_disk_gib,settings.cache_probation,settings.state_write_gib_per_day,
                        settings.cache_idle_minutes,settings.mtp_window,draft_blocks,settings.state_write_delay_ms);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "batched serve error: %s; use serve CKPT --streams 1..8 [--ctx 1..262144] [--prefill 0..4096] [--conversation-cache-mib 0..131072] [--host-reserve-mib 1024..131072] [--short-streams N --short-ctx T] [--cache-probation 0|1] [--cache-idle-minutes 1..10080] [--mtp-window 0..262144] [--draft-blocks FILE] [--state-dir DIR [--state-disk-gib 1..65536] [--state-disk-write-gib-per-day 0..65536] [--state-disk-write-delay-ms 0..60000]]\n", error.what());
    return 2;
  }
}
