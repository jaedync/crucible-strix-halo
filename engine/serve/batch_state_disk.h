// The disk tier of batch serve: which disk states a request can restore, the stage
// step before an open, the spill of host evictions, and the flush of a stopping engine. CPU only; batched_serve.h
// does the device copies.
#pragma once
#include "batch_conversation_cache.h"
#include "../runtime/state_disk.h"
#include <cstdio>
#include <string_view>

namespace crucible {
// The settings of the disk identity: the switches, the pack, the state format and the MTP slot rows. A saved state's
// MTP rows must fit the slot that restores it: another MTP bound is another directory.
inline std::vector<std::string> BatchStateDiskSettings(const std::string& checkpoint, int prefill_rows, bool vision,
                                                       int mtp_rows, char** environ_list) {
  auto settings = StateDiskSettings(environ_list, prefill_rows, vision);
  settings.push_back("pack=" + std::filesystem::path(checkpoint).lexically_normal().filename().string());
  settings.push_back("format=" + std::to_string(kStateVersion));
  settings.push_back("kv_code=" + StateKvCode());  // the KV storage code and layout of the K and V rows
  settings.push_back("mtp_rows=" + std::to_string(mtp_rows));
  return settings;
}
// What a request can restore, as BatchPrefixCache::Lookup decides for the host cache.
struct DiskQuery {
  std::span<const int> key;  // the prompt's cache key
  ConversationPolicy policy;
  size_t boundary = 0;       // cache_prefix_tokens
  size_t limit = 0;          // the last prompt-frontier position that this request can use (Lookup's limit)
  int rows = 0;              // the prefill grid
  bool replay = false;       // session replay: generated tails are restorable too
};
// A generated tail leaves at least one prompt row to run; a prompt frontier passes the host cache's own check.
inline bool DiskStateFits(const StateMeta& meta, const DiskQuery& query) {
  const size_t prompt = query.key.size();
  if (!meta.resume.prompt_only) return query.replay && meta.key.size() + 1 <= prompt;
  const size_t stable = query.boundary ? query.boundary : prompt;
  return meta.key.size() <= std::min(stable, query.limit) &&
         BatchCacheConsumerCompatible(meta.resume, prompt, query.boundary, query.limit, query.rows);
}

// A stage reason as one log word: fixed words and numbers only (the reader's "io errno=5" becomes "io_errno_5"), "-"
// for none.
inline std::string StageReason(std::string_view reason) {
  if (reason.empty()) return "-";
  std::string word(reason.substr(0, 24));
  for (char& c : word)
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) c = '_';
  return word;
}
// The serve log line of one stage reply: the prompt length, the floor (what the slots and the host
// cache restore), the best disk state that the read takes or that the floor beat (0: none), and the read room. No key
// and no tenant.
// `job`: the server's tag of the turn (its crucible_turn line has it as state_disk_job; 0: an older server).
inline std::string StageLine(const StageStatus& status, size_t prompt, size_t floor, uint64_t room, uint64_t job = 0) {
  char line[256];
  std::snprintf(line, sizeof line, "state_disk stage state=%s tokens=%zu floor=%zu best=%zu room=%llu reason=%s job=%llu\n",
                StageName(status.state), prompt, floor, status.tokens, static_cast<unsigned long long>(room),
                StageReason(status.reason).c_str(), static_cast<unsigned long long>(job));
  return line;
}

// An earlier change: a read is host memory inside the conversation cache budget (BatchConversationCache::MakeRoom), and
// the last guard keeps MemAvailable at or above kStageKeepFree after the read: the floor of the daily engine's rollback
// watcher and of the soak gate (10 GiB now; 12 GiB before). The guard is not the host reserve
// (the saves' rule, 8 GiB by default) and not a cache budget; a read that it refuses says "room".
inline constexpr uint64_t kStageKeepFree = uint64_t(10) << 30;  // 10 GiB
inline constexpr uint64_t kStageCap = uint64_t(16) << 30;  // all staged reads together
// The bytes that staged reads may hold in all: `staged` (held now) plus what MemAvailable (`available`, with the idle
// chunk pool) leaves above the larger of the host reserve and kStageKeepFree. `unallocated`: the bytes of reads that
// still read and have not taken their memory (StateDisk::UnallocatedReadBytes): MemAvailable still shows them.
inline uint64_t StageRoom(uint64_t staged, uint64_t available, uint64_t reserve, uint64_t unallocated = 0) {
  const uint64_t keep = std::max(reserve, kStageKeepFree);
  available -= std::min(available, unallocated);
  return std::min(kStageCap, staged + (available > keep ? available - keep : 0));
}
// The admit step of a new read: its room, or why it may not start ("budget", "room").
struct StageAdmit {
  uint64_t room = UINT64_MAX;
  const char* refused = nullptr;
  bool wait = false;  // the read waits for host memory that evicted entries free when the disk writer ends
};
// The last guard of a new read of `bytes` (design decision, Oct 20): start it when the room holds it on the real MemAvailable
// (StageRoom, without the reads that have not allocated); wait while the entries that its eviction gave to the disk
// writer (`pending`, freed when written) would make the room; else refuse ("room"). Freed bytes never count before
// they are free, so MemAvailable stays at kStageKeepFree or more during the overlap.
enum class ReadRoomStep { Start, Wait, Refuse };
inline ReadRoomStep ReadRoom(uint64_t staged, uint64_t available, uint64_t reserve, uint64_t unallocated,
                             uint64_t pending, uint64_t bytes) {
  if (staged + bytes <= StageRoom(staged, available, reserve, unallocated)) return ReadRoomStep::Start;
  return pending && staged + bytes <= StageRoom(staged, available + pending, reserve, unallocated) ? ReadRoomStep::Wait
                                                                                                  : ReadRoomStep::Refuse;
}

// An earlier change: the long turns that wait for a slot (each one's stage commands name its place in the server's FIFO)
// and the host resume points that a read took from them ("read ahead"). A read for a turn may take the live resume
// point of a session only when that session's turn waits behind it; the point goes to disk, and the session's own turn
// reads it back. Tags link each taken point to its restore (or its miss) in the serve log: counters, no key.
class BatchReadQueue {
 public:
  // A waiting turn stages at least every 20 s (server STAGE_REFRESH_SECONDS); a turn not seen for this long has left.
  static constexpr double kWaitSeconds = 25;  // review MEDIUM-3: a place older than one refresh and 5 s is old
  static constexpr size_t kMaxTaken = 64;  // taken points whose turn never came (a session that ended) leave first
  void Saw(std::span<const int> key, size_t position, double now) {
    std::erase_if(waiting_, [&](const Queued& w) { return now - w.seen > kWaitSeconds; });
    for (auto& w : waiting_)
      if (std::ranges::equal(w.key, key)) {
        w.position = position, w.seen = now;
        return;
      }
    waiting_.push_back({std::vector<int>(key.begin(), key.end()), position, now});
  }
  // The turn of `key` opened: it waits no more.
  void Opened(std::span<const int> key) {
    std::erase_if(waiting_, [&](const Queued& w) { return std::ranges::equal(w.key, key); });
  }
  // May a read for `reader` (at `position`) take the host point `point`? Only the point of a turn that waits behind
  // the reader, and never a prefix of the reader's own prompt (its own state, or a prefix that it shares).
  // Only the next turn (`position` 0) takes a point: a place above 0 can be older than the other turns' places.
  bool Behind(std::span<const int> point, std::span<const int> reader, size_t position, double now) const {
    if (position != 0 || Prefix(point, reader)) return false;
    return std::any_of(waiting_.begin(), waiting_.end(), [&](const Queued& w) {
      return w.position > position && now - w.seen <= kWaitSeconds && !std::ranges::equal(w.key, reader) &&
             Prefix(point, w.key);
    });
  }
  // A read took `point`: its tag (1, 2, ...).
  uint64_t Took(std::span<const int> point) {
    if (taken_.size() >= kMaxTaken) taken_.erase(taken_.begin());
    taken_.push_back({std::vector<int>(point.begin(), point.end()), ++tags_});
    return tags_;
  }
  // A disk restore of `key`: the tag of the taken point with this key (0: none); it leaves.
  // A longer state of the same session (the key extends the point) also counts (review MEDIUM-5 c).
  uint64_t Restored(std::span<const int> key) { return Leave([&](const Taken& t) { return Prefix(t.key, key); }); }
  // An open of `prompt` with no disk restore of its taken point: the tag and length of a taken point that the prompt
  // repeats (0: none); it leaves.
  std::pair<uint64_t, size_t> Missed(std::span<const int> prompt) {
    size_t tokens = 0;
    const uint64_t tag = Leave([&](const Taken& t) { return Prefix(t.key, prompt) && (tokens = t.key.size(), true); });
    return {tag, tokens};
  }
  // The tag of a taken point that `prompt` repeats (0: none). It stays.
  uint64_t TagOf(std::span<const int> prompt) const {
    const auto found = std::find_if(taken_.begin(), taken_.end(), [&](const Taken& t) { return Prefix(t.key, prompt); });
    return found == taken_.end() ? 0 : found->tag;
  }
  size_t Waiting() const { return waiting_.size(); }

 private:
  struct Queued { std::vector<int> key; size_t position; double seen; };
  struct Taken { std::vector<int> key; uint64_t tag; };
  static bool Prefix(std::span<const int> prefix, std::span<const int> key) {
    return prefix.size() <= key.size() && std::equal(prefix.begin(), prefix.end(), key.begin());
  }
  template<class Match>
  uint64_t Leave(Match match) {
    const auto found = std::find_if(taken_.begin(), taken_.end(), match);
    if (found == taken_.end()) return 0;
    const uint64_t tag = found->tag;
    taken_.erase(found);
    return tag;
  }
  std::vector<Queued> waiting_;
  std::vector<Taken> taken_;
  uint64_t tags_ = 0;
};

template<class Snapshot>
class BatchDiskTier {
 public:
  using Ready = typename StateDisk<Snapshot>::ReadyState;
  BatchDiskTier(StateDiskOptions options, bool compact) : disk_(std::move(options)), compact_(compact) {}
  bool Enabled() const { return disk_.Enabled(); }
  bool Compact() const { return compact_; }
  StateDisk<Snapshot>& Disk() { return disk_; }

  // A host eviction (BatchConversationCache::SetSpill). Only queues: the writer thread does the disk work. `room`: the
  // host memory that the queued state may hold, less the reads that Stage reserved and the reader has not
  // allocated yet. Returns whether the writer holds the entry.
  // `read` (fix 2, item 9): a victim of a read's eviction (BatchConversationCache::MakeRoom). Its bytes are in host
  // memory already and the read waits until the writer frees them (ReadRoom), so the queue limit does not refuse it
  // ("busy" lost the read-ahead point of x6h request 19). LastSpill: the reason of the last refusal ("" when queued).
  template<class Entry>
  bool Spill(std::span<const int> key, const ConversationPolicy& policy, const std::shared_ptr<const Entry>& entry,
             uint64_t room = UINT64_MAX, bool read = false) {
    const uint64_t reading = disk_.UnallocatedReadBytes();
    room = room == UINT64_MAX ? room : room > reading ? room - reading : 0;
    last_spill_ = disk_.Spill(key, policy, compact_, entry->retention == CacheRetention::Shared, entry->guarded, entry,
                              *entry->snapshot, false, room, !read);
    return last_spill_.empty();
  }
  const std::string& LastSpill() const { return last_spill_; }
  // A state of this key is on disk or waits for the writer: an eviction of its host copy loses nothing.
  bool Holds(std::span<const int> key, const ConversationPolicy& policy) const {
    return disk_.Present(key, policy, compact_) || disk_.Waiting(key, policy, compact_);
  }
  // A turn waits at most this long for a state that the writer has not written yet (fix 2): a write of a 4.5 GB
  // state takes about 2 s, and the queue holds a few of them; an older queued state is a stuck writer.
  static constexpr double kPendingSeconds = 120;
  // Starts the read of the longest disk state for `query` when it is longer than `floor` (what the slots and the
  // host cache hold). A None status names why (StageLine): "off", "nostate" (no disk state fits and the floor is 0),
  // "floor" (`tokens`: the best disk state, which the floor holds already; 0: no disk state, but a slot or the host
  // cache holds a state for this prompt), the admit's refusal, or "room" (the read room). A new read says "start"; a
  // poll of a read that the turn holds reports it and never starts a second read.
  // Fix 2 (cause.md items 1, 2, 5): a longer state that the writer has not written yet (a direct spill of the turn's
  // own slot state) says Reading "writing": the server waits as for a read and polls again; the poll after the write
  // starts the read of the written state.
  // `admit(meta)`: called once before a new read; makes room for it (the host cache evicts) and gives the read room.
  template<class Admit>
  StageStatus StageWith(const DiskQuery& query, size_t floor, Admit admit) {
    if (!Enabled()) return {StageState::None, 0, "off"};
    const auto fits = [&](const StateMeta& m) { return DiskStateFits(m, query); };
    if (auto held = disk_.PollStaged(query.key, query.policy, compact_, floor, fits)) return *held;
    const auto meta = disk_.Best(query.key, query.policy, compact_, fits);
    const auto pending = disk_.Pending(query.key, query.policy, compact_, fits, kPendingSeconds);
    if (pending && pending->key.size() > floor && (!meta || pending->key.size() > meta->key.size()))
      return {StageState::Reading, pending->key.size(), "writing"};
    if (!meta) return {StageState::None, 0, floor ? "floor" : "nostate"};
    if (meta->key.size() <= floor) return {StageState::None, meta->key.size(), "floor"};
    const StageAdmit admitted = admit(*meta);
    if (admitted.refused) return {StageState::None, meta->key.size(), admitted.refused};
    // The server waits as for a read and polls again; no read is staged yet (the next poll admits again).
    if (admitted.wait) return {StageState::Reading, meta->key.size(), "freeing"};
    auto status = disk_.Stage(*meta, admitted.room);
    if (status.state == StageState::None && status.reason == "memory") status.reason = "room";
    if (status.state == StageState::Reading && status.reason.empty()) status.reason = "start";
    return status;
  }
  // As StageWith, with a fixed read room and no admit step.
  StageStatus Stage(const DiskQuery& query, size_t floor, uint64_t room) {
    return StageWith(query, floor, [room](const StateMeta&) { return StageAdmit{room}; });
  }
  // The longest read state for `query`, if longer than `beat` (the host cache's state). Call Consume after the
  // restore.
  Ready Take(const DiskQuery& query, size_t beat) {
    if (!Enabled()) return {};
    auto ready = disk_.Ready(query.key, query.policy, compact_, [&](const StateMeta& m) { return DiskStateFits(m, query); });
    if (!ready.snapshot || ready.meta.key.size() <= beat) return {};
    return ready;
  }
  // The length that Take would give (0: none).
  size_t Peek(const DiskQuery& query) {
    const auto ready = Take(query, 0);
    return ready.snapshot ? ready.meta.key.size() : 0;
  }
  void Consume(const Ready& ready) { disk_.Consume(ready); }
  // A stopping engine: every host entry that is not superseded and is long enough goes to disk. Returns the count
  // queued; the caller then drains.
  template<class Cache>
  size_t FlushHost(Cache& cache) {
    size_t queued = 0;
    cache.ForEachEntry([&](std::span<const int> key, const ConversationPolicy& policy, const auto& entry, bool superseded) {
      if (superseded || key.size() < disk_.MinTokens()) return;
      queued += disk_.Spill(key, policy, compact_, entry->retention == CacheRetention::Shared, entry->guarded, entry,
                            *entry->snapshot, true).empty();
    });
    return queued;
  }
  // A stopping engine: one slot state, saved to host memory by the caller. Waits for its write, so only one such
  // state holds extra host memory at a time.
  bool FlushOne(std::span<const int> key, const ConversationPolicy& policy, bool guarded,
                const std::shared_ptr<const Snapshot>& snapshot, double seconds) {
    if (key.size() < disk_.MinTokens()) return false;
    const auto result = disk_.Spill(key, policy, compact_, false, guarded, snapshot, *snapshot, true);
    return (result.empty() || result == "queued") ? disk_.Drain(seconds) : result == "present";
  }
  bool Drain(double seconds) { return disk_.Drain(seconds); }
  // MEDIUM-5: a slot state that the host cache refused goes straight
  // to disk, else the state is lost. `refused`: the cache's reason; a state that the cache cannot hold for lack of
  // room or memory goes, a policy refusal does not (and a resident copy stays in its slot). `bytes`: the whole state
  // (the daily check counts the whole size, as Skip); `need`: the host memory that its save takes (its bytes less
  // the chunks that it shares with the slot's base, an earlier change); `room`: the host memory above the reserve. `save` makes
  // the snapshot (the device copy) only when the writer can take it; the writer holds it until its write ends.
  // Returns "" for a refusal that the disk does not take, "queued", or why not (Skip, waiting: the same state waits
  // for the writer already, memory, busy, Spill).
  template<class Save>
  std::string SpillDirect(std::span<const int> key, const ConversationPolicy& policy, bool guarded,
                          std::string_view refused, size_t bytes, size_t need, uint64_t room, Save save) {
    if (!Enabled() || !DirectRefusal(refused)) return {};
    if (key.size() < disk_.MinTokens()) return "short";
    if (auto skip = Skip(key, policy, bytes); !skip.empty()) return skip;
    if (disk_.Waiting(key, policy, compact_)) return "waiting";
    const uint64_t reading = disk_.UnallocatedReadBytes();
    if (need > (room > reading ? room - reading : 0)) return "memory";
    if (disk_.QueueFull(bytes, key, policy, compact_)) return "busy";
    std::shared_ptr<const Snapshot> snapshot;
    try {
      snapshot = save();
    } catch (const std::bad_alloc&) {
      return "memory";
    }
    if (!snapshot) return "memory";
    // The room check above used `need`, not the whole snapshot: its shared chunks are in host memory already.
    const auto result = disk_.Spill(key, policy, compact_, false, guarded, snapshot, *snapshot, false);
    return result.empty() ? "queued" : result == "queued" ? "waiting" : result;
  }
  static bool DirectRefusal(std::string_view refused) {
    return refused == "protected" || refused == "host" || refused == "memory" || refused == "budget";
  }
  // Why a stop flush of a slot state of `bytes` would write nothing ("present": its file is on disk; "daily": the
  // daily write budget), or "" when it would write. The caller then skips the device copy.
  std::string Skip(std::span<const int> key, const ConversationPolicy& policy, size_t bytes) {
    if (disk_.Present(key, policy, compact_)) return "present";
    if (disk_.OverDailyBudget(bytes)) return "daily";
    return {};
  }

 private:
  StateDisk<Snapshot> disk_;
  const bool compact_;
  std::string last_spill_;
};
}  // namespace crucible
