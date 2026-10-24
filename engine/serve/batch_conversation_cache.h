// One device worker owns this LRU. Retention classes never weaken consumer compatibility.
#pragma once
#include "../runtime/conversation_cache.h"
#include "../runtime/conversation_admission.h"
#include "../runtime/conversation_resume.h"
#include "cache_status.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace crucible {
enum class CacheRetention { Session, Shared };
// `speculative`: a grid point past the stable boundary, which only a later prompt that repeats it can use.
struct CacheFrontier {
  size_t position = 0;
  CacheRetention retention = CacheRetention::Session;
  bool speculative = false;
};
enum class SnapshotMemoryAction { Allocate, Free };
struct SnapshotMemoryLedger {
  virtual ~SnapshotMemoryLedger() = default;
  virtual void Update(SnapshotMemoryAction action, size_t bytes) noexcept = 0;
};
// A shared system-and-tools frontier sits on the same fixed grid, so another session's chunks
// before it are identical and its restore stays exact (E6).
inline std::vector<CacheFrontier> BatchCacheFrontiers(size_t length, size_t boundary, int rows, size_t shared = 0) {
  std::vector<CacheFrontier> frontiers;
  const size_t own = ConversationAdmission::ForPrompt(length, boundary, rows, 0).frontier;
  const size_t common = shared ? ConversationAdmission::ForPrompt(length, std::min(shared, boundary ? boundary : length),
                                                                  rows, 0).frontier : 0;
  if (common && common != own) frontiers.push_back({common, CacheRetention::Shared});
  if (own) frontiers.push_back({own, common == own ? CacheRetention::Shared : CacheRetention::Session});
  // A grid point inside the generation prompt: the next turn repeats the header but its last token (the reasoning
  // marker), so that state saves one chunk of prefill (an earlier soak run). Grid states are exact for any prompt
  // that repeats them; the stable boundary only predicts which states a later prompt repeats.
  const size_t header = rows > 0 && length ? (length - 1) / size_t(rows) * size_t(rows) : 0;
  if (boundary && header > own && header > common) frontiers.push_back({header, CacheRetention::Session, true});
  return frontiers;
}
// A prompt that repeats part of a cached tail but not all of it. Counts only, never tokens or text.
// `shorter`: the prompt is not longer than the tail (an edit or a retry), not a turn that follows it.
struct TailDivergence { size_t tail = 0, common = 0; bool shorter = false; };
inline std::string SessionReplayDivergenceLine(size_t prompt, size_t restored, const TailDivergence& missed) {
  char line[160];
  std::snprintf(line, sizeof line, "session_replay %s prompt=%zu restored=%zu tail=%zu first_mismatch=%zu\n",
                missed.shorter ? "shorter" : "diverged", prompt, restored, missed.tail, missed.common);
  return line;
}
// The resume record of a prompt frontier at `cursor` for a request with stable boundary `boundary`. A header
// frontier passes the boundary; its own check then uses the whole prompt (Capture checks it).
inline ConversationResume BatchFrontierResume(size_t cursor, size_t boundary) {
  ConversationResume state;
  state.length = cursor;
  state.prompt_only = true;
  state.prompt_boundary = cursor > boundary ? 0 : boundary;
  return state;
}
// The MTP state that an entry may hold: none (target state only), or the MTP rows of the whole prefix with the carry
// (mpos = length - 1). A gap or pending pairs is a fault.
// The key stays the tokens: drafts change only speed, so they must not decide whether a later turn hits.
inline bool BatchCacheMtpState(const ConversationResume& resume) {
  const bool empty = resume.mpos == 0 && !resume.have_carry && resume.pending_tokens.empty();
  const bool held = resume.length > 0 && resume.have_carry && resume.mpos == int(resume.length) - 1 &&
                    resume.pending_tokens.empty();
  return empty || held;
}
inline bool BatchCacheConsumerCompatible(const ConversationResume& resume, size_t length,
                                         size_t boundary, size_t warmup_start, int rows) {
  const size_t stable = boundary ? boundary : length;
  return rows > 0 && resume.prompt_only && resume.length > 0 && resume.length <= stable &&
         resume.length <= warmup_start && resume.length <= length && resume.length % size_t(rows) == 0 &&
         ConversationPromptCompatible(resume, length, boundary, rows);
}
// Why this saved state cannot serve a prompt of `length` rows (the turn line's miss_reason); "" when it
// can. The words are the checks of BatchCacheConsumerCompatible, one at a time, so a miss names the check that
// refused the state.
inline const char* BatchCacheRejectWord(const ConversationResume& resume, size_t length, size_t boundary,
                                        size_t warmup_start, int rows) {
  const size_t stable = boundary ? boundary : length;
  if (rows <= 0) return "rows";
  if (!resume.prompt_only) return "tail";  // a generated tail: only session replay (FindTail) restores it
  if (!resume.length) return "empty";
  if (resume.length > length) return "above_prompt";
  if (resume.length > std::min(stable, warmup_start)) return "above_limit";
  if (resume.length % size_t(rows)) return "off_grid";
  return ConversationPromptCompatible(resume, length, boundary, rows) ? "" : "schedule";
}

// Find's rule for a host frontier of `length` (saved with stable boundary `saved_boundary`) and a prompt of
// `length_of_prompt` rows with boundary `boundary` and restore limit `warmup_start` (BatchRestoreLimit). A Frontier
// point resumes only where Find would restore a host frontier of its length.
inline bool SlotFrontierFits(size_t length, size_t saved_boundary, size_t length_of_prompt, size_t boundary,
                             size_t warmup_start, int rows) {
  const size_t stable = boundary ? boundary : length_of_prompt;
  return length <= std::min(stable, warmup_start) &&
         BatchCacheConsumerCompatible(BatchFrontierResume(length, saved_boundary), length_of_prompt, boundary,
                                      warmup_start, rows);
}
// The default MemAvailable that a save keeps free (the server's --host-reserve-mib).
inline constexpr uint64_t kDefaultHostReserve = uint64_t(8) << 30;
// a save that shares host chunks of a held snapshot needs only `host_bytes` of new host memory (0: all its
// bytes). It keeps the chunks `kept` (SnapshotBytes::ShareableOwners; the caller holds their snapshot until the save
// ends), so an eviction of an entry that uses them does not free them.
struct SharedSave {
  size_t host_bytes = 0;
  std::vector<const void*> kept;
};
template<class Snapshot>
class BatchConversationCache {
  struct MemoryLease {
    const std::shared_ptr<SnapshotMemoryLedger> ledger;
    const size_t bytes;
    MemoryLease(std::shared_ptr<SnapshotMemoryLedger> ledger, size_t bytes) : ledger(std::move(ledger)), bytes(bytes) {
      if (this->ledger) this->ledger->Update(SnapshotMemoryAction::Allocate, bytes);
    }
    ~MemoryLease() { if (ledger) ledger->Update(SnapshotMemoryAction::Free, bytes); }
  };
 public:
  struct Entry {
    std::shared_ptr<const MemoryLease> memory;
    std::shared_ptr<const Snapshot> snapshot;
    CacheRetention retention;
    bool guarded = false;  // a tail of a guarded request (reasoning): a replay from it shows the client keeps it
    // Lineage for the replay diagnostic: this entry's number, and the number of the longest entry that its tokens
    // extended when it was saved (0: none). A tail's parent is the state that its request restored or passed.
    uint64_t serial = 0, parent = 0;
    // The key's leading client-prompt tokens: a frontier (and a shared prefix) is prompt tokens only, while a tail
    // holds its request's prompt and then generated tokens. The miss diagnostic tells a prompt difference from a
    // reply difference with it.
    size_t prompt = 0;
  };
  // The eviction hook holds `this`, and a copy would point at the dead original (L7, review): no copies.
  BatchConversationCache(const BatchConversationCache&) = delete;
  BatchConversationCache& operator=(const BatchConversationCache&) = delete;
  // `live`: sessions that can run at once (the slot count). Each keeps one protected resume point.
  // `probation`: a new entry leaves first and is never a protected resume point until a later request uses it or
  // its branch (a hit, a slot resume, a trusted tail, or a save that extends a used state). One-time requests (a Pi
  // compaction summary: one long prompt that never continues) then cannot push out live sessions. Speed only.
  explicit BatchConversationCache(size_t budget = 0, std::shared_ptr<SnapshotMemoryLedger> ledger = {},
                                  size_t live = kMaxRows, uint64_t host_reserve = kDefaultHostReserve,
                                  bool probation = true, double idle_seconds = kDefaultIdleSeconds)
      : ledger_(std::move(ledger)), cache_(budget), live_(std::max<size_t>(live, 1)), host_reserve_(host_reserve),
        probation_(probation), idle_seconds_(idle_seconds) {
    // An earlier change: every eviction is remembered, so a turn that finds nothing can name a state that left the cache,
    // also without a disk tier. The spill hook (the disk tier) is optional and only adds the disk's copy.
    cache_.SetEvicted([this](const Item& item, int tier) { OnEvicted(item, tier); });
  }
  // Tests set the wall clock (seconds); the default is the steady clock.
  void SetClock(std::function<double()> clock) { clock_now_ = std::move(clock); }
  uint64_t HostReserve() const { return host_reserve_; }
  std::shared_ptr<const Entry> Find(std::span<const int> ids, const ConversationPolicy& policy,
                                   size_t boundary, size_t warmup_start, int rows) {
    const size_t stable = boundary ? boundary : ids.size();
    return Confirm(cache_.FindEntry(ids, policy, std::min(stable, warmup_start), [&](const Entry& entry) {
      return BatchCacheConsumerCompatible(entry.snapshot->resume, ids.size(), boundary, warmup_start, rows);
    }));
  }
  // Session replay (G15): the longest generated tail this prompt repeats, leaving at least one row to predict.
  std::shared_ptr<const Entry> FindTail(std::span<const int> ids, const ConversationPolicy& policy) {
    if (ids.size() < 2) return {};
    return Confirm(cache_.FindEntry(ids, policy, ids.size() - 1,
                                    [](const Entry& entry) { return !entry.snapshot->resume.prompt_only; }));
  }
  // What Find and FindTail would restore, without a change to the cache (in-slot checkpoints compare against it).
  size_t PeekFind(std::span<const int> ids, const ConversationPolicy& policy, size_t boundary, size_t warmup_start,
                  int rows) const {
    const size_t stable = boundary ? boundary : ids.size();
    return cache_.Longest(ids, policy, std::min(stable, warmup_start), [&](const Entry& entry) {
      return BatchCacheConsumerCompatible(entry.snapshot->resume, ids.size(), boundary, warmup_start, rows);
    });
  }
  size_t PeekTail(std::span<const int> ids, const ConversationPolicy& policy) const {
    if (ids.size() < 2) return 0;
    return cache_.Longest(ids, policy, ids.size() - 1, [](const Entry& entry) { return !entry.snapshot->resume.prompt_only; });
  }
  // The length and pending rows of the tail that FindTail would restore, without a change to the cache ({0, 0}: none).
  std::pair<size_t, int> PeekTailState(std::span<const int> ids, const ConversationPolicy& policy) const {
    std::pair<size_t, int> best{0, 0};
    if (ids.size() < 2) return best;
    cache_.ForEach(policy, [&](std::span<const int> key, const Entry& entry) {
      if (!entry.snapshot->resume.prompt_only && key.size() < ids.size() && key.size() > best.first &&
          std::equal(key.begin(), key.end(), ids.begin()))
        best = {key.size(), entry.snapshot->resume.replay};
    });
    return best;
  }
  // with slot frontiers the host heads are the sessions without a slot. Protect no more of them than the
  // budget holds (Plan), so a handover copy can still evict the oldest waiting one. Off: the live count as before.
  void SetSlotFrontiers(bool on) { slot_frontiers_ = on; }
  // The newest tail saved after the restored entry is this conversation's last answer. A prompt that leaves it lost
  // replay by its own tokens: `common` is the first position where it differs. Compares one tail, from the restore on.
  std::optional<TailDivergence> MissedTail(std::span<const int> ids, const ConversationPolicy& policy,
                                           const Entry* restored) const {
    if (!restored) return {};
    std::span<const int> newest;
    uint64_t serial = 0;
    cache_.ForEach(policy, [&](std::span<const int> tail, const Entry& entry) {
      if (entry.parent == restored->serial && !entry.snapshot->resume.prompt_only && entry.serial > serial)
        newest = tail, serial = entry.serial;
    });
    const size_t start = restored->snapshot->hist.size(), limit = std::min(newest.size(), ids.size());
    if (!serial || start > limit) return {};
    const size_t common = size_t(std::mismatch(newest.begin() + start, newest.begin() + limit, ids.begin() + start).first -
                                 newest.begin());
    if (common >= newest.size()) return {};
    return TailDivergence{newest.size(), common, ids.size() <= newest.size()};
  }
  // A tail holds target state, and from a full-window request also the MTP rows (BatchCacheMtpState). Without them
  // the next turn starts a fresh MTP warm-up, as a prompt restore does.
  // A tail is speculative: the next prompt repeats it only when the client sends back every output token, and many
  // clients drop the reasoning. A speculative tail never pushes out a live session's resume point, its own included.
  // `trusted`: the request itself resumed from its branch's previous tail and its client keeps what this tail holds,
  // so this tail replaces the older entries of its branch, as before.
  // `resident`: a slot still holds this state, so the copy takes only free room or superseded entries; the slot
  // saves it later if it must leave the device (Release in batched_serve.h).
  // `hist` is the cache key. `tokens` is the snapshot's own history when the key differs (image rows key by image).
  template<class Create>
  bool CaptureTail(std::span<const int> hist, const ConversationPolicy& policy, size_t bytes, uint64_t available,
                   Create create, bool trusted = false, bool guarded = false, bool resident = false,
                   std::span<const int> tokens = {}, SharedSave share = {}, size_t prompt = 0) {
    if (tokens.empty()) tokens = hist;
    const size_t prompt_tokens = prompt && prompt < hist.size() ? prompt : hist.size();
    if (tokens.size() != hist.size()) throw std::invalid_argument("tail key and tokens differ in length");
    Try(hist.size());
    if (policy.spec != 0 || policy.mtp_prefix != 0 || policy.prefill_rows < 1 || hist.empty()) return Refuse("policy");
    if (!bytes || bytes > cache_.Budget()) return Refuse("budget");
    const uint64_t parent = Parent(hist, policy);  // before eviction can remove it
    const bool proven = ExtendsProven(hist, policy);  // also before eviction
    const auto marked = trusted ? Supersede(hist, policy, [](const Entry& entry) {
      return entry.retention == CacheRetention::Session;
    }) : std::set<const Item*>{};
    if (!Admit(bytes, available, resident, !trusted, share)) return Unmark(marked, refusal_);
    try {
      auto lease = std::make_shared<MemoryLease>(ledger_, bytes);
      auto snapshot = create();
      if (!snapshot || snapshot->hist.size() != hist.size() ||
          !std::equal(snapshot->hist.begin(), snapshot->hist.end(), tokens.begin()) ||
          snapshot->resume.prompt_only || snapshot->resume.length != hist.size() ||
          snapshot->resume.replay < 0 || snapshot->resume.replay > kMaxRows ||
          !BatchCacheMtpState(snapshot->resume))
        throw std::runtime_error("session replay needs a generated tail with target state or full MTP rows");
      auto entry = std::make_shared<Entry>(Entry{std::move(lease), std::move(snapshot), CacheRetention::Session, guarded,
                                                 ++serials_, parent, prompt_tokens});
      return Saved(cache_.Insert(hist, policy, bytes, std::move(entry)), trusted, trusted || proven);
    } catch (const std::bad_alloc&) {
      return Unmark(marked, "memory");
    }
  }
  // `ids` is the cache key; `tokens` as in CaptureTail. `proven`: the state extended a proven state when the slot
  // took it (the entry that it extended may be gone by now).
  template<class Create>
  bool Capture(std::span<const int> ids, const ConversationPolicy& policy, const CacheFrontier& frontier,
               size_t bytes, uint64_t available, Create create, std::span<const int> tokens = {},
               SharedSave share = {}, bool proven_before = false) {
    if (tokens.empty()) tokens = ids;
    if (tokens.size() != ids.size()) throw std::invalid_argument("prompt key and tokens differ in length");
    Try(frontier.position);
    if (policy.spec != 0 || policy.mtp_prefix != 0 || policy.prefill_rows < 1 ||
        !frontier.position || frontier.position > ids.size() || frontier.position % size_t(policy.prefill_rows))
      return Refuse("policy");
    if (!bytes || bytes > cache_.Budget()) return Refuse("budget");
    // Older Session frontiers that this one extends leave first.
    // Shared frontiers stay in plain LRU: other sessions can still restore them.
    // A tail is never superseded by a prompt frontier: replay makes it the most valuable entry (G15).
    // A speculative frontier (past the boundary) is a speculative tail: a retry of this prompt, or a next turn that
    // drops the reasoning, cannot use it. It marks nothing until a hit confirms it.
    const auto marked = frontier.speculative ? std::set<const Item*>{} :
        Supersede(ids.first(frontier.position), policy, [](const Entry& entry) {
          return entry.retention == CacheRetention::Session && entry.snapshot->resume.prompt_only;
        });
    // Before Admit: the state that this frontier extends is superseded now, so Admit evicts it first.
    const bool proven = proven_before || ExtendsProven(ids.first(frontier.position), policy);
    if (!Admit(bytes, available, false, frontier.speculative, share)) return Unmark(marked, refusal_);
    try {
      auto lease = std::make_shared<MemoryLease>(ledger_, bytes);
      auto snapshot = create();
      if (!snapshot || snapshot->hist.size() != frontier.position ||
          !std::equal(snapshot->hist.begin(), snapshot->hist.end(), tokens.begin()) ||
          !BatchCacheConsumerCompatible(snapshot->resume, ids.size(), snapshot->resume.prompt_boundary,
                                        ids.size(), policy.prefill_rows) ||
          snapshot->resume.length != frontier.position || snapshot->resume.replay ||
          !BatchCacheMtpState(snapshot->resume))
        throw std::runtime_error("batch capture needs a committed frontier with target state or full MTP rows");
      auto entry = std::make_shared<Entry>(Entry{std::move(lease), std::move(snapshot), frontier.retention, false,
                                                 ++serials_, 0, frontier.position});
      // A prompt frontier holds client tokens, so it is a confirmed resume point; a speculative one is not yet.
      return Saved(cache_.Insert(ids.first(frontier.position), policy, bytes, std::move(entry)), !frontier.speculative,
                   proven);
    } catch (const std::bad_alloc&) {
      return Unmark(marked, "memory");
    }
  }
  // a slot keeps the session frontier `ids` instead of a host copy. The older Session frontiers that it
  // extends are superseded as a Capture of it would mark them (they leave first and never spill); a speculative
  // frontier marks nothing, as in Capture.
  void KeepInSlot(std::span<const int> ids, const ConversationPolicy& policy, bool speculative) {
    if (!speculative)
      Supersede(ids, policy, [](const Entry& entry) {
        return entry.retention == CacheRetention::Session && entry.snapshot->resume.prompt_only;
      });
  }
  // Whether `ids` extends a proven state: the slot asks when it keeps a frontier, for its host copy later.
  bool ProvenBelow(std::span<const int> ids, const ConversationPolicy& policy) { return ExtendsProven(ids, policy); }
  // A slot resume copies nothing from the host, so it marks the session's newest host entry that the prompt extends
  // as used: that session stays live.
  void Touch(std::span<const int> ids, const ConversationPolicy& policy) {
    Item* best = nullptr;
    const double now = Now();
    cache_.ForEach([&](Item& item) {
      if (item.policy != policy || item.ids.size() > ids.size() || !std::equal(item.ids.begin(), item.ids.end(), ids.begin()))
        return;
      // Every host entry of the session is in use: the longest one is often the slot's copy, which is never a head,
      // so a head that only the copy refreshed would go stale while its session runs.
      item.active_at = now;
      if (!best || item.ids.size() > best->ids.size()) best = &item;
    });
    if (best) best->used = ++tick_, best->active = ++clock_, best->proven = true;
  }
  // Whether the entry with exactly this key has left probation.
  bool Proven(std::span<const int> ids, const ConversationPolicy& policy) {
    bool proven = false;
    cache_.ForEach([&](Item& item) { proven |= item.policy == policy && item.proven && std::ranges::equal(item.ids, ids); });
    return proven;
  }
  // The tail state that slot `slot` holds ("One copy"). A host entry of that state is
  // a copy: it leaves before any state that exists only on the host, and it is never a protected resume point.
  // `policy` holds the tenant: a slot's tail makes a copy only of its own tenant's entry.
  void SetResident(int slot, std::span<const int> hist, int replay, const ConversationPolicy& policy) {
    resident_[slot] = {std::vector<int>(hist.begin(), hist.end()), replay, policy};
  }
  void ClearResident(int slot) { resident_.erase(slot); }
  bool HoldsTail(std::span<const int> hist, const ConversationPolicy& policy, int replay) {
    bool held = false;
    cache_.ForEach([&](Item& item) {
      held |= item.policy == policy && !item.snapshot->snapshot->resume.prompt_only &&
              item.snapshot->snapshot->resume.replay == replay && std::ranges::equal(item.ids, hist);
    });
    return held;
  }
  // Host bytes of tail entries that a slot also holds.
  size_t CopyBytes() {
    size_t bytes = 0;
    cache_.ForEach([&](Item& item) { if (Copy(item)) bytes += item.bytes; });
    return bytes;
  }
  // The disk tier (state_disk.h): `spill` receives each evicted entry of the classes kStale to kWaiting (a state
  // that exists nowhere else) with its key. A superseded entry and a copy of a slot's state stay off the disk.
  // The writer holds a spilled entry in host memory until its write ends, so it frees nothing for the save that
  // evicted it. `room` is the host memory above the reserve that this save leaves, less the entries that the
  // hook already holds; the hook returns whether it holds this one.
  using Spill = std::function<bool(std::span<const int> key, const ConversationPolicy&,
                                   const std::shared_ptr<const Entry>&, int tier, uint64_t room)>;
  void SetSpill(Spill spill) { spill_ = std::move(spill); }
  // Every entry with its key and superseded mark, the most recently used first (a stopping engine writes them to
  // disk in this order, so a flush that runs out of time loses the oldest).
  template<class Visit>
  void ForEachEntry(Visit visit) {
    std::vector<Item*> items;
    cache_.ForEach([&](Item& item) { items.push_back(&item); });
    std::stable_sort(items.begin(), items.end(), [](const Item* a, const Item* b) { return a->used > b->used; });
    for (Item* item : items) visit(std::span<const int>(item->ids), item->policy, item->snapshot, item->superseded);
  }
  size_t Bytes() const { return cache_.Bytes(); }
  size_t Budget() const { return cache_.Budget(); }
  // An earlier change: a disk read for a waiting turn uses host memory inside this budget. `outside` gives the bytes of
  // the reads (StateDisk::StagedBytes); every save then makes room for them. MakeRoom evicts for a new read of `bytes`
  // as for the save of a client prompt's state (Admit: victims go to the spill hook); false with LastRefusal
  // "protected" or "host" when it cannot. A live resume point (kKept) leaves only when `read_ahead(key)` allows it (the
  // point of a session that waits behind the reading turn, batched_serve.h Stage), after every other victim;
  // `evicted(key, bytes, spilled)` then reports it, with whether the spill hook took it, and returns whether the point
  // may stay out (its state is on disk or queued). Fix 2 (cause.md item 8): a point that may not stay out goes back
  // into the cache with its marks, and MakeRoom returns false with LastRefusal "spill" (x6h request 19 lost its point).
  using ReadAhead = std::function<bool(std::span<const int> key)>;
  using ReadAheadEvicted = std::function<bool(std::span<const int> key, size_t bytes, bool spilled)>;
  void SetOutside(std::function<size_t()> outside) { cache_.SetOutside(std::move(outside)); }
  // `keep_free`: the host memory that the eviction keeps free when it is more than the host reserve (the read's last
  // guard, kStageKeepFree), so the entries that leave are the ones that the guard needs.
  bool MakeRoom(size_t bytes, uint64_t available, ReadAhead read_ahead = {}, ReadAheadEvicted evicted = {},
                uint64_t keep_free = 0) {
    if (!bytes || bytes > cache_.Budget()) {
      refusal_ = "budget";
      return false;
    }
    read_ahead_ = std::move(read_ahead);
    read_ahead_evicted_ = std::move(evicted);
    keep_free_ = keep_free;
    reading_ = true, read_ahead_kept_ = false;
    const struct Reset {
      BatchConversationCache& cache;
      ~Reset() { cache.read_ahead_ = {}, cache.read_ahead_evicted_ = {}, cache.keep_free_ = 0, cache.reading_ = false; }
    } reset{*this};
    if (!Admit(bytes, available)) return false;
    if (!read_ahead_kept_) return true;
    refusal_ = "spill";
    return false;
  }
  // The spill hook runs for a victim of MakeRoom (a read's eviction), not of a save.
  bool Reading() const { return reading_; }
  // Fix 2 (cause.md item 6): while an open guards its slot's Frontier points and releases its slot's tail, the entries
  // that its own restore may use (every entry whose key starts the prompt key `prompt`) never leave for those saves.
  // Test 217: the handover copy of another session's point evicted the cold session's 49,152-token entry. Empty: off.
  void Hold(std::span<const int> prompt) { held_.assign(prompt.begin(), prompt.end()); }
  // The word of the latest refusal, whatever the caller: "policy", "budget" or "memory" from a save's checks before
  // Admit, "host", "protected" or "resident" from Admit, "spill" from MakeRoom. Each site writes both `refusal_` and
  // `last_.refused`, so the latest event wins (L10, review 2: a refused save shadowed a later read's word, and the
  // stage path then named the wrong reason).
  const char* LastRefusal() const { return refusal_; }
  // A save of this state was refused after eviction (`LastRefusal` names why: host, protected or resident): a later
  // turn of the same conversation that finds nothing then names "not_saved" as its miss cause. A
  // "policy"/"budget"/"spill" refusal never tried to save ("a state of the whole prompt", "larger than the cache"),
  // and a refused speculative tail is not the conversation's state: neither is recorded (L9, review). Diagnostics only.
  void NoteRefused(std::span<const int> key, const ConversationPolicy& policy, size_t prompt = 0) {
    const char* why = LastRefusal();
    if (!(std::strcmp(why, "host") == 0 || std::strcmp(why, "protected") == 0 ||
          std::strcmp(why, "resident") == 0))
      return;
    RememberLost(lost_, key, policy, prompt, Now(), true);
  }
  size_t Evictions() const { return cache_.Evictions(); }
  size_t SupersededEvictions() const { return cache_.SupersededEvictions(); }
  // Rounds of saves and hits (those not on probation) after which a session with no new save or hit counts as idle.
  static constexpr uint64_t kIdleRounds = 16;
  // A resume point that no request used for this long (wall clock) is idle too (a review: with probation, the
  // session clock stops while only one-time requests run, so an abandoned session could stay protected for ever).
  static constexpr double kDefaultIdleSeconds = 1800;
  BatchCacheStatus Status() const { return {cache_.Entries(), cache_.Bytes(), cache_.Evictions(), attempts_, last_, {}, {}}; }
  // Test only: how many entries the current plan protects (L1: 0 outside Admit) and how many the last plan did
  // Both change neither an entry nor an order.
  size_t NewestStates() const { return newest_.size(); }
  size_t LastNewestStates() const { return newest_last_; }
  // Why Find and FindTail took nothing, and the state nearest to this prompt. Diagnostics only: it
  // changes no entry, no mark and no LRU order, so it cannot change what a later restore returns.
  // `warmup_start`: Find's restore limit; `tail`: FindTail was tried too.
  BatchCacheDiagnosis Diagnose(std::span<const int> ids, const ConversationPolicy& policy, size_t boundary,
                               size_t warmup_start, int rows, bool tail) const {
    struct Near {
      size_t shared = 0, key = 0, prompt = 0;
      double at = 0;
      bool lost = false, refused = false;
    };
    const auto better = [](const Near& a, const Near& b) {
      return a.shared > b.shared || (a.shared == b.shared && a.at > b.at);
    };
    Near live, lost;
    const char* rejected = nullptr;
    size_t rejected_key = 0;  // the longest entry whose whole key starts this prompt is the one that Find tried
    cache_.ForEachState(policy, [&](const Item& item) {
      const size_t shared = CommonPrefix(item.ids, ids);
      // An entry whose whole key starts this prompt: Find or FindTail decides it, and their refusal names the miss.
      if (shared == item.ids.size() && item.ids.size() >= rejected_key) {
        const auto& resume = item.snapshot->snapshot->resume;
        const bool tail_takes = tail && !resume.prompt_only && item.ids.size() + 1 <= ids.size();
        const char* word = BatchCacheRejectWord(resume, ids.size(), boundary, warmup_start, rows);
        if (*word && !tail_takes) rejected = word, rejected_key = item.ids.size();
      }
      const Near near{shared, item.ids.size(), item.snapshot->prompt, item.active_at};
      if (better(near, live)) live = near;
    });
    for (const Lost& record : *lost_) {
      if (!(record.policy == policy)) continue;
      const Near near{SharedLost(record, ids), record.length, record.prompt, record.at, true, record.refused};
      if (better(near, lost)) lost = near;
    }
    const Near& win = better(lost, live) ? lost : live;
    BatchCacheDiagnosis out;
    out.shared = win.shared;
    out.age_s = win.at ? std::max(0.0, Now() - win.at) : 0.0;
    if (!win.shared) out.outcome = nullptr;
    else if (win.shared >= win.key) out.outcome = "all";  // the whole kept state is shared
    else if (win.shared >= win.prompt) out.outcome = "reply_differs";
    else out.outcome = "prompt_differs";
    if (rejected) out.miss = rejected;
    else if (win.shared && win.lost) out.miss = win.refused ? "not_saved" : "evicted";
    else if (win.shared) out.miss = "diverged";
    else out.miss = "new";
    return out;
  }
 private:
  using Item = typename PrefixCache<Entry, ConversationPolicy>::Entry;
  // The states that left the host cache at once: the newest evicted entries and refused saves, so a turn
  // that finds nothing can name its cause. A state is kept as a fingerprint, never as its key: the prefix hash at each
  // kLostGrid-token grid point, the hash of the whole key, and its length (L4, review: a 130K-token state is named
  // too, which the earlier whole-key copy could not). The shared length is then reported at grid resolution, and
  // exactly when the lengths are equal. Bounded by kLostStates records and 64 KiB in all. Diagnostics only.
  struct Lost {
    std::vector<uint64_t> hashes;  // the hash of the key's first k, 2k, 3k ... tokens (k = kLostGrid), then the whole key
    ConversationPolicy policy;
    size_t length = 0, prompt = 0;
    double at = 0;      // when the state was last saved or used (an eviction) or when the save was refused
    bool refused = false;
  };
  // Four records, and each key keeps at most 2,048 grid hashes (524,288 tokens); the trim below holds the four together
  // under 32 KiB. A daily state of 130 Ki tokens needs 509 hashes, so a record is a few KiB.
  static constexpr size_t kLostStates = 4, kLostGrid = 256, kLostHashes = 2048;
  static uint64_t Mix(uint64_t hash, int token) noexcept {
    hash ^= static_cast<uint64_t>(static_cast<uint32_t>(token));
    return hash * 1099511628211ull;  // FNV-1a, a fingerprint for equality only
  }
  static constexpr uint64_t kHashSeed = 0xcbf29ce484222325ull;
  // The hash at every grid point and then the hash of the whole key (empty: a key shorter than one grid point, whose
  // whole-key hash is all that a record needs).
  static void Fingerprint(std::span<const int> key, std::vector<uint64_t>& out) {
    uint64_t hash = kHashSeed;
    for (size_t i = 0; i < key.size(); ++i) {
      hash = Mix(hash, key[i]);
      if ((i + 1) % kLostGrid == 0 && out.size() < kLostHashes) out.push_back(hash);
    }
    out.push_back(hash);
  }
  // The shared prefix of a lost state: the grid points that match, then the key's own length (exact, also inside a
  // grid block: a state shorter than one grid point is common). The report is at grid resolution above that.
  static size_t SharedLost(const Lost& lost, std::span<const int> ids) {
    if (lost.hashes.empty()) return 0;
    uint64_t hash = kHashSeed;
    size_t shared = 0;
    const size_t grid = lost.hashes.size() - 1;  // the last entry is the hash of the whole key
    for (size_t point = 0; point < std::min(grid, ids.size() / kLostGrid); ++point) {
      for (size_t i = point * kLostGrid; i < (point + 1) * kLostGrid; ++i) hash = Mix(hash, ids[i]);
      if (hash != lost.hashes[point]) return shared;  // the grid point below the first mismatch
      shared = (point + 1) * kLostGrid;
    }
    if (shared < lost.length && lost.length <= ids.size()) {
      for (size_t i = shared; i < lost.length; ++i) hash = Mix(hash, ids[i]);
      if (hash == lost.hashes.back()) return lost.length;
    }
    return shared;
  }
  static void RememberLost(const std::shared_ptr<std::vector<Lost>>& lost, std::span<const int> key,
                           const ConversationPolicy& policy, size_t prompt, double at, bool refused) noexcept {
    // A diagnostic must never stop a save or a turn (L8, review): a failed allocation drops the record only.
    try {
      if (key.empty()) return;
      std::vector<uint64_t> hashes;
      Fingerprint(key, hashes);
      if (hashes.empty()) return;
      std::erase_if(*lost, [&](const Lost& old) { return old.length == key.size() && old.hashes == hashes; });
      // A save that did not name its prompt is a prompt state: `prompt` 0 would report any difference as a reply one.
      lost->insert(lost->begin(), Lost{std::move(hashes), policy, key.size(), prompt ? prompt : key.size(), at,
                                       refused});
      while (lost->size() > kLostStates) lost->pop_back();
      size_t count = 0;
      for (size_t i = 0; i < lost->size(); ++i) {
        if ((count += (*lost)[i].hashes.size()) > kLostStates * kLostHashes / 2) { lost->resize(i); break; }
      }
    } catch (const std::bad_alloc&) {
      lost->clear();
    }
  }
  // An entry left the cache: remember it for the miss diagnostic, and hand it to the spill hook when there is one.
  // No device state is kept, only the key (a fixed count of them).
  void OnEvicted(const Item& item, int tier) {
    // A superseded entry and a copy of a slot's state leave with a successor in place: they cause no miss. A read-ahead
    // point is not lost either: the hook goes back into the cache when the disk state is not ready (L5, review). The
    // record changes only: the spill hook still gets the tiers that it got before, and not those two.
    if (tier >= kStale && tier != kReadAhead)
      RememberLost(lost_, item.ids, item.policy, item.snapshot->prompt, item.active_at, false);
    if (!spill_ || tier < kStale) return;  // a superseded entry or a copy is not a state that the disk tier keeps
    victims_->push_back({item.ids, item.policy, item.snapshot, tier, Freed(item, *kept_), item.bytes,
                         tier == kReadAhead ? std::optional<Item>(item) : std::nullopt});
  }
  static size_t CommonPrefix(std::span<const int> a, std::span<const int> b) {
    const size_t limit = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < limit && a[i] == b[i]) ++i;
    return i;
  }
  void Try(size_t tokens) { ++attempts_; last_ = {tokens, false, {}}; }
  // The longest entry that `hist` extends. It runs once per tail save, as Supersede does.
  uint64_t Parent(std::span<const int> hist, const ConversationPolicy& policy) const {
    uint64_t parent = 0;
    size_t longest = 0;
    cache_.ForEach(policy, [&](std::span<const int> ids, const Entry& entry) {
      if (ids.size() > longest && ids.size() < hist.size() && std::equal(ids.begin(), ids.end(), hist.begin()))
        parent = entry.serial, longest = ids.size();
    });
    return parent;
  }
  // Whether `ids` extends a proven state. A save asks before Admit, which can evict that state: in a full cache a
  // session's next frontier evicts the frontier that it supersedes, and the new one then landed on probation.
  bool ExtendsProven(std::span<const int> ids, const ConversationPolicy& policy) {
    bool proven = false;
    cache_.ForEach([&](Item& item) {
      proven |= item.proven && item.policy == policy && item.ids.size() < ids.size() &&
                std::equal(item.ids.begin(), item.ids.end(), ids.begin());
    });
    return proven;
  }
  bool Refuse(const char* reason) { last_.refused = reason; refusal_ = reason; return false; }
  // Marks the Session entries that `ids` extends, as PrefixCache::Supersede, and returns the new marks.
  template<class Accept>
  std::set<const Item*> Supersede(std::span<const int> ids, const ConversationPolicy& policy, Accept accept) {
    std::set<const Item*> marked;
    cache_.ForEach([&](Item& item) {
      if (item.superseded || item.policy != policy || item.ids.size() >= ids.size() ||
          !std::equal(item.ids.begin(), item.ids.end(), ids.begin()) || !accept(*item.snapshot)) return;
      item.superseded = true;
      marked.insert(&item);
    });
    return marked;
  }
  // A refused save keeps the marks of no older entry: a refused trusted tail must not demote its session's resume
  // point. Entries that a failed save already evicted are gone from the list, so only live items match.
  bool Unmark(const std::set<const Item*>& marked, const char* reason) {
    if (!marked.empty()) cache_.ForEach([&](Item& item) { if (marked.contains(&item)) item.superseded = false; });
    return Refuse(reason);
  }
  bool Copy(const Item& item) const {
    const auto& resume = item.snapshot->snapshot->resume;
    return !resume.prompt_only && std::any_of(resident_.begin(), resident_.end(), [&](const auto& held) {
      return held.second.replay == resume.replay && held.second.policy == item.policy &&
             std::ranges::equal(held.second.hist, item.ids);
    });
  }
  // `proven`: the request itself continued its branch (a trusted tail). A save that extends a proven state is proven.
  bool Saved(bool saved, bool confirmed, bool proven) {
    // An eviction outside Admit gets no room. None happens today: Insert evicts only tiers 0 and 1, which never spill.
    HandVictims(0);
    if (!saved) return Refuse("budget");
    Item& item = cache_.Newest();
    item.used = ++tick_;
    item.confirmed = confirmed ? tick_ : 0;
    item.proven = proven;
    cache_.ForEach([&](Item& other) { item.proven |= other.proven && Extends(item, other); });
    // A save on probation does not move the session clock: one-time requests never make a live session idle.
    item.active = OnProbation(item) ? clock_ : ++clock_;
    item.active_at = Now();
    return last_.saved = true;
  }
  // Whether `prefix` is a prefix of `longer`. The tail of the shorter key is compared first, at the same offsets (M1,
  // review): two conversations that share a long system prompt then fail in a few compares instead of one per token.
  // The result is exact: a prefix matches at every offset, so a mismatch in its tail proves that it is not a prefix.
  static bool Extends(const Item& longer, const Item& prefix) {
    if (longer.policy != prefix.policy || prefix.ids.size() >= longer.ids.size()) return false;
    const size_t size = prefix.ids.size();
    for (size_t i = size > 8 ? size - 8 : 0; i < size; ++i)
      if (prefix.ids[i] != longer.ids[i]) return false;
    return std::equal(prefix.ids.begin(), prefix.ids.end(), longer.ids.begin());
  }
  // A hit shows the client repeated this state: it becomes its session's resume point, and the Session entries that
  // it extends leave first. A tail stays protected only from its first hit on.
  std::shared_ptr<const Entry> Confirm(Item* hit) {
    if (!hit) return {};
    hit->used = hit->confirmed = ++tick_;
    hit->active = ++clock_;
    hit->active_at = Now();
    hit->proven = true;
    cache_.ForEach([&](Item& item) {
      if (&item != hit && item.snapshot->retention == CacheRetention::Session && Extends(*hit, item))
        item.superseded = true;
    });
    return hit->snapshot;
  }
  // Eviction classes, first to leave first. A session's resume point is its newest
  // confirmed entry that no other confirmed entry extends. The `live_` most recently used of them are live; a resume
  // point not used in the last kIdleRounds * live_ saves and hits is idle.
  enum Tier : int {
    kKept = -1,      // the resume point of a live session: never leaves
    kSuperseded = 0,
    kCopy = 1,       // a copy of a slot's state
    kStale = 2,      // a session that no request used for idle_seconds_ (wall clock): it leaves before new states
    kTrial = 3,      // on probation (no later request used it yet) and speculative (no client prompt showed it)
    kProbation = 4,  // on probation, a client prompt's state (one-time requests)
    kIdle = 5,       // an idle session's entries
    kOverShare = 6,  // extra entries of a session above its fair share
    kExtra = 7,      // other extra entries
    // An earlier change: the rank of the newest saved state of a conversation that had a turn recently (Plan's newest_ set).
    // It is not a class of Plan: the entry keeps its class, and Admit's own guards still read that class (a
    // speculative tail never takes a probation state or a waiting resume point). The rank only orders the victims:
    // that state leaves after every older or idle entry, and a recent session's resume point beyond the live count
    // (kWaiting) still outlasts it. A restore of an entry that stays returns the same state as before.
    kNewest = 8,
    kWaiting = 9,    // the resume point of a recent session beyond the live count (more sessions than slots)
    kReadAhead = 10,  // a live resume point that a disk read for a turn ahead of its session may take (MakeRoom)
  };
  bool OnProbation(const Item& item) const { return probation_ && !item.proven; }
  bool Held(const Item& item) const {
    return !held_.empty() && item.ids.size() <= held_.size() && std::equal(item.ids.begin(), item.ids.end(), held_.begin());
  }
  double Now() const {
    return clock_now_ ? clock_now_()
                      : std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  std::map<const Item*, int> Plan() {
    const double now = Now();
    std::vector<Item*> heads, confirmed;
    std::set<const Item*> copies;
    cache_.ForEach([&](Item& item) { if (Copy(item)) copies.insert(&item); });
    cache_.ForEach([&](Item& item) {
      if (!item.superseded && item.confirmed && !copies.contains(&item) && !OnProbation(item)) confirmed.push_back(&item);
    });
    for (Item* item : confirmed)
      if (std::none_of(confirmed.begin(), confirmed.end(), [&](const Item* other) { return Extends(*other, *item); }))
        heads.push_back(item);
    std::stable_sort(heads.begin(), heads.end(), [](const Item* a, const Item* b) { return a->used > b->used; });
    // See SetSlotFrontiers.
    size_t largest = 0;
    for (const Item* head : heads) largest = std::max(largest, head->bytes);
    const size_t protect = !slot_frontiers_ ? live_ :
        std::min(live_, std::max<size_t>(1, largest ? cache_.Budget() / largest : live_));
    std::map<const Item*, bool> live;
    std::set<const Item*> idle, stale;
    for (size_t i = 0; i < heads.size(); ++i) {
      if (now - heads[i]->active_at > idle_seconds_) stale.insert(heads[i]);
      if (clock_ - heads[i]->active > kIdleRounds * live_ || stale.contains(heads[i])) idle.insert(heads[i]);
      live[heads[i]] = i < protect && !idle.contains(heads[i]);
    }
    // Each other entry belongs to the longest resume point that it extends (none for shared prefixes).
    std::map<const Item*, const Item*> owner;
    std::map<const Item*, size_t> held;
    cache_.ForEach([&](Item& item) {
      const Item* best = nullptr;
      for (const Item* head : heads)
        if (Extends(item, *head) && (!best || head->ids.size() > best->ids.size())) best = head;
      owner[&item] = best;
      held[best ? best : &item] += item.bytes;
    });
    const size_t lives = std::max<size_t>(1, std::count_if(live.begin(), live.end(), [](const auto& h) { return h.second; }));
    const size_t share = cache_.Budget() / lives;
    std::map<const Item*, int> tier;
    cache_.ForEach([&](Item& item) {
      const Item* head = live.contains(&item) ? &item : owner[&item];
      if (item.superseded) tier[&item] = kSuperseded;
      else if (copies.contains(&item)) tier[&item] = kCopy;
      else if (head && stale.contains(head)) tier[&item] = kStale;
      else if (OnProbation(item)) tier[&item] = item.confirmed ? kProbation : kTrial;
      else if (live.contains(&item)) tier[&item] = live[&item] ? kKept : idle.contains(&item) ? kIdle : kWaiting;
      else if (head && idle.contains(head)) tier[&item] = kIdle;
      else tier[&item] = head && held[head] > share ? kOverShare : kExtra;
    });
    // An earlier change: the newest saved state of a conversation that had a turn recently leaves only when no older or idle
    // entry can go instead (kNewest). It is the deepest entry of its lineage (no entry that extends it), and its
    // conversation is not idle. The rule changes the order of the victims only: a restore of the entries that stay
    // returns what it returned before.
    const std::set<const Item*> headset(heads.begin(), heads.end());
    std::vector<Item*> items;
    cache_.ForEach([&](Item& item) { items.push_back(&item); });
    newest_.clear();
    // A diagnostic must never stop a save (L8, review): a failed allocation leaves the order as it was.
    try {
      std::vector<std::pair<double, Item*>> unproven;  // (active_at, item), newest first at the end of this block
      for (Item* item : items) {
        const int level = tier.at(item);
        if (item->superseded || level < kStale || level > kExtra) continue;  // never evicted, or already last
        // A session with a resume point has the cache's own clocks; a state on probation has none yet, so the idle
        // time (wall clock) bounds its protection.
        const Item* head = owner.at(item) ? owner.at(item) : headset.contains(item) ? item : nullptr;
        if (head ? idle.contains(head) || stale.contains(head) : now - item->active_at > idle_seconds_) continue;
        const bool deepest = std::none_of(items.begin(), items.end(), [&](const Item* other) {
          return other->ids.size() > item->ids.size() && Extends(*other, *item);
        });
        if (!deepest) continue;
        // A conversation's first state and a one-time request's state look the same until the next turn, so an unproven
        // state is protected too. A burst of one-time requests must not protect every one of them (M2, review): only
        // the newest `live_` (the slot count) unproven states join.
        if (OnProbation(*item)) unproven.emplace_back(item->active_at, item);
        else newest_.insert(item);
      }
      const size_t limit = std::max<size_t>(live_, 1);
      if (unproven.size() > limit) {
        std::sort(unproven.begin(), unproven.end(),
                  [](const std::pair<double, Item*>& a, const std::pair<double, Item*>& b) { return a.first > b.first; });
        unproven.resize(limit);
      }
      for (const auto& [_, item] : unproven) newest_.insert(item);
    } catch (const std::bad_alloc&) {
      newest_.clear();
    }
    newest_last_ = newest_.size();
    return tier;
  }
  // Hands this save's victims to the spill hook. The save leaves available + victims - bytes above zero; the
  // reserve takes its part, and each held victim takes its bytes from the rest. A victim that is not held frees its
  // memory when `victims` ends.
  void SpillVictims(uint64_t available, size_t bytes) {
    uint64_t have = available;
    for (const auto& victim : *victims_) have = victim.freed > UINT64_MAX - have ? UINT64_MAX : have + victim.freed;
    const uint64_t need = host_reserve_ + bytes;
    HandVictims(have > need ? have - need : 0);
  }
  void HandVictims(uint64_t room) {
    if (victims_->empty()) return;
    const auto victims = std::move(*victims_);
    victims_->clear();
    for (const auto& victim : victims) {
      const bool spilled = spill_(victim.key, victim.policy, victim.entry, victim.tier, room);
      if (spilled) room -= std::min<uint64_t>(room, victim.freed);
      if (victim.tier == kReadAhead && read_ahead_evicted_ &&
          !read_ahead_evicted_(victim.key, victim.bytes, spilled) && victim.item) {
        cache_.Reinsert(*victim.item);
        read_ahead_kept_ = true;
      }
    }
  }
  // Budget room and the host reserve (default 8 GiB of MemAvailable) together (ConversationCache ReserveHost): evicted snapshots count
  // toward the reserve, and a save that cannot fit evicts nothing. A save that only live resume points could make
  // room for is refused as "protected". A copy of a slot's state takes only superseded entries ("resident"). A
  // speculative tail never takes a waiting session's resume point: with more sessions than slots, that state is the
  // next turn of a session that has no slot (tests/slot_copy_soak_test.cpp).
  bool Admit(size_t bytes, uint64_t available, bool resident = false, bool speculative = false,
             const SharedSave& share = {}) {
    const uint64_t reserve = std::max<uint64_t>(host_reserve_, keep_free_);
    const size_t host_bytes = share.host_bytes && share.host_bytes < bytes ? share.host_bytes : bytes;
    // The eviction hook records what each victim frees with the chunks that this save keeps. The plan's protected set
    // lives only inside this call: a re-entrant Plan from HandVictims must not find it (L1, review).
    *kept_ = share.kept;
    const struct Reset {
      std::vector<const void*>& kept;
      std::set<const Item*>& newest;
      ~Reset() {
        kept.clear();
        newest.clear();
      }
    } reset{*kept_, newest_};
    const auto freed = [&](const Item& item) { return Freed(item, share.kept); };
    // No eviction needed: skip the plan.
    if (cache_.ReserveHost(bytes, host_bytes, available, reserve, [](const Item&) { return -1; }, freed)) return true;
    const auto plan = Plan();
    if (cache_.ReserveHost(bytes, host_bytes, available, reserve, [&](const Item& item) {
          const int tier = plan.at(&item);
          if (Held(item)) return int(kKept);
          // A speculative save never takes a client prompt's state that waits for its session's next turn.
          if ((resident && tier > kSuperseded) || (speculative && (tier == kWaiting || tier == kProbation)))
            return int(kKept);
          if (tier == kKept && read_ahead_ && read_ahead_(item.ids)) return int(kReadAhead);
          // An earlier change: the newest state of a recent conversation leaves after every older or idle entry. The guards
          // above keep their meaning, so this is the order of the victims only.
          if (newest_.contains(&item)) return int(kNewest);
          return tier;
        }, freed)) {
      SpillVictims(available, host_bytes);
      return true;
    }
    if (resident) {
      refusal_ = "resident";
      return false;
    }
    // An empty cache would hold it: only live resume points stand in the way.
    refusal_ = available + cache_.Bytes() >= reserve + host_bytes ? "protected" : "host";
    return false;
  }
  // The host memory that an eviction frees: the entry less its chunks that another snapshot also uses or that the
  // save keeps. Two victims that share a chunk each count it as held: the count stays on the safe side.
  static size_t Freed(const Item& item, std::span<const void* const> kept) {
    if constexpr (requires(const Snapshot& snapshot) { snapshot.bytes.OwnBytes(kept); }) {
      const auto& bytes = item.snapshot->snapshot->bytes;
      const size_t held = bytes.size() - bytes.OwnBytes(kept);
      return item.bytes > held ? item.bytes - held : 0;
    } else {
      return item.bytes;
    }
  }
  std::shared_ptr<SnapshotMemoryLedger> ledger_;
  PrefixCache<Entry, ConversationPolicy> cache_;
  const size_t live_;
  bool slot_frontiers_ = false;
  const uint64_t host_reserve_;
  const bool probation_;
  const double idle_seconds_;
  std::function<double()> clock_now_;
  struct Resident { std::vector<int> hist; int replay = 0; ConversationPolicy policy; };
  std::map<int, Resident> resident_;  // slot: tail tokens, pending replay rows and policy
  uint64_t tick_ = 0;
  // Counts saves and hits that are not on probation (without probation: every save and hit, as tick_).
  uint64_t clock_ = 0;
  const char* refusal_ = "host";
  uint64_t attempts_ = 0, serials_ = 0;
  BatchCacheSave last_;
  struct Victim {
    std::vector<int> key;
    ConversationPolicy policy;
    std::shared_ptr<const Entry> entry;
    int tier;
    size_t freed;  // the host memory that the eviction frees (Freed, with the chunks that the save keeps)
    size_t bytes;  // the entry's budget bytes
    std::optional<Item> item;  // a read-ahead point (kReadAhead) with its marks, to put back when the disk refuses it
  };
  Spill spill_;
  ReadAhead read_ahead_;  // set only inside MakeRoom
  ReadAheadEvicted read_ahead_evicted_;
  uint64_t keep_free_ = 0;  // set only inside MakeRoom
  bool reading_ = false, read_ahead_kept_ = false;  // set only inside MakeRoom
  std::vector<int> held_;  // Hold
  // Evicted in the current Admit, waiting for SpillVictims (shared with the eviction hook, so no `this` capture).
  std::shared_ptr<std::vector<Victim>> victims_ = std::make_shared<std::vector<Victim>>();
  // The chunks that the save of the current Admit keeps (for the eviction hook; empty outside Admit).
  std::shared_ptr<std::vector<const void*>> kept_ = std::make_shared<std::vector<const void*>>();
  // The states that left the host cache at once (evicted or refused), newest first (Diagnose, NoteRefused).
  std::shared_ptr<std::vector<Lost>> lost_ = std::make_shared<std::vector<Lost>>();
  // The newest saved state of each recent conversation (Plan): its victims order only.
  std::set<const Item*> newest_;
  size_t newest_last_ = 0;  // the size of the last plan's set (tests only)
};
}  // namespace crucible
