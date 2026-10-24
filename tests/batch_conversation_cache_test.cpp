#include "../engine/serve/batch_conversation_cache.h"
#include "../engine/runtime/sha256.h"
#include <cstdio>
#include <cassert>
#include <numeric>
using namespace crucible;
struct Snapshot { std::vector<int> hist; ConversationResume resume; std::vector<unsigned char> bytes; };
struct Ledger : SnapshotMemoryLedger {
  size_t allocated = 0, released = 0;
  void Update(SnapshotMemoryAction action, size_t bytes) noexcept override {
    (action == SnapshotMemoryAction::Allocate ? allocated : released) += bytes;
  }
};
static std::vector<int> Tokens(int first, size_t count) {
  std::vector<int> ids(count); std::iota(ids.begin(), ids.end(), first); return ids;
}
static std::vector<int> Fork(const std::vector<int>& base, size_t keep, int first) {
  auto ids = Tokens(first, base.size());
  std::copy(base.begin(), base.begin() + keep, ids.begin());
  return ids;
}

// Three 64-byte entries fill the budget (unless `entries` says more), so each later capture evicts one entry.
// The rule tests below save states that no later request uses, so they run without probation (Probation() tests it).
struct Cache {
  BatchConversationCache<Snapshot> cache;
  ConversationPolicy policy;
  explicit Cache(size_t live = 8, size_t entries = 3, bool probation = false)
      : cache(64 * entries, {}, live, kDefaultHostReserve, probation) { policy.prefill_rows = 8; }
  bool Capture(const std::vector<int>& ids, size_t position, CacheRetention retention) {
    return cache.Capture(ids, policy, {position, retention}, 64, size_t(9) << 30, [&] {
      auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + position);
      s->resume.length = position; s->resume.prompt_only = true; s->resume.prompt_boundary = ids.size() - 1;
      return std::shared_ptr<const Snapshot>(s);
    });
  }
  bool Tail(const std::vector<int>& ids, size_t length, bool trusted = false, bool guarded = false,
            bool resident = false, uint64_t available = uint64_t(9) << 30, size_t prompt = 0) {
    return cache.CaptureTail(std::span<const int>(ids.data(), length), policy, 64, available, [&] {
      auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + length);
      s->resume.length = length; return std::shared_ptr<const Snapshot>(s);
    }, trusted, guarded, resident, {}, {}, prompt);
  }
  size_t HitTail(const std::vector<int>& ids) {
    const auto entry = cache.FindTail(ids, policy);
    return entry ? entry->snapshot->hist.size() : 0;
  }
  size_t Hit(const std::vector<int>& ids) {
    const auto entry = cache.Find(ids, policy, ids.size() - 1, ids.size(), 8);
    return entry ? entry->snapshot->hist.size() : 0;
  }
};

// Cache policy: superseded Session frontiers leave first, Shared ones never become superseded.
static void SupersededFirst() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40);
  {
    Cache c;
    // A first turn whose own frontier equals its shared frontier keeps the Shared class.
    const auto first = BatchCacheFrontiers(12, 11, 8, 10);
    assert(first.size() == 1 && first[0].position == 8 && first[0].retention == CacheRetention::Shared);
    assert(c.Capture(x, 8, first[0].retention));
    assert(c.Capture(x, 16, CacheRetention::Session));
    assert(c.Capture(x, 24, CacheRetention::Session));  // supersedes 16, never the shared 8
    assert(c.Capture(y, 16, CacheRetention::Session));  // full: evicts 16, not the older shared 8
    assert(c.cache.Evictions() == 1 && c.cache.SupersededEvictions() == 1);
    assert(c.Hit(x) == 24 && c.Hit(Fork(x, 20, 300)) == 8 && c.Hit(y) == 16);
    assert(c.Capture(y, 24, CacheRetention::Session));  // evicts y's superseded 16
    assert(c.cache.SupersededEvictions() == 2 && c.Hit(Fork(x, 20, 300)) == 8);
  }
  {
    // Fork, then the original branch continues before memory runs low: its hit clears the mark.
    Cache c;
    const auto fork = Fork(x, 20, 200);
    assert(c.Capture(x, 8, CacheRetention::Shared) && c.Capture(x, 16, CacheRetention::Session));
    assert(c.Capture(fork, 24, CacheRetention::Session));  // marks x's 16
    assert(c.Hit(x) == 16);
    assert(c.Capture(y, 16, CacheRetention::Session));     // plain LRU now: the shared 8 leaves
    assert(c.cache.SupersededEvictions() == 0 && c.Hit(x) == 16 && c.Hit(fork) == 24);
  }
  {
    // Fork, then memory runs low before the original branch returns: it falls back to the shared frontier.
    Cache c;
    const auto fork = Fork(x, 20, 200);
    assert(c.Capture(x, 8, CacheRetention::Shared) && c.Capture(x, 16, CacheRetention::Session));
    assert(c.Capture(fork, 24, CacheRetention::Session));
    assert(c.Capture(y, 16, CacheRetention::Session));
    assert(c.cache.SupersededEvictions() == 1 && c.Hit(x) == 8 && c.Hit(fork) == 24);
  }
}

// G15 tails, with the multi-session rules: a prompt frontier never
// marks a tail. A speculative tail marks nothing; a hit confirms it and marks what it extends. A trusted tail (a
// replayed turn without reasoning) marks what it extends at once, as all tails did before.
static void TailRules() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40);
  {
    Cache c;
    assert(c.Tail(x, 21) && c.Capture(x, 24, CacheRetention::Session));  // the frontier extends the tail
    assert(c.Capture(y, 16, CacheRetention::Session) && c.Capture(z, 16, CacheRetention::Session));
    assert(c.cache.Evictions() == 1 && c.cache.SupersededEvictions() == 0);  // the unconfirmed tail left
    assert(!c.HitTail(x) && c.Hit(x) == 24);
  }
  {
    // train 12 marked x's 16 frontier at the tail save and evicted it here: x then had no resume point.
    Cache c;
    assert(c.Capture(x, 16, CacheRetention::Session) && c.Tail(x, 21));
    assert(c.Capture(y, 16, CacheRetention::Session) && c.Capture(z, 16, CacheRetention::Session));
    assert(c.cache.SupersededEvictions() == 0 && !c.HitTail(x));
    assert(c.Hit(x) == 16 && c.Hit(y) == 16 && c.Hit(z) == 16);
  }
  {
    Cache c;
    assert(c.Capture(x, 8, CacheRetention::Shared) && c.Capture(x, 16, CacheRetention::Session) && c.Tail(x, 21));
    assert(c.HitTail(x) == 21);  // the client repeated the tail: the 16 frontier is superseded, never the shared 8
    assert(c.Capture(y, 16, CacheRetention::Session));
    assert(c.cache.SupersededEvictions() == 1 && c.HitTail(x) == 21 && c.Hit(Fork(x, 12, 300)) == 8);
  }
  {
    Cache c;
    assert(c.Capture(x, 16, CacheRetention::Session) && c.Tail(x, 21, true));  // trusted: marks the 16 at once
    assert(c.Capture(y, 16, CacheRetention::Session) && c.Capture(z, 16, CacheRetention::Session));
    assert(c.cache.SupersededEvictions() == 1 && c.HitTail(x) == 21);
  }
  {
    // A guarded tail (reasoning) says so on a hit: a replay from it shows that the client keeps the reasoning.
    Cache c;
    assert(c.Tail(x, 21, false, true) && c.Tail(y, 21));
    assert(c.cache.FindTail(x, c.policy)->guarded && !c.cache.FindTail(y, c.policy)->guarded);
  }
}

// A speculative save never pushes out a live session's resume point; idle sessions and sessions above their fair
// share lose entries first.
static void SessionRules() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40), w = Tokens(301, 40);
  {
    // train 12 saved this tail and evicted x's own resume point.
    Cache c;
    assert(c.Capture(x, 16, CacheRetention::Session) && c.Capture(y, 16, CacheRetention::Session) &&
           c.Capture(z, 16, CacheRetention::Session));
    assert(!c.Tail(x, 21) && c.cache.Status().last.refused == "protected" && c.cache.Evictions() == 0);
    assert(c.Tail(x, 21, true));  // trusted: x's own 16 is superseded, so it makes the room
    assert(c.HitTail(x) == 21 && c.Hit(y) == 16 && c.Hit(z) == 16);
  }
  {
    // One live session: z, the most recent, is live. x is recent but beyond the live count (it waits for a slot):
    // its resume point outlasts z's speculative tail (tests/slot_copy_soak_test.cpp).
    Cache c(1);
    assert(c.Capture(z, 16, CacheRetention::Session) && c.Tail(z, 21) && c.Capture(x, 16, CacheRetention::Session));
    assert(c.Hit(z) == 16);
    assert(c.Capture(w, 16, CacheRetention::Session));
    assert(c.Hit(x) == 16 && c.HitTail(z) == 0 && c.Hit(w) == 16);
  }
  {
    // A speculative tail never takes a waiting session's resume point; a confirmed save does, as the last choice.
    Cache c(1, 2);
    assert(c.Capture(z, 16, CacheRetention::Session) && c.Capture(x, 16, CacheRetention::Session) && c.Hit(z) == 16);
    assert(!c.Tail(z, 21) && c.cache.Status().last.refused == "protected");
    assert(c.Tail(z, 21, true));  // trusted: z's own 16 makes the room
    assert(c.Hit(x) == 16 && c.HitTail(z) == 21);
    assert(c.Capture(w, 16, CacheRetention::Session) && c.Hit(x) == 0 && c.HitTail(z) == 21);
  }
  {
    // Two live sessions and five entries: x holds three (above its share of 2.5), so x's tails leave first,
    // although y's tail is older.
    Cache c(2, 5);
    assert(c.Capture(y, 16, CacheRetention::Session) && c.Tail(y, 21));
    assert(c.Capture(x, 16, CacheRetention::Session) && c.Tail(x, 21) && c.Tail(x, 23));
    assert(c.Capture(w, 16, CacheRetention::Session));
    assert(c.HitTail(y) == 21 && c.HitTail(x) == 23 && c.Hit(x) == 16 && c.Hit(y) == 16);
  }
  {
    // A session with no save or hit for kIdleRounds * live rounds is idle, so its resume point can leave.
    Cache c(2);
    assert(c.Capture(x, 16, CacheRetention::Session) && c.Capture(y, 16, CacheRetention::Session) &&
           c.Capture(z, 16, CacheRetention::Session));
    for (uint64_t i = 0; i < BatchConversationCache<Snapshot>::kIdleRounds * 2; ++i) assert(c.Hit(y) == 16);
    assert(c.Hit(z) == 16 && c.Tail(y, 21) && c.Hit(x) == 0 && c.HitTail(y) == 21);
  }
}

// Session replay diagnostic: a prompt that leaves the newest tail saved after its restored entry
// names where, in numbers only. Only that tail is compared, and a prompt not longer than it (a retry) is apart.
static void MissedTail() {
  Cache c;
  const auto base = Tokens(1, 40), other = Tokens(101, 40);
  assert(c.Tail(base, 21) && c.Tail(base, 30) && c.Tail(other, 35));
  const auto restored = [&](const std::vector<int>& ids) { return c.cache.FindTail(ids, c.policy); };
  const auto edited = Fork(base, 25, 500);  // repeats the 21-token tail, leaves the 30-token tail at 25
  const auto hit = restored(edited);
  assert(hit && hit->snapshot->hist.size() == 21);
  const auto missed = c.cache.MissedTail(edited, c.policy, hit.get());
  assert(missed && missed->tail == 30 && missed->common == 25 && !missed->shorter);
  assert(SessionReplayDivergenceLine(edited.size(), 21, *missed) ==
         "session_replay diverged prompt=40 restored=21 tail=30 first_mismatch=25\n");
  // Reaching the newest tail, a cold prompt, or a tail of another branch is not a divergence.
  const auto newest = restored(base);
  assert(newest->snapshot->hist.size() == 30 && !c.cache.MissedTail(base, c.policy, newest.get()));
  assert(!c.cache.MissedTail(edited, c.policy, nullptr));
  const auto peer = restored(Fork(other, 36, 900));
  assert(peer->snapshot->hist.size() == 35 && !c.cache.MissedTail(Fork(other, 36, 900), c.policy, peer.get()));
  // A prompt that is not longer than the tail (an edit or a retry) gets its own line, so the diverged count means splits.
  const auto retry = Fork(std::vector<int>(base.begin(), base.begin() + 28), 26, 700);
  const auto shorter = c.cache.MissedTail(retry, c.policy, restored(retry).get());
  assert(shorter && shorter->tail == 30 && shorter->common == 26 && shorter->shorter);
  assert(SessionReplayDivergenceLine(retry.size(), 21, *shorter).starts_with("session_replay shorter prompt=28 "));
  // The newest of two tails saved after one entry is the one compared.
  Cache s;
  assert(s.Tail(base, 21) && s.Tail(base, 30) && s.Tail(Fork(base, 24, 300), 34));  // both follow the 21-token tail
  const auto both = Fork(base, 22, 600);
  const auto sibling = s.cache.MissedTail(both, s.policy, s.cache.FindTail(both, s.policy).get());
  assert(sibling && sibling->tail == 34 && sibling->common == 22);
}

// In-slot checkpoints compare with what the host would restore: Peek gives Find's length and changes nothing.
static void Peek() {
  Cache c;
  const auto base = Tokens(1, 40);
  assert(c.Capture(base, 16, CacheRetention::Session) && c.Tail(base, 21));
  const auto edited = Fork(base, 19, 500);
  assert(c.cache.PeekTail(base, c.policy) == 21 && c.cache.PeekTail(edited, c.policy) == 0);
  assert(c.cache.PeekFind(edited, c.policy, 39, 40, 8) == 16 && c.Hit(edited) == 16);
  const auto& view = c.cache;  // const: no LRU or mark change
  assert(view.PeekTail(std::vector<int>(base.begin(), base.begin() + 21), c.policy) == 0);  // one row must run
  assert(c.HitTail(base) == 21);
}

// Review findings 1 and 2, and host copies of a slot's state ("One copy").
static void ResidentRules() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40), w = Tokens(301, 40);
  const auto S = CacheRetention::Session;
  {
    // A refused trusted tail or frontier keeps its session's resume point protected (finding 1, the 100K fourth
    // session): the marks of a refused save are undone.
    Cache c;
    assert(c.Capture(x, 16, S) && c.Capture(y, 16, S) && c.Capture(z, 16, S));
    assert(!c.Tail(x, 21, true, false, false, uint64_t(7) << 30) && c.cache.Status().last.refused == "host");
    assert(!c.cache.Capture(x, c.policy, {24, S}, 64, uint64_t(7) << 30, []() -> std::shared_ptr<const Snapshot> {
      throw std::logic_error("not reached");
    }));
    assert(!c.Capture(w, 16, S) && c.cache.Status().last.refused == "protected" && c.cache.Evictions() == 0);
    assert(c.Hit(x) == 16 && c.Hit(y) == 16 && c.Hit(z) == 16);
  }
  {
    // A slot resume with no host tail keeps its session live (finding 2): z, not the resumed x, is idle.
    Cache c(2);
    assert(c.Capture(x, 16, S) && c.Capture(y, 16, S) && c.Capture(z, 16, S));
    for (uint64_t i = 0; i < BatchConversationCache<Snapshot>::kIdleRounds * 2; ++i) assert(c.Hit(y) == 16);
    c.cache.Touch(Tokens(1, 30), c.policy);
    assert(c.Capture(w, 16, S));
    assert(c.Hit(z) == 0 && c.Hit(x) == 16);
  }
  {
    // A host copy of the tail that slot 0 holds leaves before states that exist only on the host, even when it is
    // a live resume point. A copy itself takes only free room or superseded entries.
    Cache c;
    assert(c.Capture(x, 16, S) && c.Capture(y, 16, S) && c.Tail(z, 21, true, false, true));
    c.cache.SetResident(0, std::span<const int>(z.data(), 21), 0, c.policy);
    assert(c.cache.CopyBytes() == 64 && c.cache.HoldsTail(std::span<const int>(z.data(), 21), c.policy, 0));
    assert(!c.cache.HoldsTail(std::span<const int>(z.data(), 21), c.policy, 1));
    assert(!c.Tail(w, 21, false, false, true) && c.cache.Status().last.refused == "resident");
    assert(c.Capture(w, 16, S) && !c.cache.HoldsTail(std::span<const int>(z.data(), 21), c.policy, 0));
    assert(c.Hit(x) == 16 && c.Hit(y) == 16 && c.Hit(w) == 16 && c.cache.CopyBytes() == 0);
  }
  {
    // When the slot moves on, the entry is the only copy again and keeps its protection.
    Cache c;
    assert(c.Capture(x, 16, S) && c.Capture(y, 16, S) && c.Tail(z, 21, true, false, true));
    c.cache.SetResident(0, std::span<const int>(z.data(), 21), 0, c.policy);
    c.cache.ClearResident(0);
    assert(!c.Capture(w, 16, S) && c.cache.Status().last.refused == "protected" && c.HitTail(z) == 21);
  }
}

// Cache keys: an entry is found by its key. `tokens` checks the snapshot's own history.
static void KeyedEntries() {
  Cache c;
  const auto tokens = Tokens(1, 21);
  auto key = tokens; key[3] = (1 << 30) + 7;  // an image token: same ID, another image
  assert(c.cache.CaptureTail(key, c.policy, 64, uint64_t(9) << 30, [&] {
    auto s = std::make_shared<Snapshot>(); s->hist = tokens; s->resume.length = tokens.size();
    return std::shared_ptr<const Snapshot>(s);
  }, false, false, false, tokens));
  auto prompt = Tokens(1, 30);
  assert(!c.cache.FindTail(prompt, c.policy));
  std::copy(key.begin(), key.end(), prompt.begin());
  assert(c.cache.FindTail(prompt, c.policy) && c.cache.HoldsTail(key, c.policy, 0) && !c.cache.HoldsTail(tokens, c.policy, 0));
  assert(c.cache.Capture(prompt, c.policy, {16, CacheRetention::Session}, 64, uint64_t(9) << 30, [&] {
    auto s = std::make_shared<Snapshot>(); s->hist = Tokens(1, 16); s->resume.length = 16;
    s->resume.prompt_only = true; s->resume.prompt_boundary = 29;
    return std::shared_ptr<const Snapshot>(s);
  }, Tokens(1, 30)));
  assert(c.Hit(prompt) == 16 && c.Hit(Tokens(1, 30)) == 0);
}

// Image rows key by pixels (engine/model/flashnext/multimodal.h CacheKey), but a snapshot keeps the real token IDs.
static void ImageKeys() {
  const auto ids = Tokens(1, 40);
  auto key = ids;
  for (size_t i = 4; i < 20; ++i) key[i] = -int(i) - 1;
  ConversationPolicy policy; policy.prefill_rows = 8;
  BatchConversationCache<Snapshot> cache(256);
  const auto saved = [&](size_t length, bool prompt) {
    auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + length);
    s->resume.length = length; s->resume.prompt_only = prompt; s->resume.prompt_boundary = 39;
    return std::shared_ptr<const Snapshot>(s);
  };
  assert(cache.Capture(key, policy, {16, CacheRetention::Session}, 64, size_t(9) << 30, [&] { return saved(16, true); },
                       ids));
  assert(cache.Find(key, policy, 39, 40, 8)->snapshot->hist.size() == 16);
  assert(!cache.Find(ids, policy, 39, 40, 8));  // the same pad IDs with other pixels never match
  assert(cache.CaptureTail(std::span<const int>(key.data(), 30), policy, 64, size_t(9) << 30,
                           [&] { return saved(30, false); }, false, false, false, std::span<const int>(ids.data(), 30)));
  assert(cache.FindTail(key, policy)->snapshot->hist.size() == 30);
  assert(!cache.FindTail(ids, policy));
}

// A header frontier (the grid point past the stable boundary, an earlier soak run) is speculative: only a next turn
// that repeats the header uses it. It supersedes nothing until a hit confirms it, so a retry of the same prompt and
// a next turn that drops the reasoning keep the own frontier (a review: budget 2, rows 8, prompt 25,
// boundary 22).
static void HeaderFrontier() {
  const auto prompt = Tokens(1, 25);
  const auto frontiers = BatchCacheFrontiers(25, 22, 8);
  assert(frontiers.size() == 2 && frontiers[0].position == 16 && frontiers[1].position == 24);
  assert(!frontiers[0].speculative && frontiers[1].speculative);
  const auto save = [&](Cache& c, const std::vector<int>& ids, size_t boundary, const CacheFrontier& frontier) {
    return c.cache.Capture(ids, c.policy, frontier, 64, uint64_t(9) << 30, [&] {
      auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + frontier.position);
      s->resume = BatchFrontierResume(frontier.position, boundary);
      return std::shared_ptr<const Snapshot>(s);
    });
  };
  const auto hit = [](Cache& c, const std::vector<int>& ids, size_t boundary) {
    const auto entry = c.cache.Find(ids, c.policy, boundary, ids.size(), 8);
    return entry ? entry->snapshot->hist.size() : 0;
  };
  auto next = Fork(Tokens(1, 40), 24, 300);  // repeats the header, differs at the reasoning marker
  const auto dropped = Fork(Tokens(1, 40), 20, 400);  // older turns rendered without reasoning
  const auto other = Tokens(101, 25);
  {
    Cache c(8, 2);
    assert(save(c, prompt, 22, frontiers[0]) && save(c, prompt, 22, frontiers[1]));
    assert(hit(c, prompt, 22) == 16);
    assert(save(c, other, 22, frontiers[0]));  // full: the unconfirmed header leaves, not the own frontier
    assert(hit(c, prompt, 22) == 16 && hit(c, dropped, 38) == 16);
  }
  {
    Cache c(8, 2);
    assert(save(c, prompt, 22, frontiers[0]) && save(c, prompt, 22, frontiers[1]));
    assert(hit(c, next, 38) == 24);  // the hit confirms the header and supersedes the own frontier
    assert(save(c, other, 22, frontiers[0]) && c.cache.SupersededEvictions() == 1);
    assert(hit(c, next, 38) == 24 && hit(c, prompt, 22) == 0);
  }
  {
    // No hit between the saves: a hit would clear a wrong mark on the own frontier (conversation_cache.h).
    Cache c(8, 2);
    assert(save(c, prompt, 22, frontiers[0]) && save(c, prompt, 22, frontiers[1]));
    assert(save(c, other, 22, frontiers[0]));
    assert(hit(c, prompt, 22) == 16);
  }
  {
    // One live session: A waits for a slot. B's header, speculative, never takes A's resume point.
    Cache c(1, 2);
    assert(save(c, prompt, 22, frontiers[0]) && save(c, other, 22, frontiers[0]));
    assert(!save(c, other, 22, frontiers[1]));
    assert(hit(c, prompt, 22) == 16 && hit(c, other, 22) == 16);
  }
}

// The host reserve is the server's --host-reserve-mib (default 8 GiB): a save keeps that much MemAvailable free.
static void HostReserve() {
  const auto x = Tokens(1, 40);
  const auto snapshot = [&] {
    auto s = std::make_shared<Snapshot>(); s->hist.assign(x.begin(), x.begin() + 16);
    s->resume.length = 16; s->resume.prompt_only = true; s->resume.prompt_boundary = x.size() - 1;
    return std::shared_ptr<const Snapshot>(s);
  };
  ConversationPolicy policy;
  policy.prefill_rows = 8;
  const uint64_t available = uint64_t(6) << 30;
  BatchConversationCache<Snapshot> standard(192, {}, 8);
  assert(standard.HostReserve() == kDefaultHostReserve && kDefaultHostReserve == uint64_t(8) << 30);
  assert(!standard.Capture(x, policy, {16, CacheRetention::Session}, 64, available, snapshot));
  assert(standard.Status().last.refused == "host");
  BatchConversationCache<Snapshot> lower(192, {}, 8, uint64_t(4) << 30);
  assert(lower.HostReserve() == uint64_t(4) << 30);
  assert(lower.Capture(x, policy, {16, CacheRetention::Session}, 64, available, snapshot) && lower.Bytes() == 64);
}

// The disk tier (state_disk.h): an evicted state of classes kStale to kWaiting goes to the spill
// hook with its key and entry. A superseded entry and a copy of a slot's state stay off the disk.
// the disk writer holds a spilled victim until its write ends, so a victim does not free host memory for the
// save that evicts it. The hook gets the room that the save leaves above the host reserve (available + victims -
// new bytes - reserve), less the victims that it already holds, and says whether it holds this one.
static void SpillRoom() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40), w = Tokens(301, 40), v = Tokens(401, 40);
  using Entry = BatchConversationCache<Snapshot>::Entry;
  constexpr uint64_t reserve = 1000;
  std::vector<uint64_t> rooms;
  std::vector<int> tiers;
  const auto hook = [&](std::span<const int>, const ConversationPolicy&, const std::shared_ptr<const Entry>& entry,
                        int tier, uint64_t room) {
    rooms.push_back(room), tiers.push_back(tier);
    return entry->memory->bytes <= room;
  };
  const auto save = [](BatchConversationCache<Snapshot>& cache, const std::vector<int>& ids, size_t bytes,
                       uint64_t available) {
    ConversationPolicy policy;
    policy.prefill_rows = 8;
    return cache.Capture(ids, policy, {16, CacheRetention::Session}, bytes, available, [&] {
      auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + 16);
      s->resume.length = 16; s->resume.prompt_only = true; s->resume.prompt_boundary = ids.size() - 1;
      return std::shared_ptr<const Snapshot>(s);
    });
  };
  {
    // One live session: the older resume points wait (kWaiting) and leave to disk, the least recently used first.
    BatchConversationCache<Snapshot> cache(64 * 3, {}, 1, reserve, false);
    cache.SetSpill(hook);
    assert(save(cache, x, 64, 1 << 20) && save(cache, y, 64, 1 << 20) && save(cache, z, 64, 1 << 20) && rooms.empty());
    // The budget evicts x. The save leaves 74 bytes above the reserve with x held: x's write fits.
    assert(save(cache, w, 64, reserve + 74) && rooms == std::vector<uint64_t>{74} && tiers == std::vector<int>{9});
    // The host is 30 bytes short: y's 64 bytes make the room, and only 34 stay above the reserve. y is not held.
    rooms.clear();
    assert(save(cache, v, 64, reserve + 34) && rooms == std::vector<uint64_t>{34} && cache.Bytes() == 64 * 3);
  }
  {
    // Two victims: the first held write lowers the room of the second.
    rooms.clear();
    BatchConversationCache<Snapshot> cache(64 * 3, {}, 1, reserve, false);
    cache.SetSpill(hook);
    assert(save(cache, x, 64, 1 << 20) && save(cache, y, 64, 1 << 20) && save(cache, z, 64, 1 << 20));
    assert(save(cache, w, 128, reserve + 200) && rooms == (std::vector<uint64_t>{200, 136}));
    // A host that is short by the victim's whole size: the room is 0, so nothing is held.
    rooms.clear();
    BatchConversationCache<Snapshot> tight(64 * 3, {}, 1, reserve, false);
    tight.SetSpill(hook);
    assert(save(tight, x, 64, 1 << 20) && save(tight, y, 64, 1 << 20) && save(tight, z, 64, 1 << 20));
    assert(save(tight, w, 64, reserve) && rooms == std::vector<uint64_t>{0});
  }
}

static void SpillHook() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40), w = Tokens(301, 40);
  const auto S = CacheRetention::Session;
  using Entry = BatchConversationCache<Snapshot>::Entry;
  struct Spilled { std::vector<int> key; int tier; size_t hist; };
  std::vector<Spilled> spilled;
  const auto hook = [&](std::span<const int> key, const ConversationPolicy& policy, const std::shared_ptr<const Entry>& entry,
                        int tier, uint64_t) {
    assert(policy.prefill_rows == 8 && entry && entry->snapshot);
    spilled.push_back({std::vector<int>(key.begin(), key.end()), tier, entry->snapshot->hist.size()});
    return true;
  };
  {
    Cache c;
    c.cache.SetSpill(hook);
    assert(c.Capture(x, 8, CacheRetention::Shared) && c.Capture(x, 16, S) && c.Capture(x, 24, S));
    assert(c.Capture(y, 16, S) && c.cache.SupersededEvictions() == 1 && spilled.empty());
  }
  {
    Cache c(2);
    c.cache.SetSpill(hook);
    assert(c.Capture(x, 16, S) && c.Hit(x) == 16 && c.Capture(y, 16, S) && c.Capture(z, 16, S));
    for (uint64_t i = 0; i < BatchConversationCache<Snapshot>::kIdleRounds * 2; ++i) assert(c.Hit(y) == 16);
    assert(c.Hit(z) == 16 && c.Tail(y, 21) && c.Hit(x) == 0);
    assert(spilled.size() == 1 && spilled[0].key == std::vector<int>(x.begin(), x.begin() + 16) &&
           spilled[0].tier == 5 && spilled[0].hist == 16);  // kIdle
  }
  spilled.clear();
  {
    Cache c;
    c.cache.SetSpill(hook);
    assert(c.Capture(x, 16, S) && c.Capture(y, 16, S) && c.Tail(z, 21, true, false, true));
    c.cache.SetResident(0, std::span<const int>(z.data(), 21), 0, c.policy);
    assert(c.Capture(w, 16, S) && spilled.empty());
    // A stopping engine writes what the host holds: every entry with its key and its superseded mark.
    // The most recently used first: x after its hit.
    assert(c.Hit(x) == 16);
    c.cache.Touch(std::span<const int>(y.data(), 20), c.policy);  // a slot resume: used moves, the list order stays
    std::vector<int> firsts;
    c.cache.ForEachEntry([&](std::span<const int> key, const ConversationPolicy&, const std::shared_ptr<const Entry>& entry,
                             bool superseded) {
      assert(entry && entry->snapshot->hist.size() == key.size() && !superseded);
      firsts.push_back(key.front());
    });
    assert((firsts == std::vector<int>{y.front(), x.front(), w.front()}));  // z, a copy of a slot state, left
  }
}

// Probation (design decision, Oct 8): a new entry is on probation until a later request uses it (a hit, a slot resume, or a
// trusted tail of its branch). Probation entries leave first and never become a protected resume point, so one-time
// requests (Pi compaction summaries: one long prompt that never continues) cannot push out live sessions.
static void Probation() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40);
  const auto S = CacheRetention::Session;
  {
    Cache c(2, 4, true);
    assert(c.Capture(x, 16, S) && c.Hit(x) == 16 && c.Capture(y, 16, S) && c.Hit(y) == 16);
    // Twenty one-time requests while x and y wait (the user reads): each saves two frontiers and a speculative tail.
    // Without probation x and y count as idle after 32 saves and leave before the newest one-time states.
    for (int k = 0; k < 20; ++k) {
      const auto once = Tokens(1000 + 100 * k, 40);
      c.Capture(once, 16, S);
      c.Capture(once, 24, S);
      c.Tail(once, 30);
    }
    assert(c.Hit(x) == 16 && c.Hit(y) == 16);
  }
  {
    // A new session's first state is on probation, then a hit makes it a resume point like any other.
    Cache c(1, 2, true);
    assert(c.Capture(x, 16, S) && c.Hit(x) == 16);
    assert(c.Capture(z, 16, S));  // on probation: x stays the live session
    assert(c.Capture(y, 16, S) && c.Hit(x) == 16 && c.Hit(z) == 0);  // z left first
    assert(c.Hit(y) == 16);       // y is proven now, and the most recent session
    assert(c.Capture(z, 16, S) && c.Hit(y) == 16 && c.Hit(x) == 0);
  }
  {
    // A trusted tail (its request resumed from its branch's tail) and a frontier that extends a proven state are
    // proven at once.
    Cache c(2, 4, true);
    assert(c.Capture(x, 16, S) && c.Hit(x) == 16 && c.Capture(x, 24, S));
    assert(c.Tail(y, 21, true));
    assert(c.cache.Proven(std::span<const int>(x.data(), 24), c.policy) &&
           c.cache.Proven(std::span<const int>(y.data(), 21), c.policy));
    assert(c.Capture(z, 16, S) && !c.cache.Proven(std::span<const int>(z.data(), 16), c.policy));
    c.cache.Touch(Tokens(201, 30), c.policy);  // a slot resume of z's session
    assert(c.cache.Proven(std::span<const int>(z.data(), 16), c.policy));
  }
  {
    // in a full cache, the next frontier of a proven session supersedes that session's state and evicts it
    // first. The frontier still extends a proven state, so it stays proven: one-time requests cannot push it out
    // while its session decodes a long answer (t17 S5-long lost two resume points of 123K and 164K tokens).
    Cache c(2, 3, true);
    assert(c.Capture(x, 16, S) && c.Hit(x) == 16 && c.Capture(y, 16, S) && c.Hit(y) == 16);
    assert(c.Capture(z, 16, S));  // on probation; the cache is full
    assert(c.Capture(x, 24, S) && c.cache.SupersededEvictions() == 1);
    assert(c.cache.Proven(std::span<const int>(x.data(), 24), c.policy));
    for (int k = 0; k < 4; ++k) c.Capture(Tokens(1000 + 100 * k, 40), 16, S);
    assert(c.Hit(x) == 24 && c.Hit(y) == 16);
  }
}

// with probation, an abandoned proven session stayed protected for ever while two new
// sessions on probation evicted each other (0 hits of 400). A proven state that no request used for longer than
// the idle time (default 30 minutes, wall clock) is idle, so it can leave (to disk with the tier on).
static void StaleSession() {
  const auto x = Tokens(1, 40), a = Tokens(1000, 40), b = Tokens(2000, 40);
  const auto S = CacheRetention::Session;
  double now = 0;
  Cache c(1, 2, true);
  c.cache.SetClock([&] { return now; });
  assert(c.Capture(x, 16, S) && c.Hit(x) == 16);
  int early = 0, hits = 0;
  for (int round = 0; round < 200; ++round) {
    now += 60;  // one minute a round
    int got = c.Hit(a) == 16;  // each new session's turn: a lookup, then its save (the probe's order)
    c.Capture(a, 16, S);
    got += c.Hit(b) == 16;
    c.Capture(b, 16, S);
    (round < 25 ? early : hits) += got;
  }
  // Within 30 minutes the abandoned session keeps its state. Then it leaves, and the new sessions hit as often as
  // without probation (the probe: 199 of 400 in 200 rounds, one a round; here 170 rounds after the first 30).
  assert(early == 0);
  assert(hits >= 165);
  assert(c.Hit(x) == 0);
  // A session that a request uses within the idle time stays.
  now = 0;
  Cache d(1, 2, true);
  d.cache.SetClock([&] { return now; });
  assert(d.Capture(x, 16, S) && d.Hit(x) == 16);
  for (int round = 0; round < 100; ++round) {
    now += 60;
    if (round % 20 == 0) assert(d.Hit(x) == 16);
    d.Capture(a, 16, S);
    d.Capture(b, 16, S);
  }
  assert(d.Hit(x) == 16);
}

// a session that runs only from its slot (slot resumes call Touch) for 40 minutes,
// with no new host save or hit. Its host head x30 (a confirmed tail) must stay fresh: Touch's longest prefix is the
// slot's copy x35, which is never a head. Without the fix x30 went stale and left before one-time requests.
static void SlotOnlySession() {
  const auto x = Tokens(1, 40);
  const auto S = CacheRetention::Session;
  double now = 0;
  Cache c(2, 4, true);
  c.cache.SetClock([&] { return now; });
  assert(c.Capture(x, 16, S) && c.Hit(x) == 16);
  assert(c.Tail(x, 30, true) && c.HitTail(Tokens(1, 33)) == 30);
  assert(c.Tail(x, 35, false, false, true));
  c.cache.SetResident(0, std::span<const int>(x.data(), 35), 0, c.policy);
  for (int minute = 5; minute <= 40; minute += 5) {
    now += 300;
    c.cache.Touch(std::span<const int>(x.data(), 38), c.policy);
  }
  for (int k = 1; k <= 4; ++k) c.Capture(Tokens(1000 * k, 40), 16, S);  // one-time requests (probation)
  assert(c.HitTail(Tokens(1, 33)) == 30);
}

// a save that shares host chunks with its base needs only its new bytes of host memory. (The chunks that the
// save keeps and the real freed bytes of victims: tests/tail_incr_test.cpp SpillRoomWithSharedChunks.)
static void SharedSaveRoom() {
  const auto ids = Tokens(1, 40);
  ConversationPolicy policy;
  policy.prefill_rows = 8;
  const uint64_t reserve = uint64_t(8) << 30;
  const auto make = [&](size_t length) {
    auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + long(length));
    s->resume.length = length; return std::shared_ptr<const Snapshot>(s);
  };
  const auto run = [&](uint64_t available, size_t host_bytes) {
    BatchConversationCache<Snapshot> cache(1000, {}, 8, reserve, false);
    assert(cache.CaptureTail(std::span<const int>(ids.data(), 30), policy, 200, reserve + 300, [&] { return make(30); }, true));
    const bool saved = cache.CaptureTail(std::span<const int>(ids.data(), 35), policy, 200, available,
                                         [&] { return make(35); }, true, false, false, {}, SharedSave{host_bytes, {}});
    return std::pair{saved, cache.Bytes()};
  };
  // 40 new bytes fit in 50 free bytes: the older tail stays, although the whole save (200) does not fit.
  assert(run(reserve + 50, 40) == std::pair(true, size_t(400)));
  assert(run(reserve + 50, 0) == std::pair(true, size_t(200)));
  // 30 short: the older tail leaves.
  assert(run(reserve + 10, 40) == std::pair(true, size_t(200)));
}

// a slot keeps a session frontier. The older Session frontiers that it extends
// are superseded with no bytes; the host copy later keeps its proven flag; and the live sessions without a slot
// are protected only as far as the budget holds them, so a handover copy can still evict the oldest one.
static void SlotFrontiers() {
  const auto S = CacheRetention::Session;
  const auto x = Tokens(1, 40), y = Tokens(101, 40);
  ConversationPolicy policy;
  policy.prefill_rows = 8;
  {
    Cache c;
    assert(c.Capture(x, 8, S) && c.Hit(x) == 8 && c.cache.ProvenBelow(std::span<const int>(x.data(), 16), c.policy));
    assert(!c.cache.ProvenBelow(std::span<const int>(y.data(), 16), c.policy));
    // Three live heads fill the budget. A speculative point marks nothing (a fourth state finds no room); a confirmed
    // one supersedes frontier 8, which leaves first.
    assert(c.Capture(y, 8, S) && c.Capture(Tokens(201, 40), 8, S));
    c.cache.KeepInSlot(std::span<const int>(x.data(), 16), c.policy, true);
    assert(!c.Capture(Tokens(301, 40), 8, S) && c.cache.SupersededEvictions() == 0);
    c.cache.KeepInSlot(std::span<const int>(x.data(), 16), c.policy, false);
    assert(c.Capture(Tokens(301, 40), 8, S) && c.cache.SupersededEvictions() == 1 && c.Hit(x) == 0);
  }
  // The host copy of a slot frontier keeps the proven flag that the slot recorded (the entry that it extended can
  // be gone); with probation on it is then no one-time state.
  {
    Cache c(8, 3, true);
    const auto make = [&](size_t length) {
      auto s = std::make_shared<Snapshot>(); s->hist.assign(x.begin(), x.begin() + long(length));
      s->resume.length = length; s->resume.prompt_only = true; s->resume.prompt_boundary = 39;
      return std::shared_ptr<const Snapshot>(s);
    };
    assert(c.cache.Capture(x, c.policy, {16, S}, 64, size_t(9) << 30, [&] { return make(16); }, {}, {}, true));
    assert(c.cache.Proven(std::span<const int>(x.data(), 16), c.policy));
    assert(c.cache.Capture(y, c.policy, {16, S}, 64, size_t(9) << 30, [&] {
      auto s = std::make_shared<Snapshot>(*make(16)); s->hist.assign(y.begin(), y.begin() + 16);
      return std::shared_ptr<const Snapshot>(s); }));
    assert(!c.cache.Proven(std::span<const int>(y.data(), 16), c.policy));
  }
  // Three heads of 64, 32 and 128 bytes in a 256-byte budget: two of the largest fit, so with slot frontiers two are
  // live (the newest) and the oldest waits. A new state of 96 bytes evicts it. Without slot frontiers all three stay
  // protected, as before: the save is refused.
  for (const bool on : {true, false}) {
    BatchConversationCache<Snapshot> cache(256, {}, 8, kDefaultHostReserve, false);
    cache.SetSlotFrontiers(on);
    const auto save = [&](const std::vector<int>& ids, size_t bytes) {
      return cache.Capture(ids, policy, {16, S}, bytes, size_t(9) << 30, [&] {
        auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + 16);
        s->resume.length = 16; s->resume.prompt_only = true; s->resume.prompt_boundary = 39;
        return std::shared_ptr<const Snapshot>(s);
      });
    };
    assert(save(x, 64) && save(y, 32) && save(Tokens(201, 40), 128) && cache.Bytes() == 224);
    if (!on) {
      assert(!save(Tokens(301, 40), 96) && cache.Evictions() == 0 && cache.Status().last.refused == "protected");
      continue;
    }
    assert(save(Tokens(301, 40), 96) && cache.Bytes() == 256 && cache.Evictions() == 1);
    assert(!cache.Find(x, policy, 39, 40, 8) && cache.Find(y, policy, 39, 40, 8));
  }
  // SlotFrontierFits is Find's rule. A host frontier of each length, saved with its own stable
  // boundary, restores for a prompt exactly when the rule accepts a Frontier point of that length.
  const auto z = Tokens(1, 64);
  size_t accepted = 0, refused = 0;
  for (size_t length = 8; length <= 56; length += 4)
    for (const size_t saved_boundary : {size_t(0), size_t(20), size_t(40)})
      for (const size_t prompt : {size_t(41), size_t(57), size_t(64)})
        for (const size_t boundary : {size_t(0), size_t(16), size_t(32), size_t(48)})
          for (const size_t warmup : {size_t(24), prompt - 1}) {
            if (length > prompt) continue;
            BatchConversationCache<Snapshot> cache(1 << 20, {}, 8, kDefaultHostReserve, false);
            const bool saved = cache.Capture(z, policy, {length, S, length > saved_boundary && saved_boundary}, 64,
                                             size_t(9) << 30, [&] {
              auto s = std::make_shared<Snapshot>(); s->hist.assign(z.begin(), z.begin() + long(length));
              s->resume = BatchFrontierResume(length, saved_boundary);
              return std::shared_ptr<const Snapshot>(s);
            });
            if (!saved) continue;
            const std::span<const int> ids(z.data(), prompt);
            const bool found = cache.PeekFind(ids, policy, boundary, warmup, 8) == length;
            assert(found == SlotFrontierFits(length, saved_boundary, prompt, boundary, warmup, 8));
            ++(found ? accepted : refused);
          }
  assert(accepted > 20 && refused > 20);
  // ResumeHost reads PeekTailState: the length and pending rows that FindTail restores, with no change to the cache.
  {
    Cache c;
    assert(c.cache.PeekTailState(x, c.policy) == std::pair(size_t(0), 0));
    assert(c.cache.CaptureTail(std::span<const int>(x.data(), 30), c.policy, 64, size_t(9) << 30, [&] {
      auto s = std::make_shared<Snapshot>(); s->hist.assign(x.begin(), x.begin() + 30);
      s->resume.length = 30; s->resume.replay = 3; return std::shared_ptr<const Snapshot>(s); }));
    assert(c.Tail(x, 20) && c.cache.PeekTailState(x, c.policy) == std::pair(size_t(30), 3));
    const auto found = c.cache.FindTail(x, c.policy);
    assert(found && found->snapshot->hist.size() == 30 && found->snapshot->resume.replay == 3);
    assert(c.cache.PeekTailState(std::span<const int>(x.data(), 30), c.policy) == std::pair(size_t(20), 0));
  }
}

// W1 LocalTenantUnchanged: a fixed trace of captures, tails, finds, touches and slot marks under budget
// pressure, with probation and slot frontiers on. Its decision log is the log of train 23 (7b3106e8): with the local
// tenant only, the tenant key changes no decision. A changed log prints it.
// An earlier change moved the digest on purpose: the newest state of a recent conversation now leaves last
// (kNewest). With that rank off, this trace gives the digest of train 23, ede7ee6b...69ce5, so the tenant key still
// changes no decision.
static const std::string kLocalTrace23 = "a9ac1f406282eee975ff2d0ab5208e728ea3b05f76cdddc601ddabab9e6894f6";
static void LocalTenantUnchanged() {
  BatchConversationCache<Snapshot> cache(64 * 7, {}, 3, kDefaultHostReserve, true);
  cache.SetSlotFrontiers(true);
  ConversationPolicy policy;
  policy.prefill_rows = 8;
  uint32_t seed = 124;
  const auto next = [&] { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fffu; };
  std::vector<std::vector<int>> sessions(5);
  std::vector<std::string> trace;
  const auto log = [&](const std::string& what, size_t value) {
    trace.push_back(what + "=" + std::to_string(value) + " entries=" + std::to_string(cache.Status().entries) +
                    " bytes=" + std::to_string(cache.Bytes()) + " evicted=" + std::to_string(cache.Evictions()) +
                    " refused=" + cache.Status().last.refused);
  };
  const auto make = [](std::span<const int> ids, bool prompt) {
    auto s = std::make_shared<Snapshot>();
    s->hist.assign(ids.begin(), ids.end());
    s->resume.length = ids.size();
    s->resume.prompt_only = prompt;
    s->resume.prompt_boundary = prompt ? ids.size() : 0;
    return std::shared_ptr<const Snapshot>(s);
  };
  for (int step = 0; step < 400; ++step) {
    const size_t s = next() % sessions.size();
    auto& ids = sessions[s];
    if (ids.empty() || ids.size() > 120) ids = Tokens(int(s % 2) * 1000 + 1, 16 + next() % 8);
    if (next() % 4 == 0 && ids.size() > 9) ids.resize(ids.size() - next() % 8);
    auto more = Tokens(int(next() % 5000) + 2000, 3 + next() % 20);
    ids.insert(ids.end(), more.begin(), more.end());
    const size_t boundary = ids.size() - 1 - next() % 4;
    switch (next() % 9) {
      case 0: case 1: {
        for (const auto& frontier : BatchCacheFrontiers(ids.size(), boundary, 8, s % 2 ? 16 : 0))
          log("capture", cache.Capture(ids, policy, frontier, 64, size_t(9) << 30,
                                       [&] { return make(std::span<const int>(ids).first(frontier.position), true); },
                                       ids));
        break;
      }
      case 2: case 3: {
        const bool trusted = next() % 2, guarded = next() % 3 == 0, resident = next() % 3 == 0;
        log("tail", cache.CaptureTail(ids, policy, 64, size_t(9) << 30, [&] { return make(ids, false); }, trusted,
                                      guarded, resident, ids));
        if (resident) cache.SetResident(int(s % 3), ids, 0, policy);
        break;
      }
      case 4: {
        const auto tail = cache.FindTail(ids, policy);
        log("find_tail", tail ? tail->snapshot->hist.size() : 0);
        const auto entry = cache.Find(ids, policy, boundary, ids.size() - 1, 8);
        log("find", entry ? entry->snapshot->hist.size() : 0);
        break;
      }
      case 5:
        log("peek", cache.PeekFind(ids, policy, boundary, ids.size() - 1, 8) + 1000 * cache.PeekTail(ids, policy));
        log("peek_state", cache.PeekTailState(ids, policy).first);
        break;
      case 6:
        cache.Touch(ids, policy);
        log("touch", cache.HoldsTail(ids, policy, 0));
        break;
      case 7: {
        const size_t below = ids.size() / 8 * 8;
        log("proven", cache.ProvenBelow(std::span<const int>(ids).first(below), policy));
        cache.KeepInSlot(std::span<const int>(ids).first(below), policy, next() % 2);
        log("keep", below);
        break;
      }
      default:
        cache.ClearResident(int(next() % 3));
        log("copy_bytes", cache.CopyBytes());
        break;
    }
  }
  crucible::Sha256 hash;
  for (const auto& line : trace) {
    hash.Update(reinterpret_cast<const uint8_t*>(line.data()), line.size());
    hash.Update(reinterpret_cast<const uint8_t*>("\n"), 1);
  }
  const std::string digest = hash.HexDigest();
  if (digest != kLocalTrace23) {
    for (const auto& line : trace) std::printf("%s\n", line.c_str());
    std::printf("trace digest %s\n", digest.c_str());
    std::fflush(stdout);
  }
  assert(digest == kLocalTrace23);
}

// TenantsNeverShareRam (W1, design section 3.5): two tenants with the same tokens. Find, FindTail, PeekFind,
// PeekTail, PeekTailState, HoldsTail, Proven and ProvenBelow of tenant B never see an entry of tenant A. B's calls
// (Touch, KeepInSlot, a slot mark, saves and a trusted tail that would supersede A's branch) change none of A's
// entries: A's later hits, evictions and spills are the same as when B sends other tokens. A spill carries the
// entry's own policy, so a state of A goes to disk as A's.
static void TenantsNeverShareRam() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), w = Tokens(201, 40), other = Tokens(301, 40);
  ConversationPolicy a, b;
  a.prefill_rows = b.prefill_rows = 8;
  a.tenant[0] = 1;
  b.tenant[31] = 2;
  constexpr auto S = CacheRetention::Session;
  const auto make = [](std::span<const int> ids, bool prompt) {
    auto s = std::make_shared<Snapshot>();
    s->hist.assign(ids.begin(), ids.end());
    s->resume.length = ids.size();
    s->resume.prompt_only = prompt;
    s->resume.prompt_boundary = prompt ? ids.size() : 0;
    return std::shared_ptr<const Snapshot>(s);
  };
  // A's turns with B's calls between them; B sends `bx`. Returns what A finds at the end and A's spills.
  const auto run = [&](const std::vector<int>& bx) {
    BatchConversationCache<Snapshot> cache(64 * 5, {}, 2, kDefaultHostReserve, false);
    std::vector<std::pair<std::vector<int>, ConversationPolicy>> spilled;
    cache.SetSpill([&](std::span<const int> key, const ConversationPolicy& policy, const auto&, int, uint64_t) {
      assert(policy == a || policy == b);
      spilled.emplace_back(std::vector<int>(key.begin(), key.end()), policy);
      return true;
    });
    const auto capture = [&](const ConversationPolicy& p, const std::vector<int>& ids, size_t at) {
      return cache.Capture(ids, p, {at, S}, 64, size_t(9) << 30,
                           [&] { return make(std::span<const int>(ids).first(at), true); }, ids);
    };
    const auto tail = [&](const ConversationPolicy& p, const std::vector<int>& ids, size_t length, bool trusted) {
      return cache.CaptureTail(std::span<const int>(ids).first(length), p, 64, size_t(9) << 30,
                               [&] { return make(std::span<const int>(ids).first(length), false); }, trusted);
    };
    assert(capture(a, x, 16) && tail(a, x, 21, false) && cache.FindTail(x, a));
    // B repeats A's tokens: no hit, no peek, no tail state, no proof, no held tail.
    assert(!cache.Find(bx, b, 39, 39, 8) && !cache.FindTail(bx, b) && cache.PeekFind(bx, b, 39, 39, 8) == 0);
    assert(cache.PeekTail(bx, b) == 0 && cache.PeekTailState(bx, b).first == 0 && !cache.ProvenBelow(bx, b));
    assert(!cache.HoldsTail(std::span<const int>(bx).first(21), b, 0) && !cache.Proven(std::span<const int>(bx).first(16), b));
    cache.Touch(bx, b);
    cache.KeepInSlot(std::span<const int>(bx).first(16), b, false);
    cache.SetResident(0, std::span<const int>(bx).first(21), 0, b);
    assert(cache.CopyBytes() == 0);  // B's slot mark makes no copy of A's tail
    // B saves: its own entries; its trusted tail supersedes none of A's.
    assert(capture(b, bx, 16) && tail(b, bx, 24, true));
    assert(cache.PeekTail(x, a) == 21 && cache.PeekTail(bx, b) == 24 && cache.FindTail(bx, b));
    cache.ClearResident(0);
    // Pressure: A's later captures evict.
    capture(a, y, 16);
    capture(a, w, 16);
    capture(a, y, 32);
    for (int session = 0; session < 4; ++session) {
      const auto more = Tokens(1000 + 100 * session, 40);
      capture(a, more, 16);
      tail(a, more, 21, false);
    }
    const std::vector<size_t> found{cache.PeekFind(x, a, 39, 39, 8), cache.PeekTail(x, a),
                                    cache.PeekFind(y, a, 39, 39, 8), cache.PeekFind(w, a, 39, 39, 8)};
    std::vector<std::vector<int>> of_a;
    for (const auto& [key, policy] : spilled)
      if (policy == a) of_a.push_back(key);
    return std::pair{found, of_a};
  };
  const auto same = run(x), apart = run(other);
  assert(same == apart && !same.second.empty());
}

// An earlier change: a turn that cached nothing names its cause and the state nearest to its prompt (the turn line's
// miss_reason, closest_shared_tokens, closest_age_s and closest_outcome). Numbers and fixed words only.
static void Diagnosis() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40);
  const auto S = CacheRetention::Session;
  const auto diagnose = [](Cache& c, const std::vector<int>& ids, size_t boundary, size_t warmup, int rows) {
    return c.cache.Diagnose(ids, c.policy, boundary, warmup, rows, false);
  };
  {
    // An empty cache: nothing shares the prompt's first token (a new conversation, or a fresh engine).
    Cache c;
    const auto d = diagnose(c, x, 39, 40, 8);
    assert(std::string_view(d.miss) == "new" && d.shared == 0 && !d.outcome);
  }
  {
    // The nearest state shares 8 tokens and then the prompt differs (the client forked its conversation).
    Cache c;
    assert(c.Capture(x, 16, S));
    const auto d = diagnose(c, Fork(x, 8, 300), 39, 40, 8);
    assert(std::string_view(d.miss) == "diverged" && d.shared == 8);
    assert(std::string_view(d.outcome) == "prompt_differs");
  }
  {
    // The shared part reaches the tokens that the earlier turn generated: the reply differs, not the prompt.
    Cache c;
    assert(c.Tail(Tokens(1, 45), 45, false, false, false, uint64_t(9) << 30, 40));
    const auto d = diagnose(c, Fork(Tokens(1, 45), 42, 700), 44, 45, 8);
    assert(std::string_view(d.miss) == "diverged" && d.shared == 42);
    assert(std::string_view(d.outcome) == "reply_differs");
  }
  {
    // A state of the whole prompt: one prompt row must run to give the first token, and a state off the grid.
    Cache c;
    assert(c.Capture(x, 16, S));
    const auto limited = diagnose(c, x, 39, 8, 8);
    assert(std::string_view(limited.miss) == "above_limit" && limited.shared == 16);
    assert(std::string_view(limited.outcome) == "all");
    assert(std::string_view(diagnose(c, x, 39, 40, 5).miss) == "off_grid");
  }
  {
    // The state left the cache: the miss names it, with its length and its age.
    Cache c(1, 2);
    assert(c.Capture(x, 16, S) && c.Capture(y, 16, S) && c.Capture(z, 16, S));
    assert(c.cache.Evictions() == 1 && c.Hit(x) == 0);
    const auto d = diagnose(c, x, 39, 40, 8);
    assert(std::string_view(d.miss) == "evicted" && d.shared == 16 && std::string_view(d.outcome) == "all");
    assert(d.age_s >= 0);
  }
  {
    // A save of the state was refused: the cause of a later miss of the same conversation.
    Cache c;
    c.cache.NoteRefused(std::span<const int>(x.data(), 16), c.policy, 16);
    const auto d = diagnose(c, x, 39, 40, 8);
    assert(std::string_view(d.miss) == "not_saved" && d.shared == 16 && std::string_view(d.outcome) == "all");
  }
}

// An earlier change: per-tier counters (lookup, hit, reused tokens) for the slot, host and disk tiers. The restore hook counts
// the host and disk tiers; the session counts the slot tier (BatchCacheHooks::slot_served, batch_session_test).
static void TierCounters() {
  BatchCacheCounters counters;
  counters.Count(BatchCacheTier::Host, 0);
  counters.Count(BatchCacheTier::Host, 100);
  counters.Count(BatchCacheTier::Disk, 300);
  assert(counters.host.lookups == 2 && counters.host.hits == 1 && counters.host.tokens == 100);
  assert(counters.disk.lookups == 1 && counters.disk.hits == 1 && counters.disk.tokens == 300);
  assert(counters.slot.lookups == 0 && counters.slot.hits == 0 && counters.slot.tokens == 0);
  counters.Count(BatchCacheTier::Slot, 40);
  assert(counters.slot.lookups == 1 && counters.slot.hits == 1 && counters.slot.tokens == 40);
  counters.Count(BatchCacheTier::Slot, 0);  // a slot that held nothing: a lookup, not a hit
  assert(counters.slot.lookups == 2 && counters.slot.hits == 1 && counters.slot.tokens == 40);
}

// An earlier change: the eviction rule keeps the newest saved state of a conversation that had a turn recently. An older
// entry of that conversation, or another session's idle state, can go instead.
static void NewestStateKept() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), w = Tokens(5000, 40);
  const auto S = CacheRetention::Session;
  // Filler sessions move the cache's own clock, so that y is idle after its one use. They are recent (kWaiting).
  const auto fillers = int(BatchConversationCache<Snapshot>::kIdleRounds) + 1;
  const auto others = [&](Cache& c) {
    for (int k = 0; k < fillers; ++k) {
      const auto other = Tokens(1000 + 100 * k, 40);
      assert(c.Capture(other, 16, S) && c.Hit(other) == 16);
    }
  };
  {
    // The recent session x saved a frontier and then its tail. Its older frontier leaves instead of the tail.
    Cache c(1, size_t(fillers) + 3, true);
    assert(c.Capture(y, 16, S) && c.Hit(y) == 16);
    assert(c.Capture(x, 16, S) && c.Tail(x, 21));
    others(c);
    assert(c.Capture(w, 16, S));  // the cache is full: one entry leaves
    assert(c.HitTail(x) == 21 && c.Hit(x) == 0 && c.Hit(y) == 16);
  }
  {
    // The recent session x holds only its tail: the idle session's state leaves instead.
    Cache c(1, size_t(fillers) + 2, true);
    assert(c.Capture(y, 16, S) && c.Hit(y) == 16);
    assert(c.Tail(x, 21));
    others(c);
    assert(c.Capture(w, 16, S));
    assert(c.HitTail(x) == 21 && c.Hit(y) == 0);
  }
  {
    // The rule protects a recent conversation only: an old one-time state is not protected, so a session's tail that
    // a request saved after it leaves the old state's room.
    double now = 1000;
    Cache c(1, size_t(fillers) + 2, true);
    c.cache.SetClock([&] { return now; });
    assert(c.Capture(y, 16, S));  // a one-time prompt, then nothing for longer than the idle time
    now += BatchConversationCache<Snapshot>::kDefaultIdleSeconds + 1;
    assert(c.Tail(x, 21));
    others(c);
    assert(c.Capture(w, 16, S));
    assert(c.HitTail(x) == 21 && c.Hit(y) == 0);
  }
}

// Review M2: a burst of one-time requests must not protect every state that it saves; only the newest `live_` unproven
// states join the protected set (a conversation's first state and a one-time state look the same until the next turn).
static void NewestBound() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40), w = Tokens(301, 40);
  const auto S = CacheRetention::Session;
  Cache c(1, 3, true);  // one live session: the bound is one unproven state; three entries fit
  assert(c.Capture(x, 16, S) && c.Capture(y, 16, S) && c.Capture(z, 16, S));
  assert(c.cache.LastNewestStates() == 0);  // no plan yet
  assert(c.Capture(w, 16, S));              // the cache is full: one entry leaves, and the plan runs
  assert(c.cache.LastNewestStates() == 1);  // x, y and z are unproven: only the newest of them is protected
  assert(c.cache.NewestStates() == 0);      // L1: the set lives inside Admit only
  assert(c.Hit(x) == 0 && c.Hit(y) == 16 && c.Hit(z) == 16 && c.Hit(w) == 16);
}

// Review L4: a state larger than 64 Ki tokens is named too (the record keeps a fingerprint, never the key). The shared
// length of a lost state is the whole length when the keys are equal, else the last grid point that matched.
static void LostFingerprint() {
  // A 65,544-token state: the whole-key copy of the first version kept nothing above 65,536 tokens, so this is the case
  // that the fingerprint adds (the daily profile holds 32K to 130K states).
  const auto x = Tokens(1, 70000), y = Tokens(100000, 70000), z = Tokens(200000, 70000);
  const auto S = CacheRetention::Session;
  Cache c(1, 2);
  assert(c.Capture(x, 65544, S) && c.Capture(y, 65544, S) && c.Capture(z, 65544, S));
  assert(c.Hit(x) == 0);  // x left the cache
  const auto same = c.cache.Diagnose(x, c.policy, 69999, 70000, 8, false);
  assert(std::string_view(same.miss) == "evicted" && same.shared == 65544);
  auto forked = x;
  forked[40000] += 7;  // inside the 256-token block that starts at 39,936
  const auto near = c.cache.Diagnose(forked, c.policy, 69999, 70000, 8, false);
  assert(std::string_view(near.miss) == "evicted" && near.shared == 39936);
  assert(std::string_view(near.outcome) == "prompt_differs");
}

// Review L9: only a save that the host had room to keep but did not take is a loss. A refused state of the whole prompt
// ("policy") or one larger than the cache ("budget") is never recorded as "not_saved".
static void RefusalWords() {
  const auto x = Tokens(1, 40);
  const auto S = CacheRetention::Session;
  Cache c(1, 3);
  assert(c.cache.Status().last.refused.empty());
  // A budget refusal: one byte more than the whole cache.
  assert(!c.cache.CaptureTail(std::span<const int>(x.data(), 16), c.policy, 64 * 3 + 1, size_t(9) << 30, [&] {
    auto s = std::make_shared<Snapshot>();
    s->hist.assign(x.begin(), x.begin() + 16); s->resume.length = 16; s->resume.prompt_only = false;
    return std::shared_ptr<const Snapshot>(s);
  }));
  assert(c.cache.LastRefusal() == std::string("budget"));
  c.cache.NoteRefused(std::span<const int>(x.data(), 16), c.policy, 16);
  assert(std::string_view(c.cache.Diagnose(x, c.policy, 39, 40, 8, false).miss) == "new");
  // A host refusal: the reserve cannot be met even with every entry gone (the whole-key state is 3 entries wide).
  c.cache.NoteRefused(std::span<const int>(x.data(), 16), c.policy, 16);  // the refusal word is still "budget"
  assert(std::string_view(c.cache.Diagnose(x, c.policy, 39, 40, 8, false).miss) == "new");
  (void)S;
}

// Review 2 (L10): the word of a refusal is the word of the latest one. A save that the budget refused must not shadow a
// later read's refusal (the stage path names the wrong reason, batched_serve.h:511).
static void LatestRefusal() {
  const auto a = Tokens(2001, 40), b = Tokens(2101, 40);
  Cache c(2, 2);  // two live resume points fill the budget; no probation
  const uint64_t memory = uint64_t(64) << 30;
  assert(c.Capture(a, 16, CacheRetention::Session) && c.Capture(b, 16, CacheRetention::Session));
  // A save of the whole budget plus one byte: the budget refuses it before any eviction.
  assert(!c.cache.CaptureTail(std::span<const int>(a.data(), 16), c.policy, 129, size_t(9) << 30, [&] {
    auto s = std::make_shared<Snapshot>();
    s->hist.assign(a.begin(), a.begin() + 16); s->resume.length = 16; s->resume.prompt_only = false;
    return std::shared_ptr<const Snapshot>(s);
  }));
  assert(c.cache.LastRefusal() == std::string("budget"));
  // A read needs room and the two live points are protected: the latest word is that refusal, not the save's.
  assert(!c.cache.MakeRoom(64, memory));
  assert(c.cache.LastRefusal() == std::string("protected"));
}

int main() {
  TenantsNeverShareRam();
  LocalTenantUnchanged();
  Diagnosis();
  LatestRefusal();
  LostFingerprint();
  RefusalWords();
  NewestBound();
  TierCounters();
  NewestStateKept();
  SlotFrontiers();
  SharedSaveRoom();
  SlotOnlySession();
  StaleSession();
  Probation();
  SpillHook();
  SpillRoom();
  HostReserve();
  SupersededFirst();
  HeaderFrontier();
  KeyedEntries();
  TailRules();
  SessionRules();
  MissedTail();
  Peek();
  ResidentRules();
  ImageKeys();
  auto ledger = std::make_shared<Ledger>();
  std::vector<int> ids(40); std::iota(ids.begin(), ids.end(), 1);
  ConversationPolicy policy; policy.prefill_rows = 8;
  const auto snapshot = [&](size_t frontier) {
    auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin()+frontier);
    s->resume.length = frontier; s->resume.prompt_only = true; s->resume.prompt_boundary = 39;
    s->bytes.resize(32); return std::shared_ptr<const Snapshot>(s);
  };
  {
    BatchConversationCache<Snapshot> cache(128, ledger);
    assert(cache.Capture(ids, policy, {16, CacheRetention::Session}, 64, size_t(9)<<30, [&] { return snapshot(16); }));
    assert(cache.Find(ids, policy, 39, 40, 8)->retention == CacheRetention::Session);
    assert(!cache.Find(ids, policy, 8, 40, 8));
    assert(!cache.Find(ids, policy, 39, 8, 8));
    assert(cache.Capture(ids, policy, {24, CacheRetention::Shared}, 64, size_t(9)<<30, [&] { return snapshot(24); }));
    auto live_peer = cache.Find(ids, policy, 39, 40, 8);
    assert(live_peer->retention == CacheRetention::Shared);
    const auto peer_hist = live_peer->snapshot->hist;
    assert(!cache.Find(ids, policy, 8, 40, 8));
    assert(cache.Find(ids, policy, 16, 40, 8)->snapshot->hist.size() == 16);
    auto spec = policy; spec.spec = 2;
    assert(!cache.Find(ids, spec, 39, 40, 8));
    assert(!cache.Capture(ids, spec, {32, CacheRetention::Session}, 64, size_t(9)<<30, [&] { return snapshot(32); }));
    assert(!cache.Capture(ids, policy, {31, CacheRetention::Session}, 64, size_t(9)<<30, [&] { return snapshot(31); }));
    assert(!cache.Capture(ids, policy, {32, CacheRetention::Session}, 64, size_t(7)<<30, [&] { return snapshot(32); }));
    assert(!cache.Capture(ids, policy, {32, CacheRetention::Session}, 64, size_t(9)<<30,
                         []() -> std::shared_ptr<const Snapshot> { throw std::bad_alloc(); }));
    assert(cache.Bytes() <= 128);
    assert(cache.Capture(ids, policy, {32, CacheRetention::Session}, 64, size_t(9)<<30, [&] { return snapshot(32); }));
    assert(cache.Find(ids, policy, 39, 40, 8)->snapshot->hist.size() == 32);
    assert(cache.Evictions() > 0 && live_peer->snapshot->hist == peer_hist);
    assert(!BatchCacheConsumerCompatible(live_peer->snapshot->resume, 40, 8, 40, 8));
    assert(BatchCacheFrontiers(40, 39, 8).front().position == 32);
    assert(BatchCacheFrontiers(7, 6, 8).empty());
    const auto both = BatchCacheFrontiers(40, 39, 8, 20);
    assert(both.size() == 2 && both[0].position == 16 && both[0].retention == CacheRetention::Shared);
    assert(both[1].position == 32 && both[1].retention == CacheRetention::Session);
    assert(BatchCacheFrontiers(40, 39, 8, 35).size() == 1);  // same grid cell as the session frontier
    // A shared prefix that ends in the session frontier cell keeps the shared class (engine-tests 2768c27).
    assert(BatchCacheFrontiers(40, 39, 8, 35).front().retention == CacheRetention::Shared);
    assert(BatchCacheFrontiers(40, 39, 8, 5).size() == 1);   // below one chunk: nothing to share
    assert(BatchCacheFrontiers(40, 12, 8, 30).front().position == 8);  // never beyond the stable boundary
    // A grid point inside the generation prompt (an earlier soak run: prompt 57,345 = 14 x 4096 + 1, boundary
    // 57,341). The next turn repeats the header up to its last token, so the 57,344 state is saved too.
    const auto header = BatchCacheFrontiers(57345, 57341, 4096);
    assert(header.size() == 2 && header[0].position == 53248 && header[1].position == 57344);
    assert(header[1].retention == CacheRetention::Session);
    assert(BatchCacheFrontiers(57346, 57341, 4096).size() == 2 && BatchCacheFrontiers(57344, 57341, 4096).size() == 1);
    assert(BatchCacheFrontiers(57345, 0, 4096).size() == 1);  // no boundary: the frontier is already the last grid point
    // The header state is valid for a later prompt that repeats it, although it passes the capturing boundary.
    {
      BatchConversationCache<Snapshot> header_cache(1000, ledger);
      std::vector<int> prompt(25); std::iota(prompt.begin(), prompt.end(), 1);
      assert(header_cache.Capture(prompt, policy, BatchCacheFrontiers(25, 22, 8)[1], 64, size_t(9)<<30, [&] {
        auto s = std::make_shared<Snapshot>(); s->hist.assign(prompt.begin(), prompt.begin() + 24);
        s->resume.length = 24; s->resume.prompt_only = true; s->resume.prompt_boundary = 0; s->bytes.resize(32);
        return std::shared_ptr<const Snapshot>(s);
      }));
      auto next = prompt; next[24] = 99; next.resize(40, 7);
      assert(header_cache.Find(next, policy, 38, 40, 8)->snapshot->hist.size() == 24);
    }
  }
  {
    // Session replay (G15): at a full cache the host-memory reserve counts the entries that the budget evicts.
    // Measured: a 106K session (served layout, 10 GiB) kept no third tail, because the reserve check ran on the
    // memory measured before eviction freed the older tail and frontier.
    BatchConversationCache<Snapshot> full(256, ledger);
    const auto make = [&](size_t length) {
      auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + length);
      s->resume.length = length; s->bytes.resize(32);
      return std::shared_ptr<const Snapshot>(s);
    };
    const uint64_t reserve = uint64_t(8) << 30;
    assert(full.CaptureTail(std::span<const int>(ids.data(), 21), policy, 200, reserve + 300, [&] { return make(21); }));
    // 100 bytes of host memory beyond the reserve, but saving evicts the older 200-byte tail first.
    assert(full.CaptureTail(std::span<const int>(ids.data(), 30), policy, 200, reserve + 100, [&] { return make(30); }));
    assert(full.FindTail(ids, policy)->snapshot->hist.size() == 30 && full.Bytes() == 200);
    // Without enough memory even after eviction, nothing is saved, and nothing is evicted: the old tail stays.
    assert(!full.CaptureTail(std::span<const int>(ids.data(), 35), policy, 200, reserve - 300, [&] { return make(35); }));
    assert(full.Bytes() == 200 && full.FindTail(ids, policy)->snapshot->hist.size() == 30);
    // Each try is visible (crucible_turn): its length, the result, and why a save was refused.
    assert(full.Status().attempts == 3 && full.Status().entries == 1 && full.Status().evictions == 1);
    assert(full.Status().last.tokens == 35 && !full.Status().last.saved && full.Status().last.refused == "host");
    assert(!full.CaptureTail(std::span<const int>(ids.data(), 36), policy, 300, reserve + 900, [&] { return make(36); }));
    assert(full.Status().last.refused == "budget" && full.Status().attempts == 4);
    // The hit above confirmed the 30 tail: a speculative tail cannot push it out, a trusted one replaces it.
    assert(!full.CaptureTail(std::span<const int>(ids.data(), 37), policy, 200, reserve + 900, [&] { return make(37); }));
    assert(full.Status().last.refused == "protected" && full.Bytes() == 200);
    assert(full.CaptureTail(std::span<const int>(ids.data(), 37), policy, 200, reserve + 900, [&] { return make(37); },
                            true));
    assert(full.Status().last.saved && full.Status().last.tokens == 37 && full.Status().last.refused.empty());
  }
  {
    // Host memory, not the budget, can force eviction: evict only what the reserve needs.
    BatchConversationCache<Snapshot> roomy(1000, ledger);
    const auto make = [&](size_t length) {
      auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + length);
      s->resume.length = length; s->bytes.resize(32);
      return std::shared_ptr<const Snapshot>(s);
    };
    const uint64_t reserve = uint64_t(8) << 30;
    assert(roomy.CaptureTail(std::span<const int>(ids.data(), 10), policy, 200, reserve + 1000, [&] { return make(10); }));
    // The budget has room, but host memory is 100 bytes short: the older tail leaves, the new one is saved.
    assert(roomy.CaptureTail(std::span<const int>(ids.data(), 20), policy, 200, reserve + 100, [&] { return make(20); }));
    assert(roomy.Bytes() == 200 && roomy.FindTail(ids, policy)->snapshot->hist.size() == 20);
  }
  {
    // Session replay (G15): tails at any length, found only by FindTail; the default Find never returns them.
    BatchConversationCache<Snapshot> cache(256, ledger);
    const auto tail = [&](size_t length, int replay) {
      auto s = std::make_shared<Snapshot>(); s->hist.assign(ids.begin(), ids.begin() + length);
      s->resume.length = length; s->resume.replay = replay; s->bytes.resize(32);
      return std::shared_ptr<const Snapshot>(s);
    };
    const std::span<const int> hist(ids.data(), 21);
    assert(cache.CaptureTail(hist, policy, 64, size_t(9)<<30, [&] { return tail(21, 3); }));
    assert(!cache.Find(ids, policy, 39, 40, 8));
    const auto found = cache.FindTail(ids, policy);
    assert(found && found->snapshot->resume.replay == 3 && found->snapshot->hist.size() == 21);
    // The whole prompt must not come from a tail: one row has to predict the first token.
    assert(!cache.FindTail(std::span<const int>(ids.data(), 21), policy));
    auto edited = ids; edited[5] = 0;
    assert(!cache.FindTail(edited, policy));
    // A prompt state, a wrong length, or MTP state is an engine fault (as in Capture), not a cache miss.
    const auto refused = [&](auto create) {
      try { cache.CaptureTail(hist, policy, 64, size_t(9)<<30, create); assert(false); }
      catch (const std::runtime_error&) {}
    };
    refused([&] { return snapshot(21); });
    refused([&] { return tail(20, 0); });
    refused([&] {
      auto s = std::make_shared<Snapshot>(*tail(21, 0)); s->resume.mpos = 4; return std::shared_ptr<const Snapshot>(s);
    });
    auto spec = policy; spec.spec = 2;
    assert(!cache.CaptureTail(hist, spec, 64, size_t(9)<<30, [&] { return tail(21, 0); }));
    assert(!cache.CaptureTail(hist, policy, 64, size_t(7)<<30, [&] { return tail(21, 0); }));
    assert(cache.Bytes() <= 256);
  }
  assert(ledger->allocated == ledger->released);
}
