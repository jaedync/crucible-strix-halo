// The miss ledger: for each open, where the best state for its prompt was (a slot tail, a slot
// point, a host entry, a disk state or a lost record), and why the open did not use it. Closed word sets and numbers
// only, never tokens, text or a hash of text. Diagnostics only: nothing here changes what a restore returns.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace crucible {
// Why the best state was not usable (the turn line's lost_cause):
//   none      the open used the best state (or a state within kLedgerSlack tokens of it)
//   new       no state or record shares half of the prompt or more (new text, or a prompt that changed early)
//   handover  a slot state (a tail or a Frontier point) left its slot at another session's open with no copy
//   evict     a host entry left the host cache (no disk tier, or its disk copy is not known)
//   idle      a host entry of an idle session left the host cache
//   budget    the disk's daily write budget refused the state's write
//   spill     the disk refused or lost the state's write (memory, busy, a failed write)
//   stop      the engine stopped and its flush did not write the state
//   switch    the state was on disk under another state epoch (an engine switch)
//   undo      a live state shares half of the prompt or more, and the prompt leaves it before its end (an undo or edit)
//   unknown   a live state holds the shared prefix and the open did not use it (a reject rule or a refused read)
inline constexpr std::array<const char*, 11> kLostCauses{"none", "new", "handover", "evict", "idle", "budget",
                                                          "spill", "stop", "switch", "undo", "unknown"};
// Where the best state was (best_tier).
inline constexpr std::array<const char*, 6> kBestTiers{"slot", "point", "host", "disk", "lost", "none"};
// What the open used (resume_tier) and the point kind (resume_point).
inline constexpr std::array<const char*, 6> kResumeTiers{"slot", "point", "host", "disk", "block", "none"};
inline constexpr std::array<const char*, 8> kResumePoints{"frontier", "pre", "start", "end", "grid", "header",
                                                           "tail", "none"};
// What a disk read waited for (stage_wait_cause).
inline constexpr std::array<const char*, 4> kStageWaitCauses{"none", "room", "reader", "writer"};
// Where a state that left a slot went (the state_release line).
inline constexpr std::array<const char*, 4> kReleaseOutcomes{"resident", "host", "disk_queued", "lost"};
// A gap below this many tokens is no miss: the lost records keep their shared length at this grid.
inline constexpr size_t kLedgerSlack = 256;

// The word of `set` that equals `word` (its static storage), or nullptr.
template<size_t N>
const char* LedgerWord(std::string_view word, const std::array<const char*, N>& set) {
  for (const char* item : set)
    if (word == item) return item;
  return nullptr;
}
// A refusal word for a log line: a lowercase identifier of at most 24 characters, "-" for none or for anything else.
inline std::string LedgerToken(std::string_view word) {
  if (word.empty() || word.size() > 24) return "-";
  for (const char c : word)
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return "-";
  return std::string(word);
}

// The nearest state of one tier: the tokens that it shares with the prompt, its own length, and for a lost record its
// cause and whether its disk copy exists. `lineage`: the engine's conversation counter of that state (0: none).
struct LedgerNear {
  size_t shared = 0, length = 0;
  const char* cause = nullptr;
  uint64_t lineage = 0;
};
struct LedgerTiers {
  LedgerNear slot, point, host, disk, lost;
};
// The ledger of one open (the turn line's fields).
struct BatchLedger {
  size_t best_shared = 0;
  const char* best_tier = "none";
  const char* lost_cause = "none";
  size_t disk_shared = 0;
  uint64_t lineage = 0;
  const char* resume_tier = "none";
  const char* resume_point = "none";
  size_t resume_gap = 0;
};

// The ledger from the tiers' nearest states and what the open restored (`resumed` tokens from `resume_tier` with
// point kind `resume_point`). A tie goes to the live tier first (slot, point, host, disk, then lost).
inline BatchLedger LedgerVerdict(const LedgerTiers& tiers, size_t prompt, size_t resumed,
                                 const char* resume_tier = "none", const char* resume_point = "none") {
  BatchLedger out;
  out.resume_tier = LedgerWord(resume_tier ? resume_tier : "", kResumeTiers) ? LedgerWord(resume_tier, kResumeTiers) : "none";
  out.resume_point = LedgerWord(resume_point ? resume_point : "", kResumePoints) ? LedgerWord(resume_point, kResumePoints) : "none";
  out.disk_shared = tiers.disk.shared;
  const std::array<std::pair<const char*, const LedgerNear*>, 5> order{{{"slot", &tiers.slot}, {"point", &tiers.point},
                                                                       {"host", &tiers.host}, {"disk", &tiers.disk},
                                                                       {"lost", &tiers.lost}}};
  const LedgerNear* win = nullptr;
  for (const auto& [tier, near] : order)
    if (near->shared && (!win || near->shared > win->shared)) win = near, out.best_tier = LedgerWord(tier, kBestTiers);
  out.best_shared = win ? win->shared : 0;
  out.lineage = win ? win->lineage : 0;
  if (resumed > out.best_shared) {
    out.best_shared = resumed;
    out.best_tier = LedgerWord(out.resume_tier, kBestTiers) ? LedgerWord(out.resume_tier, kBestTiers) : "none";
  }
  out.resume_gap = out.best_shared - std::min(resumed, out.best_shared);
  if (!resumed && out.best_shared < kLedgerSlack) out.lost_cause = "new";
  else if (out.resume_gap < kLedgerSlack) out.lost_cause = "none";
  else if (out.best_shared * 2 < prompt) out.lost_cause = "new";  // defect 2: a record that shares only the block
  else if (std::string_view(out.best_tier) == "lost")
    out.lost_cause = win && win->cause && LedgerWord(win->cause, kLostCauses) ? LedgerWord(win->cause, kLostCauses) : "unknown";
  else out.lost_cause = win && win->shared < win->length ? "undo" : "unknown";
  return out;
}

// The engine line of one state that leaves a slot (Release, Handover, FlushToDisk): where it went and the refusal
// word. Numbers and closed-set words only; a word outside its set prints as "-".
inline std::string ReleaseLine(const char* kind, int slot, size_t tokens, const char* outcome, std::string_view refusal,
                               size_t bytes, double seconds, uint64_t lineage) {
  const char* word = LedgerWord(outcome ? outcome : "", kReleaseOutcomes);
  char line[256];
  std::snprintf(line, sizeof line, "state_release kind=%s slot=%d tokens=%zu outcome=%s refusal=%s bytes=%zu "
                "seconds=%.6f lineage=%llu\n", LedgerToken(kind ? kind : "").c_str(), slot, tokens, word ? word : "-",
                LedgerToken(refusal).c_str(), bytes, seconds, static_cast<unsigned long long>(lineage));
  return line;
}
// slots_durable: the disk index probe of a full key (a SHA-256 of the key, about 0.4 ms at 150K tokens) runs once for
// each key until the disk changes. `written` grows when the disk adds a state, `removed` when it removes one. A held
// key stays held until a removal; a key that was not held is probed again after a write. The memo keeps a 64-bit
// digest of each key and its length in memory only (never on a line or a file), at most kLimit entries. A digest
// collision can only count a slot wrong, and the count changes no decision.
template<class Tenant>
class DurableMemo {
 public:
  static constexpr size_t kLimit = 64;
  template<class Probe>
  bool Held(std::span<const int> key, const Tenant& tenant, uint64_t written, uint64_t removed, Probe&& probe) {
    const uint64_t digest = Digest(key);
    auto seen = std::find_if(seen_.begin(), seen_.end(), [&](const Seen& entry) {
      return entry.digest == digest && entry.tokens == key.size() && entry.tenant == tenant;
    });
    if (seen == seen_.end()) {
      if (seen_.size() >= kLimit) seen_.erase(seen_.begin());
      seen_.push_back({digest, key.size(), tenant, false, 0, 0, false});
      seen = seen_.end() - 1;
    }
    if (!seen->probed || seen->removed != removed || (!seen->held && seen->written != written)) {
      seen->held = probe();
      seen->written = written;
      seen->removed = removed;
      seen->probed = true;
    }
    return seen->held;
  }
  size_t Size() const { return seen_.size(); }

 private:
  struct Seen {
    uint64_t digest;
    size_t tokens;
    Tenant tenant;
    bool held;
    uint64_t written, removed;
    bool probed;
  };
  static uint64_t Digest(std::span<const int> key) {
    uint64_t hash = 1469598103934665603ull;  // FNV-1a over the token values
    for (const int token : key) hash = (hash ^ uint32_t(token)) * 1099511628211ull;
    return hash;
  }
  std::vector<Seen> seen_;
};
}  // namespace crucible
