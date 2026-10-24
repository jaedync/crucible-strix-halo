// Cache visibility for crucible_turn: the Oct 5 incident refused every large save, and no log showed it.
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace crucible {
// The cache tiers that a turn's open asks, in their order: the slot (its resident tail
// and its checkpoints), the host memory cache, and the disk. The turn line carries one counters entry for each.
enum class BatchCacheTier : int { Slot = 0, Host = 1, Disk = 2 };
struct BatchTierCounters {
  uint64_t lookups = 0, hits = 0, tokens = 0;
};
// What each tier was asked, what it served, and the tokens that its restores reused. Cumulative since the engine
// started; a turn's line reports the values at its done event. Numbers only, never tokens or text.
struct BatchCacheCounters {
  BatchTierCounters slot, host, disk;
  // One lookup of `tier` that served `tokens` (0: the tier held nothing for the turn).
  void Count(BatchCacheTier tier, size_t tokens) noexcept {
    BatchTierCounters& found = Of(tier);
    ++found.lookups;
    if (!tokens) return;
    ++found.hits;
    found.tokens += tokens;
  }
  BatchTierCounters& Of(BatchCacheTier tier) noexcept {
    return tier == BatchCacheTier::Slot ? slot : tier == BatchCacheTier::Host ? host : disk;
  }
  const BatchTierCounters& Of(BatchCacheTier tier) const noexcept {
    return tier == BatchCacheTier::Slot ? slot : tier == BatchCacheTier::Host ? host : disk;
  }
};
// Why a turn's open cached nothing, and the state nearest to it (the turn line's miss_reason,
// closest_shared_tokens, closest_age_s and closest_outcome). Fixed words and numbers only, never tokens or text.
// The words of `miss` (engine/cache_status.h below, batch_conversation_cache.h Diagnose):
//   none           the turn restored state
//   bypass         the client's cache_control is not "auto"
//   too_short      the prompt has fewer than 2 tokens (nothing can be restored)
//   new            no cached state shares the prompt's first token (a new conversation, or a fresh engine)
//   diverged       the nearest state shares `shared` tokens and then differs (the client's prompt changed)
//   evicted        the nearest state left the host cache (it is on its way to the disk, or gone)
//   not_saved      a save of the nearest state was refused (the turn line's cache_save_refused names why)
//   above_limit    a cached state of the whole prompt: one prompt row must run to give the first token
//   above_prompt   a state longer than the prompt
//   off_grid       a state off the prefill grid
//   schedule       a state of another chunk schedule (boundary or step)
//   rows           a request without a prefill grid
//   tail           a generated tail at a session replay that is off (FindTail holds it)
//   empty          a state of no tokens
//   unknown        the open cached nothing and the worker could not name why
// The server adds "ssd_read_failed" when the turn's own disk read failed (server/metrics.py miss_reason).
struct BatchCacheDiagnosis {
  const char* miss = nullptr;   // nullptr: not reported (the session says "none" for a hit, "unknown" otherwise)
  const char* outcome = nullptr;  // "all" (the whole kept state is shared), "prompt_differs", "reply_differs"
  size_t shared = 0;            // tokens shared with the nearest state
  double age_s = 0;             // seconds since that state was last saved or used
};
// The latest save try. `refused` names why it was not saved: "budget" (larger than the whole cache), "host" (the
// 8 GiB host reserve, even after eviction), "protected" (only live sessions' resume points could make the room),
// "resident" (a copy of a state that its slot still holds, with no free room: the slot keeps it), "policy" (a state
// this cache never holds), or "memory" (allocation).
struct BatchCacheSave {
  size_t tokens = 0;
  bool saved = false;
  std::string refused;
};
struct BatchCacheStatus {
  size_t entries = 0, bytes = 0, evictions = 0;
  uint64_t attempts = 0;
  BatchCacheSave last;
  std::optional<uint64_t> host_available;  // MemAvailable when the status was read
  std::optional<uint64_t> pinned_bytes;    // SnapshotChunkPool::PinnedBytes when the status was read
  BatchCacheCounters counters;             // per-tier lookups, hits and reused tokens
};
}  // namespace crucible
