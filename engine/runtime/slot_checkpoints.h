// In-slot resume points. The device bytes live with the cache hooks; this header
// keeps which positions each slot can resume from, so the rules stay CPU-testable.
//
// A checkpoint holds the fixed-size state of a slot at one position: GDN state, conv ring, pending GDN rows, PLE
// ring, the compact raw ring and the open QSA pool block. KV rows below the position are not copied: the slot keeps
// them, because a stream only appends rows until something resets it or restores over it. So a checkpoint stays
// valid while its slot keeps every row below it.
//
// Start, Grid and End checkpoints resume as a replay tail does (one small-row forward first, then bulk chunks from
// there): contract A, the G15 class (docs/SESSION_REPLAY.md).
//
// An End checkpoint with outputs (re-send fix A) also keeps what the prompt's last row gave: the first
// token's greedy ID, its logits row and the MTP carry. An identical prompt then resumes at the End point with no prompt
// row: every target byte and the first token are copies of the run that saved it, so its output bits are that run's.
// A Pre checkpoint (fix B) is a chunk end at the existing 2,048-row core cut of a long prompt's final
// chunk: a prompt that extends the old one inside its last message resumes there. It resumes as an End point does.
//
// A Frontier checkpoint is a session prompt frontier of a normal turn: a
// grid-aligned state on the default chunk schedule. It resumes as a host frontier hit does (no head forward), and
// the host keeps no copy of it until the slot must give it up (the session's Open guard).
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>
#include "tenant.h"

namespace crucible {
enum class CheckpointKind { Start, Grid, End, Frontier, Pre };
// Two buffers for Start, Grid and End points and one for the newest Frontier point (option B2):
// the same device bytes as three general buffers.
constexpr int kGeneralCheckpoints = 2, kFrontierCheckpoint = kGeneralCheckpoints;
constexpr int kSlotCheckpoints = kGeneralCheckpoints + 1;
// Fix A: the host bytes of the logits rows of `streams` slots (vocabulary `vocab`): one row per general buffer (an End
// point keeps one) and the in-flight row of a final chunk (Request::outputs) before its End save.
constexpr size_t EndLogitsBytes(int streams, size_t vocab) {
  return size_t(streams) * size_t(kGeneralCheckpoints + 1) * vocab * sizeof(float);
}
// A Frontier point sits on a whole QSA pool block (128 rows): its host copy (SlotCheckpointIO::SaveFrom) then reads
// no open pool block. With prefill rows that are not a multiple of 128 (--prefill 0: 8) the host copies the frontier.
constexpr size_t kSlotFrontierRows = 128;
// Whether the slot may keep a frontier at `position`, and resume from it as a host hit: a whole pool block and a
// whole prefill chunk (`rows`: the prefill rows), with no pending rows.
inline bool SlotFrontierGrid(size_t position, int rows, int replay = 0) {
  return position && position % kSlotFrontierRows == 0 && rows > 0 && position % size_t(rows) == 0 && replay == 0;
}

// What the host copy of a Frontier point needs (the metadata of a host frontier at its position).
struct SlotFrontier {
  size_t boundary = 0;       // the request's stable boundary (BatchFrontierResume)
  bool speculative = false;  // a header grid point past the boundary; a hit confirms it
  bool proven = false;       // it extends a proven host state
};

// What the last prompt row gave (fix A). `logits`: shared, so copies of a point do not copy the row (about 1 MiB).
// `carry`: the checkpoint buffer holds the MTP carry of the last prompt row (a speculative request warmed it).
struct PromptOutputs {
  int greedy = -1;
  std::shared_ptr<const std::vector<float>> logits;
  bool carry = false;
};

struct SlotCheckpoint {
  std::vector<int> hist;  // the tokens below the checkpoint
  std::vector<int> key;   // their cache key (empty: the tokens); image rows key by image, not by pad token IDs
  int replay = 0;         // pending GDN rows (a turn start on a restored tail)
  bool guarded = false;   // the state came from a guarded request's tail
  CheckpointKind kind = CheckpointKind::End;
  uint64_t owner = 0;     // the request that took it
  SlotFrontier frontier;  // kind Frontier only
  StateTenant tenant = kLocalTenant;  // W1: whose state it is; only that tenant's requests resume from it
  std::shared_ptr<const PromptOutputs> outputs;  // kind End only: the prompt end of fix A
};
// Whether `point` resumes a prompt of its own length (fix A): an End point with outputs. Every other point needs the
// last prompt row to run.
inline bool ResumesWhole(const SlotCheckpoint& point) {
  return point.kind == CheckpointKind::End && point.outputs && point.outputs->logits;
}
// Where a new Frontier point goes: the slot holds it already (`index`: its buffer), or a free buffer `index`, or
// neither (the host copies it).
struct FrontierPick { bool held = false; std::optional<int> index; };

class SlotCheckpoints {
 public:
  explicit SlotCheckpoints(int slots = 0) : held_(size_t(slots)) {}
  // The general buffer for a new Start, Grid or End checkpoint whose cache key is `key`, or nullopt when the slot
  // already holds that state. A request's later grid point replaces its earlier one; else an empty buffer, else the
  // shortest checkpoint leaves (a Frontier point in a general buffer too: the session moved past it).
  // Invariant: every point is a prefix of the rows on its slot (Keep drops the others), so a point of the same length
  // has the same key. One with another key or another tenant would be stale: it is the one to replace.
  std::optional<int> Victim(int slot, std::span<const int> key, CheckpointKind kind, uint64_t owner,
                            const StateTenant& tenant) const {
    if (kind == CheckpointKind::Frontier) throw std::invalid_argument("a Frontier point takes FrontierVictim");
    const auto& held = Slot(slot);
    for (int i = 0; i < kSlotCheckpoints; ++i)
      if (held[i] && held[i]->key.size() == key.size()) {
        if (held[i]->tenant == tenant && std::equal(key.begin(), key.end(), held[i]->key.begin())) return std::nullopt;
        if (i < kGeneralCheckpoints) return i;
      }
    if (kind == CheckpointKind::Grid)
      for (int i = 0; i < kGeneralCheckpoints; ++i)
        if (held[i] && held[i]->owner == owner && held[i]->kind == CheckpointKind::Grid) return i;
    int victim = 0;
    for (int i = 0; i < kGeneralCheckpoints; ++i) {
      if (!held[i]) return i;
      if (held[i]->hist.size() < held[victim]->hist.size()) victim = i;
    }
    return victim;
  }
  // The buffer for a new Frontier point of request `owner` (MEDIUM-4):
  // - the Frontier buffer when it is empty or holds an older request's point (by the invariant that point is a
  //   prefix of this one: the session moved past it, as a host frontier supersedes the ones that it extends);
  // - else an empty general buffer, so the request's confirmed frontier and its header grid point both stay;
  // - else none: the host copies this frontier as before.
  FrontierPick FrontierVictim(int slot, std::span<const int> key, uint64_t owner, const StateTenant& tenant) const {
    const auto& held = Slot(slot);
    for (int i = 0; i < kSlotCheckpoints; ++i)
      if (held[i] && held[i]->tenant == tenant && std::ranges::equal(held[i]->key, key))
        return held[i]->kind == CheckpointKind::Frontier ? FrontierPick{true, i} : FrontierPick{};
    const auto& frontier = held[kFrontierCheckpoint];
    if (!frontier || frontier->owner != owner) return {false, kFrontierCheckpoint};
    for (int i = 0; i < kGeneralCheckpoints; ++i)
      if (!held[i]) return {false, i};
    return {};
  }
  void Store(int slot, int index, SlotCheckpoint checkpoint) {
    if (index < 0 || index >= kSlotCheckpoints || checkpoint.hist.empty() ||
        (index == kFrontierCheckpoint && checkpoint.kind != CheckpointKind::Frontier))
      throw std::invalid_argument("slot checkpoint needs a buffer index of its kind and tokens");
    if (checkpoint.key.empty()) checkpoint.key = checkpoint.hist;
    if (checkpoint.key.size() != checkpoint.hist.size())
      throw std::invalid_argument("slot checkpoint key and tokens differ in length");
    Slot(slot)[size_t(index)] = std::move(checkpoint);
  }
  // The longest checkpoint of `tenant` whose key `key` (a prompt's cache key) extends by at least one token: the last
  // prompt row must still run. An End point with outputs also serves a key of its own length (ResumesWhole, fix A).
  // `accept` can refuse a point (the host restore rule for Frontier points, MEDIUM-2).
  template <class Accept>
  std::optional<int> Best(int slot, std::span<const int> key, const StateTenant& tenant, Accept accept) const {
    std::optional<int> best;
    const auto& held = Slot(slot);
    for (int i = 0; i < kSlotCheckpoints; ++i) {
      const auto& item = held[i];
      if (!item || item->tenant != tenant || item->key.size() > key.size() ||
          (item->key.size() == key.size() && !ResumesWhole(*item)) ||
          (best && item->key.size() <= held[*best]->key.size()))
        continue;
      if (std::equal(item->key.begin(), item->key.end(), key.begin()) && accept(*item)) best = i;
    }
    return best;
  }
  std::optional<int> Best(int slot, std::span<const int> key, const StateTenant& tenant) const {
    return Best(slot, key, tenant, [](const SlotCheckpoint&) { return true; });
  }
  // The slot's Frontier points, the longest first (the guard copies them in this order, so a shorter one shares the
  // host chunks of the longer one, an earlier change).
  std::vector<int> Frontiers(int slot) const {
    std::vector<int> found;
    const auto& held = Slot(slot);
    for (int i = 0; i < kSlotCheckpoints; ++i)
      if (held[i] && held[i]->kind == CheckpointKind::Frontier) found.push_back(i);
    std::stable_sort(found.begin(), found.end(), [&](int a, int b) { return held[a]->hist.size() > held[b]->hist.size(); });
    return found;
  }
  // The longest Frontier point of the slot (0: none): what another session's open would make the host copy.
  size_t FrontierLength(int slot) const {
    const auto found = Frontiers(slot);
    return found.empty() ? 0 : Slot(slot)[size_t(found.front())]->hist.size();
  }
  // A hit on a speculative Frontier point shows that a later prompt repeats it.
  void Confirm(int slot, int index) {
    auto& item = Slot(slot).at(size_t(index));
    if (item && item->kind == CheckpointKind::Frontier) item->frontier.speculative = false;
  }
  // The longest checkpoint length that the prompt key `key` repeats, for slot affinity (0: none).
  size_t Common(int slot, std::span<const int> key, const StateTenant& tenant) const {
    const auto best = Best(slot, key, tenant);
    return best ? Slot(slot)[size_t(*best)]->hist.size() : 0;
  }
  // The slot keeps only its rows below `keep` (a restore at `keep`, or 0 for a reset or a host restore).
  void Keep(int slot, size_t keep) {
    for (auto& item : Slot(slot))
      if (item && item->hist.size() > keep) item.reset();
  }
  const SlotCheckpoint& At(int slot, int index) const {
    const auto& item = Slot(slot).at(size_t(index));
    if (!item) throw std::logic_error("slot checkpoint is empty");
    return *item;
  }

 private:
  using Held = std::array<std::optional<SlotCheckpoint>, kSlotCheckpoints>;
  Held& Slot(int slot) { return held_.at(size_t(slot)); }
  const Held& Slot(int slot) const { return held_.at(size_t(slot)); }
  std::vector<Held> held_;
};
}  // namespace crucible
