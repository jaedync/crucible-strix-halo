// One caller owns the engine. The frontend selects ready requests without blocking peers.
#pragma once
#include "../runtime/slot_checkpoints.h"
#include "serve_protocol.h"
#include "draft_tally.h"
#include "../runtime/batch_spec_contract.h"
#include "cache_status.h"
#include "../runtime/prefill_interleave.h"
#include "../model/flashnext/prefill_launch.h"
#include "../runtime/draft_warmup.h"
#include "../runtime/packed_rows_option.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <limits>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace crucible {
// Only pre-device Open validation uses this type. All other failures keep fail-closed semantics.
class RecoverableOpenError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};
inline bool IsRecoverableBatchError(const std::exception& error, std::string_view op, bool poisoned) {
  return op == "open" && !poisoned && dynamic_cast<const RecoverableOpenError*>(&error) != nullptr;
}

// A speculative request's draft counters over its whole life (BatchSpecDecoder): drafts proposed and accepted, and
// draft-and-verify steps. The done event carries them so the server can log them per turn.
struct BatchDraftStats {
  long proposed = 0, accepted = 0, steps = 0;
  std::optional<DraftTally> tally;  // counters by position and by token; unset: only the three above
};
// The same three counters during decode (crucible_decode, live drafts for a log reader): a speculative request's first token
// event of each spec_step or verify_step carries its counts after that step. Counters only, never token IDs.
struct BatchDraftCounts {
  long proposed = 0, accepted = 0, steps = 0;
};

struct BatchEvent {
  uint64_t request_id = 0, slot_generation = 0;
  std::optional<int> token;
  bool done = false, stopped = false;
  size_t prompt_tokens = 0, completion_tokens = 0;
  double prefill_seconds = 0, decode_seconds = 0;
  size_t cached_tokens = 0, cache_bytes = 0;
  double restore_seconds = 0, capture_seconds = 0;
  bool cache_shared = false;
  int slot_context = 0;  // set on the opened event
  bool cache_tail = false;  // the restore was a previous turn's generated tail (session replay, G15)
  bool slot_resume = false; // that tail was still on the request's slot: no host restore
  bool checkpoint_resume = false;  // the restore was an in-slot checkpoint (slot_checkpoints.h)
  bool frontier_copy = false;  // the restore was a copy of another slot's Frontier point
  bool end_resume = false;  // the open resumed the whole prompt from an End point with outputs (re-send fix A)
  int tail_replay = 0;      // that tail's pending GDN rows (evidence for the replay gate)
  // Opened events of image requests: the prefill commands that run encoder slices before the first prompt chunk.
  std::optional<int> vision_slices;
  // A request that ended without a token because an input check failed after its open (an image file that changed).
  // The text is a fixed engine message; the request is closed and its slot is free.
  std::optional<std::string> error;
  // Done events only, when the status hook is set: this request's largest saved prompt frontier, its saved tail,
  // its last refused save, and the cache after the tail save (crucible_turn).
  size_t cache_saved_tokens = 0, cache_tail_tokens = 0, cache_entries = 0, cache_evictions = 0;
  std::string cache_save_refused;
  std::optional<uint64_t> host_available;
  std::optional<uint64_t> cache_pinned_bytes;  // the snapshot pool's pinned chunks, each once (soak memory rule)
  bool cache_reported = false;  // the fields above are set
  // The open's cache diagnosis (boards 364, 426): why it cached nothing, the state nearest to its prompt, and the
  // per-tier counters at the done event. `nullptr` words: not reported (an older worker).
  const char* cache_miss = nullptr;
  const char* closest_outcome = nullptr;
  size_t closest_tokens = 0;
  double closest_age_s = 0;
  BatchCacheCounters counters;
  bool counters_reported = false;
  std::optional<BatchDraftStats> draft;  // done events of speculative requests, when the spec hooks report them
  std::optional<BatchDraftCounts> step_draft;  // token events: the counts after this step (BatchDraftCounts)
  std::vector<TokenLogprob> logprobs;  // the first token's top-N, only when the request asked
};

// The done event's draft counters (JSON members with a leading comma); empty for a request without speculation, so
// its done line stays byte-identical.
inline std::string BatchDraftFields(const BatchEvent& event) {
  if (!event.done || !event.draft) return {};
  return ",\"draft_proposed\":" + std::to_string(event.draft->proposed) +
         ",\"draft_accepted\":" + std::to_string(event.draft->accepted) +
         ",\"spec_steps\":" + std::to_string(event.draft->steps) + (event.draft->tally ? event.draft->tally->Fields() : "");
}

// A token event's step counters (JSON members with a leading comma), with the done event's names; empty for every
// token event without them, so plain requests and the other tokens of a step keep their bytes.
inline std::string BatchStepDraftFields(const BatchEvent& event) {
  if (!event.token || !event.step_draft) return {};
  return ",\"draft_proposed\":" + std::to_string(event.step_draft->proposed) +
         ",\"draft_accepted\":" + std::to_string(event.step_draft->accepted) +
         ",\"spec_steps\":" + std::to_string(event.step_draft->steps);
}

// A fixed log word of the cache diagnosis: a lowercase identifier only, so no request text can reach the line.
inline bool BatchCacheWord(const char* word) {
  if (!word) return false;
  size_t length = 0;
  for (; word[length]; ++length)
    if (!((word[length] >= 'a' && word[length] <= 'z') || word[length] == '_') || length >= 24) return false;
  return length > 0;
}
// The done event's cache counters (JSON members with a leading comma). Empty without the status hook.
inline std::string BatchTierFields(const BatchCacheCounters& counters) {
  const auto pair = [](const char* tier, const BatchTierCounters& found) {
    return std::string(",\"cache_") + tier + "_lookups\":" + std::to_string(found.lookups) + ",\"cache_" + tier +
           "_hits\":" + std::to_string(found.hits) + ",\"cache_" + tier + "_tokens\":" + std::to_string(found.tokens);
  };
  return pair("slot", counters.slot) + pair("host", counters.host) + pair("disk", counters.disk);
}
// The done event's cache fields (JSON members with a leading comma); empty when the cache did not report.
inline std::string BatchCacheFields(const BatchEvent& event) {
  if (!event.done || !event.cache_reported) return {};
  std::string out = ",\"cache_saved_tokens\":" + std::to_string(event.cache_saved_tokens) +
                    ",\"cache_tail_tokens\":" + std::to_string(event.cache_tail_tokens) +
                    ",\"cache_entries\":" + std::to_string(event.cache_entries) +
                    ",\"cache_evictions\":" + std::to_string(event.cache_evictions);
  // Reasons are fixed identifiers (cache_status.h); anything else is not printed into the JSON line.
  if (!event.cache_save_refused.empty() &&
      std::all_of(event.cache_save_refused.begin(), event.cache_save_refused.end(),
                  [](char c) { return c >= 'a' && c <= 'z'; }))
    out += ",\"cache_save_refused\":\"" + event.cache_save_refused + "\"";
  if (event.host_available) out += ",\"host_free_mib\":" + std::to_string(*event.host_available >> 20);
  if (event.cache_pinned_bytes) out += ",\"cache_pinned_bytes\":" + std::to_string(*event.cache_pinned_bytes);
  if (event.slot_resume) out += ",\"slot_resume\":true";
  if (event.checkpoint_resume) out += ",\"checkpoint_resume\":true";
  if (event.frontier_copy) out += ",\"frontier_copy\":true";
  if (event.end_resume) out += ",\"end_resume\":true";
  // The cache diagnosis and the per-tier counters; fixed words and numbers only.
  if (BatchCacheWord(event.cache_miss)) out += ",\"miss_reason\":\"" + std::string(event.cache_miss) + "\"";
  if (BatchCacheWord(event.closest_outcome)) {
    char age[32];
    std::snprintf(age, sizeof age, "%.3f", event.closest_age_s);
    out += ",\"closest_shared_tokens\":" + std::to_string(event.closest_tokens) + ",\"closest_age_s\":" + age +
           ",\"closest_outcome\":\"" + std::string(event.closest_outcome) + "\"";
  }
  if (event.counters_reported) out += BatchTierFields(event.counters);
  return out;
}

// A restore result. `shared` marks a hit on a system-and-tools frontier (G7).
struct BatchRestore {
  size_t tokens = 0;
  bool shared = false;
  // Session replay (G15): a generated tail can carry verified rows that still wait for GDN replay.
  int replay = 0;
  bool tail = false;
  bool slot = false;  // the slot already holds this state; nothing was copied
  bool guarded = false;  // the tail came from a guarded request (its output held reasoning)
  // The tier that served this restore, and the diagnosis of a miss. The restore hook fills
  // both; a slot resume or a slot checkpoint sets the tier itself and leaves the diagnosis at a hit.
  BatchCacheTier tier = BatchCacheTier::Slot;
  BatchCacheDiagnosis diag;
  BatchRestore(size_t tokens = 0, bool shared = false, int replay = 0, bool tail = false, bool slot = false,
               bool guarded = false)
      : tokens(tokens), shared(shared), replay(replay), tail(tail), slot(slot), guarded(guarded) {}
};
// The tail that a finished request left on its slot (session replay): its length, pending GDN rows, and whether
// its request was guarded.
struct BatchResident { size_t length = 0; int replay = 0; bool guarded = false; };
// A slot resume restores a host tail instead only when that tail is longer, or has the same tokens with other
// pending rows; otherwise the slot state is the state the restore would write.
inline bool HostTailFirst(size_t length, int replay, const BatchResident& resident) {
  return length > resident.length || (length == resident.length && replay != resident.replay);
}
// The disk read for a request (state_disk.h): none (no longer state on disk, or no disk tier), reading, ready, or
// failed. `tokens`: the length of the disk state.
// `reason`: why a read did or did not start (BatchDiskTier::Stage, StageReason), for the server's turn line.
struct BatchStage { std::string state = "none"; size_t tokens = 0; std::string reason; uint64_t tag = 0; };
// An earlier change: where a staging turn waits. `position`: its place among the server's waiting long turns (0: next in
// line); `read`: false for a turn behind the first, which only says that it waits and reports a read that it holds.
// `in_slot(key)`: `key` is a prefix of a state that a slot holds or runs (that session's points never leave for a read).
struct BatchStageQueue {
  size_t position = 0;
  bool read = true;
  std::function<bool(std::span<const int> key)> in_slot;
  uint64_t job = 0;  // the server's tag of the turn (log lines only, fix 2 item 10)
};
// How a finished request started: from a previous tail (`replayed`), and whether that tail was guarded. `slot`: the
// slot that holds the finished state.
struct BatchTailSource { bool replayed = false, guarded = false; int slot = -1; };

// Re-send fix B: the shortest prompt that ends a chunk at its final chunk's core cut, so that an
// extended prompt resumes there. Below it the extra launch is a larger share of the prefill.
inline constexpr size_t kPreEndMinTokens = 32768;

// The output rows that an expected tail counts at most: the reserve stays near the usual tail, and a longer
// output pins its extra chunks at the save (about 300 rows per 8 MB chunk).
inline constexpr size_t kTailExpectRows = 4096;

// Worker-owned prefix cache callbacks (G2). Both run inside the device
// command of the request they serve, so a restore or capture never sits between a peer's launches.
struct BatchCacheHooks {
  // Restores the longest eligible prefix into the selected stream. Returns its length, 0 on a miss.
  std::function<BatchRestore(const ServeRequest&)> restore;
  // Runs after a non-final prompt chunk that ends at `cursor`. It may save that frontier.
  std::function<void(const ServeRequest&, size_t cursor)> capture;
  std::function<size_t()> bytes;
  // Session replay (G15, opt-in): runs on the finished request's selected stream at the end of the command that
  // published its last token, before any later command can reuse the slot. `source`: how the request started.
  // Returns the tail state that the slot now holds, if any.
  std::function<std::optional<BatchResident>(const ServeRequest&, std::span<const int> generated, int replay,
                                             BatchTailSource source)> tail;
  // Optional, with `tail`: the request on `slot` may save a tail of about `rows` rows when it ends (its open);
  // 0: no tail is due (after its tail save, or a close). No device work: the host pins the memory of that save while
  // the batch decodes, so the save on the device worker does not.
  std::function<void(const ServeRequest&, int slot, size_t rows)> expect;
  // Optional, with `tail`: the request repeats the tail that its slot still holds. Returns that slot state
  // (`slot` set, nothing copied) or a longer host state that it restored; both give the same output as a restore.
  std::function<BatchRestore(const ServeRequest&, BatchResident)> resume;
  // Optional, all three (slot_checkpoints.h): `checkpoint_save` copies the selected stream's fixed-size state into
  // buffer (slot, index); `checkpoint_load` writes it back over the slot, which still holds every row below it;
  // `peek` is the length that `restore` would give, without a change to the cache.
  std::function<void(int slot, int index)> checkpoint_save;
  std::function<void(int slot, int index, const SlotCheckpoint&)> checkpoint_load;
  std::function<size_t(const ServeRequest&)> peek;
  // Optional: a checkpoint resume left the slot's previous tail at `common` (the replay divergence line, counts only).
  std::function<void(size_t prompt, size_t restored, size_t tail, size_t common)> diverged;
  // Optional, with `resume`: the next request on `slot` overwrites the tail that the slot holds (`resumed`: it
  // continues from that tail). Runs on the selected slot before any write, so the host can save the state that
  // leaves the device ("One copy").
  std::function<void(int slot, bool resumed)> release;
  // Optional, all five with the checkpoint hooks: slot frontiers.
  // `frontier`: the session frontier at `cursor` of a normal turn may stay in the slot instead of a host copy. The
  // hook calls `keep` with the frontier's metadata; it returns true when the slot kept it (no host capture there).
  std::function<bool(const ServeRequest&, size_t cursor, const std::function<bool(const SlotFrontier&)>& keep)> frontier;
  // `frontier_fits`: the host restore rule (Find: boundary, BatchRestoreLimit, prompt check) for a Frontier point.
  std::function<bool(const ServeRequest&, const SlotCheckpoint&)> frontier_fits;
  // `handover`: copies Frontier point (slot, index) to the host. Runs on the selected slot before an open writes it.
  std::function<void(int slot, int index, const SlotCheckpoint&)> handover;
  // `drop`: the open resumes from the slot's own Frontier point, so the tail that the slot holds leaves unsaved.
  std::function<void(int slot)> drop;
  // `resume_host`: with `resume`, whether it would restore a host tail instead of the slot's tail (HostTailFirst).
  std::function<bool(const ServeRequest&, BatchResident)> resume_host;
  // Optional, with the slot frontier hooks: restores a transient host copy of
  // Frontier point (`from`, `index`) of another slot into the selected slot, or refuses (0) with a logged reason.
  // It may select the donor's stream; the session selects the target again on every exit.
  std::function<BatchRestore(const ServeRequest&, int from, int index, const SlotCheckpoint&)> frontier_copy;
  // Optional: the cache key of `tokens` (the request's prompt, or a prefix or extension of it). Slot resume and slot
  // checkpoints compare keys, not token IDs. Without the hook the key is RequestCacheKey: two images have the same
  // pad token IDs, so image tokens key by image (text keys equal the tokens).
  std::function<std::vector<int>(const ServeRequest&, std::span<const int> tokens)> key;
  // Optional, the disk tier (state_disk.h): start or report the read of a disk state longer than `floor` (what a
  // slot already holds for the prompt) and longer than the host cache's state. No device work.
  // `queue`: the turn's place in the server's queue and the slots' keys.
  std::function<BatchStage(const ServeRequest&, size_t floor, const BatchStageQueue& queue)> stage;
  // Optional (fix 2, cause.md item 6): from before an open guards its slot's Frontier points until after its restore,
  // the host cache keeps the entries that the open's prompt may restore (`request`); nullptr ends the hold.
  std::function<void(const ServeRequest* request)> hold;
  // Optional: the ID of the request whose open runs now (log tags of the restore and handover lines, item 10).
  std::function<void(uint64_t id)> request;
  // Optional, log lines only (re-send fix A and B): an open resumed the whole prompt (`tokens`) from an End point
  // with outputs; a prompt saved a Pre point at `tokens`.
  std::function<void(int slot, size_t tokens)> end_resumed;
  std::function<void(int slot, size_t tokens)> pre_saved;
  // Optional: totals and the latest save try. A capture or tail hook that tried a save raises `attempts`.
  std::function<BatchCacheStatus()> status;
  // Optional: a slot resume or a slot checkpoint served `tokens` (0: the slot held nothing for the
  // turn). The per-tier counters of the turn line count the slot tier only; the restore hook counts host and disk.
  std::function<void(size_t tokens)> slot_served;
  // Optional, with `capture`: the first position after `cursor` that capture may save, or SIZE_MAX. Bulk launches that merge
  // grid chunks end there (PrefillLaunchRows). Without it, a worker with `capture` never merges chunks.
  std::function<size_t(const ServeRequest&, size_t cursor)> next_capture;
  // Optional, x4-tail: a joined launch may pass one capture point. `cut_ready`:
  // the request may take a cut. `cut_prepare`: the device copies that put the selected slot's state at `position` in
  // the cut buffer during the launch (Engine::SetPrefillJoin); `window_start` is the start of the launch that ends at
  // `position` without joins, for the buffer's retained ledger. After the launch `cut_save` copies the cut buffer to
  // checkpoint buffer (slot, index), as checkpoint_save copies the slot, and `cut_capture` saves the host frontier
  // at `position` from it, as capture saves it from the slot.
  std::function<bool(const ServeRequest&)> cut_ready;
  std::function<std::vector<PrefillCutCopy>(size_t position, size_t window_start)> cut_prepare;
  std::function<void(int slot, int index)> cut_save;
  std::function<void(const ServeRequest&, size_t position)> cut_capture;
};

// Speculative decode callbacks (G6a). Each runs inside the command of the
// request it serves, after SelectStream, so per-request draft state never meets a peer's launches.
struct BatchSpecHooks {
  // Returns an empty string when this engine can decode the request's resolved policy exactly.
  std::function<std::string(const ServeRequest&)> check;
  std::function<void(uint64_t id, int slot, const ServeRequest&, size_t cached)> open;
  // Runs after each prompt chunk [start, start+rows); `bulk` marks the GEMM prefill path.
  std::function<void(uint64_t id, size_t start, int rows, bool bulk)> chunk;
  // Optional: the first row of chunk [start, start+rows) whose final trunk stacks `chunk` reads (rows if none).
  // The engine may skip the last layer's output for rows below it. Without it, every row is kept.
  std::function<int(uint64_t id, size_t start, int rows)> stacks;
  std::function<void(uint64_t id, int first)> first;
  // One solo draft-and-verify step. Returns the accepted tokens plus the correction (1 to 8 IDs).
  // `remaining` counts tokens still allowed; `position` is the absolute position of the next token (keyed law).
  std::function<std::vector<int>(uint64_t id, int last, size_t remaining, uint64_t position)> step;
  std::function<void(uint64_t id)> close;
  // Optional: the request's draft counters, read at its done event before `close` erases them.
  std::function<BatchDraftStats(uint64_t id)> stats;
  // Optional: the three scalar counters only (no tally copy), read after each speculative step (BatchDraftCounts).
  std::function<BatchDraftCounts(uint64_t id)> counts;
  // Target rows that still wait for GDN replay (session replay saves them with the tail).
  std::function<int(uint64_t id)> replay;
  // Load-adaptive decode (G6c): a speculative request may join a packed step. `commit` runs first on the
  // request's stream. When verified rows still wait for GDN replay, it runs one solo plain row and returns its
  // greedy ID, because a packed pass needs committed state. `packed` then sees each such request's row of the
  // pass with the token it emitted, so the draft model can catch up before the next solo step.
  std::function<std::optional<int>(uint64_t id, int last)> commit;
  std::function<void(const std::vector<uint64_t>& ids, const std::vector<int>& rows, const std::vector<int>& tokens)> packed;
  // Packed verification (G6d). `propose` drafts up to `cap` tokens on the request's stream and returns its verify
  // rows (the last token first) with its pending replay count. After one shared pass, `accept` verifies the
  // request's rows [first, first + rows) and returns the emitted tokens; `settle` then runs deferred draft work.
  std::function<VerifySpan(uint64_t id, int last, size_t remaining, uint64_t position, int cap)> propose;
  std::function<std::vector<int>(uint64_t id, int first, std::span<const int> greedy)> accept;
  std::function<void()> settle;
  // Optional: drafts a request could copy from its own history now (prompt lookup), 0 if none. VerifyStep gives
  // spare rows to such requests first (DraftCapsFor). Like a cap, it changes speed only.
  std::function<int(uint64_t id, int last)> want;
  // Optional: propose for several requests at once (grouped MTP passes). Same contract as `propose` per request;
  // `last`, `remaining`, `position` and `cap` are per request, in the order of `ids`.
  std::function<std::vector<VerifySpan>(const std::vector<uint64_t>& ids, const std::vector<int>& last,
                                        const std::vector<size_t>& remaining, const std::vector<uint64_t>& position,
                                        const std::vector<int>& cap)> propose_all;
};

// Spare rows of a packed verify pass, after one row per request, go to speculative requests in equal shares;
// `turn` rotates the remainder. A cap changes only how many drafts a request proposes, never its output. One span
// holds at most kMaxRows rows (SpecSpansError), so no cap exceeds kMaxRows - 1; that binds only above 8 rows.
inline std::vector<int> DraftCaps(const std::vector<bool>& speculative, int rows, uint64_t turn) {
  std::vector<int> caps(speculative.size(), 0);
  std::vector<size_t> order;
  for (size_t i = 0; i < speculative.size(); ++i) if (speculative[i]) order.push_back(i);
  const int spare = rows - int(speculative.size());
  if (order.empty() || spare <= 0) return caps;
  const int share = spare / int(order.size()), extra = spare % int(order.size());
  for (size_t k = 0; k < order.size(); ++k) {
    const size_t index = order[(k + turn) % order.size()];
    caps[index] = std::min(kMaxRows - 1, share + (int(k) < extra ? 1 : 0));
  }
  return caps;
}

// DraftCaps with priority: requests that want copy rows (want > 0) take spare rows first, longest want first
// (index order on ties), each up to its want. The rows left go to the other speculative requests as DraftCaps does.
inline std::vector<int> DraftCapsFor(const std::vector<bool>& speculative, const std::vector<int>& want, int rows,
                                     uint64_t turn) {
  std::vector<int> caps(speculative.size(), 0);
  int spare = rows - int(speculative.size());
  std::vector<size_t> order;
  for (size_t i = 0; i < speculative.size(); ++i) if (speculative[i] && want[i] > 0) order.push_back(i);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return want[a] > want[b]; });
  for (const size_t i : order) {
    caps[i] = std::max(0, std::min({want[i], spare, kMaxRows - 1}));
    spare -= caps[i];
  }
  std::vector<bool> rest(speculative.size(), false);
  for (size_t i = 0; i < speculative.size(); ++i) rest[i] = speculative[i] && want[i] <= 0;
  // DraftCaps reserves one row per listed request; every request here already holds its row.
  const auto shared = DraftCaps(rest, std::max(0, spare) + int(rest.size()), turn);
  for (size_t i = 0; i < caps.size(); ++i) if (rest[i]) caps[i] = shared[i];
  return caps;
}

template <class E>
class BatchSession {
 public:
  BatchSession(E& engine, int slots, uint64_t cache_budget = 0, int short_slots = 0, int short_ctx = 0)
      : engine_(engine), slots_(slots) {
    if (slots < 1 || slots > kMaxRows) throw std::invalid_argument("batch slots must be in [1,8]");
    if (short_slots < 0 || short_slots >= slots || (short_slots && (short_ctx < 1 || short_ctx >= engine.Context())))
      throw std::invalid_argument("short slots need 1..slots-1 slots and a context below the full context");
    // Live states and the whole cache budget share one admission check.
    engine_.ConfigureServeStreams(slots, cache_budget, short_slots, short_ctx);
    resident_.resize(size_t(slots));
    checkpoints_ = SlotCheckpoints(slots);
    for (int slot = 0; slot < slots; ++slot) {
      contexts_.push_back(slot < slots - short_slots ? engine.Context() : short_ctx);
      // Short slots may carry verification headroom beyond their admitted context.
      if (engine_.StreamCapacity(slot) < contexts_.back()) throw std::logic_error("slot capacity below its class");
    }
  }
  void SetSpecHooks(BatchSpecHooks hooks) { spec_ = std::move(hooks); }
  // Re-send fix A (always on with slot checkpoints): End points keep the prompt outputs; an identical prompt resumes
  // at its End point with no prompt row.
  // Re-send fix B: a prompt of at least `min_tokens` tokens ends a chunk at its final chunk's
  // core cut (`core_rows` past the chunk start, an existing cut of the GDN and QSA core parts) and keeps a Pre point
  // there. 0: off.
  void SetPreEnd(size_t min_tokens, int core_rows = kPrefillLaunchCoreRows) {
    if (core_rows <= kMaxRows) throw std::invalid_argument("a pre-end cut needs core parts of more than 8 rows");
    pre_end_min_ = min_tokens;
    core_rows_ = core_rows;
  }
  const std::vector<int>& SlotContexts() const { return contexts_; }
  void SetCacheHooks(BatchCacheHooks hooks) { cache_ = std::move(hooks); }
  size_t Active() const { return active_.size(); }
  // The prompt cursor after the last Prefill. The frontend ranks waiting turns by the rows that remain (SRPT).
  // An encoder slice moves no prompt row, so it leaves no cursor.
  std::optional<size_t> PrefillCursor() const { return prefill_cursor_; }
  bool Poisoned() const { return poisoned_; }

  std::vector<BatchEvent> Open(uint64_t id, const ServeRequest& request, bool deferred = false) {
    SettlePrompt();
    ValidateOpen(id, request);
    const int slot = FreeSlot(request);
    if (generations_[slot] == std::numeric_limits<uint64_t>::max())
      throw std::overflow_error("sequence slot generation limit reached; restart the worker");
    auto state = std::make_unique<Request>(id, slot, ++generations_[slot], request);
    std::optional<BatchResident> resident;
    const bool held = resident_[size_t(slot)].has_value();
    const std::vector<int> key = Key(request, request.ids);
    if (cache_.resume && Resumable(slot, request, key))
      resident = BatchResident{resident_[size_t(slot)]->key.size(), resident_[size_t(slot)]->replay,
                               resident_[size_t(slot)]->guarded};
    // frontier-copy: another slot's Frontier point that is longer than the host state and this slot's own point.
    const auto donor = resident ? std::nullopt : Donor(slot, request, key);
    const auto point = resident || donor ? std::nullopt : Checkpoint(slot, request, key);
    const bool from_frontier = point && checkpoints_.At(slot, *point).kind == CheckpointKind::Frontier;
    // Where the prompt leaves the slot's previous tail, for the divergence line of a checkpoint resume.
    std::optional<std::pair<size_t, size_t>> left;
    if (point && resident_[size_t(slot)]) {
      const auto& tail = resident_[size_t(slot)]->key;
      const size_t limit = std::min(tail.size(), key.size());
      const size_t common = size_t(std::mismatch(tail.begin(), tail.begin() + std::ptrdiff_t(limit), key.begin()).first -
                                   tail.begin());
      if (common < tail.size()) left = {tail.size(), common};
    }
    resident_[size_t(slot)].reset();  // this request writes the slot
    DeviceUpdate([&] {
      engine_.SelectStream(slot);
      if (cache_.request) cache_.request(id);
      // The hold ends on every exit of this block, also on a throw.
      if (cache_.hold) cache_.hold(&request);
      const struct Unhold {
        const BatchCacheHooks& hooks;
        ~Unhold() { if (hooks.hold) hooks.hold(nullptr); }
      } unhold{cache_};
      // a Frontier point that this open does not keep goes to the host before any write.
      GuardFrontiers(slot, request, resident, point);
      // HIGH-2: a resume from the slot's own Frontier point leaves the slot's tail, which the prompt does not repeat.
      if (held && from_frontier && cache_.drop) cache_.drop(slot);
      else if (held && cache_.release) cache_.release(slot, resident.has_value());
      if (!resident && !point) engine_.Reset();
      Request& owned = *state;
      active_.emplace(id, std::move(state));
      if (cache_.restore) {
        const double start = Clock();
        BatchRestore hit;
        if (point) {
          const SlotCheckpoint& saved = checkpoints_.At(slot, *point);
          cache_.checkpoint_load(slot, *point, saved);
          if (saved.kind == CheckpointKind::Frontier) {
            // A host frontier hit at the same position (MEDIUM-1): no head forward; merged launches, captures and
            // no Start point, as after that hit.
            hit = BatchRestore{saved.hist.size()};
            checkpoints_.Confirm(slot, *point);
          } else {
            owned.grid_point = saved.kind == CheckpointKind::Grid;
            hit = BatchRestore{saved.hist.size(), false, saved.replay, true, false, saved.guarded};
            // Fix A: the End point of this prompt (Best gives a point of the prompt's length only then).
            if (saved.hist.size() == owned.options.ids.size() && ResumesWhole(saved)) owned.end_outputs = saved.outputs;
          }
        } else {
          if (donor) hit = CopyFrontier(slot, owned.options, *donor);
          owned.frontier_copy = hit.tokens > 0;
          if (!owned.frontier_copy)
            hit = resident ? cache_.resume(owned.options, *resident) : cache_.restore(owned.options);
        }
        // A host restore resets the stream itself; a miss after all must not keep the old slot state.
        if (resident && !hit.slot && !hit.tokens) engine_.Reset();
        if (hit.slot && (!resident || hit.tokens != resident->length || !hit.tail))
          throw std::logic_error("slot resume must keep the resident tail");
        // Only a slot or checkpoint resume keeps the slot's rows (below the restored length).
        checkpoints_.Keep(slot, hit.slot || point ? hit.tokens : 0);
        owned.from_checkpoint = point.has_value();
        if (left && left->second > hit.tokens && cache_.diverged)
          cache_.diverged(owned.options.ids.size(), hit.tokens, left->first, left->second);
        owned.cached = owned.cursor = hit.tokens;
        owned.cache_shared = hit.shared;
        owned.replay = hit.replay;
        owned.from_tail = hit.tail;
        owned.slot_resume = hit.slot;
        owned.tail_guarded = hit.tail && hit.guarded;
        owned.tail_replay = hit.tail ? hit.replay : 0;
        // An earlier change: the miss cause and the state nearest to the prompt, and the tier that served this open.
        owned.diag = hit.diag;
        if (hit.tokens) owned.diag.miss = "none";
        else if (!owned.diag.miss) owned.diag.miss = "unknown";
        // An earlier change: the open asks the slot tier first; the restore hook counts the host and disk tiers.
        const bool slot_served = point || donor || hit.slot || owned.frontier_copy;
        if (cache_.slot_served) cache_.slot_served(slot_served ? hit.tokens : 0);
        // The slot keeps the copied state as its own Frontier point: the host holds no copy of it (frontier-copy.md).
        if (owned.frontier_copy) KeepFrontier(owned);
        owned.restore = Clock() - start;
        // The last prompt row must run here: it produces the first token's logits. Fix A: an End point with outputs
        // holds that row's results.
        if (owned.cached >= owned.options.ids.size() && !(owned.end_outputs && owned.cached == owned.options.ids.size()))
          throw std::logic_error("prefix cache restored the whole prompt");
        owned.end_resumed = owned.end_outputs != nullptr;
        if (owned.end_resumed && cache_.end_resumed) cache_.end_resumed(slot, owned.cached);
        // A turn that starts on a tail can be retried or edited from here (an End resume is that point already).
        if (hit.tail && hit.tokens && !owned.end_outputs)
          SaveCheckpoint(owned, CheckpointKind::Start, hit.replay, owned.tail_guarded);
      }
      // On the selected slot, before any hook can select another stream.
      if (cache_.tail && cache_.expect && owned.options.cache_control != "bypass")
        cache_.expect(owned.options, slot, owned.options.ids.size() +
                                           std::min(size_t(std::max(owned.options.max_tokens, 0)), kTailExpectRows));
      // Every open rebinds, so a text request never sees a previous request's images. The encoder work waits
      // for this request's prefill commands, so peers decode between its slices.
      if constexpr (requires { engine_.BindImages(engine_.ActiveSequence(), owned.options.images, size_t{0}, true); }) {
        owned.vision = engine_.BindImages(engine_.ActiveSequence(), owned.options.images, owned.cached, true);
        owned.vision_slices = engine_.VisionSlicesLeft(engine_.ActiveSequence());
      }
      if (owned.spec) spec_.open(id, slot, owned.options, owned.cached);
    });
    if (deferred) {
      BatchEvent opened{id, generations_[slot], {}, false, false, request.ids.size()};
      opened.slot_context = contexts_[slot];
      opened.cached_tokens = active_.at(id)->cached;
      opened.cache_miss = active_.at(id)->diag.miss;
      opened.closest_tokens = active_.at(id)->diag.shared;
      opened.closest_age_s = active_.at(id)->diag.age_s;
      opened.closest_outcome = active_.at(id)->diag.outcome;
      if (!request.images.empty()) opened.vision_slices = active_.at(id)->vision_slices;
      return {opened};
    }
    for (;;) {
      auto events = Prefill(id);
      if (!events.empty()) return events;
    }
  }

  // every Frontier point as (slot, buffer index, point), the longest of each slot first. A stopping engine
  // writes them to disk before the held tails (BatchPrefixCache::FlushToDisk).
  std::vector<std::tuple<int, int, SlotCheckpoint>> FrontierPoints() const {
    std::vector<std::tuple<int, int, SlotCheckpoint>> points;
    for (int slot = 0; slot < slots_; ++slot)
      for (const int index : checkpoints_.Frontiers(slot)) points.emplace_back(slot, index, checkpoints_.At(slot, index));
    return points;
  }

  // The disk tier: start the read of a disk state for `request` before its open, so the open restores from memory
  // and the device worker never waits for the disk. Reads no device state and changes no slot.
  BatchStage Stage(const ServeRequest& request, size_t position = 0, bool read = true, uint64_t job = 0) const {
    RequireHealthy();
    if (!cache_.stage || request.cache_control != "auto" || request.ids.size() < 2) return {};
    const std::vector<int> key = Key(request, request.ids);
    size_t floor = 0;
    for (int slot = 0; slot < slots_; ++slot) {
      if (Resumable(slot, request, key)) floor = std::max(floor, resident_[size_t(slot)]->key.size());
      if (cache_.checkpoint_load)
        if (const auto best = BestPoint(slot, request, key)) floor = std::max(floor, checkpoints_.At(slot, *best).hist.size());
    }
    // The slots' keys, made only when a read must evict: the held states and the running requests' prompts.
    std::optional<std::vector<std::vector<int>>> slot_keys;
    const auto in_slot = [&](std::span<const int> prefix) {
      if (!slot_keys) {
        slot_keys.emplace();
        for (const auto& held : resident_) if (held) slot_keys->push_back(held->key);
        for (const auto& [id, state] : active_) slot_keys->push_back(Key(state->options, state->options.ids));
      }
      return std::any_of(slot_keys->begin(), slot_keys->end(), [&](const std::vector<int>& held) {
        return prefix.size() <= held.size() && std::equal(prefix.begin(), prefix.end(), held.begin());
      });
    };
    return cache_.stage(request, floor, BatchStageQueue{position, read, in_slot, job});
  }

  // Interleaved prefill: verify passes for `decoders` run between the layers of a bulk
  // chunk, every `every` layers; `part` receives each pass's events, tail saves done, before the chunk continues.
  // Decoders that finish drop out of later passes. A small-row chunk runs no pass.
  using PartSink = std::function<void(std::vector<BatchEvent>)>;
  std::vector<BatchEvent> Prefill(uint64_t id, const std::vector<uint64_t>& decoders = {}, int every = 0,
                                  const PartSink& part = {}) {
    SettlePrompt();
    RequireHealthy();
    if (!active_.contains(id) || active_.at(id)->last >= 0)
      throw std::invalid_argument("request is not waiting for prefill");
    if (!decoders.empty()) {
      if (every <= 0 || !part) throw std::invalid_argument("interleaved prefill needs a layer interval and a sink");
      if (std::find(decoders.begin(), decoders.end(), id) != decoders.end())
        throw std::invalid_argument("interleaved prefill cannot decode its own request");
      CheckVerify(decoders);
    }
    // The next image's pixels are read and checked on the host before any device work, outside DeviceUpdate: a file
    // that changed after the open ends only this request, and the session stays healthy (review F1).
    prefill_cursor_.reset();  // a command that ends early (below) names no cursor
    if constexpr (requires { engine_.PrepareVisionPixels(engine_.ActiveSequence()); }) {
      const Request& waiting = *active_.at(id);
      engine_.SelectStream(waiting.slot);
      // The longest chunk this command can run (a merged bulk launch, below, can pass the grid); a head chunk is
      // shorter, so at most it reads one chunk early.
      const size_t step = size_t(engine_.PrefillRows() > 0
                                     ? std::max(engine_.PrefillRows(), EngineLaunchRows(engine_) + EngineJoinRows(engine_))
                                     : kMaxRows);
      const size_t end = waiting.cursor + std::min(step, waiting.options.ids.size() - waiting.cursor);
      try {
        if (engine_.VisionSlicesBefore(engine_.ActiveSequence(), end) > 0)
          engine_.PrepareVisionPixels(engine_.ActiveSequence());
      }
      catch (const std::runtime_error& error) { return FailRequest(id, error.what()); }
      // The host buffer of one image (up to 12 MB) is the only allocation here; no state changed.
      catch (const std::bad_alloc&) { return FailRequest(id, ImageBufferError()); }
    }
    PrefillBetween between{every, [&] {
      std::vector<uint64_t> live;
      for (const uint64_t decoder : decoders) if (active_.contains(decoder)) live.push_back(decoder);
      if (live.empty()) return;
      auto events = VerifyEvents(live, CheckVerify(live));
      SaveTails(&events);
      part(std::move(events));
    }};
    return DeviceUpdate([&]() -> std::vector<BatchEvent> {
    Request& state = *active_.at(id);
    engine_.SelectStream(state.slot);
    prefill_cursor_.reset();
    if (state.end_outputs) return ResumeEnd(state);
    const int prefill = engine_.PrefillRows();
    const int step = prefill > 0 ? prefill : kMaxRows;
    const auto& ids = state.options.ids;
    // A restored tail first runs one small-row Forward (with its pending GDN rows, if any), as serial RunPrompt does.
    const int replay = state.replay;
    // A Grid checkpoint is a chunk end of a replayed turn, after its head: the resume continues that turn's chunk
    // schedule, so it gives the same rows. A second head would shift every later chunk end.
    const bool head = replay || (state.from_tail && state.cursor == state.cached && !state.grid_point);
    const bool bulk = prefill > 0 && !head;
    int rows = int(std::min<size_t>(head ? std::min(step, kMaxRows) : step, ids.size() - state.cursor));
    std::optional<JoinedLaunch> joined;
    const size_t pre = bulk ? PreEndPoint(state, step) : 0;
    // Whole grid chunks merge into one launch when no capture point falls between them and the launch ends before
    // the MTP warm-up rows; a replayed turn keeps the grid (it saves a slot checkpoint at each chunk end). With
    // decoders waiting the grid stays: a merged layer is longer, so the decode gap grows (3 x 32K, interleave 10:
    // 0.75 s against 0.65 s) and the extra passes cancel the launch gain (TTFT 27.1 s against 27.0 s).
    if (bulk && !state.from_tail && decoders.empty() && EngineLaunchRows(engine_) > step &&
        (!cache_.capture || cache_.next_capture)) {
      size_t stop = cache_.capture ? cache_.next_capture(state.options, state.cursor) : SIZE_MAX;
      if (state.spec) stop = std::min(stop, DraftWarmupStart(ids.size(), state.options.mtp_prefix));
      if (pre && pre > state.cursor) stop = std::min(stop, pre);
      rows = PrefillLaunchRows(state.cursor, ids.size(), step, EngineLaunchRows(engine_), stop);
      // x4-tail: with join rows the prompt's last partial chunk joins the launch before it, and a launch may pass one
      // capture point (the cut). The engine gets the spec right before the launch (below).
      // Fix B: a join that reaches over the Pre point is dropped (the plain launch above ends at the point), and no
      // launch from the point on joins: the plan without joins has no ledger off the grid.
      if (EngineJoinRows(engine_) > 0 && !(pre && state.cursor >= pre)) {
        auto join = JoinLaunch(state, step);
        if (!(pre > state.cursor && pre < state.cursor + size_t(join.rows))) {
          rows = join.rows;
          joined = std::move(join);
        }
      }
    }
    // Fix B: the chunk that holds the Pre point ends there (a launch never passes it, above).
    if (pre > state.cursor && pre < state.cursor + size_t(rows)) rows = int(pre - state.cursor);
    const bool final = state.cursor + rows == ids.size();
    // The images that this chunk reads get their encoder slices first, one slice per command, so peers decode
    // between them; the embeddings of images behind the cursor are freed after the chunk.
    if constexpr (requires { engine_.VisionSlicesBefore(engine_.ActiveSequence(), size_t{0}); })
      if (engine_.VisionSlicesBefore(engine_.ActiveSequence(), state.cursor + size_t(rows)) > 0) {
        state.vision += engine_.EncodeVisionSlice(engine_.ActiveSequence());
        return {};
      }
    const double start = Clock();
    const int keep = !state.spec ? rows : spec_.stacks ? spec_.stacks(id, state.cursor, rows) : 0;
    // The PLE rows of the next two chunks start their disk reads now (CRUCIBLE_PLE_HOST).
    if constexpr (requires { engine_.HintPle(ids, 0, 0); })
      if (bulk) engine_.HintPle(ids, state.cursor + rows, state.cursor + rows + 2 * size_t(step));
    if (joined) {
      if (joined->cut) {
        joined->spec.copies = cache_.cut_prepare(joined->cut, joined->cut_window);
        if (joined->spec.copies.empty()) throw std::logic_error("a prefill cut needs device copies");
      }
      if constexpr (requires { engine_.SetPrefillJoin(PrefillJoinSpec{}); }) engine_.SetPrefillJoin(joined->spec);
    }
    const auto output = bulk
        ? engine_.Prefill(ids.data() + state.cursor, rows, "", final, decoders.empty() ? nullptr : &between, keep)
        : engine_.Forward(ids.data() + state.cursor, rows, replay, rows);
    state.replay = 0;
    // x4-tail: the frontier inside the launch is saved first, as the launch that ended there saved it.
    if (joined && joined->cut) SaveCut(state, joined->cut);
    // A final chunk samples before the speculative hook: MTP pairing reuses the logits buffer, and serial
    // RunPrompt also reads the prompt logits before it pairs the MTP rows.
    if (final && output.empty()) throw std::runtime_error("engine returned no prompt prediction");
    const int row = bulk ? 0 : rows - 1;
    // Fix A: the End point keeps the first token's greedy ID and logits row, copied before MTP pairing reuses the
    // buffer. The carry is the final chunk's MTP carry, which a speculative request warms before the End save.
    if (final && cache_.checkpoint_save && state.options.cache_control != "bypass")
      state.outputs = std::make_shared<const PromptOutputs>(PromptOutputs{
          output.back(), std::make_shared<const std::vector<float>>(engine_.CopyLogits(1, row)), state.spec});
    const int token = final ? Sample(state, output.back(), row, state.outputs ? state.outputs->logits : nullptr) : -1;
    // A first token that does not end its request is published before the rest of the prompt end:
    // SettlePrompt runs it before any later device work, in the old order. Outputs do not depend on drafts.
    if (final && !Ends(state, token)) {
      prompt_end_ = PromptEnd{id, state.cursor, rows, bulk, token};
      state.cursor += rows;
      prefill_cursor_ = state.cursor;
      // The MTP rows take token embeddings only (no image overlay), so the image rows go now.
      if constexpr (requires { engine_.ReleaseImageRows(engine_.ActiveSequence(), size_t{0}); })
        engine_.ReleaseImageRows(engine_.ActiveSequence(), state.cursor);
      state.prefill += Clock() - start;
      return Publish(id, token);
    }
    if (state.spec) spec_.chunk(id, state.cursor, rows, bulk);
    state.cursor += rows;
    prefill_cursor_ = state.cursor;
    if constexpr (requires { engine_.ReleaseImageRows(engine_.ActiveSequence(), size_t{0}); })
      engine_.ReleaseImageRows(engine_.ActiveSequence(), state.cursor);
    if (!final) {
      // A replayed turn saves no prompt frontiers (below), so its slot keeps the latest chunk boundary instead.
      if (state.from_tail) SaveCheckpoint(state, CheckpointKind::Grid, 0, false);
      state.prefill += Clock() - start;
      // Fix B: the Pre point is not a grid point, so no host frontier or slot Frontier point is due there.
      if (state.cursor == pre) {
        if (SaveCheckpoint(state, CheckpointKind::Pre, 0, false) && cache_.pre_saved) cache_.pre_saved(state.slot, pre);
        return {};
      }
      // A replayed turn's rows follow a shifted chunk schedule, so its prompt frontiers are not default-path state.
      if (cache_.capture && !state.from_tail) {
        const double saved = Clock();
        const uint64_t tried = CacheAttempts();
        if (!KeepFrontier(state)) cache_.capture(state.options, state.cursor);
        NoteSave(tried, state.saved_tokens, state.refused);
        state.capture += Clock() - saved;
      }
      return {};
    }
    FinishPrompt(state, token);
    state.prefill += Clock() - start;
    return Publish(id, token);
    });
  }

  // The rest of a published prompt end: the final chunk's MTP warm-up, the draft start and the End
  // checkpoint. The serve loop calls it after each command's events; every command calls it first.
  void SettlePrompt() {
    if (!prompt_end_) return;
    const PromptEnd end = *std::exchange(prompt_end_, std::nullopt);
    if (!active_.contains(end.id)) return;
    DeviceUpdate([&] {
      Request& state = *active_.at(end.id);
      const double start = Clock();
      engine_.SelectStream(state.slot);
      if (state.spec) spec_.chunk(end.id, end.start, end.rows, end.bulk);
      FinishPrompt(state, end.token);
      state.prefill += Clock() - start;
    });
  }
  // The serve loop's call after a command's boundary: the text of a failed prompt end (the session is then poisoned),
  // which the loop sends as an error event.
  std::optional<std::string> TrySettlePrompt() {
    try { SettlePrompt(); }
    catch (const std::exception& error) { return std::string(error.what()); }
    return std::nullopt;
  }

  std::vector<BatchEvent> Step(const std::vector<uint64_t>& ids) {
    SettlePrompt();
    RequireHealthy();
    if (ids.empty() || ids.size() > size_t(slots_))
      throw std::invalid_argument("batch step must contain one to eight active requests");
    std::set<uint64_t> unique;
    for (uint64_t id : ids) {
      if (!unique.insert(id).second || !active_.contains(id) || active_.at(id)->last < 0)
        throw std::invalid_argument("batch step contains a duplicate or inactive request");
      if (active_.at(id)->spec && (!spec_.commit || !spec_.packed))
        throw std::invalid_argument("this worker decodes a speculative request only with spec_step");
    }
    return DeviceUpdate([&] {
    std::vector<BatchEvent> events;
    std::vector<uint64_t> batch;
    std::vector<int> tokens, slots;
    for (uint64_t id : ids) {
      Request& state = *active_.at(id);
      if (state.spec) {
        engine_.SelectStream(state.slot);
        const double start = Clock();
        if (const auto greedy = spec_.commit(id, state.last)) {
          const int token = Sample(state, *greedy, 0);
          spec_.packed({id}, {0}, {token});
          state.decode += Clock() - start;
          const auto one = Publish(id, token);
          events.insert(events.end(), one.begin(), one.end());
          continue;
        }
      }
      batch.push_back(id);
      tokens.push_back(state.last);
      slots.push_back(state.slot);
    }
    if (batch.empty()) return events;
    const double start = Clock();
    const auto greedy = engine_.ForwardBatch(tokens, slots);
    if (greedy.size() != batch.size()) throw std::runtime_error("engine returned the wrong batch width");
    std::vector<int> output;
    for (size_t row = 0; row < batch.size(); ++row)
      output.push_back(Sample(*active_.at(batch[row]), greedy[row], int(row)));
    // The draft catch-up reads the pass's rows, so it runs before any publication can close a request.
    std::vector<uint64_t> spec_ids;
    std::vector<int> spec_rows, spec_tokens;
    for (size_t row = 0; row < batch.size(); ++row)
      if (active_.at(batch[row])->spec) {
        spec_ids.push_back(batch[row]); spec_rows.push_back(int(row)); spec_tokens.push_back(output[row]);
      }
    if (!spec_ids.empty()) spec_.packed(spec_ids, spec_rows, spec_tokens);
    const double elapsed = Clock() - start;
    for (size_t row = 0; row < batch.size(); ++row) {
      active_.at(batch[row])->decode += elapsed;
      const auto one = Publish(batch[row], output[row]);
      events.insert(events.end(), one.begin(), one.end());
    }
    return events;
    });
  }

  // Rows of one packed verification pass (packed_rows_option.h): kMaxRows unless set. A fixed cap of 9 to 16 needs
  // an engine that runs such passes (Engine::MaxPassRows); else this refuses it. With wide_from > 0 (automatic),
  // a pass of fewer decoders uses kMaxRows, and the setting falls back to kMaxRows where a wide pass cannot run
  // (fewer slots than wide_from: then the engine is not asked, because its first MaxPassRows call builds the
  // 16-row plans). Speed only: the rows never change a request's output.
  void SetPassRows(int rows, int wide_from = 0) {
    static_assert(kMaxRows == 8 && kPackedRows == 16, "the messages name the limits");
    if (rows < kMaxRows || rows > kPackedRows || wide_from < 0) throw std::invalid_argument("packed verify rows must be 8 to 16");
    if (wide_from && rows > kMaxRows && (slots_ < wide_from || engine_.MaxPassRows() < rows)) rows = kMaxRows;
    else if (!wide_from && rows > kMaxRows && rows > engine_.MaxPassRows())
      throw std::invalid_argument("this engine runs packed verify passes of at most 8 rows");
    pass_rows_ = rows;
    wide_from_ = rows > kMaxRows ? wide_from : 0;
  }
  int PassRows() const { return pass_rows_; }
  int WideFrom() const { return wide_from_; }

  // One packed verification pass over every listed request (G6d). Speculative requests propose drafts within
  // their row caps; plain requests take one row. Each request's output equals its solo steps.
  std::vector<BatchEvent> VerifyStep(const std::vector<uint64_t>& ids) {
    SettlePrompt();
    const auto speculative = CheckVerify(ids);
    return DeviceUpdate([&] { return VerifyEvents(ids, speculative); });
  }

 private:
  std::vector<bool> CheckVerify(const std::vector<uint64_t>& ids) const {
    RequireHealthy();
    if (ids.empty() || ids.size() > size_t(slots_))
      throw std::invalid_argument("verify step must contain one to eight active requests");
    std::set<uint64_t> unique;
    std::vector<bool> speculative;
    for (uint64_t id : ids) {
      if (!unique.insert(id).second || !active_.contains(id) || active_.at(id)->last < 0)
        throw std::invalid_argument("verify step contains a duplicate or inactive request");
      speculative.push_back(active_.at(id)->spec);
      if (speculative.back() && (!spec_.propose || !spec_.accept || !spec_.settle))
        throw std::invalid_argument("this worker has no packed verification");
    }
    return speculative;
  }
  std::vector<BatchEvent> VerifyEvents(const std::vector<uint64_t>& ids, const std::vector<bool>& speculative) {
    const double start = Clock();
    std::vector<int> want(ids.size(), 0);
    if (spec_.want)
      for (size_t i = 0; i < ids.size(); ++i)
        if (speculative[i]) want[i] = spec_.want(ids[i], active_.at(ids[i])->last);
    const int cap_rows = wide_from_ && ids.size() < size_t(wide_from_) ? kMaxRows : pass_rows_;
    const auto caps = DraftCapsFor(speculative, want, cap_rows, verify_turn_++);
    std::vector<VerifySpan> spans;
    std::vector<int> firsts;
    int rows = 0;
    std::map<uint64_t, VerifySpan> grouped;
    if (spec_.propose_all) {
      std::vector<uint64_t> group;
      std::vector<int> lasts, group_caps;
      std::vector<size_t> remaining;
      std::vector<uint64_t> positions;
      for (size_t i = 0; i < ids.size(); ++i) {
        const Request& state = *active_.at(ids[i]);
        if (!state.spec) continue;
        group.push_back(ids[i]);
        lasts.push_back(state.last);
        remaining.push_back(size_t(state.options.max_tokens) - state.count);
        positions.push_back(state.options.ids.size() + state.count);
        group_caps.push_back(caps[i]);
      }
      const auto proposed = spec_.propose_all(group, lasts, remaining, positions, group_caps);
      if (proposed.size() != group.size()) throw std::logic_error("grouped proposal count does not match");
      for (size_t k = 0; k < group.size(); ++k) grouped.emplace(group[k], proposed[k]);
    }
    for (size_t i = 0; i < ids.size(); ++i) {
      Request& state = *active_.at(ids[i]);
      VerifySpan span{state.slot, {state.last}, 0, 1};
      if (state.spec && grouped.contains(ids[i])) {
        span = grouped.at(ids[i]);
        span.stream = state.slot;
        if (span.tokens.empty() || span.tokens.front() != state.last || span.tokens.size() > size_t(1 + caps[i]))
          throw std::logic_error("speculative proposal does not fit its row cap");
      } else if (state.spec) {
        engine_.SelectStream(state.slot);
        span = spec_.propose(ids[i], state.last, size_t(state.options.max_tokens) - state.count,
                             state.options.ids.size() + state.count, caps[i]);
        span.stream = state.slot;
        if (span.tokens.empty() || span.tokens.front() != state.last || span.tokens.size() > size_t(1 + caps[i]))
          throw std::logic_error("speculative proposal does not fit its row cap");
      }
      firsts.push_back(rows);
      rows += int(span.tokens.size());
      spans.push_back(std::move(span));
    }
    const auto greedy = engine_.ForwardVerifySpans(spans);
    if (greedy.size() != size_t(rows)) throw std::runtime_error("engine returned the wrong verify width");
    // Check mode: copy every row now; draft proposals for the next step reuse the logits buffer.
    std::vector<std::vector<float>> checked;
    if (check_) for (int row = 0; row < rows; ++row) checked.push_back(engine_.CopyLogits(1, row));
    std::vector<std::vector<int>> outputs;
    for (size_t i = 0; i < ids.size(); ++i) {
      Request& state = *active_.at(ids[i]);
      const auto span = std::span<const int>(greedy).subspan(size_t(firsts[i]), spans[i].tokens.size());
      if (!state.spec) { outputs.push_back({Sample(state, span[0], firsts[i])}); continue; }
      outputs.push_back(spec_.accept(ids[i], firsts[i], span));
      if (outputs.back().empty() || outputs.back().size() > spans[i].tokens.size())
        throw std::runtime_error("speculative acceptance returned no tokens");
    }
    if (std::find(speculative.begin(), speculative.end(), true) != speculative.end()) spec_.settle();
    if (check_) CheckRows(ids, spans, firsts, outputs, checked);
    const double elapsed = Clock() - start;
    std::vector<BatchEvent> events;
    for (size_t i = 0; i < ids.size(); ++i) {
      active_.at(ids[i])->decode += elapsed;
      // Read before Publish: a request that ends in this step loses its counters at its close.
      auto counts = speculative[i] ? StepCounts(ids[i]) : std::nullopt;
      for (const int token : outputs[i]) {
        if (token < 0 || token >= kVocab) throw std::runtime_error("engine returned an invalid token ID");
        auto one = Publish(ids[i], token);
        if (counts) one.front().step_draft = std::exchange(counts, std::nullopt);
        events.insert(events.end(), one.begin(), one.end());
        if (!active_.contains(ids[i])) break;
      }
    }
    return events;
  }
  // Rows [0, n) of a span have fully accepted prefixes (row 0 reads the last token; row k the k-th accepted
  // draft), so their logits depend only on the request: equal keys across runs must hash equal.
  void CheckRows(const std::vector<uint64_t>& ids, const std::vector<VerifySpan>& spans, const std::vector<int>& firsts,
                 const std::vector<std::vector<int>>& outputs, const std::vector<std::vector<float>>& checked) {
    for (size_t i = 0; i < ids.size(); ++i) {
      const Request& state = *active_.at(ids[i]);
      const uint64_t prompt = Fnv1a(std::span<const int>(state.options.ids));
      const uint64_t position = state.options.ids.size() + state.count;
      for (size_t k = 0; k < outputs[i].size(); ++k) {
        const auto& row = checked.at(size_t(firsts[i]) + k);
        check_.Row(prompt, position + k, spans[i].tokens.at(k), Fnv1a(row.data(), row.size() * sizeof(float)));
      }
    }
  }

 public:
  std::vector<BatchEvent> SpecStep(uint64_t id) {
    SettlePrompt();
    RequireHealthy();
    if (!active_.contains(id) || active_.at(id)->last < 0 || !active_.at(id)->spec)
      throw std::invalid_argument("spec_step needs an active speculative request after prefill");
    return DeviceUpdate([&] {
      Request& state = *active_.at(id);
      engine_.SelectStream(state.slot);
      const double start = Clock();
      const auto tokens = spec_.step(id, state.last, size_t(state.options.max_tokens) - state.count,
                                     state.options.ids.size() + state.count);
      if (tokens.empty() || tokens.size() > size_t(kMaxRows)) throw std::runtime_error("spec step returned no tokens");
      state.decode += Clock() - start;
      std::vector<BatchEvent> events;
      auto counts = StepCounts(id);  // before Publish, as in VerifyEvents
      // Publish stops at the first stop ID or the token limit, as serial generation trims its window.
      for (const int token : tokens) {
        if (token < 0 || token >= kVocab) throw std::runtime_error("engine returned an invalid token ID");
        auto one = Publish(id, token);
        if (counts) one.front().step_draft = std::exchange(counts, std::nullopt);
        events.insert(events.end(), one.begin(), one.end());
        if (!active_.contains(id)) break;
      }
      return events;
    });
  }

  void Close(uint64_t id) {
    SettlePrompt();
    RequireHealthy();
    const auto found = active_.find(id);
    if (found == active_.end()) throw std::invalid_argument("request is not active");
    if (cache_.expect) cache_.expect(found->second->options, found->second->slot, 0);
    slot_used_[size_t(found->second->slot)] = ++resident_tick_;
    const bool spec = found->second->spec;
    active_.erase(found);
    if (spec) spec_.close(id);
  }

 private:
  // Ends a request before any device work of its command: one error event, then the slot is free.
  std::vector<BatchEvent> FailRequest(uint64_t id, const char* message) {
    BatchEvent failed{id, active_.at(id)->generation};
    failed.error = message;
    Close(id);
    return {failed};
  }
  struct Request {
    const uint64_t id;
    const int slot;
    const uint64_t generation;
    const ServeRequest options;
    int last = -1;
    size_t count = 0, cursor = 0, cached = 0;
    bool cache_shared = false, from_tail = false, slot_resume = false, tail_guarded = false, from_checkpoint = false;
    bool frontier_copy = false;  // the open restored a copy of another slot's Frontier point
    bool grid_point = false;  // resumed from a Grid checkpoint
    std::shared_ptr<const PromptOutputs> end_outputs;  // fix A: the End point's outputs; set until the first token
    std::shared_ptr<const PromptOutputs> outputs;      // fix A: this prompt's outputs, for its End point
    bool end_resumed = false;  // fix A: the open restored the whole prompt (the event's end_resume)
    int tail_replay = 0;
    int replay = 0;
    std::vector<TokenLogprob> top;  // set by Sample for a scoring request
    size_t saved_tokens = 0;
    std::string refused;
    std::vector<int> generated;  // kept only when the tail hook is set
    // The open's cache diagnosis: the miss word and the state nearest to the prompt.
    BatchCacheDiagnosis diag;
    // Serial generation runs one plain step when a single token is requested; so does batch serving.
    const bool spec;
    double prefill = 0, decode = 0, restore = 0, capture = 0, vision = 0;
    int vision_slices = 0;
    Request(uint64_t id, int slot, uint64_t generation, const ServeRequest& options)
        : id(id), slot(slot), generation(generation), options(options),
          spec(options.spec > 0 && options.max_tokens > 1) {}
  };
  // A published first token whose prompt end waits for SettlePrompt; `start`: the final chunk's first row.
  struct PromptEnd { uint64_t id; size_t start; int rows; bool bulk; int token; };
  void FinishPrompt(Request& state, int token) {
    if (state.spec) spec_.first(state.id, token);
    // The prompt end: a next prompt that leaves this request's output (a token split, dropped reasoning, a retry)
    // resumes here and prefills only the rest. With outputs (fix A) an identical prompt resumes here with no row.
    SaveCheckpoint(state, CheckpointKind::End, 0, false, std::exchange(state.outputs, nullptr));
  }
  // Fix A: the prompt end of an End resume. No engine call: the first token comes from the saved greedy ID, or from
  // the saved logits row by the keyed law (seed, prompt length), as the run that saved the point drew it. Then the
  // prompt end as after a final chunk with no rows: the draft start; the End point is on the slot already.
  std::vector<BatchEvent> ResumeEnd(Request& state) {
    const double start = Clock();
    const auto outputs = std::exchange(state.end_outputs, nullptr);
    // The bound image rows go now, as after a final chunk: no prompt row reads them.
    if constexpr (requires { engine_.ReleaseImageRows(engine_.ActiveSequence(), size_t{0}); })
      engine_.ReleaseImageRows(engine_.ActiveSequence(), state.cursor);
    const int token = Sample(state, outputs->greedy, 0, outputs->logits);
    FinishPrompt(state, token);
    state.prefill += Clock() - start;
    return Publish(state.id, token);
  }
  // Fix B: the Pre point of a prompt, or 0. It is the existing core cut of the final chunk (`core_rows_` past the
  // chunk start on the zero-origin grid of `step` rows), so the cut does not change GDN or QSA arithmetic
  // Only on the default chunk schedule (no tail, a restore on the grid), and only when
  // both new pieces have more than 8 rows.
  size_t PreEndPoint(const Request& state, int step) const {
    const size_t length = state.options.ids.size();
    if (!pre_end_min_ || length < pre_end_min_ || !cache_.checkpoint_save || state.options.cache_control == "bypass" ||
        state.from_tail || step <= core_rows_ || step % core_rows_ || state.cached % size_t(step))
      return 0;
    const size_t point = (length - 1) / size_t(step) * size_t(step) + size_t(core_rows_);
    return point < length && length - point > size_t(kMaxRows) ? point : 0;
  }
  bool Stops(const Request& state, int token) const {
    const auto& stops = state.options.stop_ids;
    return std::find(stops.begin(), stops.end(), token) != stops.end();
  }
  // Whether publishing `token` ends the request (Publish).
  bool Ends(const Request& state, int token) const {
    return Stops(state, token) || state.count + 1 >= size_t(state.options.max_tokens);
  }
  static double Clock() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void RequireHealthy() const {
    if (poisoned_) throw std::runtime_error("batch worker state is uncertain; restart the worker");
  }
  template <class F>
  auto DeviceUpdate(F&& operation) {
    RequireHealthy();
    try {
      if constexpr (std::is_void_v<decltype(operation())>) { operation(); SaveTails(nullptr); }
      else if constexpr (std::is_same_v<decltype(operation()), std::vector<BatchEvent>>) {
        auto result = operation(); SaveTails(&result); return result;
      }
      else { auto result = operation(); SaveTails(nullptr); return result; }
    }
    catch (...) { poisoned_ = true; throw; }
  }
  struct FinishedTail {
    uint64_t id; int slot; ServeRequest options; std::vector<int> generated; int replay; BatchTailSource source;
  };
  // A slot's state after its request finished: the saved tail's tokens, until the next request on the slot.
  // `key`: the cache key of the held tail's tokens.
  // `tenant`: the tenant of the request that left the tail. Only that tenant resumes it.
  struct Resident {
    std::vector<int> key;
    int replay = 0;
    bool guarded = false;
    uint64_t used = 0;
    StateTenant tenant = kLocalTenant;
  };
  // Runs after the command's last publication and draft settle, so every stream holds its final state.
  // The command's done events then report the tail saves and the cache they leave.
  void SaveTails(std::vector<BatchEvent>* events) {
    for (const auto& tail : tails_) {
      engine_.SelectStream(tail.slot);
      const uint64_t tried = CacheAttempts();
      // The tail is due now: the reserve helper must not pin a new share for this slot during its copy.
      if (cache_.expect) cache_.expect(tail.options, tail.slot, 0);
      const auto kept = cache_.tail(tail.options, tail.generated, tail.replay, tail.source);
      if (kept && kept->length >= tail.options.ids.size() &&
          kept->length - tail.options.ids.size() <= tail.generated.size()) {
        std::vector<int> tokens(tail.options.ids);
        tokens.insert(tokens.end(), tail.generated.begin(),
                      tail.generated.begin() + std::ptrdiff_t(kept->length - tail.options.ids.size()));
        resident_[size_t(tail.slot)] = Resident{Key(tail.options, tokens), kept->replay, kept->guarded, ++resident_tick_,
                                                tail.options.tenant};
      }
      if (events)
        for (auto& event : *events)
          if (event.done && event.request_id == tail.id) NoteSave(tried, event.cache_tail_tokens, event.cache_save_refused);
    }
    tails_.clear();
    // Decode steps without a done event skip the status read (it reads /proc/meminfo).
    if (!events || !cache_.status ||
        std::none_of(events->begin(), events->end(), [](const BatchEvent& event) { return event.done; })) return;
    const auto status = cache_.status();
    for (auto& event : *events) {
      if (!event.done) continue;
      event.cache_entries = status.entries;
      event.cache_bytes = status.bytes;
      event.cache_evictions = status.evictions;
      event.host_available = status.host_available;
      event.cache_pinned_bytes = status.pinned_bytes;
      // A worker that counts the cache tiers reports them: with the slot hook, never a bare mock.
      event.counters = status.counters;
      event.counters_reported = bool(cache_.slot_served);
      event.cache_reported = true;
    }
  }
  // x4-tail: a bulk launch with joins at the request's cursor. `spec.ledger` is
  // the start of the last launch that the plan without joins runs in it, and `cut_window` the start of that plan's
  // launch that ends at the cut.
  struct JoinedLaunch {
    int rows = 0;
    size_t cut = 0, cut_window = 0;
    PrefillJoinSpec spec;
  };
  JoinedLaunch JoinLaunch(const Request& state, int step) const {
    const ServeRequest& request = state.options;
    const size_t end = request.ids.size();
    const int launch = EngineLaunchRows(engine_);
    const size_t warm = state.spec ? DraftWarmupStart(end, request.mtp_prefix) : SIZE_MAX;
    const auto next = [&](size_t at) { return cache_.capture ? cache_.next_capture(request, at) : SIZE_MAX; };
    // The stop of the plan without joins (above).
    const auto plain = [&](size_t at) { return std::min(next(at), warm); };
    PrefillJoinInput in;
    in.at = state.cursor;
    in.end = end;
    in.step = step;
    in.launch = launch;
    in.join = EngineJoinRows(engine_);
    in.first = next(state.cursor);
    in.second = in.first == SIZE_MAX ? SIZE_MAX : next(in.first);
    in.cut = cache_.capture && cache_.cut_ready && cache_.cut_prepare && cache_.cut_capture && cache_.cut_ready(request);
    // The warm-up start stops a launch unless one grid chunk holds the window: then the MTP warms per grid chunk of
    // the launch (BatchSpecHooks::chunk), the calls of the plan without joins.
    in.warm = request.mtp_prefix > 0 && request.mtp_prefix <= step ? SIZE_MAX : warm;
    const auto plan = PrefillJoinPlan(in);
    JoinedLaunch out;
    out.rows = plan.rows;
    out.cut = plan.cut;
    out.spec.ledger = int(PrefillPlainStart(state.cursor, state.cursor + size_t(plan.rows), end, step, launch, plain) -
                          state.cursor);
    if (plan.cut) {
      out.spec.cut = int(plan.cut - state.cursor);
      out.cut_window = PrefillPlainStart(state.cursor, plan.cut, end, step, launch, plain);
    }
    return out;
  }
  // x4-tail: the frontier at `cut` (inside the launch that just ran) from the cut buffer: a slot frontier, or a host
  // frontier as the capture after a launch that ended there saves it.
  void SaveCut(Request& state, size_t cut) {
    const double saved = Clock();
    const uint64_t tried = CacheAttempts();
    if (!KeepFrontierAt(state, cut, true)) cache_.cut_capture(state.options, cut);
    NoteSave(tried, state.saved_tokens, state.refused);
    state.capture += Clock() - saved;
  }
  uint64_t CacheAttempts() const { return cache_.status ? cache_.status().attempts : 0; }
  // Records the save that a hook just tried, if it tried one.
  void NoteSave(uint64_t tried, size_t& tokens, std::string& refused) const {
    if (!cache_.status) return;
    const auto status = cache_.status();
    if (status.attempts == tried) return;
    if (status.last.saved) tokens = std::max(tokens, status.last.tokens);
    else refused = status.last.refused;
  }
  void ValidateOpen(uint64_t id, const ServeRequest& request) const {
    RequireHealthy();
    if (id == 0 || active_.contains(id)) throw std::invalid_argument("request ID must be unique and positive");
    if (active_.size() >= size_t(slots_)) throw std::invalid_argument("all sequence slots are in use");
    if (request.spec > 0 && request.max_tokens > 1) {
      const std::string error = spec_.check ? spec_.check(request) : "this batch worker has no speculative decoder";
      if (!error.empty()) throw RecoverableOpenError(error);
    }
    if (!ValidTokenIds(request.ids, kVocab) || !ValidTokenIds(request.stop_ids, kVocab) ||
        !RequestFitsContext(request.ids.size(), request.max_tokens, engine_.Context()) ||
        !request.sampling.Error(kVocab).empty())
      throw RecoverableOpenError("invalid batch request sampling, token IDs, or context bounds");
    if (!request.images.empty()) {
      bool vision = false;
      if constexpr (requires { engine_.HasVision(); }) vision = engine_.HasVision();
      if (!vision) throw RecoverableOpenError("this engine runs without the vision tower; start serve with --vision 1");
      // The pixel files must exist before the open changes state; a slice reads one and checks its digest.
      for (const auto& image : request.images) {
        try { CheckImageFile(image); }
        catch (const std::runtime_error& error) { throw RecoverableOpenError(error.what()); }
      }
    }
  }
  // The smallest free slot class that holds the prompt and all requested tokens; the frontend mirrors this rule.
  // Within the class, a returning session takes the slot that still holds its tail, else its own older slot when the
  // prompt repeats almost all of that state; then an empty slot; then a slot that shares more than the system prompt
  // (its own older slot); else the slot left longest ago, so other sessions keep their states.
  int FreeSlot(const ServeRequest& request) const {
    const std::vector<int> key = Key(request, request.ids);
    int best = -1;
    std::tuple<int, size_t, uint64_t> best_key{};
    for (int slot = 0; slot < slots_; ++slot) {
      bool used = false;
      for (const auto& [id, state] : active_) used |= state->slot == slot;
      if (used || !RequestFitsContext(request.ids.size(), request.max_tokens, contexts_[slot])) continue;
      const auto rank = Affinity(slot, request, key);
      if (best < 0 || contexts_[slot] < contexts_[best] || (contexts_[slot] == contexts_[best] && rank > best_key))
        best = slot, best_key = rank;
    }
    if (best < 0) throw RecoverableOpenError("no free sequence slot holds this request; retry after a close");
    return best;
  }
  std::vector<int> Key(const ServeRequest& request, std::span<const int> tokens) const {
    return cache_.key ? cache_.key(request, tokens) : RequestCacheKey(request, tokens);
  }
  // `key`: the request's prompt key. Only the tail's own tenant resumes it.
  bool Resumable(int slot, const ServeRequest& request, const std::vector<int>& key) const {
    const auto& held = resident_[size_t(slot)];
    return held && request.cache_control == "auto" && held->tenant == request.tenant && held->key.size() < key.size() &&
           std::equal(held->key.begin(), held->key.end(), key.begin());
  }
  std::tuple<int, size_t, uint64_t> Affinity(int slot, const ServeRequest& request, const std::vector<int>& key) const {
    const auto& held = resident_[size_t(slot)];
    if (held && Resumable(slot, request, key)) return {4, held->key.size(), 0};
    // A slot checkpoint that the prompt repeats is the session's own older state (with the pick rule of Checkpoint).
    const auto best = request.cache_control == "auto" ? BestPoint(slot, request, key) : std::nullopt;
    const size_t point = best ? checkpoints_.At(slot, *best).hist.size() : 0;
    // a tail of another tenant shares nothing with this prompt, so its slot ranks as the slot left longest ago.
    const size_t common = held && held->tenant == request.tenant
                              ? size_t(std::mismatch(held->key.begin(), held->key.end(), key.begin(), key.end()).first -
                                       held->key.begin()) : 0;
    if (point > size_t(request.shared_prefix_tokens)) return {3, std::max(point, common), 0};
    // a slot that holds only a Frontier point is not empty (HIGH-1). Among the slots that other sessions
    // left, the smallest loss goes first (the Frontier point that the handover must copy), then the oldest.
    const size_t loss = checkpoints_.FrontierLength(slot);
    const size_t keep = std::numeric_limits<size_t>::max() - loss;
    if (!held) return loss ? std::tuple<int, size_t, uint64_t>{0, keep, std::numeric_limits<uint64_t>::max() - slot_used_[size_t(slot)]}
                           : std::tuple<int, size_t, uint64_t>{2, 0, 0};
    if (common > size_t(request.shared_prefix_tokens) && common >= held->key.size() - held->key.size() / 16)
      return {3, common, 0};
    if (common > size_t(request.shared_prefix_tokens)) return {1, common, 0};
    return {0, keep, std::numeric_limits<uint64_t>::max() - held->used};
  }
  // The slot checkpoint to resume from, when it is longer than what the host cache would restore.
  // `key`: the request's prompt key.
  std::optional<int> Checkpoint(int slot, const ServeRequest& request, const std::vector<int>& key) const {
    if (!cache_.checkpoint_load || !cache_.restore || request.cache_control != "auto") return std::nullopt;
    const auto best = BestPoint(slot, request, key);
    if (!best || (cache_.peek && cache_.peek(request) >= checkpoints_.At(slot, *best).hist.size())) return std::nullopt;
    return best;
  }
  // MEDIUM-2: a Frontier point resumes only where the host rule (Find) would restore a host frontier of its length,
  // so the slot and the host give the same restore for one prompt.
  std::optional<int> BestPoint(int slot, const ServeRequest& request, std::span<const int> key) const {
    return checkpoints_.Best(slot, key, request.tenant, [&](const SlotCheckpoint& point) {
      return point.kind != CheckpointKind::Frontier || (cache_.frontier_fits && cache_.frontier_fits(request, point));
    });
  }
  // frontier-copy: the longest Frontier point of another slot that the host rule
  // accepts for `request` (the lowest slot on a tie), when it is longer than the host state and this slot's own
  // point. Rows below a Frontier point never change while its slot decodes, so a busy slot can give it. Only a point of
  // the request's own tenant can be a donor.
  std::optional<std::pair<int, int>> Donor(int slot, const ServeRequest& request, const std::vector<int>& key) const {
    if (!cache_.frontier_copy || !cache_.peek || !cache_.restore || !FrontiersOn() || request.cache_control != "auto")
      return std::nullopt;
    const auto own = BestPoint(slot, request, key);
    size_t longest = std::max(cache_.peek(request), own ? checkpoints_.At(slot, *own).hist.size() : 0);
    std::optional<std::pair<int, int>> donor;
    for (int from = 0; from < slots_; ++from) {
      if (from == slot) continue;
      const auto best = checkpoints_.Best(from, key, request.tenant, [&](const SlotCheckpoint& point) {
        return point.kind == CheckpointKind::Frontier && cache_.frontier_fits(request, point);
      });
      if (best && checkpoints_.At(from, *best).hist.size() > longest)
        donor = {from, *best}, longest = checkpoints_.At(from, *best).hist.size();
    }
    return donor;
  }
  // Restores donor point (from, index) into `slot` with a slot-to-slot device copy (SlotCheckpointIO::CopyTo; no
  // host stage). A copy hit confirms a speculative point, as a Frontier hit does.
  BatchRestore CopyFrontier(int slot, const ServeRequest& request, std::pair<int, int> donor) {
    // the hook reads the donor's stream. The target is selected again on every exit, so the
    // fallback restore and an error never write the donor's slot.
    const struct Reselect { E& engine; int slot; ~Reselect() { engine.SelectStream(slot); } } reselect{engine_, slot};
    const auto [from, index] = donor;
    const SlotCheckpoint& point = checkpoints_.At(from, index);
    if (point.tenant != request.tenant) throw std::logic_error("a frontier copy must restore the donor point");
    const BatchRestore hit = cache_.frontier_copy(request, from, index, point);
    if (!hit.tokens) return hit;
    if (hit.tokens != point.hist.size() || hit.tail || hit.slot || hit.shared)
      throw std::logic_error("a frontier copy must restore the donor point");
    if (point.frontier.speculative) checkpoints_.Confirm(from, index);
    return hit;
  }
  bool FrontiersOn() const {
    return cache_.frontier && cache_.frontier_fits && cache_.handover && cache_.checkpoint_save &&
           cache_.checkpoint_load && (!cache_.resume || cache_.resume_host);
  }
  // the session frontier at the cursor stays in the slot (a Frontier point) when the slot has a buffer for it.
  bool KeepFrontier(Request& state) { return KeepFrontierAt(state, state.cursor, false); }
  // x4-tail: `from_cut`: the state at `cursor` is in the cut buffer (a joined launch passed it), not in the slot.
  bool KeepFrontierAt(Request& state, size_t cursor, bool from_cut) {
    if (!FrontiersOn() || state.options.cache_control == "bypass" || (from_cut && !cache_.cut_save)) return false;
    return cache_.frontier(state.options, cursor, [&](const SlotFrontier& meta) {
      const std::span<const int> below(state.options.ids.data(), cursor);
      auto key = Key(state.options, below);
      const auto pick = checkpoints_.FrontierVictim(state.slot, key, state.id, state.options.tenant);
      if (!pick.held && !pick.index) return false;
      if (pick.held) {
        // The slot holds this state already; a confirmed frontier confirms it (LOW-3).
        if (!meta.speculative) checkpoints_.Confirm(state.slot, *pick.index);
      } else {
        if (from_cut) cache_.cut_save(state.slot, *pick.index);
        else cache_.checkpoint_save(state.slot, *pick.index);
        checkpoints_.Store(state.slot, *pick.index, {std::vector<int>(below.begin(), below.end()), std::move(key), 0,
                                                     false, CheckpointKind::Frontier, state.id, meta,
                                                     state.options.tenant});
      }
      state.saved_tokens = std::max(state.saved_tokens, cursor);
      return true;
    });
  }
  // HIGH-1: the Frontier points that this open would drop go to the host first, the longest first. The open keeps
  // them (Keep, below) only on a slot resume (every point lies below the tail), unless `resume` restores a host tail
  // instead, or on a checkpoint load at or above them. A reset, a host restore or another session keeps none.
  void GuardFrontiers(int slot, const ServeRequest& request, const std::optional<BatchResident>& resident,
                      std::optional<int> point) {
    if (!FrontiersOn()) return;
    size_t keep = 0;
    if (resident) keep = cache_.resume_host && cache_.resume_host(request, *resident) ? 0 : SIZE_MAX;
    else if (point) keep = checkpoints_.At(slot, *point).hist.size();
    for (const int index : checkpoints_.Frontiers(slot))
      if (checkpoints_.At(slot, index).hist.size() > keep) cache_.handover(slot, index, checkpoints_.At(slot, index));
  }
  // Returns whether it saved a point (false: no checkpoints, a bypass request, or the slot holds that state).
  bool SaveCheckpoint(Request& state, CheckpointKind kind, int replay, bool guarded,
                      std::shared_ptr<const PromptOutputs> outputs = nullptr) {
    if (!cache_.checkpoint_save || state.options.cache_control == "bypass") return false;
    const std::span<const int> below(state.options.ids.data(), state.cursor);
    auto key = Key(state.options, below);
    const auto victim = checkpoints_.Victim(state.slot, key, kind, state.id, state.options.tenant);
    if (!victim) return false;
    cache_.checkpoint_save(state.slot, *victim);
    checkpoints_.Store(state.slot, *victim, {std::vector<int>(below.begin(), below.end()), std::move(key),
                                             replay, guarded, kind, state.id, {}, state.options.tenant,
                                             std::move(outputs)});
    return true;
  }
  // `saved`: the row's logits, copied already (fix A); else they are copied from the engine's row `row`.
  int Sample(Request& state, int greedy, int row, const std::shared_ptr<const std::vector<float>>& saved = nullptr) {
    if (state.options.sampling.temperature == 0) {
      if (greedy < 0 || greedy >= kVocab) throw std::runtime_error("engine returned an invalid token ID");
      if (state.options.top_logprobs) {
        // The row the sampler would copy. The token stays the device's greedy ID; the event carries both.
        state.top = TopLogprobs(engine_.CopyLogits(1, row), state.options.top_logprobs);
      }
      return greedy;
    }
    const auto logits = saved ? *saved : engine_.CopyLogits(1, row);
    // Keyed law: the draw depends only on (seed, absolute position, token), as in serial generation.
    return sampling::FromLogits(logits, state.options.sampling)
        .KeyedSample(state.options.sampling.seed, state.options.ids.size() + state.count);
  }
  std::optional<BatchDraftCounts> StepCounts(uint64_t id) const {
    if (!spec_.counts) return std::nullopt;
    return spec_.counts(id);
  }
  std::vector<BatchEvent> Publish(uint64_t id, int token) {
    Request& state = *active_.at(id);
    state.last = token;
    ++state.count;
    const bool stopped = Stops(state, token);
    const BatchEvent event{id, state.generation, token, false, stopped, state.options.ids.size(), state.count,
                           state.prefill, state.decode, state.cached, cache_.bytes ? cache_.bytes() : 0,
                           state.restore, state.capture, state.cache_shared};
    std::vector<BatchEvent> events{event};
    events.back().cache_tail = state.from_tail;
    events.back().logprobs = std::move(state.top);
    events.back().tail_replay = state.tail_replay;
    events.back().slot_resume = state.slot_resume;
    events.back().checkpoint_resume = state.from_checkpoint;
    events.back().frontier_copy = state.frontier_copy;
    events.back().end_resume = state.end_resumed;
    if (cache_.tail) state.generated.push_back(token);
    if (stopped || state.count >= size_t(state.options.max_tokens)) {
      BatchEvent done = events.front();
      done.token.reset();
      done.logprobs.clear();
      done.done = true;
      done.cache_saved_tokens = state.saved_tokens;
      done.cache_save_refused = state.refused;
      done.cache_miss = state.diag.miss;
      done.closest_tokens = state.diag.shared;
      done.closest_age_s = state.diag.age_s;
      done.closest_outcome = state.diag.outcome;
      done.slot_context = contexts_[size_t(state.slot)];
      const bool spec = state.spec;
      if (spec && spec_.stats) done.draft = spec_.stats(id);
      events.push_back(done);
      if (cache_.tail && state.options.cache_control != "bypass")
        tails_.push_back({id, state.slot, state.options, std::move(state.generated),
                          spec && spec_.replay ? spec_.replay(id) : 0,
                          {state.from_tail && !state.from_checkpoint, state.tail_guarded, state.slot}});
      slot_used_[size_t(state.slot)] = ++resident_tick_;
      active_.erase(id);
      if (spec) spec_.close(id);
    }
    return events;
  }
  E& engine_;
  const int slots_;
  std::vector<int> contexts_;
  bool poisoned_ = false;
  std::optional<size_t> prefill_cursor_;
  std::optional<PromptEnd> prompt_end_;
  InterleaveCheck check_;
  BatchCacheHooks cache_;
  BatchSpecHooks spec_;
  std::vector<FinishedTail> tails_;
  std::vector<std::optional<Resident>> resident_;
  SlotCheckpoints checkpoints_;
  uint64_t resident_tick_ = 0;
  std::array<uint64_t, kMaxRows> slot_used_{};  // per slot: when its last request ended (Affinity)
  std::array<uint64_t, kMaxRows> generations_{};
  uint64_t verify_turn_ = 0;
  size_t pre_end_min_ = 0;    // SetPreEnd (0: off)
  int core_rows_ = kPrefillLaunchCoreRows;
  int pass_rows_ = kMaxRows;  // SetPassRows
  int wide_from_ = 0;         // SetPassRows: decoders from which a pass uses pass_rows_ (0: always)
  std::map<uint64_t, std::unique_ptr<Request>> active_;
};
}  // namespace crucible
