// The Oct 5 served-cache incident on the CPU: the served layout's batch cache, measured snapshot sizes, and the
// host memory those snapshots pin, driven by the incident's prompt sequence (numbers only, no content).
// In the served build: once the 10 GiB cache was full, the 8 GiB host check ran before eviction, so every
// frontier above about 62K tokens was refused and one session re-prefilled up to 166K tokens per turn.
#include "../engine/serve/batch_conversation_cache.h"
#include <cassert>
#include <cstdio>
#include <map>
using namespace crucible;
struct Snapshot { std::vector<int> hist; ConversationResume resume; };

namespace {
constexpr int kRows = 4096;                                  // --prefill 4096
constexpr size_t kBudget = size_t(10240) << 20;              // --conversation-cache-mib 10240
// MemAvailable at 08:16, after load and before the first request (20.06 GiB). Snapshots are
// pinned host memory, so MemAvailable falls by the cache's bytes (20.06 to 9.6 GiB by 08:23 in the incident).
constexpr uint64_t kIdleAvailable = uint64_t(20.06 * 1073741824.0);
// Least-squares fit of 14 measured snapshot sizes (8K to 109K tokens, development measurements): 89 KB max error.
size_t SnapshotBytes(size_t tokens) { return size_t(152292328 + 32264 * tokens); }
constexpr size_t kTemplateTail = 4;  // "<|im_start|>assistant\n<think>\n" after cache_prefix_tokens
constexpr size_t kSharedPrefix = 11000;  // system and tools block, below the first user turn

struct Turn { int session; size_t prompt, completion; };
struct Result { size_t cached, prefilled, appended; };

class ServedCache {
 public:
  explicit ServedCache(bool replay) : replay_(replay) { policy_.prefill_rows = kRows; }
  // One request: restore, prefill in 4096-row chunks with frontier captures, then (replay) the generated tail.
  Result Run(const Turn& turn) {
    const auto ids = Stream(turn.session, turn.prompt);
    size_t cached = 0;
    bool from_tail = false;
    if (replay_)
      if (const auto tail = cache_.FindTail(ids, policy_)) cached = tail->snapshot->hist.size(), from_tail = true;
    if (!cached)
      if (const auto entry = cache_.Find(ids, policy_, ids.size() - kTemplateTail, ids.size() - 1, kRows))
        cached = entry->snapshot->hist.size();
    // As BatchSession: a frontier restore keeps the 4096-row grid, a replayed turn captures no prompt frontiers.
    const auto frontiers = BatchCacheFrontiers(ids.size(), ids.size() - kTemplateTail, kRows, kSharedPrefix);
    for (size_t cursor = cached + kRows; !from_tail && cursor < ids.size(); cursor += kRows)
      for (const auto& frontier : frontiers)
        if (frontier.position == cursor) Capture(ids, frontier);
    if (replay_) Tail(turn.session, turn.prompt + turn.completion - 1, from_tail);
    const size_t seen = seen_[turn.session];
    seen_[turn.session] = replay_ ? turn.prompt + turn.completion - 1 : turn.prompt;
    return {cached, turn.prompt - cached, turn.prompt - std::min(seen, turn.prompt)};
  }
  size_t refused = 0;

 private:
  static std::vector<int> Stream(int session, size_t length) {
    std::vector<int> ids(length);
    for (size_t i = 0; i < length; ++i) ids[i] = session * 1000000 + int(i);
    return ids;
  }
  uint64_t Available() const { return kIdleAvailable - cache_.Bytes(); }
  void Capture(const std::vector<int>& ids, const CacheFrontier& frontier) {
    const size_t position = frontier.position;
    refused += !cache_.Capture(ids, policy_, frontier, SnapshotBytes(position), Available(), [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist.assign(ids.begin(), ids.begin() + position);
      s->resume.length = position; s->resume.prompt_only = true; s->resume.prompt_boundary = ids.size() - kTemplateTail;
      return std::shared_ptr<const Snapshot>(s);
    });
  }
  // As BatchPrefixCache: a tail of a replayed turn without reasoning is trusted.
  void Tail(int session, size_t length, bool trusted) {
    const auto hist = Stream(session, length);
    refused += !cache_.CaptureTail(hist, policy_, SnapshotBytes(length), Available(), [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist = hist; s->resume.length = length;
      return std::shared_ptr<const Snapshot>(s);
    }, trusted);
  }
  const bool replay_;
  ConversationPolicy policy_;
  BatchConversationCache<Snapshot> cache_{kBudget};
  std::map<int, size_t> seen_;
};

// Requests from 08:16 to 08:39 in arrival order (prompt tokens only). Session 2 is the long agent session; 1 and 3 are
// shorter sessions that filled the cache first. Completions are not recorded there; 400 tokens stand in for them.
std::vector<Turn> IncidentTurns() {
  const std::vector<std::pair<int, size_t>> prompts{
      {1, 12341}, {3, 12492}, {1, 13031}, {1, 13466}, {1, 14213}, {1, 19765}, {1, 20565}, {1, 22258},
      {2, 12834}, {2, 17510}, {2, 22749}, {2, 23468}, {2, 24994}, {2, 25558}, {2, 27451}, {2, 29404},
      {2, 31016}, {2, 31607}, {2, 32332}, {2, 33560}, {2, 34299}, {2, 54752}, {2, 74721}, {2, 94222},
      {2, 113374}, {2, 134864}, {2, 155379}, {2, 179809}, {2, 183834}, {2, 187552}, {2, 206318}, {2, 209228},
      {2, 219071}};
  std::vector<Turn> turns;
  for (const auto& [session, prompt] : prompts) turns.push_back({session, prompt, 400});
  return turns;
}

// tools/agent_soak.py multi: the long session to 220K beside two short sessions to 40K, round robin. The system
// and tools block (about 12.2K tokens) is shared, as for Pi agents with one setup.
std::vector<Turn> MultiTurns() {
  const std::vector<size_t> steps{690, 435, 750, 5550, 800, 1700, 4700, 5250, 720, 1500, 560, 1900, 1950, 1600, 600,
                                  700, 1250, 740, 20450, 19950, 19500, 19150, 21500, 20500, 24400, 4000, 3700, 18750,
                                  2900, 9850};
  std::map<int, size_t> prompt{{1, 12200}, {2, 12200}, {3, 12200}};
  std::map<int, size_t> step{{1, 0}, {2, 0}, {3, 0}};
  std::vector<Turn> turns;
  for (bool any = true; any;) {
    any = false;
    for (int session : {1, 2, 3}) {
      const size_t limit = session == 1 ? 220000 : 40000;
      if (prompt[session] > limit) continue;
      turns.push_back({session, prompt[session], 120});
      any = true;
      size_t next = steps[step[session]++ % steps.size()];
      if (session != 1) while (next >= 10000) next = steps[step[session]++ % steps.size()];
      prompt[session] += 120 + next;
    }
  }
  return turns;
}

// The soak rule (tools/agent_soak.py): a turn re-prefills at most its appended tokens plus one chunk, or plus the
// 8-row decode-path head chunk with session replay.
bool Soak(bool replay, bool print, const std::vector<Turn>& turns = IncidentTurns()) {
  ServedCache cache(replay);
  bool pass = true;
  for (const auto& turn : turns) {
    const auto r = cache.Run(turn);
    const size_t margin = replay ? kMaxRows : kRows;
    const bool ok = !r.appended || r.prefilled <= r.appended + margin;
    pass = pass && ok;
    if (print || !ok)
      std::printf("replay=%d session=%d prompt=%zu cached=%zu prefilled=%zu appended=%zu %s\n", int(replay),
                  turn.session, turn.prompt, r.cached, r.prefilled, r.appended, ok ? "ok" : "FAIL");
  }
  std::printf("replay=%d refused=%zu %s\n", int(replay), cache.refused, pass ? "PASS" : "FAIL");
  return pass;
}
// The Oct 6 multi-session failure (decode worker, crucible-t12 h2h, results/h2h requests 59-70): four sessions share a
// 31K history, then run code, tool, code_sampled, think and prose turns at the same time. The think turn reaches
// the answer room after one token: the server cancels the first job and runs a continuation job, which saves a tail
// that holds the reasoning. The client keeps only the answer, so the next prompt repeats the prompt and not that
// tail. On crucible-t12 the continuation tails superseded each session's resume point, the next turn of all four
// sessions missed, and each re-prefilled 33K tokens.
class AgentCache {
 public:
  struct Job { std::vector<int> ids; size_t boundary = 0, cached = 0; bool from_tail = false; };
  explicit AgentCache(bool replay) : replay_(replay) { policy_.prefill_rows = kRows; }
  // Restore, then prefill in 4096-row chunks with the frontier captures of BatchSession.
  Job Open(std::vector<int> ids, size_t boundary) {
    Job job{std::move(ids), boundary};
    const auto& r = job.ids;
    if (replay_)
      if (const auto tail = cache_.FindTail(r, policy_)) job.cached = tail->snapshot->hist.size(), job.from_tail = true;
    if (!job.cached)
      if (const auto entry = cache_.Find(r, policy_, boundary, r.size() - 1, kRows)) job.cached = entry->snapshot->hist.size();
    const auto frontiers = BatchCacheFrontiers(r.size(), boundary, kRows, 0);
    for (size_t cursor = job.cached + kRows; !job.from_tail && cursor < r.size(); cursor += kRows)
      for (const auto& frontier : frontiers)
        if (frontier.position == cursor) Capture(r, boundary, frontier);
    return job;
  }
  // A finished job keeps its prompt and every output token but the last (ConversationTailState). The server guards
  // a continuation job and a turn with reasoning: their tails are never trusted.
  void Tail(const Job& job, const std::vector<int>& generated, bool guard) {
    if (!replay_ || generated.size() < 2) return;
    std::vector<int> hist(job.ids);
    hist.insert(hist.end(), generated.begin(), generated.end() - 1);
    refused += !cache_.CaptureTail(hist, policy_, SnapshotBytes(hist.size()), Available(), [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist = hist; s->resume.length = hist.size();
      return std::shared_ptr<const Snapshot>(s);
    }, job.from_tail && !guard);
  }
  size_t refused = 0;

 private:
  uint64_t Available() const { return kIdleAvailable - cache_.Bytes(); }
  void Capture(const std::vector<int>& ids, size_t boundary, const CacheFrontier& frontier) {
    const size_t position = frontier.position;
    refused += !cache_.Capture(ids, policy_, frontier, SnapshotBytes(position), Available(), [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist.assign(ids.begin(), ids.begin() + position);
      s->resume.length = position; s->resume.prompt_only = true; s->resume.prompt_boundary = boundary;
      return std::shared_ptr<const Snapshot>(s);
    });
  }
  const bool replay_;
  ConversationPolicy policy_;
  BatchConversationCache<Snapshot> cache_{kBudget};
};

struct Workload { const char* name; size_t completion; bool think; size_t tool_result; };

// tools/h2h_decode.py: each turn appends a user request; a tool turn also appends a short tool result. The think turn
// sends max_tokens 768, so the answer room (1024) closes the think block after one token.
bool AgentSoak(int sessions, size_t base_tokens, int rounds, bool print) {
  constexpr int kGenPrompt[] = {7, 8, 9, 10};  // "<|im_start|>assistant\n<think>\n"
  constexpr int kImEnd = 11;
  constexpr size_t kUser = 60, kClose = 25, kCap = 1;
  const Workload workloads[] = {{"code", 512, false, 0}, {"tool", 400, false, 120}, {"code_sampled", 512, false, 0},
                                {"think", 768, true, 0}, {"prose", 512, false, 0}};
  int fresh = 1 << 24;
  const auto Fresh = [&](size_t count) { std::vector<int> ids(count); for (auto& id : ids) id = fresh++; return ids; };
  const auto Join = [](std::vector<int> a, const std::vector<int>& b) { a.insert(a.end(), b.begin(), b.end()); return a; };
  AgentCache cache(true);
  std::vector<int> base(base_tokens);
  for (size_t i = 0; i < base_tokens; ++i) base[i] = int(i) + 100;
  std::vector<std::vector<int>> history(sessions);
  std::vector<size_t> previous(sessions, 0);
  bool pass = true;
  const auto Prompt = [&](int s) {
    auto ids = Join(history[s], Fresh(kUser));
    ids.insert(ids.end(), std::begin(kGenPrompt), std::end(kGenPrompt));
    return ids;
  };
  // Warm-up: each session sends its first prompt alone with max_tokens 1, so the shared frontiers exist.
  for (int s = 0; s < sessions; ++s) {
    history[s] = Join(base, Fresh(kUser));
    auto ids = Join(history[s], {7, 8, 9, 10});
    cache.Open(ids, ids.size() - 4);
  }
  for (int round = 0; round < rounds; ++round)
    for (const auto& work : workloads) {
      // All sessions open, prefill and decode at the same time; tails are saved as each job finishes.
      std::vector<AgentCache::Job> first(sessions), cont(sessions);
      std::vector<std::vector<int>> output(sessions);
      for (int s = 0; s < sessions; ++s) {
        auto ids = Prompt(s);
        const size_t boundary = ids.size() - 4;
        first[s] = cache.Open(std::move(ids), boundary);
      }
      for (int s = 0; s < sessions; ++s) {
        if (!work.think) { output[s] = Fresh(work.completion); continue; }
        // http.py close_thinking: cancel the first job (no tail), then continue on prompt + output + close.
        output[s] = Join(Fresh(kCap), Fresh(kClose));
        cont[s] = cache.Open(Join(first[s].ids, output[s]), first[s].boundary);
      }
      for (int s = 0; s < sessions; ++s) {
        if (!work.think) { cache.Tail(first[s], output[s], false); continue; }
        const auto answer = Fresh(work.completion - kCap - kClose);
        cache.Tail(cont[s], answer, true);
        output[s] = Join(output[s], answer);
      }
      for (int s = 0; s < sessions; ++s) {
        const auto& job = first[s];
        const size_t prefilled = job.ids.size() - job.cached, appended = job.ids.size() - previous[s];
        const bool ok = !previous[s] || prefilled <= appended + kRows;
        pass = pass && ok;
        if (print || !ok)
          std::printf("agents=%d base=%zu %s session=%d prompt=%zu cached=%zu prefilled=%zu appended=%zu %s\n", sessions,
                      base_tokens, work.name, s, job.ids.size(), job.cached, prefilled, appended, ok ? "ok" : "FAIL");
        previous[s] = job.ids.size();
        // The client repeats the answer. Without thinking it repeats every output token (replay hits); after a
        // think turn it renders only the answer, so the prompt differs from the output after the header.
        history[s] = work.think ? Join(Join(std::vector<int>(job.ids.begin(), job.ids.begin() + job.boundary), {7, 8}),
                                       Fresh(work.completion / 2))
                                : Join(job.ids, output[s]);
        history[s].push_back(kImEnd);
        if (work.tool_result) history[s] = Join(history[s], Fresh(work.tool_result));
      }
    }
  std::printf("agents=%d base=%zu refused=%zu %s\n", sessions, base_tokens, cache.refused, pass ? "PASS" : "FAIL");
  return pass;
}
}  // namespace


// The Oct 28 memory incident: the daily engine needed --host-reserve-mib 11264 to keep MemAvailable above the
// 10,240 MiB soak floor. The disk writer's queue holds evicted snapshots in pinned memory beside the cache, and neither
// the cache budget nor the 8 GiB default reserve counted it. This is the CPU memory model of tools/agent_soak.py agents
// (4 Pi sessions from 100K to 190K tokens) with the real 9,700 MiB layout: MemAvailable = baseline - cache - queue.
namespace queue_model {
constexpr uint64_t kMiB = uint64_t(1) << 20;
constexpr uint64_t kBaseline = 22167 * kMiB;         // MemAvailable + pinned peak of the r11 soak (night log Oct 27)
constexpr size_t kCacheBudget = size_t(7948) << 20;  // 9,700 MiB less the slot checkpoints and the cut buffer
constexpr uint64_t kFloor = 10240 * kMiB, kMargin = 1000 * kMiB;  // the soak floor, and the gate's margin
constexpr uint64_t kQueueLimit = uint64_t(4) << 30;  // min(8 GiB, host reserve 8 GiB / 2), batched_serve.h OpenDisk
size_t StateBytes(size_t tokens) { return size_t(27460) * tokens; }  // 3,112,899,584 B at 113,380 tokens (soak r11)

// counted: the fix (queue in Outside, one floor, spill ceiling). `reserve` and `queue_limit`: the server's
// --host-reserve-mib and half of it. `restore`: a turn also restores the longest disk state of its session.
struct Config {
  bool counted;
  uint64_t reserve = kDefaultHostReserve, queue_limit = uint64_t(4) << 30;
  bool restore = false;
  bool direct = false;  // cache-direct: a direct spill reads the device in chunks (a lazy snapshot), no full host copy
  // cache-direct: only a resumed release saves lazily. A release to another session and a handover still copy the whole
  // tail: this share (percent) of the direct spills is of that kind (Oct 28 soak: 4 of 86 releases were not resumed).
  unsigned eager_percent = 0;
};
struct Outcome {
  uint64_t lowest = UINT64_MAX;
  size_t direct = 0, spilled = 0, dropped = 0, refused = 0, hits = 0, turns = 0, misses = 0, prefilled = 0;
  uint64_t direct_peak = 0;  // cache-direct: the most host bytes that one direct spill held in the writer queue
  uint64_t eager_peak = 0;   // the same for the spills that copy the whole tail (eager_percent)
  size_t eager = 0;
};

constexpr uint64_t kRestBytes = 250 * 1000 * 1000, kStagingBytes = 8 * 8 * kMiB;

class QueueCache {
 public:
  explicit QueueCache(Config config) : config_(config), cache_(kCacheBudget, {}, 4, config.reserve) {
    policy_.prefill_rows = kRows;
    if (config.counted) {
      cache_.SetHostFloor(kFloor);
      cache_.SetSpillSlack(uint64_t(2) << 30);
      cache_.SetDirectMargin(uint64_t(1088) << 20);  // kDirectMargin, batch_state_disk.h
      cache_.SetQueued([this] { return size_t(Queued()); });
    }
    cache_.SetSpill([this](std::span<const int> key, const ConversationPolicy&, const std::shared_ptr<const Cache::Entry>& entry,
                           int, uint64_t room) { return Spill(key, entry->memory->bytes, !entry->snapshot->resume.prompt_only, room); });
  }
  void Turn(int session, size_t prompt) {
    const auto ids = Stream(session, prompt);
    size_t cached = 0;
    bool from_tail = false;
    if (const auto tail = cache_.FindTail(ids, policy_)) cached = tail->snapshot->hist.size(), from_tail = true;
    if (!cached)
      if (const auto entry = cache_.Find(ids, policy_, ids.size() - kTemplateTail, ids.size() - 1, kRows))
        cached = entry->snapshot->hist.size();
    // The disk tier: the longest state of this session that is below the prompt (written or queued).
    if (config_.restore)
      for (const auto& [key, tail] : disk_[session])
        if (key < prompt && key > cached) cached = key, from_tail = tail;
    ++out.turns, out.hits += cached > 0, out.misses += cached == 0, out.prefilled += prompt - cached;
    const auto frontiers = BatchCacheFrontiers(ids.size(), ids.size() - kTemplateTail, kRows, kSharedPrefix);
    for (size_t cursor = cached + kRows; !from_tail && cursor < ids.size(); cursor += kRows)
      for (const auto& frontier : frontiers)
        if (frontier.position == cursor) Save(ids, frontier.position, true, frontier);
    Save(Stream(session, prompt + 79), prompt + 79, false, {});
  }
  Outcome out;

 private:
  using Cache = BatchConversationCache<Snapshot>;
  static std::vector<int> Stream(int session, size_t length) {
    std::vector<int> ids(length);
    for (size_t i = 0; i < length; ++i) ids[i] = session * 1000000 + int(i);
    return ids;
  }
  uint64_t Queued() const {
    uint64_t sum = 0;
    for (const auto bytes : queue_) sum += bytes;
    return sum;
  }
  uint64_t Pinned() const { return cache_.Bytes() + Queued(); }
  uint64_t Available() const { return kBaseline - Pinned(); }
  // The disk writer ends one write per save: a state of 3 GB takes 0.4 to 2.4 s, about one prefill chunk (1 s).
  void Write() {
    if (!queue_.empty()) queue_.erase(queue_.begin());
  }
  // StateDisk::Spill: refused as "memory" when the payload is over the room, as "busy" when a state waits and the
  // waiting bytes pass the queue limit (the first job of the queue is the one that the writer writes).
  bool Spill(std::span<const int> key, size_t bytes, bool tail, uint64_t room) {
    const bool taken = bytes <= room && !Busy(bytes);
    if (taken) queue_.push_back(bytes), ++out.spilled, Written(key.size(), key.front(), tail);
    else ++out.dropped;
    return taken;
  }
  bool Busy(size_t bytes) const {
    const uint64_t waiting = queue_.size() > 1 ? Queued() - queue_.front() : 0;
    return waiting && waiting + bytes > config_.queue_limit;
  }
  // A state that the writer took is on disk for the next turn (it waits for the write, as a read waits).
  void Written(size_t length, int first, bool tail) {
    disk_[first / 1000000].emplace_back(length, tail);
  }
  // BatchPrefixCache::Direct: a slot tail that the cache refused goes to the writer when the host keeps the reserve
  // (and, with the fix, the budget room); the writer holds the new snapshot, so it pins its bytes now.
  void Direct(const std::vector<int>& ids, size_t bytes) {
    // Chunked: the writer pins the rest of the state (fixed state, partial blocks; 250 MB in the Oct 28 soak
    // log) and reads the full blocks from the device through 8 io_threads x 8 MiB of staging.
    const bool eager = config_.direct && (calls_++ * 7919) % 100 < config_.eager_percent;  // 7919: spread the eager ones
    const uint64_t host = config_.direct && !eager ? std::min<uint64_t>(bytes, kRestBytes + kStagingBytes) : bytes;
    const uint64_t room = cache_.DirectRoom(Available());
    if (host > room || Busy(host)) return void(++out.dropped);
    queue_.push_back(host), ++out.direct;
    (eager ? out.eager_peak : out.direct_peak) = std::max<uint64_t>(eager ? out.eager_peak : out.direct_peak, host);
    out.eager += eager;
    Written(ids.size(), ids.front(), true);
  }
  void Save(const std::vector<int>& ids, size_t position, bool frontier_save, const CacheFrontier& frontier) {
    Write();
    const auto make = [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist.assign(ids.begin(), ids.begin() + position);
      s->resume.length = position;
      s->resume.prompt_only = frontier_save;
      if (frontier_save) s->resume.prompt_boundary = ids.size() - kTemplateTail;
      return std::shared_ptr<const Snapshot>(s);
    };
    const size_t bytes = StateBytes(position);
    const bool saved = frontier_save ? cache_.Capture(ids, policy_, frontier, bytes, Available(), make)
                                     : cache_.CaptureTail(ids, policy_, bytes, Available(), make, true);
    out.refused += !saved;
    if (!saved && !frontier_save) Direct(ids, bytes);
    // The save allocates its chunks now, with the evicted states still queued: the dip of the incident.
    out.lowest = std::min(out.lowest, Available());
  }
  Config config_;
  ConversationPolicy policy_;
  Cache cache_;
  std::vector<uint64_t> queue_;
  size_t calls_ = 0;
  std::map<int, std::vector<std::pair<size_t, bool>>> disk_;  // session: (key length, tail)
};

// 4 sessions in turn, from 100K tokens by 700 to 190K: the cold prefills fill the cache, then each turn saves a tail.
Outcome Run(Config config, int sessions = 4) {
  QueueCache cache(config);
  for (size_t prompt = 100000; prompt <= 190000; prompt += 700)
    for (int session = 1; session <= sessions; ++session) cache.Turn(session, prompt + size_t(session) * 40);
  return cache.out;
}

bool Soak(bool print) {
  const Outcome before = Run({false}), after = Run({true});
  const auto line = [&](const char* name, const Outcome& o) {
    std::printf("queue_model %s lowest=%llu MiB spilled=%zu direct=%zu dropped=%zu refused=%zu hits=%zu/%zu\n",
                name, static_cast<unsigned long long>(o.lowest / kMiB), o.spilled, o.direct, o.dropped, o.refused, o.hits, o.turns);
  };
  if (print) line("trunk", before), line("fixed", after);
  // The model reproduces the incident: the trunk rules dip under the floor plus the margin; the fix does not, and the
  // disk tier still takes states.
  const bool reproduced = before.lowest < kFloor + kMargin;
  const bool bounded = after.lowest >= kFloor + kMargin;
  const bool disk = after.spilled > 0;
  std::printf("queue_model reproduced=%d bounded=%d disk=%d %s\n", int(reproduced), int(bounded), int(disk),
              reproduced && bounded && disk ? "PASS" : "FAIL");
  return reproduced && bounded && disk;
}

// more sessions than slots (an agent and its subagents). The control is the production rule
// before the fix (trunk with --host-reserve-mib 11264, queue limit half of it). The fix must keep the memory floor and
// must not lose resumable state against it: every refused slot tail that the control writes to disk, the fix writes too.
bool MoreSessionsThanSlots(bool print) {
  bool pass = true;
  for (const int sessions : {4, 6, 8}) {
    const uint64_t reserve = 11264 * kMiB;
    const Outcome control = Run({false, reserve, reserve / 2, true}, sessions), fixed = Run({true, kDefaultHostReserve, kQueueLimit, true}, sessions);
    const auto line = [&](const char* name, const Outcome& o) {
      std::printf("queue_model sessions=%d %s lowest=%llu MiB misses=%zu/%zu direct=%zu dropped=%zu spilled=%zu prefilled=%.2f Mtok\n",
                  sessions, name, static_cast<unsigned long long>(o.lowest / kMiB), o.misses, o.turns, o.direct, o.dropped,
                  o.spilled, double(o.prefilled) / 1e6);
    };
    if (print) line("control", control), line("fixed", fixed);
    const bool floor = fixed.lowest >= 11240 * kMiB, misses = fixed.misses <= control.misses,
               prefilled = double(fixed.prefilled) <= 1.10 * double(control.prefilled);
    std::printf("queue_model sessions=%d lowest=%d misses=%d prefilled=%d %s\n", sessions, int(floor), int(misses), int(prefilled),
                floor && misses && prefilled ? "PASS" : "FAIL");
    pass = pass && floor && misses && prefilled;
  }
  return pass;
}

// cache-direct: a direct spill must not take a full host snapshot of the slot tail. The model is the
// 6-session case of the Oct 29 review: the control is production's --host-reserve-mib 11264; "snapshot" is the direct
// spill of the cache-mem2 branch (the whole tail pinned until its write ends); "chunked" holds the fixed state and the
// partial blocks only (about 250 MB, state_disk direct log of the Oct 28 soak) beside a small staging buffer.
bool DirectPeak(bool print) {
  constexpr uint64_t kBound = 512 * kMiB;  // the rest of the state (250 MB) and the staging buffer (32 MiB) with room to spare
  bool pass = true;
  for (const int sessions : {4, 6, 8}) {
    const uint64_t reserve = 11264 * kMiB;
    const Outcome control = Run({false, reserve, reserve / 2, true}, sessions);
    const Outcome chunked = Run({true, kDefaultHostReserve, kQueueLimit, true, true}, sessions);
    if (print)
      std::printf("direct_model sessions=%d control_peak=%llu MiB chunked_peak=%llu MiB chunked_lowest=%llu MiB direct=%zu dropped=%zu prefilled=%.2f/%.2f Mtok\n",
                  sessions, static_cast<unsigned long long>(control.direct_peak / kMiB),
                  static_cast<unsigned long long>(chunked.direct_peak / kMiB), static_cast<unsigned long long>(chunked.lowest / kMiB),
                  chunked.direct, chunked.dropped, double(chunked.prefilled) / 1e6, double(control.prefilled) / 1e6);
    const bool peak = chunked.direct_peak <= kBound, floor = chunked.lowest >= 11240 * kMiB,
               misses = chunked.misses <= control.misses, prefilled = chunked.prefilled <= control.prefilled;
    std::printf("direct_model sessions=%d peak=%d lowest=%d misses=%d prefilled=%d %s\n", sessions, int(peak), int(floor), int(misses),
                int(prefilled), peak && floor && misses && prefilled ? "PASS" : "FAIL");
    pass = pass && peak && floor && misses && prefilled;
    // Only a resumed release is lazy: 5% of the direct spills (a release to another session, a handover) still copy the
    // whole tail. The soak lowest, not this model, decides about --host-reserve-mib; the bar is 10,240 MiB.
    const Outcome mixed = Run({true, kDefaultHostReserve, kQueueLimit, true, true, 5}, sessions);
    const bool bar = mixed.lowest >= 10240 * kMiB, lazy_peak = mixed.direct_peak <= kBound;
    if (print)
      std::printf("direct_model sessions=%d mixed(5%% eager) lazy_peak=%llu MiB eager_peak=%llu MiB lowest=%llu MiB direct=%zu eager=%zu dropped=%zu\n",
                  sessions, static_cast<unsigned long long>(mixed.direct_peak / kMiB), static_cast<unsigned long long>(mixed.eager_peak / kMiB),
                  static_cast<unsigned long long>(mixed.lowest / kMiB), mixed.direct, mixed.eager, mixed.dropped);
    std::printf("direct_model sessions=%d mixed bar=%d lazy_peak=%d %s\n", sessions, int(bar), int(lazy_peak), bar && lazy_peak ? "PASS" : "FAIL");
    pass = pass && bar && lazy_peak;
  }
  return pass;
}
}  // namespace queue_model

int main(int argc, char**) {
  const bool print = argc > 1;
  const bool frontiers = Soak(false, print), replay = Soak(true, print);
  const bool multi = Soak(false, print, MultiTurns()), multi_replay = Soak(true, print, MultiTurns());
  // The h2h layout of Oct 6: --streams 4 --ctx 131072 --conversation-cache-mib 10240, a 31K shared history. At 60K
  // the speculative tails no longer fit beside the four resume points, so they must not push those out. At 75K the
  // four resume points alone overflow 10 GiB: that needs the slot-resident resume (BatchSession), not this cache.
  const bool agents = AgentSoak(4, 31100, 2, print), served = AgentSoak(3, 31100, 2, print);
  const bool crowded = AgentSoak(4, 60000, 2, print);
  const bool queue = queue_model::Soak(print), sessions = queue_model::MoreSessionsThanSlots(print),
             direct = queue_model::DirectPeak(print);
  std::fflush(stdout);
  const bool pass = frontiers && replay && multi && multi_replay && agents && served && crowded && queue && sessions && direct;
  assert(pass);
  return pass ? 0 : 1;
}
