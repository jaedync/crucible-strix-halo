// The state-machine fuzz at the tier level (start plan B1b step 2, first part): the real BatchConversationCache, the
// real disk tier (BatchDiskTier, StateDisk on a test folder, with write_delay_ms for a slow writer) and a mock of the
// tier glue of engine/serve/batched_serve.h (no CPU test includes that file): Release (batched_serve.h Release: the
// tail capture, then Direct and SpillDirect, then DrainHandover), Stage (batched_serve.h Stage: StageWith, the read's
// MakeRoom with the read-ahead rule, ReadRoom) and Lookup (batched_serve.h Lookup: Take, then FindTail). N slots hold
// the tails of the sessions that ran there; N + 1 to 2N sessions take turns in a seeded order.
//
// The reference model knows each session's newest tail (the prompt plus the reply less one token); it does not copy a
// rule of the cache or the disk. Properties (plan B1b):
//   F1 no silent loss: a turn resumes at its session's newest tail, or the miss ledger names a cause (required on
//      this tree; the long mode counts it).
//   F2 exactness: a restored state is a prefix of the prompt, and its bytes are the bytes that were saved for it.
//   F3 no orphan: after each step, each session's newest tail is in a slot, in the host cache, on the disk or in the
//      writer's queue, or the lost list holds a record with a cause that shares half of it.
//   F4 bounded memory: host bytes (cache + writer queue + staged reads) stay at or below budget + slack + one state.
//   F6 bounded stall: no handover waits for the writer more than 2 s (order L14). Counted: B3 step 2 owns it (the long
//      mode of 2,000 seeds counts 52, with the drain of the whole writer queue).
// Seeds: CRUCIBLE_FUZZ_SEEDS (default 6, about 2 s); `--long N` runs N seeds and prints `F2 fail=.. F3 fail=.. F4 fail=..`.
// Named seeds: L4 (a refused read: the base refuses it; B3 reads it), L14 (a slow writer: the drain passes 2 s),
// L4-slot and L4-run (b3-fix: a budget of 2 states, idle slot tails and running turns protect their restore sources).
//
// b3-fix (soak spb3d-n6b, Nov 10): `in_slot` is the real rule of batch_session.h Stage: a point is in a slot when it is
// a PREFIX of a slot's idle tail or of a running turn's prompt (the mock before compared equal keys only, so the
// restore source of a turn was never in a slot here, and the fuzz missed 5 lost turns of the soak). Running turns:
// on some seeds up to N - 1 turns run at a time (batch_session.h `active_`: their prompts are in the slots, and a
// handover takes only a slot that runs no turn); one of them, in a seeded order, ends when one more opens.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

std::mutex pinned_mutex;
std::map<uintptr_t, size_t> pinned;
template<class T> int hipHostMalloc(T** out, size_t n) {
  std::lock_guard lock(pinned_mutex);
  *out = static_cast<T*>(std::malloc(n));
  pinned[uintptr_t(*out)] = n;
  return 0;
}
int hipHostFree(void* p) {
  std::lock_guard lock(pinned_mutex);
  assert(pinned.erase(uintptr_t(p)) == 1);
  std::free(p);
  return 0;
}
constexpr int hipMemcpyDeviceToHost = 0, hipMemcpyHostToDevice = 1;
int hipMemcpy(void* out, const void* in, size_t n, int) { std::memcpy(out, in, n); return 0; }
int hipStreamSynchronize(int) { return 0; }
#define HIP_OK(expr) assert((expr) == 0)
#include "../engine/serve/batch_state_disk.h"
#include "../engine/serve/handover_spill.h"
#include "support/disk_test_dir.h"
#include <chrono>
#include <thread>

using namespace crucible;
namespace fs = std::filesystem;

struct Snapshot {
  std::vector<int> hist;
  RetainedPrefix retained;
  ConversationResume resume;
  SnapshotBytes bytes;
  std::vector<std::pair<int, int>> qsa_layout;
};
using Tier = BatchDiskTier<Snapshot>;
using Cache = BatchConversationCache<Snapshot>;

namespace {
constexpr size_t kTokenBytes = 64;                        // a state's bytes: 64 a token
constexpr uint64_t kFloor = kStageKeepFree;               // the host floor of the server (and of the read room)
ConversationPolicy Policy() {
  ConversationPolicy policy;
  policy.prefill_rows = 8;
  return policy;
}
// The bytes of a state depend only on its tokens: a restore that gives other bytes breaks F2.
uint8_t ByteOf(std::span<const int> hist, size_t at) { return uint8_t(at * 31 + hist.size() * 7 + (hist.empty() ? 0 : hist.back())); }
std::shared_ptr<const Snapshot> Make(std::span<const int> hist) {
  auto s = std::make_shared<Snapshot>();
  s->hist.assign(hist.begin(), hist.end());
  s->resume.length = hist.size();
  s->resume.prompt_only = false;
  s->bytes = SnapshotBytes(hist.size() * kTokenBytes);
  s->bytes.ForEachPiece(0, s->bytes.size(), [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
    for (size_t i = 0; i < count; ++i) chunk.host[inside + i] = ByteOf(hist, at + i);
  });
  return s;
}
bool BytesOf(const Snapshot& s) {
  bool equal = s.bytes.size() == s.hist.size() * kTokenBytes;
  s.bytes.ForEachPiece(0, s.bytes.size(), [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
    for (size_t i = 0; equal && i < count; ++i) equal = chunk.host[inside + i] == ByteOf(s.hist, at + i);
  });
  return equal;
}
fs::path Home() {
  static const fs::path home = [] {
    const DiskTestDir dir = PickDiskTestDir("state_machine_fuzz_test");
    assert(dir.direct);
    ::setenv("HOME", dir.path.c_str(), 1);
    return dir.path;
  }();
  return home;
}
bool Prefix(std::span<const int> a, std::span<const int> b) {
  return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

struct Counts { size_t restarts = 0, turns = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0, f6 = 0, reads = 0, refused_reads = 0; };

// One run: `slots` slots, `sessions` sessions, `turns` turns in a seeded order.
class Fuzz {
 public:
  // `live`: the live resume points that the cache protects (the server: the slot count).
  Fuzz(uint64_t seed, int slots, int sessions, uint64_t headroom_states, int write_delay_ms, size_t first_tokens,
       size_t live = 0, size_t budget_states = 3)
      : rng_(seed), slots_(size_t(slots)), sessions_(size_t(sessions)) {
    StateDiskOptions options;
    options.root = Home() / ("s" + std::to_string(seed) + "-" + std::to_string(slots) + "-" + std::to_string(sessions));
    fs::remove_all(options.root);
    options.identity = std::string(64, '7');
    options.directory = std::string(16, '7') + "-" + std::string(16, '7') + "-" + std::string(16, '7');
    options.min_tokens = 12;
    options.log = [](const std::string&) {};
    options.free_space = [] { return uint64_t(1) << 40; };
    options.write_delay_ms = write_delay_ms;
    root_ = options.root;
    options_ = options;
    live_ = live ? live : size_t(slots);
    state_ = first_tokens * kTokenBytes;
    // The host cache holds about 3 tails, as 9,700 MiB holds 2.8 tails of 130K; the soaks' 8.33 GB entry budget holds 2
    // to 2.5 states of 123K to 153K (b3-fix: budget_states 2).
    budget_ = state_ * budget_states;
    headroom_ = budget_ + headroom_states * state_;  // serve: MemAvailable holds the cache and about 1 state more
    Open();
    for (size_t s = 0; s < sessions_; ++s) {
      std::vector<int> ids(first_tokens);
      std::iota(ids.begin(), ids.end(), int(s) * 1000000 + 1);
      newest_.push_back({});
      convo_.push_back(std::move(ids));
    }
    slot_.assign(slots_, -1);
    resident_.assign(slots_, {});
  }
  // A new process on the same state folder (OpenDisk of batched_serve.h).
  void Open() {
    tier_ = std::make_unique<Tier>(options_, false);
    cache_ = std::make_unique<Cache>(budget_, nullptr, live_, kFloor, false);
    // As OpenDisk (batched_serve.h): the reads and the queue count beside the cache, the floor, the slack, the hook.
    cache_->SetOutside([this] { return size_t(tier_->Disk().StagedBytes()); });
    cache_->SetQueued([this] { return size_t(cache_->Reading() ? 0 : tier_->Disk().QueuedBytes()); });
    cache_->SetHostFloor(kFloor);
    cache_->SetSpillSlack(state_ * 2);  // serve: 2 GiB beside states that share 89% of their blocks
    cache_->SetSpill([this](std::span<const int> key, const ConversationPolicy& policy, const std::shared_ptr<const Cache::Entry>& entry,
                            int, uint64_t room) { return tier_->Spill(key, policy, entry, room, cache_->Reading()); });
    tier_->Attach(*cache_);
  }
  // A stop and a start (batched_serve.h FlushToDisk, order L6): each slot tail is saved and written (FlushOne), then the
  // host entries (FlushHost), the drain, the stop records and the lost list; then a new process on the same folder.
  void Restart() {
    FinishAll();
    for (size_t slot = 0; slot < slots_; ++slot) {
      if (slot_[slot] < 0) continue;
      const auto& tail = resident_[slot];
      tier_->FlushOne(tail, Policy(), false, Make(tail), 30);
      slot_[slot] = -1;
      resident_[slot].clear();
    }
    tier_->FlushHost(*cache_);
    tier_->Drain(30);
    tier_->NoteStop(*cache_);
    tier_->SaveLedger(*cache_);
    cache_.reset();
    tier_.reset();
    Open();
    ++counts.restarts;
  }
  ~Fuzz() {
    tier_->Drain(30);
    tier_.reset();
    std::error_code error;
    fs::remove_all(root_, error);
  }
  // The simulated MemAvailable: the floor and the headroom less what the cache, the queue and the reads pin.
  uint64_t Available() const {
    const uint64_t used = cache_->Bytes() + tier_->Disk().QueuedBytes() + tier_->Disk().StagedBytes();
    const uint64_t total = kFloor + headroom_;
    return total > used ? total - used : 0;
  }
  void Turn(size_t s) {
    ++counts.turns;
    if (settle_) tier_->Drain(5);  // the client's think time: the writer ends its queue before the turn
    for (size_t r = 0; r < running_.size(); ++r)  // a session runs one turn at a time
      if (slot_[size_t(running_[r].slot)] == int(s)) { Finish(r); break; }
    auto prompt = convo_[s];
    const size_t append = 16 + rng_() % 64;
    for (size_t i = 0; i < append; ++i) prompt.push_back(int(s) * 1000000 + int(prompt.size()) + 1);
    // The slot: the one that holds the session, else a free one, else the least recently used (a handover).
    int slot = -1;
    for (size_t i = 0; i < slots_; ++i) if (slot_[i] == int(s)) slot = int(i);
    size_t resumed = 0;
    if (slot >= 0 && Prefix(resident_[size_t(slot)], prompt)) {
      resumed = resident_[size_t(slot)].size();
    } else {
      if (slot < 0) slot = Pick();
      Stage(prompt);
      // batched_serve.h Open: the open holds its prompt while it releases the slot (Fix 2 item 6, Hold), so the
      // release capture never takes an entry that its restore may use.
      cache_->Hold(prompt);
      if (slot_[size_t(slot)] >= 0) Release(size_t(slot));
      resumed = Lookup(prompt);
      cache_->Hold({});
    }
    if (resumed + 4 < newest_[s].size() && Prefix(newest_[s], prompt)) {
      const auto ledger = cache_->LedgerOf(prompt, Policy());
      const bool named = ledger.lost.cause && ledger.lost.shared * 2 >= newest_[s].size();
      if (!named) ++counts.f1;
    }
    // The turn runs: the slot holds its tail (the prompt and a reply, less one token).
    auto tail = prompt;
    for (int i = 0; i < 8; ++i) tail.push_back(int(s) * 1000000 + int(tail.size()) + 1);
    tail.pop_back();
    slot_[size_t(slot)] = int(s);
    used_[size_t(slot)] = ++clock_;
    convo_[s] = tail;
    convo_[s].push_back(int(s) * 1000000 + int(tail.size()) + 1);
    if (overlap_) {
      // The turn runs on (its prompt is in the slot, batch_session.h `active_`, and the slot holds no idle tail); a
      // running turn, in a seeded order, ends when N turns would run (turns of other lengths end in another order).
      running_.push_back({slot, prompt, tail});
      resident_[size_t(slot)].clear();
      if (running_.size() + 1 > slots_) Finish(size_t(rng_() % running_.size()));
    } else {
      resident_[size_t(slot)] = tail;
      newest_[s] = tail;
    }
    Check();
  }
  // Running turn `r` ends: its slot holds its tail.
  void Finish(size_t r) {
    const Running& run = running_[r];
    used_[size_t(run.slot)] = ++clock_;  // the handover order (Pick) is the order in which the slots went idle
    resident_[size_t(run.slot)] = run.tail;
    newest_[size_t(slot_[size_t(run.slot)])] = run.tail;
    running_.erase(running_.begin() + std::ptrdiff_t(r));
  }
  void FinishAll() { while (!running_.empty()) Finish(0); }
  void Run(size_t turns) {
    used_.assign(slots_, 0);
    for (size_t t = 0; t < turns; ++t) {
      if (restart_every_ && t && t % restart_every_ == 0) Restart();
      Turn(size_t(rng_() % sessions_));
    }
    FinishAll();
  }
  Counts counts;
  std::unique_ptr<Tier> tier_;
  std::unique_ptr<Cache> cache_;

 private:
  // A free slot, else the least recently used slot that runs no turn (a handover).
  int Pick() {
    for (size_t i = 0; i < slots_; ++i) if (slot_[i] < 0) return int(i);
    int pick = -1;
    for (size_t i = 0; i < slots_; ++i)
      if (!Runs(i) && (pick < 0 || used_[i] < used_[size_t(pick)])) pick = int(i);
    return pick;
  }
  // batch_session.h Stage `in_slot`: `point` is a prefix of a slot's idle tail or of a running turn's prompt.
  bool InSlot(std::span<const int> point) const {
    for (size_t i = 0; i < slots_; ++i) if (Prefix(point, resident_[i])) return true;
    return std::any_of(running_.begin(), running_.end(), [&](const Running& run) { return Prefix(point, run.prompt); });
  }
  bool Runs(size_t slot) const {
    return std::any_of(running_.begin(), running_.end(), [&](const Running& run) { return run.slot == int(slot); });
  }
  // batched_serve.h Release: the tail capture; when it is refused, Direct (SpillDirect with the lazy host part), then the
  // bounded drain of a handover.
  void Release(size_t slot) {
    const auto& tail = resident_[slot];
    const size_t bytes = tail.size() * kTokenBytes;
    const bool saved = cache_->CaptureTail(tail, Policy(), bytes, Available(), [&] { return Make(tail); }, true);
    if (!saved) cache_->NoteRefused(tail, Policy(), tail.size());
    if (!saved) {
      const size_t host = bytes / 16;  // the lazy plan pins the head and the tail of the state only
      const std::string refused = cache_->Status().last.refused;
      const auto result = tier_->SpillDirect(tail, Policy(), false, refused, bytes, host, cache_->DirectRoom(Available()),
                                             [&] { return Make(tail); }, host);
      cache_->NoteSpill(tail, Policy(), result);
      if (result == "queued") {
        const auto start = std::chrono::steady_clock::now();
        const auto plan = PlanReleaseSpill(false, true, bytes, 1);
        DrainHandover(plan, true, [&](double seconds) { return tier_->Drain(seconds); });
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(2)) ++counts.f6;
      }
    }
    slot_[slot] = -1;
    resident_[slot].clear();
  }
  // batched_serve.h Stage, polled as the server polls a turn that holds a free slot (at most 5 s here).
  void Stage(const std::vector<int>& prompt) {
    const DiskQuery query{prompt, Policy(), prompt.size() - 1, prompt.size() - 1, 8, true};
    const size_t floor = cache_->PeekTail(prompt, Policy());
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
      const auto status = tier_->StageWith(query, floor, [&](const StateMeta& meta) {
        // batched_serve.h Stage `behind`: a point that the read queue lets go (MayTakeFor: start plan B3 and b3-fix).
        BatchReadQueue queue;
        const auto nobody = [&](std::span<const int> point) {
          return queue.MayTakeFor(point, prompt, 0, InSlot(point), tier_->Disk().MinTokens());
        };
        const auto taken = [&](std::span<const int> point, size_t, bool spilled) {
          return spilled || tier_->Holds(point, Policy());
        };
        if (!cache_->MakeRoom(meta.payload, Available() + tier_->Disk().QueuedBytes(), nobody, taken, kFloor))
          return StageAdmit{0, cache_->LastRefusal() == std::string_view("protected") ? "protected" : "room"};
        auto& disk = tier_->Disk();
        switch (ReadRoom(disk.StagedBytes(), Available(), cache_->HostReserve(), disk.UnallocatedReadBytes(),
                         disk.QueuedBytes(), meta.payload)) {
          case ReadRoomStep::Start: return StageAdmit{StageRoom(disk.StagedBytes(), Available(), cache_->HostReserve())};
          case ReadRoomStep::Wait: return StageAdmit{0, nullptr, true};
          case ReadRoomStep::Refuse: break;
        }
        return StageAdmit{0, "room"};
      });
      if (status.state == StageState::None) {
        if (status.reason == "room" || status.reason == "protected") ++counts.refused_reads;
        return;
      }
      if (status.state != StageState::Reading || std::chrono::steady_clock::now() > until) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  // batched_serve.h Lookup: a ready disk read, then the host tail.
  size_t Lookup(const std::vector<int>& prompt) {
    const DiskQuery query{prompt, Policy(), prompt.size() - 1, prompt.size() - 1, 8, true};
    const size_t beat = cache_->PeekTail(prompt, Policy());
    if (const auto ready = tier_->Take(query, beat); ready.snapshot) {
      ++counts.reads;
      if (!Prefix(ready.snapshot->hist, prompt) || !BytesOf(*ready.snapshot)) ++counts.f2;
      const size_t tokens = ready.snapshot->hist.size();
      tier_->Consume(ready);
      return tokens;
    }
    if (const auto entry = cache_->FindTail(prompt, Policy())) {
      if (!Prefix(entry->snapshot->hist, prompt) || !BytesOf(*entry->snapshot)) ++counts.f2;
      return entry->snapshot->hist.size();
    }
    return 0;
  }
  void Check() {
    // F4: the cache, the writer queue and the reads stay within the budget, the slack and one state.
    const uint64_t held = cache_->Bytes() + tier_->Disk().QueuedBytes() + tier_->Disk().StagedBytes();
    if (held > budget_ + state_ * 2 + state_ / 2) ++counts.f4;
    // F3: each newest tail is somewhere, or the lost list names it.
    for (size_t s = 0; s < sessions_; ++s) {
      const auto& tail = newest_[s];
      if (tail.empty()) continue;
      bool found = false;
      for (size_t i = 0; i < slots_; ++i) found |= resident_[i] == tail;
      for (const Running& run : running_)  // the session's turn runs on from it
        found |= slot_[size_t(run.slot)] == int(s) && Prefix(tail, run.prompt);
      auto next = tail;  // a prompt that repeats the tail (PeekTail leaves one row to prefill)
      next.push_back(-1);
      found = found || cache_->PeekTail(next, Policy()) >= tail.size() || tier_->Holds(tail, Policy());
      if (!found) {
        const auto ledger = cache_->LedgerOf(tail, Policy());
        found = ledger.lost.cause && ledger.lost.shared * 2 >= tail.size();
      }
      if (!found) ++counts.f3;
    }
  }
  std::mt19937_64 rng_;
  size_t slots_, sessions_;
  fs::path root_;
  size_t state_ = 0, budget_ = 0;
  uint64_t headroom_ = 0;
  std::vector<std::vector<int>> convo_, newest_, resident_;
  std::vector<int> slot_;
  std::vector<uint64_t> used_;
  struct Running { int slot; std::vector<int> prompt, tail; };
  std::vector<Running> running_;  // the running turns, oldest first
  uint64_t clock_ = 0;
  StateDiskOptions options_;
  size_t live_ = 0;

 public:
  bool settle_ = false;
  bool overlap_ = false;  // up to N - 1 turns run at a time
  size_t restart_every_ = 0;
};

Counts RunSeed(uint64_t seed) {
  std::mt19937_64 pick(seed);
  const int slots = std::array<int, 4>{2, 4, 6, 8}[pick() % 4];
  const int sessions = slots + 1 + int(pick() % size_t(slots));
  // Odd seeds protect one live point only, so the reads run and F2 checks the bytes of disk restores; even seeds use
  // the server's rule (one live point a slot).
  Fuzz fuzz(seed, slots, sessions, pick() % 3, 0, 2048 + pick() % 2048, seed % 2 ? 1 : 0);
  fuzz.settle_ = seed % 3 != 0;
  fuzz.overlap_ = seed / 2 % 2 == 1;  // seeds 2, 3, 6, 7, ...: running turns
  fuzz.restart_every_ = seed % 4 == 0 ? size_t(sessions) * 2 : 0;
  fuzz.Run(size_t(sessions) * 8);
  return fuzz.counts;
}
}  // namespace

int main(int argc, char** argv) {
  size_t seeds = 6;
  if (const char* text = std::getenv("CRUCIBLE_FUZZ_SEEDS")) seeds = size_t(std::strtoull(text, nullptr, 10));
  if (argc == 3 && std::string(argv[1]) == "--long") seeds = size_t(std::strtoull(argv[2], nullptr, 10));
  Counts all;
  for (uint64_t seed = 1; seed <= seeds; ++seed) {
    const Counts c = RunSeed(seed);
    all.turns += c.turns, all.f1 += c.f1, all.f2 += c.f2, all.f3 += c.f3, all.f4 += c.f4, all.f6 += c.f6;
    all.reads += c.reads, all.refused_reads += c.refused_reads, all.restarts += c.restarts;
  }
  std::printf("F2 fail=%zu F3 fail=%zu F4 fail=%zu (seeds %zu, turns %zu; counted: F1 %zu, F6 %zu, reads %zu, refused "
              "reads %zu, restarts %zu)\n", all.f2, all.f3, all.f4, seeds, all.turns, all.f1, all.f6, all.reads, all.refused_reads,
              all.restarts);
  std::fflush(stdout);
  {
    // Seed L14: a slow writer (1.2 s a write) and two handovers with queued writes: the second drain waits for the whole
    // queue, more than 2 s (F6). Recorded RED on the base (the plan's step 2 owns it).
    Fuzz fuzz(9001, 2, 4, 0, 1200, 2048);
    fuzz.Run(8);
    std::printf("seed L14: F6 %zu (%s on this tree)\n", fuzz.counts.f6, fuzz.counts.f6 ? "RED" : "GREEN");
  }
  {
    // Seed L4: 6 sessions on 4 slots, the server's protection (4 live points), little headroom: the turns whose state
    // is on the disk find the host cache full of live points with no disk copy, and the read is refused ("protected").
    // RED on the base (the read-ahead rule takes only points of turns that wait behind); GREEN with B3 MayTake.
    Fuzz fuzz(9002, 4, 6, 1, 0, 2048);
    fuzz.Run(36);
    std::printf("seed L4: refused reads %zu, reads %zu (%s on this tree)\n", fuzz.counts.refused_reads, fuzz.counts.reads,
                fuzz.counts.refused_reads ? "RED" : "GREEN");
  }
  // b3-fix seeds L4-slot, L4-run and L8-run: a budget of 2 states (as the soaks' 8.33 GB entry budget), so the host
  // points are the restore sources of the sessions in the slots. L4-slot: the order of L4 (6 sessions on 4 slots, idle
  // slot tails). L4-run (6 sessions) and L8-run (8 sessions) on 4 slots: up to 3 turns run at a time (as the 6x4 and
  // 8x4 soaks). With the B3 rule (no in-slot point leaves) they refuse 4, 13 and 8 reads ("protected") and lose as
  // many turns (F1); with MayTakeFor an in-slot point leaves through the disk.
  struct Named { const char* name; uint64_t seed; int sessions; bool overlap; };
  constexpr std::array<Named, 3> kNamed{{{"L4-slot", 9002, 6, false}, {"L4-run", 9003, 6, true}, {"L8-run", 9006, 8, true}}};
  std::array<Counts, 3> named{};
  for (size_t n = 0; n < kNamed.size(); ++n) {
    Fuzz fuzz(kNamed[n].seed, 4, kNamed[n].sessions, 1, 0, 2048, 0, 2);
    fuzz.overlap_ = kNamed[n].overlap;
    fuzz.Run(kNamed[n].overlap ? size_t(kNamed[n].sessions) * 8 : 36);
    named[n] = fuzz.counts;
    const Counts& c = named[n];
    std::printf("seed %s: F1 %zu, refused reads %zu, reads %zu, turns %zu, F2 %zu F3 %zu F4 %zu (%s on this tree)\n",
                kNamed[n].name, c.f1, c.refused_reads, c.reads, c.turns, c.f2, c.f3, c.f4,
                c.f1 || c.refused_reads ? "RED" : "GREEN");
  }
  std::fflush(stdout);
  assert(all.turns > 0 && all.reads > 0 && (seeds < 4 || all.restarts > 0));
  assert(all.f2 == 0 && all.f3 == 0 && all.f4 == 0);
  // F1 holds on this tree (B3: the read may take an idle live point, MayTake). With the base rule (a read takes only the
  // points of turns that wait behind it) the same seeds give F1 17 and 17 refused reads in 408 turns (order L4).
  assert(all.f1 == 0 && all.refused_reads == 0);
  for (const Counts& c : named) {
    assert(c.f2 == 0 && c.f3 == 0 && c.f4 == 0);
    assert(c.f1 == 0 && c.refused_reads == 0 && c.reads > 0);
  }
  std::printf("state_machine_fuzz_test passed\n");
  return 0;
}
