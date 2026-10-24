// The disk tier of batch serve (engine/serve/batch_state_disk.h): host evictions reach the disk, a new process finds
// them, and a request restores only a state that the host cache's own rules would accept.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
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
#include "support/disk_test_dir.h"
#include <chrono>
#include <thread>
#include <atomic>
#include <future>

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

fs::path Home() {
  static const fs::path home = [] {
    const DiskTestDir dir = PickDiskTestDir("batch_state_disk_test");
    // The spill must take O_DIRECT, as state_disk_test requires (a tmpfs root hides the disk path).
    assert(dir.direct);
    ::setenv("HOME", dir.path.c_str(), 1);
    return dir.path;
  }();
  return home;
}
StateDiskOptions Options() {
  StateDiskOptions options;
  options.root = Home() / "states";
  options.identity = std::string(64, '7');
  options.directory = std::string(16, '7') + "-" + std::string(16, '7') + "-" + std::string(16, '7');
  options.min_tokens = 12;
  options.log = [](const std::string&) {};
  return options;
}
ConversationPolicy Policy() {
  ConversationPolicy policy;
  policy.prefill_rows = 8;
  return policy;
}
std::vector<int> Tokens(int first, size_t count) {
  std::vector<int> ids(count);
  std::iota(ids.begin(), ids.end(), first);
  return ids;
}
std::shared_ptr<const Snapshot> Make(std::span<const int> hist, bool tail, size_t boundary) {
  auto s = std::make_shared<Snapshot>();
  s->hist.assign(hist.begin(), hist.end());
  s->resume.length = hist.size();
  s->resume.prompt_only = !tail;
  s->resume.prompt_boundary = tail ? 0 : boundary;
  s->bytes = SnapshotBytes(3000 + hist.size());
  s->bytes.ForEachPiece(0, s->bytes.size(), [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
    for (size_t i = 0; i < count; ++i) chunk.host[inside + i] = uint8_t(at + i + hist.size());
  });
  return s;
}
bool Capture(Cache& cache, const std::vector<int>& ids, size_t position) {
  return cache.Capture(ids, Policy(), {position, CacheRetention::Session}, 64, uint64_t(64) << 30,
                       [&] { return Make(std::span<const int>(ids).first(position), false, ids.size() - 1); });
}
bool Tail(Cache& cache, const std::vector<int>& ids, size_t length) {
  return cache.CaptureTail(std::span<const int>(ids).first(length), Policy(), 64, uint64_t(64) << 30,
                           [&] { return Make(std::span<const int>(ids).first(length), true, 0); });
}
DiskQuery Query(const std::vector<int>& prompt, bool replay = true) {
  return {prompt, Policy(), prompt.size() - 1, prompt.size() - 1, 8, replay};
}
StageStatus StageWait(Tier& tier, const DiskQuery& query, size_t floor) {
  for (int i = 0; i < 2000; ++i) {
    const auto status = tier.Stage(query, floor, UINT64_MAX);
    if (status.state != StageState::Reading) return status;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  assert(false);
  return {};
}

// Stage reserves a read before the reader allocates it, so MemAvailable does not show those bytes
// yet. A spill's room leaves them out until the read state holds its memory.
void SpillRoomLeavesUnallocatedReads() {
  auto options = Options();
  options.root = Home() / "states-room";
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<int> waiting = 0;
  options.before_read = [&] { ++waiting; gate.wait(); };
  Tier tier(options, false);
  const auto x = Tokens(1, 40), y = Tokens(101, 40);
  const auto xs = Make(std::span<const int>(x).first(16), false, x.size() - 1);
  assert(tier.Disk().Spill(xs->hist, Policy(), false, false, false, xs, *xs).empty() && tier.Drain(30));
  assert(tier.Stage(Query(x), 0, UINT64_MAX).state == StageState::Reading);
  while (waiting == 0) std::this_thread::yield();
  const uint64_t pending = xs->bytes.size();
  assert(tier.Disk().UnallocatedReadBytes() == pending);
  const auto ys = Make(std::span<const int>(y).first(16), false, y.size() - 1);
  const auto entry = std::make_shared<const Cache::Entry>(Cache::Entry{{}, ys, CacheRetention::Session});
  const uint64_t need = ys->bytes.size();
  assert(!tier.Spill(std::span<const int>(y).first(16), Policy(), entry, pending + need - 1));  // refused: memory
  release.set_value();
  assert(StageWait(tier, Query(x), 0).state == StageState::Ready && tier.Disk().UnallocatedReadBytes() == 0);
  assert(tier.Spill(std::span<const int>(y).first(16), Policy(), entry, need) && tier.Drain(30));
}

// MEDIUM-5: a slot state that the host cache refuses goes straight to
// disk when host memory holds its new bytes, and the next process restores it. The device copy runs only when the
// writer takes the state.
void DirectSpill() {
  auto options = Options();
  options.root = Home() / "states-direct";
  options.queue_bytes = 1;  // one state that waits for the writer fills the queue
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<bool> hold = false;
  std::atomic<int> waiting = 0;
  options.before_write = [&] { if (hold) { ++waiting; gate.wait(); } };
  const auto a = Tokens(601, 40), b = Tokens(701, 40), c = Tokens(801, 40);
  const std::span<const int> tail(b.data(), 30);
  std::shared_ptr<const Snapshot> saved;
  int saves = 0;
  const auto save = [&] { ++saves; return saved = Make(tail, true, 0); };
  const size_t bytes = 3030, need = 1000;
  {
    Tier tier(options, false);
    // The host cache holds a's frontier, which a speculative tail may not take: it refuses "protected".
    Cache cache(64, {}, 1);
    assert(Capture(cache, a, 16) && !Tail(cache, b, 30));
    const std::string refused = cache.Status().last.refused;
    assert(refused == "protected");
    // A policy refusal never goes to disk; no room for the new bytes: refused before the device copy.
    assert(tier.SpillDirect(tail, Policy(), false, "policy", bytes, need, UINT64_MAX, save).empty());
    assert(tier.SpillDirect(tail, Policy(), false, "resident", bytes, need, UINT64_MAX, save).empty());
    assert(tier.SpillDirect(tail, Policy(), false, refused, bytes, need, need - 1, save) == "memory" && saves == 0);
    // `need`, not the whole state, must fit the room (the shared chunks are in host memory already).
    assert(tier.SpillDirect(tail, Policy(), false, refused, bytes, need, need, save) == "queued" && saves == 1);
    assert(tier.Drain(30) && tier.Disk().Present(tail, Policy(), false));
    // The same state again: on disk, no device copy.
    assert(tier.SpillDirect(tail, Policy(), false, "host", bytes, need, UINT64_MAX, save) == "present" && saves == 1);
    // The writer holds one state and another one waits: the queue is full, so no device copy.
    hold = true;
    // cw is of another branch: a waiting prefix of the new state would give its place (fix 3, TailTakesPrefixQueuePlace).
    const auto e = Tokens(901, 40);
    const auto cs = Make(std::span<const int>(c).first(20), true, 0), cw = Make(std::span<const int>(e).first(21), true, 0);
    assert(tier.Disk().Spill(cs->hist, Policy(), false, false, false, cs, *cs).empty());
    while (waiting == 0) std::this_thread::yield();
    // The writer writes this state already: no device copy.
    assert(tier.SpillDirect(cs->hist, Policy(), false, "protected", bytes, need, UINT64_MAX, save) == "waiting" &&
           saves == 1);
    assert(tier.Disk().Spill(cw->hist, Policy(), false, false, false, cw, *cw).empty());
    assert(tier.SpillDirect(std::span<const int>(c).first(30), Policy(), false, "budget", bytes, need, UINT64_MAX,
                            save) == "busy" && saves == 1);
    release.set_value();
    assert(tier.Drain(30));
  }
  {
    auto small = options;
    small.write_per_day = 1000;
    Tier limited(std::move(small), false);
    assert(limited.SpillDirect(std::span<const int>(c).first(30), Policy(), false, "memory", bytes, need,
                               UINT64_MAX, save) == "daily" && saves == 1);
  }
  // The next process restores the state that went straight to disk, with the bytes that Save wrote.
  options.before_write = {};
  Tier tier(options, false);
  assert(StageWait(tier, Query(b), 0).state == StageState::Ready);
  const auto ready = tier.Take(Query(b), 0);
  assert(ready.snapshot && ready.meta.key.size() == 30 && !ready.meta.resume.prompt_only);
  assert(ready.snapshot->hist == saved->hist && ready.snapshot->bytes == saved->bytes);
  tier.Consume(ready);
}

// An earlier change: the running tier (no restart) writes a grid-aligned prompt frontier when the host cache
// evicts it, and the session's next prompt (the same prefix, a dropped reasoning block, new tokens) stages and takes it.
// H2 and H3 of the brief (Best, Matches, DiskStateFits) in the soak's storage mode (CRUCIBLE_COMPACT_STATE=1).
void InRunFrontier() {
  auto options = Options();
  options.root = Home() / "states-inrun";
  Tier tier(options, true);
  const auto turn0 = Tokens(1001, 30);  // a 24-token grid frontier (3 chunks of 8), the generation header after it
  auto turn1 = std::vector<int>(turn0.begin(), turn0.begin() + 27);
  for (int i = 0; i < 12; ++i) turn1.push_back(5000 + i);  // the reasoning is gone; a tool result follows
  DiskQuery query{turn1, Policy(), turn1.size() - 3, turn1.size() - 1, 8, true};
  const auto frontier = Make(std::span<const int>(turn0).first(24), false, turn0.size() - 1);
  // The turn arrives while the host cache holds the frontier: the floor beats the disk (nothing on disk yet).
  auto status = tier.Stage(query, 24, UINT64_MAX);
  assert(status.state == StageState::None && status.reason == "floor" && status.tokens == 0);
  assert(StageLine(status, turn1.size(), 24, 7) ==
         "state_disk stage state=none tokens=39 floor=24 best=0 room=7 reason=floor job=0\n");
  assert(tier.Stage(query, 0, UINT64_MAX).reason == "nostate");
  // The host cache evicts it while the turn waits: the writer puts it on disk in this process.
  const auto entry = std::make_shared<const Cache::Entry>(Cache::Entry{{}, frontier, CacheRetention::Session});
  assert(tier.Spill(frontier->hist, Policy(), entry, UINT64_MAX) && tier.Drain(30));
  status = tier.Stage(query, 24, UINT64_MAX);  // a host entry of the same length still wins
  assert(status.state == StageState::None && status.tokens == 24 && status.reason == "floor");
  // The read room is the last guard ("room"), as the host reserve was before.
  status = tier.Stage(query, 12, frontier->bytes.size() - 1);
  assert(status.state == StageState::None && status.reason == "room");
  status = tier.Stage(query, 12, UINT64_MAX);
  assert(status.state == StageState::Reading && status.reason == "start" && status.tokens == 24);
  // A poll of the held read never starts a second read, even when a longer state reaches the disk (rule 1).
  const auto longer = Make(std::span<const int>(turn1).first(32), false, turn1.size() - 3);
  const auto longer_entry = std::make_shared<const Cache::Entry>(Cache::Entry{{}, longer, CacheRetention::Session});
  assert(tier.Spill(longer->hist, Policy(), longer_entry, UINT64_MAX) && tier.Drain(30));
  status = StageWait(tier, query, 12);
  assert(status.state == StageState::Ready && status.tokens == 24 && status.reason.empty());
  assert(tier.Disk().StagedBytes() == frontier->bytes.size());
  const auto ready = tier.Take(query, 12);
  assert(ready.snapshot && ready.meta.key.size() == 24 && ready.meta.resume.prompt_only);
  assert(ready.snapshot->hist == frontier->hist && ready.snapshot->bytes == frontier->bytes);  // Save's bytes
  tier.Consume(ready);
  assert(tier.Disk().StagedBytes() == 0);
  // With the read gone, the next stage takes the longer state.
  assert(StageWait(tier, query, 12).tokens == 32);
}

// An earlier change: the read room keeps MemAvailable at kStageKeepFree (10,240 MiB now; 12,288 MiB before) or
// more after the read; a larger host reserve wins; all staged reads together stay under kStageCap.
void StageRoomRule() {
  constexpr uint64_t mib = uint64_t(1) << 20;
  static_assert(kStageKeepFree == 10240 * mib);
  assert(StageRoom(0, 15000 * mib, 8192 * mib) == (15000 - 10240) * mib);
  assert(StageRoom(0, 10000 * mib, 8192 * mib) == 0);
  assert(StageRoom(3000 * mib, 10000 * mib, 8192 * mib) == 3000 * mib);  // held reads are in MemAvailable already
  assert(StageRoom(0, 20000 * mib, 16384 * mib) == (20000 - 16384) * mib);
  // Review MEDIUM-1: two reads in flight. The first one (3000 MiB) still reads and has not taken its memory, so
  // MemAvailable shows it: the room leaves it out, and no second read fits above 10,240 MiB.
  assert(StageRoom(3000 * mib, 13000 * mib, 8192 * mib, 3000 * mib) == 3000 * mib);
  assert(StageRoom(3000 * mib, 13000 * mib, 8192 * mib, 0) == (3000 + 13000 - 10240) * mib);
  assert(StageRoom(3000 * mib, 2000 * mib, 8192 * mib, 3000 * mib) == 3000 * mib);  // no underflow
  // The read room rule (design decision, Oct 20): a 110K read is 2.9 GiB.
  const uint64_t gib10 = (uint64_t(1) << 30) / 10;  // 0.1 GiB
  // An earlier change: at the daily day low (MemAvailable 13.7 GiB) the read starts at once; after it 10.8 GiB stay free.
  assert(ReadRoom(0, 137 * gib10, 8192 * mib, 0, 0, 29 * gib10) == ReadRoomStep::Start && 137 - 29 >= 100);
  // (i) At 11.7 GiB its eviction gave 3 GiB of host entries to the disk writer: it waits (no "room").
  assert(ReadRoom(0, 117 * gib10, 8192 * mib, 0, 30 * gib10, 29 * gib10) == ReadRoomStep::Wait);
  // The writes end and free the 3 GiB (14.7 GiB): it starts; after its allocation 11.8 GiB, not below 10 GiB.
  assert(ReadRoom(0, 147 * gib10, 8192 * mib, 0, 0, 29 * gib10) == ReadRoomStep::Start && 147 - 29 >= 100);
  // A second read in flight that has not allocated takes the room again: the next one waits or is refused.
  assert(ReadRoom(29 * gib10, 147 * gib10, 8192 * mib, 29 * gib10, 0, 29 * gib10) == ReadRoomStep::Refuse);
  // (ii) No entry to free: refused ("room").
  assert(ReadRoom(0, 117 * gib10, 8192 * mib, 0, 0, 29 * gib10) == ReadRoomStep::Refuse);
  // Not enough even after the writes: refused at once, no wait.
  assert(ReadRoom(0, 117 * gib10, 8192 * mib, 0, 2 * gib10, 29 * gib10) == ReadRoomStep::Refuse);
  assert(StageRoom(0, uint64_t(64) << 30, 8192 * mib) == kStageCap);
  // The daily a4v profile: a 110,592-token state is 3,040,047,104 bytes. It needs about 13,140 MiB of MemAvailable.
  const uint64_t state = 3040047104;
  assert(StageRoom(0, 10240 * mib + state, 8192 * mib) >= state && StageRoom(0, 10240 * mib + state - 1, 8192 * mib) < state);
  assert(StageReason("") == "-" && StageReason("io errno=5") == "io_errno_5" && StageReason("Room") == "_oom");
}

// An earlier change (decision A): a read is inside the cache budget. The cache evicts for it as for a save; a live resume
// point leaves only when the read-ahead rule allows it (its session waits behind the reading turn), never a staged read
// (outside bytes), and each such eviction goes to the spill hook and is reported.
void MakeRoomForRead() {
  const auto a = Tokens(2001, 20), b = Tokens(2101, 20), c = Tokens(2201, 20);
  Cache cache(64 * 2, {}, 2, 0, false);  // two live resume points fill the budget; no probation
  std::vector<size_t> spilled;
  cache.SetSpill([&](std::span<const int> key, const ConversationPolicy&, const std::shared_ptr<const Cache::Entry>&, int,
                     uint64_t) { spilled.push_back(key.size()); return true; });
  size_t outside = 0;
  cache.SetOutside([&] { return outside; });
  assert(Capture(cache, a, 16) && Capture(cache, b, 16));
  const uint64_t memory = uint64_t(64) << 30;
  // Without the rule the live points stay: "protected" (the read says "budget").
  assert(!cache.MakeRoom(64, memory) && std::string_view(cache.LastRefusal()) == "protected" && spilled.empty());
  // The rule allows only b's point: b leaves to disk and is reported; a stays.
  std::vector<std::pair<size_t, bool>> taken;
  const auto only_b = [&](std::span<const int> key) { return std::equal(key.begin(), key.end(), b.begin()); };
  assert(cache.MakeRoom(64, memory, only_b, [&](std::span<const int> key, size_t bytes, bool held) {
    assert(bytes == 64);
    taken.emplace_back(key.size(), held);
    return held;
  }));
  assert((taken == std::vector<std::pair<size_t, bool>>{{16, true}}) && (spilled == std::vector<size_t>{16}));
  assert(cache.Find(a, Policy(), 19, 19, 8) && !cache.Find(b, Policy(), 19, 19, 8));
  // The read now holds its bytes outside the entries: a save makes room for them too (c evicts nothing protected and
  // is refused), and a second read cannot take the first one's bytes or a's protected point.
  outside = 64;
  assert(!Capture(cache, c, 16) && cache.Find(a, Policy(), 19, 19, 8));
  assert(!cache.MakeRoom(64, memory, [](std::span<const int>) { return false; }) && cache.Find(a, Policy(), 19, 19, 8));
  // The rule is gone after MakeRoom: a later save never takes a live point by it.
  outside = 0;
  assert(Capture(cache, c, 16) && cache.Find(a, Policy(), 19, 19, 8) && spilled.size() == 1);
}

// Review MEDIUM-2: the read's eviction keeps `keep_free` (kStageKeepFree) of host memory, not only the host reserve, so
// the eviction frees what the last guard needs and is not wasted.
void MakeRoomKeepsFree() {
  const auto a = Tokens(2301, 20), b = Tokens(2401, 20);
  Cache cache(64 * 4, {}, 2, 0, false);  // the budget holds a read without an eviction
  cache.SetSpill([](std::span<const int>, const ConversationPolicy&, const std::shared_ptr<const Cache::Entry>&, int,
                    uint64_t) { return true; });
  assert(Capture(cache, a, 16) && Capture(cache, b, 16));
  const uint64_t keep = uint64_t(1) << 30;
  size_t taken = 0;
  const auto only_b = [&](std::span<const int> key) { return std::equal(key.begin(), key.end(), b.begin()); };
  const auto count = [&](std::span<const int>, size_t, bool) { ++taken; return true; };
  // Host reserve 0: the memory holds the read, nothing leaves.
  assert(cache.MakeRoom(64, keep + 32, only_b, count) && taken == 0 && cache.Find(b, Policy(), 19, 19, 8));
  // With keep_free, the read needs keep + 64 bytes: b's point leaves (the rule allows it), a stays.
  assert(cache.MakeRoom(64, keep + 32, only_b, count, keep) && taken == 1 && !cache.Find(b, Policy(), 19, 19, 8));
  assert(cache.Find(a, Policy(), 19, 19, 8));
  // Not enough even then: refused, nothing more leaves.
  assert(!cache.MakeRoom(64, 32, only_b, count, keep) && taken == 1 && cache.Find(a, Policy(), 19, 19, 8));
}

// An earlier change (decision A): which live points a read may take, and the tags that link a taken point to its turn.
void ReadQueueRule() {
  BatchReadQueue queue;
  const auto shared = Tokens(1, 8);
  auto with = [&](int first) { auto key = shared; for (int i = 0; i < 12; ++i) key.push_back(first + i); return key; };
  const auto reader = with(100), behind = with(200), ahead = with(300), idle = with(400), slot = with(500);
  const auto point = [](const std::vector<int>& key) { return std::span<const int>(key).first(16); };
  // Review MEDIUM-3: only the next turn (position 0) may take a point, and the places are fresh (kWaitSeconds 25).
  queue.Saw(reader, 0, 10);
  queue.Saw(ahead, 1, 10);
  queue.Saw(behind, 2, 10);
  assert(BatchReadQueue::kWaitSeconds == 25);
  assert(queue.Behind(point(behind), reader, 0, 20));     // a session behind the reader: its point may leave
  assert(queue.Behind(point(ahead), reader, 0, 20));      // (place 1 is behind place 0 too)
  assert(!queue.Behind(point(behind), reader, 1, 20));    // a reader that is not next takes nothing
  assert(!queue.Behind(point(reader), reader, 0, 20));    // the reader's own state
  assert(!queue.Behind(shared, reader, 0, 20));           // a prefix that the reader shares
  assert(!queue.Behind(point(idle), reader, 0, 20));      // an idle session: no queued turn
  assert(!queue.Behind(point(slot), reader, 0, 20));      // a session in a slot has no queued turn either
  assert(!queue.Behind(point(behind), reader, 0, 10 + BatchReadQueue::kWaitSeconds + 1));  // a turn that left
  queue.Saw(behind, 0, 30);                               // it is next now: the old reader's place is old
  queue.Saw(reader, 1, 30);
  assert(!queue.Behind(point(behind), reader, 1, 31) && queue.Behind(point(reader), behind, 0, 31));
  queue.Saw(behind, 2, 32);
  queue.Saw(reader, 0, 32);
  queue.Opened(behind);                                   // it opened: no longer waits
  assert(!queue.Behind(point(behind), reader, 0, 33) && queue.Waiting() == 2);
  // Tags: a restore of the taken point names it; an open without that restore is a miss; others are 0.
  const uint64_t one = queue.Took(point(behind)), two = queue.Took(point(idle));
  assert(one == 1 && two == 2 && queue.TagOf(behind) == 1 && queue.TagOf(reader) == 0 && queue.TagOf(point(idle)) == 2);
  // Review MEDIUM-5 (c): a restore of a longer state of the same session also names the taken point.
  assert(queue.Restored(point(reader)) == 0 && queue.Restored(behind) == 1 && queue.Restored(point(behind)) == 0);
  assert(queue.Missed(reader) == std::make_pair(uint64_t(0), size_t(0)));
  assert(queue.Missed(idle) == std::make_pair(uint64_t(2), size_t(16)) && queue.Missed(idle).first == 0);
}

// Fix 2 (cause.md items 1, 2, 5): the turn of a session whose slot state went straight to disk a moment before (the
// daily slot-3 turn of Oct 21, request 832; the same-second case) finds that state while it waits for the writer or
// while the writer writes it: Stage says Reading "writing", and the poll after the write reads the bytes that Save
// wrote. On f3a2a3a1 Stage said None "nostate" and the turn prefilled its whole prompt.
void StageWaitsForWriter() {
  auto options = Options();
  options.root = Home() / "states-writing";
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<int> waiting = 0;
  options.before_write = [&] { ++waiting; gate.wait(); };
  Tier tier(options, false);
  const auto a = Tokens(3001, 40), b = Tokens(3101, 40);
  const std::span<const int> tail(b.data(), 30);
  std::shared_ptr<const Snapshot> saved;
  const auto save = [&] { return saved = Make(tail, true, 0); };
  // The writer writes a's state now; b's slot tail goes straight to disk and waits behind it.
  const auto as = Make(std::span<const int>(a).first(20), true, 0);
  assert(tier.Disk().Spill(as->hist, Policy(), false, false, false, as, *as).empty());
  while (waiting == 0) std::this_thread::yield();
  assert(tier.SpillDirect(tail, Policy(), false, "protected", 3030, 1000, UINT64_MAX, save) == "queued");
  assert(!tier.Disk().Present(tail, Policy(), false) && tier.Holds(tail, Policy()));
  // In the queue (item 5: Best does not see it; Pending does).
  auto status = tier.Stage(Query(b), 0, UINT64_MAX);
  assert(status.state == StageState::Reading && status.reason == "writing" && status.tokens == 30);
  assert(StageLine(status, 40, 0, 7, 12) ==
         "state_disk stage state=reading tokens=40 floor=0 best=30 room=7 reason=writing job=12\n");
  // In the writer's hands.
  status = tier.Stage(Query(a), 0, UINT64_MAX);
  assert(status.state == StageState::Reading && status.reason == "writing" && status.tokens == 20);
  // A floor (a slot or the host cache) that holds as much starts no wait; a queued state older than the bound counts
  // as a stuck writer (Pending's max_age).
  status = tier.Stage(Query(b), 30, UINT64_MAX);
  assert(status.state == StageState::None && status.reason == "floor" && status.tokens == 0);
  const auto any = [](const StateMeta&) { return true; };
  assert(tier.Disk().Pending(tail, Policy(), false, any, Tier::kPendingSeconds));
  assert(!tier.Disk().Pending(tail, Policy(), false, any, -1));
  assert(Tier::kPendingSeconds == 120);
  release.set_value();
  status = StageWait(tier, Query(b), 0);
  assert(status.state == StageState::Ready && status.tokens == 30);
  const auto ready = tier.Take(Query(b), 0);
  assert(ready.snapshot && ready.meta.key.size() == 30 && !ready.meta.resume.prompt_only);
  assert(ready.snapshot->hist == saved->hist && ready.snapshot->bytes == saved->bytes);  // Save's bytes
  tier.Consume(ready);
  assert(tier.Drain(30));
}

// The same-second proof flag (serve --state-disk-write-delay-ms): each write waits, and the start line says so
// (the SWITCH line of the GPU test); unset, the start line has no such field.
void WriteDelay() {
  auto options = Options();
  options.root = Home() / "states-delay";
  std::vector<std::string> lines;
  std::mutex lines_mutex;
  options.log = [&](const std::string& line) { std::lock_guard lock(lines_mutex); lines.push_back(line); };
  options.write_delay_ms = 300;
  const auto a = Tokens(3201, 40);
  const auto as = Make(std::span<const int>(a).first(20), true, 0);
  {
    Tier tier(options, false);
    const auto start = std::chrono::steady_clock::now();
    assert(tier.Disk().Spill(as->hist, Policy(), false, false, false, as, *as).empty());
    // Inside the delay the state is not on disk yet; the stage waits for it.
    assert(tier.Stage(Query(a), 0, UINT64_MAX).reason == "writing");
    assert(tier.Drain(30) && tier.Disk().Present(as->hist, Policy(), false));
    assert(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(300));
  }
  // Every tier has stopped: no thread logs now.
  assert(std::any_of(lines.begin(), lines.end(), [](const std::string& line) {
    return line.find("write_delay_ms=300") != std::string::npos;
  }));
  options.write_delay_ms = 0;
  lines.clear();
  { Tier tier(options, false); }
  assert(std::none_of(lines.begin(), lines.end(), [](const std::string& line) {
    return line.find("write_delay_ms") != std::string::npos;
  }));
}

// Fix 2 (cause.md items 7 and 8; x6h request 19): a read-ahead point whose spill the disk refuses (and whose state is
// not on disk) stays in the cache, and the read is refused with LastRefusal "spill". A point that the disk holds
// already may leave without a spill.
void ReadAheadKeptOnRefusal() {
  const auto a = Tokens(2501, 20), b = Tokens(2601, 20);
  Cache cache(64 * 2, {}, 2, 0, false);
  bool accept = false;
  cache.SetSpill([&](std::span<const int>, const ConversationPolicy&, const std::shared_ptr<const Cache::Entry>&, int,
                     uint64_t) { return accept; });
  assert(Capture(cache, a, 16) && Capture(cache, b, 16));
  const uint64_t memory = uint64_t(64) << 30;
  const auto only_b = [&](std::span<const int> key) { return std::equal(key.begin(), key.end(), b.begin()); };
  std::vector<std::pair<size_t, bool>> taken;
  bool on_disk = false;
  const auto report = [&](std::span<const int> key, size_t, bool spilled) {
    taken.emplace_back(key.size(), spilled);
    return spilled || on_disk;
  };
  const size_t bytes = cache.Bytes();
  assert(!cache.MakeRoom(64, memory, only_b, report) && std::string_view(cache.LastRefusal()) == "spill");
  assert((taken == std::vector<std::pair<size_t, bool>>{{16, false}}));
  assert(cache.Find(b, Policy(), 19, 19, 8) && cache.Find(a, Policy(), 19, 19, 8) && cache.Bytes() == bytes);
  assert(!cache.Reading());
  // The point is on disk already: it leaves, and the read gets its room.
  on_disk = true;
  assert(cache.MakeRoom(64, memory, only_b, report) && !cache.Find(b, Policy(), 19, 19, 8));
  assert(cache.Find(a, Policy(), 19, 19, 8));
}

// Fix 2 (cause.md item 9): a victim of a read's eviction is in host memory already, so the writer queue limit does not
// refuse it ("busy"); a victim of a save still meets the limit. LastSpill names the refusal.
void ReadVictimSkipsQueueLimit() {
  auto options = Options();
  options.root = Home() / "states-readvictim";
  options.queue_bytes = 1;
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<int> waiting = 0;
  options.before_write = [&] { ++waiting; gate.wait(); };
  Tier tier(options, false);
  const auto a = Tokens(3301, 40), b = Tokens(3401, 40), c = Tokens(3501, 40);
  const auto as = Make(std::span<const int>(a).first(20), true, 0);
  assert(tier.Disk().Spill(as->hist, Policy(), false, false, false, as, *as).empty());
  while (waiting == 0) std::this_thread::yield();
  const auto bs = Make(std::span<const int>(b).first(20), true, 0), cs = Make(std::span<const int>(c).first(20), true, 0);
  assert(tier.Disk().Spill(bs->hist, Policy(), false, false, false, bs, *bs).empty());  // the queue is full now
  const auto entry = std::make_shared<const Cache::Entry>(Cache::Entry{{}, cs, CacheRetention::Session});
  assert(!tier.Spill(cs->hist, Policy(), entry, UINT64_MAX) && tier.LastSpill() == "busy");
  assert(tier.Spill(cs->hist, Policy(), entry, UINT64_MAX, true) && tier.LastSpill().empty());
  assert(tier.Holds(cs->hist, Policy()));
  // The room (memory) still refuses a read's victim.
  const auto ds = Make(Tokens(3601, 20), true, 0);
  const auto d_entry = std::make_shared<const Cache::Entry>(Cache::Entry{{}, ds, CacheRetention::Session});
  assert(!tier.Spill(ds->hist, Policy(), d_entry, 1, true) && tier.LastSpill() == "memory");
  release.set_value();
  assert(tier.Drain(30) && tier.Disk().Present(cs->hist, Policy(), false));
}

// Fix 2 (cause.md item 6; GPU test 217): while an open holds its prompt, the entries that its restore may use never
// leave for another save (the handover copy of another session's Frontier point, or the release of its slot's tail).
void HoldKeepsPromptEntries() {
  const auto a = Tokens(2701, 20), b = Tokens(2801, 20), c = Tokens(2901, 20);
  const auto run = [&](bool hold) {
    Cache cache(64 * 2, {}, 1, 0, false);  // one live session: b is kept, a waits (tier kWaiting)
    assert(Capture(cache, a, 16) && Capture(cache, b, 16));
    if (hold) cache.Hold(a);
    const bool saved = Capture(cache, c, 16);
    cache.Hold({});
    return std::make_pair(saved, bool(cache.Find(a, Policy(), 19, 19, 8)));
  };
  assert(run(false) == std::make_pair(true, false));  // without the hold, the save takes a's entry
  assert(run(true) == std::make_pair(false, true));   // with the hold, the save is refused and a's entry stays
}

// The side effect of the hold (review question): while an open holds its prompt, the release capture of the slot's old
// tail (another session) cannot take the held entry. The capture is refused, and the refused tail takes the direct
// spill path (batched_serve.h Release -> Direct -> SpillDirect) with the bytes that Save wrote.
void HoldSendsReleasedTailToDisk() {
  auto options = Options();
  options.root = Home() / "states-hold";
  Tier tier(options, false);
  const auto a = Tokens(3701, 40), b = Tokens(3801, 40);
  Cache cache(64 * 1, {}, 1, 0, false);  // room for one entry: a's entry, which the open of a's turn holds
  assert(Capture(cache, a, 16));
  cache.Hold(a);
  assert(!Tail(cache, b, 30));
  const std::string refused = cache.Status().last.refused;
  assert(refused == "protected" || refused == "host");
  cache.Hold({});
  assert(cache.Find(a, Policy(), 19, 19, 8));
  const std::span<const int> tail(b.data(), 30);
  std::shared_ptr<const Snapshot> saved;
  const auto save = [&] { return saved = Make(tail, true, 0); };
  assert(tier.SpillDirect(tail, Policy(), false, refused, 3030, 1000, UINT64_MAX, save) == "queued");
  assert(tier.Drain(30) && tier.Disk().Present(tail, Policy(), false));
  const auto status = StageWait(tier, Query(b), 0);
  assert(status.state == StageState::Ready && status.tokens == 30);
  const auto ready = tier.Take(Query(b), 0);
  assert(ready.snapshot && ready.snapshot->bytes == saved->bytes);
  tier.Consume(ready);
}

// Fix 3: an open on a slot sends the slot's old session (2) to disk in two direct
// spills: its grid frontier (16 tokens; 49,152 in G) and then its turn-1 tail (30 tokens, off the 8-row grid; 50,341
// in G). The writer writes session 1's tail, and session 1's frontier waits, so the frontier takes the last queue
// room and the tail was refused "busy" (state_disk direct ... result=refused reason=busy). Session 2's turn 2 then
// restored only the frontier and prefilled the rest (REFERENCE-DIFFERS). The newer state of a branch takes the queue
// place of its own older prefix that waits for the writer: the tail is queued, the frontier leaves the queue
// ("state_disk queue_drop ... reason=superseded"), and turn 2 restores the tail with the bytes that Save wrote.
void TailTakesPrefixQueuePlace() {
  auto options = Options();
  options.root = Home() / "states-f3";
  std::vector<std::string> lines;
  std::mutex lines_mutex;
  options.log = [&](const std::string& line) { std::lock_guard lock(lines_mutex); lines.push_back(line); };
  // Two states wait at most beside the one that the writer writes (G: 4 GiB, states of about 1.75 GB). Make gives a
  // state of 3000 + rows bytes.
  options.queue_bytes = 3016 + 3030 + 24;
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<int> waiting = 0;
  options.before_write = [&] { ++waiting; gate.wait(); };
  Tier tier(options, false);
  const auto s1 = Tokens(4001, 40), s2 = Tokens(4101, 40);
  const std::span<const int> s1_tail(s1.data(), 31), s1_front(s1.data(), 16);
  const std::span<const int> s2_tail(s2.data(), 30), s2_front(s2.data(), 16);
  // The writer writes session 1's tail; session 1's frontier waits.
  const auto d = Make(s1_tail, true, 0), c = Make(s1_front, false, s1.size() - 1);
  assert(tier.Disk().Spill(d->hist, Policy(), false, false, false, d, *d).empty());
  while (waiting == 0) std::this_thread::yield();
  assert(tier.Disk().Spill(c->hist, Policy(), false, false, false, c, *c).empty());
  // Session 2's frontier goes straight to disk and waits too.
  std::weak_ptr<const Snapshot> front;
  const auto save_front = [&] {
    auto s = Make(s2_front, false, s2.size() - 1);
    front = s;
    return s;
  };
  assert(tier.SpillDirect(s2_front, Policy(), false, "protected", 3016, 3016, UINT64_MAX, save_front) == "queued");
  assert(tier.Holds(s2_front, Policy()));
  // Review rev-f3 item 3: turn 2 polls while the frontier waits ("writing" 16), and so does a prompt that repeats the
  // frontier but not the tail (a new answer to the same last message).
  const auto turn2 = Tokens(4101, 40);
  auto regen = Tokens(4101, 20);
  regen.resize(40, 999);
  auto status = tier.Stage(Query(turn2), 0, UINT64_MAX);
  assert(status.state == StageState::Reading && status.reason == "writing" && status.tokens == 16);
  status = tier.Stage(Query(regen), 0, UINT64_MAX);
  assert(status.state == StageState::Reading && status.reason == "writing" && status.tokens == 16);
  // Session 2's tail: on 170d4ef5 "busy" (the device copy did not run and the state was lost).
  std::shared_ptr<const Snapshot> saved;
  const auto save_tail = [&] { return saved = Make(s2_tail, true, 0); };
  assert(tier.SpillDirect(s2_tail, Policy(), false, "protected", 3030, 1000, UINT64_MAX, save_tail) == "queued");
  // The frontier left the queue and its bytes are free; session 1's frontier (another branch) still waits.
  assert(front.expired() && !tier.Holds(s2_front, Policy()) && tier.Holds(s2_tail, Policy()));
  assert(tier.Holds(c->hist, Policy()) && tier.Disk().QueuedBytes() == d->bytes.size() + 3016 + 3030);
  {
    std::lock_guard lock(lines_mutex);
    assert(std::count(lines.begin(), lines.end(),
                      "state_disk queue_drop tokens=16 bytes=3016 reason=superseded by=30") == 1);
  }
  // Session 2's turn 2 waits for the tail ("writing"), not for the frontier, and restores it.
  status = tier.Stage(Query(turn2), 0, UINT64_MAX);
  assert(status.state == StageState::Reading && status.reason == "writing" && status.tokens == 30);
  // The repeated prompt does not fit the tail, and its frontier left: the next poll is final (no wait, no hang).
  status = tier.Stage(Query(regen), 0, UINT64_MAX);
  assert(status.state == StageState::None && status.reason == "nostate" && status.tokens == 0);
  release.set_value();
  status = StageWait(tier, Query(turn2), 0);
  assert(status.state == StageState::Ready && status.tokens == 30);
  const auto ready = tier.Take(Query(turn2), 0);
  assert(ready.snapshot && ready.meta.key.size() == 30 && !ready.meta.resume.prompt_only);
  assert(ready.snapshot->hist == saved->hist && ready.snapshot->bytes == saved->bytes);  // Save's bytes
  tier.Consume(ready);
  assert(tier.Drain(30) && tier.Disk().QueuedBytes() == 0);
  status = StageWait(tier, Query(regen), 0);
  assert(status.state == StageState::None && status.reason == "nostate");
  assert(tier.Disk().Present(d->hist, Policy(), false) && tier.Disk().Present(c->hist, Policy(), false));
  assert(tier.Disk().Present(s2_tail, Policy(), false) && !tier.Disk().Present(s2_front, Policy(), false));
}

// Fix 3, the limits of the queue place rule: a prefix leaves only when that makes the room, never for nothing; a
// shared state, another branch and the state that the writer writes never leave; the default (no limit hit) drops
// nothing.
void PrefixStaysWithoutRoom() {
  auto options = Options();
  options.root = Home() / "states-f3b";
  options.queue_bytes = 3016 * 3 + 2;  // three 16-row states wait; a 30-row state replacing one of them does not fit
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<int> waiting = 0;
  options.before_write = [&] { ++waiting; gate.wait(); };
  Tier tier(options, false);
  const auto a = Tokens(4201, 40), b = Tokens(4301, 40), s = Tokens(4401, 40), t = Tokens(4501, 40);
  const auto first = [](const std::vector<int>& ids, size_t n) { return std::span<const int>(ids).first(n); };
  // The writer writes s's 12-row state (a prefix of s's tail; it stays in the writer's hands).
  const auto s12 = Make(first(s, 12), false, s.size() - 1);
  assert(tier.Disk().Spill(s12->hist, Policy(), false, false, false, s12, *s12).empty());
  while (waiting == 0) std::this_thread::yield();
  const auto as = Make(first(a, 16), false, a.size() - 1), bs = Make(first(b, 16), false, b.size() - 1);
  const auto s16 = Make(first(s, 16), false, s.size() - 1);
  assert(tier.Disk().Spill(as->hist, Policy(), false, false, false, as, *as).empty());
  assert(tier.Disk().Spill(bs->hist, Policy(), false, false, false, bs, *bs).empty());
  assert(tier.Disk().Spill(s16->hist, Policy(), false, false, false, s16, *s16).empty());
  int saves = 0;
  const auto save = [&] { ++saves; return Make(first(s, 30), true, 0); };
  // Dropping s's 16-row prefix would leave 3016 * 2 + 3030 bytes waiting: more than the limit. Nothing leaves.
  assert(tier.SpillDirect(first(s, 30), Policy(), false, "protected", 3030, 3030, UINT64_MAX, save) == "busy");
  assert(saves == 0 && tier.Holds(s16->hist, Policy()) && tier.Holds(s12->hist, Policy()));
  assert(tier.Disk().QueuedBytes() == 3012 + 3016 * 3);
  // A shared prefix never leaves (other sessions restore it); a state of another branch is not a prefix.
  auto shared_options = options;
  shared_options.root = Home() / "states-f3c";
  shared_options.queue_bytes = 3016 + 2;
  shared_options.before_write = [&] { ++waiting; gate.wait(); };
  waiting = 0;
  Tier other(shared_options, false);
  const auto t12 = Make(first(t, 12), false, t.size() - 1), t16 = Make(first(t, 16), false, t.size() - 1);
  assert(other.Disk().Spill(t12->hist, Policy(), false, false, false, t12, *t12).empty());
  while (waiting == 0) std::this_thread::yield();
  assert(other.Disk().Spill(t16->hist, Policy(), false, true, false, t16, *t16).empty());  // shared
  const auto t30 = Make(first(t, 30), true, 0);
  assert(other.Disk().Spill(t30->hist, Policy(), false, false, false, t30, *t30) == "busy");
  assert(other.Holds(t16->hist, Policy()));
  release.set_value();
  assert(tier.Drain(30) && other.Drain(30));
  assert(tier.Disk().QueuedBytes() == 0 && other.Disk().QueuedBytes() == 0);
}

int main() {
  const auto x = Tokens(1, 40), y = Tokens(101, 40), z = Tokens(201, 40), w = Tokens(301, 40);
  std::shared_ptr<const Snapshot> x_tail;
  {
    Tier tier(Options(), false);
    assert(tier.Enabled());
    Cache cache(64 * 3, {}, 2);
    cache.SetSpill([&](std::span<const int> key, const ConversationPolicy& policy,
                       const std::shared_ptr<const Cache::Entry>& entry, int, uint64_t room) {
      return tier.Spill(key, policy, entry, room);
    });
    // x's tail is confirmed by a hit (its 16-token frontier becomes superseded), then x idles while y and z run.
    assert(Capture(cache, x, 16) && Tail(cache, x, 21));
    const auto hit = cache.FindTail(Tokens(1, 30), Policy());
    assert(hit && hit->snapshot->hist.size() == 21);
    x_tail = hit->snapshot;
    assert(Capture(cache, y, 16) && Capture(cache, z, 16));  // the superseded frontier leaves, not to disk
    assert(tier.Drain(30) && tier.Disk().Entries() == 0);
    for (uint64_t i = 0; i < Cache::kIdleRounds * 2; ++i)
      assert(cache.Find(y, Policy(), 39, 39, 8) && cache.Find(z, Policy(), 39, 39, 8));
    assert(Capture(cache, w, 16));  // the idle tail leaves to disk
    assert(tier.Drain(30) && tier.Disk().Entries() == 1);
    // A stopping engine writes the host entries that are not superseded.
    const size_t before = tier.Disk().Entries();
    assert(tier.FlushHost(cache) >= 1 && tier.Drain(30) && tier.Disk().Entries() > before);
  }
  {
    // A stop flush skips superseded entries: v's frontier, superseded by its tail.
    auto options = Options();
    options.root = Home() / "states-flush";  // another identity in the same root would remove x's directory
    Tier tier(std::move(options), false);
    Cache cache(64 * 4, {}, 2);
    const auto v = Tokens(401, 40);
    assert(Capture(cache, v, 16) && Tail(cache, v, 21) && cache.FindTail(Tokens(401, 30), Policy()));
    assert(tier.FlushHost(cache) == 1 && tier.Drain(30) && tier.Disk().Entries() == 1);
    assert(!tier.Disk().Present(std::vector<int>(v.begin(), v.begin() + 16), Policy(), false));
    assert(tier.Disk().Present(std::vector<int>(v.begin(), v.begin() + 21), Policy(), false));
  }
  Tier tier(Options(), false);
  const auto prompt = Tokens(1, 30);  // x's tail (21 tokens) and 9 more
  // A floor at the state's length (a slot holds it) starts no read.
  assert(tier.Stage(Query(prompt), 21, UINT64_MAX).state == StageState::None);
  // Without session replay a tail is not restorable, and x's superseded frontier never went to disk.
  assert(tier.Stage(Query(prompt, false), 0, UINT64_MAX).state == StageState::None);
  const auto status = StageWait(tier, Query(prompt), 0);
  assert(status.state == StageState::Ready && status.tokens == 21);
  assert(!tier.Take(Query(prompt), 21).snapshot);  // the host cache holds as much: no disk restore
  const auto ready = tier.Take(Query(prompt), 16);
  assert(ready.snapshot && ready.meta.key.size() == 21 && !ready.meta.resume.prompt_only);
  assert(ready.snapshot->hist == x_tail->hist && ready.snapshot->bytes == x_tail->bytes);
  assert(ready.snapshot->resume.length == 21 && ready.snapshot->resume.next_id == x_tail->resume.next_id);
  tier.Consume(ready);
  assert(tier.Peek(Query(prompt)) == 0);
  // A prompt frontier fits only the host rules: y's 16-token frontier, but not past the request's limit.
  const auto y_prompt = Tokens(101, 30);
  assert(StageWait(tier, Query(y_prompt, false), 0).tokens == 16);
  auto short_limit = Query(y_prompt, false);
  short_limit.limit = 15;
  assert(tier.Stage(short_limit, 0, UINT64_MAX).state == StageState::None && tier.Peek(short_limit) == 0);
  assert(tier.Peek(Query(y_prompt)) == 16);
  // A prompt that leaves the state's tokens finds nothing.
  auto other = prompt;
  other[5] = 999;
  assert(tier.Stage(Query(other), 0, UINT64_MAX).state == StageState::None);
  // A stop flush skips the device copy of a slot state whose file is on disk, or that the daily budget refuses.
  assert(tier.Skip(x_tail->hist, Policy(), 3021) == "present");
  assert(tier.Skip(Tokens(501, 30), Policy(), 3030).empty());
  {
    auto small = Options();
    small.write_per_day = 1000;
    Tier limited(std::move(small), false);
    assert(limited.Skip(Tokens(501, 30), Policy(), 3030) == "daily");
  }
  SpillRoomLeavesUnallocatedReads();
  DirectSpill();
  InRunFrontier();
  StageRoomRule();
  MakeRoomForRead();
  MakeRoomKeepsFree();
  ReadQueueRule();
  StageWaitsForWriter();
  WriteDelay();
  ReadAheadKeptOnRefusal();
  ReadVictimSkipsQueueLimit();
  HoldKeepsPromptEntries();
  HoldSendsReleasedTailToDisk();
  TailTakesPrefixQueuePlace();
  PrefixStaysWithoutRoom();
  fs::remove_all(Home());
  {
    // the disk identity names the MTP slot rows, so a state saved with another MTP bound
    // is in another directory and never restores into a smaller slot.
    char* none[] = {nullptr};
    const auto a = BatchStateDiskSettings("/m/x/../pack-a", 4096, false, 33280, none);
    const auto b = BatchStateDiskSettings("/m/pack-a", 4096, false, 106496, none);
    assert(std::find(a.begin(), a.end(), "mtp_rows=33280") != a.end());
    assert(std::find(a.begin(), a.end(), "pack=pack-a") != a.end());
    assert(std::find(b.begin(), b.end(), "mtp_rows=106496") != b.end());
    // the format version and the KV storage code are in the identity, so a state of another format or KV code
    // is in another directory.
    assert(std::find(a.begin(), a.end(), "format=2") != a.end());
    assert(std::find(a.begin(), a.end(), "kv_code=" + StateKvCode()) != a.end());
    assert(StateDiskIdentityFor("e", "c", a).directory != StateDiskIdentityFor("e", "c", b).directory);
    assert(StateDiskIdentityFor("e", "c", a).directory ==
           StateDiskIdentityFor("e", "c", BatchStateDiskSettings("/m/x/../pack-a", 4096, false, 33280, none)).directory);
  }
  std::puts("batch_state_disk_test passed");
}
