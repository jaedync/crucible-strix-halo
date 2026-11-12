// The miss ledger on the real BatchConversationCache and the disk tier: a state leaves by each cause
// (handover, evict, idle, budget, spill, stop, switch), and the next open of its prompt reports that lost_cause with
// best_tier and best_shared_tokens. A record that shares less than half of the prompt gives `new`. A refused stage
// reports the length of the refused disk state. The flows copy batched_serve.h (Release, Handover, OpenDisk,
// FlushToDisk); each step names its line there. Synthetic token ids only.
//
// MISS_LEDGER_JSON=PATH writes one record for each cause case.
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
#include "../engine/serve/miss_ledger.h"
#include "support/disk_test_dir.h"
#include <chrono>
#include <fstream>
#include <future>
#include <string>
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
constexpr size_t kState = 2048;     // the tokens of each saved state (8 lost-record grid points)
constexpr size_t kBytes = 100;      // the budget bytes of each entry
constexpr size_t kTwo = 2 * kBytes + kBytes / 2;  // a host cache that holds two entries

fs::path Home() {
  static const fs::path home = [] {
    const DiskTestDir dir = PickDiskTestDir("miss_ledger_test");
    ::setenv("HOME", dir.path.c_str(), 1);
    return dir.path;
  }();
  return home;
}
StateDiskOptions Options(const std::string& root, char identity = '7') {
  StateDiskOptions options;
  options.root = Home() / root;
  options.identity = std::string(64, identity);
  options.directory = std::string(16, identity) + "-" + std::string(16, identity) + "-" + std::string(16, identity);
  options.min_tokens = 12;
  options.log = [](const std::string&) {};
  options.free_space = [] { return uint64_t(1) << 40; };
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
std::vector<int> Join(std::vector<int> a, const std::vector<int>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
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
// A prompt frontier of `position` tokens of `ids` (batched_serve.h Capture).
bool Save(Cache& cache, const std::vector<int>& ids, size_t position = kState, size_t bytes = kBytes) {
  return cache.Capture(ids, Policy(), {position, CacheRetention::Session}, bytes, uint64_t(64) << 30,
                       [&] { return Make(std::span<const int>(ids).first(position), false, ids.size() - 1); });
}
// A conversation: kState + 52 tokens of its own ids, so its frontier at kState is a resume point.
std::vector<int> Conversation(int n) { return Tokens(n * 1000000, kState + 52); }
// The next prompt of that conversation: its frontier, then new text.
std::vector<int> Next(int n, size_t more = 500) {
  auto ids = Conversation(n);
  ids.resize(kState);
  return Join(std::move(ids), Tokens(n * 1000000 + 700000, more));
}
DiskQuery Query(const std::vector<int>& prompt) { return {prompt, Policy(), prompt.size() - 1, prompt.size() - 1, 8, true}; }
// The open's ledger with no slot state (batched_serve.h Lookup, then BatchSession::Open's verdict): a miss.
BatchLedger Open(Cache& cache, const std::vector<int>& prompt, Tier* tier = nullptr) {
  const LedgerTiers tiers = tier ? tier->Ledger(cache, Query(prompt)) : cache.LedgerOf(prompt, Policy());
  return LedgerVerdict(tiers, prompt.size(), 0);
}
// The disk spill hook of batched_serve.h OpenDisk (`room`: what the test lets the writer hold).
void Wire(Cache& cache, Tier& tier, uint64_t room = UINT64_MAX) {
  cache.SetSpill([&cache, &tier, room](std::span<const int> key, const ConversationPolicy& policy,
                                      const std::shared_ptr<const Cache::Entry>& entry, int, uint64_t) {
    return tier.Spill(key, policy, entry, room, cache.Reading());
  });
  tier.Attach(cache);
}

struct Case { std::string name, want; BatchLedger got; };
std::vector<Case> cases;
void Expect(const std::string& name, const std::string& want, const BatchLedger& got, const char* tier, size_t shared) {
  cases.push_back({name, want, got});
  std::printf("case=%s lost_cause=%s best_tier=%s best_shared=%zu disk_shared=%zu\n", name.c_str(), got.lost_cause,
              got.best_tier, got.best_shared, got.disk_shared);
  assert(want == got.lost_cause);
  assert(std::string(tier) == got.best_tier);
  assert(got.best_shared == shared);
}

// A host entry leaves the host cache with no disk tier: `evict`. Its record shares the whole state.
void Evict() {
  Cache cache(kTwo, {}, 1, 1000, false);
  assert(Save(cache, Conversation(1)) && Save(cache, Conversation(2)) && Save(cache, Conversation(3)));
  Expect("evict", "evict", Open(cache, Next(1)), "lost", kState);
}
// Defect 2: the record shares less than half of the prompt (only the block): `new`, not the eviction.
void HalfRule() {
  Cache cache(kTwo, {}, 1, 1000, false);
  assert(Save(cache, Conversation(1)) && Save(cache, Conversation(2)) && Save(cache, Conversation(3)));
  const auto prompt = Next(1, 3 * kState);
  const BatchLedger got = Open(cache, prompt);
  assert(std::string(got.lost_cause) == "new" && std::string(got.best_tier) == "lost" && got.best_shared == kState);
  // The old miss word is kept as it was (add fields only, rename none).
  assert(std::string(cache.Diagnose(prompt, Policy(), prompt.size() - 1, prompt.size() - 1, 8, true).miss) == "evicted");
}
// The resume point of a session that no request used for the idle time leaves first: `idle`.
void Idle() {
  double now = 0;
  Cache cache(kTwo, {}, 1, 1000, false, 100);
  cache.SetClock([&] { return now; });
  assert(Save(cache, Conversation(1)));
  now = 500;
  assert(Save(cache, Conversation(2)) && Save(cache, Conversation(3)));
  Expect("idle", "idle", Open(cache, Next(1)), "lost", kState);
}
// The daily write budget refuses the spill of the evicted state: `budget`.
void Budget() {
  auto options = Options("states-budget");
  options.write_per_day = 1;
  Tier tier(options, false);
  Cache cache(kTwo, {}, 1, 1000, false);
  Wire(cache, tier);
  assert(Save(cache, Conversation(1)) && Save(cache, Conversation(2)) && Save(cache, Conversation(3)));
  assert(tier.Drain(30));
  Expect("budget", "budget", Open(cache, Next(1), &tier), "lost", kState);
}
// The writer cannot hold the evicted state (no host room): `spill`.
void Spill() {
  Tier tier(Options("states-spill"), false);
  Cache cache(kTwo, {}, 1, 1000, false);
  Wire(cache, tier, 0);
  assert(Save(cache, Conversation(1)) && Save(cache, Conversation(2)) && Save(cache, Conversation(3)));
  assert(tier.Drain(30));
  Expect("spill", "spill", Open(cache, Next(1), &tier), "lost", kState);
}
// Defect 1: an evicted state whose write ends is on disk: the best tier is the disk, and the record is no loss.
void Durable() {
  auto options = Options("states-durable");
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  options.before_write = [&] { gate.wait(); };
  Tier tier(options, false);
  Cache cache(kTwo, {}, 1, 1000, false);
  Wire(cache, tier);
  assert(Save(cache, Conversation(1)) && Save(cache, Conversation(2)) && Save(cache, Conversation(3)));
  // The write waits: the state is queued, so the disk tier holds it (Pending).
  LedgerTiers tiers = tier.Ledger(cache, Query(Next(1)));
  assert(tiers.disk.shared == kState && tiers.lost.shared == 0);
  release.set_value();
  assert(tier.Drain(30));
  tiers = tier.Ledger(cache, Query(Next(1)));
  assert(tiers.disk.shared == kState && tiers.lost.shared == 0);
  const BatchLedger got = LedgerVerdict(tiers, Next(1).size(), 0);
  assert(std::string(got.best_tier) == "disk" && got.disk_shared == kState);
  // The open that reads it from disk uses the best state.
  const BatchLedger hit = LedgerVerdict(tiers, Next(1).size(), kState, "disk");
  assert(std::string(hit.lost_cause) == "none" && hit.resume_gap == 0 && std::string(hit.resume_tier) == "disk");
}
// A slot's tail leaves its slot at another session's open and the host cache refuses its copy (batched_serve.h
// Release: CaptureTail, then NoteRefused; no disk tier, so no direct spill): `handover`. Today it reports not_saved.
void Handover() {
  Cache cache(kTwo, {}, 1, 1000, false);
  assert(Save(cache, Conversation(1)));
  const auto tail = Join(Conversation(2), Tokens(2000000 + 900000, 100));
  const bool saved = cache.CaptureTail(tail, Policy(), 2 * kBytes, uint64_t(64) << 30,
                                       [&] { return Make(tail, true, 0); }, true);
  assert(!saved && std::string(cache.LastRefusal()) == "protected");
  cache.NoteRefused(tail, Policy(), Conversation(2).size(), "handover");
  const auto prompt = Join(tail, Tokens(2000000 + 950000, 400));
  Expect("handover", "handover", Open(cache, prompt), "lost", tail.size());
  // The legacy word stays.
  assert(std::string(cache.Diagnose(prompt, Policy(), prompt.size() - 1, prompt.size() - 1, 8, true).miss) == "not_saved");
}
// The engine stops and its flush writes nothing (no time left, batched_serve.h FlushToDisk): the next start reads the
// ledger from the state root's side folder, and the open reports `stop`.
void Stop() {
  const auto options = Options("states-stop");
  {
    Tier tier(options, false);
    Cache cache(kTwo, {}, 1, 1000, false);
    Wire(cache, tier);
    assert(Save(cache, Conversation(1)));
    tier.NoteStop(cache);
    assert(tier.SaveLedger(cache));
  }
  Tier tier(options, false);
  Cache cache(kTwo, {}, 1, 1000, false);
  Wire(cache, tier);
  Expect("stop", "stop", Open(cache, Next(1), &tier), "lost", kState);
}
// The stop flush wrote the state; the next start has another state epoch, so the state is not readable: `switch`.
// With the same epoch the state is on disk (best tier disk, no loss).
void Switch() {
  const auto options = Options("states-switch");
  {
    Tier tier(options, false);
    Cache cache(kTwo, {}, 1, 1000, false);
    Wire(cache, tier);
    assert(Save(cache, Conversation(1)));
    assert(tier.FlushHost(cache) == 1 && tier.Drain(30));
    tier.NoteStop(cache);
    assert(tier.SaveLedger(cache));
  }
  {
    Tier same(options, false);
    Cache cache(kTwo, {}, 1, 1000, false);
    Wire(cache, same);
    const BatchLedger got = Open(cache, Next(1), &same);
    assert(std::string(got.best_tier) == "disk" && got.best_shared == kState && got.disk_shared == kState);
  }
  Tier other(Options("states-switch", '8'), false);
  Cache cache(kTwo, {}, 1, 1000, false);
  Wire(cache, other);
  Expect("switch", "switch", Open(cache, Next(1), &other), "lost", kState);
}
// The lineage counter: a later state of a conversation inherits it, another conversation gets a new one, and a state
// that a lost record holds half of inherits the record's number.
void Lineage() {
  Cache cache(kTwo, {}, 1, 1000, false);
  assert(Save(cache, Conversation(1), 1024));
  const uint64_t first = cache.LedgerOf(Next(1), Policy()).host.lineage;
  assert(first > 0);
  assert(Save(cache, Conversation(1)));
  assert(cache.LedgerOf(Next(1), Policy()).host.lineage == first);
  assert(Save(cache, Conversation(2)) && Save(cache, Conversation(3)) && Save(cache, Conversation(4)));
  const uint64_t two = cache.LedgerOf(Next(2), Policy()).lost.lineage;
  assert(two && two != first);
  // Conversation 1 left the host cache; its next turn's frontier inherits the record's number.
  const auto later = Join(Next(1, 600), Tokens(1000000 + 800000, 100));
  const LedgerTiers before = cache.LedgerOf(later, Policy());
  assert(before.lost.lineage == first);
  assert(Save(cache, later, kState + 512));
  assert(cache.LedgerOf(later, Policy()).host.lineage == first);
}
// The lost list holds 64 records and its file stays under 1 MiB.
void Bounds() {
  Cache cache(kTwo, {}, 1, 1000, false);
  for (int n = 1; n <= 70; ++n) assert(Save(cache, Conversation(n)));
  assert(cache.LostRecords() == 64);
  assert(cache.LedgerBytes().size() <= (size_t(1) << 20));
}
// A refused stage reports the length of the disk state that it refused, and what a read waits for.
void Stage() {
  Tier tier(Options("states-stage"), false);
  const auto ids = Conversation(1);
  const auto state = Make(std::span<const int>(ids).first(kState), false, ids.size() - 1);
  assert(tier.Disk().Spill(state->hist, Policy(), false, false, false, state, *state).empty() && tier.Drain(30));
  const auto refused = tier.StageWith(Query(Next(1)), 0, [](const StateMeta&) { return StageAdmit{0, "room"}; });
  assert(refused.state == StageState::None && refused.tokens == kState);
  assert(std::string(StageWaitCause(refused)) == "room");
  assert(std::string(StageWaitCause({StageState::Reading, kState, "start"})) == "reader");
  assert(std::string(StageWaitCause({StageState::Reading, kState, "writing"})) == "writer");
  assert(std::string(StageWaitCause({StageState::Reading, kState, "freeing"})) == "room");
  assert(std::string(StageWaitCause({StageState::None, 0, "nostate"})) == "none");
  assert(std::string(StageWaitCause({StageState::Ready, kState, "start"})) == "none");
}
// The verdict's rules on their own: closed words, the half rule on a live state, an undo, and the slot tiers first.
void Verdict() {
  LedgerTiers tiers;
  tiers.host = {kState, 3 * kState, nullptr, 5};
  // The prompt leaves a live state before its end: an undo.
  BatchLedger got = LedgerVerdict(tiers, kState + 100, 0);
  assert(std::string(got.lost_cause) == "undo" && std::string(got.best_tier) == "host" && got.lineage == 5);
  tiers.slot = {kState, kState, nullptr, 6};
  got = LedgerVerdict(tiers, kState + 100, 0);
  assert(std::string(got.best_tier) == "slot" && std::string(got.lost_cause) == "unknown");
  got = LedgerVerdict(tiers, kState + 100, kState, "slot", "tail");
  assert(std::string(got.lost_cause) == "none" && std::string(got.resume_point) == "tail");
  got = LedgerVerdict(LedgerTiers{}, 100, 0, "bogus", "bogus");
  assert(std::string(got.lost_cause) == "new" && std::string(got.resume_tier) == "none" &&
         std::string(got.resume_point) == "none" && std::string(got.best_tier) == "none");
  const std::string line = ReleaseLine("tail", 2, kState, "lost", "protected", 9, 0.5, 3);
  assert(line == "state_release kind=tail slot=2 tokens=2048 outcome=lost refusal=protected bytes=9 seconds=0.500000 "
                 "lineage=3\n");
  assert(ReleaseLine("tail", 2, 1, "gone", "a b", 0, 0, 0).find("outcome=- refusal=-") != std::string::npos);
}

void WriteJson() {
  const char* path = std::getenv("MISS_LEDGER_JSON");
  if (!path) return;
  std::ofstream out(path);
  out << "[";
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& c = cases[i];
    out << (i ? "," : "") << "\n {\"case\": \"" << c.name << "\", \"want\": \"" << c.want << "\", \"lost_cause\": \""
        << c.got.lost_cause << "\", \"best_tier\": \"" << c.got.best_tier << "\", \"best_shared_tokens\": "
        << c.got.best_shared << ", \"disk_shared_tokens\": " << c.got.disk_shared << "}";
  }
  out << "\n]\n";
}
}  // namespace

int main() {
  Verdict();
  Evict();
  HalfRule();
  Idle();
  Budget();
  Spill();
  Durable();
  Handover();
  Stop();
  Switch();
  Lineage();
  Bounds();
  Stage();
  WriteJson();
  std::printf("miss_ledger_test: %zu cause cases passed\n", cases.size());
  return 0;
}
