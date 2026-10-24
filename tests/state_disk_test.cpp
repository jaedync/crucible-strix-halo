// NVMe tier for conversation states (engine/runtime/state_disk.h): format, index, writer, reader, budgets, privacy.
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <mutex>
#include <regex>
#include <vector>

std::mutex pinned_mutex;
std::map<uintptr_t, size_t> pinned;
// Page-aligned, as hipHostMalloc gives: so the payload chunks can take direct I/O in place, as on the test machine.
template<class T> int hipHostMalloc(T** out, size_t n) {
  std::lock_guard lock(pinned_mutex);
  *out = static_cast<T*>(std::aligned_alloc(4096, (n + 4095) / 4096 * 4096));
  pinned[uintptr_t(*out)] = n;
  return 0;
}
int hipHostFree(void* p) {
  std::lock_guard lock(pinned_mutex);
  assert(pinned.erase(uintptr_t(p)) == 1);
  std::free(p);
  return 0;
}
bool IsPinned(const void* p) {
  std::lock_guard lock(pinned_mutex);
  const auto x = uintptr_t(p);
  auto it = pinned.upper_bound(x);
  return it != pinned.begin() && x < (--it)->first + it->second;
}
constexpr int hipMemcpyDeviceToHost = 0, hipMemcpyHostToDevice = 1;
int hipMemcpy(void* out, const void* in, size_t n, int) { std::memcpy(out, in, n); return 0; }
int hipStreamSynchronize(int) { return 0; }
#define HIP_OK(expr) assert((expr) == 0)
#include "../engine/runtime/state_disk.h"
#include "support/disk_test_dir.h"
#include <chrono>
#include <fstream>
#include <future>
#include <thread>
#include <fcntl.h>
#include <sys/stat.h>

using namespace crucible;
namespace fs = std::filesystem;

struct TestSnapshot {
  std::vector<int> hist;
  RetainedPrefix retained;
  ConversationResume resume;
  SnapshotBytes bytes;
  std::vector<std::pair<int, int>> qsa_layout;
  uint16_t kv_quant = 0;  // as a ConversationSnapshot with a KV quant field: stored when the type has it
};
using Disk = StateDisk<TestSnapshot>;
constexpr size_t kChunk = SnapshotChunkPool::kChunkBytes;

struct Logs {
  std::mutex mutex;
  std::vector<std::string> lines;
  std::function<void(const std::string&)> Sink() {
    return [this](const std::string& line) { std::lock_guard lock(mutex); lines.push_back(line); };
  }
  bool Has(const std::string& text) {
    std::lock_guard lock(mutex);
    for (const auto& line : lines) if (line.find(text) != std::string::npos) return true;
    return false;
  }
};

bool direct_root = false;
fs::path Root() {
  static const fs::path root = [] {
    const DiskTestDir dir = PickDiskTestDir("state_disk_test");
    direct_root = dir.direct;
    fs::create_directories(dir.path / "home");
    ::setenv("HOME", (dir.path / "home").c_str(), 1);
    return dir.path / "home";
  }();
  return root;
}

// An identity directory name for test identity `id` (three groups of 16 hex digits).
std::string DirName(char id) {
  const std::string group(16, "0123456789abcdef"[id % 16]);
  return group + "-" + group + "-" + group;
}

StateDiskOptions Options(const std::string& identity, Logs& logs, const std::string& name = "states") {
  StateDiskOptions options;
  options.root = Root() / ".cache" / name;
  options.identity = std::string(64, identity[0]);
  options.directory = DirName(identity[0]);
  options.min_tokens = 4;
  options.io_threads = 3;
  options.log = logs.Sink();
  return options;
}

ConversationPolicy Policy(int rows = 16) {
  ConversationPolicy policy;
  policy.prefill_rows = rows;
  return policy;
}

std::shared_ptr<TestSnapshot> Make(const std::vector<int>& hist, size_t bytes, uint8_t seed, bool tail = false) {
  auto snapshot = std::make_shared<TestSnapshot>();
  snapshot->hist = hist;
  RetainedPrefix::FromFields({hist.size(), hist.size() - 1, hist.size() - 2, 1}, snapshot->retained);
  snapshot->resume.length = hist.size();
  snapshot->resume.prompt_only = !tail;
  snapshot->resume.replay = tail ? 2 : 0;
  snapshot->resume.next_id = tail ? 77 : -1;
  snapshot->resume.prompt_boundary = 3;
  snapshot->resume.logits = {1.5f, -2.25f};
  snapshot->resume.pending_tokens = {5, 6};
  snapshot->qsa_layout = {{3, 0}, {7, 64}};
  snapshot->kv_quant = uint16_t(0x0108 + seed);
  snapshot->bytes = SnapshotBytes(bytes);
  snapshot->bytes.ForEachPiece(0, bytes, [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
    for (size_t i = 0; i < count; ++i) chunk.host[inside + i] = uint8_t((at + i) * 131 + seed);
  });
  return snapshot;
}

void Spill(Disk& disk, const std::shared_ptr<TestSnapshot>& snapshot, const ConversationPolicy& policy = Policy(),
           bool compact = false, std::string expect = "") {
  const auto result = disk.Spill(snapshot->hist, policy, compact, false, false, snapshot, *snapshot);
  assert(result == expect);
}

std::vector<int> Tokens(size_t n, int salt = 0) {
  std::vector<int> ids(n);
  for (size_t i = 0; i < n; ++i) ids[i] = int(i * 7 + salt) % 1000;
  return ids;
}

auto Any = [](const StateMeta&) { return true; };

// Stages `meta` and waits for the read.
StageStatus StageWait(Disk& disk, const StateMeta& meta) {
  for (int i = 0; i < 2000; ++i) {
    const auto status = disk.Stage(meta, UINT64_MAX);
    if (status.state != StageState::Reading) return status;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  assert(false);
  return {};
}

void Equal(const TestSnapshot& a, const TestSnapshot& b) {
  assert(a.hist == b.hist);
  assert(a.retained.Fields() == b.retained.Fields());
  assert(a.resume.length == b.resume.length && a.resume.mpos == b.resume.mpos && a.resume.replay == b.resume.replay);
  assert(a.resume.next_id == b.resume.next_id && a.resume.have_carry == b.resume.have_carry);
  assert(a.resume.pending_tokens == b.resume.pending_tokens && a.resume.logits == b.resume.logits);
  assert(a.resume.prompt_only == b.resume.prompt_only && a.resume.prompt_boundary == b.resume.prompt_boundary);
  assert(a.qsa_layout == b.qsa_layout && a.kv_quant == b.kv_quant);
  assert(a.bytes == b.bytes);
}

mode_t Mode(const fs::path& path) {
  struct stat info{};
  assert(::stat(path.c_str(), &info) == 0);
  return info.st_mode & 0777;
}

void RoundTripAndRebuild() {
  Logs logs;
  const auto snapshot = Make(Tokens(40), 2 * kChunk + 12345, 9);
  fs::path file;
  {
    Disk disk(Options("a", logs));
    assert(disk.Enabled());
    Spill(disk, snapshot);
    assert(disk.Drain(30));
    assert(disk.Entries() == 1);
    // A second spill of the same state writes nothing.
    Spill(disk, snapshot, Policy(), false, "present");
    const auto prompt = Tokens(50);
    const auto meta = disk.Best(prompt, Policy(), false, Any);
    assert(meta && meta->key == snapshot->hist && meta->resume.length == 40 && !meta->shared);
    file = disk.Directory() / meta->name;
    assert(Mode(disk.Directory()) == 0700 && Mode(file) == 0600);
    assert(fs::file_size(file) % 4096 == 0);
    // The padding after the payload is zero, never stale memory.
    std::ifstream in(file, std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(in)), {});
    const size_t payload_end = data.size() - (4096 - (2 * kChunk + 12345) % 4096);
    for (size_t i = payload_end; i < data.size(); ++i) assert(data[i] == 0);
    const auto status = StageWait(disk, *meta);
    assert(status.state == StageState::Ready && status.tokens == 40);
    const auto ready = disk.Ready(prompt, Policy(), false, Any);
    assert(ready.snapshot);
    Equal(*ready.snapshot, *snapshot);
    assert(disk.StagedBytes() > 0);
    disk.Consume(ready);
    assert(disk.StagedBytes() == 0 && !disk.Ready(prompt, Policy(), false, Any).snapshot);
  }
  // A new process with the same identity rebuilds the index from the file headers.
  Disk disk(Options("a", logs));
  assert(disk.Entries() == 1);
  const auto meta = disk.Best(Tokens(41), Policy(), false, Any);
  assert(meta && StageWait(disk, *meta).state == StageState::Ready);
  Equal(*disk.Ready(Tokens(41), Policy(), false, Any).snapshot, *snapshot);
  assert(logs.Has("state_disk start entries=1"));
}

// a composed snapshot (shared chunks of a held one, its own bytes packed) spills the same bytes as a flat one,
// and the read gives a flat snapshot with those bytes.
void ComposedSnapshotSpills() {
  Logs logs;
  const auto base = Make(Tokens(40), 3 * kChunk + 12345, 4);
  std::vector<uint8_t> device = base->bytes.Copy();
  device.back() ^= 0x5a;  // a byte past the shared chunks changes, as a later row of the slot would
  auto composed = std::make_shared<TestSnapshot>(*base);
  std::vector<SnapshotRange> copies;
  composed->bytes = SnapshotBytes::Compose({{device.data(), 100}, {device.data() + 100, device.size() - 100,
                                                                   &base->bytes, 100}}, copies);
  CopySnapshotRanges(copies, composed->bytes, true);
  assert(composed->bytes.HeldElsewhere() == 2 * kChunk && composed->bytes.Copy() == device);
  Disk disk(Options("composed", logs));
  Spill(disk, composed);
  assert(disk.Drain(30));
  const auto meta = disk.Best(Tokens(50), Policy(), false, Any);
  assert(meta && StageWait(disk, *meta).state == StageState::Ready);
  const auto ready = disk.Ready(Tokens(50), Policy(), false, Any);
  assert(ready.snapshot && ready.snapshot->bytes.Flat() && ready.snapshot->bytes.Copy() == device);
  disk.Consume(ready);
}

void KeyMatchesExactly() {
  Logs logs;
  Disk disk(Options("b", logs));
  const auto snapshot = Make(Tokens(20), 5000, 1, true);
  Spill(disk, snapshot);
  assert(disk.Drain(30));
  auto prompt = Tokens(30);
  assert(disk.Best(prompt, Policy(), false, Any));
  prompt[19] += 1;  // the last stored token differs
  assert(!disk.Best(prompt, Policy(), false, Any));
  assert(!disk.Best(Tokens(19), Policy(), false, Any));        // shorter than the state
  assert(!disk.Best(Tokens(30), Policy(32), false, Any));      // another policy
  assert(!disk.Best(Tokens(30), Policy(), true, Any));         // another storage mode
  assert(!disk.Best(Tokens(30), Policy(), false, [](const StateMeta& m) { return m.resume.prompt_only; }));
  // Too short a state is never written.
  Spill(disk, Make(Tokens(3), 100, 1), Policy(), false, "short");
}

void IdentitiesAndLocks() {
  Logs logs;
  fs::path old_dir;
  {
    Disk old(Options("c", logs, "ids"));
    Spill(old, Make(Tokens(10), 3000, 2));
    assert(old.Drain(30));
    old_dir = old.Directory();
    // A second engine of the same identity must not share the directory.
    Disk twin(Options("c", logs, "ids"));
    assert(!twin.Enabled() && logs.Has("state_disk off reason=locked"));
    // Another identity cannot remove a directory that a live engine holds.
    Disk other(Options("d", logs, "ids"));
    assert(other.Enabled() && fs::exists(old_dir));
  }
  // An unrelated file in the root: the root is not the tier's own folder, so the engine refuses it and removes
  // nothing. Without it, a stale temp file in the identity directory leaves.
  { std::ofstream(Root() / ".cache" / "ids" / "notes.txt") << "x"; }
  const fs::path own = Root() / ".cache" / "ids" / DirName('e');
  fs::create_directories(own);
  std::ofstream(own / "abc.state.tmp.1") << "partial";
  bool refused = false;
  try { Disk mixed(Options("e", logs, "ids")); } catch (const std::invalid_argument&) { refused = true; }
  assert(refused && fs::exists(old_dir) && fs::exists(own / "abc.state.tmp.1"));
  fs::remove(Root() / ".cache" / "ids" / "notes.txt");
  Disk next(Options("e", logs, "ids"));
  assert(next.Enabled() && next.Entries() == 0);
  assert(!fs::exists(old_dir));  // the old identity's directory leaves at start
  assert(!fs::exists(own / "abc.state.tmp.1"));
  // A state root outside the home directory is refused.
  auto outside = Options("f", logs);
  outside.root = "/var/tmp/crucible-states-test";
  bool threw = false;
  try { Disk bad(outside); } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
}

// A rollback from the block storage format: this binary removes the newer
// binary's directory with its block files, and a stray block file in its own directory leaves at start.
void OldBinaryRemovesBlocks() {
  Logs logs;
  const fs::path root = Root() / ".cache" / "rollback";
  const fs::path newer = root / DirName('i');
  const std::string digest(64, 'a');
  fs::create_directories(newer);
  for (const std::string name : {digest + ".block", digest + ".block.tmp.77", digest + ".block.dead.3",
                                 digest + ".state"})
    std::ofstream(newer / name) << "v2";
  {
    Disk disk(Options("h", logs, "rollback"));
    assert(disk.Enabled());
    assert(!fs::exists(newer));  // without the block names, the directory and its blocks stay
    Spill(disk, Make(Tokens(10), 3000, 4));
    assert(disk.Drain(30));
    std::ofstream(disk.Directory() / (digest + ".block")) << "stray";
    std::ofstream(disk.Directory() / (digest + ".block.tmp.9")) << "stray";
  }
  Disk again(Options("h", logs, "rollback"));
  assert(again.Enabled() && again.Entries() == 1);
  assert(!fs::exists(again.Directory() / (digest + ".block")));
  assert(!fs::exists(again.Directory() / (digest + ".block.tmp.9")));
  // Other names stay: a block name is a suffix, not a substring of any name.
  const fs::path other = root / DirName('j');
  fs::create_directories(other);
  std::ofstream(other / "notes.blocks") << "keep";
  Disk third(Options("k", logs, "rollback"));
  assert(fs::exists(other / "notes.blocks"));
}

void CorruptFilesFail() {
  Logs logs;
  Disk disk(Options("g", logs));
  const auto snapshot = Make(Tokens(12), kChunk + 777, 3);
  Spill(disk, snapshot);
  assert(disk.Drain(30));
  const auto meta = disk.Best(Tokens(12), Policy(), false, Any);
  assert(meta);
  {
    std::fstream file(disk.Directory() / meta->name, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(std::streamoff(fs::file_size(disk.Directory() / meta->name) - 5000));
    file.put('\x5a');
  }
  const auto status = StageWait(disk, *meta);
  assert(status.state == StageState::Failed && status.reason == "checksum");
  assert(disk.Entries() == 0 && !fs::exists(disk.Directory() / meta->name));
  assert(!disk.Ready(Tokens(12), Policy(), false, Any).snapshot);
  // A damaged header drops the file at the next start.
  Spill(disk, snapshot);
  assert(disk.Drain(30));
  const auto again = disk.Best(Tokens(12), Policy(), false, Any);
  const fs::path path = disk.Directory() / again->name;
  {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(100);
    file.put('\x01');
  }
}

void DamagedHeaderLeaves() {
  Logs logs;
  std::optional<Disk> disk(std::in_place, Options("g", logs));
  assert(disk->Entries() == 0);
  for (const auto& item : fs::directory_iterator(disk->Directory()))
    assert(!item.path().string().ends_with(".state"));
}

void BudgetAndFullDisk() {
  Logs logs;
  auto options = Options("h", logs);
  const size_t each = 3 * 4096 + 4096;  // a small header and three pages of payload
  options.budget = 2 * each + 9000;     // room for two files, not three
  uint64_t free_bytes = uint64_t(1) << 40;
  options.free_space = [&] { return free_bytes; };
  Disk disk(options);
  const auto unrelated = Make(Tokens(10, 500), 3 * 4096, 4);
  const auto older = Make(Tokens(10), 3 * 4096, 5);
  auto longer_ids = Tokens(10);
  longer_ids.push_back(999);
  const auto newer = Make(longer_ids, 3 * 4096, 6);
  Spill(disk, unrelated);
  assert(disk.Drain(30));
  Spill(disk, older);
  assert(disk.Drain(30));
  Spill(disk, newer);
  assert(disk.Drain(30));
  // The state that a longer state of its branch extends leaves first, before the older unrelated state.
  assert(disk.Entries() == 2);
  assert(disk.Best(Tokens(20, 500), Policy(), false, Any));
  assert(disk.Best(longer_ids, Policy(), false, Any)->key == longer_ids);
  assert(logs.Has("state_disk evict"));
  // A write that the free space refuses evicts nothing, even when the evictions would not free enough.
  free_bytes = 4096;
  const auto evicted = disk.Stats().evicted;
  Spill(disk, Make(Tokens(30, 500), 3 * 4096, 8));
  assert(disk.Drain(30) && disk.Entries() == 2 && disk.Stats().evicted == evicted);
  // A disk without the reserve refuses the spill and the server goes on.
  auto full = Options("i", logs);
  full.free_space = [] { return uint64_t(1) << 20; };
  Disk small(full);
  const auto snapshot = Make(Tokens(10), 4096, 7);
  std::weak_ptr<TestSnapshot> weak = snapshot;
  Spill(small, snapshot);
  assert(small.Drain(30));
  assert(small.Entries() == 0 && small.Stats().refused == 1 && logs.Has("result=refused reason=full"));
  for (const auto& item : fs::directory_iterator(small.Directory()))
    assert(item.path().filename() == ".lock");
}

void QueueLimitAndMemory() {
  Logs logs;
  auto options = Options("j", logs);
  options.queue_bytes = 10000;
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<int> waiting = 0;
  options.before_write = [&] { ++waiting; gate.wait(); };
  Disk disk(options);
  auto first = Make(Tokens(10), 6000, 1);
  std::weak_ptr<TestSnapshot> weak = first;
  Spill(disk, first);
  while (waiting == 0) std::this_thread::yield();
  // The writer holds the first state. The limit counts only the states that wait, so one more state waits beside
  // it (a full-size tail must not find the queue full while the writer writes the last one); a third one would pass
  // the limit.
  assert(disk.QueuedBytes() == 6000 && !disk.QueueFull(6000));
  // The salts give each state its own branch (a waiting prefix of the new state gives its place, below).
  auto second = Make(Tokens(11, 1), 6000, 2);
  std::weak_ptr<TestSnapshot> second_weak = second;
  Spill(disk, second);
  second.reset();
  assert(disk.QueueFull(6000) && !disk.QueueFull(4000));
  Spill(disk, Make(Tokens(12, 2), 6000, 3), Policy(), false, "busy");
  // Fix 3: a newer state of the waiting state's own branch takes its place. The
  // waiting state leaves the queue and host memory; the state in the writer stays.
  assert(disk.QueueFull(6000, Tokens(12, 2), Policy(), false) && !disk.QueueFull(6000, Tokens(13, 1), Policy(), false));
  assert(disk.QueueFull(6000, Tokens(13, 0), Policy(), false));  // it extends only the state in the writer
  Spill(disk, Make(Tokens(13, 1), 6000, 4));
  assert(second_weak.expired() && disk.QueuedBytes() == 12000);
  assert(logs.Has("state_disk queue_drop tokens=11 bytes=6000 reason=superseded by=13"));
  assert(!disk.Waiting(Tokens(11, 1), Policy(), false) && disk.Waiting(Tokens(13, 1), Policy(), false));
  first.reset();
  assert(!weak.expired());  // the queue keeps the evicted bytes until the write completes
  release.set_value();
  assert(disk.Drain(30));
  assert(weak.expired() && disk.Entries() == 2 && disk.QueuedBytes() == 0);
  assert(disk.Present(Tokens(13, 1), Policy(), false) && !disk.Present(Tokens(11, 1), Policy(), false));
}

// the writer holds an evicted state in host memory until the write ends. The cache passes the room that its
// save leaves above the host reserve; a state that does not fit is refused ("memory") and leaves as with no disk tier.
void HostRoom() {
  Logs logs;
  Disk disk(Options("v", logs, "host-room"));
  const auto state = Make(Tokens(12), 6000, 3);
  assert(disk.Spill(state->hist, Policy(), false, false, false, state, *state, false, 5999) == "memory");
  assert(logs.Has("bytes=6000 result=refused reason=memory") && disk.Entries() == 0);
  assert(disk.Spill(state->hist, Policy(), false, false, false, state, *state, false, 6000).empty());
  assert(disk.Drain(30) && disk.Entries() == 1);
  // A state on disk already holds no memory: "present" comes first.
  assert(disk.Spill(state->hist, Policy(), false, false, false, state, *state, false, 0) == "present");
  // A stopping engine's flush ignores the room: its states are in host memory anyway.
  const auto other = Make(Tokens(13), 6000, 4);
  assert(disk.Spill(other->hist, Policy(), false, false, false, other, *other, true, 0).empty());
  assert(disk.Drain(30) && disk.Entries() == 2);
}

void LogsHoldNoTokens() {
  Logs logs;
  {
    Disk disk(Options("k", logs));
    Spill(disk, Make(Tokens(10, 123), 4096, 1));
    assert(disk.Drain(30));
    const auto meta = disk.Best(Tokens(10, 123), Policy(), false, Any);
    StageWait(disk, *meta);
  }
  const std::regex safe(R"(state_disk [a-z]+( [a-z_]+=[a-z0-9_.\-]+)*)");
  std::lock_guard lock(logs.mutex);
  assert(!logs.lines.empty());
  for (const auto& line : logs.lines) assert(std::regex_match(line, safe));
}

void IdentityText() {
  // The directory and key change with the binary, the checkpoint metadata and the engine settings.
  const std::string b1(64, '1'), b2(64, '2'), c1(64, '3');
  const auto a = StateDiskIdentityFor(b1, c1, {"CRUCIBLE_PLE_HOST=1", "prefill_rows=4096"});
  const auto b = StateDiskIdentityFor(b2, c1, {"CRUCIBLE_PLE_HOST=1", "prefill_rows=4096"});
  const auto c = StateDiskIdentityFor(b1, c1, {"CRUCIBLE_PLE_HOST=0", "prefill_rows=4096"});
  const auto d = StateDiskIdentityFor(b1, c1, {"prefill_rows=4096", "CRUCIBLE_PLE_HOST=1"});
  assert(a.sha != b.sha && a.sha != c.sha && a.sha == d.sha && a.directory != b.directory);
  assert(std::regex_match(a.directory, std::regex("[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}")));
  const char* env[] = {"PATH=/usr/bin", "CRUCIBLE_FUSE=2", "HOME=/x", "CRUCIBLE_PLE_HOST=1", nullptr};
  const auto settings = StateDiskSettings(const_cast<char**>(env), 4096, false);
  assert((settings == std::vector<std::string>{"CRUCIBLE_FUSE=2", "CRUCIBLE_PLE_HOST=1", "prefill_rows=4096",
                                               "vision=0"}));
  // The MTP sidecar enters the identity by file content, as the ablation file does: another file at the same path
  // must miss (a stale sidecar would restore wrong MTP rows).
  {
    const fs::path file = Root() / "sidecar.bin";
    const std::string path_entry = "CRUCIBLE_MTP_SIDECAR=" + file.string();
    const char* sidecar_env[] = {path_entry.c_str(), nullptr};
    const auto identity = [&](const std::string& content) {
      { std::ofstream(file, std::ios::binary | std::ios::trunc) << content; }
      return StateDiskIdentityFor(b1, c1, StateDiskSettings(const_cast<char**>(sidecar_env), 4096, false));
    };
    const auto one = identity("sidecar one");
    const auto again = identity("sidecar one");
    const auto other = identity("sidecar two");
    assert(one.sha == again.sha && one.sha != other.sha && one.directory != other.directory);
    const auto settings_with = StateDiskSettings(const_cast<char**>(sidecar_env), 4096, false);
    assert(std::find_if(settings_with.begin(), settings_with.end(),
                        [](const std::string& item) { return item.rfind("mtp_sidecar_sha256=", 0) == 0; }) !=
           settings_with.end());
    fs::remove(file);
  }
  // The BF16 n-gram table enters by size, modification time, inode and ctime: a replaced file at the same path must
  // miss.
  {
    const fs::path file = Root() / "ngram_bf16.safetensors";
    const std::string path_entry = "CRUCIBLE_PLE_BF16=" + file.string();
    const char* bf16_env[] = {path_entry.c_str(), "CRUCIBLE_PLE_HOST=1", nullptr};
    const auto identity = [&] {
      return StateDiskIdentityFor(b1, c1, StateDiskSettings(const_cast<char**>(bf16_env), 4096, false));
    };
    const auto missing = identity();
    { std::ofstream(file, std::ios::binary | std::ios::trunc) << "table one"; }
    const auto one = identity();
    assert(one.sha == identity().sha && one.sha != missing.sha);
    { std::ofstream(file, std::ios::binary | std::ios::trunc) << "table one, longer"; }
    assert(identity().sha != one.sha);
    // A replacement with the same size and modification time (touch -r) must also miss: a rename
    // gives a new inode, and an in-place rewrite gives a new ctime, which no user tool can set back.
    const auto same_size_and_mtime = [&](const fs::path& target, const char* content) {
      struct stat before{};
      assert(::stat(file.c_str(), &before) == 0);
      { std::ofstream(target, std::ios::binary | std::ios::trunc) << content; }
      const struct timespec times[2] = {before.st_atim, before.st_mtim};
      assert(::utimensat(AT_FDCWD, target.c_str(), times, 0) == 0);
      if (target != file) fs::rename(target, file);
      struct stat after{};
      assert(::stat(file.c_str(), &after) == 0);
      assert(after.st_size == before.st_size && after.st_mtim.tv_sec == before.st_mtim.tv_sec &&
             after.st_mtim.tv_nsec == before.st_mtim.tv_nsec);
    };
    { std::ofstream(file, std::ios::binary | std::ios::trunc) << "table one"; }
    const auto base = identity();
    same_size_and_mtime(Root() / "ngram_bf16.next", "table two");  // made while the old file exists: a new inode
    const auto renamed = identity();
    assert(renamed.sha != base.sha);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // past a coarse timestamp tick
    same_size_and_mtime(file, "table 3!!");
    assert(identity().sha != renamed.sha);
    fs::remove(file);
  }
  // The checkpoint identity also names each weights file by size and modification time.
  const fs::path pack = Root() / "pack";
  fs::create_directories(pack);
  { std::ofstream(pack / "config.json") << "{}"; }
  { std::ofstream(pack / "model-00001.safetensors") << "abc"; }
  const std::string first = CheckpointSha256(pack);
  assert(first == CheckpointSha256(pack));
  { std::ofstream(pack / "model-00001.safetensors", std::ios::app) << "d"; }
  const std::string grown = CheckpointSha256(pack);
  assert(grown != first);
  { std::ofstream(pack / "notes.txt") << "x"; }
  assert(CheckpointSha256(pack) == grown);
  fs::remove_all(pack);
}

// SSD wear guards (design decision, Oct 8): byte counters, a daily write budget that survives a restart and also stops a stop
// flush, and no second write of a state whose file is on disk.
void WearGuards() {
  Logs logs;
  std::string today = "2026-10-08";
  auto options = Options("l", logs, "wear");
  options.today = [&] { return today; };
  const auto one = Make(Tokens(10), 3 * 4096, 1);
  const uint64_t file = 4096 + 3 * 4096;  // header page and payload
  options.write_per_day = 2 * file + 100;  // two files a day
  {
    Disk disk(options);
    Spill(disk, one);
    assert(disk.Drain(30));
    Spill(disk, one, Policy(), false, "present");  // the same full key: no second write
    auto two_ids = Tokens(10); two_ids.push_back(5);
    Spill(disk, Make(two_ids, 3 * 4096, 2));
    assert(disk.Drain(30));
    auto three_ids = two_ids; three_ids.push_back(6);
    const auto three = Make(three_ids, 3 * 4096, 3);
    Spill(disk, three, Policy(), false, "daily");
    // A stop flush obeys the budget too, and nothing stalls.
    assert(disk.Spill(three->hist, Policy(), false, false, false, three, *three, true) == "daily");
    const auto stats = disk.Stats();
    assert(stats.written == 2 && stats.written_bytes == 2 * file && stats.present == 1 && stats.refused_daily == 2);
    assert(disk.DayWritten() == 2 * file);
    assert(disk.Present(one->hist, Policy(), false) && !disk.Present(three->hist, Policy(), false));
    const auto meta = disk.Best(one->hist, Policy(), false, Any);
    StageWait(disk, *meta);
    assert(disk.Stats().read_bytes == 3 * 4096);
  }
  assert(logs.Has("result=refused reason=daily") && logs.Has("day_written=") && logs.Has("day_read=12288"));
  assert(Mode(Root() / ".cache" / "wear" / "write-ledger") == 0600);
  {
    // A restart on the same day keeps the day's total; a new day starts at zero.
    Disk disk(options);
    assert(disk.DayWritten() == 2 * file);
    auto ids = Tokens(10, 7);
    const auto other = Make(ids, 3 * 4096, 4);
    Spill(disk, other, Policy(), false, "daily");
    today = "2026-10-09";
    Spill(disk, other);
    assert(disk.Drain(30) && disk.DayWritten() == file && disk.Stats().written == 1);
  }
  // A budget of zero: the tier reads but never writes.
  auto read_only = Options("l", logs, "wear");
  read_only.write_per_day = 0;
  read_only.today = [&] { return today; };
  Disk disk(read_only);
  Spill(disk, Make(Tokens(10, 9), 4096, 5), Policy(), false, "daily");
}

// F3 (a poll keeps a read state), F4 (eviction stays fast with 200 files), and the mutants that
// survived (directory modes, the queue limit at a stop, the file size check, the header checksum).
void ReadStaysWhilePolled() {
  Logs logs;
  auto options = Options("m", logs);
  options.stage_seconds = 0.3;
  Disk disk(options);
  Spill(disk, Make(Tokens(10), 4096, 1));
  assert(disk.Drain(30));
  const auto meta = disk.Best(Tokens(11), Policy(), false, Any);
  assert(StageWait(disk, *meta).state == StageState::Ready);
  for (int i = 0; i < 4; ++i) {  // 0.6 s in all, each poll within the 0.3 s limit
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    assert(disk.Stage(*meta, UINT64_MAX).state == StageState::Ready);
  }
  assert(disk.Ready(Tokens(11), Policy(), false, Any).snapshot);
  std::this_thread::sleep_for(std::chrono::milliseconds(400));  // no poll: the read state leaves memory
  assert(!disk.Ready(Tokens(11), Policy(), false, Any).snapshot);
}

// A disk budget that holds about 190 of the files that the first pass wrote.
uint64_t disk_budget_for_test(int files, Logs& logs, const StateDiskOptions& options) {
  Disk disk(options);
  assert(disk.Entries() == size_t(files));
  return disk.Bytes() - 10 * (disk.Bytes() / files);
}

void EvictionScales() {
  Logs logs;
  auto options = Options("n", logs);
  options.min_tokens = 1;
  const uint64_t file = 4096 * 2;  // a small header page and one payload page
  // 200 sessions of 16K to 21K tokens behind one 12K-token system prompt, then 40 more: each new write evicts one.
  std::vector<int> shared = Tokens(12000, 77);
  const auto session = [&](int k) {
    auto ids = shared;
    // Lengths differ, so each pair compares its common 12K-token prefix (a pairwise check costs 48 KB a pair).
    for (int i = 0; i < 4000 + (k % 50) * 100; ++i) ids.push_back((k * 131 + i * 7) % 240000);
    return ids;
  };
  {
    auto first = options;
    Disk disk(first);
    for (int k = 0; k < 200; ++k) Spill(disk, Make(session(k), 64, uint8_t(k)));
    assert(disk.Drain(120) && disk.Entries() == 200);
  }
  options.budget = disk_budget_for_test(200, logs, options);
  Disk disk(options);
  assert(disk.Entries() >= 185 && disk.Entries() < 200);  // the start trims to the budget
  std::atomic<bool> done = false;
  std::atomic<double> worst = 0;
  std::thread probe([&] {
    const auto key = session(3);
    while (!done) {
      const auto t0 = std::chrono::steady_clock::now();
      disk.Present(key, Policy(), false);  // takes the same lock as the device thread
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (ms > worst) worst = ms;
    }
  });
  for (int k = 200; k < 240; ++k) Spill(disk, Make(session(k), 64, uint8_t(k)));
  assert(disk.Drain(120));
  done = true;
  probe.join();
  assert(disk.Entries() < 200 && disk.Stats().evicted >= 40);
  std::printf("eviction with 200 files: worst lock wait %.2f ms\n", worst.load());
  // The pairwise check of 76c2d0d9 waited about 38 ms here (-O2); the one-pass check about 1.3 ms. Sanitizers slow both.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
  assert(worst < 200);
#else
  assert(worst < 15);
#endif
  (void)file;
}

void DirectoryModes() {
  Logs logs;
  auto options = Options("o", logs, "modes");
  fs::create_directories(options.root / options.directory);
  fs::permissions(options.root, fs::perms::all);                        // 0777
  fs::permissions(options.root / options.directory, fs::perms::all);    // 0777
  { std::ofstream(options.root / "write-ledger") << ""; }
  Disk disk(options);
  assert(disk.Enabled() && Mode(options.root) == 0700 && Mode(disk.Directory()) == 0700);
  // A root that holds other files (for example --state-dir ~/.cache) is not the tier's own folder: refused, and its
  // mode stays.
  auto shared = Options("o", logs, "shared");
  fs::create_directories(shared.root);
  { std::ofstream(shared.root / "fontconfig") << "x"; }
  fs::permissions(shared.root, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
                                   fs::perms::others_read | fs::perms::others_exec);  // 0755
  bool refused = false;
  try { Disk other(shared); } catch (const std::invalid_argument&) { refused = true; }
  assert(refused && Mode(shared.root) == 0755 && !fs::exists(shared.root / shared.directory));
  // A new root (and each folder that the tier creates on the way) gets mode 0700.
  auto fresh = Options("o", logs, "new/states");
  Disk created(fresh);
  assert(created.Enabled() && Mode(fresh.root) == 0700 && Mode(fresh.root.parent_path()) == 0700);
}

// Fix 3 (review rev-f3 items 2 and 4): a new state takes the queue place of the first waiting prefix of its branch, and
// every waiting prefix of that branch leaves at once (12 and 16 tokens). A job behind them stays and keeps its order.
// The same 16-token key with another policy, storage mode or tenant is another branch: it stays.
void QueueDropsPrefixesInPlace() {
  Logs logs;
  auto options = Options("y", logs, "states-f3");
  options.queue_bytes = 6500;
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<int> waiting = 0;
  options.before_write = [&] { ++waiting; gate.wait(); };
  Disk disk(options);
  Spill(disk, Make(Tokens(10, 9), 1000, 1));  // the writer holds another branch
  while (waiting == 0) std::this_thread::yield();
  const auto b = Tokens(30, 3);
  const std::vector<int> b12(b.begin(), b.begin() + 12), b16(b.begin(), b.begin() + 16);
  Spill(disk, Make(b12, 1000, 2));
  Spill(disk, Make(b16, 1000, 3));
  const auto behind = Tokens(20, 5);
  Spill(disk, Make(behind, 1000, 4));
  ConversationPolicy tenant_policy = Policy();  // the policy holds the tenant
  tenant_policy.tenant[0] = 1;
  const auto kp = Make(b16, 1000, 5), kc = Make(b16, 1000, 6), kt = Make(b16, 1000, 7);
  assert(disk.Spill(b16, Policy(8), false, false, false, kp, *kp).empty());
  assert(disk.Spill(b16, Policy(), true, false, false, kc, *kc).empty());
  assert(disk.Spill(b16, tenant_policy, false, false, false, kt, *kt).empty());
  // 6,000 bytes wait: a 1,500-byte state passes the limit unless both prefixes leave.
  assert(disk.QueueFull(1500) && !disk.QueueFull(1500, b, Policy(), false));
  Spill(disk, Make(b, 1500, 8, true));
  assert(logs.Has("state_disk queue_drop tokens=12 bytes=1000 reason=superseded by=30"));
  assert(logs.Has("state_disk queue_drop tokens=16 bytes=1000 reason=superseded by=30"));
  assert(disk.QueuedBytes() == 1000 + 1500 + 4 * 1000);
  assert(!disk.Waiting(b12, Policy(), false) && !disk.Waiting(b16, Policy(), false) && disk.Waiting(b, Policy(), false));
  assert(disk.Waiting(behind, Policy(), false) && disk.Waiting(b16, Policy(8), false));
  assert(disk.Waiting(b16, Policy(), true) && disk.Waiting(b16, tenant_policy, false));
  release.set_value();
  assert(disk.Drain(30) && disk.QueuedBytes() == 0);
  assert(disk.Present(b, Policy(), false) && !disk.Present(b12, Policy(), false) && !disk.Present(b16, Policy(), false));
  assert(disk.Present(behind, Policy(), false) && disk.Present(b16, Policy(8), false));
  assert(disk.Present(b16, Policy(), true) && disk.Present(b16, tenant_policy, false));
  // The writer order: the new state in the place of the 12-token prefix, before the job behind it.
  const auto written = [&](const std::string& tokens) {
    std::lock_guard lock(logs.mutex);
    const auto found = std::find_if(logs.lines.begin(), logs.lines.end(), [&](const std::string& line) {
      return line.rfind("state_disk spill tokens=" + tokens + " ", 0) == 0 && line.find("result=written") != std::string::npos;
    });
    assert(found != logs.lines.end());
    return found - logs.lines.begin();
  };
  assert(written("30") < written("20"));
}

void FlushIgnoresQueueLimit() {
  Logs logs;
  auto options = Options("p", logs);
  options.queue_bytes = 5000;
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<int> waiting = 0;
  options.before_write = [&] { ++waiting; gate.wait(); };
  Disk disk(options);
  Spill(disk, Make(Tokens(10), 4096, 1));
  while (waiting == 0) std::this_thread::yield();
  Spill(disk, Make(Tokens(11, 1), 4096, 2));  // waits beside the state in the writer (another branch)
  const auto third = Make(Tokens(12, 2), 4096, 3);
  Spill(disk, third, Policy(), false, "busy");
  assert(disk.Spill(third->hist, Policy(), false, false, false, third, *third, true).empty());  // a stop flush
  release.set_value();
  assert(disk.Drain(30) && disk.Entries() == 3);
}

uint64_t Field(const fs::path& path, std::streamoff offset) {
  uint64_t value = 0;
  std::ifstream in(path, std::ios::binary);
  in.seekg(offset);
  in.read(reinterpret_cast<char*>(&value), 8);
  return value;
}
void Flip(const fs::path& path, uint64_t offset) {
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  file.seekg(std::streamoff(offset));
  const char c = char(file.get());
  file.seekp(std::streamoff(offset));
  file.put(char(c ^ 0x10));
}
void SizeAndHeaderChecks() {
  Logs logs;
  fs::path path;
  {
    Disk disk(Options("q", logs));
    Spill(disk, Make(Tokens(10), 3 * 4096, 1));
    assert(disk.Drain(30));
    const auto meta = disk.Best(Tokens(10), Policy(), false, Any);
    path = disk.Directory() / meta->name;
    // A file that grew after the index read its header fails the read.
    { std::ofstream(path, std::ios::app | std::ios::binary) << std::string(4096, 'x'); }
    assert(StageWait(disk, *meta).state == StageState::Failed);
    Spill(disk, Make(Tokens(10), 3 * 4096, 1));
    assert(disk.Drain(30));
    // Version 2: a changed checksum of the last payload chunk is in the resume section, which only a read
    // checks. The read fails and the state leaves.
    Flip(path, Field(path, 24) - 9);
    assert(StageWait(disk, *meta).reason == "header" && disk.Entries() == 0);
    Spill(disk, Make(Tokens(10), 3 * 4096, 1));
    assert(disk.Drain(30));
  }
  // A changed byte of the index section: the header checksum finds it at the next start.
  Flip(path, Field(path, 48) - 9);
  Disk disk(Options("q", logs));
  assert(disk.Entries() == 0);
}

// the mutants that survived. N4: the writer reads the shared ledger before a write, so
// another engine's writes count. N6 and N7: the superseded marks at insert and at removal. N10: the space that
// eviction frees counts toward the free space. N11: a stop ends a write in progress at its next chunk.
void SharedLedger() {
  Logs logs;
  const std::string today = "2026-10-10";
  const uint64_t file = 4096 + 3 * 4096;
  auto a_options = Options("s", logs, "ledger");
  a_options.today = [&] { return today; };
  a_options.write_per_day = 2 * file + file / 2;
  auto b_options = Options("t", logs, "ledger");
  b_options.today = a_options.today;
  b_options.write_per_day = a_options.write_per_day;
  Disk a(a_options);
  Disk b(b_options);
  Spill(a, Make(Tokens(10), 3 * 4096, 1));
  assert(a.Drain(30) && a.DayWritten() == file);
  Spill(b, Make(Tokens(10, 3), 3 * 4096, 2));
  assert(b.Drain(30) && b.DayWritten() == 2 * file);
  Spill(a, Make(Tokens(10, 5), 3 * 4096, 3));  // A's own view is one file, so the spill queues
  assert(a.Drain(30) && a.Entries() == 1 && a.Stats().refused_daily == 1 && a.DayWritten() == 2 * file);
}

void SupersededMarks() {
  Logs logs;
  const uint64_t file = 4096 + 3 * 4096;
  auto options = Options("u", logs, "marks");
  options.budget = 3 * file + 100;
  const auto unrelated = Make(Tokens(10, 500), 3 * 4096, 1);
  const auto older = Make(Tokens(10), 3 * 4096, 2);
  auto longer = Tokens(10);
  longer.push_back(999);
  const auto newer = Make(longer, 3 * 4096, 3);
  {
    Disk disk(options);
    for (const auto& snapshot : {unrelated, older, newer}) {
      Spill(disk, snapshot);
      assert(disk.Drain(30));
    }
    // The insert of `newer` marked `older`: a fourth, unrelated write evicts it, not the least recently used.
    Spill(disk, Make(Tokens(10, 600), 3 * 4096, 4));
    assert(disk.Drain(30) && disk.Entries() == 3);
    assert(!disk.Present(older->hist, Policy(), false) && disk.Present(unrelated->hist, Policy(), false));
  }
  fs::remove_all(options.root);
  Disk disk(options);
  for (const auto& snapshot : {unrelated, older, newer}) {
    Spill(disk, snapshot);
    assert(disk.Drain(30));
  }
  // `newer` leaves (its read fails), so `older` is not superseded any more: the least recently used leaves.
  const auto meta = disk.Best(longer, Policy(), false, Any);
  { std::ofstream(disk.Directory() / meta->name, std::ios::app | std::ios::binary) << std::string(4096, 'x'); }
  assert(StageWait(disk, *meta).state == StageState::Failed && disk.Entries() == 2);
  Spill(disk, Make(Tokens(10, 600), 3 * 4096, 4));
  Spill(disk, Make(Tokens(10, 700), 3 * 4096, 5));
  assert(disk.Drain(30) && disk.Entries() == 3);
  assert(disk.Present(older->hist, Policy(), false) && !disk.Present(unrelated->hist, Policy(), false));
}

void EvictionFreesSpace() {
  Logs logs;
  const uint64_t file = 4096 + 3 * 4096;
  auto options = Options("w", logs, "space");
  options.budget = 2 * file + 100;
  options.free_reserve = 4096;
  uint64_t free_bytes = uint64_t(1) << 40;
  options.free_space = [&] { return free_bytes; };
  Disk disk(options);
  Spill(disk, Make(Tokens(10, 1), 3 * 4096, 1));
  Spill(disk, Make(Tokens(10, 2), 3 * 4096, 2));
  assert(disk.Drain(30) && disk.Entries() == 2);
  free_bytes = file + options.free_reserve - 1;  // not enough alone; enough with the file that eviction frees
  Spill(disk, Make(Tokens(10, 3), 3 * 4096, 3));
  assert(disk.Drain(30) && disk.Entries() == 2 && disk.Stats().evicted == 1 && disk.Present(Tokens(10, 3), Policy(), false));
}

void StopEndsAWrite() {
  Logs logs;
  auto options = Options("x", logs, "stop");
  options.io_threads = 1;
  std::promise<void> in_chunk;
  std::atomic<bool> first = true;
  options.before_chunk = [&] {
    if (!first.exchange(false)) return;
    in_chunk.set_value();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));  // the destructor sets the stop flag meanwhile
  };
  {
    Disk disk(options);
    Spill(disk, Make(Tokens(10), 4 * SnapshotChunkPool::kChunkBytes, 1));
    in_chunk.get_future().wait();
  }
  options.before_chunk = nullptr;
  Disk disk(options);
  assert(disk.Entries() == 0 && logs.Has("result=refused reason=stopped"));
  for (const auto& item : fs::directory_iterator(disk.Directory())) assert(item.path().filename() == ".lock");
}

// Oct 9 device run: every spill of the engine was refused with reason=io. Direct I/O from or into hipHostMalloc pages
// fails (the test machine: EFAULT), so a payload chunk must go through an aligned bounce buffer, both ways.
void DirectIoFromPinnedMemory() {
  Root();
  assert(direct_root);  // the test exercises O_DIRECT on a real filesystem
  Logs logs;
  auto options = Options("p", logs, "pinned");
  options.io_fault = [](const void* buffer) { return IsPinned(buffer) ? EFAULT : 0; };
  Disk disk(options);
  const auto snapshot = Make(Tokens(30), 3 * kChunk + 4096, 3);
  Spill(disk, snapshot);
  assert(disk.Drain(30));
  assert(disk.Entries() == 1 && disk.Stats().refused == 0 && !logs.Has("result=refused"));
  const auto meta = disk.Best(Tokens(31), Policy(), false, Any);
  assert(meta && StageWait(disk, *meta).state == StageState::Ready);
  Equal(*disk.Ready(Tokens(31), Policy(), false, Any).snapshot, *snapshot);
}

// An I/O error names its errno in the spill and read lines (numbers only), and a read error keeps the file.
void IoErrorsNameErrno() {
  Logs logs;
  auto options = Options("q", logs, "errno");
  std::atomic<int> fault = EIO;
  options.io_fault = [&](const void*) { return fault.load(); };
  Disk disk(options);
  Spill(disk, Make(Tokens(20), kChunk + 100, 4));
  assert(disk.Drain(30) && disk.Entries() == 0);
  assert(logs.Has("result=refused reason=io errno=5 "));
  fault = 0;
  const auto snapshot = Make(Tokens(20), kChunk + 100, 4);
  Spill(disk, snapshot);
  assert(disk.Drain(30) && disk.Entries() == 1);
  fault = EFAULT;
  const auto meta = disk.Best(Tokens(21), Policy(), false, Any);
  assert(meta && StageWait(disk, *meta).state == StageState::Failed);
  assert(logs.Has("result=failed reason=io errno=14") && disk.Entries() == 1);
  // an open error of the state file that is not ENOENT (EACCES here, EMFILE or EIO in service) is
  // an I/O error: the state stays. (As root the open does not fail.)
  fault = 0;
  const fs::path path = disk.Directory() / meta->name;
  fs::permissions(path, fs::perms::none);
  if (geteuid() != 0) {
    assert(StageWait(disk, *meta).state == StageState::Failed);
    assert(logs.Has("result=failed reason=io errno=13") && disk.Entries() == 1 && fs::exists(path));
  }
  fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write);
  assert(StageWait(disk, *meta).state == StageState::Ready);
}

// One thread reuses its bounce buffer, so the buffer holds the full chunks before the last partial chunk. The padding
// after the payload stays zero only because the writer clears it. (With more threads than chunks every buffer is new
// and zero, and the check passes without the clear.)
void PaddingIsZeroWithReusedBuffer() {
  Root();
  assert(direct_root);
  Logs logs;
  auto options = Options("r", logs, "reuse");
  options.io_threads = 1;
  Disk disk(options);
  constexpr size_t kPayload = 2 * kChunk + 12345;  // two full chunks, then a partial one
  static_assert(kPayload % 4096 != 0);
  const auto snapshot = Make(Tokens(40), kPayload, 11);
  Spill(disk, snapshot);
  assert(disk.Drain(30) && disk.Entries() == 1);
  const auto meta = disk.Best(Tokens(41), Policy(), false, Any);
  assert(meta);
  std::ifstream in(disk.Directory() / meta->name, std::ios::binary);
  std::vector<char> data((std::istreambuf_iterator<char>(in)), {});
  const size_t padded_end = data.size();
  const size_t payload_end = padded_end - (4096 - kPayload % 4096);
  assert(payload_end < padded_end);
  for (size_t i = payload_end; i < padded_end; ++i) assert(data[i] == 0);
  assert(StageWait(disk, *meta).state == StageState::Ready);
  Equal(*disk.Ready(Tokens(41), Policy(), false, Any).snapshot, *snapshot);
}

// The checksum algorithm, written again here: the stored sums and the file bytes must not change when the writer
// computes the sum on the bounce buffer instead of on the pinned chunk.
uint64_t ReferenceChecksum(const uint8_t* p, size_t n) {
  constexpr uint64_t kMul = 0x9E3779B97F4A7C15ull;
  uint64_t lane[4] = {0x243F6A8885A308D3ull, 0x13198A2E03707344ull, 0xA4093822299F31D0ull, 0x082EFA98EC4E6C89ull};
  size_t i = 0;
  for (; i + 32 <= n; i += 32)
    for (int k = 0; k < 4; ++k) {
      uint64_t word;
      std::memcpy(&word, p + i + 8 * k, 8);
      lane[k] = (lane[k] ^ word) * kMul;
      lane[k] ^= lane[k] >> 29;
    }
  uint64_t h = uint64_t(n) * kMul;
  for (const uint64_t v : lane) h = (h ^ v) * kMul, h ^= h >> 31;
  for (; i < n; ++i) h = (h ^ p[i]) * kMul, h ^= h >> 31;
  return h;
}

// For several shapes and thread counts: the header holds the reference sum of every chunk, the payload in the file is
// the state byte for byte, and the state reads back.
void StoredSumsAndBytesMatchReference() {
  Root();
  assert(direct_root);
  const size_t shapes[] = {1000, kChunk, kChunk + 1, 2 * kChunk + 12345, 3 * kChunk + 4096};
  int id = 0;
  for (const int threads : {1, 8}) {
    for (const size_t payload : shapes) {
      Logs logs;
      auto options = Options("s", logs, "sums" + std::to_string(id));
      options.io_threads = threads;
      Disk disk(options);
      const auto snapshot = Make(Tokens(40), payload, uint8_t(20 + id++));
      Spill(disk, snapshot);
      assert(disk.Drain(30) && disk.Entries() == 1);
      const auto meta = disk.Best(Tokens(41), Policy(), false, Any);
      assert(meta);
      std::ifstream in(disk.Directory() / meta->name, std::ios::binary);
      const std::vector<char> data((std::istreambuf_iterator<char>(in)), {});
      const auto header = ParseStateHeader(reinterpret_cast<const uint8_t*>(data.data()), data.size());
      assert(header.sums.size() == (payload + kChunk - 1) / kChunk);
      std::vector<uint8_t> expect(payload);
      snapshot->bytes.ForEachPiece(0, payload, [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
        std::memcpy(expect.data() + at, chunk.host + inside, count);
      });
      for (size_t i = 0; i < header.sums.size(); ++i) {
        const size_t length = std::min(kChunk, payload - i * kChunk);
        assert(header.sums[i] == ReferenceChecksum(expect.data() + i * kChunk, length));
      }
      assert(data.size() == header.header_bytes + StateRoundUp(payload));
      assert(std::memcmp(data.data() + header.header_bytes, expect.data(), payload) == 0);
      assert(StageWait(disk, *meta).state == StageState::Ready);
      Equal(*disk.Ready(Tokens(41), Policy(), false, Any).snapshot, *snapshot);
    }
  }
}

int main() {
  DirectIoFromPinnedMemory();
  IoErrorsNameErrno();
  PaddingIsZeroWithReusedBuffer();
  StoredSumsAndBytesMatchReference();
  RoundTripAndRebuild();
  ComposedSnapshotSpills();
  KeyMatchesExactly();
  IdentitiesAndLocks();
  OldBinaryRemovesBlocks();
  CorruptFilesFail();
  DamagedHeaderLeaves();
  BudgetAndFullDisk();
  QueueLimitAndMemory();
  QueueDropsPrefixesInPlace();
  HostRoom();
  LogsHoldNoTokens();
  IdentityText();
  WearGuards();
  ReadStaysWhilePolled();
  EvictionScales();
  DirectoryModes();
  FlushIgnoresQueueLimit();
  SizeAndHeaderChecks();
  SharedLedger();
  SupersededMarks();
  EvictionFreesSpace();
  StopEndsAWrite();
  fs::remove_all(Root().parent_path());
  std::puts("state_disk_test passed");
}
