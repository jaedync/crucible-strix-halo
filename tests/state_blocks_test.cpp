// Block storage of the NVMe tier: the block plan, the formats, the
// SHA-256 path, and the store (shared blocks, references, eviction, crash states, damage, pins, prune, limits).
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <regex>
#include <vector>

std::mutex pinned_mutex;
std::map<uintptr_t, size_t> pinned;
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
constexpr int hipMemcpyDeviceToHost = 0, hipMemcpyHostToDevice = 1;
int hipMemcpy(void* out, const void* in, size_t n, int) { std::memcpy(out, in, n); return 0; }
int hipStreamSynchronize(int) { return 0; }
#define HIP_OK(expr) assert((expr) == 0)
#include "../engine/runtime/state_disk.h"
#include "support/disk_test_dir.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <thread>

using namespace crucible;
namespace fs = std::filesystem;

struct TestSnapshot {
  std::vector<int> hist;
  RetainedPrefix retained;
  ConversationResume resume;
  SnapshotBytes bytes;
  std::vector<std::pair<int, int>> qsa_layout;
};
using Disk = StateDisk<TestSnapshot>;
using Layout = std::vector<std::pair<int, int>>;

// Every log line of every disk in this test: the pattern check runs at the end (no token, no tenant byte).
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
} logs;

fs::path Root() {
  static const fs::path root = [] {
    const DiskTestDir dir = PickDiskTestDir("state_blocks_test");
    assert(dir.direct);  // the O_DIRECT path, as state_disk_test
    fs::create_directories(dir.path / "home");
    ::setenv("HOME", (dir.path / "home").c_str(), 1);
    return dir.path / "home";
  }();
  return root;
}

// The test geometry: a fixed-size head, a full layer, a compact layer and the MTP layer, then the stacks.
constexpr uint64_t kHead = 4096 + 100, kTail = 777;
constexpr size_t kRows = 128;  // the block rows of these tests (the engine uses kStateBlockRows)
const Layout kLayout = {{3, 0}, {7, kQsaRawSlots}, {kLayers, 0}};

std::string DirName(char id) {
  const std::string group(16, "0123456789abcdef"[id % 16]);
  return group + "-" + group + "-" + group;
}
StateDiskOptions Options(char id, const std::string& name) {
  StateDiskOptions options;
  options.root = Root() / ".cache" / name;
  options.identity = std::string(64, id);
  options.directory = DirName(id);
  options.min_tokens = 4;
  options.io_threads = 3;
  options.log = logs.Sink();
  options.block_rows = kRows;
  options.block_head_bytes = kHead;
  options.block_tail_bytes = kTail;
  options.verify_digest = true;
  return options;
}
ConversationPolicy Policy(const StateTenant& tenant = kLocalTenant) {
  ConversationPolicy policy;
  policy.prefill_rows = 16;
  policy.tenant = tenant;
  return policy;
}
std::vector<int> Tokens(size_t n, int salt = 0) {
  std::vector<int> ids(n);
  for (size_t i = 0; i < n; ++i) ids[i] = int(i * 7 + salt) % 1000;
  return ids;
}
auto Any = [](const StateMeta&) { return true; };

uint64_t SpanBytes(const Layout& layout, size_t rows, size_t mpos) {
  uint64_t bytes = 0;
  for (const auto& [layer, slots] : layout)
    for (const auto& span : QsaSnapshotSpans(0, slots, layer == kLayers ? mpos : rows)) bytes += span.bytes;
  return bytes;
}
SnapshotGeometry Geometry(const TestSnapshot& s, uint64_t head = kHead, uint64_t tail = kTail) {
  SnapshotGeometry g;
  g.layout = s.qsa_layout;
  g.rows = s.hist.size();
  g.mpos = size_t(std::max(s.resume.mpos, 0));
  g.head = head;
  g.tail = tail;
  g.total = s.bytes.size();
  return g;
}

// The byte of block `index` of `lane` at stream position `at`. Equal functions give equal blocks in two states.
using Fill = std::function<uint8_t(int lane, uint64_t index, uint64_t at)>;
// Not periodic in the stream position: a gather that moves a range gives other bytes.
uint8_t BlockByte(int lane, uint64_t index, uint64_t at) {
  return uint8_t((uint32_t(at) * 0x9E3779B1u) >> 24) + uint8_t(index * 13 + lane * 29 + 1);
}

void SetStream(SnapshotBytes& bytes, std::span<const StreamRange> ranges, const std::function<uint8_t(uint64_t)>& f) {
  ForEachStreamPiece(ranges, 0, RangeBytes(ranges), [&](uint64_t offset, uint64_t at, uint64_t count) {
    bytes.ForEachPiece(size_t(offset), size_t(count), [&](const SnapshotChunk& chunk, size_t inside, size_t n, size_t done) {
      for (size_t i = 0; i < n; ++i) chunk.host[inside + i] = f(at + done + i);
    });
  });
}
// Without `fill`, the block bytes depend on the first token: the turns of one session share their blocks, and
// unrelated sessions (Tokens with another salt) do not.
std::shared_ptr<TestSnapshot> Make(const std::vector<int>& hist, size_t mpos, uint8_t seed, Fill fill = nullptr,
                                   const Layout& layout = kLayout) {
  if (!fill) {
    const uint8_t session = uint8_t(hist[0] * 7);
    fill = [session](int lane, uint64_t index, uint64_t at) { return uint8_t(BlockByte(lane, index, at) + session); };
  }
  auto s = std::make_shared<TestSnapshot>();
  s->hist = hist;
  RetainedPrefix::FromFields({hist.size(), hist.size() - 1, hist.size() - 2, 1}, s->retained);
  s->resume.length = hist.size();
  s->resume.mpos = int(mpos);
  s->resume.prompt_only = true;
  s->resume.next_id = -1;
  s->resume.logits = {1.5f, -2.25f, float(seed)};
  s->resume.pending_tokens = {5, 6};
  s->qsa_layout = layout;
  const uint64_t total = kHead + SpanBytes(layout, hist.size(), mpos) + kTail;
  s->bytes = SnapshotBytes(total);
  s->bytes.ForEachPiece(0, total, [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
    for (size_t i = 0; i < count; ++i) chunk.host[inside + i] = uint8_t((at + i) * 131 + seed);
  });
  const auto plan = PlanBlocks(Geometry(*s), kRows);
  assert(!plan.whole);
  for (const auto& block : plan.blocks)
    SetStream(s->bytes, block.ranges, [&](uint64_t at) { return fill(block.lane, block.index, at); });
  return s;
}
// The first `rows` of `hist`'s tokens as a longer turn of the same session (same block bytes).
std::vector<int> Prefix(const std::vector<int>& hist, size_t rows) { return {hist.begin(), hist.begin() + rows}; }

std::string Spill(Disk& disk, const std::shared_ptr<TestSnapshot>& s, bool shared = false,
                  const StateTenant& tenant = kLocalTenant, bool flush = false) {
  return disk.Spill(s->hist, Policy(tenant), false, shared, false, s, *s, flush, UINT64_MAX);
}
void SpillOk(Disk& disk, const std::shared_ptr<TestSnapshot>& s, bool shared = false,
             const StateTenant& tenant = kLocalTenant) {
  assert(Spill(disk, s, shared, tenant).empty());
  assert(disk.Drain(60));
}
StageStatus StageWait(Disk& disk, const StateMeta& meta) {
  for (int i = 0; i < 5000; ++i) {
    const auto status = disk.Stage(meta, UINT64_MAX);
    if (status.state != StageState::Reading) return status;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  assert(false);
  return {};
}
void Equal(const TestSnapshot& a, const TestSnapshot& b) {
  assert(a.hist == b.hist && a.retained.Fields() == b.retained.Fields());
  assert(a.resume.length == b.resume.length && a.resume.mpos == b.resume.mpos);
  assert(a.resume.logits == b.resume.logits && a.resume.pending_tokens == b.resume.pending_tokens);
  assert(a.qsa_layout == b.qsa_layout);
  assert(a.bytes == b.bytes);
}
// Restores the state with `s`'s key and checks every field and byte.
void Restores(Disk& disk, const TestSnapshot& s, const StateTenant& tenant = kLocalTenant) {
  const auto meta = disk.Best(s.hist, Policy(tenant), false, Any);
  assert(meta && meta->key == s.hist);
  assert(StageWait(disk, *meta).state == StageState::Ready);
  const auto ready = disk.Ready(s.hist, Policy(tenant), false, Any);
  assert(ready.snapshot && ready.meta.name == meta->name);
  Equal(*ready.snapshot, s);
  disk.Consume(ready);
}
std::vector<fs::path> Files(const fs::path& dir, const std::string& suffix) {
  std::vector<fs::path> files;
  for (const auto& item : fs::directory_iterator(dir))
    if (item.path().filename().string().ends_with(suffix)) files.push_back(item.path());
  std::sort(files.begin(), files.end());
  return files;
}
// The bytes of the state and block files on disk: the index's `Bytes()` must equal it after each step.
uint64_t DiskBytes(const fs::path& dir) {
  uint64_t bytes = 0;
  for (const auto& suffix : {".state", ".block"})
    for (const auto& file : Files(dir, suffix)) bytes += fs::file_size(file);
  return bytes;
}
// Train 22's name test: a rollback to it removes every file that matches, so the directory must
// hold no other name than these and `.lock` (Oct 15).
bool Train22Name(const std::string& name) {
  const auto has = [&](const char* part) { return name.find(part) != std::string::npos; };
  return name.size() > 6 && (name.ends_with(".state") || has(".state.tmp.") || has(".state.dead.") ||
                             name.ends_with(".block") || has(".block.tmp.") || has(".block.dead."));
}
void OnlyTrain22Names(const fs::path& dir) {
  for (const auto& item : fs::directory_iterator(dir)) {
    const std::string name = item.path().filename();
    assert(name == ".lock" || Train22Name(name));
  }
}
std::string Find(const std::string& text) {
  std::lock_guard lock(logs.mutex);
  for (auto it = logs.lines.rbegin(); it != logs.lines.rend(); ++it) if (it->find(text) != std::string::npos) return *it;
  return {};
}
uint64_t Field(const std::string& line, const std::string& name) {
  const auto at = line.find(" " + name + "=");
  assert(at != std::string::npos);
  return std::stoull(line.substr(at + name.size() + 2));
}
// The header size of a state or block file (the payload starts there).
uint64_t HeaderBytes(const fs::path& file) {
  uint64_t bytes = 0;
  std::ifstream in(file, std::ios::binary);
  in.seekg(16);
  in.read(reinterpret_cast<char*>(&bytes), 8);
  return bytes;
}
void Corrupt(const fs::path& file, uint64_t offset) {
  std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
  f.seekg(std::streamoff(offset));
  const char c = char(f.get());
  f.seekp(std::streamoff(offset));
  f.put(char(c ^ 0x5a));
}

// ---- the plan ----

// For random layouts and row counts, the plan is a partition of [0, total); blocks exist only for full rows of each
// lane; each block of a lane has the same stream size (two states share it).
void PlanCover() {
  std::mt19937 random(7);
  for (int round = 0; round < 400; ++round) {
    Layout layout;
    for (int layer = 0; layer < 3; ++layer)
      if (random() % 2) layout.emplace_back(layer * 4 + 3, random() % 2 ? kQsaRawSlots : 0);
    if (random() % 2) layout.emplace_back(kLayers, random() % 2 ? kQsaRawSlots : 0);
    const size_t rows = random() % (6 * kRows + 2), mpos = random() % (rows + 1);
    const uint64_t head = random() % 9000, tail = random() % 3000;
    SnapshotGeometry g;
    g.layout = layout;
    g.rows = rows;
    g.mpos = mpos;
    g.head = head;
    g.tail = tail;
    g.total = head + SpanBytes(layout, rows, mpos) + tail;
    const auto plan = PlanBlocks(g, kRows);
    assert(!plan.whole);
    std::vector<std::span<const StreamRange>> lists{plan.rest};
    size_t lanes[2] = {0, 0};
    for (const auto& block : plan.blocks) {
      lists.emplace_back(block.ranges);
      assert(block.index == lanes[block.lane]++);
      const bool trunk = std::any_of(layout.begin(), layout.end(), [](const auto& l) { return l.first != kLayers; });
      const bool mtp = std::any_of(layout.begin(), layout.end(), [](const auto& l) { return l.first == kLayers; });
      assert(block.lane == 0 ? trunk : mtp);
      assert(RangeBytes(block.ranges) == RangeBytes(plan.blocks[block.lane == 0 ? 0 : lanes[0]].ranges));
    }
    const bool trunk = std::any_of(layout.begin(), layout.end(), [](const auto& l) { return l.first != kLayers; });
    const bool mtp = std::any_of(layout.begin(), layout.end(), [](const auto& l) { return l.first == kLayers; });
    assert(lanes[0] == (trunk ? rows / kRows : 0));
    assert(lanes[1] == (mtp ? mpos / kRows : 0));
    assert(RangesCover(lists, g.total));
  }
  // The cover check finds a gap and an overlap.
  const std::vector<StreamRange> a{{0, 10}, {10, 5}}, gap{{0, 10}, {11, 4}}, overlap{{0, 10}, {9, 6}};
  assert(RangesCover({a}, 15) && !RangesCover({gap}, 15) && !RangesCover({overlap}, 15) && !RangesCover({a}, 16));
}

// QsaSnapshotSpans gives the sizes of the formulas that Spans used before (conversation_snapshot.h, train 22).
void PlanSpansMatchSave() {
  for (size_t rows = 0; rows < 3000; rows += rows < 300 ? 1 : 97)
    for (const int slots : {0, kQsaRawSlots}) {
      const auto spans = QsaSnapshotSpans(0, slots, rows);
      assert(spans[0].bytes == rows * 512 * 2 && spans[1].bytes == rows * 512 * 2);
      assert(spans[2].bytes == QsaRawSnapshotBytes(slots, rows));
      assert(spans[3].bytes == (rows + 127) / 128 * 4096 * sizeof(float));  // QsaPoolFloats (layers/ops.h)
    }
  assert(KvqPositionBytes(0) == 1024);
}

// A total that the spans do not explain, a KV layout that this binary does not know, and block rows that are not a
// multiple of the pool block all give one state file (no blocks), and that state still round-trips.
void PlanWholeOnMismatch() {
  const auto s = Make(Tokens(3 * kRows + 5), 2 * kRows, 1);
  auto g = Geometry(*s);
  assert(!PlanBlocks(g, kRows).whole && !PlanBlocks(g, kRows).blocks.empty());
  for (const uint64_t total : {g.total - 1, g.total + 1}) {
    auto other = g;
    other.total = total;
    const auto plan = PlanBlocks(other, kRows);
    assert(plan.whole && plan.blocks.empty() && plan.rest.size() == 1 && plan.rest[0].bytes == total);
  }
  auto unknown = g;
  unknown.kvl = 7;
  assert(PlanBlocks(unknown, kRows).whole);
  assert(PlanBlocks(g, 0).whole && PlanBlocks(g, 64).whole);
  // The disk tier with another head size: one state file, the same bytes back.
  auto options = Options('a', "whole");
  options.block_head_bytes = kHead + 1;
  Disk disk(options);
  SpillOk(disk, s);
  assert(disk.Blocks() == 0 && Find("result=written").find("plan=whole") != std::string::npos);
  Restores(disk, *s);
}

// Each KV layout that this binary knows gives blocks (the only one today: fp16, layout 0).
void PlanEachKvLayout() {
  const auto s = Make(Tokens(2 * kRows), kRows, 2);
  for (const int kvl : {0}) {
    auto g = Geometry(*s);
    g.kvl = kvl;
    const auto plan = PlanBlocks(g, kRows);
    assert(!plan.whole && plan.blocks.size() == 3);
    assert(RangeBytes(plan.blocks[0].ranges) == 2 * kRows * KvqPositionBytes(kvl) + kRows * kQsaRawDim * 4 + 16384 +
                                                 2 * kRows * KvqPositionBytes(kvl) + 16384);
  }
}

// ---- digests and formats ----

// The SHA-NI path gives the plain digest: FIPS 180-2 vectors and random lengths, in random update pieces.
void Sha256Paths() {
  const auto hex = [](const std::string& text, bool ni) {
    Sha256::UseExtensions() = ni;
    Sha256 hash;
    hash.Update(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    return hash.HexDigest();
  };
  for (const bool ni : {false, true}) {
    assert(hex("", ni) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(hex("abc", ni) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", ni) ==
           "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  }
  std::mt19937 random(3);
  for (int round = 0; round < 60; ++round) {
    std::vector<uint8_t> data(random() % 20000);
    for (auto& b : data) b = uint8_t(random());
    std::array<uint8_t, 32> digests[2];
    for (const bool ni : {false, true}) {
      Sha256::UseExtensions() = ni;
      Sha256 hash;
      for (size_t at = 0; at < data.size();) {
        const size_t n = std::min<size_t>(data.size() - at, random() % 300);
        hash.Update(data.data() + at, n);
        at += n;
      }
      digests[ni] = hash.Digest();
    }
    assert(digests[0] == digests[1]);
  }
  Sha256::UseExtensions() = true;
  std::printf("sha256: extensions %s\n", Sha256::Accelerated() ? "on" : "off");
}

// the digest of a composed snapshot equals the digest of its flat copy.
void ComposedDigestEqualsFlat() {
  const auto base = Make(Tokens(3 * kRows), 2 * kRows, 4);
  std::vector<uint8_t> device = base->bytes.Copy();
  device.back() ^= 0x5a;
  std::vector<SnapshotRange> copies;
  SnapshotBytes composed = SnapshotBytes::Compose({{device.data(), 100}, {device.data() + 100, device.size() - 100,
                                                                         &base->bytes, 100}}, copies);
  CopySnapshotRanges(copies, composed, true);
  assert(composed.Copy() == device);
  SnapshotBytes flat(device.size());
  flat.ForEachPiece(0, device.size(), [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
    std::memcpy(chunk.host + inside, device.data() + at, count);
  });
  const auto plan = PlanBlocks(Geometry(*base), kRows);
  for (const auto& block : plan.blocks) assert(DigestRanges(composed, block.ranges) == DigestRanges(flat, block.ranges));
  assert(DigestRanges(composed, plan.rest) == DigestRanges(flat, plan.rest));
}

// A state file whose ranges miss or overlap one byte is refused: the parser checks the cover itself.
void BadCoverRefused() {
  const auto s = Make(Tokens(2 * kRows + 3), kRows, 5);
  const auto plan = PlanBlocks(Geometry(*s), kRows);
  StateMeta meta;
  meta.key = s->hist;
  meta.policy = Policy();
  std::vector<StateBlockEntry> entries;
  for (const auto& block : plan.blocks) entries.push_back({DigestRanges(s->bytes, block.ranges), RangeBytes(block.ranges), block.ranges});
  const auto build = [&](const std::vector<StreamRange>& rest) {
    return BuildStateHeader(std::string(64, 'a'), meta, *s, kRows, entries, rest,
                            std::vector<uint64_t>(StateChunks(RangeBytes(rest))));
  };
  const auto good = build(plan.rest);
  const auto header = ParseStateHeader(good.data(), good.size());
  assert(header.blocks.size() == plan.blocks.size() && header.rest == plan.rest && header.meta.key == s->hist);
  assert(ParseStateHeader(good.data(), good.size(), false).blocks.size() == plan.blocks.size());
  for (const int change : {-1, +1}) {
    auto rest = plan.rest;
    rest.back().bytes += change;  // the last byte missing, or one byte past the snapshot
    const auto bad = build(rest);
    bool threw = false;
    try { ParseStateHeader(bad.data(), bad.size()); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
  }
  auto overlap = plan.rest;
  overlap[1].offset -= 1;
  const auto bad = build(overlap);
  bool threw = false;
  try { ParseStateHeader(bad.data(), bad.size(), false); } catch (const std::runtime_error&) { threw = true; }
  assert(threw);
}

// ---- the store ----

// Several row counts around the block edges: write, read, every field and byte equal; a restart restores too.
void BlocksRoundTrip() {
  for (const size_t rows : {4 * kRows, 4 * kRows + 1, 5 * kRows - 1, 5 * kRows, 1000 * size_t(1)}) {
    const auto s = Make(Tokens(rows, int(rows)), rows - rows / 3, uint8_t(rows));
    const std::string name = "trip" + std::to_string(rows);
    {
      Disk disk(Options('b', name));
      SpillOk(disk, s);
      assert(disk.Entries() == 1 && disk.Blocks() == rows / kRows + (rows - rows / 3) / kRows);
      assert(disk.Bytes() == DiskBytes(disk.Directory()));
      Restores(disk, *s);
    }
    Disk disk(Options('b', name));
    assert(disk.Entries() == 1 && disk.Blocks() == rows / kRows + (rows - rows / 3) / kRows);
    Restores(disk, *s);
    fs::remove_all(Root() / ".cache" / name);
  }
}

// The restore with blocks equals the restore of the full-state file (block_rows 0) and the saved bytes.
void BlocksEqualWholeFile() {
  const auto s = Make(Tokens(3 * kRows + 17), 2 * kRows + 1, 6);
  auto whole = Options('c', "eq-whole");
  whole.block_rows = 0;
  Disk a(Options('c', "eq-blocks")), b(whole);
  SpillOk(a, s);
  SpillOk(b, s);
  assert(a.Blocks() == 5 && b.Blocks() == 0 && Files(b.Directory(), ".block").empty());
  std::shared_ptr<const TestSnapshot> read[2];
  for (Disk* disk : {&a, &b}) {
    const auto meta = disk->Best(s->hist, Policy(), false, Any);
    assert(StageWait(*disk, *meta).state == StageState::Ready);
    read[disk == &b] = disk->Ready(s->hist, Policy(), false, Any).snapshot;
  }
  Equal(*read[0], *read[1]);
  Equal(*read[0], *s);
}

// B extends A by 2 trunk blocks: B writes those 2 blocks (and its MTP block) and its state file; the ledger and the
// counters agree; A and B both restore. A state whose blocks are all on disk writes only its state file.
void BlocksShared() {
  const auto ids = Tokens(5 * kRows + 9);
  const auto a = Make(Prefix(ids, 3 * kRows + 2), kRows + 5, 7);
  const auto b = Make(ids, 2 * kRows + 1, 8);
  Disk disk(Options('d', "shared"));
  SpillOk(disk, a);
  const uint64_t after_a = disk.DayWritten();
  assert(disk.Blocks() == 4 && after_a == DiskBytes(disk.Directory()));
  SpillOk(disk, b);
  const auto line = Find("result=written");
  assert(Field(line, "blocks") == 7 && Field(line, "new_blocks") == 3 && Field(line, "shared") == 4);
  assert(disk.Blocks() == 7 && disk.Stats().blocks_written == 7 && disk.Stats().blocks_shared == 4);
  const auto b_state = disk.Best(b->hist, Policy(), false, Any);
  uint64_t new_blocks = 0;
  for (const auto& file : Files(disk.Directory(), ".block")) new_blocks += fs::file_size(file);
  new_blocks -= after_a - fs::file_size(disk.Directory() / disk.Best(a->hist, Policy(), false, Any)->name);
  assert(disk.DayWritten() - after_a == new_blocks + b_state->file_bytes && Field(line, "written") == new_blocks + b_state->file_bytes);
  assert(disk.Stats().written_bytes == disk.DayWritten() && disk.Bytes() == DiskBytes(disk.Directory()));
  Restores(disk, *a);
  Restores(disk, *b);
  // C: A's rows and blocks with other tail bytes (another decode result): no new block, one state file.
  auto c_ids = Prefix(ids, 3 * kRows + 2);
  c_ids.back() = 999;
  const auto c = Make(c_ids, kRows + 5, 9);
  const uint64_t before = disk.DayWritten();
  SpillOk(disk, c);
  assert(Field(Find("result=written"), "new_blocks") == 0 && disk.Blocks() == 7);
  assert(disk.DayWritten() - before == disk.Best(c->hist, Policy(), false, Any)->file_bytes);
  Restores(disk, *c);
}

// C has A's tokens and other bytes in block 0: C writes its own block 0, and each restores its own bytes. A name by
// the token key alone would give C A's block.
void SameKeyOtherBytes() {
  const auto ids = Tokens(2 * kRows + 3);
  const auto a = Make(ids, 0, 10);
  const auto other = [](int lane, uint64_t index, uint64_t at) {
    return uint8_t(BlockByte(lane, index, at) + (index == 0 ? 1 : 0));
  };
  auto c = Make(ids, 0, 10, other);
  Disk disk(Options('e', "samekey"));
  SpillOk(disk, a);
  assert(disk.Blocks() == 2);
  Restores(disk, *a);
  auto c_ids = ids;
  c_ids.push_back(5);  // a longer key, so C's state file is new; its block 0 differs from A's
  c = Make(c_ids, 0, 10, other);
  SpillOk(disk, c);
  assert(disk.Blocks() == 3 && Field(Find("result=written"), "new_blocks") == 1);
  Restores(disk, *a);
  Restores(disk, *c);
}

// TenantsNeverShareDisk (W1): the same key and bytes in two tenants, each tenant from the policy: two sets of
// block files; a lookup of tenant 2 never finds tenant 1's state. Train 23's TenantsDoNotShare, with one tenant source.
void TenantsNeverShareDisk() {
  StateTenant one{}, two{};
  one[0] = 1;
  two[31] = 2;
  const auto s = Make(Tokens(2 * kRows + 1), kRows, 11);
  fs::path dir;
  {
    Disk disk(Options('g', "tenants"));
    dir = disk.Directory();
    SpillOk(disk, s, false, one);
    assert(disk.Blocks() == 3 && !disk.Best(s->hist, Policy(two), false, Any) && !disk.Present(s->hist, Policy(two), false));
    assert(!disk.Best(s->hist, Policy(), false, Any));  // the local tenant neither
    SpillOk(disk, s, false, two);
    assert(disk.Blocks() == 6 && disk.Entries() == 2 && Field(Find("result=written"), "new_blocks") == 3);
    const auto first = disk.Best(s->hist, Policy(one), false, Any), second = disk.Best(s->hist, Policy(two), false, Any);
    assert(first && second && first->name != second->name && first->blocks != second->blocks);
    Restores(disk, *s, two);
    // Tenant 1's state is staged; tenant 2's lookup does not see it.
    assert(StageWait(disk, *first).state == StageState::Ready);
    assert(!disk.Ready(s->hist, Policy(two), false, Any).snapshot);
    assert(disk.Ready(s->hist, Policy(one), false, Any).snapshot);
  }
  // A block file whose header names another tenant than its name leaves at start, with its states.
  const auto block = Files(dir, ".block").front();
  std::vector<uint8_t> bytes(4096);
  { std::ifstream in(block, std::ios::binary); in.read(reinterpret_cast<char*>(bytes.data()), 4096); }
  auto header = ParseBlockHeader(bytes.data(), bytes.size());
  header.tenant[5] ^= 1;
  const auto changed = BuildBlockHeader(header);
  { std::fstream out(block, std::ios::in | std::ios::out | std::ios::binary); out.write(reinterpret_cast<const char*>(changed.data()), std::streamsize(changed.size())); }
  Disk disk(Options('g', "tenants"));
  assert(disk.Entries() == 1 && disk.Blocks() == 3 && !fs::exists(block));
  assert(Field(Find("state_disk start"), "broken") == 2 && Field(Find("state_disk start"), "orphans") == 2);
}

StateTenant Tenant() {
  StateTenant tenant{};
  for (size_t i = 0; i < tenant.size(); ++i) tenant[i] = uint8_t(0xa5 ^ (i * 29));
  return tenant;
}
// The compile-time member count of the policy with the tenant is 16 (15 plus the KV fields of a later train, plus
// 1), so a std::array member does not miscount by brace elision. WritePolicy and ReadPolicy round-trip
// it, and two policies that differ only in the tenant are not equal.
void PolicyFieldCount() {
  static_assert(StateFieldCount<ConversationPolicy>() == 16 + StateKvFields<ConversationPolicy>());
  static_assert(std::is_same_v<decltype(ConversationPolicy::tenant), StateTenant>);
  ConversationPolicy policy;
  policy.prefill_rows = 4096;
  policy.draft_selection = {3, 5};
  policy.tenant = Tenant();
  StateBytesOut out;
  WritePolicy(out, policy);
  StateBytesIn in(out.Bytes().data(), out.Bytes().size());
  const ConversationPolicy read = ReadPolicy(in);
  assert(read == policy && read.tenant == Tenant() && in.At() == out.Bytes().size());
  // The tenant comes first, as the raw tenant before the policy in train 23's names and headers.
  const StateTenant tenant = Tenant();
  assert(std::equal(tenant.begin(), tenant.end(), out.Bytes().begin()));
  ConversationPolicy local = policy;
  local.tenant = kLocalTenant;
  assert(!(local == policy) && ConversationPolicy{}.tenant == kLocalTenant);
}

// W1 requirement 6: the local tenant's state and block files (names and bytes) equal train 23's (7b3106e8), so
// the format and its checks are unchanged. The disk identity includes the engine binary (StateDiskIdentityFor), so the
// first start of any new binary still uses a new directory and prefills each session once. A changed file prints its
// digest.
const std::string kLocalFiles23 = "9ba0407d086f443e0952e108ae1ed0267cc363b3aab23bd494597ace277b2c21";
void LocalTenantFilesUnchanged() {
  const auto s = Make(Tokens(2 * kRows + 1), kRows, 11);
  const auto options = Options('h', "local-files");
  {
    Disk disk(options);
    SpillOk(disk, s);
  }
  std::vector<fs::path> files = Files(options.root / DirName('h'), ".state");
  for (const auto& block : Files(options.root / DirName('h'), ".block")) files.push_back(block);
  assert(files.size() == 4);
  Sha256 hash;
  for (const auto& file : files) {
    const std::string name = file.filename().string();
    std::ifstream in(file, std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    hash.Update(reinterpret_cast<const uint8_t*>(name.data()), name.size());
    hash.Update(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
  }
  const std::string digest = hash.HexDigest();
  if (digest != kLocalFiles23) std::printf("local files digest %s\n", digest.c_str());
  std::fflush(stdout);
  assert(digest == kLocalFiles23);
}

// Budget eviction of A keeps B's shared blocks; eviction of B removes its blocks. The index bytes equal the files on
// disk after each step.
void RefsAndEviction() {
  const auto ids = Tokens(4 * kRows);
  const auto a = Make(Prefix(ids, 2 * kRows + 1), 0, 12);
  const auto b = Make(ids, 0, 13);
  const auto other = Make(Tokens(4 * kRows, 500), 0, 14);
  auto options = Options('h', "refs");
  Disk probe(Options('h', "refs-probe"));
  SpillOk(probe, b);
  SpillOk(probe, other);
  options.budget = probe.Bytes() + 4096;  // B and one more like it; with A too, one state must leave
  Disk disk(options);
  SpillOk(disk, a);
  SpillOk(disk, b);
  assert(disk.Entries() == 2 && disk.Blocks() == 4 && disk.Bytes() == DiskBytes(disk.Directory()));
  SpillOk(disk, other);  // A is superseded: it leaves, and its blocks stay for B
  assert(disk.Entries() == 2 && !disk.Present(a->hist, Policy(), false) && disk.Blocks() == 8);
  assert(disk.Bytes() == DiskBytes(disk.Directory()) && disk.Bytes() <= options.budget);
  Restores(disk, *b);
  const auto last = Make(Tokens(4 * kRows, 900), 0, 15);
  SpillOk(disk, last);  // B was read after `other` was written: `other` is the least recently used and leaves
  assert(disk.Present(b->hist, Policy(), false) && !disk.Present(other->hist, Policy(), false));
  assert(disk.Blocks() == 8 && disk.Bytes() == DiskBytes(disk.Directory()));
  const auto fifth = Make(Tokens(4 * kRows, 950), 0, 16);
  SpillOk(disk, fifth);  // B leaves with its 4 blocks
  assert(!disk.Present(b->hist, Policy(), false) && disk.Blocks() == 8 && disk.Bytes() == DiskBytes(disk.Directory()));
  assert(disk.Stats().evicted == 3);
}

// Blocks with no state (a stop between the blocks and the state file) leave at start.
void CrashOrphanBlocks() {
  const auto s = Make(Tokens(3 * kRows), kRows, 17);
  fs::path dir;
  {
    Disk disk(Options('i', "orphans"));
    SpillOk(disk, s);
    dir = disk.Directory();
  }
  fs::remove(Files(dir, ".state").front());
  Disk disk(Options('i', "orphans"));
  assert(disk.Entries() == 0 && disk.Blocks() == 0 && Files(dir, ".block").empty());
  assert(Field(Find("state_disk start"), "orphans") == 4);
}

// A state that lists a removed block leaves at start (broken=1); its other blocks are orphans; other states stay.
void CrashMissingBlock() {
  const auto a = Make(Tokens(2 * kRows), 0, 18), e = Make(Tokens(2 * kRows, 300), 0, 19);
  fs::path dir;
  std::vector<std::string> a_blocks;
  {
    Disk disk(Options('j', "missing"));
    SpillOk(disk, a);
    SpillOk(disk, e);
    a_blocks = disk.Best(a->hist, Policy(), false, Any)->blocks;
    dir = disk.Directory();
  }
  fs::remove(dir / a_blocks[0]);
  Disk disk(Options('j', "missing"));
  assert(disk.Entries() == 1 && disk.Present(e->hist, Policy(), false) && disk.Blocks() == 2);
  const auto line = Find("state_disk start");
  assert(Field(line, "broken") == 1 && Field(line, "orphans") == 1 && !fs::exists(dir / a_blocks[1]));
  Restores(disk, *e);
}

// Temporary files leave at start; a block file cut short leaves with its states.
void CrashTornFiles() {
  const auto a = Make(Tokens(2 * kRows), 0, 20), e = Make(Tokens(2 * kRows, 300), 0, 21);
  fs::path dir;
  std::vector<std::string> a_blocks;
  {
    Disk disk(Options('k', "torn"));
    SpillOk(disk, a);
    SpillOk(disk, e);
    a_blocks = disk.Best(a->hist, Policy(), false, Any)->blocks;
    dir = disk.Directory();
  }
  std::ofstream(dir / (a_blocks[0] + ".tmp.77")) << "partial";
  std::ofstream(dir / "abc.state.tmp.5") << "partial";
  fs::resize_file(dir / a_blocks[1], fs::file_size(dir / a_blocks[1]) - 4096);
  Disk disk(Options('k', "torn"));
  assert(disk.Entries() == 1 && disk.Present(e->hist, Policy(), false));
  assert(!fs::exists(dir / (a_blocks[0] + ".tmp.77")) && !fs::exists(dir / "abc.state.tmp.5"));
  assert(!fs::exists(dir / a_blocks[0]) && !fs::exists(dir / a_blocks[1]));
  assert(disk.Bytes() == DiskBytes(dir));
}

// A block file of another identity, or a valid block of another size under a listed block's name (a size or count
// mismatch), leaves at start with the states that list it; a block that another state uses under its own name stays.
void ForeignBlockHeader() {
  const auto ids = Tokens(3 * kRows);
  const auto a = Make(Prefix(ids, 2 * kRows), 0, 22), b = Make(ids, 0, 23);
  const auto e = Make(Tokens(2 * kRows, 300), 0, 24);
  fs::path dir;
  std::vector<std::string> a_blocks, b_blocks, e_blocks;
  {
    Disk disk(Options('l', "foreign"));
    for (const auto& s : {a, b, e}) SpillOk(disk, s);
    a_blocks = disk.Best(a->hist, Policy(), false, Any)->blocks;
    b_blocks = disk.Best(b->hist, Policy(), false, Any)->blocks;
    e_blocks = disk.Best(e->hist, Policy(), false, Any)->blocks;
    dir = disk.Directory();
  }
  assert(b_blocks.size() == 3 && b_blocks[0] == a_blocks[0] && b_blocks[2] != a_blocks[1]);
  // Another identity in the header of b's third block (its own name).
  std::vector<uint8_t> bytes(4096);
  { std::ifstream in(dir / b_blocks[2], std::ios::binary); in.read(reinterpret_cast<char*>(bytes.data()), 4096); }
  auto header = ParseBlockHeader(bytes.data(), bytes.size());
  header.identity = std::string(64, 'z');
  const auto changed = BuildBlockHeader(header);
  { std::fstream out(dir / b_blocks[2], std::ios::in | std::ios::out | std::ios::binary); out.write(reinterpret_cast<const char*>(changed.data()), std::streamsize(changed.size())); }
  // E's second block replaced by a valid block file of another size and digest.
  fs::copy_file(dir / b_blocks[1], dir / e_blocks[1], fs::copy_options::overwrite_existing);
  Disk disk(Options('l', "foreign"));
  assert(disk.Entries() == 1 && disk.Present(a->hist, Policy(), false));
  assert(!fs::exists(dir / b_blocks[2]) && !fs::exists(dir / e_blocks[1]) && !fs::exists(dir / e_blocks[0]));
  assert(fs::exists(dir / a_blocks[0]) && fs::exists(dir / a_blocks[1]) && disk.Blocks() == 2);
  assert(disk.Bytes() == DiskBytes(dir));
  Restores(disk, *a);
}

// One changed byte in a shared block: the read fails (checksum), the block and every state that lists it leave.
void DamagedBlockAtRead() {
  const auto ids = Tokens(3 * kRows);
  const auto a = Make(Prefix(ids, 2 * kRows), 0, 25), b = Make(ids, 0, 26), e = Make(Tokens(2 * kRows, 300), 0, 27);
  Disk disk(Options('m', "damaged"));
  for (const auto& s : {a, b, e}) SpillOk(disk, s);
  const auto meta = disk.Best(a->hist, Policy(), false, Any);
  Corrupt(disk.Directory() / meta->blocks[0], HeaderBytes(disk.Directory() / meta->blocks[0]) + 1000);
  const auto status = StageWait(disk, *meta);
  assert(status.state == StageState::Failed && status.reason == "checksum");
  assert(disk.Entries() == 1 && disk.Present(e->hist, Policy(), false) && !fs::exists(disk.Directory() / meta->blocks[0]));
  assert(!disk.Ready(a->hist, Policy(), false, Any).snapshot && disk.Blocks() == 2);
  assert(disk.Bytes() == DiskBytes(disk.Directory()));
  // A changed byte in a state file's rest leaves only that state.
  const auto e_meta = disk.Best(e->hist, Policy(), false, Any);
  SpillOk(disk, b);
  Corrupt(disk.Directory() / e_meta->name, HeaderBytes(disk.Directory() / e_meta->name) + 100);
  assert(StageWait(disk, *e_meta).reason == "checksum" && disk.Entries() == 1 && disk.Blocks() == 3);
  Restores(disk, *b);
}

// An eviction during a read does not remove its blocks: the read holds the state file and pins the blocks. The
// blocks leave after the read.
void ReadPinsBlocks() {
  const auto a = Make(Tokens(2 * kRows), kRows, 28), d = Make(Tokens(2 * kRows, 300), kRows, 29);
  auto options = Options('n', "pins");
  std::atomic<bool> once = true;
  Disk* self = nullptr;
  options.before_read = [&] {
    if (!once.exchange(false)) return;
    assert(Spill(*self, d).empty() && self->Drain(60));  // needs A's state file: A leaves during the read
  };
  Disk probe(Options('n', "pins-probe"));
  SpillOk(probe, a);
  options.budget = 2 * probe.Bytes() - 4096;
  Disk disk(options);
  self = &disk;
  SpillOk(disk, a);
  const auto meta = disk.Best(a->hist, Policy(), false, Any);
  assert(StageWait(disk, *meta).state == StageState::Ready);
  assert(!disk.Present(a->hist, Policy(), false) && disk.Present(d->hist, Policy(), false));
  const auto ready = disk.Ready(a->hist, Policy(), false, Any);
  assert(ready.snapshot);
  Equal(*ready.snapshot, *a);
  for (const auto& block : meta->blocks) assert(!fs::exists(disk.Directory() / block));
  assert(disk.Blocks() == 3 && disk.Bytes() == DiskBytes(disk.Directory()));
}

// A write hook moves the gather of each new block after the hash: the spill is refused (digest), and no block and
// no state stay.
void WriteDigestChecked() {
  auto options = Options('o', "digest");
  options.write_ranges = [](std::vector<StreamRange>& ranges) { std::rotate(ranges.begin(), ranges.begin() + 1, ranges.end()); };
  Disk disk(options);
  const auto s = Make(Tokens(2 * kRows + 4), 0, 30);
  assert(Spill(disk, s).empty() && disk.Drain(60));
  assert(disk.Entries() == 0 && disk.Blocks() == 0 && logs.Has("result=refused reason=digest"));
  for (const auto& item : fs::directory_iterator(disk.Directory())) assert(item.path().filename() == ".lock");
}

// A and B share a block; a read of A finds it bad while a spill of C pins it: A and B leave, C is refused (damaged),
// and the block leaves after the last pin.
void BadBlockWhileShared() {
  const auto ids = Tokens(4 * kRows);
  const auto a = Make(Prefix(ids, kRows + 3), 0, 31), b = Make(Prefix(ids, 2 * kRows + 3), 0, 32);
  const auto c = Make(ids, 0, 33);
  auto options = Options('p', "badshared");
  Disk* self = nullptr;
  std::string shared;
  std::atomic<bool> armed = false;
  options.after_blocks = [&] {  // C's new blocks are on disk; its pin holds A's block 0
    if (!armed) return;
    const auto meta = self->Best(a->hist, Policy(), false, Any);
    Corrupt(self->Directory() / shared, HeaderBytes(self->Directory() / shared) + 10);
    assert(StageWait(*self, *meta).reason == "checksum");
    assert(self->Entries() == 0 && fs::exists(self->Directory() / shared));  // pinned: the file stays
  };
  Disk disk(options);
  self = &disk;
  SpillOk(disk, a);
  SpillOk(disk, b);
  shared = disk.Best(a->hist, Policy(), false, Any)->blocks[0];
  armed = true;
  assert(Spill(disk, c).empty() && disk.Drain(60));
  assert(logs.Has("result=refused reason=damaged") && disk.Entries() == 0 && disk.Blocks() == 0);
  for (const auto& item : fs::directory_iterator(disk.Directory())) assert(item.path().filename() == ".lock");
}

// Two equal blocks in one state: one block file, two references, a correct restore, and a clean removal.
void DuplicateBlockInState() {
  const Fill same = [](int lane, uint64_t, uint64_t at) { return uint8_t(at * 3 + lane + 1); };
  const auto s = Make(Tokens(3 * kRows), 0, 34, same, {{3, 0}});
  auto options = Options('q', "dup");
  {
    Disk disk(options);
    SpillOk(disk, s);
    const auto meta = disk.Best(s->hist, Policy(), false, Any);
    assert(meta->blocks.size() == 3 && meta->blocks[0] == meta->blocks[1] && disk.Blocks() == 1);
    assert(Field(Find("result=written"), "new_blocks") == 1 && disk.Bytes() == DiskBytes(disk.Directory()));
    Restores(disk, *s);
  }
  options.budget = 4096;  // the restart evicts the state: both references go, and the block with them
  Disk disk(options);
  assert(disk.Entries() == 0 && disk.Blocks() == 0 && Files(disk.Directory(), ".block").empty());
}

// 6 turns of one session: the newest 2 older states stay; a shared state and a state with a read in progress stay,
// and that read succeeds.
void PruneBranch() {
  const auto ids = Tokens(7 * kRows);
  std::vector<std::shared_ptr<TestSnapshot>> turns;
  for (size_t k = 1; k <= 6; ++k) turns.push_back(Make(Prefix(ids, k * kRows + 1), 0, uint8_t(40 + k)));
  {
    Disk disk(Options('r', "prune"));
    for (size_t k = 0; k < 5; ++k) SpillOk(disk, turns[k]);
    assert(disk.Entries() == 3 && disk.Stats().pruned == 2 && Field(Find("result=written"), "pruned") == 1);
    assert(!disk.Present(turns[0]->hist, Policy(), false) && !disk.Present(turns[1]->hist, Policy(), false));
    assert(disk.Blocks() == 5 && disk.Bytes() == DiskBytes(disk.Directory()));
    Restores(disk, *turns[2]);
  }
  auto options = Options('r', "prune2");
  std::atomic<bool> once = true;
  Disk* self = nullptr;
  options.before_read = [&] {
    if (!once.exchange(false)) return;
    for (size_t k = 2; k < 6; ++k) assert(Spill(*self, turns[k]).empty() && self->Drain(60));
  };
  Disk disk(options);
  self = &disk;
  SpillOk(disk, turns[0]);
  SpillOk(disk, turns[1], true);
  const auto meta = disk.Best(turns[0]->hist, Policy(), false, Any);
  assert(StageWait(disk, *meta).state == StageState::Ready);  // turns 3 to 6 spill during the read
  assert(disk.Present(turns[0]->hist, Policy(), false) && disk.Present(turns[1]->hist, Policy(), false));
  assert(!disk.Present(turns[2]->hist, Policy(), false) && disk.Entries() == 5);
  const auto ready = disk.Ready(turns[0]->hist, Policy(), false, Any);
  Equal(*ready.snapshot, *turns[0]);
}

// The state file limit (max_states): the least recently used leaves.
void StateFileLimit() {
  auto options = Options('s', "limit");
  options.max_states = 8;
  Disk disk(options);
  for (int k = 0; k < 9; ++k) SpillOk(disk, Make(Tokens(kRows + 1, 100 * k), 0, uint8_t(k)));
  assert(disk.Entries() == 8 && !disk.Present(Tokens(kRows + 1, 0), Policy(), false));
  assert(disk.Present(Tokens(kRows + 1, 800), Policy(), false) && disk.Bytes() == DiskBytes(disk.Directory()));
}

// A stop after the last new block: the state file is written, and the next start restores it.
void StopFinishesState() {
  auto options = Options('t', "stop");
  std::promise<void> blocks_done;
  options.after_blocks = [&] {
    blocks_done.set_value();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));  // the destructor sets the stop flag meanwhile
  };
  const auto s = Make(Tokens(3 * kRows), kRows, 50);
  {
    Disk disk(options);
    assert(Spill(disk, s).empty());
    blocks_done.get_future().wait();
  }
  options.after_blocks = nullptr;
  Disk disk(options);
  assert(disk.Entries() == 1 && disk.Blocks() == 4);
  Restores(disk, *s);
}

// The ledger counts each block at its rename: an I/O error in the third of 3 new blocks leaves the
// first 2 in the ledger, and no file.
void LedgerPerBlock() {
  auto options = Options('u', "ledger");
  options.io_threads = 1;
  std::atomic<int> calls = 0;
  options.io_fault = [&](const void*) { return ++calls == 3 ? EIO : 0; };
  Disk disk(options);
  const auto s = Make(Tokens(3 * kRows), 0, 51, nullptr, {{3, 0}});
  assert(Spill(disk, s).empty() && disk.Drain(60));
  assert(logs.Has("result=refused reason=io errno=5") && disk.Entries() == 0 && disk.Blocks() == 0);
  const auto plan = PlanBlocks(Geometry(*s), kRows);
  BlockHeader header{options.identity, kLocalTenant, {}, RangeBytes(plan.blocks[0].ranges), 0,
                     std::vector<uint64_t>(StateChunks(RangeBytes(plan.blocks[0].ranges)))};
  const uint64_t block_file = BuildBlockHeader(header).size() + StateRoundUp(header.payload);
  assert(disk.DayWritten() == 2 * block_file && disk.Stats().written_bytes == 2 * block_file);
  for (const auto& item : fs::directory_iterator(disk.Directory())) assert(item.path().filename() == ".lock");
}

// The queue-time daily check subtracts the full blocks of the longest state on disk that the new state extends
// with a budget below B's whole size, B still queues and writes.
void QueueEstimate() {
  const auto ids = Tokens(4 * kRows);
  const auto a = Make(Prefix(ids, 3 * kRows + 1), 0, 52), b = Make(ids, 0, 53);
  auto options = Options('v', "estimate");
  options.today = [] { return std::string("2026-10-17"); };
  Disk probe(Options('v', "estimate-probe"));
  SpillOk(probe, a);
  options.write_per_day = probe.DayWritten() + b->bytes.size();
  Disk disk(options);
  SpillOk(disk, a);
  assert(disk.DayWritten() == probe.DayWritten());
  assert(Spill(disk, b).empty() && disk.Drain(60) && disk.Present(b->hist, Policy(), false));
  // Without A on disk the whole size counts: an unrelated state of B's size is refused.
  assert(Spill(disk, Make(Tokens(4 * kRows, 300), 0, 54)) == "daily");
}

// After spills, reads, evictions, damage and a stop, the directory holds only the names that train 22 removes.
void OnlyTrain22NamesStay() {
  for (const auto& item : fs::recursive_directory_iterator(Root() / ".cache"))
    if (item.is_directory() && item.path().filename().string().size() == 50) OnlyTrain22Names(item.path());
}

// No token, no key byte and no tenant byte in any line: fixed words and numbers only.
void LogsHoldNoTokens() {
  const std::regex safe(R"(state_disk [a-z]+( [a-z_]+=[a-z0-9_.\-]+)*)");
  std::lock_guard lock(logs.mutex);
  assert(logs.lines.size() > 50);
  for (const auto& line : logs.lines) {
    if (!std::regex_match(line, safe)) std::printf("bad log line: %s\n", line.c_str());
    assert(std::regex_match(line, safe));
  }
}

// TenantNotInLogs (W1, design section 9.4): a tenant's spill, read and restore write no tenant byte to a log
// line or a file name, in hex (either case) or raw. Runs before LogsHoldNoTokens checks the pattern of every line.
void TenantNotInLogs() {
  const StateTenant tenant = Tenant();
  std::string lower, upper, raw(tenant.begin(), tenant.end());
  for (const uint8_t byte : tenant) {
    char two[3];
    std::snprintf(two, sizeof(two), "%02x", byte);
    lower += two;
    std::snprintf(two, sizeof(two), "%02X", byte);
    upper += two;
  }
  const auto s = Make(Tokens(2 * kRows + 1, 5), kRows, 31);
  fs::path dir;
  {
    Disk disk(Options('t', "tenant-logs"));
    dir = disk.Directory();
    SpillOk(disk, s, false, tenant);
    Restores(disk, *s, tenant);
  }
  Disk disk(Options('t', "tenant-logs"));  // the start's index lines too
  assert(disk.Entries() == 1);
  std::vector<std::string> names;
  for (const auto& item : fs::recursive_directory_iterator(Root())) names.push_back(item.path().string());
  std::lock_guard lock(logs.mutex);
  for (const auto& texts : {std::cref(logs.lines), std::cref(names)})
    for (const auto& text : texts.get())
      for (const auto& form : {lower, upper, raw, lower.substr(0, 16), upper.substr(0, 16)})
        assert(text.find(form) == std::string::npos);
}

// Bench, not a gate: Rebuild time and the longest lock wait of a lookup with many states (IndexScale). The states
// share one block, so the files stay small; the keys have the real length.
void IndexScale() {
  const size_t states = std::getenv("STATE_BLOCKS_SCALE") ? 1000 : 100, tokens = 100000;
  auto options = Options('w', "scale");
  options.max_states = 2 * states;
  options.min_tokens = 1;
  options.block_rows = 0;  // the bench measures the index, not the blocks
  const auto start = std::chrono::steady_clock::now();
  {
    Disk disk(options);
    for (size_t k = 0; k < states; ++k) {
      auto s = std::make_shared<TestSnapshot>();
      s->hist = Tokens(tokens, int(k));
      s->resume.length = tokens;
      s->bytes = SnapshotBytes(4096);
      assert(disk.Spill(s->hist, Policy(), false, false, false, s, *s).empty());
      if (k % 32 == 31) assert(disk.Drain(120));
    }
    assert(disk.Drain(120) && disk.Entries() == states);
  }
  const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  Disk disk(options);
  assert(disk.Entries() == states);
  const auto line = Find("state_disk start");
  const auto key = Tokens(tokens, 3);
  double worst = 0;
  for (int i = 0; i < 50; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    disk.Best(key, Policy(), false, Any);
    worst = std::max(worst, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::printf("index scale: %zu states of %zu tokens, write %.1f s, %s, worst Best %.2f ms\n", states, tokens, write_s,
              line.substr(line.find("rebuild_ms")).substr(0, 24).c_str(), worst);
  fs::remove_all(options.root);
}

int main() {
  PlanCover();
  PlanSpansMatchSave();
  PlanWholeOnMismatch();
  PlanEachKvLayout();
  Sha256Paths();
  ComposedDigestEqualsFlat();
  BadCoverRefused();
  BlocksRoundTrip();
  BlocksEqualWholeFile();
  BlocksShared();
  SameKeyOtherBytes();
  PolicyFieldCount();
  TenantsNeverShareDisk();
  LocalTenantFilesUnchanged();
  RefsAndEviction();
  CrashOrphanBlocks();
  CrashMissingBlock();
  CrashTornFiles();
  ForeignBlockHeader();
  DamagedBlockAtRead();
  ReadPinsBlocks();
  WriteDigestChecked();
  BadBlockWhileShared();
  DuplicateBlockInState();
  PruneBranch();
  StateFileLimit();
  StopFinishesState();
  LedgerPerBlock();
  QueueEstimate();
  OnlyTrain22NamesStay();
  TenantNotInLogs();
  LogsHoldNoTokens();
  IndexScale();
  fs::remove_all(Root().parent_path());
  std::puts("state_blocks_test passed");
}
