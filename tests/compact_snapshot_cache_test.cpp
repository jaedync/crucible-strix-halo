// The real cache and snapshot code must preserve pinned compact rings at every wrap boundary.
#include "../engine/model/flashnext/geometry.h"
#include "../engine/model/flashnext/fixed_state.h"
#include "../engine/runtime/retained_prefix.h"
#include "../engine/runtime/bounds.h"
#include "../engine/layers/cache_storage.h"
#include "../engine/runtime/conversation_cache.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace crucible;
using __half = uint16_t;
constexpr int kPend = 258;
constexpr size_t QsaPoolFloats(int n) { return size_t((n + 127) / 128) * 4096; }
struct DeviceSpan { const void* pointer; size_t bytes; };
constexpr int hipMemcpyDeviceToHost = 0, hipMemcpyHostToDevice = 1;
std::map<uintptr_t, size_t> pinned;

bool IsPinned(const void* pointer, size_t count) {
  const auto address = uintptr_t(pointer);
  for (const auto& [base, bytes] : pinned)
    if (address >= base && address - base <= bytes && count <= bytes - (address - base)) return true;
  return false;
}
template<class T> int hipHostMalloc(T** output, size_t bytes) {
  *output = static_cast<T*>(std::malloc(bytes));
  if (!*output) return 1;
  pinned[uintptr_t(*output)] = bytes;
  return 0;
}
int hipHostFree(void* pointer) {
  assert(pinned.erase(uintptr_t(pointer)) == 1);
  std::free(pointer);
  return 0;
}
[[gnu::noinline]] int hipMemcpy(void* output, const void* input, size_t bytes, int direction) {
  assert(IsPinned(direction == hipMemcpyDeviceToHost ? output : input, bytes));
  std::memcpy(output, input, bytes);
  return 0;
}
// Save fences the default stream since 070d45e; the stub copies synchronously.
int hipStreamSynchronize(int stream) { assert(stream == 0); return 0; }
#define HIP_OK(expression) assert((expression) == 0)
struct QsaCache { __half* k; __half* v; float* ik_raw; float* ik_pool; int raw_slots = 0; };
struct CacheValues {
  std::vector<__half> k, v;
  std::vector<float> raw, pool;
  void Allocate(int capacity, bool compact) {
    k.resize(size_t(capacity) * 512); v.resize(k.size());
    raw.resize(size_t(compact ? kQsaRawSlots : capacity) * 128);
    pool.resize(QsaPoolFloats(capacity));
  }
  QsaCache Descriptor(bool compact) {
    return {k.data(), v.data(), raw.data(), pool.data(), compact ? kQsaRawSlots : 0};
  }
};
struct SequenceState {
  const int capacity;
  const int mtp_capacity = capacity;
  const bool compact_state;
  std::vector<int> hist;
  RetainedPrefix retained;
  std::map<int, float*> gstate, ghist, gpend;
  bool gdn_lazy = false;  // as engine/runtime/sequence_state.h
  std::map<int, QsaCache> qc;
  // As engine/runtime/sequence_state.h: the one list of the fixed-size buffers.
  std::vector<DeviceSpan> FixedSpans() const {
    std::vector<DeviceSpan> spans;
    crucible::VisitFixedState(*this, sizeof(float) * 48 * kMaxRows * kPend,
                              [&](const void* pointer, size_t bytes) { spans.push_back({pointer, bytes}); });
    return spans;
  }
  std::vector<float> ple = std::vector<float>(16 * kWide);
  float* ple_ring = ple.data();
  CacheValues trunk, mtp;
  explicit SequenceState(int cap, bool with_mtp = true)
      : capacity(cap), compact_state(CompactStateFlag(std::getenv("CRUCIBLE_COMPACT_STATE"))) {
    trunk.Allocate(cap, compact_state); qc[3] = trunk.Descriptor(compact_state);
    if (with_mtp) AddMtp();
  }
  void AddMtp() { mtp.Allocate(capacity, compact_state); qc[kLayers] = mtp.Descriptor(compact_state); }
  void Reset() { hist.clear(); retained.Reset(); std::fill(ple.begin(), ple.end(), 0); }
};
struct Engine {
  std::vector<std::unique_ptr<SequenceState>> sequences_;
  std::vector<float> logits = std::vector<float>(kVocab);
  float* logits_ = logits.data();
  bool fail_mtp = false;
  void EnsureMtpCache(SequenceState& state, int position) {
    if (fail_mtp || position != 0) throw std::runtime_error("MTP allocation failed");
    state.AddMtp();
  }
};
#ifndef CRUCIBLE_SNAPSHOT_HEADER
#define CRUCIBLE_SNAPSHOT_HEADER "../engine/runtime/conversation_snapshot.h"
#endif
#include CRUCIBLE_SNAPSHOT_HEADER

bool ChunksPinned(const SnapshotBytes& bytes) {
  return std::all_of(bytes.Chunks().begin(), bytes.Chunks().end(), [](const SnapshotChunk& chunk) {
    return IsPinned(chunk.host, SnapshotChunkPool::kChunkBytes);
  });
}

size_t ExpectedSnapshotBytes(size_t length, int mpos) {
  const auto cache = [](size_t n) {
    return n ? n * 2048 + kQsaRawSlots * 128 * sizeof(float) + QsaPoolFloats(int(n)) * sizeof(float) : 0;
  };
  return (16 + kMaxRows + 2) * kWide * sizeof(float) + cache(length) + cache(size_t(mpos));
}
void Fill(CacheValues& values, uint32_t salt) {
  std::fill(values.k.begin(), values.k.end(), __half(salt));
  std::fill(values.v.begin(), values.v.end(), __half(salt + 1));
  for (size_t i = 0; i < values.raw.size(); ++i)
    values.raw[i] = std::bit_cast<float>(i % 17 ? 0x3f000000u + salt + uint32_t(i) : 0x80000000u);
  std::fill(values.pool.begin(), values.pool.end(), float(salt + 2));
}
template<class F> void Reject(F function) {
  bool rejected = false;
  try { function(); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
}

void CacheRoundTrip(size_t length, bool with_mtp) {
  Engine engine;
  engine.sequences_.push_back(std::make_unique<SequenceState>(80, with_mtp));
  engine.sequences_.push_back(std::make_unique<SequenceState>(96, false));
  auto& source = *engine.sequences_[0];
  auto& destination = *engine.sequences_[1];
  assert(source.compact_state && source.trunk.raw.size() == 16 * 128);
  source.hist.resize(length, 7); assert(source.retained.Forward(int(length), 0, int(length)).empty());
  Fill(source.trunk, 11); if (with_mtp) Fill(source.mtp, 23);
  source.ple[0] = 41;
  std::vector<float> stack((kMaxRows + 1) * kWide, 31), carry(kWide, 32);
  ConversationResume resume;
  resume.length = length; resume.mpos = with_mtp ? int(length - 1) : 0;
  resume.next_id = 17; resume.logits.resize(kVocab, 2);
  const size_t bytes = ConversationStateIO::Bytes(engine, source, resume, stack.data(), carry.data());
  PrefixCache<ConversationSnapshot, ConversationPolicy> cache(bytes);
  assert(cache.Reserve(bytes));
  const auto snapshot = ConversationStateIO::Save(engine, source, resume, stack.data(), carry.data());
  assert(snapshot->bytes.size() == ExpectedSnapshotBytes(length, resume.mpos));
  assert(ChunksPinned(snapshot->bytes));
  const auto expected_trunk = source.trunk.raw, expected_mtp = source.mtp.raw;
  const ConversationPolicy policy{.spec = with_mtp ? 7 : 0, .draft_selection = {}};
  assert(cache.Insert(source.hist, policy, bytes, snapshot));
  std::fill(source.trunk.raw.begin(), source.trunk.raw.end(), 99);
  auto ids = source.hist; ids.push_back(29);
  const auto found = cache.Find(ids, policy);
  assert(found == snapshot && cache.Bytes() == bytes);
  const auto restored = ConversationStateIO::Restore(engine, destination, *found, stack.data(), carry.data());
  assert(restored.length == length && restored.mpos == resume.mpos && destination.hist == snapshot->hist);
  assert(std::memcmp(destination.trunk.raw.data(), expected_trunk.data(), 16 * 128 * sizeof(float)) == 0);
  if (resume.mpos)
    assert(std::memcmp(destination.mtp.raw.data(), expected_mtp.data(), 16 * 128 * sizeof(float)) == 0);
  assert(destination.qc.contains(kLayers) == with_mtp && destination.ple[0] == 41);
  const auto saved_again = ConversationStateIO::Save(engine, destination, restored, stack.data(), carry.data());
  assert(saved_again->bytes == snapshot->bytes && ChunksPinned(saved_again->bytes));
  assert(source.trunk.raw.front() == 99 && stack[0] == 31 && carry[0] == 32 && engine.logits[0] == 2);
}

void CacheRejectsIncompatibleRestore() {
  Engine engine;
  engine.sequences_.push_back(std::make_unique<SequenceState>(80));
  engine.sequences_.push_back(std::make_unique<SequenceState>(80, false));
  auto& source = *engine.sequences_[0];
  auto& destination = *engine.sequences_[1];
  source.hist.resize(49, 7); Fill(source.trunk, 11); Fill(source.mtp, 23);
  std::vector<float> stack((kMaxRows + 1) * kWide, 31), carry(kWide, 32);
  ConversationResume resume; resume.length = 49; resume.mpos = 45;
  const auto snapshot = ConversationStateIO::Save(engine, source, resume, stack.data(), carry.data());
  const size_t bytes = ConversationStateIO::Bytes(engine, source, resume, stack.data(), carry.data());
  PrefixCache<ConversationSnapshot> cache(bytes);
  assert(cache.Insert(source.hist, 0, bytes, snapshot));
  const auto found = cache.Find(source.hist, 0);
  destination.hist = {91}; destination.ple[0] = 92;
  auto truncated = *found; truncated.bytes.pop_back();
  Reject([&] { ConversationStateIO::Restore(engine, destination, truncated, stack.data(), carry.data()); });
  assert(destination.hist == std::vector<int>{91} && destination.ple[0] == 92 && !destination.qc.contains(kLayers));
  engine.fail_mtp = true;
  Reject([&] { ConversationStateIO::Restore(engine, destination, *found, stack.data(), carry.data()); });
  assert(destination.hist == std::vector<int>{91} && destination.ple[0] == 92 && !destination.qc.contains(kLayers));
  engine.fail_mtp = false;
  destination.qc[3].raw_slots = 0;
  Reject([&] { ConversationStateIO::Restore(engine, destination, *found, stack.data(), carry.data()); });
  assert(destination.hist == std::vector<int>{91} && destination.ple[0] == 92 && !destination.qc.contains(kLayers));
}

int main() {
  assert(setenv("CRUCIBLE_COMPACT_STATE", "1", 1) == 0);
  for (size_t length : {49, 1, 3, 16, 17, 33, 65}) {
    CacheRoundTrip(length, false);
    CacheRoundTrip(length, true);
    assert(pinned.empty());
  }
  CacheRejectsIncompatibleRestore();
  assert(pinned.empty());
}
