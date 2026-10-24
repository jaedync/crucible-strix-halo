// an incremental save (ConversationStateIO::Save with a base) gives the same bytes as a full save, and a
// restore of it writes the same device bytes. Fake device memory, as compact_snapshot_cache_test.
#include "../engine/model/flashnext/geometry.h"
#include "../engine/model/flashnext/fixed_state.h"
#include "../engine/runtime/retained_prefix.h"
#include "../engine/runtime/bounds.h"
#include "../engine/layers/cache_storage.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
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
int copies = 0;
size_t copied = 0;
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
int hipHostFree(void* pointer) { assert(pinned.erase(uintptr_t(pointer)) == 1); std::free(pointer); return 0; }
int hipMemcpy(void* output, const void* input, size_t bytes, int direction) {
  assert(IsPinned(direction == hipMemcpyDeviceToHost ? output : input, bytes));
  std::memcpy(output, input, bytes);
  ++copies;
  if (direction == hipMemcpyDeviceToHost) copied += bytes;
  return 0;
}
int hipStreamSynchronize(int stream) { assert(stream == 0); return 0; }
// Slot checkpoint buffers (slot_checkpoint_io.h): fake device memory.
constexpr int hipMemcpyDeviceToDevice = 2;
int hipMalloc(uint8_t** output, size_t bytes) { *output = static_cast<uint8_t*>(std::malloc(bytes)); return *output ? 0 : 1; }
int hipFree(void* pointer) { std::free(pointer); return 0; }
int hipMemcpyAsync(void* output, const void* input, size_t bytes, int direction, int stream) {
  assert(direction == hipMemcpyDeviceToDevice && stream == 0);
  std::memcpy(output, input, bytes);
  return 0;
}
#define HIP_OK(expression) assert((expression) == 0)
struct QsaCache { __half* k; __half* v; float* ik_raw; float* ik_pool; int raw_slots = 0; };
struct Layer {
  std::vector<__half> k, v;
  std::vector<float> raw, pool;
  QsaCache Allocate(int capacity, bool compact) {
    k.resize(size_t(capacity) * 512); v.resize(k.size());
    raw.resize(size_t(compact ? kQsaRawSlots : capacity) * 128);
    pool.resize(QsaPoolFloats(capacity));
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
  std::map<int, QsaCache> qc;
  std::vector<DeviceSpan> FixedSpans() const {
    std::vector<DeviceSpan> spans;
    crucible::VisitFixedState(*this, sizeof(float) * 48 * kMaxRows * kPend,
                              [&](const void* pointer, size_t bytes) { spans.push_back({pointer, bytes}); });
    return spans;
  }
  std::vector<float> ple = std::vector<float>(16 * kWide);
  float* ple_ring = ple.data();
  // One GDN layer (state, history, pending rows): the fixed spans that a checkpoint buffer holds besides the PLE ring.
  std::vector<float> gdn_state = std::vector<float>(kGdnStateBytes / sizeof(float));
  std::vector<float> gdn_hist = std::vector<float>(kGdnHistBytes / sizeof(float));
  std::vector<float> gdn_pend = std::vector<float>(size_t(48) * kMaxRows * kPend);
  std::map<int, Layer> layers;
  SequenceState(int cap, bool compact, bool mtp) : capacity(cap), compact_state(compact) {
    for (int l : {3, 7}) qc[l] = layers[l].Allocate(cap, compact);
    if (mtp) qc[kLayers] = layers[kLayers].Allocate(cap, compact);
    gstate[0] = gdn_state.data();
    ghist[0] = gdn_hist.data();
    gpend[0] = gdn_pend.data();
  }
  // As the engine's Reset: the GDN state and history and the PLE ring go to zero; the pending rows stay.
  void Reset() {
    hist.clear();
    retained.Reset();
    std::fill(ple.begin(), ple.end(), 0);
    std::fill(gdn_state.begin(), gdn_state.end(), 0);
    std::fill(gdn_hist.begin(), gdn_hist.end(), 0);
  }
};
struct Engine {
  std::vector<std::unique_ptr<SequenceState>> sequences_;
  std::vector<float> logits = std::vector<float>(kVocab);
  float* logits_ = logits.data();
  void EnsureMtpCache(SequenceState& state, int) { state.qc[kLayers] = state.layers[kLayers].Allocate(state.capacity, state.compact_state); }
};
#include "../engine/runtime/conversation_snapshot.h"
#include "../engine/serve/batch_conversation_cache.h"
#include "../engine/serve/slot_base.h"
#include "../engine/runtime/slot_checkpoint_io.h"

// Writes rows [from, to) of every QSA layer, the pool blocks that hold them, the compact rings and the fixed state,
// as a forward of those rows does.
void Forward(SequenceState& s, size_t from, size_t to, uint32_t salt) {
  for (auto& [l, layer] : s.layers) {
    for (size_t row = from; row < to; ++row)
      for (size_t i = 0; i < 512; ++i) {
        layer.k[row * 512 + i] = __half(row * 31 + i + salt + l);
        layer.v[row * 512 + i] = __half(row * 17 + i + salt + l);
      }
    if (s.compact_state) {
      for (size_t i = 0; i < layer.raw.size(); ++i) layer.raw[i] = float(salt + i % 97 + to);
    } else {
      for (size_t row = from; row < to; ++row)
        for (size_t i = 0; i < 128; ++i) layer.raw[row * 128 + i] = float(row + i + salt);
    }
    if (to > from)
      for (size_t block = from / 128; block <= (to - 1) / 128; ++block)
        for (size_t i = 0; i < 4096; ++i) layer.pool[block * 4096 + i] = float(block * 7 + i + salt + to);
  }
  std::fill(s.ple.begin(), s.ple.end(), float(salt + to));
  for (auto* gdn : {&s.gdn_state, &s.gdn_hist, &s.gdn_pend})
    for (size_t i = 0; i < gdn->size(); ++i) (*gdn)[i] = float(salt + to + i % 89);
  s.hist.resize(to, int(salt % 1000));
  assert(s.retained.Forward(int(to - from), 0, int(to - from)).empty());
}
ConversationResume Resume(size_t length, bool mtp, int replay = 0) {
  ConversationResume r;
  r.length = length;
  r.mpos = mtp ? int(length) : 0;
  r.replay = replay;
  r.next_id = 3;
  r.logits.resize(kVocab, 1);
  return r;
}
std::vector<__half> Device(const SequenceState& s) {
  std::vector<__half> out;
  for (const auto& [l, layer] : s.layers) {
    out.insert(out.end(), layer.k.begin(), layer.k.end());
    out.insert(out.end(), layer.v.begin(), layer.v.end());
  }
  return out;
}

void SameAsFullSave(bool compact, bool mtp, int replay) {
  constexpr int kRows = 24576;  // 24 MB of K per layer: three whole chunks can be shared
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(kRows, compact, mtp));
  e.sequences_.push_back(std::make_unique<SequenceState>(kRows, compact, mtp));
  SequenceState& s = *e.sequences_[0];
  std::vector<float> stk((kMaxRows + 1) * kWide, 5), carry(kWide, 6);
  const size_t n1 = 12000, n2 = 20000;
  Forward(s, 0, n1, 1);
  auto base = ConversationStateIO::Save(e, s, Resume(n1, mtp, replay), stk.data(), carry.data());
  // The slot replays its pending rows and decodes on: rows from n1 - replay change, nothing below.
  Forward(s, n1 - size_t(replay), n2, 2);
  std::fill(stk.begin(), stk.end(), 7.0f);
  const auto r2 = Resume(n2, mtp);
  const size_t full_copied = (copied = 0, ConversationStateIO::Save(e, s, r2, stk.data(), carry.data()), copied);
  const auto full = ConversationStateIO::Save(e, s, r2, stk.data(), carry.data());
  copied = 0;
  SaveBase from{base, n1 - size_t(replay)};
  assert(!ConversationStateIO::SharedSpans(s, n2, from).empty());
  const auto incremental = ConversationStateIO::Save(e, s, r2, stk.data(), carry.data(), from);
  // Bit for bit the same bytes, with fewer bytes copied from the device.
  assert(incremental->bytes == full->bytes && incremental->bytes.size() == full->bytes.size());
  const size_t shared = ConversationStateIO::SharedBytes(s, n2, from);
  assert(shared > 0 && copied + shared == full_copied && incremental->bytes.HeldElsewhere() == shared);
  // A restore of it writes the device bytes of the slot, also after the base is gone.
  const auto want = Device(s);
  SequenceState& other = *e.sequences_[1];
  Forward(other, 0, 100, 9);
  std::vector<float> stk2(stk.size(), 0), carry2(kWide, 0);
  from = {};
  base.reset();
  assert(incremental->bytes.HeldElsewhere() == 0);
  ConversationStateIO::Restore(e, other, *incremental, stk2.data(), carry2.data());
  assert(Device(other) == want && other.ple == s.ple && stk2 == stk && other.hist == s.hist);
  for (const auto& [l, layer] : s.layers) assert(other.layers.at(l).raw == layer.raw && other.layers.at(l).pool == layer.pool);
}

void RulesThatLimitSharing() {
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(24576, false, true));
  SequenceState& s = *e.sequences_[0];
  std::vector<float> stk((kMaxRows + 1) * kWide, 5), carry(kWide, 6);
  Forward(s, 0, 12000, 1);
  const auto base = ConversationStateIO::Save(e, s, Resume(12000, true), stk.data(), carry.data());
  Forward(s, 12000, 16000, 2);
  const auto r = Resume(16000, true);
  // Rows round down to a pool block; fewer than 128 rows share nothing.
  assert(ConversationStateIO::SharedSpans(s, 16000, {base, 127}).empty());
  const auto spans = ConversationStateIO::SharedSpans(s, 16000, {base, 12000});
  assert(spans.size() == 2 * 4 && spans[0].bytes == 11904 * 1024);  // 93 blocks of 128 rows; K, V, raw and pool
  // A base of another trunk layout shares nothing (here: a compact ring against raw rows).
  e.sequences_.push_back(std::make_unique<SequenceState>(24576, true, true));
  SequenceState& compact = *e.sequences_[1];
  Forward(compact, 0, 12000, 1);
  const auto compact_base = ConversationStateIO::Save(e, compact, Resume(12000, true), stk.data(), carry.data());
  assert(ConversationStateIO::SharedSpans(s, 16000, {compact_base, 12000}).empty());
  assert(ConversationStateIO::Save(e, s, r, stk.data(), carry.data(), {compact_base, 12000})->bytes.HeldElsewhere() == 0);
  // A base whose size does not match its layout shares nothing.
  auto damaged = std::make_shared<ConversationSnapshot>(*base);
  damaged->bytes.pop_back();
  assert(ConversationStateIO::SharedSpans(s, 16000, {damaged, 12000}).empty());
  // The rule the caller keeps: if the slot wrote a row below the shared rows after the base, the bytes differ. So the
  // base must hold the slot's rows (BatchPrefixCache tracks it).
  for (auto& [l, layer] : s.layers)
    for (size_t row = 0; row < 11904; row += 100) { layer.k[row * 512] ^= 1; layer.v[row * 512] ^= 1; }
  const auto full = ConversationStateIO::Save(e, s, r, stk.data(), carry.data());
  const auto wrong = ConversationStateIO::Save(e, s, r, stk.data(), carry.data(), {base, 12000});
  assert(wrong->bytes.HeldElsewhere() > 0);
  assert(!(wrong->bytes == full->bytes));
}

// the pending replay rows of the new save were written after the base was held, so they never share.
void PendingRowsNeverShare() {
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(24576, false, false));
  SequenceState& s = *e.sequences_[0];
  std::vector<float> stk((kMaxRows + 1) * kWide, 5), carry(kWide, 6);
  Forward(s, 0, 20000, 1);
  const auto base = ConversationStateIO::Save(e, s, Resume(20000, false), stk.data(), carry.data());
  Forward(s, 20000, 24000, 2);
  const auto r = Resume(24000, false, 8000);
  assert(ConversationStateIO::Settled(r) == 16000);
  const SaveBase from{base, 20000};
  const size_t settled = ConversationStateIO::SharedBytes(s, 16000, from), all = ConversationStateIO::SharedBytes(s, 24000, from);
  assert(settled > 0 && settled < all);  // 1 against 2 whole K chunks per layer
  const auto saved = ConversationStateIO::Save(e, s, r, stk.data(), carry.data(), from);
  assert(saved->bytes.HeldElsewhere() == settled);
  assert(saved->bytes == ConversationStateIO::Save(e, s, r, stk.data(), carry.data())->bytes);
}

// the slot base lifecycle of BatchPrefixCache (slot_base.h) with real snapshots.
void SlotBaseLifecycle() {
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(24576, false, false));
  SequenceState& s = *e.sequences_[0];
  std::vector<float> stk((kMaxRows + 1) * kWide, 5), carry(kWide, 6);
  Forward(s, 0, 12000, 1);
  auto snapshot = ConversationStateIO::Save(e, s, Resume(12000, false, 37), stk.data(), carry.data());
  SlotBases<ConversationSnapshot, 4> bases;
  assert(!bases.Get(2).snapshot && bases.Get(2).rows == 0);
  // Hold: a restore or a save; the pending replay rows are not the slot's yet.
  bases.Hold(2, snapshot);
  assert(bases.Get(2).snapshot == snapshot && bases.Get(2).rows == 12000 - 37 && !bases.Get(1).snapshot);
  // A checkpoint resume keeps only the rows below its own pending rows; a longer checkpoint changes nothing.
  bases.Limit(2, 9000, 100);
  assert(bases.Get(2).rows == 8900);
  bases.Limit(2, 11000, 0);
  assert(bases.Get(2).rows == 8900);
  bases.Limit(2, 50, 100);
  assert(bases.Get(2).rows == 0 && ConversationStateIO::SharedSpans(s, 12000, {bases.Get(2).snapshot, 0}).empty());
  // The restore hook clears it.
  bases.Hold(2, snapshot);
  bases.Clear(2);
  assert(!bases.Get(2).snapshot);
  // Weak: an evicted snapshot that nothing else holds is gone, and the next save copies everything.
  bases.Hold(3, snapshot);
  snapshot.reset();
  assert(!bases.Get(3).snapshot);
}

// an eviction through Admit counts what the victim really frees (SnapshotBytes::OwnBytes), less the
// chunks that the save keeps, and the spill room uses that count.
void SpillRoomWithSharedChunks() {
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(24576, false, false));
  SequenceState& s = *e.sequences_[0];
  std::vector<float> stk((kMaxRows + 1) * kWide, 5), carry(kWide, 6);
  Forward(s, 0, 20000, 1);
  const auto r1 = Resume(20000, false);
  const size_t bytes1 = ConversationStateIO::Bytes(e, s, r1, stk.data(), carry.data()) + kConversationPolicyBytes;
  const auto base = ConversationStateIO::Save(e, s, r1, stk.data(), carry.data());
  Forward(s, 20000, 24000, 2);
  const auto r2 = Resume(24000, false);
  const size_t bytes2 = ConversationStateIO::Bytes(e, s, r2, stk.data(), carry.data()) + kConversationPolicyBytes;
  const SaveBase from{base, 20000};
  const size_t shared = ConversationStateIO::SharedBytes(s, 24000, from);
  const SharedSave share{bytes2 - shared, ConversationStateIO::SharedOwners(s, 24000, from)};
  assert(shared > 0 && share.kept.size() * SnapshotChunkPool::kChunkBytes == shared);
  assert(base->bytes.OwnBytes() == base->bytes.size() && base->bytes.OwnBytes(share.kept) == base->bytes.size() - shared);
  const uint64_t reserve = 1 << 20;
  ConversationPolicy policy;
  policy.prefill_rows = 8;
  for (const bool shares : {true, false}) {
    BatchConversationCache<ConversationSnapshot> cache(bytes1 + bytes2 + 1000, {}, 1, reserve, false);
    std::vector<std::pair<uint64_t, size_t>> spilled;  // room, victim length
    cache.SetSpill([&](std::span<const int>, const ConversationPolicy&, const auto& entry, int, uint64_t room) {
      spilled.emplace_back(room, entry->snapshot->hist.size());
      return false;
    });
    // Session A's tail (the base), then session C, then session B's tail. One session is live (C), so A waits and
    // spills when B's save evicts it.
    const std::vector<int> b_key(24000, 5), c_key(100, 6);
    const auto other = [&](const std::vector<int>& key, const ConversationSnapshot& from) {
      auto copy = std::make_shared<ConversationSnapshot>(from);
      copy->hist = key;
      copy->resume.length = key.size();
      return std::shared_ptr<const ConversationSnapshot>(copy);
    };
    assert(cache.CaptureTail(base->hist, policy, bytes1, uint64_t(1) << 40, [&] { return base; }, true));
    assert(cache.CaptureTail(c_key, policy, 1000, uint64_t(1) << 40, [&] { return other(c_key, *base); }, true));
    // The host has room for B's new bytes only after A leaves.
    const uint64_t available = reserve + share.host_bytes - 1;
    const bool saved = cache.CaptureTail(b_key, policy, bytes2, available, [&] {
      return other(b_key, *ConversationStateIO::Save(e, s, r2, stk.data(), carry.data(), from));
    }, true, false, false, {}, shares ? share : SharedSave{});
    assert(saved && spilled.size() == 1 && spilled[0].second == 20000);
    // A frees its bytes less the chunks that B keeps (without sharing: all of A).
    const uint64_t freed = shares ? bytes1 - shared : bytes1;
    const uint64_t need = reserve + (shares ? share.host_bytes : bytes2);
    assert(spilled[0].first == (available + freed > need ? available + freed - need : 0));
  }
  // Admit refuses the save when A, less the chunks that the save keeps, frees too little; the same host need without
  // the kept chunks counts all of A and saves. Only C (live) and A are in the cache.
  for (const bool keeps : {true, false}) {
    BatchConversationCache<ConversationSnapshot> cache(bytes1 + bytes2 + 1000, {}, 1, reserve, false);
    const std::vector<int> b_key(24000, 5), c_key(100, 6);
    const auto other = [&](const std::vector<int>& key, const ConversationSnapshot& from) {
      auto copy = std::make_shared<ConversationSnapshot>(from);
      copy->hist = key;
      copy->resume.length = key.size();
      return std::shared_ptr<const ConversationSnapshot>(copy);
    };
    assert(cache.CaptureTail(base->hist, policy, bytes1, uint64_t(1) << 40, [&] { return base; }, true));
    assert(cache.CaptureTail(c_key, policy, 1000, uint64_t(1) << 40, [&] { return other(c_key, *base); }, true));
    // Short by one byte more than A frees with the kept chunks.
    const uint64_t available = reserve + share.host_bytes - (bytes1 - shared) - 1;
    const bool saved = cache.CaptureTail(b_key, policy, bytes2, available, [&] {
      return other(b_key, *ConversationStateIO::Save(e, s, r2, stk.data(), carry.data(), from));
    }, true, false, false, {}, keeps ? share : SharedSave{share.host_bytes, {}});
    // The refusal reads "protected": an empty cache would have room (the reason line counts whole entries).
    assert(saved == !keeps && (!keeps || cache.Status().last.refused == "protected"));
    assert(keeps ? cache.Bytes() == bytes1 + 1000 : cache.Bytes() == bytes2 + 1000);
  }
}

// the host copy of a Frontier point, made while the slot holds later rows, has the bytes of a save at the
// point.
ConversationResume FrontierResume(size_t length) {
  ConversationResume r;
  r.length = length;
  r.prompt_only = true;
  return r;
}
void SaveFromCheckpoint(bool compact, bool mtp, bool with_base, bool truncated) {
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(24576, compact, mtp));
  SequenceState& s = *e.sequences_[0];
  std::vector<float> stk((kMaxRows + 1) * kWide, 0), carry(kWide, 0);  // the shared zero pair of target state
  const size_t f0 = 12032, f = 20096, n = 24000;  // grid points (multiples of 128); f0 holds one whole K chunk
  Forward(s, 0, f0, 1);
  const auto base = ConversationStateIO::Save(e, s, FrontierResume(f0), stk.data(), carry.data());
  Forward(s, f0, f, 2);
  const auto want = ConversationStateIO::Save(e, s, FrontierResume(f), stk.data(), carry.data());
  const size_t want_bytes = ConversationStateIO::Bytes(e, s, FrontierResume(f), stk.data(), carry.data());
  SlotCheckpointIO::Buffer buffer(SlotCheckpointIO::Bytes(s));
  SlotCheckpointIO::Save(s, buffer);
  const SlotCheckpoint point{s.hist, {}, 0, false, CheckpointKind::Frontier, 1, {f, false, false}};
  // The slot decodes on: the fixed state, the compact ring, the PLE ring and the pool blocks from f change.
  Forward(s, f, n, 3);
  if (truncated) s.hist.resize(n - 300);  // a tail save truncates the stream to its kept length
  const SaveBase from = with_base ? SaveBase{base, f0} : SaveBase{};
  copied = 0;
  const auto got = SlotCheckpointIO::SaveFrom(e, s, buffer, point, FrontierResume(f), stk.data(), carry.data(), from);
  assert(got->bytes == want->bytes && got->hist == want->hist && got->resume.length == f);
  assert(std::memcmp(&got->retained, &want->retained, sizeof(RetainedPrefix)) == 0);
  assert(got->bytes.HeldElsewhere() == (with_base ? ConversationStateIO::SharedBytes(s, f, from) : 0));
  assert(!with_base || got->bytes.HeldElsewhere() > 0);
  assert(ConversationStateIO::BytesAt(e, s, FrontierResume(f), f, stk.data(), carry.data()) == want_bytes);
  // A point whose rows the slot no longer holds, or a copy with MTP rows, is refused.
  SlotCheckpoint other = point;
  other.hist[5] += 1;
  bool threw = false;
  try { SlotCheckpointIO::SaveFrom(e, s, buffer, other, FrontierResume(f), stk.data(), carry.data()); }
  catch (const std::logic_error&) { threw = true; }
  assert(threw);
  auto with_mtp = FrontierResume(f);
  with_mtp.mpos = int(f);
  threw = false;
  try { SlotCheckpointIO::SaveFrom(e, s, buffer, point, with_mtp, stk.data(), carry.data()); }
  catch (const std::logic_error&) { threw = true; }
  assert(threw);
}

// fc-d2d: SlotCheckpointIO::CopyTo writes to the target slot the bytes
// that SaveFrom and then ConversationStateIO::Restore write, with no host stage (no pinned chunk, no host copy).
struct SlotBytes {
  std::map<int, Layer> layers;
  std::vector<float> ple, gdn_state, gdn_hist, gdn_pend;
  std::vector<int> hist;
  RetainedPrefix retained;
  bool mtp;
  explicit SlotBytes(const SequenceState& s)
      : layers(s.layers), ple(s.ple), gdn_state(s.gdn_state), gdn_hist(s.gdn_hist), gdn_pend(s.gdn_pend),
        hist(s.hist), retained(s.retained), mtp(s.qc.contains(kLayers)) {}
  // The fixed state only: GDN state, history and pending rows (review-fcd2d LOW-3).
  bool SameGdn(const SlotBytes& o) const {
    return gdn_state == o.gdn_state && gdn_hist == o.gdn_hist && gdn_pend == o.gdn_pend;
  }
  bool operator==(const SlotBytes& o) const {
    if (!SameGdn(o) || layers.size() != o.layers.size() || ple != o.ple || hist != o.hist || mtp != o.mtp ||
        std::memcmp(&retained, &o.retained, sizeof(RetainedPrefix)) != 0)
      return false;
    for (const auto& [l, layer] : layers) {
      const auto other = o.layers.find(l);
      if (other == o.layers.end() || layer.k != other->second.k || layer.v != other->second.v ||
          layer.raw != other->second.raw || layer.pool != other->second.pool)
        return false;
    }
    return true;
  }
};
void CopyToEqualsHostCopy(bool compact, bool donor_mtp, bool target_mtp, bool truncated) {
  constexpr int kRows = 24576;
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(kRows, compact, donor_mtp));
  e.sequences_.push_back(std::make_unique<SequenceState>(kRows, compact, target_mtp));
  e.sequences_.push_back(std::make_unique<SequenceState>(kRows, compact, target_mtp));
  SequenceState& s = *e.sequences_[0];
  SequenceState& host = *e.sequences_[1];  // the host path: SaveFrom, then Restore
  SequenceState& copy = *e.sequences_[2];  // CopyTo
  std::vector<float> stk((kMaxRows + 1) * kWide, 0), carry(kWide, 0);  // the shared zero pair of target state
  const size_t f = 20096, n = 24000;
  Forward(s, 0, f, 1);
  SlotCheckpointIO::Buffer buffer(SlotCheckpointIO::Bytes(s));
  SlotCheckpointIO::Save(s, buffer);
  const SlotCheckpoint point{s.hist, {}, 0, false, CheckpointKind::Frontier, 1, {f, false, false}};
  // The donor decodes on: the fixed state, the compact ring, the PLE ring and the pool blocks from f change.
  Forward(s, f, n, 3);
  if (truncated) s.hist.resize(n - 300);
  // Both targets hold another session's rows in every span, also above f (a restore leaves those rows).
  Forward(host, 0, kRows, 7);
  Forward(copy, 0, kRows, 7);
  assert(SlotBytes(host) == SlotBytes(copy));
  const auto r = FrontierResume(f);
  const SlotBytes donor_before(s);
  ConversationStateIO::Restore(e, host, *SlotCheckpointIO::SaveFrom(e, s, buffer, point, r, stk.data(), carry.data()),
                               stk.data(), carry.data());
  const auto pinned_before = pinned;
  const int copies_before = copies;
  const auto stk_before = stk, carry_before = carry;
  SlotCheckpointIO::CopyTo(e, s, buffer, point, r, stk.data(), carry.data(), copy);
  assert(pinned == pinned_before && copies == copies_before);  // no host stage
  assert(SlotBytes(copy).SameGdn(SlotBytes(host)));  // the GDN spans come from the checkpoint buffer
  assert(SlotBytes(copy) == SlotBytes(host));
  assert(copy.hist == point.hist && copy.qc.contains(kLayers) == (donor_mtp || target_mtp));
  // The copy only reads the donor and its checkpoint buffer; the shared pair stays.
  assert(SlotBytes(s) == donor_before && stk == stk_before && carry == carry_before);
}
// Every refusal comes before the first write: the target keeps its bytes.
void CopyToRefuses() {
  constexpr int kRows = 24576;
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(kRows, false, true));
  e.sequences_.push_back(std::make_unique<SequenceState>(kRows, false, true));
  e.sequences_.push_back(std::make_unique<SequenceState>(16384, false, true));
  e.sequences_.push_back(std::make_unique<SequenceState>(kRows, true, true));
  SequenceState& s = *e.sequences_[0];
  SequenceState& target = *e.sequences_[1];
  SequenceState& small = *e.sequences_[2];    // capacity below f
  SequenceState& compact = *e.sequences_[3];  // another trunk layout
  std::vector<float> stk((kMaxRows + 1) * kWide, 0), carry(kWide, 0);
  const size_t f = 20096;
  Forward(s, 0, f, 1);
  SlotCheckpointIO::Buffer buffer(SlotCheckpointIO::Bytes(s));
  SlotCheckpointIO::Save(s, buffer);
  const SlotCheckpoint point{s.hist, {}, 0, false, CheckpointKind::Frontier, 1, {f, false, false}};
  Forward(s, f, 24000, 3);
  Forward(target, 0, kRows, 7);
  Forward(small, 0, 16384, 7);
  Forward(compact, 0, kRows, 7);
  const auto refuses = [&](SequenceState& to, const SlotCheckpoint& at, const ConversationResume& r, auto error) {
    const SlotBytes before(to);
    bool threw = false;
    try { SlotCheckpointIO::CopyTo(e, s, buffer, at, r, stk.data(), carry.data(), to); }
    catch (const decltype(error)&) { threw = true; }
    assert(threw && SlotBytes(to) == before);
  };
  // Rows gone from the donor: another token below f, or a stream truncated below f.
  SlotCheckpoint other = point;
  other.hist[5] += 1;
  refuses(target, other, FrontierResume(f), std::logic_error(""));
  const auto hist = s.hist;
  s.hist.resize(f - 1);
  refuses(target, point, FrontierResume(f), std::logic_error(""));
  s.hist = hist;
  // A copy with MTP rows, another length, or the donor as its own target.
  auto with_mtp = FrontierResume(f);
  with_mtp.mpos = int(f);
  refuses(target, point, with_mtp, std::logic_error(""));
  refuses(target, point, FrontierResume(f - 128), std::logic_error(""));
  refuses(s, point, FrontierResume(f), std::logic_error(""));
  // A target with too few rows or another trunk layout.
  refuses(small, point, FrontierResume(f), std::runtime_error(""));
  refuses(compact, point, FrontierResume(f), std::runtime_error(""));
  // The good copy still runs after the refusals.
  SlotCheckpointIO::CopyTo(e, s, buffer, point, FrontierResume(f), stk.data(), carry.data(), target);
  assert(target.hist == point.hist);
}

// An earlier change: a checkpoint buffer with a tail (fix A keeps the MTP carry after the spans) takes spans only
// up to the tail, and a save never writes the tail.
void SaveKeepsTheTail() {
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(24576, true, false));
  SequenceState& s = *e.sequences_[0];
  Forward(s, 0, 20000, 1);
  const size_t spans = SlotCheckpointIO::Bytes(s), tail = kWide * sizeof(float);
  SlotCheckpointIO::Buffer buffer(spans + tail, tail);
  assert(buffer.bytes == spans + tail && buffer.tail == tail);
  std::memset(buffer.device + spans, 0x5A, tail);
  SlotCheckpointIO::Save(s, buffer);
  for (size_t i = 0; i < tail; ++i) assert(buffer.device[spans + i] == 0x5A);
  // Spans that reach into the tail are refused, also when the whole buffer would hold them.
  SlotCheckpointIO::Buffer small(spans + tail - 1, tail);
  bool threw = false;
  try { SlotCheckpointIO::Save(s, small); } catch (const std::logic_error&) { threw = true; }
  assert(threw);
  threw = false;
  try { SlotCheckpointIO::Buffer wrong(tail - 1, tail); } catch (const std::invalid_argument&) { threw = true; }
  assert(threw);
}

int main() {
  SaveKeepsTheTail();
  for (bool compact : {false, true})
    for (bool donor_mtp : {false, true})
      for (bool target_mtp : {false, true})
        for (bool truncated : {false, true}) CopyToEqualsHostCopy(compact, donor_mtp, target_mtp, truncated);
  CopyToRefuses();
  for (bool compact : {false, true})
    for (bool mtp : {false, true})
      for (bool with_base : {false, true})
        for (bool truncated : {false, true}) SaveFromCheckpoint(compact, mtp, with_base, truncated);
  for (bool compact : {false, true})
    for (bool mtp : {false, true})
      for (int replay : {0, 37}) SameAsFullSave(compact, mtp, replay);
  RulesThatLimitSharing();
  PendingRowsNeverShare();
  SlotBaseLifecycle();
  SpillRoomWithSharedChunks();
  std::puts("tail_incr_test passed (with the slot frontier copy)");
  return 0;
}
