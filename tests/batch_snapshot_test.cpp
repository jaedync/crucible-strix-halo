#include "../engine/model/flashnext/geometry.h"
#include "../engine/model/flashnext/fixed_state.h"
#include "../engine/runtime/retained_prefix.h"
#include "../engine/runtime/bounds.h"
#include "../engine/layers/cache_storage.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdint>
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
bool fail_pinned_allocation = false;
int snapshot_fences = 0;
int hipStreamSynchronize(int stream) { assert(stream == 0); ++snapshot_fences; return 0; }
bool IsPinned(const void* p, size_t n) {
  const auto address = uintptr_t(p);
  for (const auto& [base, bytes] : pinned)
    if (address >= base && address - base <= bytes && n <= bytes - (address - base)) return true;
  return false;
}
template<class T> int hipHostMalloc(T** out, size_t n) {
  if (fail_pinned_allocation) return 1;
  *out = static_cast<T*>(std::malloc(n));
  if (!*out) return 1;
  pinned[uintptr_t(*out)] = n;
  return 0;
}
int hipHostFree(void* p) { assert(pinned.erase(uintptr_t(p)) == 1); std::free(p); return 0; }
inline int hipMemcpy(void* out, const void* in, size_t n, int direction) {
  assert(IsPinned(direction == hipMemcpyDeviceToHost ? out : in, n));
  std::memcpy(out, in, n); return 0;
}
#define HIP_OK(expr) assert((expr) == 0)
struct QsaCache { __half* k; __half* v; float* ik_raw; float* ik_pool; int raw_slots = 0; };
struct SequenceState {
  const int capacity, mtp_capacity;
  bool compact_state = false;
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
  float* ple_ring;
  std::vector<float> g, h, p, ple, raw, pool, trunk_raw, trunk_pool;
  std::vector<__half> k, v, trunk_k, trunk_v;
  explicit SequenceState(int cap, int mtp_rows = 0) : capacity(cap), mtp_capacity(mtp_rows ? mtp_rows : cap), g(kGdnStateBytes / sizeof(float)), h(16 * 10240),
      p(48 * kMaxRows * kPend), ple(16 * kWide), raw(cap * 128), pool(QsaPoolFloats(cap)),
      trunk_raw(cap * 128), trunk_pool(QsaPoolFloats(cap)), k(cap * 512), v(cap * 512),
      trunk_k(cap * 512), trunk_v(cap * 512) {
    gstate[0] = g.data(); ghist[0] = h.data(); gpend[0] = p.data(); ple_ring = ple.data();
    qc[3] = {trunk_k.data(), trunk_v.data(), trunk_raw.data(), trunk_pool.data()};
    qc[kLayers] = {k.data(), v.data(), raw.data(), pool.data()};
  }
  void Reset() {
    hist.clear(); retained.Reset();
    std::fill(g.begin(), g.end(), 0); std::fill(h.begin(), h.end(), 0); std::fill(ple.begin(), ple.end(), 0);
  }
};
struct Engine {
  std::vector<std::unique_ptr<SequenceState>> sequences_;
  std::vector<float> logits = std::vector<float>(kVocab);
  float* logits_ = logits.data();
  bool mtp_fail = false;
  void EnsureMtpCache(SequenceState& state, int position) {
    if (mtp_fail || position != 0) throw std::runtime_error("MTP allocation failed");
    state.raw.resize(size_t(state.compact_state ? 16 : state.capacity) * 128);
    state.qc[kLayers] = {state.k.data(), state.v.data(), state.raw.data(), state.pool.data(),
                        state.compact_state ? 16 : 0};
  }
};
#include "../engine/runtime/conversation_snapshot.h"

template<class F> void Reject(F run) {
  bool rejected = false;
  try { run(); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
}
int main() {
  Engine e;
  e.sequences_.push_back(std::make_unique<SequenceState>(4));
  e.sequences_.push_back(std::make_unique<SequenceState>(8));
  SequenceState& a = *e.sequences_[0];
  SequenceState& b = *e.sequences_[1];
  a.hist = {1, 2}; assert(a.retained.Forward(2, 0, 2).empty());
  a.g[0] = 11; a.h[0] = 12; a.p[0] = 13; a.ple[0] = 14;
  std::fill(a.k.begin(), a.k.end(), 19); std::fill(a.v.begin(), a.v.end(), 20);
  std::fill(a.raw.begin(), a.raw.end(), 21); std::fill(a.pool.begin(), a.pool.end(), 22);
  std::fill(a.trunk_k.begin(), a.trunk_k.end(), 29);
  std::vector<float> stk((kMaxRows + 1) * kWide, 31), carry(kWide, 32);
  std::vector<float> peer_stk(stk.size(), 41), peer_carry(kWide, 42);
  b.hist = {8}; b.g[0] = 81;
  ConversationResume r{2, 3, 0, 7, true, {7}, std::vector<float>(kVocab, 2)};
  const auto snapshot = ConversationStateIO::Save(e, a, r, stk.data(), carry.data());
  assert(snapshot_fences == 1);
  assert(ConversationStateIO::Bytes(e, a, r, stk.data(), carry.data()) >= snapshot->bytes.size());
  // SaveBytes (the bytes that the pool pins before a tail save) equals the bytes that Save copies,
  // with MTP rows and without them.
  {
    auto rows = r; rows.mpos = 2;
    assert(ConversationStateIO::SaveBytes(a, 2, true) ==
           ConversationStateIO::Save(e, a, rows, stk.data(), carry.data())->bytes.size());
    rows.mpos = 0;
    assert(ConversationStateIO::SaveBytes(a, 2, false) ==
           ConversationStateIO::Save(e, a, rows, stk.data(), carry.data())->bytes.size());
    // Before the stream has its MTP cache, SaveBytes counts the MTP rows that the cache will hold.
    e.sequences_.push_back(std::make_unique<SequenceState>(4));
    SequenceState& fresh = *e.sequences_.back();
    fresh.hist = {1, 2}; assert(fresh.retained.Forward(2, 0, 2).empty());
    rows.mpos = 2;
    const size_t allocated = ConversationStateIO::Save(e, fresh, rows, stk.data(), carry.data())->bytes.size();
    fresh.qc.erase(kLayers);
    assert(ConversationStateIO::SaveBytes(fresh, 2, true) == allocated);
    // Rows clamp to the capacity (4): 2 more rows of K, V and raw keys in the trunk layer and the MTP layer.
    assert(ConversationStateIO::SaveBytes(fresh, 9, true) == allocated + 2 * 2 * (2 * 512 * 2 + 128 * 4));
    e.sequences_.pop_back();
  }
  a.hist = {9}; a.g[0] = 91; a.h[0] = 92; a.p[0] = 93; a.ple[0] = 94;
  std::fill(a.k.begin(), a.k.end(), 99); std::fill(stk.begin(), stk.end(), 51);
  std::fill(a.trunk_k.begin(), a.trunk_k.end(), 98);
  std::fill(carry.begin(), carry.end(), 52);
  const auto restored = ConversationStateIO::Restore(e, a, *snapshot, stk.data(), carry.data());
  assert(restored.mpos == 3 && restored.length == 2 && restored.pending_tokens == r.pending_tokens);
  assert(a.hist == std::vector<int>({1, 2}) && a.g[0] == 11 && a.h[0] == 12 && a.p[0] == 13 && a.ple[0] == 14);
  assert(a.k[3 * 512 - 1] == 19 && a.k[3 * 512] == 99);
  assert(a.trunk_k[2 * 512 - 1] == 29 && a.trunk_k[2 * 512] == 98);
  assert(stk[0] == 31 && carry[0] == 32 && e.logits[0] == 2);
  assert(b.hist == std::vector<int>({8}) && b.g[0] == 81 && peer_stk[0] == 41 && peer_carry[0] == 42);
  auto prefix = a.retained; assert(prefix.Forward(1, 0, 1).empty());
  auto invalid = r; invalid.mpos = 5;
  Reject([&] { ConversationStateIO::Save(e, a, invalid, stk.data(), carry.data()); });
  // a slot whose MTP cache holds fewer positions than its context refuses MTP rows above them.
  e.sequences_.push_back(std::make_unique<SequenceState>(8, 2));
  SequenceState& c = *e.sequences_[2];
  c.hist = {1, 2}; assert(c.retained.Forward(2, 0, 2).empty());
  auto two = r; two.mpos = 2;
  const auto fits = ConversationStateIO::Save(e, c, two, stk.data(), carry.data());
  Reject([&] { ConversationStateIO::Save(e, c, r, stk.data(), carry.data()); });
  Reject([&] { ConversationStateIO::Restore(e, c, *snapshot, peer_stk.data(), peer_carry.data()); });
  assert(c.hist == std::vector<int>({1, 2}));
  assert(ConversationStateIO::Restore(e, c, *fits, peer_stk.data(), peer_carry.data()).mpos == 2);
  e.sequences_.pop_back();
  auto bad = *snapshot; bad.resume.mpos = 5;
  a.g[0] = 61;
  Reject([&] { ConversationStateIO::Restore(e, a, bad, stk.data(), carry.data()); });
  assert(a.g[0] == 61);
  bad = *snapshot; bad.resume.logits.resize(1);
  Reject([&] { ConversationStateIO::Restore(e, a, bad, stk.data(), carry.data()); });
  assert(a.g[0] == 61);
  bad = *snapshot; bad.bytes.pop_back();
  Reject([&] { ConversationStateIO::Restore(e, a, bad, stk.data(), carry.data()); });
  assert(a.g[0] == 61);
  // A cache entry can move to another owned slot without changing the source slot.
  std::fill(b.k.begin(), b.k.end(), 71);
  e.logits[0] = 72;
  const auto peer_resume = ConversationStateIO::Restore(e, b, *snapshot, peer_stk.data(), peer_carry.data());
  assert(peer_resume.length == 2 && peer_resume.mpos == 3);
  assert(b.hist == snapshot->hist && b.g[0] == 11 && b.h[0] == 12 && b.p[0] == 13 && b.ple[0] == 14);
  assert(b.k[3 * 512 - 1] == 19 && b.k[3 * 512] == 71);
  assert(peer_stk[0] == 31 && peer_carry[0] == 32 && e.logits[0] == 2);
  assert(a.g[0] == 61 && a.hist == snapshot->hist && stk[0] == 31 && carry[0] == 32);
  const auto peer_copy = ConversationStateIO::Save(e, b, peer_resume, peer_stk.data(), peer_carry.data());
  assert(peer_copy->bytes == snapshot->bytes && peer_copy->hist == snapshot->hist);
  const size_t allocations = pinned.size();
  fail_pinned_allocation = true;
  bool allocation_failed = false;
  try { ConversationStateIO::Save(e, a, r, stk.data(), carry.data()); }
  catch (const std::bad_alloc&) { allocation_failed = true; }
  fail_pinned_allocation = false;
  assert(allocation_failed && pinned.size() == allocations && a.g[0] == 61);
  try {
    PinnedAllocator<float>{}.allocate(std::numeric_limits<size_t>::max());
    assert(false);
  } catch (const std::bad_array_new_length&) {}
  assert(pinned.size() == allocations);
  SequenceState foreign(4);
  Reject([&] { ConversationStateIO::Save(e, foreign, r, stk.data(), carry.data()); });
  Reject([&] { ConversationStateIO::Save(e, a, r, nullptr, carry.data()); });
  const auto migrated = ConversationStateIO::Restore(e, b, *snapshot, peer_stk.data(), peer_carry.data());
  assert(migrated.mpos == 3 && b.hist == snapshot->hist && b.g[0] == 11 && a.g[0] == 61);
  assert(peer_stk[0] == 31 && peer_carry[0] == 32);
  b.Reset();
  const ConversationResume empty;
  const auto zero = ConversationStateIO::Save(e, b, empty, peer_stk.data(), peer_carry.data());
  assert(ConversationStateIO::Restore(e, b, *zero, peer_stk.data(), peer_carry.data()).mpos == 0);
  assert(b.hist.empty());
  a.compact_state = true;
  a.qc[3].raw_slots = 16;
  a.trunk_raw.resize(16 * 128); a.qc[3].ik_raw = a.trunk_raw.data();
  a.qc[kLayers].raw_slots = 16;
  a.raw.resize(16 * 128); a.qc[kLayers].ik_raw = a.raw.data();
  std::fill(a.raw.begin(), a.raw.end(), 37);
  const auto compact = ConversationStateIO::Save(e, a, r, stk.data(), carry.data());
  assert(compact->qsa_layout == (std::vector<std::pair<int, int>>{{3, 16}, {kLayers, 16}}));
  std::fill(a.raw.begin(), a.raw.end(), 38);
  ConversationStateIO::Restore(e, a, *compact, stk.data(), carry.data());
  assert(a.raw.back() == 37);
  a.qc[kLayers].raw_slots = 0;
  Reject([&] { ConversationStateIO::Restore(e, a, *compact, stk.data(), carry.data()); });
  a.qc[kLayers].raw_slots = 16;
  auto compact_rows = r; compact_rows.mpos = 2;
  const size_t compact_mtp = ConversationStateIO::Save(e, a, compact_rows, stk.data(), carry.data())->bytes.size();
  a.qc.erase(kLayers);
  assert(ConversationStateIO::SaveBytes(a, 2, true) == compact_mtp);  // compact: the MTP raw ring, not raw rows
  auto serial = r; serial.mpos = 0;
  const auto absent = ConversationStateIO::Save(e, a, serial, stk.data(), carry.data());
  assert(absent->qsa_layout == (std::vector<std::pair<int, int>>{{3, 16}}));
  assert(ConversationStateIO::SaveBytes(a, 2, false) == absent->bytes.size());
  Reject([&] { ConversationStateIO::Save(e, a, r, stk.data(), carry.data()); });
  auto invalid_absent = *absent; invalid_absent.resume.mpos = 1;
  Reject([&] { ConversationStateIO::Restore(e, a, invalid_absent, stk.data(), carry.data()); });
  assert(ConversationStateIO::Restore(e, a, *absent, stk.data(), carry.data()).mpos == 0);
  b.compact_state = true; b.qc.erase(kLayers); b.g[0] = 81;
  b.qc[3].raw_slots = 16;
  b.trunk_raw.resize(16 * 128); b.qc[3].ik_raw = b.trunk_raw.data();
  auto truncated = *compact; truncated.bytes.pop_back();
  Reject([&] { ConversationStateIO::Restore(e, b, truncated, peer_stk.data(), peer_carry.data()); });
  assert(b.g[0] == 81 && !b.qc.contains(kLayers));
  e.mtp_fail = true;
  Reject([&] { ConversationStateIO::Restore(e, b, *compact, peer_stk.data(), peer_carry.data()); });
  assert(b.g[0] == 81 && !b.qc.contains(kLayers));
  e.mtp_fail = false;
  assert(ConversationStateIO::Restore(e, b, *compact, peer_stk.data(), peer_carry.data()).mpos == 3);
  assert(b.qc.at(kLayers).raw_slots == 16 && b.raw.back() == 37);
  assert(ConversationStateIO::Restore(e, b, *absent, peer_stk.data(), peer_carry.data()).mpos == 0);
  assert(b.qc.contains(kLayers));
  // a lazy save keeps the full blocks below the settled rows on the device. Rows from the
  // settled rows up change after the save (the resumed request runs on); the snapshot still equals the eager one.
  {
    const size_t rows = 3 * kStateBlockRows + 500;
    e.sequences_.push_back(std::make_unique<SequenceState>(int(rows) + 8));
    SequenceState& big = *e.sequences_.back();
    big.hist.assign(rows, 3);
    for (auto* v : {&big.g, &big.h, &big.p, &big.ple, &big.raw, &big.pool, &big.trunk_raw, &big.trunk_pool})
      for (size_t i = 0; i < v->size(); ++i) (*v)[i] = float(i % 251) + 1;
    for (auto* v : {&big.k, &big.v, &big.trunk_k, &big.trunk_v})
      for (size_t i = 0; i < v->size(); ++i) (*v)[i] = __half(i % 253 + 1);
    std::vector<float> lstk((kMaxRows + 1) * kWide, 5), lcarry(kWide, 6);
    ConversationResume lr{rows, 0, 5, 7, true, {7}, std::vector<float>(kVocab, 2)};
    const auto eager = ConversationStateIO::Save(e, big, lr, lstk.data(), lcarry.data());
    const auto plan = ConversationStateIO::PlanLazy(big, lr, kStateBlockRows, ConversationStateIO::HeadBytes(big),
                                                    ConversationStateIO::kTailBytes);
    assert(plan.rows == 3 * kStateBlockRows && plan.bytes > 0 && plan.bytes < eager->bytes.size());
    const auto source = std::make_shared<LazySource>(DeviceLazyReader(), plan.rows);
    const auto lazy = ConversationStateIO::SaveLazy(e, big, lr, lstk.data(), lcarry.data(), plan, source);
    assert(lazy->bytes.size() == eager->bytes.size());
    assert(lazy->bytes.LazyBytes() == plan.bytes && lazy->bytes.HostBytes() == eager->bytes.size() - plan.bytes);
    assert(lazy->hist == eager->hist && lazy->resume.length == eager->resume.length);
    // Shadow mode: the eager copy of the lazy rows is kept; a walk that reads other bytes is refused and counted.
    {
      const auto shadow_source = std::make_shared<LazySource>(DeviceLazyReader(), plan.rows);
      const auto shadowed = ConversationStateIO::SaveLazy(e, big, lr, lstk.data(), lcarry.data(), plan, shadow_source, true);
      assert(shadowed->bytes.Copy() == eager->bytes.Copy() && shadow_source->Mismatches() == 0);
      assert(shadow_source->Compared() == plan.bytes);
      const auto saved = std::array<__half, 4>{big.k[10], big.v[10], big.trunk_k[10], big.trunk_v[10]};
      for (auto* v : {&big.k, &big.v, &big.trunk_k, &big.trunk_v}) (*v)[10] = __half(77);  // rows below the settled rows change
      bool refused = false;
      try { shadowed->bytes.Copy(); } catch (const SnapshotLost&) { refused = true; }
      big.k[10] = saved[0], big.v[10] = saved[1], big.trunk_k[10] = saved[2], big.trunk_v[10] = saved[3];
      assert(refused && shadow_source->Mismatches() == 1);
    }
    // The slot keeps writing from the settled rows: K, V, raw keys and the pool of every layer, rows >= 3 blocks.
    const size_t first = plan.rows;
    for (auto* v : {&big.k, &big.v, &big.trunk_k, &big.trunk_v})
      for (size_t i = first * 512; i < v->size(); ++i) (*v)[i] = 9999;
    for (auto* v : {&big.raw, &big.trunk_raw})
      for (size_t i = first * 128; i < v->size(); ++i) (*v)[i] = -1;
    for (auto* v : {&big.pool, &big.trunk_pool})
      for (size_t i = first / 128 * 4096; i < v->size(); ++i) (*v)[i] = -2;
    assert(lazy->bytes.Copy() == eager->bytes.Copy());
    // A write below the plan's rows ends the source: the next walk refuses (SnapshotLost).
    source->Invalidate();
    bool lost = false;
    try { lazy->bytes.Copy(); } catch (const SnapshotLost&) { lost = true; }
    assert(lost);
    e.sequences_.pop_back();
  }
}
