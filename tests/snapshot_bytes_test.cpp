// Snapshot chunk pool and chunked byte copies (CPU path; the device kernel runs in the snapshot gates).
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

constexpr int hipMemcpyDeviceToHost = 0, hipMemcpyHostToDevice = 1;
// The reserve helper pins from its own thread, so the fake runtime locks its table.
std::mutex fake_runtime;
std::map<uintptr_t, size_t> pinned;
std::atomic<int> allocations = 0, frees = 0, fences = 0, worker_allocations = 0;
std::atomic<bool> fail_allocation = false;
const std::thread::id worker = std::this_thread::get_id();  // the test's main thread plays the device worker
template<class T> int hipHostMalloc(T** out, size_t n) {
  if (fail_allocation) return 1;
  std::lock_guard lock(fake_runtime);
  *out = static_cast<T*>(std::malloc(n));
  pinned[uintptr_t(*out)] = n;
  ++allocations;
  if (std::this_thread::get_id() == worker) ++worker_allocations;
  return 0;
}
std::function<void()> on_free;  // runs inside each free (the pool lock test)
int hipHostFree(void* p) {
  if (on_free) on_free();
  std::lock_guard lock(fake_runtime);
  assert(pinned.erase(uintptr_t(p)) == 1); std::free(p); ++frees; return 0;
}
size_t PinnedCount() { std::lock_guard lock(fake_runtime); return pinned.size(); }
bool IsPinned(const void* p, size_t n) {
  std::lock_guard lock(fake_runtime);
  const auto address = uintptr_t(p);
  for (const auto& [base, bytes] : pinned)
    if (address >= base && address - base <= bytes && n <= bytes - (address - base)) return true;
  return false;
}
int hipMemcpy(void* out, const void* in, size_t n, int direction) {
  assert(IsPinned(direction == hipMemcpyDeviceToHost ? out : in, n));
  std::memcpy(out, in, n);
  return 0;
}
int hipStreamSynchronize(int stream) { assert(stream == 0); ++fences; return 0; }
#define HIP_OK(expr) assert((expr) == 0)
#include "../engine/runtime/snapshot_bytes.h"
using namespace crucible;

struct Span { const void* pointer; size_t bytes; };
constexpr size_t kChunk = SnapshotChunkPool::kChunkBytes;
// Waits until the reserve helper has nothing left to pin (it pins, or the limit, the check or a failure stops it).
void Settle(SnapshotChunkPool& pool) {
  for (int i = 0; i < 5000 && pool.Filling(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  assert(!pool.Filling());
}

int main() {
  auto& pool = SnapshotChunkPool::Instance();
  // Spans that straddle chunk edges, including odd sizes, round-trip byte for byte.
  std::vector<std::vector<uint8_t>> device;
  for (size_t size : {kChunk - 3, size_t(5), kChunk + 17, size_t(16), 2 * kChunk}) {
    std::vector<uint8_t> span(size);
    for (size_t i = 0; i < size; ++i) span[i] = uint8_t(i * 131 + size);
    device.push_back(std::move(span));
  }
  std::vector<Span> spans;
  size_t total = 0;
  for (const auto& span : device) { spans.push_back({span.data(), span.size()}); total += span.size(); }
  {
    SnapshotBytes bytes(total);
    assert(bytes.size() == total && bytes.Chunks().size() == (total + kChunk - 1) / kChunk);
    const int before = fences;
    CopySnapshotSpans(spans, bytes, true);
    assert(fences == before + 1);  // the copy waits before the snapshot is published
    std::vector<uint8_t> expected;
    for (const auto& span : device) expected.insert(expected.end(), span.begin(), span.end());
    assert(bytes.Copy() == expected);
    // Restore writes the same bytes back.
    auto restored = device;
    for (auto& span : restored) std::fill(span.begin(), span.end(), 0xEE);
    std::vector<Span> targets;
    for (const auto& span : restored) targets.push_back({span.data(), span.size()});
    CopySnapshotSpans(targets, bytes, false);
    assert(restored == device);
    // A copy is deep and equal; a shortened copy differs.
    SnapshotBytes copy = bytes;
    assert(copy == bytes && copy.Chunks().front().host != bytes.Chunks().front().host);
    copy.pop_back();
    assert(!(copy == bytes));
    // Spans that do not cover the snapshot are rejected.
    spans.pop_back();
    try { CopySnapshotSpans(spans, bytes, true); assert(false); } catch (const std::logic_error&) {}
  }
  // Without an idle limit every released chunk goes back to the system (the default for tools and gates).
  assert(pool.IdleBytes() == 0 && allocations == frees + 0 && pinned.empty());
  // With a limit, released chunks are reused without an allocation.
  pool.SetPinnedLimit(3 * kChunk);
  { SnapshotBytes warm(3 * kChunk); }
  assert(pool.IdleBytes() == 3 * kChunk);
  const int allocated = allocations;
  fail_allocation = true;
  {
    SnapshotBytes reused(2 * kChunk + 1);
    assert(allocations == allocated && pool.IdleBytes() == 0);
    // The pool is empty and the system refuses: no partial snapshot, and the chunks go back to the pool.
    try { SnapshotBytes more(kChunk); assert(false); } catch (const std::bad_alloc&) {}
  }
  assert(pool.IdleBytes() == 3 * kChunk);
  fail_allocation = false;
  // A larger release than the limit returns the excess; a lower limit trims at once.
  { SnapshotBytes big(5 * kChunk); }
  assert(pool.IdleBytes() == 3 * kChunk);
  pool.SetPinnedLimit(kChunk);
  assert(pool.IdleBytes() == kChunk && pinned.size() == 1);
  // the limit holds for every pinned chunk, in use and idle. Chunks in use may pass it (a disk write holds an
  // evicted snapshot while the next save fills); when they come back, the pool keeps only what fits.
  pool.SetPinnedLimit(3 * kChunk);
  {
    SnapshotBytes live(2 * kChunk);  // takes the idle chunk and one new chunk
    { SnapshotBytes held(2 * kChunk); assert(pinned.size() == 4 && pool.IdleBytes() == 0); }
    assert(pinned.size() == 3 && pool.IdleBytes() == kChunk);
    // frontier-copy (soak memory rule): PinnedBytes counts every pinned chunk once, in use and idle.
    assert(pool.PinnedBytes() == 3 * kChunk);
    // A transient snapshot (a frontier copy that lives for one open): when it ends, the chunks in use (pinned less
    // idle) are the old count again.
    const size_t in_use = pool.PinnedBytes() - pool.IdleBytes();
    { SnapshotBytes transient(kChunk); assert(pool.PinnedBytes() - pool.IdleBytes() == in_use + kChunk); }
    assert(pool.PinnedBytes() - pool.IdleBytes() == in_use && pool.PinnedBytes() == 3 * kChunk);
  }
  assert(pinned.size() == 3 && pool.IdleBytes() == 3 * kChunk && pool.PinnedBytes() == 3 * kChunk);
  {
    // a full cache of three 2-chunk states (limit 6 chunks). Each save evicts the oldest to a
    // disk write that holds it while the new state fills, then the write ends. Pinned memory passes the limit by one
    // state at most while a write runs, never grows across saves, and is back at the limit after each write.
    pool.SetPinnedLimit(6 * kChunk);
    std::deque<SnapshotBytes> live;
    for (int i = 0; i < 3; ++i) live.emplace_back(2 * kChunk);
    size_t peak = 0;
    for (int save = 0; save < 20; ++save) {
      SnapshotBytes held = std::move(live.front());  // the writer's copy of the victim
      live.pop_front();
      live.emplace_back(2 * kChunk);
      peak = std::max(peak, pinned.size());
      { SnapshotBytes done = std::move(held); }  // the write ends
      assert(pinned.size() == 6 && pool.IdleBytes() == 0);
    }
    assert(peak == 8);
  }
  assert(pinned.size() == 6 && pool.IdleBytes() == 6 * kChunk);
  // hipHostFree synchronizes the device, so the pool frees its excess chunks outside its lock. Take, Give
  // and IdleBytes on the device worker must not wait for a writer that frees hundreds of chunks.
  int unlocked = 0;
  on_free = [&] {
    auto other = std::async(std::launch::async, [&] { return pool.IdleBytes(); });
    assert(other.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    ++unlocked;
  };
  pool.SetPinnedLimit(4 * kChunk);  // a lower limit frees two idle chunks
  assert(unlocked == 2 && pinned.size() == 4);
  { SnapshotBytes over(6 * kChunk); }  // takes 4 idle and 2 new chunks; the release frees 2
  assert(unlocked == 4 && pinned.size() == 4 && pool.IdleBytes() == 4 * kChunk);
  on_free = nullptr;
  pool.SetPinnedLimit(0);
  assert(pool.IdleBytes() == 0 && pinned.empty());
  {
    // a tail save that pinned its chunks on the device worker stopped every decoder (3.5 GB, 0.65 s at 128K).
    // The reserve helper pins the expected bytes on its own thread first; the save then takes idle chunks only.
    pool.SetPinnedLimit(8 * kChunk);
    const int before = worker_allocations;
    pool.Reserve(3 * kChunk);
    Settle(pool);
    assert(pool.IdleBytes() == 3 * kChunk && worker_allocations == before && PinnedCount() == 3);
    {
      SnapshotBytes save(3 * kChunk);
      assert(worker_allocations == before);
      // The reserve holds while the target stands: the helper pins the next save's chunks.
      Settle(pool);
      assert(pool.IdleBytes() == 3 * kChunk && PinnedCount() == 6);
    }
    // A released save gives its chunks back; the pool keeps them up to the limit, as before.
    assert(pool.IdleBytes() == 6 * kChunk);
    // The helper never passes the pinned limit.
    pool.Reserve(20 * kChunk);
    Settle(pool);
    assert(PinnedCount() == 8 && pool.IdleBytes() == 8 * kChunk && worker_allocations == before);
    // It pins nothing while the host check refuses (the host reserve), and again when a new target allows it.
    pool.Reserve(0);
    Settle(pool);
    pool.SetPinnedLimit(0);
    pool.SetPinnedLimit(8 * kChunk);
    std::atomic<int> checks = 0;
    pool.Reserve(2 * kChunk, [&] { ++checks; return false; });
    Settle(pool);
    assert(checks >= 1 && PinnedCount() == 0);
    pool.Reserve(2 * kChunk, [] { return true; });
    Settle(pool);
    assert(PinnedCount() == 2);
    // an exception in the pin step (here the host check) stops the helper; it does not end the
    // process, and a new target starts it again.
    pool.Reserve(3 * kChunk, [] () -> bool { throw std::runtime_error("host check"); });
    Settle(pool);
    assert(PinnedCount() == 2);
    pool.Reserve(3 * kChunk, [] { return true; });
    Settle(pool);
    assert(PinnedCount() == 3);
    pool.Reserve(0);
    pool.SetPinnedLimit(0);
    pool.SetPinnedLimit(8 * kChunk);
    pool.Reserve(2 * kChunk);
    Settle(pool);
    assert(PinnedCount() == 2);
    // A failed pin stops the helper until the next target; a save then pins on the worker as before.
    fail_allocation = true;
    pool.Reserve(4 * kChunk);
    Settle(pool);
    assert(PinnedCount() == 2);
    fail_allocation = false;
    { SnapshotBytes more(3 * kChunk); assert(worker_allocations == before + 1); }
    // Zero stops the growth; the stop waits for the helper, so no pin runs after it (a stopping server).
    pool.Reserve(0);
    pool.StopReserve();
    const size_t kept = PinnedCount();
    { SnapshotBytes large(5 * kChunk); }
    assert(PinnedCount() == std::max<size_t>(kept, 5));
    pool.SetPinnedLimit(0);
    assert(pool.IdleBytes() == 0 && PinnedCount() == 0);
    // A new target after the stop starts the helper again.
    pool.SetPinnedLimit(2 * kChunk);
    pool.Reserve(2 * kChunk);
    Settle(pool);
    assert(PinnedCount() == 2);
    pool.StopReserve();
    pool.SetPinnedLimit(0);
    assert(pool.IdleBytes() == 0 && PinnedCount() == 0);
  }
  {
    // a composed snapshot shares the whole chunks of a held snapshot with the same bytes and copies the rest.
    assert(PinnedCount() == 0);
    std::vector<uint8_t> a(3 * kChunk + kChunk / 2), b(100);
    for (size_t i = 0; i < a.size(); ++i) a[i] = uint8_t(i * 7 + 1);
    for (size_t i = 0; i < b.size(); ++i) b[i] = uint8_t(i * 5 + 3);
    auto parent = std::make_unique<SnapshotBytes>(a.size());
    CopySnapshotSpans(std::vector<Span>{{a.data(), a.size()}}, *parent, true);
    assert(PinnedCount() == 4 && parent->Flat() && parent->HeldElsewhere() == 0);
    // The device bytes after the parent changed only past the parent's rows (the exactness rule of the caller).
    a[a.size() - 1] ^= 0xff;
    std::vector<SnapshotRange> copies;
    const std::vector<SnapshotBytes::Part> parts{{a.data(), a.size(), parent.get(), 0}, {b.data(), b.size()}};
    assert(parent->ShareableBytes(0, a.size()) == 3 * kChunk);
    SnapshotBytes child = SnapshotBytes::Compose(parts, copies);
    // Chunks 0 to 2 are shared; the last half chunk of `a` and all of `b` go into one new chunk.
    assert(PinnedCount() == 5 && child.size() == a.size() + b.size());
    assert(copies.size() == 2 && copies[0].offset == 3 * kChunk && copies[0].bytes == kChunk / 2 &&
           copies[1].device == b.data() && copies[1].offset == a.size());
    const int fences_before = fences;
    CopySnapshotRanges(copies, child, true);
    assert(fences == fences_before + 1);
    std::vector<uint8_t> want(a);
    want.insert(want.end(), b.begin(), b.end());
    assert(child.Copy() == want);
    assert(child.HeldElsewhere() == 3 * kChunk && parent->HeldElsewhere() == 3 * kChunk);
    // A part that starts inside a chunk copies the partial chunks at both ends.
    std::vector<SnapshotRange> shifted_copies;
    const SnapshotBytes shifted = SnapshotBytes::Compose({{a.data() + 100, a.size() - 100, parent.get(), 100}},
                                                         shifted_copies);
    assert(shifted_copies.size() == 2 && shifted_copies[0].bytes == kChunk - 100 &&
           shifted_copies[1].offset == 3 * kChunk - 100 && shifted.HeldElsewhere() == 2 * kChunk);
    CopySnapshotRanges(shifted_copies, shifted, true);
    assert(shifted.Copy() == std::vector<uint8_t>(a.begin() + 100, a.end()));
    // A snapshot composed from a composed one shares the same chunks.
    std::vector<SnapshotRange> again_copies;
    const SnapshotBytes again = SnapshotBytes::Compose({{want.data(), want.size(), &child, 0}}, again_copies);
    CopySnapshotRanges(again_copies, again, true);
    assert(again.Copy() == want && again_copies.size() == 1 && again_copies[0].bytes == kChunk / 2 + b.size());
    // A restore of a composed snapshot writes its logical bytes; a copy of it is flat and equal.
    std::vector<uint8_t> restored(want.size(), 0);
    CopySnapshotSpans(std::vector<Span>{{restored.data(), restored.size()}}, child, false);
    assert(restored == want);
    const SnapshotBytes flat = child;
    assert(flat.Flat() && flat == child);
    // The parent leaves: the child keeps the shared chunks and its bytes; the parent's last chunk goes.
    const size_t before = PinnedCount();
    parent.reset();
    assert(PinnedCount() == before - 1 && child.Copy() == want && child.HeldElsewhere() == 3 * kChunk);
    SnapshotBytes shorter = child;
    shorter.pop_back();
    assert(shorter.size() == want.size() - 1);
  }
  assert(PinnedCount() == 0);
  // Moves transfer chunks without a copy.
  SnapshotBytes first(10);
  const auto* host = first.Chunks().front().host;
  SnapshotBytes second = std::move(first);
  assert(first.empty() && second.Chunks().front().host == host);
  return 0;
}
