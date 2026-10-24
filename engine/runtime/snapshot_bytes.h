// Snapshot storage in fixed pinned chunks from one process-wide pool (oracle-cache item 8).
// A save takes chunks without hipHostMalloc or a zero fill once the pool is warm, and an evicted snapshot gives its
// chunks back. The chunk size bounds the memory that a snapshot holds beyond its own bytes.
#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace crucible {
struct SnapshotChunk { uint8_t* host = nullptr; uint8_t* device = nullptr; };

class SnapshotChunkPool {
 public:
  static constexpr size_t kChunkBytes = size_t(8) << 20;
  static SnapshotChunkPool& Instance() {
    // Never destroyed: process exit returns the pages, and a destructor could run after the HIP runtime.
    static auto* pool = new SnapshotChunkPool;
    return *pool;
  }
  // Idle chunks stay pinned while all pinned chunks, in use and idle, fit in this many bytes (the conversation cache
  // budget, which admission already reserves). Chunks in use may pass it for a while (a disk write holds an evicted
  // snapshot, an earlier change); when they come back, the excess goes back to the system. Zero returns each released chunk at once.
  void SetPinnedLimit(size_t bytes) {
    std::vector<SnapshotChunk> excess;
    {
      std::lock_guard lock(mutex_);
      pinned_limit_ = bytes;
      blocked_ = false;
      excess = Trim();
    }
    wake_.notify_all();
    Free(excess);
  }
  size_t IdleBytes() const {
    std::lock_guard lock(mutex_);
    return idle_.size() * kChunkBytes;
  }
  // Every pinned chunk once, in use and idle: entries that share chunks
  // count them at full size each, so the soak memory rule reads this, not the entry bytes.
  size_t PinnedBytes() const {
    std::lock_guard lock(mutex_);
    return pinned_ * kChunkBytes;
  }
  // a save pins each new chunk on the calling thread (about 5.5 GB/s on the test machine), so a 3.5 GB tail save stopped
  // every decoder of the batch worker for 0.65 s. A helper thread pins idle chunks before the saves instead: it keeps
  // `bytes` idle, inside the pinned limit, and pins a chunk only when `may_pin` allows it (the host reserve). A pin
  // failure or a refusal stops it until the next call. Zero stops the growth; idle chunks stay, as evicted ones do.
  void Reserve(size_t bytes, std::function<bool()> may_pin = {}) {
    {
      std::lock_guard lock(mutex_);
      reserve_ = bytes;
      may_pin_ = std::move(may_pin);
      blocked_ = false;
      if (bytes && !helper_.joinable()) {
        stop_ = false;
        // Without a thread the saves pin on the caller, as before the reserve.
        try { helper_ = std::thread([this] { Fill(); }); } catch (const std::exception&) {}
      }
    }
    wake_.notify_all();
  }
  // Ends the helper and waits for its pin in progress: call it before the HIP runtime can go away.
  void StopReserve() {
    std::thread helper;
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
      reserve_ = 0;
      helper = std::move(helper_);
    }
    wake_.notify_all();
    if (helper.joinable()) helper.join();
  }
  // The helper pins now or has chunks left to pin (tests wait for it).
  bool Filling() const {
    std::lock_guard lock(mutex_);
    return pinning_ || (helper_.joinable() && !stop_ && !blocked_ && Short());
  }
  SnapshotChunk Take() {
    {
      std::unique_lock lock(mutex_);
      if (!idle_.empty()) {
        const SnapshotChunk chunk = idle_.back();
        idle_.pop_back();
        lock.unlock();
        wake_.notify_all();  // the helper pins a replacement while the target stands
#if CRUCIBLE_DEBUG_NONFINITE & 2
        std::memset(chunk.host, 0xFF, kChunkBytes);  // a restore must read only bytes that a save wrote
#endif
        return chunk;
      }
    }
    const SnapshotChunk chunk = Pin();
#if CRUCIBLE_DEBUG_NONFINITE & 2
    std::memset(chunk.host, 0xFF, kChunkBytes);
#endif
    std::lock_guard lock(mutex_);
    ++pinned_;
    return chunk;
  }
  void Give(SnapshotChunk chunk) {
    std::vector<SnapshotChunk> excess;
    {
      std::lock_guard lock(mutex_);
      idle_.push_back(chunk);
      excess = Trim();
    }
    Free(excess);
  }

 private:
  static SnapshotChunk Pin() {
    SnapshotChunk chunk;
    if (hipHostMalloc(&chunk.host, kChunkBytes) != 0) throw std::bad_alloc();
#ifdef __HIPCC__
    void* device = nullptr;
    if (hipHostGetDevicePointer(&device, chunk.host, 0) != hipSuccess) {
      HIP_OK(hipHostFree(chunk.host));
      throw std::bad_alloc();
    }
    chunk.device = static_cast<uint8_t*>(device);
#else
    chunk.device = chunk.host;
#endif
    return chunk;
  }
  // Under the lock: the idle bytes are below the reserve and one more chunk fits the limit.
  bool Short() const {
    return idle_.size() * kChunkBytes < reserve_ && (pinned_ + 1) * kChunkBytes <= pinned_limit_;
  }
  // The helper thread: pins outside the lock, so Take, Give and IdleBytes on the device worker never wait for a pin.
  void Fill() {
    for (;;) {
      std::function<bool()> may_pin;
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] { return stop_ || (!blocked_ && Short()); });
        if (stop_) return;
        try {
          may_pin = may_pin_;
        } catch (const std::exception&) {
          blocked_ = true;
          continue;
        }
        pinning_ = true;
      }
      // No exception leaves the helper (it would end the process): a failed check or pin stops it until the next
      // Reserve or SetPinnedLimit.
      std::optional<SnapshotChunk> chunk;
      try {
        if (!may_pin || may_pin()) chunk = Pin();
      } catch (const std::exception&) {}
      std::vector<SnapshotChunk> excess;
      {
        std::lock_guard lock(mutex_);
        pinning_ = false;
        bool stop = !chunk;
        try {
          if (chunk) {
            idle_.push_back(*chunk);
            ++pinned_;
            chunk.reset();  // the pool holds it now
            excess = Trim();  // a Take on the worker pinned at the same time and passed the limit
          }
        } catch (const std::exception&) {
          stop = true;
        }
        if (stop) blocked_ = true;
      }
      if (chunk) HIP_OK(hipHostFree(chunk->host));  // the pool could not keep it
      Free(excess);
    }
  }
  // Under the lock: the idle chunks above the limit leave the pool.
  std::vector<SnapshotChunk> Trim() {
    std::vector<SnapshotChunk> excess;
    while (!idle_.empty() && pinned_ * kChunkBytes > pinned_limit_) {
      excess.push_back(idle_.back());
      idle_.pop_back();
      --pinned_;
    }
    return excess;
  }
  // Outside the lock: hipHostFree synchronizes the device, so a disk writer that frees hundreds of chunks must not
  // stop the device worker's Take, Give and IdleBytes.
  static void Free(const std::vector<SnapshotChunk>& chunks) {
    for (const auto& chunk : chunks) HIP_OK(hipHostFree(chunk.host));
  }
  mutable std::mutex mutex_;
  std::vector<SnapshotChunk> idle_;
  size_t pinned_limit_ = 0, pinned_ = 0;  // pinned_: chunks in use and idle
  // The reserve helper. Never joined at exit: the pool is never destroyed (StopReserve ends it).
  std::condition_variable wake_;
  std::thread helper_;
  std::function<bool()> may_pin_;
  size_t reserve_ = 0;
  bool stop_ = false, blocked_ = false, pinning_ = false;
};

// A pinned chunk that goes back to the pool when its last user goes.
class PinnedChunk {
 public:
  explicit PinnedChunk(SnapshotChunk chunk) : chunk_(chunk) {}
  PinnedChunk(const PinnedChunk&) = delete;
  PinnedChunk& operator=(const PinnedChunk&) = delete;
  ~PinnedChunk() { SnapshotChunkPool::Instance().Give(chunk_); }
  const SnapshotChunk& Get() const { return chunk_; }
  static std::shared_ptr<const PinnedChunk> Take() {
    const SnapshotChunk chunk = SnapshotChunkPool::Instance().Take();
    try {
      return std::make_shared<const PinnedChunk>(chunk);
    } catch (...) {
      SnapshotChunkPool::Instance().Give(chunk);
      throw;
    }
  }
 private:
  SnapshotChunk chunk_;
};

// A device copy of a snapshot: `bytes` at `device`, to or from the logical offset `offset`.
struct SnapshotRange { const void* device; size_t bytes, offset; };

// The bytes of one snapshot in logical order, as extents of pinned chunks. Contents start unspecified (no fill):
// Save writes them all. A flat snapshot (a full save, a disk read) has one extent per chunk. A composed snapshot
// shares the whole chunks of a held snapshot that has the same bytes, and packs its other bytes into chunks
// of its own.
class SnapshotBytes {
 public:
  // A logical part for Compose: `bytes` at `device`. With `from`, that snapshot holds the same bytes at `from_offset`.
  struct Part { const void* device = nullptr; size_t bytes = 0; const SnapshotBytes* from = nullptr; size_t from_offset = 0; };
  SnapshotBytes() = default;
  explicit SnapshotBytes(size_t size) {
    for (size_t at = 0; at < size; at += SnapshotChunkPool::kChunkBytes)
      Push(PinnedChunk::Take(), 0, std::min(SnapshotChunkPool::kChunkBytes, size - at));
  }
  // A copy is deep and flat.
  SnapshotBytes(const SnapshotBytes& other) : SnapshotBytes(other.size_) {
    other.ForEachPiece(0, other.size_, [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
      ForEachPiece(at, count, [&](const SnapshotChunk& mine, size_t into, size_t part, size_t done) {
        std::memcpy(mine.host + into, chunk.host + inside + done, part);
      });
    });
  }
  SnapshotBytes& operator=(const SnapshotBytes& other) {
    if (this != &other) *this = SnapshotBytes(other);
    return *this;
  }
  SnapshotBytes(SnapshotBytes&& other) noexcept
      : extents_(std::exchange(other.extents_, {})), chunks_(std::exchange(other.chunks_, {})),
        starts_(std::exchange(other.starts_, {})), size_(std::exchange(other.size_, 0)) {}
  SnapshotBytes& operator=(SnapshotBytes&& other) noexcept {
    if (this != &other) {
      extents_ = std::exchange(other.extents_, {});
      chunks_ = std::exchange(other.chunks_, {});
      starts_ = std::exchange(other.starts_, {});
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  // Builds a snapshot from `parts` in logical order and appends the device ranges that the caller must copy into it
  // to `copies`. From a part with `from`, it shares the extents of `from` that use a whole chunk; it copies the rest
  // of that part, as it copies a part without `from`. So a shared chunk holds only bytes of this snapshot.
  static SnapshotBytes Compose(const std::vector<Part>& parts, std::vector<SnapshotRange>& copies) {
    struct Segment { std::shared_ptr<const PinnedChunk> shared; const uint8_t* device; size_t bytes; };
    std::vector<Segment> plan;
    size_t own = 0;
    const auto copy = [&](const void* device, size_t bytes) {
      if (!bytes) return;
      auto* at = static_cast<const uint8_t*>(device);
      if (!plan.empty() && !plan.back().shared && plan.back().device + plan.back().bytes == at)
        plan.back().bytes += bytes;
      else
        plan.push_back({nullptr, at, bytes});
      own += bytes;
    };
    for (const auto& part : parts) {
      if (!part.from) {
        copy(part.device, part.bytes);
        continue;
      }
      part.from->ForEachExtent(part.from_offset, part.bytes, [&](const Extent& extent, size_t inside, size_t count,
                                                                 size_t at) {
        if (WholeChunk(extent) && inside == 0 && count == extent.bytes)
          plan.push_back({extent.owner, static_cast<const uint8_t*>(part.device) + at, count});
        else
          copy(static_cast<const uint8_t*>(part.device) + at, count);
      });
    }
    std::vector<std::shared_ptr<const PinnedChunk>> fresh;
    for (size_t at = 0; at < own; at += SnapshotChunkPool::kChunkBytes) fresh.push_back(PinnedChunk::Take());
    SnapshotBytes out;
    size_t packed = 0;  // bytes in the fresh chunks so far
    for (const auto& segment : plan) {
      if (segment.shared) {
        out.Push(segment.shared, 0, segment.bytes);
        continue;
      }
      copies.push_back({segment.device, segment.bytes, out.size_});
      for (size_t done = 0; done < segment.bytes;) {
        const size_t inside = packed % SnapshotChunkPool::kChunkBytes;
        const size_t count = std::min(segment.bytes - done, SnapshotChunkPool::kChunkBytes - inside);
        out.Push(fresh[packed / SnapshotChunkPool::kChunkBytes], inside, count);
        packed += count;
        done += count;
      }
    }
    return out;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  // Tests shorten a copy to model a corrupt snapshot.
  void pop_back() {
    if (!size_) return;
    --size_;
    if (--extents_.back().bytes == 0) {
      extents_.pop_back();
      chunks_.pop_back();
      starts_.pop_back();
    }
  }
  // One chunk per extent, in logical order. In a flat snapshot, chunk i holds the logical bytes from i x 8 MB.
  const std::vector<SnapshotChunk>& Chunks() const { return chunks_; }
  bool Flat() const {
    for (size_t index = 0; index < extents_.size(); ++index)
      if (extents_[index].inside || starts_[index] != index * SnapshotChunkPool::kChunkBytes) return false;
    return true;
  }
  // Calls f(chunk, offset_in_chunk, bytes, offset_in_range) for each chunk piece of [offset, offset + bytes).
  template<class F> void ForEachPiece(size_t offset, size_t bytes, F f) const {
    ForEachExtent(offset, bytes, [&](const Extent& extent, size_t inside, size_t count, size_t at) {
      f(extent.owner->Get(), extent.inside + inside, count, at);
    });
  }
  // Logical bytes in chunks that another snapshot also uses: an eviction of this snapshot does not free them.
  size_t HeldElsewhere() const { return size_ - OwnBytes(); }
  // Logical bytes that an eviction of this snapshot frees: its chunks that no other snapshot uses and that are not in
  // `kept` (the chunks that a save keeps, ShareableOwners). O(E log E) in the extents.
  size_t OwnBytes(std::span<const void* const> kept = {}) const {
    std::vector<const void*> owners;
    owners.reserve(extents_.size());
    for (const auto& extent : extents_) owners.push_back(extent.owner.get());
    std::sort(owners.begin(), owners.end());
    std::vector<const void*> sorted_kept(kept.begin(), kept.end());
    std::sort(sorted_kept.begin(), sorted_kept.end());
    size_t own = 0;
    for (const auto& extent : extents_) {
      const void* owner = extent.owner.get();
      const auto [first, last] = std::equal_range(owners.begin(), owners.end(), owner);
      if (extent.owner.use_count() > last - first) continue;  // another snapshot holds this chunk too
      if (std::binary_search(sorted_kept.begin(), sorted_kept.end(), owner)) continue;
      own += extent.bytes;
    }
    return own;
  }
  // The chunks of [offset, offset + bytes) that a Compose part with this snapshot as `from` shares.
  std::vector<const void*> ShareableOwners(size_t offset, size_t bytes) const {
    std::vector<const void*> owners;
    ForEachExtent(offset, bytes, [&](const Extent& extent, size_t inside, size_t count, size_t) {
      if (WholeChunk(extent) && inside == 0 && count == extent.bytes) owners.push_back(extent.owner.get());
    });
    return owners;
  }
  // The bytes of [offset, offset + bytes) that a Compose part with this snapshot as `from` shares.
  size_t ShareableBytes(size_t offset, size_t bytes) const {
    size_t shared = 0;
    ForEachExtent(offset, bytes, [&](const Extent& extent, size_t inside, size_t count, size_t) {
      if (WholeChunk(extent) && inside == 0 && count == extent.bytes) shared += count;
    });
    return shared;
  }
  std::vector<uint8_t> Copy() const {
    std::vector<uint8_t> out(size_);
    ForEachPiece(0, size_, [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
      std::memcpy(out.data() + at, chunk.host + inside, count);
    });
    return out;
  }
  bool operator==(const SnapshotBytes& other) const { return size_ == other.size_ && Copy() == other.Copy(); }

 private:
  struct Extent { std::shared_ptr<const PinnedChunk> owner; size_t inside, bytes; };
  static bool WholeChunk(const Extent& extent) {
    return extent.inside == 0 && extent.bytes == SnapshotChunkPool::kChunkBytes;
  }
  // Calls f(extent, offset_in_extent, bytes, offset_in_range) for each extent piece of [offset, offset + bytes).
  template<class F> void ForEachExtent(size_t offset, size_t bytes, F f) const {
    if (offset > size_ || bytes > size_ - offset) throw std::out_of_range("snapshot byte range");
    if (!bytes) return;
    size_t index = size_t(std::upper_bound(starts_.begin(), starts_.end(), offset) - starts_.begin()) - 1;
    for (size_t done = 0; done < bytes; ++index) {
      const Extent& extent = extents_[index];
      const size_t inside = offset + done - starts_[index];
      const size_t count = std::min(bytes - done, extent.bytes - inside);
      f(extent, inside, count, done);
      done += count;
    }
  }
  // Appends an extent; one that continues the last extent in the same chunk joins it.
  void Push(std::shared_ptr<const PinnedChunk> owner, size_t inside, size_t bytes) {
    if (!extents_.empty() && extents_.back().owner == owner &&
        extents_.back().inside + extents_.back().bytes == inside) {
      extents_.back().bytes += bytes;
    } else {
      chunks_.push_back(owner->Get());
      starts_.push_back(size_);
      extents_.push_back({std::move(owner), inside, bytes});
    }
    size_ += bytes;
  }
  std::vector<Extent> extents_;
  std::vector<SnapshotChunk> chunks_;  // extents_[i].owner->Get()
  std::vector<size_t> starts_;         // the logical offset of extents_[i]
  size_t size_ = 0;
};

struct SnapshotPiece { const void* src; void* dst; size_t bytes; };
#ifdef __HIPCC__
// blockIdx.y picks a piece and x strides across it. A byte copy, so the saved bits equal the device bits.
__global__ void SnapshotCopyKernel(const SnapshotPiece* pieces, int first) {
  const SnapshotPiece piece = pieces[first + int(blockIdx.y)];
  const size_t start = size_t(blockIdx.x) * blockDim.x + threadIdx.x, stride = size_t(gridDim.x) * blockDim.x;
  if (((uintptr_t(piece.src) | uintptr_t(piece.dst) | piece.bytes) & 15) == 0) {
    const auto* src = static_cast<const uint4*>(piece.src);
    auto* dst = static_cast<uint4*>(piece.dst);
    for (size_t i = start; i < piece.bytes / 16; i += stride) dst[i] = src[i];
  } else {
    const auto* src = static_cast<const uint8_t*>(piece.src);
    auto* dst = static_cast<uint8_t*>(piece.dst);
    for (size_t i = start; i < piece.bytes; i += stride) dst[i] = src[i];
  }
}
#endif
// Copies each range to (save) or from (restore) its logical bytes in the snapshot chunks and waits for the copy. One
// kernel on the null stream moves every piece: on gfx1151 it reached 86 GB/s where one hipMemcpy per span reached
// 58 GB/s (bench/snapshot_copy_bench.hip). Null-stream order matches the hipMemcpy calls it replaces.
inline void CopySnapshotRanges(const std::vector<SnapshotRange>& ranges, const SnapshotBytes& bytes, bool to_host) {
  std::vector<SnapshotPiece> pieces;
  for (const auto& range : ranges)
    bytes.ForEachPiece(range.offset, range.bytes, [&](const SnapshotChunk& chunk, size_t inside, size_t count, size_t at) {
      auto* device = static_cast<uint8_t*>(const_cast<void*>(range.device)) + at;
      if (to_host) pieces.push_back({device, chunk.device + inside, count});
      else pieces.push_back({chunk.device + inside, device, count});
    });
#ifdef __HIPCC__
  // The piece table lives in pinned memory that the kernel reads; the wait below frees it for the next copy.
  static std::mutex mutex;
  static SnapshotPiece* table = nullptr;
  static SnapshotPiece* device_table = nullptr;
  static size_t capacity = 0;
  std::lock_guard lock(mutex);
  if (pieces.size() > capacity) {
    if (table) HIP_OK(hipHostFree(table));
    table = device_table = nullptr;
    capacity = 0;
    const size_t wanted = std::max<size_t>(pieces.size(), 4096);
    if (hipHostMalloc(&table, wanted * sizeof(SnapshotPiece)) != hipSuccess) throw std::bad_alloc();
    void* mapped = nullptr;
    HIP_OK(hipHostGetDevicePointer(&mapped, table, 0));
    device_table = static_cast<SnapshotPiece*>(mapped);
    capacity = wanted;
  }
  std::copy(pieces.begin(), pieces.end(), table);
  constexpr int kMaxGridY = 65535;
  for (size_t first = 0; first < pieces.size(); first += kMaxGridY) {
    const unsigned count = unsigned(std::min<size_t>(kMaxGridY, pieces.size() - first));
    hipLaunchKernelGGL(SnapshotCopyKernel, dim3(16, count), dim3(256), 0, 0, device_table, int(first));
    HIP_OK(hipGetLastError());
  }
#else
  for (const auto& piece : pieces)
    HIP_OK(hipMemcpy(piece.dst, piece.src, piece.bytes, to_host ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice));
#endif
  HIP_OK(hipStreamSynchronize(0));
}
// Copies the spans, in order, to (save) or from (restore) the whole snapshot.
template<class Span>
void CopySnapshotSpans(const std::vector<Span>& spans, const SnapshotBytes& bytes, bool to_host) {
  std::vector<SnapshotRange> ranges;
  size_t offset = 0;
  for (const auto& span : spans) {
    ranges.push_back({span.pointer, span.bytes, offset});
    offset += span.bytes;
  }
  if (offset != bytes.size()) throw std::logic_error("snapshot spans do not cover the snapshot bytes");
  CopySnapshotRanges(ranges, bytes, to_host);
}
}  // namespace crucible
