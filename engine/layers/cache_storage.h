// Exact storage addresses and allocation ownership, independent of HIP execution.
#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string_view>
#include "../model/flashnext/geometry.h"

namespace crucible {
constexpr int kQsaRawSlots = 16, kQsaRawDim = 128, kQsaRawDecodeRows = kMaxRows;
// Preparation must retain the incomplete four-token pool and every verify row.
static_assert(kQsaRawSlots >= kMaxRows + 3, "increase the raw ring before increasing verification rows");
#ifdef __HIPCC__
#define CACHE_STORAGE_HD __host__ __device__
#else
#define CACHE_STORAGE_HD
#endif
CACHE_STORAGE_HD constexpr size_t QsaRawOffset(int slots, int position, int dimension) {
  return size_t(slots ? position % slots : position) * kQsaRawDim + dimension;
}
CACHE_STORAGE_HD constexpr bool QsaRawUsesChunk(int slots, int rows) {
  return slots != 0 && rows > kQsaRawDecodeRows;
}
CACHE_STORAGE_HD constexpr bool QsaRawFromChunk(int slots, int rows, int pos0, int position) {
  return QsaRawUsesChunk(slots, rows) && position >= pos0;
}
#undef CACHE_STORAGE_HD

inline bool CompactStateFlag(const char* value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("CRUCIBLE_COMPACT_STATE must be 0 or 1");
}
// The engine's view of the flag. The train 3 failure came from mismatched check
// commands; matched checks pass. It stays opt-in.
inline bool EngineCompactState(const char* value) {
  return CompactStateFlag(value);
}
inline std::array<size_t, 4> QsaCacheBytes(int context, bool bounded) {
  if (context < 1 || context > 262144) throw std::invalid_argument("QSA cache context is out of range");
  return {size_t(context) * 1024, size_t(context) * 1024,
          size_t(bounded ? kQsaRawSlots : context) * kQsaRawDim * sizeof(float),
          size_t((context + 127) / 128) * 4096 * sizeof(float)};
}
inline bool CacheAllocationFits(size_t free, size_t needed, size_t reserve) {
  return free > reserve && needed <= free - reserve;
}
// mtp_rows: the positions of the MTP cache.
inline size_t QsaSequenceCacheBytes(int context, bool compact, bool mtp, int mtp_rows) {
  if (mtp_rows < 1 || mtp_rows > context) throw std::invalid_argument("MTP cache rows are out of range");
  const auto bytes = QsaCacheBytes(context, compact), own = QsaCacheBytes(mtp_rows, compact);
  return 12ull * (bytes[0] + bytes[1] + bytes[2] + bytes[3]) +
         (compact && !mtp ? 0 : own[0] + own[1] + own[2] + own[3]);
}
inline size_t QsaSequenceCacheBytes(int context, bool compact, bool mtp) {
  return QsaSequenceCacheBytes(context, compact, mtp, context);
}
// Serve admission charges the lazy MTP cache. A speculative request promotes it later;
// a promotion that fails after admission would make that request depend on server load.
inline size_t ServeQsaCacheBytes(int context, bool compact, int mtp_rows) {
  return QsaSequenceCacheBytes(context, compact, true, mtp_rows);
}
inline size_t ServeQsaCacheBytes(int context, bool compact) {
  return ServeQsaCacheBytes(context, compact, context);
}
// Full storage always holds its MTP cache; a compact slot may not have one yet.
inline size_t MissingMtpCacheBytes(int mtp_rows, bool compact, bool has_mtp) {
  if (!compact || has_mtp) return 0;
  const auto bytes = QsaCacheBytes(mtp_rows, true);
  return bytes[0] + bytes[1] + bytes[2] + bytes[3];
}
inline size_t QsaRawSnapshotBytes(int slots, size_t length) {
  return length ? size_t(slots ? slots : length) * kQsaRawDim * sizeof(float) : 0;
}
// The K (and V) bytes of one position in KV layout `kvl` (0: fp16 rows of 512 halves). The only K/V size formula of a
// snapshot: a new layout adds its size here (kv-h9-store section 5).
inline size_t KvqPositionBytes(int kvl) {
  if (kvl != 0) throw std::invalid_argument("unknown KV layout");
  return 512 * 2;
}
// One span of a QSA layer in a conversation snapshot: `bytes` in all; each `unit_rows` positions take `unit_bytes`, in
// position order (0: the span does not grow by position, as the compact raw ring).
struct QsaSnapshotSpan { size_t bytes, unit_rows, unit_bytes; };
// The four spans of one QSA layer with `rows` positions, in Save order: K, V, raw keys, pool. ConversationStateIO::Spans
// and the block plan of the NVMe tier (state_blocks.h) both use it, so they cannot disagree.
inline std::array<QsaSnapshotSpan, 4> QsaSnapshotSpans(int kvl, int raw_slots, size_t rows) {
  const size_t kv = KvqPositionBytes(kvl), pool = 4096 * sizeof(float);
  const size_t raw_unit = raw_slots ? 0 : kQsaRawDim * sizeof(float);
  return {{{rows * kv, 1, kv},
           {rows * kv, 1, kv},
           {QsaRawSnapshotBytes(raw_slots, rows), raw_slots ? size_t(0) : size_t(1), raw_unit},
           {(rows + 127) / 128 * pool, 128, pool}}};
}

template<class Caches, class Clear>
bool ResetQsaCache(const Caches& caches, int layer, int context, Clear clear) {
  const auto found = caches.find(layer);
  if (found == caches.end()) return false;
  const auto& cache = found->second;
  if (cache.raw_slots != 0 && cache.raw_slots != kQsaRawSlots)
    throw std::invalid_argument("unsupported raw cache layout");
  const auto bytes = QsaCacheBytes(context, cache.raw_slots != 0);
  const std::array<void*, 4> pointers{cache.k, cache.v, cache.ik_raw, cache.ik_pool};
  for (size_t i = 0; i < pointers.size(); ++i) clear(pointers[i], bytes[i]);
  return true;
}

class CacheBufferTransaction {
 public:
  CacheBufferTransaction(const std::array<size_t, 4>& bytes,
      const std::function<void*(size_t)>& allocate, std::function<void(void*)> release)
      : release_(std::move(release)) {
    try {
      for (size_t i = 0; i < pointers_.size(); ++i) {
        pointers_[i] = allocate(bytes[i]);
        if (!pointers_[i]) throw std::runtime_error("QSA cache allocation failed; reduce context or stream count");
      }
    } catch (...) { Clear(); throw; }
  }
  CacheBufferTransaction(const CacheBufferTransaction&) = delete;
  CacheBufferTransaction& operator=(const CacheBufferTransaction&) = delete;
  ~CacheBufferTransaction() { Clear(); }
  const std::array<void*, 4>& Pointers() const { return pointers_; }
  void Commit() { committed_ = true; }
 private:
  void Clear() {
    if (!committed_) for (void* p : pointers_) if (p) release_(p);
    pointers_ = {};
  }
  std::array<void*, 4> pointers_{};
  std::function<void(void*)> release_;
  bool committed_ = false;
};
}  // namespace crucible
