#include "../engine/layers/cache_storage.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>
using namespace crucible;

template<class F> void Reject(F run) {
  bool rejected = false;
  try { run(); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
}

void AllocationTest() {
  for (int fail = 0; fail < 4; ++fail) {
    std::map<void*, size_t> live;
    int calls = 0;
    Reject([&] {
      CacheBufferTransaction tx({32, 64, 128, 256}, [&](size_t bytes) -> void* {
        if (calls++ == fail) return nullptr;
        void* p = new char[bytes]; live[p] = bytes; return p;
      }, [&](void* p) { assert(live.erase(p) == 1); delete[] static_cast<char*>(p); });
    });
    assert(live.empty());
  }
  std::map<void*, size_t> live;
  auto alloc = [&](size_t bytes) { void* p = new char[bytes]; live[p] = bytes; return p; };
  auto release = [&](void* p) { assert(live.erase(p) == 1); delete[] static_cast<char*>(p); };
  Reject([&] { CacheBufferTransaction tx({1, 2, 3, 4}, alloc, release); throw std::runtime_error("publish"); });
  assert(live.empty());
  std::array<void*, 4> owned;
  { CacheBufferTransaction tx({1, 2, 3, 4}, alloc, release); owned = tx.Pointers(); tx.Commit(); }
  assert(live.size() == 4);
  for (void* p : owned) release(p);
}

void LifetimeTest() {
  for (int pos : {0, 1, 2, 3, 13, 14, 15, 16, 1023, 65535, 131071}) {
    for (int rows : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 2048}) {
      std::vector<int> ring(kQsaRawSlots * kQsaRawDim, -1);
      for (int p = std::max(0, pos - kQsaRawSlots); p < pos; ++p)
        for (int j = 0; j < kQsaRawDim; ++j) ring[QsaRawOffset(kQsaRawSlots, p, j)] = p * 128 + j;
      if (!QsaRawUsesChunk(kQsaRawSlots, rows))
        for (int p = pos; p < pos + rows; ++p)
          for (int j = 0; j < 128; ++j) ring[QsaRawOffset(kQsaRawSlots, p, j)] = p * 128 + j;
      for (int b = pos / 4; b < (pos + rows) / 4; ++b)
        for (int t = 0; t < 4; ++t) for (int j = 0; j < 128; ++j) {
          const int p = b * 4 + t;
          const int value = QsaRawFromChunk(kQsaRawSlots, rows, pos, p) ? p * 128 + j
                            : ring[QsaRawOffset(kQsaRawSlots, p, j)];
          assert(value == p * 128 + j);
        }
      if (QsaRawUsesChunk(kQsaRawSlots, rows))
        for (int p = pos + std::max(0, rows - kQsaRawSlots); p < pos + rows; ++p)
          for (int j = 0; j < 128; ++j) ring[QsaRawOffset(kQsaRawSlots, p, j)] = p * 128 + j;
      const int keep_begin = rows <= 8 ? 1 : rows;
      for (int keep = keep_begin; keep <= rows; ++keep)
        for (int p = std::max(0, (pos + keep) / 4 * 4); p < pos + keep; ++p)
          for (int j = 0; j < 128; ++j) assert(ring[QsaRawOffset(kQsaRawSlots, p, j)] == p * 128 + j);
    }
  }
}

int main() {
  assert(!CompactStateFlag(nullptr) && !CompactStateFlag("0") && CompactStateFlag("1"));
  Reject([] { CompactStateFlag("yes"); });
  // Compact state passed its matched device checks.
  // It stays opt-in: unset and 0 keep full storage, and malformed values still fail.
  assert(!EngineCompactState(nullptr) && !EngineCompactState("0") && EngineCompactState("1"));
  Reject([] { EngineCompactState("yes"); });
  Reject([] { EngineCompactState(""); });
  assert(!CacheAllocationFits(9, 1, 9));
  assert(!CacheAllocationFits(10, 2, 9));
  assert(CacheAllocationFits(10, 1, 9));
  assert(CacheAllocationFits(size_t(-1), 1, 9));
  for (int ctx : {1, 127, 128, 129, 16384, 65536, 131072, 262144}) {
    const auto full = QsaCacheBytes(ctx, false), bounded = QsaCacheBytes(ctx, true);
    assert(full[0] == size_t(ctx) * 1024 && full[1] == full[0]);
    assert(full[2] == size_t(ctx) * 512 && bounded[2] == 8192);
    assert(full[3] == size_t((ctx + 127) / 128) * 16384);
    assert(QsaRawSnapshotBytes(16, ctx) == 8192 && QsaRawSnapshotBytes(16, 0) == 0);
    assert(QsaRawSnapshotBytes(0, ctx) == size_t(ctx) * 512);
    const size_t one_bounded = bounded[0] + bounded[1] + bounded[2] + bounded[3];
    assert(QsaSequenceCacheBytes(ctx, false, false) == QsaSequenceCacheBytes(ctx, false, true));
    assert(QsaSequenceCacheBytes(ctx, true, true) - QsaSequenceCacheBytes(ctx, true, false) == one_bounded);
    if (ctx >= 16)
      assert(QsaSequenceCacheBytes(ctx, false, true) - QsaSequenceCacheBytes(ctx, true, true)
             == 13ull * (ctx - 16) * 512);
    // Serve admission charges the lazy MTP cache, so a later promotion cannot fail on memory.
    for (bool compact : {false, true})
      assert(ServeQsaCacheBytes(ctx, compact) == QsaSequenceCacheBytes(ctx, compact, true));
    assert(ServeQsaCacheBytes(ctx, true) > QsaSequenceCacheBytes(ctx, true, false));
    // Admission also charges the MTP cache that an existing compact slot still lacks.
    assert(MissingMtpCacheBytes(ctx, true, false) == QsaSequenceCacheBytes(ctx, true, true) - QsaSequenceCacheBytes(ctx, true, false));
    assert(MissingMtpCacheBytes(ctx, true, true) == 0 && MissingMtpCacheBytes(ctx, false, false) == 0);
  }
  // the MTP cache holds its own row count; the 12 trunk caches keep the context.
  for (int ctx : {32768, 106496, 131072, 196608, 262144})
    for (bool compact : {false, true}) {
      const int rows = std::min(ctx, 33280);
      assert(QsaSequenceCacheBytes(ctx, compact, true, ctx) == QsaSequenceCacheBytes(ctx, compact, true));
      assert(ServeQsaCacheBytes(ctx, compact, ctx) == ServeQsaCacheBytes(ctx, compact));
      assert(QsaSequenceCacheBytes(ctx, true, false, rows) == QsaSequenceCacheBytes(ctx, true, false));
      const size_t saved = ServeQsaCacheBytes(ctx, compact) - ServeQsaCacheBytes(ctx, compact, rows);
      assert(saved == size_t(ctx - rows) * (compact ? 2176 : 2688));
      assert(MissingMtpCacheBytes(rows, true, false) ==
             QsaSequenceCacheBytes(ctx, true, true, rows) - QsaSequenceCacheBytes(ctx, true, false, rows));
    }
  assert(ServeQsaCacheBytes(196608, false) - ServeQsaCacheBytes(196608, false, 33280) == 439025664ull);
  Reject([] { QsaSequenceCacheBytes(1024, false, true, 1025); });
  Reject([] { QsaSequenceCacheBytes(1024, false, true, 0); });
  Reject([] { QsaCacheBytes(0, true); });
  // YaRN contexts: the storage formula holds to 1,048,576; the mode decides the cap.
  assert(QsaCacheBytes(524288, true)[0] == size_t(524288) * 1024 && QsaCacheBytes(1048576, false)[2] == size_t(1048576) * 512);
  Reject([] { QsaCacheBytes(1048577, true); });
  AllocationTest(); LifetimeTest();
}
