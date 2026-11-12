// the lazy save plan. CPU only (the fake runtime copies with memcpy).
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>
template<class T> int hipHostMalloc(T** out, size_t n) { *out = static_cast<T*>(std::malloc(n)); return 0; }
int hipHostFree(void* p) { std::free(p); return 0; }
int hipStreamSynchronize(int) { return 0; }
constexpr int hipMemcpyDeviceToHost = 0, hipMemcpyHostToDevice = 1;
int hipMemcpy(void* out, const void* in, size_t n, int) { std::memcpy(out, in, n); return 0; }
#define HIP_OK(expr) assert((expr) == 0)
#include "../engine/layers/cache_storage.h"
#include "../engine/runtime/lazy_snapshot.h"
using namespace crucible;

struct Span { const void* pointer; size_t bytes; };

int main() {
  // One trunk layer with a compact raw ring (kQsaRawSlots): K, V and the pool grow by rows; the ring stays in `rest`.
  // Compact raw ring (the engine's compact state) and the raw key rows that grow with the context (raw_slots 0).
  for (const auto& layout : std::vector<std::vector<std::pair<int, int>>>{{{0, kQsaRawSlots}, {1, kQsaRawSlots}}, {{0, 0}, {1, 0}}})
  for (const size_t rows : {size_t(4095), size_t(4096), size_t(9000), size_t(12288), size_t(20000)}) {
    SnapshotGeometry g;
    g.layout = layout;
    g.rows = rows;
    g.mpos = 0;
    g.head = 1000;
    g.tail = 777;
    uint64_t total = g.head + g.tail;
    for (const auto& [layer, raw] : layout) {
      (void)layer;
      for (const auto& span : QsaSnapshotSpans(0, raw, rows)) total += span.bytes;
    }
    g.total = total;
    for (const size_t settled : {size_t(0), size_t(4095), rows, rows > 100 ? rows - 100 : 0}) {
      const auto lazy = LazyIntervals(g, kStateBlockRows, settled);
      const size_t full = std::min(rows, settled) / kStateBlockRows;
      if (!full) { assert(lazy.empty()); continue; }
      // Lazy bytes: `full` blocks of every row-growing span of every layer.
      uint64_t expected = 0;
      for (const auto& [layer, raw] : layout) {
        (void)layer;
        for (const auto& span : QsaSnapshotSpans(0, raw, rows))
          if (span.unit_rows) expected += full * (kStateBlockRows / span.unit_rows * span.unit_bytes);
      }
      assert(LazyBytesOf(lazy) == expected);
      // The parts cover the spans exactly once, in order; the lazy ones name `source`, the others are copied.
      std::vector<std::vector<uint8_t>> device;
      std::vector<Span> spans;
      auto add = [&](size_t bytes) {
        device.emplace_back(bytes);
        for (size_t i = 0; i < bytes; ++i) device.back()[i] = uint8_t(i * 31 + device.size());
      };
      add(g.head);
      for (const auto& [layer, raw] : layout) {
        (void)layer;
        for (const auto& span : QsaSnapshotSpans(0, raw, rows)) if (span.bytes) add(span.bytes);
      }
      add(g.tail);
      for (const auto& d : device) spans.push_back({d.data(), d.size()});
      auto source = std::make_shared<LazySource>([](const void* from, void* to, size_t n) { std::memcpy(to, from, n); }, full * kStateBlockRows);
      const auto parts = SplitLazyParts(spans, lazy, source);
      uint64_t eager = 0, lazily = 0;
      for (const auto& part : parts) (part.lazy ? lazily : eager) += part.bytes;
      assert(lazily == expected && eager + lazily == total);
      std::vector<SnapshotRange> copies;
      SnapshotBytes bytes = SnapshotBytes::Compose(parts, copies);
      CopySnapshotRanges(copies, bytes, true);
      std::vector<uint8_t> want;
      for (const auto& d : device) want.insert(want.end(), d.begin(), d.end());
      assert(bytes.size() == total && bytes.HostBytes() == eager && bytes.Copy() == want);
      // The block plan of the lazy snapshot reads the same bytes as the plan of a full one (what the writer hashes).
      const BlockPlan plan = PlanBlocks(g, kStateBlockRows);
      for (const auto& block : plan.blocks) {
        std::vector<uint8_t> a(RangeBytes(block.ranges)), b(a.size());
        GatherRanges(bytes, block.ranges, 0, a.size(), a.data());
        uint64_t at = 0;
        for (const auto& range : block.ranges) {
          std::memcpy(b.data() + at, want.data() + range.offset, range.bytes);
          at += range.bytes;
        }
        assert(a == b);
      }
      const SnapshotBytes full_copy = bytes;
      assert(DigestRanges(bytes, plan.rest) == DigestRanges(full_copy, plan.rest));
    }
  }
  // block_rows 0 or a geometry that the plan does not explain: nothing is lazy.
  SnapshotGeometry g;
  const std::vector<std::pair<int, int>> one{{0, kQsaRawSlots}};
  g.layout = one, g.rows = 10000, g.head = 0, g.tail = 0, g.total = 5;
  assert(LazyIntervals(g, kStateBlockRows, 10000).empty() && LazyIntervals(g, 0, 10000).empty());
  std::printf("lazy_snapshot_test passed\n");
  return 0;
}
