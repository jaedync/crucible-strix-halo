// the plan of a lazy save. The full blocks of a slot tail below its settled rows stay on the
// device, and the disk writer reads them in chunks through a small pinned staging buffer (LazySource). The rest of the
// state (the fixed state, the rows of the partial blocks, the MTP lane, the stacks) is copied to the host at once.
// Pure functions on offsets; CPU only.
#pragma once
#include "snapshot_bytes.h"
#include "state_blocks.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace crucible {
// The snapshot ranges of every lane-0 block whose rows all lie below `settled`, sorted and merged. Empty when the
// geometry gives no blocks (`block_rows` 0, or spans that the plan does not explain): the caller then saves everything.
inline std::vector<StreamRange> LazyIntervals(const SnapshotGeometry& geometry, size_t block_rows, size_t settled) {
  const BlockPlan plan = PlanBlocks(geometry, block_rows);
  std::vector<StreamRange> out;
  if (plan.whole) return out;
  for (const auto& block : plan.blocks)
    if (block.lane == 0 && (block.index + 1) * block_rows <= settled)
      out.insert(out.end(), block.ranges.begin(), block.ranges.end());
  std::sort(out.begin(), out.end(), [](const StreamRange& a, const StreamRange& b) { return a.offset < b.offset; });
  std::vector<StreamRange> merged;
  for (const auto& range : out) AppendRange(merged, range.offset, range.bytes);
  return merged;
}
inline uint64_t LazyBytesOf(std::span<const StreamRange> intervals) { return RangeBytes(intervals); }

// Cuts the spans (a device pointer and a byte count each, in snapshot order) at the lazy intervals: the bytes inside an
// interval become parts that read `source`; the others are copied. The parts cover the spans exactly once.
template<class Span>
std::vector<SnapshotBytes::Part> SplitLazyParts(const std::vector<Span>& spans, std::span<const StreamRange> lazy,
                                                const std::shared_ptr<LazySource>& source) {
  std::vector<SnapshotBytes::Part> parts;
  uint64_t offset = 0;
  size_t first = 0;
  const auto add = [&](const uint8_t* device, uint64_t bytes, bool read) {
    if (!bytes) return;
    SnapshotBytes::Part part;
    part.device = device;
    part.bytes = size_t(bytes);
    if (read) part.lazy = source;
    parts.push_back(std::move(part));
  };
  for (const auto& span : spans) {
    const auto* base = static_cast<const uint8_t*>(span.pointer);
    const uint64_t end = offset + span.bytes;
    uint64_t cursor = offset;
    while (first < lazy.size() && lazy[first].offset + lazy[first].bytes <= offset) ++first;
    for (size_t i = first; i < lazy.size() && lazy[i].offset < end; ++i) {
      const uint64_t from = std::max<uint64_t>(lazy[i].offset, offset), to = std::min<uint64_t>(lazy[i].offset + lazy[i].bytes, end);
      if (from >= to) continue;
      add(base + (cursor - offset), from - cursor, false);
      add(base + (from - offset), to - from, true);
      cursor = to;
    }
    add(base + (cursor - offset), end - cursor, false);
    offset = end;
  }
  return parts;
}
}  // namespace crucible
