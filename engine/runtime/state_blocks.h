// Block storage for the NVMe tier. The per-position spans of every 4,096 rows of a
// conversation snapshot go to one block file, named by the SHA-256 of its bytes, so two states that hold the same
// bytes share the file. This file holds the pure parts: the block plan, the range copies and the block file format.
// state_disk.h owns the files, the index and the threads. CPU only.
#pragma once
#include "sha256.h"
#include "snapshot_bytes.h"
#include "tenant.h"
#include "../layers/cache_storage.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace crucible {
inline constexpr size_t kStateBlockRows = 4096;
// Who a state belongs to (tenant.h): the tenant of its policy. A state and a block are shared only inside one
// tenant, so no tenant can learn from a fast spill or read that another one holds the same state.
using StateDigest = std::array<uint8_t, 32>;

// Bytes [offset, offset + bytes) of a snapshot, as one piece of a file stream (block or state file).
struct StreamRange {
  uint64_t offset = 0, bytes = 0;
  bool operator==(const StreamRange&) const = default;
};
inline uint64_t RangeBytes(std::span<const StreamRange> ranges) {
  uint64_t bytes = 0;
  for (const auto& range : ranges) bytes += range.bytes;
  return bytes;
}
// A block file's content: the listed snapshot ranges, in order. `lane` 0: the trunk QSA layers; 1: the MTP layer
// (its own row count, mpos). `index`: the block number in its lane.
struct PlannedBlock {
  int lane = 0;
  uint64_t index = 0;
  std::vector<StreamRange> ranges;
};
struct BlockPlan {
  std::vector<PlannedBlock> blocks;
  std::vector<StreamRange> rest;  // everything else, in snapshot order: it goes to the state file
  bool whole = false;               // the spans did not explain the snapshot size: no blocks
};
// The geometry of a snapshot that the plan needs (ConversationStateIO::Spans order): `head` bytes of fixed-size state,
// then each QSA layer of `layout` (layer `mtp_layer` holds `mpos` rows, the others `rows`), then `tail` bytes (the
// request stacks). `kvl`: the KV layout of the K and V rows.
struct SnapshotGeometry {
  std::span<const std::pair<int, int>> layout;
  int kvl = 0;
  size_t rows = 0, mpos = 0;
  uint64_t head = 0, tail = 0, total = 0;
  int mtp_layer = kLayers;
};
// Adds `range` to `ranges`, merged with the last one when they touch (fewer ranges, the same bytes).
inline void AppendRange(std::vector<StreamRange>& ranges, uint64_t offset, uint64_t bytes) {
  if (!bytes) return;
  if (!ranges.empty() && ranges.back().offset + ranges.back().bytes == offset) ranges.back().bytes += bytes;
  else ranges.push_back({offset, bytes});
}
// Plans the blocks of one snapshot. Block b of a lane holds rows [b block_rows, (b + 1) block_rows) of every span that
// grows by rows, and exists only when its rows are complete. `block_rows` 0, or a size that the spans do not explain,
// gives no blocks: the state is then one state file and still restores the same bytes.
inline BlockPlan PlanBlocks(const SnapshotGeometry& g, size_t block_rows) {
  BlockPlan plan;
  const auto whole = [&] {
    plan = {};
    plan.whole = true;
    AppendRange(plan.rest, 0, g.total);
    return plan;
  };
  if (!block_rows || block_rows % 128) return whole();
  std::vector<PlannedBlock> lanes[2];
  uint64_t offset = 0;
  AppendRange(plan.rest, 0, g.head);
  offset += g.head;
  try {
    for (const auto& [layer, raw_slots] : g.layout) {
      const int lane = layer == g.mtp_layer ? 1 : 0;
      const size_t rows = lane ? g.mpos : g.rows, full = rows / block_rows;
      auto& blocks = lanes[lane];
      while (blocks.size() < full) blocks.push_back({lane, blocks.size(), {}});
      for (const auto& span : QsaSnapshotSpans(g.kvl, raw_slots, rows)) {
        if (!span.unit_rows) { AppendRange(plan.rest, offset, span.bytes); offset += span.bytes; continue; }
        const uint64_t step = block_rows / span.unit_rows * span.unit_bytes;
        for (size_t b = 0; b < full; ++b) AppendRange(blocks[b].ranges, offset + b * step, step);
        AppendRange(plan.rest, offset + full * step, span.bytes - full * step);
        offset += span.bytes;
      }
    }
  } catch (const std::invalid_argument&) {
    return whole();  // a KV layout that this binary does not know
  }
  AppendRange(plan.rest, offset, g.tail);
  if (offset + g.tail != g.total) return whole();
  for (auto& lane : lanes)
    for (auto& block : lane) plan.blocks.push_back(std::move(block));
  return plan;
}
// True when the ranges of every block and the rest cover [0, total) exactly once. The reader checks it on the state
// file's own lists, so a restore never depends on this binary's plan.
inline bool RangesCover(const std::vector<std::span<const StreamRange>>& lists, uint64_t total) {
  std::vector<StreamRange> all;
  for (const auto& list : lists) all.insert(all.end(), list.begin(), list.end());
  std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
  uint64_t at = 0;
  for (const auto& range : all) {
    if (!range.bytes || range.offset != at || range.bytes > total - at) return false;
    at += range.bytes;
  }
  return at == total;
}
// Calls f(snapshot_offset, stream_at, count) for the stream bytes [from, from + bytes) of `ranges` (the ranges laid end
// to end form the stream of one file).
template<class F>
void ForEachStreamPiece(std::span<const StreamRange> ranges, uint64_t from, uint64_t bytes, F f) {
  uint64_t start = 0;
  for (const auto& range : ranges) {
    if (!bytes) return;
    const uint64_t end = start + range.bytes;
    if (from < end) {
      const uint64_t inside = from - start, count = std::min(bytes, range.bytes - inside);
      f(range.offset + inside, from, count);
      from += count, bytes -= count;
    }
    start = end;
  }
}
// Copies the stream bytes [from, from + bytes) of `ranges` out of `source` to `out`.
inline void GatherRanges(const SnapshotBytes& source, std::span<const StreamRange> ranges, uint64_t from,
                         uint64_t bytes, uint8_t* out) {
  ForEachStreamPiece(ranges, from, bytes, [&](uint64_t offset, uint64_t at, uint64_t count) {
    source.ForEachPiece(size_t(offset), size_t(count), [&](const SnapshotChunk& chunk, size_t inside, size_t n, size_t done) {
      std::memcpy(out + (at - from) + done, chunk.host + inside, n);
    });
  });
}
// Copies `bytes` stream bytes from `in` to their snapshot offsets in `target` (the stream starts at `from`).
inline void ScatterRanges(SnapshotBytes& target, std::span<const StreamRange> ranges, uint64_t from, uint64_t bytes,
                          const uint8_t* in) {
  ForEachStreamPiece(ranges, from, bytes, [&](uint64_t offset, uint64_t at, uint64_t count) {
    target.ForEachPiece(size_t(offset), size_t(count), [&](const SnapshotChunk& chunk, size_t inside, size_t n, size_t done) {
      std::memcpy(chunk.host + inside, in + (at - from) + done, n);
    });
  });
}
// The SHA-256 of the stream of `ranges`, read from the snapshot's own memory (composed snapshots too).
inline StateDigest DigestRanges(const SnapshotBytes& source, std::span<const StreamRange> ranges) {
  Sha256 hash;
  for (const auto& range : ranges)
    source.ForEachPiece(size_t(range.offset), size_t(range.bytes), [&](const SnapshotChunk& chunk, size_t inside, size_t n, size_t) {
      hash.Update(chunk.host + inside, n);
    });
  return hash.Digest();
}
inline std::string DigestHex(const StateDigest& digest) {
  static const char* hex = "0123456789abcdef";
  std::string out;
  for (const uint8_t b : digest) out += hex[b >> 4], out += hex[b & 15];
  return out;
}
// A block file's name: the identity, the tenant, the digest and the size. The token key is not in it: the digest
// decides that two blocks hold the same bytes.
inline std::string BlockFileNameFor(const std::string& identity, const StateTenant& tenant, const StateDigest& digest,
                                    uint64_t bytes) {
  Sha256 hash;
  const std::string tag = "crucible-block-v1";
  hash.Update(reinterpret_cast<const uint8_t*>(tag.data()), tag.size());
  const uint64_t size = identity.size();
  hash.Update(reinterpret_cast<const uint8_t*>(&size), 8);
  hash.Update(reinterpret_cast<const uint8_t*>(identity.data()), identity.size());
  hash.Update(tenant.data(), tenant.size());
  hash.Update(digest.data(), digest.size());
  hash.Update(reinterpret_cast<const uint8_t*>(&bytes), 8);
  return hash.HexDigest() + ".block";
}
}  // namespace crucible
