// Included after Engine. Snapshots never refer to mutable engine storage.
#pragma once
#include "conversation_resume.h"
#include "../layers/cache_storage.h"
#include "../gfx1151/pinned_allocator.h"
#include "snapshot_bytes.h"
#include <memory>
#include <span>
#include <stdexcept>

struct ConversationSnapshot {
  std::vector<int> hist;
  RetainedPrefix retained;
  ConversationResume resume;
  SnapshotBytes bytes;
  std::vector<std::pair<int, int>> qsa_layout;
};
// a held snapshot whose rows [0, rows) the stream still holds bit for bit (BatchPrefixCache tracks it per slot).
struct SaveBase {
  std::shared_ptr<const ConversationSnapshot> snapshot;
  size_t rows = 0;
};
struct ConversationStateIO {
  // The bytes that a save of `n` rows takes from `base` instead of the device: per trunk QSA layer, K and V rows,
  // raw key rows (not the compact ring) and the pool blocks below the shared rows. The MTP layer, the fixed-size state
  // and the stacks are always copied. Empty when the trunk layouts or the base size do not match.
  struct SharedSpan { const void* pointer; size_t base_offset, bytes; };
  static constexpr size_t kSharedRowStep = 128;  // a pool block: rows below it share no block with later rows
  // `n`: the settled rows of the new save (Settled: its length less its pending replay rows, which the last verify
  // pass wrote after the base was held; a review L3).
  static size_t Settled(const ConversationResume& r) {
    const size_t replay = size_t(std::max(r.replay, 0));
    return r.length > replay ? r.length - replay : 0;
  }
  static std::vector<SharedSpan> SharedSpans(const SequenceState& state, size_t n, const SaveBase& base) {
    if (!base.snapshot) return {};
    const ConversationSnapshot& held = *base.snapshot;
    const size_t held_rows = held.hist.size(), replay = size_t(std::max(held.resume.replay, 0));
    size_t rows = std::min({base.rows, n, held_rows > replay ? held_rows - replay : 0});
    rows -= rows % kSharedRowStep;
    if (!rows) return {};
    auto trunk = held.qsa_layout;
    std::erase_if(trunk, [](const auto& item) { return item.first == kLayers; });
    if (trunk != Layout(state, false)) return {};
    size_t offset = 0;
    for (const auto& span : state.FixedSpans()) offset += span.bytes;
    std::vector<SharedSpan> shared;
    for (const auto& [l, c] : state.qc) {
      if (l == kLayers) continue;
      const size_t kv = held_rows * 512 * sizeof(__half), raw = QsaRawSnapshotBytes(c.raw_slots, held_rows);
      const size_t pool = QsaPoolFloats(int(held_rows)) * sizeof(float);
      shared.push_back({c.k, offset, rows * 512 * sizeof(__half)});
      shared.push_back({c.v, offset + kv, rows * 512 * sizeof(__half)});
      if (!c.raw_slots) shared.push_back({c.ik_raw, offset + 2 * kv, QsaRawSnapshotBytes(0, rows)});
      shared.push_back({c.ik_pool, offset + 2 * kv + raw, QsaPoolFloats(int(rows)) * sizeof(float)});
      offset += 2 * kv + raw + pool;
    }
    // The rest of the base: its MTP rows (if any) and the two stacks. Another size means another layout: share nothing.
    const auto mtp = std::find_if(held.qsa_layout.begin(), held.qsa_layout.end(),
                                  [](const auto& item) { return item.first == kLayers; });
    const size_t mpos = size_t(std::max(held.resume.mpos, 0));
    if (mtp != held.qsa_layout.end())
      offset += 2 * mpos * 512 * sizeof(__half) + QsaRawSnapshotBytes(mtp->second, mpos) +
                (mpos ? QsaPoolFloats(int(mpos)) * sizeof(float) : 0);
    offset += sizeof(float) * (kMaxRows + 1) * kWide + sizeof(float) * kWide;
    if (offset != held.bytes.size()) return {};
    return shared;
  }
  // The bytes of those spans that a save shares (whole host chunks only, SnapshotBytes::Compose).
  static size_t SharedBytes(const SequenceState& state, size_t n, const SaveBase& base) {
    size_t bytes = 0;
    for (const auto& span : SharedSpans(state, n, base)) bytes += base.snapshot->bytes.ShareableBytes(span.base_offset, span.bytes);
    return bytes;
  }
  // The chunks of the base that those spans share: the save keeps them, so an eviction of the base does not free them.
  static std::vector<const void*> SharedOwners(const SequenceState& state, size_t n, const SaveBase& base) {
    std::vector<const void*> owners;
    for (const auto& span : SharedSpans(state, n, base)) {
      const auto more = base.snapshot->bytes.ShareableOwners(span.base_offset, span.bytes);
      owners.insert(owners.end(), more.begin(), more.end());
    }
    return owners;
  }
  static void Validate(Engine& e, const SequenceState& state, const ConversationResume& r,
                       size_t length, float* stk, float* carry, bool require_mtp = true) {
    const bool owned = std::any_of(e.sequences_.begin(), e.sequences_.end(),
                                  [&](const auto& s) { return s.get() == &state; });
    if (!owned || !stk || !carry) throw std::runtime_error("snapshot needs an owned stream and request stacks");
    if (r.length != length || length > size_t(state.capacity) || r.mpos < 0 || r.mpos > state.mtp_capacity)
      throw std::runtime_error("invalid conversation snapshot position");
    if (require_mtp && r.mpos > 0 && !state.qc.contains(kLayers))
      throw std::runtime_error("snapshot MTP position needs an allocated MTP cache");
    if (!r.logits.empty() && r.logits.size() != kVocab)
      throw std::runtime_error("snapshot target logit shape changed");
  }
  static std::vector<std::pair<int, int>> Layout(const SequenceState& state, bool include_mtp = true) {
    std::vector<std::pair<int, int>> layout;
    for (const auto& [layer, cache] : state.qc) {
      if (layer == kLayers && !include_mtp) continue;
      if (cache.raw_slots != 0 && cache.raw_slots != kQsaRawSlots)
        throw std::runtime_error("unsupported raw cache layout");
      layout.emplace_back(layer, cache.raw_slots);
    }
    return layout;
  }
  // The snapshot bytes before the QSA layers (the fixed-size state) and after them (the request stacks), in Spans
  // order. The NVMe tier's block plan needs them (state_blocks.h); another size there gives one state file.
  static uint64_t HeadBytes(const SequenceState& state) {
    uint64_t bytes = 0;
    for (const auto& span : state.FixedSpans()) bytes += span.bytes;
    return bytes;
  }
  static constexpr uint64_t kTailBytes = sizeof(float) * (kMaxRows + 1) * kWide + sizeof(float) * kWide;
  static std::vector<DeviceSpan> Spans(const SequenceState& state, size_t n, int mpos,
                                       float* stk, float* carry, bool include_mtp = true) {
    std::vector<DeviceSpan> spans;
    const auto add = [&](void* p, size_t bytes) { if (bytes) spans.push_back({p, bytes}); };
    for (const auto& span : state.FixedSpans()) add(const_cast<void*>(span.pointer), span.bytes);
    for (const auto& [l, c] : state.qc) {
      if (l == kLayers && !include_mtp) continue;
      const size_t length = l == kLayers ? size_t(mpos) : n;
      // The sizes of the NVMe tier's block plan (state_blocks.h): one formula for both.
      const auto sizes = QsaSnapshotSpans(0, c.raw_slots, length);
      add(c.k, sizes[0].bytes);
      add(c.v, sizes[1].bytes);
      add(c.ik_raw, sizes[2].bytes);
      add(c.ik_pool, sizes[3].bytes);
    }
    add(stk, sizeof(float) * (kMaxRows + 1) * kWide);
    add(carry, sizeof(float) * kWide);
    return spans;
  }
  // The MTP rows that a restore allocates the MTP cache for (the snapshot holds them; the stream does not yet).
  static size_t MtpBytes(const SequenceState& state, size_t n) {
    return n * 2048 + QsaRawSnapshotBytes(state.compact_state ? kQsaRawSlots : 0, n) + QsaPoolFloats(int(n)) * sizeof(float);
  }
  // The bytes that Save copies for `rows` rows of this stream, with MTP rows up to `rows` when `mtp`, also before the
  // stream has its MTP cache.
  static size_t SaveBytes(const SequenceState& state, size_t rows, bool mtp) {
    rows = std::min(rows, size_t(state.capacity));
    const size_t mpos = mtp ? std::min(rows, size_t(state.mtp_capacity)) : 0;
    size_t bytes = 0;
    for (const auto& span : Spans(state, rows, int(mpos), nullptr, nullptr)) bytes += span.bytes;
    if (mpos && !state.qc.contains(kLayers)) bytes += MtpBytes(state, mpos);
    return bytes;
  }
  static size_t Bytes(Engine& e, const SequenceState& state, const ConversationResume& r,
                      float* stk, float* carry) {
    return BytesAt(e, state, r, state.hist.size(), stk, carry);
  }
  // The bytes of a save of the first `length` rows (a Frontier point below the stream's rows, slot_checkpoint_io.h).
  static size_t BytesAt(Engine& e, const SequenceState& state, const ConversationResume& r, size_t length,
                        float* stk, float* carry) {
    Validate(e, state, r, length, stk, carry);
    size_t bytes = sizeof(ConversationSnapshot) + length * sizeof(int) * 2;
    bytes += r.logits.size() * sizeof(float) + r.pending_tokens.size() * sizeof(int);
    bytes += Layout(state).size() * sizeof(std::pair<int, int>);
    for (const auto& span : Spans(state, r.length, r.mpos, stk, carry)) bytes += span.bytes;
    return bytes;
  }
  // With `base`, the snapshot shares the base's whole chunks of SharedSpans and copies only the other bytes.
  static std::shared_ptr<const ConversationSnapshot> Save(Engine& e, const SequenceState& state,
      const ConversationResume& r, float* stk, float* carry, const SaveBase& base = {}) {
    return SaveAt(e, state, r, stk, carry, base, state.hist, state.retained, {});
  }
  // A span of the stream whose bytes a save reads from another device buffer of the same size.
  struct SpanSource { const void* live; const void* source; size_t bytes; };
  // Points each span of `spans` that `sources` names at its other buffer (SaveAt; SlotCheckpointIO::CopyTo).
  static void UseSources(std::vector<DeviceSpan>& spans, std::span<const SpanSource> sources) {
    for (auto& span : spans)
      for (const auto& source : sources)
        if (span.pointer == source.live) {
          if (span.bytes != source.bytes) throw std::logic_error("snapshot span source has another size");
          span.pointer = source.source;
        }
  }
  // Save of the first `hist.size()` rows, with the host-side history and counters of that state, and some spans
  // read from `sources` (a Frontier point: SlotCheckpointIO::SaveFrom). The other spans are the stream's own.
  static std::shared_ptr<const ConversationSnapshot> SaveAt(Engine& e, const SequenceState& state,
      const ConversationResume& r, float* stk, float* carry, const SaveBase& base, std::span<const int> hist,
      const RetainedPrefix& retained, std::span<const SpanSource> sources) {
    Validate(e, state, r, hist.size(), stk, carry);
    auto snapshot = std::make_shared<ConversationSnapshot>();
    snapshot->hist.assign(hist.begin(), hist.end());
    snapshot->retained = retained;
    snapshot->resume = r;
    snapshot->qsa_layout = Layout(state);
    auto spans = Spans(state, r.length, r.mpos, stk, carry);
    UseSources(spans, sources);
    const auto shared = SharedSpans(state, Settled(r), base);
    if (shared.empty()) {
      size_t total = 0;
      for (const auto& span : spans) total += span.bytes;
      snapshot->bytes = SnapshotBytes(total);
      // Publish only a complete snapshot before decode or a following request can overwrite its source: the copy waits.
      CopySnapshotSpans(spans, snapshot->bytes, true);
      return snapshot;
    }
    std::vector<SnapshotBytes::Part> parts;
    for (const auto& span : spans) {
      const auto found = std::find_if(shared.begin(), shared.end(), [&](const auto& item) { return item.pointer == span.pointer; });
      if (found == shared.end() || !found->bytes) {
        parts.push_back({span.pointer, span.bytes});
        continue;
      }
      parts.push_back({span.pointer, found->bytes, &base.snapshot->bytes, found->base_offset});
      if (span.bytes > found->bytes)
        parts.push_back({static_cast<const uint8_t*>(span.pointer) + found->bytes, span.bytes - found->bytes});
    }
    std::vector<SnapshotRange> copies;
    snapshot->bytes = SnapshotBytes::Compose(parts, copies);
    CopySnapshotRanges(copies, snapshot->bytes, true);
    return snapshot;
  }
  static ConversationResume Restore(Engine& e, SequenceState& state, const ConversationSnapshot& snapshot,
                                    float* stk, float* carry) {
    Validate(e, state, snapshot.resume, snapshot.hist.size(), stk, carry, false);
    const bool mtp = std::any_of(snapshot.qsa_layout.begin(), snapshot.qsa_layout.end(),
                               [](const auto& item) { return item.first == kLayers; });
    auto trunk_layout = snapshot.qsa_layout;
    std::erase_if(trunk_layout, [](const auto& item) { return item.first == kLayers; });
    if (trunk_layout != Layout(state, false) || (snapshot.resume.mpos > 0 && !mtp))
      throw std::runtime_error("snapshot cache layout or MTP presence changed");
    if (mtp) {
      auto expected = Layout(state, false);
      expected.emplace_back(kLayers, state.compact_state ? kQsaRawSlots : 0);
      if (expected != snapshot.qsa_layout) throw std::runtime_error("snapshot raw cache layout changed");
    }
    size_t total = 0;
    for (const auto& span : Spans(state, snapshot.hist.size(), snapshot.resume.mpos, stk, carry, mtp))
      total += span.bytes;
    if (mtp && !state.qc.contains(kLayers) && snapshot.resume.mpos > 0) total += MtpBytes(state, snapshot.resume.mpos);
    if (total != snapshot.bytes.size()) throw std::runtime_error("snapshot state layout changed");
    if (mtp && !state.qc.contains(kLayers)) e.EnsureMtpCache(state, 0);
    const auto spans = Spans(state, snapshot.hist.size(), snapshot.resume.mpos, stk, carry, mtp);
    state.Reset();
    CopySnapshotSpans(spans, snapshot.bytes, false);
    state.hist = snapshot.hist;
    state.retained = snapshot.retained;
    if (!snapshot.resume.logits.empty()) {
      const PinnedVector<float> logits(snapshot.resume.logits.begin(), snapshot.resume.logits.end());
      HIP_OK(hipMemcpy(e.logits_, logits.data(), sizeof(float) * kVocab, hipMemcpyHostToDevice));
    }
    // RunGen rebuilds the request's proposal head before restore. Another request's
    // observed suffix must not change proposal probabilities or mapping ownership.
    return snapshot.resume;
  }
  // Restore's MTP step for a slot write with no snapshot (SlotCheckpointIO::CopyTo): Engine::EnsureMtpCache is
  // private, and this struct is its friend.
  static void EnsureMtpCache(Engine& e, SequenceState& state) {
    if (!state.qc.contains(kLayers)) e.EnsureMtpCache(state, 0);
  }
};
