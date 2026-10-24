// Included after Engine, as conversation_snapshot.h. Device copies for in-slot checkpoints (slot_checkpoints.h).
//
// After position p a stream changes only its fixed-size state and, per QSA layer, the compact raw ring and the pool
// block that holds p (rows up to p - 1 can share a four-row pool group with later rows). Every other row below p is
// written once. So these spans at p, with the rows that the slot keeps, are the bytes that a full snapshot at p
// restores: ConversationStateIO::Spans writes the same raw ring and the same pool floats up to the block of p.
#pragma once
#include "conversation_snapshot.h"
#include "slot_checkpoints.h"

struct SlotCheckpointIO {
  static constexpr size_t kPoolBlockFloats = 4096;  // QsaPoolFloats: 4096 floats per 128 positions
  // The trunk's QSA layers only: a tail and a checkpoint hold target state; a speculative request warms its MTP rows
  // again after either restore.
  static std::vector<DeviceSpan> Spans(const SequenceState& state, size_t position) {
    std::vector<DeviceSpan> spans = state.FixedSpans();
    for (const auto& [layer, cache] : state.qc) {
      if (layer == kLayers) continue;
      if (cache.raw_slots) spans.push_back({cache.ik_raw, size_t(cache.raw_slots) * kQsaRawDim * sizeof(float)});
      if (position % 128)
        spans.push_back({cache.ik_pool + position / 128 * kPoolBlockFloats, kPoolBlockFloats * sizeof(float)});
    }
    return spans;
  }
  // The largest checkpoint of a stream (a position inside a pool block).
  static size_t Bytes(const SequenceState& state) {
    size_t bytes = 0;
    for (const auto& span : Spans(state, 1)) bytes += span.bytes;
    return bytes;
  }
  // One device buffer per checkpoint. Copies run on the null stream, in order with the forwards around them.
  struct Buffer {
    uint8_t* device = nullptr;
    size_t bytes = 0, position = 0;  // the spans' layout depends on the position
    size_t tail = 0;                 // bytes after the spans that Save never writes (fix A: the MTP carry)
    crucible::RetainedPrefix retained;
    explicit Buffer(size_t size, size_t tail_bytes = 0) : bytes(size), tail(tail_bytes) {
      if (tail > size) throw std::invalid_argument("slot checkpoint tail is larger than its buffer");
      HIP_OK(hipMalloc(&device, size));
#if CRUCIBLE_DEBUG_NONFINITE & 2
      HIP_OK(hipMemset(device, 0xFF, size));  // a restore must read only bytes that Save wrote
#endif
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (device) HIP_OK(hipFree(device)); }
  };
  static void Save(const SequenceState& state, Buffer& buffer) {
    size_t offset = 0;
    for (const auto& span : Spans(state, state.hist.size())) {
      if (offset + span.bytes > buffer.bytes - buffer.tail) throw std::logic_error("slot checkpoint buffer is too small");
      HIP_OK(hipMemcpyAsync(buffer.device + offset, span.pointer, span.bytes, hipMemcpyDeviceToDevice, 0));
      offset += span.bytes;
    }
    buffer.retained = state.retained;
    buffer.position = state.hist.size();
  }
  // The stream must still hold every row below the checkpoint: the session drops a slot's checkpoints when a reset
  // or a host restore writes the slot (SlotCheckpoints::Keep).
  static void Restore(SequenceState& state, const Buffer& buffer, const crucible::SlotCheckpoint& point) {
    const size_t length = point.hist.size();
    if (buffer.position != length || state.hist.size() < length ||
        !std::equal(point.hist.begin(), point.hist.end(), state.hist.begin()))
      throw std::logic_error("slot checkpoint rows are no longer on the slot");
    size_t offset = 0;
    for (const auto& span : Spans(state, length)) {
      HIP_OK(hipMemcpyAsync(const_cast<void*>(span.pointer), buffer.device + offset, span.bytes,
                            hipMemcpyDeviceToDevice, 0));
      offset += span.bytes;
    }
    state.hist = point.hist;
    state.retained = buffer.retained;
  }
  // the host snapshot of a Frontier point while the stream holds later rows.
  // The buffer holds Spans(state, length) as Save copied them; the rows below the point are the stream's own, written
  // once. So these are the bytes that ConversationStateIO::Save wrote at the point with target state (`r`: mpos 0).
  // `E`: the engine (a template here: this runtime header names no serve-level type).
  template <class E>
  static std::shared_ptr<const ConversationSnapshot> SaveFrom(E& e, const SequenceState& state,
      const Buffer& buffer, const crucible::SlotCheckpoint& point, const ConversationResume& r, float* stk,
      float* carry, const SaveBase& base = {}) {
    const auto sources = Sources(state, buffer, point, r);
    return ConversationStateIO::SaveAt(e, state, r, stk, carry, base, point.hist, buffer.retained, sources);
  }
  // fc-d2d: the frontier copy without a host stage. A host snapshot
  // needs cache room, and with host-only session tails in the budget there was none (refused, a full prefill). This
  // writes the device bytes that SaveFrom and then ConversationStateIO::Restore write to `target`, in the same order:
  // the source spans are SaveAt's with SaveFrom's sources (the fixed state and the compact rings from the checkpoint
  // buffer, the rows below the point from the donor), the destination spans are Restore's, and the steps are
  // Restore's (EnsureMtpCache, Reset, the span copies, hist, retained). Target state has no logits and no MTP rows.
  // The span pairs are the bytes of one snapshot, so the pair with one pointer (the shared stack pair) is a copy to
  // itself and is skipped. Every check and every host allocation comes before the first write to the target.
  template <class E>
  static void CopyTo(E& e, const SequenceState& donor, const Buffer& buffer, const crucible::SlotCheckpoint& point,
                     const ConversationResume& r, float* stk, float* carry, SequenceState& target) {
    if (&target == &donor) throw std::logic_error("slot frontier copy needs a target slot that is not the donor");
    const size_t length = point.hist.size();
    const auto sources = Sources(donor, buffer, point, r);
    if (!r.logits.empty()) throw std::logic_error("slot frontier copy needs target state with no logits");
    ConversationStateIO::Validate(e, donor, r, length, stk, carry);
    ConversationStateIO::Validate(e, target, r, length, stk, carry, false);
    auto spans = ConversationStateIO::Spans(donor, length, r.mpos, stk, carry);
    ConversationStateIO::UseSources(spans, sources);
    // Restore's layout rules, with the donor's layout as the snapshot's.
    const auto layout = ConversationStateIO::Layout(donor);
    const bool mtp = donor.qc.contains(kLayers);
    auto trunk = layout;
    std::erase_if(trunk, [](const auto& item) { return item.first == kLayers; });
    if (trunk != ConversationStateIO::Layout(target, false))
      throw std::runtime_error("snapshot cache layout or MTP presence changed");
    if (mtp) {
      auto expected = ConversationStateIO::Layout(target, false);
      expected.emplace_back(kLayers, target.compact_state ? kQsaRawSlots : 0);
      if (expected != layout) throw std::runtime_error("snapshot raw cache layout changed");
    }
    // With mpos 0 the MTP layer has no span, so these are Restore's spans also before EnsureMtpCache.
    const auto destinations = ConversationStateIO::Spans(target, length, r.mpos, stk, carry, mtp);
    bool same = destinations.size() == spans.size();
    for (size_t i = 0; same && i < spans.size(); ++i) same = destinations[i].bytes == spans[i].bytes;
    if (!same) throw std::runtime_error("snapshot state layout changed");
    std::vector<int> hist(point.hist);
    if (mtp) ConversationStateIO::EnsureMtpCache(e, target);
    target.Reset();
    for (size_t i = 0; i < spans.size(); ++i)
      if (destinations[i].pointer != spans[i].pointer)
        HIP_OK(hipMemcpyAsync(const_cast<void*>(destinations[i].pointer), spans[i].pointer, spans[i].bytes,
                              hipMemcpyDeviceToDevice, 0));
    // As Restore: the target's state is complete when the open goes on.
    HIP_OK(hipStreamSynchronize(0));
    target.hist = std::move(hist);
    target.retained = buffer.retained;
  }

 private:
  // SaveFrom's rules and sources: a grid point with target state whose rows the stream still holds, and the spans of
  // Spans(state, length) that Save copied into the buffer.
  static std::vector<ConversationStateIO::SpanSource> Sources(const SequenceState& state, const Buffer& buffer,
      const crucible::SlotCheckpoint& point, const ConversationResume& r) {
    const size_t length = point.hist.size();
    if (buffer.position != length || r.length != length || r.mpos != 0 || r.have_carry || length % 128 ||
        state.hist.size() < length || !std::equal(point.hist.begin(), point.hist.end(), state.hist.begin()))
      throw std::logic_error("slot frontier copy needs a grid point whose rows are still on the slot");
    std::vector<ConversationStateIO::SpanSource> sources;
    size_t offset = 0;
    for (const auto& span : Spans(state, length)) {
      sources.push_back({span.pointer, buffer.device + offset, span.bytes});
      offset += span.bytes;
    }
    return sources;
  }
};
