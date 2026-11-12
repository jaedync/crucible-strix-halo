// Only mutable sequence state belongs here. Weights and projection scratch stay shared.
#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include "../model/flashnext/model.h"
#include "../model/flashnext/multimodal.h"
#include "../exl3/plans.h"
#include "retained_prefix.h"
#include "spec_span.h"
#include "../model/flashnext/fixed_state.h"
#include "rope_mode.h"

namespace crucible {
struct SequenceState {
  const int capacity;
  // Positions of the MTP cache (qc[kLayers]). A batch slot may hold fewer than `capacity` (MtpSlotRows).
  const int mtp_capacity;
  const bool compact_state;
  std::vector<int> hist;
  RetainedPrefix retained;
  std::map<int, QsaCache> qc;
  std::map<int, float*> gstate, ghist, gpend;
  // a bulk prefill or a restore may have left a GDN state in the lazy form (S and a per-head scale, see
  // engine/layers/kernels/prefill_gdn.h). The first small-row forward makes it S_true (Engine::MaterializeGdn).
  bool gdn_lazy = false;
  float* ple_ring = nullptr;
  // Image request context (engine/model/flashnext/multimodal.h). Reset() keeps it: the request owner binds it before
  // the prompt and clears it when the request ends, so text requests always see a null table.
  struct ImageSpan { int offset, count; const uint16_t* embeds; };
  int* mrope = nullptr;  // device [capacity][4], allocated by the first image request
  bool mrope_on = false;
  std::vector<ImageSpan> image_spans;
  // The images whose embeddings this request needs, in prompt order (image_spans[i] belongs to vision_images[i]).
  // An image's embeddings exist from its first encoder slice until the prompt cursor passes its end, so a batch
  // slot holds only the images that overlap one prompt chunk (train13-audit L1).
  struct BoundImage { ImageInput image; uint16_t* embeds = nullptr; };
  std::vector<BoundImage> vision_images;
  size_t vision_next = 0;     // the first image whose encoder pass is not complete
  int vision_slice = 0;       // the next slice of vision_images[vision_next]
  float* vision_x = nullptr;  // the encoder residual stream of this request, while an image is in progress
  std::vector<uint8_t> vision_rgb;  // verified host pixels of vision_images[vision_next], read before its first slice
  const int* Mrope() const { return mrope_on ? mrope : nullptr; }
  // The RoPE mode (rope_mode.h): the process start mode at construction; only
  // Engine::SetStreamRope changes it, and only while hist is empty. Reset() keeps it.
  RopeMode rope_mode = RopeMode::Plain;

  // One layer's pending GDN rows: [48 heads][pend rows][kPend]; 16 pend rows with a verify span (spec_span.h).
  static size_t PendBytes() { return sizeof(float) * 48 * SpecPendRows(SpecSpanRows()) * kPend; }
  static size_t FixedBytes() {
    return 36ull * (kGdnStateBytes + kGdnHistBytes + PendBytes()) + kPleRingBytes;
  }
  // The fixed-size buffers (fixed_state.h): every copy of a sequence's state visits this one list.
  std::vector<DeviceSpan> FixedSpans() const {
    std::vector<DeviceSpan> spans;
    VisitFixedState(*this, PendBytes(), [&](const void* pointer, size_t bytes) { spans.push_back({pointer, bytes}); });
    return spans;
  }
  static size_t Bytes(int ctx, bool compact = false, bool mtp = false, int mtp_rows = 0) {
    return FixedBytes() + QsaSequenceCacheBytes(ctx, compact, mtp, MtpCapacity(ctx, mtp_rows));
  }
  static size_t ServeBytes(int ctx, bool compact, int mtp_rows = 0) {
    return FixedBytes() + ServeQsaCacheBytes(ctx, compact, MtpCapacity(ctx, mtp_rows));
  }
  // mtp_rows 0: the MTP cache holds the whole context.
  static int MtpCapacity(int ctx, int mtp_rows) { return mtp_rows > 0 ? std::min(ctx, mtp_rows) : ctx; }

  std::vector<DeviceSpan> Spans() const {
    const int ctx = capacity;
    std::vector<DeviceSpan> spans;
    for (const auto& [l, c] : qc) {
      const int rows = l == kLayers ? mtp_capacity : ctx;
      spans.push_back({c.k, (size_t)rows * 512 * sizeof(__half)});
      spans.push_back({c.v, (size_t)rows * 512 * sizeof(__half)});
      spans.push_back({c.ik_raw, size_t(c.raw_slots ? c.raw_slots : rows) * 128 * sizeof(float)});
      spans.push_back({c.ik_pool, QsaPoolFloats(rows) * sizeof(float)});
    }
    for (const auto& span : FixedSpans()) spans.push_back(span);
    return spans;
  }

  SequenceState(const Model& model, int ctx, bool compact = false, int mtp_rows = 0)
      : capacity(ctx), mtp_capacity(MtpCapacity(ctx, mtp_rows)), compact_state(compact) {
    for (int l = 0; l <= kLayers; ++l) {
      if (l == kLayers && compact) continue;
      if (l == kLayers || model.L[l].attn) {
        const int rows = l == kLayers ? mtp_capacity : ctx;
        QsaCache& c = qc[l];
        c.k = Zeros<__half>((size_t)rows * 512);
        c.v = Zeros<__half>((size_t)rows * 512);
        c.raw_slots = compact ? kQsaRawSlots : 0;
        c.ik_raw = Zeros<float>(size_t(c.raw_slots ? c.raw_slots : rows) * 128);
        c.ik_pool = Zeros<float>(QsaPoolFloats(rows));
      } else {
        gstate[l] = Zeros<float>(kGdnStateBytes / sizeof(float));
        ghist[l] = Zeros<float>(kGdnHistBytes / sizeof(float));
        gpend[l] = Zeros<float>(PendBytes() / sizeof(float));
      }
    }
    ple_ring = Zeros<float>(kPleRingBytes / sizeof(float));
#if CRUCIBLE_DEBUG_NONFINITE & 16
    // a fresh sequence has no KV row, so every row is above its length. Reads above the length
    // must not reach an output. Not for gen --jobs: ResetForJob zeroes the KV again, so only the first job checks it.
    const auto spans = Spans();  // four KV spans per QSA cache first, then the fixed state
    for (size_t i = 0; i < 4 * qc.size(); ++i) HIP_OK(hipMemset(const_cast<void*>(spans[i].pointer), 0xFF, spans[i].bytes));
#endif
    // The admission formula and the copy list must name the same buffers.
    if (!FixedStateMatches(*this, PendBytes(), FixedBytes()))
      throw std::logic_error("sequence state buffers do not match SequenceState::FixedBytes");
  }
  void AllocateMtpCache() {
    if (qc.contains(kLayers)) return;
    const auto bytes = QsaCacheBytes(mtp_capacity, compact_state);
    CacheBufferTransaction pending(bytes, [](size_t count) -> void* {
      void* pointer = nullptr;
      const hipError_t error = hipMalloc(&pointer, count);
      if (error != hipSuccess) throw std::runtime_error(std::string("MTP cache allocation failed: ") + hipGetErrorString(error));
      return pointer;
    }, [](void* pointer) { HIP_OK(hipFree(pointer)); });
    const auto& pointers = pending.Pointers();
    for (size_t i = 0; i < pointers.size(); ++i) {
#if CRUCIBLE_DEBUG_NONFINITE & 16
      const hipError_t error = hipMemset(pointers[i], 0xFF, bytes[i]);  // an earlier change, KV tail (see the constructor)
#else
      const hipError_t error = hipMemset(pointers[i], 0, bytes[i]);
#endif
      if (error != hipSuccess) throw std::runtime_error(std::string("MTP cache initialization failed: ") + hipGetErrorString(error));
    }
    qc.emplace(kLayers, QsaCache{static_cast<__half*>(pointers[0]), static_cast<__half*>(pointers[1]),
                               static_cast<float*>(pointers[2]), static_cast<float*>(pointers[3]),
                               compact_state ? kQsaRawSlots : 0});
    pending.Commit();
  }
  SequenceState(const SequenceState&) = delete;
  SequenceState& operator=(const SequenceState&) = delete;

  ~SequenceState() {
    for (const auto& [l, c] : qc)
      for (void* p : {(void*)c.k, (void*)c.v, (void*)c.ik_raw, (void*)c.ik_pool}) HIP_OK(hipFree(p));
    for (const auto* group : {&gstate, &ghist, &gpend})
      for (const auto& [l, p] : *group) HIP_OK(hipFree(p));
    HIP_OK(hipFree(ple_ring));
    if (mrope) HIP_OK(hipFree(mrope));
    for (const auto& image : vision_images) if (image.embeds) HIP_OK(hipFree(image.embeds));
    if (vision_x) HIP_OK(hipFree(vision_x));
  }

  void Reset() {
    for (const auto& [l, s] : gstate) HIP_OK(hipMemset(s, 0, kGdnStateBytes));  // with the scale (0 means 1)
    gdn_lazy = false;
    for (const auto& [l, h] : ghist) HIP_OK(hipMemset(h, 0, sizeof(float) * 16 * kGdnQkvWidth));
    HIP_OK(hipMemset(ple_ring, 0, sizeof(float) * 16 * kWide));
    hist.clear();
    retained.Reset();
  }
};
}  // namespace crucible
