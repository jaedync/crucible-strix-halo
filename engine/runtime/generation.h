// Resident requests accept a draft only when the serial target verifies it.
#pragma once
#include "bounds.h"
#include "draft_stop.h"
#include "../model/flashnext/geometry.h"
#include "prompt.h"
#include <algorithm>
#include <chrono>
#include <span>
#include <string>
#include <stdexcept>
#include <vector>

namespace crucible {
struct GenerationResult {
  std::vector<int> ids;
  double prefill_seconds = 0, decode_seconds = 0;
  long forwards = 0, accepted = 0, attempted = 0;
  bool stopped = false;
};
inline double GenerationNow() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// copy(dst, src, float_count) preserves stream order. emit() sees verified tokens only.
template <class Engine, class Copy, class Emit>
GenerationResult GenerateExact(Engine& eng, std::span<const int> ids, int n, int chunk, int spec,
                               std::span<const int> stops, const std::string& dump,
                               float* stk, float* carry, Copy copy, Emit emit, bool profile = false) {
  if (!ValidTokenIds(ids, kVocab) || !ValidTokenIds(stops, kVocab) ||
      !RequestFitsContext(ids.size(), n, eng.Context()) || chunk < 1 || chunk > kMaxRows ||
      spec < 0 || spec >= kMaxRows)
    throw std::runtime_error("invalid generation request; check prompt IDs, output length, chunk, spec, and context");
  eng.Reset();
  GenerationResult result;
  int mpos = 0;
  auto rows = [&](float* dst, int dst_row, int r0, int count) {
    copy(dst + (size_t)dst_row * kWide, eng.H() + (size_t)r0 * kWide, (size_t)count * kWide);
  };
  auto push = [&](int id) {
    if ((int)result.ids.size() >= n || result.stopped) return;
    result.ids.push_back(id);
    result.stopped = std::find(stops.begin(), stops.end(), id) != stops.end();
    emit(id);
  };
  double t0 = GenerationNow();
  const std::vector<int> last = RunPrompt(eng, ids, chunk, spec > 0, dump, stk, carry, mpos, copy,
      [&](float* st, const int* toks, int count, int pos) { eng.Mtp(st, toks, count, pos); }, [](int) {});
  result.prefill_seconds = GenerationNow() - t0;
  if (profile) eng.EnableProfile();
  t0 = GenerationNow();
  push(last.back());
  if (spec == 0) {
    while ((int)result.ids.size() < n && !result.stopped) {
      push(eng.Forward(&result.ids.back(), 1, 0, 1, dump)[0]);
      ++result.forwards;
    }
  } else {
    copy(stk, carry, kWide);
    std::vector<int> ctoks{result.ids.back()};
    int replay = 0;
    while ((int)result.ids.size() < n && !result.stopped) {
      const int rc = (int)ctoks.size();
      const size_t base = eng.Length();
      const int depth = BoundedDraftDepth(spec, eng.Context(), (int)base, mpos, rc);
      if (depth == 0) {
        push(eng.Forward(&result.ids.back(), 1, replay, 1)[0]);
        replay = 0;
        ++result.forwards;
        continue;
      }
      // No draft past the token limit or after a stop ID can commit a token (draft_stop.h).
      const int room = DraftRoom(size_t(n) - result.ids.size());
      std::vector<int> drafts{eng.Mtp(stk, ctoks.data(), rc, mpos)[rc - 1]};
      mpos += rc;
      for (int k = 1; k < depth && ChainMayGrow(drafts, stops, room); ++k) {
        rows(stk, kMaxRows, k == 1 ? rc - 1 : 0, 1);
        drafts.push_back(eng.Mtp(stk + (size_t)kMaxRows * kWide, &drafts.back(), 1, mpos + k - 1)[0]);
      }
      TrimDrafts(drafts, stops, room);
      const int proposed = (int)drafts.size();
      std::vector<int> input{result.ids.back()};
      input.insert(input.end(), drafts.begin(), drafts.end());
      const auto ys = eng.Forward(input.data(), proposed + 1, replay, 1);
      int accepted = 0;
      while (accepted < proposed && drafts[accepted] == ys[accepted]) ++accepted;
      eng.Truncate(base + accepted + 1);
      rows(stk, 0, 0, accepted + 1);
      ctoks.assign(ys.begin(), ys.begin() + accepted + 1);
      for (int i = 0; i <= accepted; ++i) push(ys[i]);
      replay = accepted;
      result.accepted += accepted;
      result.attempted += proposed;
      ++result.forwards;
    }
  }
  result.decode_seconds = GenerationNow() - t0;
  return result;
}
}  // namespace crucible
