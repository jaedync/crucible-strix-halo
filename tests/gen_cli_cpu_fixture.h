// Shared CPU engine for the generation-driver tests (gen_cli_cpu_test, gen_jobs_cpu_test).
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cassert>
#include "model/flashnext/geometry.h"
#include "runtime/bounds.h"
#include "runtime/retained_prefix.h"
#include "runtime/draft_policy.h"
#include "runtime/prefill_rows.h"
#include "cli/gen_job_state.h"
#include <functional>
using namespace crucible;
using hipEvent_t = int;
constexpr int hipMemcpyDeviceToDevice = 0;
#define HIP_OK(value) do { (void)(value); } while (0)
struct DeviceSpan { const void* pointer; size_t bytes; };
int hipMemset(void* pointer, int value, size_t bytes) { std::memset(pointer, value, bytes); return 0; }
int hipEventCreate(int* value) { *value = 1; return 0; }
int hipEventRecord(int, int) { return 0; }
int hipEventSynchronize(int) { return 0; }
int hipEventElapsedTime(float* value, int, int) { *value = 0; return 0; }
int hipStreamSynchronize(int) { return 0; }
int hipDeviceSynchronize() { return 0; }
int hipFree(void* pointer) { delete[] static_cast<float*>(pointer); return 0; }
int hipMemcpyAsync(void* destination, const void* source, size_t bytes, int, int) {
  if (bytes) std::memmove(destination, source, bytes);
  return 0;
}
template <class T> T* Zeros(size_t count) { return new T[count]{}; }
double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::vector<int> ParseIds(const std::string& text) { return ParseDraftSelection(text); }

struct Engine {
  int ctx, prefill_rows, mtp_calls = 0, last_prefill_rows = 0;
  // Every token the target and the draft head predict, and the rows of each Forward call (the verify rows).
  int forward_token = 0, mtp_token = 0;
  std::vector<std::vector<int>> forwards;
  std::vector<int> history;
  std::vector<float> hidden = std::vector<float>(64 * kWide);
  std::vector<float> snapshots = std::vector<float>(64 * kWide);
  RetainedPrefix retained;
  Engine(int context, int prefill) : ctx(context), prefill_rows(prefill) {}
  int PrefillRows() const { return prefill_rows; }
  float* PromptStacks() { return snapshots.data(); }
  void DraftHead(const std::vector<int>&, bool, bool, bool) {}
  void ObserveDraft(const int*, int) {}
  void Reset() { history.clear(); retained.Reset(); mtp_calls = 0; }
  // gen --jobs: host-only state. leave_dirty names a field that ResetForJob skips.
  std::string leave_dirty;
  std::function<void()> on_reset;
  bool touched = false;
  void ResetForJob() {
    if (on_reset) on_reset();
    if (leave_dirty != "history") { history.clear(); retained.Reset(); }
    if (leave_dirty != "sequence") touched = false;
  }
  GenJobState JobState(const std::vector<DeviceSpan>& scratch) const {
    GenJobState state;
    state.history = history.size();
    state.retained_clean = retained.Clean();
    state.sequence_zero = !touched;
    for (const DeviceSpan& span : scratch) {
      const auto* bytes = static_cast<const unsigned char*>(span.pointer);
      state.scratch_zero = state.scratch_zero && std::all_of(bytes, bytes + span.bytes, [](unsigned char b) { return !b; });
    }
    return state;
  }
  void ResetMtp() {}
  int Context() const { return ctx; }
  size_t Length() const { return history.size(); }
  float* H() { return hidden.data(); }
  void Truncate(size_t count) { assert(retained.Truncate(count).empty()); history.resize(count); }
  std::vector<int> Forward(const int* tokens, int rows, int replay, int commit, const std::string& = "", int = -1) {
    assert(ContextRowsFit(int(history.size()), rows, ctx, kMaxRows));
    assert(retained.Forward(rows, replay, commit).empty());
    touched = true;
    forwards.emplace_back(tokens, tokens + rows);
    history.insert(history.end(), tokens, tokens + rows);
    return std::vector<int>(rows, forward_token);
  }
  std::vector<int> Prefill(const int* tokens, int rows, const std::string& = "", bool = true, const void* = nullptr,
                           int = 0) {
    assert(ContextRowsFit(int(history.size()), rows, ctx, kMaxPrefillRows));
    assert(retained.Forward(rows, 0, rows).empty());
    touched = true;
    history.insert(history.end(), tokens, tokens + rows);
    last_prefill_rows = rows;
    return {forward_token};
  }
  // The real engine keeps bulk stacks in its padded buffer; this fake keeps them in hidden.
  void CopyPrefillStacks(float* dst, int first, int rows) {
    assert(first >= 0 && rows >= 1 && first + rows <= last_prefill_rows && first + rows <= 64);
    std::memmove(dst, hidden.data() + size_t(first) * kWide, size_t(rows) * kWide * sizeof(float));
  }
  std::vector<int> Mtp(const float*, const int*, int rows, int pos, bool = true, bool = false) {
    assert(ContextRowsFit(pos, rows, ctx, kMaxRows));
    ++mtp_calls;
    return std::vector<int>(rows, mtp_token);
  }
  float DraftMargin() const { return 2; }
  int DraftVocabulary() const { return 101; }
  float* logits() { return nullptr; }
  void DumpState(const std::string&, int = 0, const float* = nullptr, const float* = nullptr) {}
  void ClearExpertCounts() {}
  void EnableProfile(bool = true) {}
  void Report(int) {}
  void Synchronize() {}
  int Vocabulary() const { return 101; }
  std::vector<float> CopyLogits(int rows, int = 0) const { return std::vector<float>(rows * 101); }
  std::pair<std::vector<float>, std::vector<int>> DraftLogits(int) const {
    return {std::vector<float>(101), {}};
  }
};
void CopyRows(Engine& engine, float* destination, int dst_row, int src_row, int rows) {
  if (rows) std::memmove(destination + size_t(dst_row) * kWide, engine.H() + size_t(src_row) * kWide,
                        size_t(rows) * kWide * sizeof(float));
}
void QsaWriteCapture(const std::string&, const float*, size_t) {}
