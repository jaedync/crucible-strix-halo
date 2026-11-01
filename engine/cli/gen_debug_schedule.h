// gen --debug-schedule. Diagnostic only, off unless the option is
// given; serve never reaches it. It runs the prompt in an explicit chunk schedule, so a per-layer --dump of a cold
// prefill can be compared with the schedule of a server extend (a slot resume runs an 8-row Forward head, then bulk
// chunks from the resume point). --debug-restore-at K saves a slot checkpoint at position K with the serve code
// (SlotCheckpointIO), writes other rows on the slot, restores the checkpoint and checks the state bytes.
// Greedy with spec 0 only: no MTP rows, no drafts. Included by gen_cli.h after slot_checkpoint_io.h.
#pragma once
#include <chrono>
#include <filesystem>

struct DebugChunk {
  int rows = 0;
  bool forward = false;  // the small-row Forward path (a resume head), not bulk Prefill
  bool predict = false;  // a bulk chunk that ends a request's prompt (the seed); the last chunk always predicts
};

// "8192,708": bulk chunks; "f8": a small-row Forward of 8 rows; "512!": a bulk chunk with predict.
// Throws std::invalid_argument on a bad item.
inline std::vector<DebugChunk> ParseDebugSchedule(const std::string& text) {
  std::vector<DebugChunk> chunks;
  std::stringstream items(text);
  std::string item;
  while (std::getline(items, item, ',')) {
    DebugChunk c;
    std::string digits = item;
    if (!digits.empty() && digits.front() == 'f') { c.forward = true; digits.erase(0, 1); }
    if (!digits.empty() && digits.back() == '!') { c.predict = true; digits.pop_back(); }
    if (c.forward && c.predict) throw std::invalid_argument("a Forward chunk takes no '!'");
    c.rows = GenNumber<int>(digits);
    if (c.rows < 1 || (c.forward && c.rows > kMaxRows)) throw std::invalid_argument("bad chunk rows: " + item);
    chunks.push_back(c);
  }
  if (chunks.empty()) throw std::invalid_argument("empty schedule");
  return chunks;
}

inline double DebugSeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void AppendStepLogits(const std::string& path, const float* device_row) {
  if (path.empty()) return;
  std::vector<float> host(kVocab);
  HIP_OK(hipMemcpy(host.data(), device_row, sizeof(float) * kVocab, hipMemcpyDeviceToHost));
  crucible::AppendDump(path, host.data(), host.size());
}

// The bytes that a slot checkpoint at `position` saves and restores, copied to the host.
inline std::vector<std::vector<uint8_t>> DebugSpanBytes(const SequenceState& state, size_t position) {
  HIP_OK(hipDeviceSynchronize());
  std::vector<std::vector<uint8_t>> bytes;
  for (const auto& span : SlotCheckpointIO::Spans(state, position)) {
    bytes.emplace_back(span.bytes);
    HIP_OK(hipMemcpy(bytes.back().data(), span.pointer, span.bytes, hipMemcpyDeviceToHost));
  }
  return bytes;
}

// Spans that differ, and the first one ("-1" when all are equal).
inline std::string DebugSpanDiff(const std::vector<std::vector<uint8_t>>& a, const std::vector<std::vector<uint8_t>>& b) {
  if (a.size() != b.size()) return "span count differs";
  int count = 0, first = -1;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) { ++count; if (first < 0) first = int(i); }
  return std::to_string(count) + " of " + std::to_string(a.size()) + " spans differ, first " + std::to_string(first);
}

template <class E> void DebugDumpState(E& eng, const GenOptions& o, const char* name) {
  if (o.state_dump.empty()) return;
  const std::string dir = o.state_dump + "/" + name;
  std::filesystem::create_directories(dir);
  HIP_OK(hipDeviceSynchronize());
  eng.DumpState(dir);
}

// The serve slot resume: SlotCheckpointIO::Save at the prompt end, later writes on the slot (here one Forward of
// kMaxRows other rows, as the seed's own decode steps), then SlotCheckpointIO::Restore of that End point.
template <class E> void DebugRestore(E& eng, const GenOptions& o, size_t position) {
  SequenceState& state = eng.ActiveSequence();
  DebugDumpState(eng, o, "pre");
  const auto before = DebugSpanBytes(state, position);
  SlotCheckpointIO::Buffer buffer(SlotCheckpointIO::Bytes(state));
  SlotCheckpointIO::Save(state, buffer);
  crucible::SlotCheckpoint point;
  point.hist = state.hist;
  std::vector<int> other(kMaxRows);
  for (int j = 0; j < kMaxRows; ++j) other[j] = o.ids[position - 1 - size_t(j)];
  eng.Forward(other.data(), kMaxRows, 0, kMaxRows);
  const auto clobbered = DebugSpanBytes(state, position);
  SlotCheckpointIO::Restore(state, buffer, point);
  const auto after = DebugSpanBytes(state, position);
  std::fprintf(stderr, "debug restore at %zu: after the other rows %s; after the restore %s; length %zu\n", position,
               DebugSpanDiff(before, clobbered).c_str(), DebugSpanDiff(before, after).c_str(), eng.Length());
  DebugDumpState(eng, o, "post");
}

template <class E> GenResult RunDebugSchedule(E& eng, const GenOptions& o) {
  if (o.sampling.temperature > 0 || (o.spec > 0 && o.n > 1) || o.conversation)
    throw std::invalid_argument("--debug-schedule needs greedy spec 0 without a conversation");
  const std::vector<DebugChunk> chunks = ParseDebugSchedule(o.debug_schedule);
  size_t total = 0;
  for (const auto& c : chunks) total += size_t(c.rows);
  if (total != o.ids.size()) throw std::invalid_argument("the schedule rows must sum to the prompt length");
  const int step = std::max(eng.PrefillRows(), 1);
  GenResult res;
  eng.Reset();
  if (!o.step_logits.empty()) std::filesystem::remove(o.step_logits);
  const double t0 = DebugSeconds();
  std::vector<int> last;
  int row = 0;
  size_t i = 0;
  for (const auto& c : chunks) {
    if (o.debug_restore_at && i == o.debug_restore_at) DebugRestore(eng, o, i);
    else if (o.debug_state_at && i == o.debug_state_at) DebugDumpState(eng, o, "at");
    const bool final = i + size_t(c.rows) == o.ids.size();
    if (c.forward) {
      last = eng.Forward(o.ids.data() + i, c.rows, 0, c.rows, o.dump);
      row = c.rows - 1;
    } else {
      // As RunPrompt and batch serve: the PLE rows of the next two chunks start their reads now.
      eng.HintPle(o.ids, i + size_t(c.rows), i + size_t(c.rows) + 2 * size_t(step));
      last = eng.Prefill(o.ids.data() + i, c.rows, o.dump, final || c.predict, nullptr, c.rows);
      row = 0;
    }
    std::fprintf(stderr, "debug schedule: %s %d rows at %zu\n", c.forward ? "forward" : "prefill", c.rows, i);
    i += size_t(c.rows);
  }
  HIP_OK(hipStreamSynchronize(0));
  res.prefill_s = DebugSeconds() - t0;
  AppendStepLogits(o.step_logits, eng.logits() + size_t(row) * kVocab);
  res.gen = {last.back()};
  const double t1 = DebugSeconds();
  for (int k = 1; k < o.n; ++k) {
    res.gen.push_back(eng.Forward(&res.gen.back(), 1, 0, 1)[0]);
    AppendStepLogits(o.step_logits, eng.logits());
  }
  HIP_OK(hipStreamSynchronize(0));
  res.decode_s = DebugSeconds() - t1;
  res.forwards = long(res.gen.size()) - 1;
  return res;
}
