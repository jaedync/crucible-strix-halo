// This probe uses the production prompt code with one explicit row-schedule substitution.
// Two modes:
//   split2048 CKPT OUT IDS CUT IDS CUT IDS CUT   the 2048 cut qualification
//   split2048 pre CKPT OUT IDS LENGTH...         re-send fix B: each LENGTH-token prefix of IDS
//     runs the batch-serve schedule with fix B off ("whole") and on ("split", "restored": a save at the Pre point,
//     a clear, a restore, the rest). The MTP is off, so the state bytes are target state only.
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include "split2048_prompt.h"
#include "split2048_schedule.h"
#include <filesystem>

namespace {
void Clear(Engine& e) {
  e.Reset();
  for (const auto& span : e.ActiveSequence().Spans())
    HIP_OK(hipMemset(const_cast<void*>(span.pointer), 0, span.bytes));
}
void WriteBytes(const std::filesystem::path& path, const void* data, size_t bytes) {
  std::ofstream output(path, std::ios::binary);
  if (!output || !output.write(static_cast<const char*>(data), bytes))
    throw std::runtime_error("cannot write probe capture");
}
std::vector<int> ReadIds(const char* path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read probe IDs");
  std::stringstream text; text << input.rdbuf();
  return ParseIds(text.str());
}
std::vector<int> Schedule(int length, int cut) {
  std::vector<int> result;
  bool found = false;
  for (int at = 0; at < length; at += 4096) {
    const int end = std::min(at + 4096, length);
    if (at < cut && cut < end) {
      if ((cut - at) % kPrefillCoreRows || std::min(cut - at, end - cut) <= kMaxRows)
        throw std::runtime_error("invalid core cut");
      found = true;
      result.push_back(cut - at); result.push_back(end - cut);
    } else result.push_back(end - at);
  }
  if (cut && !found) throw std::runtime_error("cut is not an interior core boundary");
  return result;
}
// The batch-serve schedules of one cold prompt with fix B off and on (bench/split2048_schedule.h).
struct PreCase { size_t pre = 0; std::vector<int> whole, split; };
PreCase PreSchedules(size_t length, int step, int launch) {
  PreCase c;
  c.pre = split2048::PrePoint(length, step, kPrefillCoreRows, crucible::kPreEndMinTokens, kMaxRows);
  if (!c.pre) throw std::runtime_error("the prompt has no Pre point");
  c.whole = split2048::ServeSchedule(length, step, launch, 0);
  c.split = split2048::ServeSchedule(length, step, launch, c.pre);
  if (!split2048::CutOnlyAt(c.whole, c.split, c.pre, step)) throw std::runtime_error("the cut is not fix B's cut");
  return c;
}
// `schedule` empty: RunPrompt's own grid schedule. Else RunSplitPrompt with these rows; `cut` > 0 is a chunk end.
void Capture(Engine& e, const std::vector<int>& ids, const std::vector<int>& schedule, int cut,
             const std::filesystem::path& root, float* stk, float* carry, bool restore_cut = false,
             bool mtp_on = true) {
  std::filesystem::create_directory(root);
  Clear(e);
  HIP_OK(hipMemset(stk, 0, (kMaxRows + 1) * kWide * sizeof(float)));
  HIP_OK(hipMemset(carry, 0, kWide * sizeof(float)));
  ConversationHooks hooks;
  hooks.admit_prompt = [&](size_t n) { return n == ids.size() || (restore_cut && n == size_t(cut)); };
  hooks.save_generated_tail = false;
  hooks.save = [&](const ConversationResume& resume, float* s, float* c) {
    const auto snapshot = ConversationStateIO::Save(e, e.ActiveSequence(), resume, s, c);
    if (restore_cut && resume.length == size_t(cut)) {
      Clear(e);
      HIP_OK(hipMemset(s, 0, (kMaxRows + 1) * kWide * sizeof(float)));
      HIP_OK(hipMemset(c, 0, kWide * sizeof(float)));
      ConversationStateIO::Restore(e, e.ActiveSequence(), *snapshot, s, c);
      std::ofstream marker(root / "restore.json");
      if (!marker) throw std::runtime_error("cannot write restore marker");
      marker << "{\"length\":" << resume.length << ",\"bytes\":" << snapshot->bytes.size() << "}\n";
      return;
    }
    const auto state_bytes = snapshot->bytes.Copy();
    WriteBytes(root / "state.bin", state_bytes.data(), state_bytes.size());
    WriteBytes(root / "logits.f32", resume.logits.data(), resume.logits.size() * sizeof(float));
    std::ofstream meta(root / "state.json");
    meta << "{\"length\":" << resume.length << ",\"mpos\":" << resume.mpos
         << ",\"have_carry\":" << (resume.have_carry ? "true" : "false")
         << ",\"next_id\":" << resume.next_id << ",\"bytes\":" << snapshot->bytes.size() << "}\n";
  };
  const auto copy = [](float* dst, const float* src, size_t count) {
    HIP_OK(hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToDevice, 0));
  };
  const auto mtp = [&](float* stacks, const int* tokens, int rows, int pos) { e.Mtp(stacks, tokens, rows, pos, false); };
  int mpos = 0;
  const double start = Now();
  const auto last = !schedule.empty() ? RunSplitPrompt(e, std::span<const int>(ids), 8, mtp_on, "", stk, carry,
      mpos, copy, mtp, [](int) {}, 256, {}, &hooks, schedule)
      : RunPrompt(e, std::span<const int>(ids), 8, mtp_on, "", stk, carry, mpos,
      copy, mtp, [](int) {}, 256, {}, &hooks);
  HIP_OK(hipDeviceSynchronize());
  const double seconds = Now() - start;
  std::vector<int> generated{last.back()};
  for (int i = 1; i < 8; ++i) generated.push_back(e.Forward(&generated.back(), 1, 0, 1).at(0));
  WriteBytes(root / "ids.i32", generated.data(), generated.size() * sizeof(int));
  std::printf("RESULT %s rows=%zu cut=%d seconds=%.6f mpos=%d\n", root.c_str(), ids.size(), cut, seconds, mpos);
  std::fflush(stdout);
}
void WriteSchedule(const std::filesystem::path& path, size_t length, const PreCase& c) {
  std::ofstream out(path);
  const auto list = [&](const std::vector<int>& rows) {
    for (size_t i = 0; i < rows.size(); ++i) out << (i ? "," : "") << rows[i];
  };
  out << "{\"length\":" << length << ",\"pre\":" << c.pre << ",\"whole\":[";
  list(c.whole);
  out << "],\"split\":[";
  list(c.split);
  out << "]}\n";
  if (!out) throw std::runtime_error("cannot write the schedule");
}
int PreMain(int argc, char** argv) {
  if (argc < 6) return 2;
  const auto source = ReadIds(argv[4]);
  std::vector<size_t> lengths;
  for (int i = 5; i < argc; ++i) lengths.push_back(std::stoul(argv[i]));
  const size_t longest = *std::max_element(lengths.begin(), lengths.end());
  if (source.size() < longest || !ValidTokenIds(source, kVocab)) return 2;
  const std::filesystem::path root(argv[3]);
  if (!std::filesystem::create_directory(root)) return 2;
  // The context holds the longest prompt and the 8 continuation IDs.
  Engine e(argv[2], int((longest + 8 + 4095) / 4096 * 4096), 4096);
  const int step = e.PrefillRows(), launch = EngineLaunchRows(e);
  std::printf("PRE step=%d launch=%d cases=%zu\n", step, launch, lengths.size());
  float* stk = Zeros<float>((kMaxRows + 1) * kWide);
  float* carry = Zeros<float>(kWide);
  for (size_t k = 0; k < lengths.size(); ++k) {
    const std::vector<int> ids(source.begin(), source.begin() + std::ptrdiff_t(lengths[k]));
    const PreCase c = PreSchedules(ids.size(), step, launch);
    const auto case_root = root / ("case" + std::to_string(k));
    std::filesystem::create_directory(case_root);
    WriteSchedule(case_root / "schedule.json", ids.size(), c);
    Capture(e, ids, c.whole, 0, case_root / "whole", stk, carry, false, false);
    Capture(e, ids, c.split, int(c.pre), case_root / "split", stk, carry, false, false);
    Capture(e, ids, c.split, int(c.pre), case_root / "restored", stk, carry, true, false);
  }
  HIP_OK(hipFree(stk)); HIP_OK(hipFree(carry));
  return 0;
}
}

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "pre") return PreMain(argc, argv);
    if (argc != 9) return 2;
    const std::filesystem::path root(argv[2]);
    if (!std::filesystem::create_directory(root)) return 2;
    Engine e(argv[1], 24576, 4096);
    e.DraftHead({}, true);
    float* stk = Zeros<float>((kMaxRows + 1) * kWide);
    float* carry = Zeros<float>(kWide);
    for (int i = 3; i < 9; i += 2) {
      const auto ids = ReadIds(argv[i]);
      const int cut = std::stoi(argv[i + 1]);
      if (ids.empty() || ids.size() > 20480 || !ValidTokenIds(ids, kVocab)) return 2;
      const auto case_root = root / ("case" + std::to_string((i - 3) / 2));
      std::filesystem::create_directory(case_root);
      Capture(e, ids, {}, 0, case_root / "whole", stk, carry);
      Capture(e, ids, Schedule(int(ids.size()), cut), cut, case_root / "split", stk, carry);
      Capture(e, ids, Schedule(int(ids.size()), cut), cut, case_root / "restored", stk, carry, true);
    }
    HIP_OK(hipFree(stk)); HIP_OK(hipFree(carry));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "split2048 probe: %s\n", error.what()); return 1;
  }
}
