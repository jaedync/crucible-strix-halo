// In-slot checkpoint gate: a restore from a checkpoint at p, after the slot ran past p, must give the
// same state bytes, IDs, logits and layer dumps as a restore from a full snapshot at p. Run it once with
// CRUCIBLE_COMPACT_STATE=0 and once with 1. Usage: slot_checkpoint_gate CKPT IDS_FILE NEW_OUTPUT_DIR
#define ENGINE_LIBRARY
#include "../engine/engine.hip"
#include <filesystem>

namespace {
struct EngineStateIO {
  static SequenceState& State(Engine& e, int stream) { return *e.sequences_.at(stream); }
};
struct Observation { std::vector<int> ids; std::vector<float> logits; };
void Record(Engine& e, Observation& out, const std::vector<int>& ids, int count) {
  out.ids.insert(out.ids.end(), ids.begin(), ids.end());
  const size_t start = out.logits.size();
  out.logits.resize(start + size_t(count) * kVocab);
  HIP_OK(hipMemcpy(out.logits.data() + start, e.logits(), size_t(count) * kVocab * sizeof(float), hipMemcpyDeviceToHost));
}
// The batch schedule after a tail-like restore: one small-row forward with the pending rows, bulk chunks, then decode.
Observation Continue(Engine& e, const std::vector<int>& suffix, int replay, const std::string& dump) {
  Observation out;
  const int head = std::min<int>(kMaxRows, int(suffix.size()));
  Record(e, out, e.Forward(suffix.data(), head, replay, head, dump + "/head"), head);
  for (size_t i = size_t(head); i < suffix.size();) {
    const int rows = int(std::min<size_t>(64, suffix.size() - i));
    Record(e, out, e.Prefill(suffix.data() + i, rows, dump + "/bulk"), 1);
    i += size_t(rows);
  }
  int token = out.ids.back();
  for (int step = 0; step < 8; ++step) {
    const auto next = e.Forward(&token, 1, 0, 1, dump + "/decode");
    Record(e, out, next, 1);
    token = next.back();
  }
  return out;
}
bool DumpsEqual(const std::string& a, const std::string& b) {
  for (const char* part : {"/head", "/bulk", "/decode"})
    for (int l = 0; l <= kLayers; ++l) {
      char name[32]; std::snprintf(name, sizeof(name), "/h_%02d.f32", l);
      std::ifstream fa(a + part + name, std::ios::binary), fb(b + part + name, std::ios::binary);
      const std::string av(std::istreambuf_iterator<char>(fa), {}), bv(std::istreambuf_iterator<char>(fb), {});
      if (av.empty() || av != bv) return false;
    }
  return true;
}
// Reach a position: bulk chunks of 64, then decode-path rows; `pending` rows stay uncommitted (a verify window).
int Reach(Engine& e, const std::vector<int>& ids, int bulk, int forward, int pending) {
  for (int i = 0; i < bulk; i += 64) e.Prefill(ids.data() + i, 64, "");
  if (forward) e.Forward(ids.data() + bulk, forward, 0, forward);
  if (pending) e.Forward(ids.data() + bulk + forward, pending + 1, 0, 1);
  return pending;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4 || std::filesystem::exists(argv[3])) return 2;
  setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0);
  std::ifstream file(argv[2]); std::stringstream text; text << file.rdbuf();
  const auto source = ParseIds(text.str());
  if (source.size() < 24) return 2;
  std::vector<int> ids(512);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = source[i % source.size()];
  Engine e(argv[1], 512, 64);
  e.ConfigureStreams(1);
  e.SelectStream(0);
  SequenceState& state = EngineStateIO::State(e, 0);
  float* stk = Zeros<float>((kMaxRows + 1) * kWide);
  float* carry = Zeros<float>(kWide);
  // {bulk rows, decode-path rows, pending rows}: grid points inside and at the end of a pool block, an open pool
  // group, an open group in a new block, a verify window.
  const int cases[][3] = {{64, 0, 0}, {128, 0, 0}, {64, 6, 0}, {128, 3, 0}, {64, 5, 2}};
  int failures = 0;
  for (int c = 0; c < 5; ++c) {
    e.Reset();
    const int replay = Reach(e, ids, cases[c][0], cases[c][1], cases[c][2]);
    const size_t p = e.Length();
    ConversationResume resume;
    resume.length = p; resume.replay = replay;
    const auto full = ConversationStateIO::Save(e, state, resume, stk, carry);
    SlotCheckpointIO::Buffer buffer(SlotCheckpointIO::Bytes(state));
    SlotCheckpointIO::Save(state, buffer);
    const crucible::SlotCheckpoint point{state.hist, {}, replay, false, crucible::CheckpointKind::End, 1};
    // The slot runs on past p: new rows, pool blocks, rings and GDN state.
    std::vector<int> other(ids.rbegin(), ids.rbegin() + 69);
    if (replay) e.Forward(other.data(), replay + 1, replay, replay + 1);
    e.Prefill(other.data(), 64, "");
    e.Forward(other.data() + 64, 5, 0, 5);
    const std::vector<int> suffix(ids.begin() + std::ptrdiff_t(p), ids.begin() + std::ptrdiff_t(p) + 77);
    const std::string root = std::string(argv[3]) + "/" + std::to_string(c);
    for (const char* side : {"/a/head", "/a/bulk", "/a/decode", "/b/head", "/b/bulk", "/b/decode"})
      std::filesystem::create_directories(root + side);
    SlotCheckpointIO::Restore(state, buffer, point);
    const auto from_point = ConversationStateIO::Save(e, state, resume, stk, carry);
    const auto a = Continue(e, suffix, replay, root + "/a");
    e.Reset();
    ConversationStateIO::Restore(e, state, *full, stk, carry);
    const auto from_full = ConversationStateIO::Save(e, state, resume, stk, carry);
    const auto b = Continue(e, suffix, replay, root + "/b");
    const bool bytes = from_point->bytes == full->bytes && from_full->bytes == full->bytes &&
                       from_point->hist == full->hist;
    const bool outputs = a.ids == b.ids && a.logits.size() == b.logits.size() &&
                         !std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float));
    const bool dumps = DumpsEqual(root + "/a", root + "/b");
    std::printf("%s case=%d p=%zu replay=%d compact=%d: state bytes %s, %zu IDs and %zu logits %s, dumps %s\n",
                bytes && outputs && dumps ? "PASS" : "FAIL", c, p, replay, int(state.compact_state),
                bytes ? "equal" : "DIFFER", a.ids.size(), a.logits.size(), outputs ? "equal" : "DIFFER",
                dumps ? "equal" : "DIFFER");
    failures += !(bytes && outputs && dumps);
  }
  for (void* p : {stk, carry}) HIP_OK(hipFree(p));
  return failures ? 1 : 0;
}
