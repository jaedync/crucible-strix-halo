// Y2: BatchSession::Open sets the RoPE mode of the request's slot before its first
// forward, never resumes a state of the other mode, and counts the Way A crossings (a YaRN open that pays a full
// prefill although a plain state of the same conversation is on a slot or in the host cache).
#include "batch_mock_engine.h"
#include "../engine/runtime/rope_policy.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using crucible::RopeMode;
namespace {
struct RopeMock : MockEngine {
  std::map<int, RopeMode> rope;
  std::vector<std::pair<int, int>> sets;  // (stream, forward calls before the set)
  // As Engine::SetStreamRope: the same mode is a no-op; another mode needs an empty stream.
  void SetStreamRope(int stream, RopeMode mode) {
    if (StreamRope(stream) != mode && !histories[stream].empty())
      throw std::logic_error("a RoPE mode change needs an empty sequence");
    rope[stream] = mode;
    sets.push_back({stream, calls});
  }
  RopeMode StreamRope(int stream) const {
    const auto found = rope.find(stream);
    return found == rope.end() ? RopeMode::Plain : found->second;
  }
};

crucible::ServeRequest Yarn(crucible::ServeRequest r) {
  r.rope = RopeMode::Yarn2;
  r.plain_tenant = r.tenant;
  r.tenant = crucible::RopeTenant(r.tenant, RopeMode::Yarn2);
  return r;
}

// The open sets the mode on the slot before any forward of the request; the next open of another mode resets.
void ModeTests() {
  setenv("CRUCIBLE_ROPE_MODE", "auto:2", 1);
  RopeMock engine;
  crucible::BatchSession<RopeMock> session(engine, 1);
  const auto first = Complete(session, 1, Yarn(Request(3)));
  assert(first.size() == 5 && engine.rope.at(0) == RopeMode::Yarn2);
  assert(!engine.sets.empty() && engine.sets.back().first == 0);
  const int calls_before = engine.sets.back().second;
  (void)calls_before;
  // A plain request on the same slot: the slot is reset first, so the change is allowed.
  Complete(session, 2, Request(4));
  assert(engine.rope.at(0) == RopeMode::Plain);
  // The set runs before the request's first forward.
  RopeMock fresh;
  crucible::BatchSession<RopeMock> one(fresh, 1);
  Complete(one, 7, Yarn(Request(5)));
  assert(fresh.sets.size() == 1 && fresh.sets[0].second == 0);
  unsetenv("CRUCIBLE_ROPE_MODE");
}

// An engine without SetStreamRope serves plain requests only.
void OldEngineTests() {
  MockEngine engine;
  crucible::BatchSession<MockEngine> session(engine, 1);
  bool refused = false;
  try {
    session.Open(1, Yarn(Request(3)));
  } catch (const crucible::RecoverableOpenError& error) {
    refused = std::string(error.what()).find("YaRN") != std::string::npos;
  }
  assert(refused && session.Active() == 0);
  assert(Complete(session, 2, Request(3)).size() == 5);
}

struct Held { std::vector<int> hist; std::vector<int> stream; crucible::StateTenant tenant; };

// Session replay on one slot: a plain turn, then a turn that grows past the gate (YaRN), then a YaRN turn.
void ResumeTests() {
  setenv("CRUCIBLE_ROPE_MODE", "auto:2", 1);
  RopeMock engine;
  engine.prefill_rows = 4;
  crucible::BatchSession<RopeMock> session(engine, 1);
  std::vector<Held> host;
  std::vector<std::string> log;
  crucible::BatchCacheHooks hooks;
  hooks.restore = [&](const crucible::ServeRequest& r) -> crucible::BatchRestore {
    log.push_back("restore");
    for (const auto& t : host)
      if (t.tenant == r.tenant && t.hist.size() < r.ids.size() && std::equal(t.hist.begin(), t.hist.end(), r.ids.begin())) {
        engine.histories[engine.selected] = t.stream;
        return {t.hist.size(), false, 0, true, false, false};
      }
    return 0;
  };
  hooks.capture = [](const crucible::ServeRequest&, size_t) {};
  hooks.tail = [&](const crucible::ServeRequest& r, std::span<const int> generated, int,
                   crucible::BatchTailSource) -> std::optional<crucible::BatchResident> {
    std::vector<int> hist(r.ids);
    hist.insert(hist.end(), generated.begin(), generated.end() - 1);
    host.push_back({hist, engine.histories[engine.selected], r.tenant});
    return crucible::BatchResident{hist.size(), 0, false};
  };
  hooks.resume = [&](const crucible::ServeRequest&, crucible::BatchResident resident) -> crucible::BatchRestore {
    log.push_back("resume");
    return {resident.length, false, resident.replay, true, true, resident.guarded};
  };
  hooks.release = [](int, bool) {};
  hooks.overwrite = [](int) {};
  session.SetCacheHooks(hooks);
  const auto finish = [&](uint64_t id, const crucible::ServeRequest& r) {
    auto events = session.Open(id, r);
    for (;;) {
      for (const auto& event : events)
        if (event.done) return event;
      events = session.Step({id});
    }
  };
  auto turn1 = Request(1, 3);
  turn1.ids = {1, 2, 3, 4};
  const auto done1 = finish(1, turn1);
  assert(done1.rope == RopeMode::Plain && done1.rope_counters.yarn_opens == 0 && done1.rope_report);
  const auto tail = host.back().hist;
  // Turn 2 repeats turn 1's tail and grows past the gate: YaRN, its own tenant, no resume, one full prefill.
  auto turn2 = Request(1, 3);
  turn2.ids = tail;
  turn2.ids.insert(turn2.ids.end(), {9, 9});
  log.clear();
  const auto done2 = finish(2, Yarn(turn2));
  assert(done2.cached_tokens == 0 && std::count(log.begin(), log.end(), "resume") == 0);
  assert(done2.rope == RopeMode::Yarn2 && engine.rope.at(0) == RopeMode::Yarn2);
  assert(done2.rope_counters.yarn_opens == 1 && done2.rope_counters.crossings == 1);
  assert(done2.rope_counters.crossing_tokens == tail.size());
  // Turn 3: YaRN again, it repeats turn 2's tail, so it resumes on the slot (same mode, same tenant).
  const auto tail2 = host.back().hist;
  auto turn3 = Request(1, 3);
  turn3.ids = tail2;
  turn3.ids.push_back(7);
  log.clear();
  const auto done3 = finish(3, Yarn(turn3));
  assert(done3.slot_resume && done3.cached_tokens == tail2.size());
  assert(done3.rope_counters.yarn_opens == 2 && done3.rope_counters.crossings == 1);
  // Defense: a mode that does not match the state tenant is refused before any state changes.
  auto forged = Request(1, 3);
  forged.ids = host.back().hist;
  forged.ids.push_back(5);
  forged.tenant = crucible::RopeTenant(crucible::kLocalTenant, RopeMode::Yarn2);  // a YaRN tenant, plain mode
  bool refused = false;
  try {
    session.Open(4, forged);
  } catch (const crucible::RecoverableOpenError& error) {
    refused = std::string(error.what()).find("RoPE mode") != std::string::npos;
  }
  assert(refused && session.Active() == 0 && engine.rope.at(0) == RopeMode::Yarn2);
  // The done event fields: fixed words and numbers.
  const std::string fields = crucible::BatchRopeFields(done3);
  assert(fields == ",\"rope\":\"yarn2\",\"rope_yarn_opens\":2,\"rope_crossings\":1,\"rope_crossing_tokens\":" +
                       std::to_string(tail.size()));
  unsetenv("CRUCIBLE_ROPE_MODE");
}

// Without a YaRN word the done event has no rope field (today's bytes).
void PlainProcessTests() {
  unsetenv("CRUCIBLE_ROPE_MODE");
  RopeMock engine;
  crucible::BatchSession<RopeMock> session(engine, 1);
  auto events = session.Open(1, Request(3, 1));
  const auto done = std::find_if(events.begin(), events.end(), [](const crucible::BatchEvent& e) { return e.done; });
  assert(done != events.end() && !done->rope_report && crucible::BatchRopeFields(*done).empty());
}
}  // namespace

int main() {
  ModeTests();
  OldEngineTests();
  ResumeTests();
  PlainProcessTests();
  return 0;
}
