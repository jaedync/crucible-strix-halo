// One copy of each session state ("One copy"): the h2h agent turns of AgentSoak
// (served_cache_soak_test.cpp) with one slot per session and slot-resident resume, run twice on the same cache:
// - "two copies" (cm2, c1a8039): every tail is saved to the host at once, so the host often holds the state that its
//   slot also holds;
// - "one copy": a tail copy takes only free room, and a slot saves its tail when the next request overwrites it.
// It prints the host bytes per session at each turn end (all, and the part that copies a slot) and the tokens that
// each turn and each continuation job prefill. Numbers only, no content.
#include "../engine/serve/batch_conversation_cache.h"
#include "../engine/serve/batch_session.h"
#include <cassert>
#include <cstdio>
#include <optional>
using namespace crucible;
struct Snapshot { std::vector<int> hist; ConversationResume resume; };

namespace {
constexpr int kRows = 4096;
constexpr uint64_t kIdleAvailable = uint64_t(40) << 30;  // the budget binds, not host memory
size_t SnapshotBytes(size_t tokens) { return size_t(152292328 + 32264 * tokens); }  // served_cache_soak_test.cpp

struct Totals {
  size_t turns = 0, misses = 0, prefilled = 0, cont_prefilled = 0, cont_max = 0, spills = 0, refused = 0;
  double host = 0, copies = 0;  // sums of host bytes per session at turn ends
  size_t ends = 0;
};

class SlotAgents {
 public:
  struct Job { std::vector<int> ids; size_t boundary = 0, cached = 0; int slot = -1; bool from_tail = false; };
  SlotAgents(size_t budget, int slots, bool one_copy)
      : one_copy_(one_copy), cache_(budget, {}, size_t(slots)), held_(size_t(slots)), busy_(size_t(slots)) {
    policy_.prefill_rows = kRows;
  }
  Job Open(std::vector<int> ids, size_t boundary) {
    Job job{std::move(ids), boundary};
    const auto& r = job.ids;
    job.slot = PickSlot(r);
    busy_[size_t(job.slot)] = true;
    auto& held = held_[size_t(job.slot)];
    const bool resumable = held && held->hist.size() < r.size() && std::equal(held->hist.begin(), held->hist.end(), r.begin());
    std::optional<BatchResident> resident;
    if (resumable) resident = BatchResident{held->hist.size(), 0, held->guarded};
    if (held) Release(job.slot, resumable);
    if (resident) {
      job.from_tail = true;
      job.cached = resident->length;
      if (const auto tail = cache_.FindTail(r, policy_)) {
        if (HostTailFirst(tail->snapshot->hist.size(), 0, *resident)) job.cached = tail->snapshot->hist.size();
      } else if (one_copy_) {
        cache_.Touch(r, policy_);
      }
      return job;
    }
    if (const auto tail = cache_.FindTail(r, policy_)) job.cached = tail->snapshot->hist.size(), job.from_tail = true;
    if (!job.cached)
      if (const auto entry = cache_.Find(r, policy_, boundary, r.size() - 1, kRows)) job.cached = entry->snapshot->hist.size();
    for (size_t cursor = job.cached + kRows; !job.from_tail && cursor < r.size(); cursor += kRows)
      for (const auto& frontier : BatchCacheFrontiers(r.size(), boundary, kRows, 0))
        if (frontier.position == cursor) Capture(r, boundary, frontier);
    return job;
  }
  // http.py close_thinking cancels the first job: no tail, the slot holds nothing usable.
  void Cancel(const Job& job) { busy_[size_t(job.slot)] = false; }
  void Tail(const Job& job, const std::vector<int>& generated, bool guard) {
    busy_[size_t(job.slot)] = false;
    if (generated.size() < 2) return;
    std::vector<int> hist(job.ids);
    hist.insert(hist.end(), generated.begin(), generated.end() - 1);
    const bool trusted = job.from_tail && !guard;
    totals.refused += !Save(hist, trusted, guard, one_copy_);
    held_[size_t(job.slot)] = Held{hist, trusted, guard, ++tick_};
    if (one_copy_) cache_.SetResident(job.slot, hist, 0, policy_);
  }
  // Host bytes per session now: all entries, and the part that copies a slot's state.
  void Sample(int sessions) {
    size_t copies = 0;
    for (const auto& held : held_)
      if (held && cache_.HoldsTail(held->hist, policy_, 0)) copies += SnapshotBytes(held->hist.size());
    totals.host += double(cache_.Bytes()) / sessions;
    totals.copies += double(copies) / sessions;
    ++totals.ends;
  }
  Totals totals;

 private:
  struct Held { std::vector<int> hist; bool trusted = false, guarded = false; uint64_t used = 0; };
  // BatchSession::FreeSlot: the slot that holds the prompt's tail, then an empty slot, then the least recent.
  int PickSlot(const std::vector<int>& r) const {
    int best = -1;
    std::tuple<int, uint64_t> key{};
    for (int slot = 0; slot < int(held_.size()); ++slot) {
      if (busy_[size_t(slot)]) continue;
      const auto& held = held_[size_t(slot)];
      const std::tuple<int, uint64_t> k =
          !held ? std::tuple<int, uint64_t>{1, 0}
          : held->hist.size() < r.size() && std::equal(held->hist.begin(), held->hist.end(), r.begin())
              ? std::tuple<int, uint64_t>{2, 0} : std::tuple<int, uint64_t>{0, ~held->used};
      if (best < 0 || k > key) best = slot, key = k;
    }
    assert(best >= 0);
    return best;
  }
  void Release(int slot, bool resumed) {
    auto held = std::move(held_[size_t(slot)]);
    held_[size_t(slot)].reset();
    if (!one_copy_) return;
    cache_.ClearResident(slot);
    if (!held || cache_.HoldsTail(held->hist, policy_, 0)) return;
    ++totals.spills;
    totals.refused += !Save(held->hist, held->trusted || resumed, held->guarded, false);
  }
  bool Save(const std::vector<int>& hist, bool trusted, bool guarded, bool resident) {
    return cache_.CaptureTail(hist, policy_, SnapshotBytes(hist.size()), kIdleAvailable, [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist = hist; s->resume.length = hist.size();
      return std::shared_ptr<const Snapshot>(s);
    }, trusted, guarded, resident);
  }
  void Capture(const std::vector<int>& ids, size_t boundary, const CacheFrontier& frontier) {
    const size_t position = frontier.position;
    cache_.Capture(ids, policy_, frontier, SnapshotBytes(position), kIdleAvailable, [&] {
      auto s = std::make_shared<Snapshot>();
      s->hist.assign(ids.begin(), ids.begin() + position);
      s->resume.length = position; s->resume.prompt_only = true; s->resume.prompt_boundary = boundary;
      return std::shared_ptr<const Snapshot>(s);
    });
  }
  const bool one_copy_;
  ConversationPolicy policy_;
  BatchConversationCache<Snapshot> cache_;
  std::vector<std::optional<Held>> held_;
  std::vector<bool> busy_;
  uint64_t tick_ = 0;
};

struct Workload { const char* name; size_t completion; bool think; size_t tool_result; };

// The h2h turn sequence of AgentSoak: code, tool, code_sampled, think (closed after one token, continued on a second
// job, the client keeps only the answer), prose. `slots` below `sessions`: the sessions take turns one at a time
// (agent sessions wait for tools and users), else all sessions run each turn at once.
Totals Run(int sessions, int slots, size_t base_tokens, int rounds, size_t budget, bool one_copy) {
  constexpr int kGenPrompt[] = {7, 8, 9, 10};
  constexpr int kImEnd = 11;
  constexpr size_t kUser = 60, kClose = 25, kCap = 1;
  const Workload workloads[] = {{"code", 512, false, 0}, {"tool", 400, false, 120}, {"code_sampled", 512, false, 0},
                                {"think", 768, true, 0}, {"prose", 512, false, 0}};
  int fresh = 1 << 24;
  const auto Fresh = [&](size_t count) { std::vector<int> ids(count); for (auto& id : ids) id = fresh++; return ids; };
  const auto Join = [](std::vector<int> a, const std::vector<int>& b) { a.insert(a.end(), b.begin(), b.end()); return a; };
  SlotAgents agents(budget, slots, one_copy);
  const bool together = slots >= sessions;
  std::vector<int> base(base_tokens);
  for (size_t i = 0; i < base_tokens; ++i) base[i] = int(i) + 100;
  std::vector<std::vector<int>> history(sessions);
  std::vector<size_t> previous(sessions, 0);
  for (int s = 0; s < sessions; ++s) {
    history[s] = Join(base, Fresh(kUser));
    agents.Cancel(agents.Open(Join(history[s], {7, 8, 9, 10}), history[s].size()));  // max_tokens 1: no tail
  }
  for (int round = 0; round < rounds; ++round)
    for (const auto& work : workloads) {
      std::vector<SlotAgents::Job> first(sessions), cont(sessions);
      std::vector<std::vector<int>> output(sessions);
      const auto open = [&](int s) {
        auto ids = Join(history[s], Fresh(kUser));
        ids.insert(ids.end(), std::begin(kGenPrompt), std::end(kGenPrompt));
        const size_t boundary = ids.size() - 4;
        first[s] = agents.Open(std::move(ids), boundary);
      };
      const auto think = [&](int s) {
        if (!work.think) { output[s] = Fresh(work.completion); return; }
        output[s] = Join(Fresh(kCap), Fresh(kClose));
        agents.Cancel(first[s]);
        cont[s] = agents.Open(Join(first[s].ids, output[s]), first[s].boundary);
        const size_t prefilled = cont[s].ids.size() - cont[s].cached;
        agents.totals.cont_prefilled += prefilled;
        agents.totals.cont_max = std::max(agents.totals.cont_max, prefilled);
      };
      const auto finish = [&](int s) {
        if (!work.think) { agents.Tail(first[s], output[s], false); return; }
        const auto answer = Fresh(work.completion - kCap - kClose);
        agents.Tail(cont[s], answer, true);
        output[s] = Join(output[s], answer);
      };
      if (together) {
        for (int s = 0; s < sessions; ++s) open(s);
        for (int s = 0; s < sessions; ++s) think(s);
        for (int s = 0; s < sessions; ++s) finish(s);
        agents.Sample(sessions);
      } else {
        for (int s = 0; s < sessions; ++s) { open(s); think(s); finish(s); agents.Sample(sessions); }
      }
      for (int s = 0; s < sessions; ++s) {
        const auto& job = first[s];
        const size_t prefilled = job.ids.size() - job.cached, appended = job.ids.size() - previous[s];
        ++agents.totals.turns;
        agents.totals.prefilled += prefilled;
        agents.totals.misses += previous[s] && prefilled > appended + kRows;
        previous[s] = job.ids.size();
        history[s] = work.think ? Join(Join(std::vector<int>(job.ids.begin(), job.ids.begin() + job.boundary), {7, 8}),
                                       Fresh(work.completion / 2))
                                : Join(job.ids, output[s]);
        history[s].push_back(kImEnd);
        if (work.tool_result) history[s] = Join(history[s], Fresh(work.tool_result));
      }
    }
  return agents.totals;
}

void Print(const char* mode, int sessions, int slots, size_t base, size_t budget, const Totals& t) {
  std::printf("%-9s sessions=%d slots=%d base=%zu cache_mib=%zu host_mib_per_session=%.0f copy_mib_per_session=%.0f "
              "misses=%zu/%zu prefilled=%zu cont_prefilled=%zu cont_max=%zu spills=%zu refused=%zu\n", mode, sessions,
              slots, base, budget >> 20, t.host / t.ends / 1048576.0, t.copies / t.ends / 1048576.0, t.misses, t.turns,
              t.prefilled, t.cont_prefilled, t.cont_max, t.spills, t.refused);
}
}  // namespace

int main() {
  bool pass = true;
  for (const int slots : {4, 3, 2})
  for (const size_t mib : {size_t(10240), size_t(16384)})
    for (const size_t base : {size_t(31100), size_t(60000), size_t(100000)}) {
      const size_t budget = mib << 20;
      const Totals two = Run(4, slots, base, 2, budget, false), one = Run(4, slots, base, 2, budget, true);
      Print("two-copy", 4, slots, base, budget, two);
      Print("one-copy", 4, slots, base, budget, one);
      // One copy never prefills more, holds fewer slot copies, and misses no more turns. Up to 60K, 10 GiB holds
      // every session's resume point, with 2 to 4 slots for 4 sessions.
      pass = pass && one.prefilled + one.cont_prefilled <= two.prefilled + two.cont_prefilled &&
             one.copies <= two.copies && one.misses <= two.misses && (base > 60000 || one.misses == 0);
    }
  std::fflush(stdout);
  assert(pass);
  return pass ? 0 : 1;
}
