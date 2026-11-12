// The miss ledger in BatchSession: an open asks the ledger hook for the nearest host, lost and disk
// states, adds the slot tails and slot points itself, and its done event carries the verdict and the writer wait in
// closed words and numbers (BatchCacheFields). A session without the hook prints no ledger field.
#include "batch_mock_engine.h"
#include <numeric>

namespace {
using crucible::BatchCacheHooks;
using crucible::BatchEvent;
using crucible::ServeRequest;

BatchEvent Run(crucible::BatchSession<MockEngine>& session, uint64_t id, const ServeRequest& request) {
  auto events = session.Open(id, request);
  while (!events.back().done) events = session.Step({id});
  return events.back();
}
bool Has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
}  // namespace

int main() {
  MockEngine engine;
  engine.context = 4096;
  engine.prefill_rows = 256;
  crucible::BatchSession<MockEngine> session(engine, 2);
  size_t restore = 0;
  BatchCacheHooks hooks;
  hooks.restore = [&](const ServeRequest& r) -> crucible::BatchRestore {
    if (!restore) return 0;
    engine.histories[engine.selected].assign(r.ids.begin(), r.ids.begin() + std::ptrdiff_t(restore));
    crucible::BatchRestore hit{restore};
    hit.tier = crucible::BatchCacheTier::Host;
    return hit;
  };
  hooks.status = [] { return crucible::BatchCacheStatus{}; };
  crucible::LedgerTiers tiers;
  tiers.lost = {768, 768, "evict", 4};
  int asked = 0;
  hooks.ledger = [&](const ServeRequest&) { ++asked; return tiers; };
  double drained = -1;
  hooks.drained = [&] { const double seconds = drained; drained = -1; return seconds; };
  session.SetCacheHooks(hooks);
  auto turn = Request(1, 2);
  turn.ids.resize(1000);
  std::iota(turn.ids.begin(), turn.ids.end(), 1);

  // A miss: the lost record holds 768 tokens of 1,000, so the turn names its cause.
  BatchEvent done = Run(session, 1, turn);
  assert(asked == 1 && done.ledger_reported && done.cache_reported);
  assert(std::string(done.ledger.lost_cause) == "evict" && std::string(done.ledger.best_tier) == "lost");
  assert(done.ledger.best_shared == 768 && done.ledger.resume_gap == 768 && done.ledger.lineage == 4);
  std::string fields = crucible::BatchCacheFields(done);
  assert(Has(fields, ",\"best_shared_tokens\":768,\"disk_shared_tokens\":0,\"resume_gap_tokens\":768,"
                     "\"best_tier\":\"lost\",\"lost_cause\":\"evict\",\"resume_tier\":\"none\",\"resume_point\":\"none\""));
  assert(!Has(fields, "handover_drain_s") && !Has(fields, "lineage"));

  // A host hit of the same 768 tokens: the open used the best state. The release before it waited for the writer.
  restore = 768;
  drained = 0.5;
  done = Run(session, 2, turn);
  assert(std::string(done.ledger.lost_cause) == "none" && done.ledger.resume_gap == 0);
  assert(std::string(done.ledger.resume_tier) == "host" && std::string(done.ledger.resume_point) == "frontier");
  fields = crucible::BatchCacheFields(done);
  assert(Has(fields, "\"lost_cause\":\"none\"") && Has(fields, ",\"handover_drain_s\":0.500000"));

  // The slot tier: a disk state of 900 tokens that this open could not use, so the disk is the best tier.
  restore = 0;
  tiers = {};
  tiers.disk = {900, 900, nullptr, 0};
  done = Run(session, 3, turn);
  assert(std::string(done.ledger.best_tier) == "disk" && std::string(done.ledger.lost_cause) == "unknown");
  assert(done.ledger.disk_shared == 900);

  // A word outside a closed set never reaches the line.
  done.ledger.best_tier = "Bad Word";
  assert(!Has(crucible::BatchCacheFields(done), "best_tier"));

  // slots_durable: a slot whose tail the disk holds counts at the next open. Without the durable hook: no field.
  assert(!Has(crucible::BatchCacheFields(done), "slots_durable"));
  hooks.tail = [&](const ServeRequest& r, std::span<const int> generated, int,
                   crucible::BatchTailSource) -> std::optional<crucible::BatchResident> {
    return crucible::BatchResident{r.ids.size() + generated.size() - 1, 0, false};
  };
  std::vector<size_t> probed;
  hooks.durable = [&](int, std::span<const int> key, const crucible::StateTenant&) {
    probed.push_back(key.size());
    return key.size() > 1000;
  };
  session.SetCacheHooks(hooks);
  done = Run(session, 5, turn);  // no slot holds a tail yet
  assert(done.slots_durable == 0 && probed.empty());
  assert(Has(crucible::BatchCacheFields(done), ",\"slots_durable\":0"));
  auto other = turn;
  other.ids.assign(turn.ids.rbegin(), turn.ids.rend());
  done = Run(session, 6, other);  // the tail of turn 5 (1,001 tokens) is on disk
  assert(done.slots_durable == 1 && probed.size() == 1 && probed[0] == 1001);
  assert(Has(crucible::BatchCacheFields(done), ",\"slots_durable\":1"));
  done = Run(session, 7, turn);  // both tails: 1,001 tokens each
  assert(done.slots_durable == 2);
  hooks.durable = {};
  hooks.tail = {};
  session.SetCacheHooks(hooks);

  // DurableMemo: one index probe for each key until the disk adds a state (`written`) or removes one (`removed`).
  crucible::DurableMemo<int> memo;
  int probes = 0;
  bool on_disk = false;
  const auto probe = [&] { ++probes; return on_disk; };
  const std::vector<int> a(2000, 7), b(2000, 8);
  assert(!memo.Held(a, 1, 0, 0, probe) && !memo.Held(a, 1, 0, 0, probe) && probes == 1);
  assert(!memo.Held(a, 2, 0, 0, probe) && probes == 2);  // another tenant is another state
  on_disk = true;
  assert(memo.Held(a, 1, 1, 0, probe) && probes == 3);   // a write: the probe runs again
  assert(memo.Held(a, 1, 2, 0, probe) && probes == 3);   // a held state stays held through writes
  on_disk = false;
  assert(!memo.Held(a, 1, 2, 1, probe) && probes == 4);  // until the disk removes a state
  assert(!memo.Held(b, 1, 2, 1, probe) && probes == 5);
  for (int key = 0; key < 100; ++key) memo.Held(std::vector<int>{key}, 1, 2, 1, probe);
  assert(memo.Size() <= crucible::DurableMemo<int>::kLimit);

  // No ledger hook: no ledger field, as before.
  hooks.ledger = {};
  hooks.drained = {};
  session.SetCacheHooks(hooks);
  done = Run(session, 4, turn);
  fields = crucible::BatchCacheFields(done);
  assert(!done.ledger_reported && !Has(fields, "lost_cause") && !Has(fields, "best_shared_tokens") &&
         !Has(fields, "handover_drain_s") && !Has(fields, "slots_durable"));
  std::printf("miss_ledger_session_test: passed\n");
  return 0;
}
