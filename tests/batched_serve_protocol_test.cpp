#include "../engine/serve/batched_serve_protocol.h"
#include <cassert>
int main() {
  using crucible::ParseBatchCommand;
  const auto open = ParseBatchCommand(R"({"op":"open","request_id":1,"request":{"ids":[1,2],"max_tokens":3,"spec":0}})", 32);
  assert(open.op == "open" && open.request_id == 1 && open.request->spec == 0);
  const auto spec = ParseBatchCommand(R"({"op":"open","request_id":3,"request":{"ids":[1,2],"max_tokens":3,"spec":7}})", 32);
  assert(spec.request->spec == 7);
  assert(ParseBatchCommand(R"({"op":"spec_step","request_id":3})", 32).request_id == 3);
  assert((ParseBatchCommand(R"({"op":"verify_step","request_ids":[4,2]})", 32).request_ids == std::vector<uint64_t>{4, 2}));
  const auto step = ParseBatchCommand(R"({"request_ids":[2,1],"op":"step"})", 32);
  assert(step.request_ids == std::vector<uint64_t>({2,1}));
  const auto close = ParseBatchCommand(R"({"op":"close","request_id":2})", 32);
  assert(close.request_id == 2);
  const auto plain = ParseBatchCommand(R"({"op":"prefill","request_id":5})", 32);
  assert(plain.request_id == 5 && plain.request_ids.empty() && plain.every == 0);
  const auto interleaved = ParseBatchCommand(R"({"op":"prefill","request_id":5,"request_ids":[2,3],"every":8})", 32);
  assert(interleaved.request_id == 5 && (interleaved.request_ids == std::vector<uint64_t>{2, 3}) && interleaved.every == 8);
  // The disk tier: stage names a request only, so the engine can start a disk read before the open.
  const auto stage = ParseBatchCommand(R"({"op":"stage","request":{"ids":[1,2],"max_tokens":3,"spec":0}})", 32);
  assert(stage.op == "stage" && stage.request && stage.request->ids.size() == 2 && stage.request_id == 0);
  assert(stage.position == 0 && stage.read);
  // An earlier change: the turn's place in the server's queue, and a turn that may not start a read.
  const auto queued = ParseBatchCommand(R"({"op":"stage","request":{"ids":[1,2],"max_tokens":3,"spec":0},"position":3,"read":0})", 32);
  assert(queued.position == 3 && !queued.read && queued.job == 0);
  // Fix 2 (cause.md item 10): the server's tag of the turn, for the stage and host_evict lines.
  const auto tagged = ParseBatchCommand(R"({"op":"stage","request":{"ids":[1,2],"max_tokens":3,"spec":0},"position":0,"read":1,"job":42})", 32);
  assert(tagged.job == 42 && tagged.read);
  for (const auto& bad : {
    R"({"op":"stage","request_id":1,"request":{"ids":[1,2],"max_tokens":3,"spec":0}})",
    R"({"op":"stage"})",
    R"({"op":"stage","request":{"ids":[1,2],"max_tokens":3,"spec":0},"read":2})",
    R"({"op":"stage","request":{"ids":[1,2],"max_tokens":3,"spec":0},"position":4097})",
    R"({"op":"close","request_id":1,"position":1})",
    R"({"op":"open","request_id":1,"request":{"ids":[1,2],"max_tokens":3,"spec":0},"job":1})",
    R"({"op":"prefill","request_id":5,"request_ids":[2]})",
    R"({"op":"prefill","request_id":5,"every":8})",
    R"({"op":"prefill","request_id":5,"request_ids":[5,2],"every":8})",
    R"({"op":"prefill","request_id":5,"request_ids":[2,2],"every":8})",
    R"({"op":"prefill","request_id":5,"request_ids":[],"every":8})",
    R"({"op":"prefill","request_id":5,"request_ids":[2],"every":0})",
    R"({"op":"prefill","request_id":5,"request_ids":[2],"every":48})",
    R"({"op":"verify_step","request_ids":[2],"every":8})",
    R"({"op":"prefill","request_ids":[2],"every":8})",
    R"({"op":"step","request_ids":[]})",
    R"({"op":"step","request_ids":[1,1]})",
    R"({"op":"verify_step","request_ids":[]})",
    R"({"op":"verify_step","request_id":3})",
    R"({"op":"close","request_id":0})",
    R"({"op":"close","request_id":1,"op":"close"})",
    R"({"op":"close","request_id":1,"unknown":2})",
    R"({"op":"close","request_id":1,"request_ids":[1]})",
    R"({"op":"open","request_id":1,"request":{"ids":[1],"spec":8}})",
    R"({"op":"spec_step","request_id":0})",
    R"({"op":"spec_step","request_ids":[1]})",
    R"({"op":"open","request_id":1,"request":{"ids":[1]}})",
    R"({"op":"open","request_id":-1,"request":{"ids":[1],"spec":0}})",
    R"({"op":"step","request_ids":[1,2,3,4,5,6,7,8,9]})",
    R"({"op":"close","request_id":1} [])"
  }) {
    try { ParseBatchCommand(bad, 32); assert(false); } catch (const std::runtime_error&) {}
  }
}
