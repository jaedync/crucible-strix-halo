// Abliteration per request: the request field "ablate", its start switch, the separate
// state tenant of an ablated request, the stream flag that BatchSession::Open sets, and the stock-row reference.
#include "batch_mock_engine.h"
#include "../engine/layers/ablation.h"
#include "../engine/serve/batched_serve_protocol.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace {
struct FlagMock : MockEngine {
  std::map<int, bool> ablate;
  void SetStreamAblate(int stream, bool on) { ablate[stream] = on; }
};

void Rejected(const std::string& line, const char* reason) {
  try {
    crucible::ParseServeRequest(line, 32);
    assert(false);
  } catch (const std::runtime_error& error) {
    assert(std::string(error.what()).find(reason) != std::string::npos);
  }
}

void ProtocolTests() {
  unsetenv("CRUCIBLE_ABLATE_MODE");
  // Without the start switch the field is refused: a stock engine never serves a request as ablated.
  Rejected(R"({"ids":[1],"max_tokens":1,"ablate":1})", "CRUCIBLE_ABLATE_MODE=request");
  // 0 is the stock request on any engine.
  assert(crucible::ParseServeRequest(R"({"ids":[1],"max_tokens":1,"ablate":0})", 32).ablate == 0);
  const auto stock = crucible::ParseServeRequest(R"({"ids":[1],"max_tokens":1})", 32);
  assert(stock.ablate == 0 && stock.tenant == crucible::kLocalTenant);
  setenv("CRUCIBLE_ABLATE_MODE", "request", 1);
  assert(crucible::AblateRequestMode());
  const auto ablated = crucible::ParseServeRequest(R"({"ids":[1],"max_tokens":1,"ablate":1})", 32);
  assert(ablated.ablate == 1);
  // The ablated request has its own state tenant: no slot, checkpoint or disk match crosses the modes.
  assert(ablated.tenant == crucible::AblatedTenant(crucible::kLocalTenant));
  assert(ablated.tenant != crucible::kLocalTenant);
  Rejected(R"({"ids":[1],"max_tokens":1,"ablate":2})", "ablate must be 0 or 1");
  Rejected(R"({"ids":[1],"max_tokens":1,"ablate":-1})", "ablate must be 0 or 1");
  // A named tenant: the derived key depends on it, and the field order does not matter.
  const std::string hex(64, 'a');
  const auto t1 = crucible::ParseServeRequest(R"({"ids":[1],"max_tokens":1,"ablate":1,"tenant":")" + hex + "\"}", 32);
  const auto t2 = crucible::ParseServeRequest(R"({"ids":[1],"max_tokens":1,"tenant":")" + hex + R"(","ablate":1})", 32);
  const auto t3 = crucible::ParseServeRequest(R"({"ids":[1],"max_tokens":1,"tenant":")" + hex + "\"}", 32);
  assert(t1.tenant == t2.tenant && t1.tenant != t3.tenant && t1.tenant != ablated.tenant);
  assert(t1.tenant == crucible::AblatedTenant(t3.tenant));
  // The batch open and stage commands carry the same request object.
  const auto open = crucible::ParseBatchCommand(
      R"({"op":"open","request_id":7,"request":{"ids":[1],"max_tokens":1,"spec":0,"ablate":1}})", 32);
  assert(open.request && open.request->ablate == 1 && open.request->tenant == ablated.tenant);
  const auto stage = crucible::ParseBatchCommand(R"({"op":"stage","request":{"ids":[1],"max_tokens":1,"spec":0,"ablate":1}})", 32);
  assert(stage.request && stage.request->tenant == ablated.tenant);
  setenv("CRUCIBLE_ABLATE_MODE", "global", 1);
  assert(!crucible::AblateRequestMode());
  unsetenv("CRUCIBLE_ABLATE_MODE");
}

// BatchSession::Open sets the stream flag of the request's slot before any forward of that request.
void SessionTests() {
  setenv("CRUCIBLE_ABLATE_MODE", "request", 1);
  FlagMock engine;
  crucible::BatchSession<FlagMock> session(engine, 2);
  auto ablated = Request(3);
  ablated.ablate = 1;
  ablated.tenant = crucible::AblatedTenant(ablated.tenant);
  session.Open(1, ablated);
  const int first = engine.selected;
  assert(engine.ablate.at(first));
  session.Open(2, Request(4));
  assert(engine.selected != first && !engine.ablate.at(engine.selected));
  assert(engine.ablate.at(first));
  unsetenv("CRUCIBLE_ABLATE_MODE");
}

// A stock row of a mixed step: H += post * y with the stock epilogue's fma, and the b = 0 identity gives the same.
void StockRowTests() {
  std::vector<float> y(crucible::kHidden), a(crucible::kHidden, 0.f), zero(crucible::kHidden, 0.f);
  a[5] = 1.f;
  for (int i = 0; i < crucible::kHidden; ++i) y[i] = 0.25f + 0.001f * float(i % 97) - 0.0003f * float(i % 13);
  const float post[crucible::kStreams] = {1.f, 0.5f, 2.f, 0.f};
  std::vector<float> stock(size_t(crucible::kStreams) * crucible::kHidden, 1.5f), identity(stock);
  crucible::AblateWriteStockRef(y.data(), post, stock.data());
  crucible::AblateWriteRef(y.data(), a.data(), zero.data(), post, identity.data());
  assert(std::memcmp(stock.data(), identity.data(), stock.size() * sizeof(float)) == 0);
  for (int d = 0; d < crucible::kHidden; ++d) assert(stock[d] == std::fmaf(post[0], y[d], 1.5f));
}
}  // namespace

int main() {
  ProtocolTests();
  SessionTests();
  StockRowTests();
  return 0;
}
