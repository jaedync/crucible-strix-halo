// W1: the parse of a request's `tenant` field (missing,
// valid, bad length, bad characters). The cache and state tests are in batch_conversation_cache_test
// (TenantsNeverShareRam), slot_frontier_test (TenantsNeverShareSlot, the frontier copy), state_blocks_test
// (PolicyFieldCount, TenantsNeverShareDisk, TenantNotInLogs) and the LocalTenantUnchanged traces.
#include "../engine/serve/batched_serve_protocol.h"
#include <cassert>
#include <string>

using namespace crucible;

namespace {
const std::string kHex = "00112233445566778899aabbccddeeff0123456789ABCDEFfedcba9876543210";
StateTenant Expected() {
  StateTenant tenant{};
  const uint8_t bytes[32] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
                             0xff, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
                             0x32, 0x10};
  std::copy(std::begin(bytes), std::end(bytes), tenant.begin());
  return tenant;
}
std::string Line(const std::string& tenant_field) {
  return R"({"ids":[1,2],"max_tokens":3,"spec":0)" + tenant_field + "}";
}

void ParseTenant() {
  // Missing: the local tenant.
  assert(ParseServeRequest(Line(""), 32).tenant == kLocalTenant);
  // Valid: 64 hex characters, either case.
  assert(ParseServeRequest(Line(R"(,"tenant":")" + kHex + "\""), 32).tenant == Expected());
  assert(ParseServeRequest(Line(R"(,"tenant":")" + std::string(64, '0') + "\""), 32).tenant == kLocalTenant);
  assert(ParseTenantHex(kHex) == Expected() && !ParseTenantHex(""));
  // Bad length or bad characters: the fixed message, which holds no part of the value.
  const std::string bad[] = {"", kHex.substr(1), kHex + "0", kHex.substr(2), std::string(63, 'a') + "g",
                             std::string(63, 'a') + " ", std::string(62, 'a') + "\\u00e9", "0x" + kHex.substr(2)};
  for (const auto& value : bad) {
    bool refused = false;
    try {
      ParseServeRequest(Line(R"(,"tenant":")" + value + "\""), 32);
    } catch (const ServeTenantError& error) {
      refused = std::string(error.what()) == "tenant must be 64 hex characters";
    }
    assert(refused);
  }
  // A tenant that is not one JSON string (another type, or a second tenant field) is a bad tenant too: the same
  // fixed message, so batch serve refuses only that open.
  const std::string other_types[] = {R"(,"tenant":5)", R"(,"tenant":null)", R"(,"tenant":true)", R"(,"tenant":["a"])",
                                     R"(,"tenant":{"a":"b"})", R"(,"tenant":")" + kHex + R"(","tenant":")" + kHex + "\""};
  for (const auto& field : other_types) {
    bool refused = false;
    try {
      ParseServeRequest(Line(field), 32);
    } catch (const ServeTenantError& error) {
      refused = std::string(error.what()) == "tenant must be 64 hex characters";
    } catch (const std::runtime_error&) {
    }
    assert(refused);
  }
  // The tenant check runs after every other check: a request with another error keeps that error.
  bool other = false;
  try {
    ParseServeRequest(R"({"ids":[1,2],"max_tokens":3,"spec":0,"tenant":"x","cache_tail":"bad"})", 32);
  } catch (const ServeTenantError&) {
  } catch (const std::runtime_error& error) {
    other = std::string(error.what()) == "cache_tail must be auto or guard";
  }
  assert(other);
}

// The batch line: a bad tenant refuses only that open (the engine answers with a recoverable error and runs on); a
// stage names no state. A valid tenant reaches the request.
void BatchTenant() {
  const auto valid = ParseBatchCommand(
      R"({"op":"open","request_id":1,"request":{"ids":[1,2],"max_tokens":3,"spec":0,"tenant":")" + kHex + "\"}}", 32);
  assert(valid.request && valid.request->tenant == Expected() && !valid.refusal);
  const auto local = ParseBatchCommand(R"({"op":"open","request_id":1,"request":{"ids":[1,2],"max_tokens":3,"spec":0}})", 32);
  assert(local.request && local.request->tenant == kLocalTenant && !local.refusal);
  const auto open = ParseBatchCommand(
      R"({"request":{"ids":[1,2],"max_tokens":3,"spec":0,"tenant":"abc"},"op":"open","request_id":7})", 32);
  assert(open.op == "open" && open.request_id == 7 && !open.request && open.refusal == "tenant must be 64 hex characters");
  const auto stage = ParseBatchCommand(R"({"op":"stage","request":{"ids":[1,2],"max_tokens":3,"spec":0,"tenant":"1"}})", 32);
  assert(stage.op == "stage" && !stage.request && stage.refusal);
  const auto typed = ParseBatchCommand(
      R"({"op":"open","request_id":8,"request":{"ids":[1,2],"max_tokens":3,"spec":0,"tenant":null}})", 32);
  assert(typed.op == "open" && typed.request_id == 8 && !typed.request && typed.refusal == "tenant must be 64 hex characters");
  // The other command checks still apply to a refused open.
  for (const auto& bad : {R"({"op":"open","request":{"ids":[1,2],"max_tokens":3,"spec":0,"tenant":"1"}})",
                          R"({"op":"close","request_id":1,"request":{"ids":[1,2],"max_tokens":3,"spec":0,"tenant":"1"}})"}) {
    bool threw = false;
    try { ParseBatchCommand(bad, 32); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
  }
}
}  // namespace

int main() {
  ParseTenant();
  BatchTenant();
  std::puts("tenant_test passed");
}
