// Y2 host policy: the length gate that picks a request's RoPE mode, the state tenant
// of a YaRN request, and the request field that the serve protocol sets from them.
#include "../engine/runtime/rope_policy.h"
#include "../engine/runtime/ablate_request.h"
#include "../engine/serve/batched_serve_protocol.h"
#include <cassert>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

using namespace crucible;
namespace {
std::string Ids(size_t count) {
  std::string out = "[";
  for (size_t i = 0; i < count; ++i) out += i ? ",1" : "1";
  return out + "]";
}
std::string Line(size_t prompt, int max_tokens, const std::string& extra = "") {
  return "{\"ids\":" + Ids(prompt) + ",\"max_tokens\":" + std::to_string(max_tokens) + extra + "}";
}
bool Refused(const std::string& line, int ctx, const char* reason) {
  try {
    ParseServeRequest(line, ctx);
  } catch (const std::runtime_error& error) {
    return std::string(error.what()).find(reason) != std::string::npos;
  }
  return false;
}

void GateTests() {
  const RopeSetting plain = ParseRopeModeWord("plain"), auto2 = ParseRopeModeWord("auto:2"),
                    auto4 = ParseRopeModeWord("auto:4"), forced2 = ParseRopeModeWord("yarn:2");
  // Fed positions = prompt + max_tokens - 1 (the last token is never fed): 262,144 of them are plain.
  assert(RopeFedPositions(1, 1) == 1 && RopeFedPositions(262144, 1) == 262144 && RopeFedPositions(100, 262045) == 262144);
  assert(RopeModeFor(262144, 1, plain) == RopeMode::Plain);
  assert(RopeModeFor(100, 262045, plain) == RopeMode::Plain);
  assert(!RopeModeFor(100, 262046, plain));  // a plain process has no mode above 262,144
  // auto:2: plain up to 262,144 (byte-identical to today), YaRN factor 2 above it, up to 524,288.
  assert(RopeModeFor(100, 262045, auto2) == RopeMode::Plain);
  assert(RopeModeFor(100, 262046, auto2) == RopeMode::Yarn2);
  assert(RopeModeFor(262145, 1, auto2) == RopeMode::Yarn2);
  assert(RopeModeFor(524288, 1, auto2) == RopeMode::Yarn2);
  assert(RopeModeFor(500000, 24289, auto2) == RopeMode::Yarn2);
  assert(!RopeModeFor(500000, 24290, auto2));
  // auto:4: the smallest factor that holds the request.
  assert(RopeModeFor(4096, 1, auto4) == RopeMode::Plain);
  assert(RopeModeFor(300000, 1, auto4) == RopeMode::Yarn2);
  assert(RopeModeFor(524289, 1, auto4) == RopeMode::Yarn4);
  assert(RopeModeFor(1048576, 1, auto4) == RopeMode::Yarn4);
  assert(!RopeModeFor(1048576, 2, auto4));
  // The server's reply reserve (server/rope.py plain_max_tokens, R = 32,768) sends a plain request clamped to fed
  // positions of 262,144 at most, and a YaRN request with its own count. The engine gate agrees on both.
  assert(RopeModeFor(100000, 162145, auto2) == RopeMode::Plain);  // 100K + 200K, clamped
  assert(RopeModeFor(250000, 8000, auto2) == RopeMode::Plain);
  assert(RopeModeFor(240000, 32000, auto2) == RopeMode::Yarn2);
  assert(RopeModeFor(258000, 8000, auto2) == RopeMode::Yarn2);
  // The forced test word: every request in its mode, also a short one.
  assert(RopeModeFor(1, 1, forced2) == RopeMode::Yarn2);
  assert(RopeModeFor(524288, 1, forced2) == RopeMode::Yarn2 && !RopeModeFor(524289, 1, forced2));
}

void TenantTests() {
  // Plain keeps the tenant: every state of today's engine keeps its key.
  assert(RopeTenant(kLocalTenant, RopeMode::Plain) == kLocalTenant);
  const StateTenant y2 = RopeTenant(kLocalTenant, RopeMode::Yarn2), y4 = RopeTenant(kLocalTenant, RopeMode::Yarn4);
  assert(y2 != kLocalTenant && y4 != kLocalTenant && y2 != y4);
  assert(RopeTenant(kLocalTenant, RopeMode::Yarn2) == y2);  // deterministic
  // The mode of a state is visible from its tenant alone (the disk tier boundary reads it).
  assert(RopeTenantMode(kLocalTenant) == RopeMode::Plain);
  assert(RopeTenantMode(y2) == RopeMode::Yarn2 && RopeTenantMode(y4) == RopeMode::Yarn4);
  const StateTenant ablated = AblatedTenant(kLocalTenant);
  assert(RopeTenantMode(ablated) == RopeMode::Plain);
  const StateTenant ablated_y2 = RopeTenant(ablated, RopeMode::Yarn2);
  assert(ablated_y2 != y2 && ablated_y2 != ablated && RopeTenantMode(ablated_y2) == RopeMode::Yarn2);
  StateTenant named{};
  named.fill(0xab);
  assert(RopeTenant(named, RopeMode::Yarn2) != y2 && RopeTenantMode(named) == RopeMode::Plain);
}

void ProtocolTests() {
  unsetenv("CRUCIBLE_ROPE_MODE");
  // Without a word: every request is plain and keeps its tenant (today's protocol).
  const auto stock = ParseServeRequest(Line(8, 4), 32);
  assert(stock.rope == RopeMode::Plain && stock.tenant == kLocalTenant && stock.plain_tenant == kLocalTenant);
  setenv("CRUCIBLE_ROPE_MODE", "auto:2", 1);
  // At and below the gate: plain, the same tenant.
  const auto at = ParseServeRequest(Line(16, 262129), 524288);
  assert(at.rope == RopeMode::Plain && at.tenant == kLocalTenant);
  // Above it: YaRN factor 2 and its own state tenant.
  const auto above = ParseServeRequest(Line(16, 262130), 524288);
  assert(above.rope == RopeMode::Yarn2 && above.tenant == RopeTenant(kLocalTenant, RopeMode::Yarn2));
  assert(above.plain_tenant == kLocalTenant);
  // The context check comes first and keeps its text.
  assert(Refused(Line(16, 524274), 524288, "must fit the configured context"));
  // A named tenant and an ablated request derive in order: ablate first, then the mode.
  setenv("CRUCIBLE_ABLATE_MODE", "request", 1);
  const std::string hex(64, 'c');
  const auto both = ParseServeRequest(Line(16, 262130, ",\"ablate\":1,\"tenant\":\"" + hex + "\""), 524288);
  const auto named = ParseServeRequest(Line(16, 4, ",\"tenant\":\"" + hex + "\""), 524288);
  assert(both.tenant == RopeTenant(AblatedTenant(named.tenant), RopeMode::Yarn2));
  assert(both.plain_tenant == AblatedTenant(named.tenant));
  unsetenv("CRUCIBLE_ABLATE_MODE");
  // The batch open and stage commands parse the same request.
  const auto open = ParseBatchCommand("{\"op\":\"open\",\"request_id\":3,\"request\":" + Line(16, 262130, ",\"spec\":0") + "}", 524288);
  assert(open.request && open.request->rope == RopeMode::Yarn2);
  const auto stage = ParseBatchCommand("{\"op\":\"stage\",\"request\":" + Line(16, 262130, ",\"spec\":0") + "}", 524288);
  assert(stage.request && stage.request->tenant == above.tenant);
  // The server's mode word (review M1, M2): every job of one request keeps the mode that the server decided, also a
  // job that feeds fewer positions (a bounded first job, a shorter continuation).
  const char* kWordError = "rope must be plain, yarn2 or yarn4 and hold the request in a mode of this engine";
  const auto short_yarn = ParseServeRequest(Line(16, 4, ",\"rope\":\"yarn2\""), 524288);
  assert(short_yarn.rope == RopeMode::Yarn2 && short_yarn.tenant == RopeTenant(kLocalTenant, RopeMode::Yarn2));
  const auto word_plain = ParseServeRequest(Line(16, 262129, ",\"rope\":\"plain\""), 524288);
  assert(word_plain.rope == RopeMode::Plain && word_plain.tenant == kLocalTenant);
  assert(Refused(Line(16, 262130, ",\"rope\":\"plain\""), 524288, kWordError));   // plain holds 262,144 at most
  assert(Refused(Line(16, 4, ",\"rope\":\"yarn4\""), 524288, kWordError));        // auto:2 allows factor 2 only
  assert(Refused(Line(16, 4, ",\"rope\":\"yarn:2\""), 524288, kWordError));
  assert(Refused(Line(16, 4, ",\"rope\":2"), 524288, ""));  // not a string: refused by the scanner
  const auto open_word = ParseBatchCommand("{\"op\":\"open\",\"request_id\":4,\"request\":" +
                                           Line(16, 4, ",\"spec\":0,\"rope\":\"yarn2\"") + "}", 524288);
  assert(open_word.request && open_word.request->rope == RopeMode::Yarn2);
  setenv("CRUCIBLE_ROPE_MODE", "auto:4", 1);
  assert(ParseServeRequest(Line(16, 4, ",\"rope\":\"yarn4\""), 1048576).rope == RopeMode::Yarn4);
  assert(Refused(Line(600000, 4, ",\"rope\":\"yarn2\""), 1048576, kWordError));  // factor 2 holds 524,288
  // A process without an auto word takes no word: the default server never sends one.
  unsetenv("CRUCIBLE_ROPE_MODE");
  assert(Refused(Line(8, 4, ",\"rope\":\"plain\""), 32, kWordError));
  setenv("CRUCIBLE_ROPE_MODE", "yarn:2", 1);
  assert(Refused(Line(8, 4, ",\"rope\":\"yarn2\""), 524288, kWordError));
  // The forced test word: every request YaRN; the process holds one mode, so the tenant stays.
  const auto forced = ParseServeRequest(Line(8, 4), 524288);
  assert(forced.rope == RopeMode::Yarn2 && forced.tenant == kLocalTenant);
  unsetenv("CRUCIBLE_ROPE_MODE");
}
// Serial serve (serve.h): a mode change resets the stream first; the same mode keeps its state.
struct SerialMock {
  int resets = 0;
  std::vector<int> hist{1, 2};
  RopeMode mode = RopeMode::Plain;
  void Reset() { ++resets; hist.clear(); }
  RopeMode StreamRope(int) const { return mode; }
  void SetStreamRope(int, RopeMode m) {
    if (m != mode && !hist.empty()) throw std::logic_error("mode change on a held sequence");
    mode = m;
  }
};
struct OldMock { void Reset() {} };
void SerialTests() {
  SerialMock eng;
  ApplyStreamRope(eng, 0, RopeMode::Plain);
  assert(eng.resets == 0 && eng.hist.size() == 2);
  ApplyStreamRope(eng, 0, RopeMode::Yarn2);
  assert(eng.resets == 1 && eng.mode == RopeMode::Yarn2);
  eng.hist = {5};
  ApplyStreamRope(eng, 0, RopeMode::Yarn2);
  assert(eng.resets == 1 && eng.hist.size() == 1);
  OldMock old;
  ApplyStreamRope(old, 0, RopeMode::Plain);
  bool refused = false;
  try { ApplyStreamRope(old, 0, RopeMode::Yarn2); } catch (const std::runtime_error&) { refused = true; }
  assert(refused);
}
}  // namespace

int main() {
  SerialTests();
  GateTests();
  TenantTests();
  ProtocolTests();
  return 0;
}
