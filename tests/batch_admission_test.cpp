// The start-time refusal text: the numbers the user needs, and suggested settings that the check really accepts.
// The suggestion is tried with the engine's own admission calculation, so a printed value must pass
// AdmissionFits and the next step up must fail.
#include "../engine/runtime/batch_admission.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>

using namespace crucible;

namespace {
constexpr uint64_t GiB = 1ull << 30;
constexpr uint64_t MiB = 1ull << 20;
constexpr int kMtpRows = 33280;              // MtpSlotRows(262144, 256): the serve default --mtp-window
constexpr int kMaxRows = 8;                  // kMaxRows: a short slot keeps kMaxRows rows of headroom
constexpr int kVisionChunk = 8;              // VisionReserveRows(0, 0): --prefill 0
constexpr uint64_t kVisionScratch = 478150656;   // the encoder scratch that VisionReserveBytes charges once
constexpr uint64_t kVisionResidual = 75497472;   // VisionSlotBytes: 4096 patches * 4 * 1152 * sizeof(float)
constexpr int kVisionMaxTokens = 4095;       // one image, 4096 patches

// One pack's state bytes, with the shape and the constants of SequenceState::ServeBytes (sequence_state.h,
// cache_storage.h, fixed_state.h) and VisionReserveBytes (engine/engine.hip). FixedBytes is 151,770,112.
struct Pack {
  bool compact = false, vision = false;
  int shorts = 0, short_ctx = 0;  // --short-streams and --short-ctx (0: no short slot)
  uint64_t Cache(int rows) const {
    return uint64_t(rows) * (compact ? 2048 : 2560) + (compact ? 8192 : 0) + 16384 * uint64_t((rows + 127) / 128);
  }
  uint64_t Serve(int context) const { return 151770112ull + 12 * Cache(context) + Cache(std::min(context, kMtpRows)); }
  // What a slot holds before the check. Compact storage omits the MTP cache until the warmup promotes it; full
  // storage always holds it (cache_storage.h, MissingMtpCacheBytes).
  uint64_t Missing(int context) const { return compact ? Cache(std::min(context, kMtpRows)) : 0; }
  uint64_t FirstSlot(int context) const { return Serve(context) - Missing(context); }
  uint64_t VisionSlot(int context) const {
    const uint64_t rows = std::min(context, kVisionChunk + 2 * kVisionMaxTokens);
    return kVisionResidual + rows * 2560 * 2;
  }
  // The server refuses --short-streams unless 0 < --short-ctx < --ctx and short streams < streams
  // (server/batch_backend.py, validate). A short slot keeps kMaxRows rows above --short-ctx, so its capacity can be
  // larger than the full context of a candidate setting.
  bool Allowed(int streams, int context) const {
    return shorts == 0 || (shorts < streams && short_ctx > 0 && short_ctx < context);
  }
  int SlotContext(int slot, int streams, int context) const {
    return slot < streams - (Allowed(streams, context) ? shorts : 0) ? context : short_ctx + kMaxRows;
  }
  // What a fresh start with `streams` slots at `context` charges: slot 0 is allocated before the check, the other
  // slots, the vision reserve and a missing MTP cache of slot 0 after it.
  AdmissionCharge Charge(int streams, int context) const {
    AdmissionCharge charge;
    if (streams < 1 || !Allowed(streams, context)) {
      charge.allowed = false;  // the request's own options refuse this setting
      return charge;
    }
    charge.allowed = true;
    for (int slot = 0; slot < streams; ++slot) charge.states_bytes += Serve(SlotContext(slot, streams, context));
    charge.first_slot_bytes = FirstSlot(SlotContext(0, streams, context));
    for (int slot = 1; slot < streams; ++slot) charge.new_states_bytes += Serve(SlotContext(slot, streams, context));
    charge.new_states_bytes += Missing(SlotContext(0, streams, context));
    if (vision) charge.new_states_bytes += kVisionScratch + uint64_t(streams) * VisionSlot(context);
    return charge;
  }
};

AdmissionTry Trial(const Pack& pack) {
  return [&pack](int streams, int context) { return pack.Charge(streams, context); };
}

// The inputs that Engine::ConfigureServeStreams passes now: values from the pack, and the requested setting.
AdmissionMemory Memory(const Pack& pack, uint64_t available, uint64_t cache, int streams, int context) {
  AdmissionMemory memory;
  memory.host_free = available;
  memory.device_free = available;
  memory.states_bytes = pack.Charge(streams, context).states_bytes;
  memory.allocated_bytes = pack.FirstSlot(context);  // slot 0, which exists before the check
  memory.first_slot_bytes = pack.FirstSlot(context);
  memory.new_states_bytes = pack.Charge(streams, context).new_states_bytes;
  memory.cache_budget = cache;
  memory.streams = streams;
  memory.context = context;
  return memory;
}

bool Has(const std::string& text, const std::string& part) {
  const bool found = text.find(part) != std::string::npos;
  if (!found) std::printf("missing: [%s]\n--- text ---\n%s\n---", part.c_str(), text.c_str());
  return found;
}
bool HasNot(const std::string& text, const std::string& part) { return text.find(part) == std::string::npos; }

// The value that the text prints after `label`, -1 for "none", -2 for a label that the text does not have.
int Printed(const std::string& text, const char* label) {
  size_t at = text.find(label);
  if (at == std::string::npos) return -2;
  at = text.find(": ", at);
  if (at == std::string::npos) return -2;
  at += 2;
  if (text.compare(at, 4, "none") == 0) return -1;
  int value = 0;
  for (; at < text.size() && (std::isdigit((unsigned char)text[at]) || text[at] == ','); ++at)
    if (text[at] != ',') value = value * 10 + (text[at] - '0');
  return value;
}

// The number that the text prints right after `label`, or -2.
int After(const std::string& text, const char* label) {
  size_t at = text.find(label);
  if (at == std::string::npos) return -2;
  at += std::strlen(label);
  while (at < text.size() && text[at] == ' ') ++at;
  int value = 0;
  bool digits = false;
  for (; at < text.size() && (std::isdigit((unsigned char)text[at]) || text[at] == ','); ++at)
    if (text[at] != ',') value = value * 10 + (text[at] - '0'), digits = true;
  return digits ? value : -2;
}

// A candidate setting, tried with the engine's calculation: the requested slot 0 is allocated now, the candidate's
// slot 0 before the check. A setting that the request's other options refuse does not count as a suggestion.
bool Passes(const Pack& pack, uint64_t available, uint64_t cache, int streams, int context, uint64_t slot_now) {
  const AdmissionCharge charge = pack.Charge(streams, context);
  if (!charge.allowed) return false;
  return AdmissionFits(AdmissionAvailableAt(available, slot_now, charge.first_slot_bytes), charge.new_states_bytes, cache);
}

// Every printed suggestion must pass the predicate, and one step more must fail. Nothing at or above the request
// may be printed, because the request does not fit.
void CheckSuggestions(const std::string& text, const Pack& pack, uint64_t available, uint64_t cache, int streams,
                      int context) {
  const uint64_t slot_now = pack.FirstSlot(context);
  const int best_streams = Printed(text, "Largest --streams at --ctx");
  const int best_context = Printed(text, "Largest --ctx at --streams");
  // A candidate that the request's other options refuse (short streams below the stream count, --short-ctx below
  // --ctx) cannot pass, so the expected answer includes that rule.
  for (int other = 1; other < streams; ++other)
    assert(Passes(pack, available, cache, other, context, slot_now) ==
           (pack.Allowed(other, context) && best_streams > 0 && other <= best_streams));
  for (int other = kAdmissionContextGrain; other < context; other += kAdmissionContextGrain)
    assert(Passes(pack, available, cache, streams, other, slot_now) ==
           (pack.Allowed(streams, other) && best_context > 0 && other <= best_context));
  assert(best_streams != streams && best_context != context);
  // The stream suggestion must be usable: the user keeps --short-streams, so it must stay below --streams.
  if (best_streams > 0) assert(pack.Allowed(best_streams, context));
  if (best_context > 0) assert(pack.Allowed(streams, best_context));
}

// BytesText with one decimal for the unit and the exact bytes once.
void CheckFormat() {
  assert(GibText(GiB) == "1.0 GiB");
  assert(GibText(8 * GiB + GiB / 2) == "8.5 GiB");
  assert(MibText(512 * MiB) == "512.0 MiB");
  assert(GroupDigits(0) == "0" && GroupDigits(999) == "999" && GroupDigits(1000) == "1,000");
  assert(BytesText(8544751616ull) == "8.0 GiB (8,544,751,616 bytes)");
  assert(BytesText(0) == "0.0 GiB (0 bytes)");
  assert(BytesText(std::numeric_limits<uint64_t>::max()).find("17179869184.0 GiB") == 0);
  assert(kAdmissionReserve == 9 * GiB);
  assert(kAdmissionOsReserve == 8 * GiB && kAdmissionPlanReserve == 512 * MiB && kAdmissionHostReserve == 512 * MiB);
}

// The accept or refuse decision is the trunk one, so a change here is a change of behaviour.
void CheckPredicate() {
  const uint64_t huge = std::numeric_limits<uint64_t>::max();
  for (const uint64_t available : {uint64_t(0), uint64_t(1), kAdmissionReserve - 1, kAdmissionReserve,
                                   kAdmissionReserve + 1, uint64_t(12) * GiB, huge})
    for (const uint64_t states : {uint64_t(0), uint64_t(1), GiB, uint64_t(30) * GiB, huge})
      for (const uint64_t cache : {uint64_t(0), uint64_t(1), GiB, huge}) {
        bool trunk;
        if (available <= kAdmissionReserve) trunk = false;
        else if (states > available - kAdmissionReserve) trunk = false;
        else trunk = cache <= available - kAdmissionReserve - states;
        assert(AdmissionFits(available, states, cache) == trunk);
      }
}
}  // namespace

int main() {
  CheckFormat();
  CheckPredicate();
  const Pack full;
  Pack compact_pack; compact_pack.compact = true;
  Pack vision_pack; vision_pack.vision = true;

  // The user's report has two measurements: MemAvailable 9,331,200 kB and device free 8,544,751,616 bytes. The check
  // keeps the smaller one, and the text shows both.
  {
    AdmissionMemory memory = Memory(full, 8'544'751'616ull, 0, 1, 256000);
    memory.host_free = 9'555'148'800ull;
    const std::string text = BatchAdmissionError(memory, Trial(full));
    assert(Has(text, "free host memory: 8.9 GiB (9,555,148,800 bytes)"));
    assert(Has(text, "free device memory: 8.0 GiB (8,544,751,616 bytes)"));
    assert(Has(text, "Free memory is at or below the reserve"));
    CheckSuggestions(text, full, 8'544'751'616ull, 0, 1, 256000);
  }

  // The user's report: 8.0 GiB free against a 9.0 GiB reserve. A lower --ctx does help, because the first slot is
  // allocated at --ctx before the check and a smaller one leaves more free.
  {
    const uint64_t available = 8'544'751'616ull;
    const std::string text = BatchAdmissionError(Memory(full, available, 0, 1, 256000), Trial(full));
    assert(!text.empty());
    assert(Has(text, "1 stream at 256,000 context needs 7.9 GiB (8,498,762,752 bytes) of state"));
    assert(Has(text, "All of that state is already allocated."));
    assert(Has(text, "free host memory: 8.0 GiB (8,544,751,616 bytes)"));
    assert(Has(text, "free device memory: 8.0 GiB (8,544,751,616 bytes)"));
    assert(Has(text, "start-time reserve: 9.0 GiB (9,663,676,416 bytes): system 8.0 GiB, plans 512.0 MiB, host 512.0 MiB"));
    assert(Has(text, "conversation cache budget: 0.0 GiB (0 bytes)"));
    assert(Has(text, "Free memory is at or below the reserve"));
    assert(Has(text, "no stream count fits"));
    assert(Has(text, "more than 9.0 GiB (9,663,676,416 bytes) free now"));
    assert(Has(text, "Stop other memory-heavy programs, if any"));
    assert(Has(text, "start with a lower --ctx"));
    assert(Has(text, "text only"));
    // The old text sent the user to --streams and --ctx, and claimed that other programs caused it.
    assert(HasNot(text, "cannot help"));
    assert(HasNot(text, "other programs hold"));
    // The largest --ctx that passes is printed, and it passes.
    assert(Printed(text, "Largest --ctx at --streams") > 0);
    CheckSuggestions(text, full, available, 0, 1, 256000);
  }
  // available == reserve: still the below-reserve case, and a lower --ctx can pass.
  {
    const uint64_t available = kAdmissionReserve;
    const std::string text = BatchAdmissionError(Memory(full, available, 0, 1, 131072), Trial(full));
    assert(Has(text, "Free memory is at or below the reserve"));
    CheckSuggestions(text, full, available, 0, 1, 131072);
  }
  // The host can be the tighter side.
  {
    AdmissionMemory memory = Memory(full, 7 * GiB, 0, 4, 32768);
    memory.device_free = 40 * GiB;
    const std::string text = BatchAdmissionError(memory, Trial(full));
    assert(Has(text, "free host memory: 7.0 GiB (7,516,192,768 bytes)"));
    assert(Has(text, "Free memory is at or below the reserve"));
    CheckSuggestions(text, full, 7 * GiB, 0, 4, 32768);
  }

  // Reviewer repro (a): the cache budget alone is bigger than the room, so no stream count fits. The old text
  // printed --streams 1, which fails.
  {
    const uint64_t available = 27'057'563'648ull, cache = 16'589ull * MiB;
    const std::string text = BatchAdmissionError(Memory(full, available, cache, 4, 262144), Trial(full));
    assert(!text.empty());
    assert(Printed(text, "Largest --streams at --ctx") == -1);
    assert(Printed(text, "Largest --ctx at --streams") > 0);
    CheckSuggestions(text, full, available, cache, 4, 262144);
  }

  // Reviewer repro (b): compact state. ServeBytes already includes the compact MTP cache, and the check charges the
  // missing MTP cache of slot 0 again. The old text printed --streams 2, which needs one byte too many.
  {
    const uint64_t available = 16'805'499'903ull;
    const std::string text = BatchAdmissionError(Memory(compact_pack, available, 0, 3, 262144), Trial(compact_pack));
    assert(!text.empty());
    assert(Printed(text, "Largest --streams at --ctx") == 1);
    assert(Printed(text, "Largest --ctx at --streams") > 0);
    assert(Has(text, "6.5 GiB (6,996,972,544 bytes) of it is already allocated"));
    CheckSuggestions(text, compact_pack, available, 0, 3, 262144);
  }

  // Reviewer round-2 repro (1): compact with --prefill 0 at a context below the MTP window. The compact constructor
  // omits the MTP cache, so the add-back of slot 0 must not include it: the old text printed --ctx 16384, which
  // needs 651,006,976 bytes against a room of 615,355,392.
  {
    Pack small; small.compact = true;  // --prefill 0 does not change the state, only the vision policy
    const uint64_t available = 9'851'212'800ull;
    const std::string text = BatchAdmissionError(Memory(small, available, 0, 2, 32768), Trial(small));
    assert(!text.empty());
    assert(Printed(text, "Largest --ctx at --streams") > 0);
    assert(Printed(text, "Largest --ctx at --streams") != 16384);
    CheckSuggestions(text, small, available, 0, 2, 32768);
  }

  // Reviewer round-2 repro (2): --short-streams 3 with --streams 4. The user keeps --short-streams, so --streams 3
  // cannot start (the server needs short streams below the stream count). The old text printed --streams 3.
  {
    Pack shorted;
    shorted.shorts = 3;
    shorted.short_ctx = 128;
    const uint64_t available = 9'977'120'768ull;
    const std::string text = BatchAdmissionError(Memory(shorted, available, 0, 4, 262144), Trial(shorted));
    assert(!text.empty());
    assert(Printed(text, "Largest --streams at --ctx") == -1);
    CheckSuggestions(text, shorted, available, 0, 4, 262144);
  }
  // A short slot keeps kMaxRows rows above --short-ctx, so a candidate whose full context is just below that capacity
  // still allocates the short slot, not a full one. The server rule is --short-ctx below --ctx, nothing more.
  {
    Pack shorted;
    shorted.shorts = 1;
    shorted.short_ctx = 128;
    const AdmissionCharge charge = shorted.Charge(2, 129);
    assert(charge.allowed);
    assert(charge.new_states_bytes == shorted.Serve(136));  // the short slot alone, at 128 + kMaxRows
    assert(!shorted.Charge(1, 129).allowed);               // short streams must stay below streams
    assert(!shorted.Charge(2, 128).allowed);               // --short-ctx must stay below --ctx
    const std::string text = BatchAdmissionError(Memory(shorted, 9'763'676'416ull, 0, 2, 200000), Trial(shorted));
    assert(!text.empty());
    CheckSuggestions(text, shorted, 9'763'676'416ull, 0, 2, 200000);
  }

  // Reviewer repro (c): vision on and --prefill 0. The vision reserve is charged for every slot, so a suggestion
  // that ignores it fails. The old text printed --streams 2 and --ctx 172,160, which both fail.
  {
    const uint64_t available = 18'360'620'032ull;
    const std::string text = BatchAdmissionError(Memory(vision_pack, available, 0, 3, 262144), Trial(vision_pack));
    assert(!text.empty());
    assert(Printed(text, "Largest --streams at --ctx") == 1);
    assert(Printed(text, "Largest --ctx at --streams") > 0);
    CheckSuggestions(text, vision_pack, available, 0, 3, 262144);
  }

  // The state fits, but the cache budget does not: the largest budget that fits.
  {
    const uint64_t available = 14 * GiB, cache = 4900ull * MiB;
    const std::string text = BatchAdmissionError(Memory(full, available, cache, 2, 65536), Trial(full));
    assert(!text.empty());
    assert(Has(text, "conversation cache budget: 4.8 GiB (5,138,022,400 bytes)"));
    assert(Has(text, "The state fits, but the conversation cache budget does not."));
    const int mib = After(text, "Reduce --conversation-cache-mib to");
    assert(mib > 0);
    const AdmissionCharge charge = full.Charge(2, 65536);
    assert(AdmissionFits(available, charge.new_states_bytes, uint64_t(mib) * MiB));
    assert(!AdmissionFits(available, charge.new_states_bytes, uint64_t(mib + 1) * MiB));
  }

  // A setting that fits has no text.
  assert(BatchAdmissionError(Memory(full, 64 * GiB, 0, 4, 262144), Trial(full)).empty());
  assert(BatchAdmissionError(Memory(compact_pack, 64 * GiB, 4 * GiB, 8, 262144), Trial(compact_pack)).empty());
  // The exact boundary of the predicate: one byte less does not fit.
  {
    const AdmissionCharge charge = full.Charge(2, 65536);
    const uint64_t exact = kAdmissionReserve + charge.new_states_bytes;
    assert(BatchAdmissionError(Memory(full, exact, 0, 2, 65536), Trial(full)).empty());
    assert(!BatchAdmissionError(Memory(full, exact - 1, 0, 2, 65536), Trial(full)).empty());
  }

  // Integer edges: no overflow and no underflow, and no suggestion when nothing can fit.
  {
    AdmissionMemory memory = Memory(full, std::numeric_limits<uint64_t>::max(), 0, 2, 262144);
    memory.new_states_bytes = std::numeric_limits<uint64_t>::max();
    const std::string text = BatchAdmissionError(memory, [](int, int) {
      AdmissionCharge charge;
      charge.states_bytes = charge.first_slot_bytes = charge.new_states_bytes = std::numeric_limits<uint64_t>::max();
      return charge;
    });
    assert(Has(text, "Largest --streams at --ctx 262,144: none."));
    assert(Has(text, "Largest --ctx at --streams 2: none."));
  }
  {
    // A huge cache budget with free memory at the top: the state fits, so the cache budget is the cause.
    const std::string text = BatchAdmissionError(
        Memory(full, std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max(), 2, 262144), Trial(full));
    assert(Has(text, "Reduce --conversation-cache-mib to"));
    assert(AdmissionFits(std::numeric_limits<uint64_t>::max(), full.Charge(2, 262144).new_states_bytes, 0));
  }
  {
    // One byte of room: only the minimum stream count fits, and the byte counts do not underflow.
    const std::string text = BatchAdmissionError(Memory(full, kAdmissionReserve + 1, 0, 2, 1), Trial(full));
    assert(Has(text, "Largest --streams at --ctx 1: 1."));
    assert(Has(text, "Largest --ctx at --streams 2: none."));
    assert(Has(text, "free for new state after the reserve and the cache: 0.0 GiB (1 bytes)"));
    CheckSuggestions(text, full, kAdmissionReserve + 1, 0, 2, 1);
  }

  const auto available = AvailableHostMemory();
  assert(available && *available > 0);
  std::printf("batch admission text: PASS\n");
  return 0;
}
