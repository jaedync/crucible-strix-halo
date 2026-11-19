// Start-time batch admission: one reserve covers the OS, the engine plans and the frontend, and the refusal text says
// what the user can do. Existing allocations have already reduced the free memory, so the check charges only the state
// that is not allocated yet, while the text names the state of the whole requested setting.
// Every suggested setting is tried with the engine's own admission calculation, so the text never prints a setting
// that the check would refuse.
// Pure: the caller measures the free memory and supplies the charge of a setting.
// CPU-visible: tests/batch_admission_test.cpp.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <stdexcept>

namespace crucible {
class AdmissionRefused : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline std::optional<uint64_t> AvailableHostMemory() {
  std::ifstream file("/proc/meminfo");
  for (std::string line; std::getline(file, line);) {
    if (!line.starts_with("MemAvailable:")) continue;
    std::istringstream input(line);
    std::string key, unit, extra;
    uint64_t kib;
    if (!(input >> key >> kib >> unit) || unit != "kB" || input >> extra ||
        kib > UINT64_MAX / 1024) return std::nullopt;
    return kib * 1024;
  }
  return std::nullopt;
}

// The reserve that start-time admission keeps free ("Memory before start"). The OS holds 8 GiB;
// the engine needs 512 MiB for plans and row buffers after one slot exists (ConfigureStreams); the frontend bounds 64
// pending JSON requests plus event queues and samplers in 512 MiB.
inline constexpr uint64_t kAdmissionOsReserve = 8ull << 30;
inline constexpr uint64_t kAdmissionPlanReserve = 512ull << 20;
inline constexpr uint64_t kAdmissionHostReserve = 512ull << 20;
inline constexpr uint64_t kAdmissionReserve =
    kAdmissionOsReserve + kAdmissionPlanReserve + kAdmissionHostReserve;

// A --ctx that the refusal suggests moves in 128 positions: QsaCacheBytes pools 128 positions at a time and the slot
// checkpoints copy pool blocks of 4096 floats (slot_checkpoint_io.h kPoolBlockFloats).
inline constexpr int kAdmissionContextGrain = 128;

// "8,544,751,616": the exact bytes, grouped for reading.
inline std::string GroupDigits(uint64_t value) {
  const std::string digits = std::to_string(value);
  std::string text;
  for (size_t i = 0; i < digits.size(); ++i) {
    if (i && (digits.size() - i) % 3 == 0) text += ',';
    text += digits[i];
  }
  return text;
}
// "7.9 GiB": binary units with one decimal, as the user report asks. No locale: "%.1f" is the C locale.
inline std::string UnitText(uint64_t bytes, uint64_t unit, const char* name) {
  char text[48];
  std::snprintf(text, sizeof text, "%.1f %s", double(bytes) / double(unit), name);
  return text;
}
inline std::string GibText(uint64_t bytes) { return UnitText(bytes, 1ull << 30, "GiB"); }
inline std::string MibText(uint64_t bytes) { return UnitText(bytes, 1ull << 20, "MiB"); }
// "7.9 GiB (8,498,752 bytes)": the unit and the exact bytes once.
inline std::string BytesText(uint64_t bytes) {
  return GibText(bytes) + " (" + GroupDigits(bytes) + " bytes)";
}
inline uint64_t SaturatedSum(uint64_t first, uint64_t second) {
  return first > UINT64_MAX - second ? UINT64_MAX : first + second;
}

// What one setting costs, computed by the caller with the engine's own admission code, so that a suggested setting
// can be tried before it is printed.
struct AdmissionCharge {
  uint64_t states_bytes = 0;       // state of every slot of the setting
  uint64_t first_slot_bytes = 0;   // what the setting's first slot holds before the check (its MTP cache may be absent)
  uint64_t new_states_bytes = 0;   // what the check charges now: new slots, the vision reserve, a missing MTP cache
  bool allowed = true;             // the request's other options accept this setting
};
// A fresh start at (streams, context): slot 0 of that setting is allocated before the check and the other slots
// after it, exactly as Engine::ConfigureServeStreams charges it.
using AdmissionTry = std::function<AdmissionCharge(int streams, int context)>;

// What the refusal needs. The caller measures the free memory and charges the requested setting.
struct AdmissionMemory {
  uint64_t host_free = 0;         // /proc/meminfo MemAvailable, bytes
  uint64_t device_free = 0;       // hipMemGetInfo free, bytes
  uint64_t states_bytes = 0;      // state of every requested slot, bytes
  uint64_t allocated_bytes = 0;   // the part of that state that the slots which exist already hold
  uint64_t new_states_bytes = 0;  // what the check charges now (vision and missing MTP cache included)
  uint64_t first_slot_bytes = 0;  // what the requested first slot holds before the check, bytes
  uint64_t cache_budget = 0;      // the unused conversation cache budget, bytes
  int streams = 0;                // the requested slots
  int context = 0;                // --ctx
};

// The start-time predicate, unchanged from trunk: free memory above the reserve, and the new state plus the cache
// budget inside it. A setting that fits returns no text.
inline bool AdmissionFits(uint64_t available, uint64_t new_states, uint64_t cache_budget) {
  if (available <= kAdmissionReserve) return false;
  const uint64_t room = available - kAdmissionReserve;
  if (new_states > room) return false;
  return cache_budget <= room - new_states;
}

// Free memory at the check for a fresh start at another context: the requested first slot is allocated now and the
// candidate's first slot before the check. A lower context frees the difference, so it can pass a reserve that the
// requested setting misses. A plan buffer that follows the context is inside the reserve, so this estimate is low.
// Both arguments are the bytes that the first slot really holds before the check, not its whole state: compact
// storage allocates the MTP cache later, so that cache cancels only when the two contexts share its size.
inline uint64_t AdmissionAvailableAt(uint64_t available, uint64_t slot_now, uint64_t slot_candidate) {
  const uint64_t before = SaturatedSum(available, slot_now);
  return before > slot_candidate ? before - slot_candidate : 0;
}

// The largest stream count below the request that the check accepts at the requested context, or 0 for none. The
// caller's charge says whether the request's other options accept the candidate, so a suggestion never contradicts
// --short-streams.
inline int LargestAdmissionStreams(const AdmissionMemory& memory, const AdmissionTry& try_setting) {
  // The context does not change, so the free memory at the check does not change either.
  const uint64_t available = std::min(memory.host_free, memory.device_free);
  for (int streams = memory.streams - 1; streams >= 1; --streams) {
    const AdmissionCharge charge = try_setting(streams, memory.context);
    if (charge.allowed && AdmissionFits(available, charge.new_states_bytes, memory.cache_budget)) return streams;
  }
  return 0;
}

// The largest context on the admission grain below the request and above `floor_context` that the check accepts at
// the requested stream count, or 0 for none. The scan is exact; it does not assume that the charge grows with the
// context.
inline int LargestAdmissionContext(const AdmissionMemory& memory, const AdmissionTry& try_setting,
                                   int floor_context = 0) {
  const uint64_t available = std::min(memory.host_free, memory.device_free);
  if (memory.streams < 1 || memory.context <= kAdmissionContextGrain) return 0;
  for (int context = (memory.context - 1) / kAdmissionContextGrain * kAdmissionContextGrain;
       context >= kAdmissionContextGrain && context > floor_context; context -= kAdmissionContextGrain) {
    const AdmissionCharge charge = try_setting(memory.streams, context);
    if (!charge.allowed) continue;
    const uint64_t at = AdmissionAvailableAt(available, memory.first_slot_bytes, charge.first_slot_bytes);
    if (AdmissionFits(at, charge.new_states_bytes, memory.cache_budget)) return context;
  }
  return 0;
}

// Memory stage 2, the start-time fit ("Memory fit at start"). A setting that does not fit the
// free memory becomes the largest one that does, in this order, and every other option stays:
// 1. the conversation cache budget, in whole MiB, down to min(budget, kFitCacheFloor);
// 2. the slot count at the same --ctx (the budget then takes the largest value that fits, up to the request);
// 3. --ctx in steps of kAdmissionContextGrain at the smallest slot count, only when no slot count fits.
// A setting that fits keeps every value. Each step only changes how many requests run at once, how much the cache
// keeps, or (step 3) which prompts the server accepts: no request's tokens depend on them (batching and cache
// invariance). The fit never changes the pack, the tokenizer, the sampling or the RoPE mode.
inline constexpr uint64_t kFitCacheFloor = 2048ull << 20;
// A fitted layout keeps this much free beyond the check. The engine measures the free memory again in
// ConfigureServeStreams, and after the --ctx step it frees, allocates and warms slot 0 between the two measures; an
// allocator rounding, a GPU page pool or another program can take some MiB there. Only a fitted layout pays it: a
// layout that fits keeps the exact check of the requested layout.
inline constexpr uint64_t kFitMargin = 256ull << 20;
// The native RoPE positions. A --ctx above it is a YaRN layout (the server sets the YaRN word from --ctx), so the fit
// keeps --ctx above it: a lower --ctx would turn YaRN off without a word.
inline constexpr int kFitPlainContext = 262144;

enum class FitStep : uint8_t { kRequested, kCache, kStreams, kContext, kRefused };
inline const char* FitStepName(FitStep step) {
  switch (step) {
    case FitStep::kRequested: return "requested";
    case FitStep::kCache: return "cache";
    case FitStep::kStreams: return "streams";
    case FitStep::kContext: return "context";
    case FitStep::kRefused: return "refused";
  }
  return "refused";
}
struct FitLimits {
  int min_streams = 1;    // the smallest slot count that the other options allow (--short-streams plus one)
  int floor_context = 0;  // the fit keeps --ctx above this value (kFitPlainContext for a YaRN layout)
};
struct FitLayout {
  FitStep step = FitStep::kRequested;
  int streams = 0, context = 0;
  uint64_t cache_budget = 0;
  uint64_t free_bytes = 0;  // the free memory at the check (the smaller of host and device)
  uint64_t need_bytes = 0;  // what the requested layout needs free: the reserve, its new state and its cache budget
  std::string reason;       // kRefused only: why no smaller layout fits
};

// The largest cache budget, at most `requested`, that the check accepts beside `new_states`; whole MiB when it is
// below the request. nullopt when the state alone does not fit.
inline std::optional<uint64_t> LargestAdmissionCache(uint64_t available, uint64_t new_states, uint64_t requested) {
  if (available <= kAdmissionReserve) return std::nullopt;
  const uint64_t room = available - kAdmissionReserve;
  if (new_states > room) return std::nullopt;
  const uint64_t free = room - new_states;
  return requested <= free ? requested : free / (1ull << 20) * (1ull << 20);
}

// Steps 1 to 3 on the free memory less kFitMargin. `layout` holds the requested values and the measures.
inline FitLayout FitWithMargin(const AdmissionMemory& requested, const AdmissionTry& try_setting, const FitLimits& limits,
                               FitLayout layout) {
  AdmissionMemory memory = requested;
  memory.host_free = memory.host_free > kFitMargin ? memory.host_free - kFitMargin : 0;
  memory.device_free = memory.device_free > kFitMargin ? memory.device_free - kFitMargin : 0;
  const uint64_t available = std::min(memory.host_free, memory.device_free);
  const uint64_t floor = std::min(memory.cache_budget, kFitCacheFloor);
  // 1. The cache budget.
  if (const auto cache = LargestAdmissionCache(available, memory.new_states_bytes, memory.cache_budget);
      cache && *cache >= floor) {
    layout.step = FitStep::kCache;
    layout.cache_budget = *cache;
    return layout;
  }
  // 2. Fewer slots at the same --ctx. The context does not change, so the free memory at the check does not either.
  AdmissionMemory at_floor = memory;
  at_floor.cache_budget = floor;
  if (const int streams = LargestAdmissionStreams(at_floor, try_setting); streams >= limits.min_streams) {
    const AdmissionCharge charge = try_setting(streams, memory.context);
    layout.step = FitStep::kStreams;
    layout.streams = streams;
    layout.cache_budget = LargestAdmissionCache(available, charge.new_states_bytes, memory.cache_budget).value_or(floor);
    return layout;
  }
  // 3. A lower --ctx at the smallest slot count. A lower --ctx frees the difference of slot 0 (AdmissionAvailableAt).
  AdmissionMemory smallest = at_floor;
  smallest.streams = limits.min_streams;
  if (const int context = LargestAdmissionContext(smallest, try_setting, limits.floor_context)) {
    const AdmissionCharge charge = try_setting(smallest.streams, context);
    const uint64_t at = AdmissionAvailableAt(available, memory.first_slot_bytes, charge.first_slot_bytes);
    layout.step = FitStep::kContext;
    layout.streams = smallest.streams;
    layout.context = context;
    layout.cache_budget = LargestAdmissionCache(at, charge.new_states_bytes, memory.cache_budget).value_or(floor);
    return layout;
  }
  layout.step = FitStep::kRefused;
  const std::string slots = std::to_string(smallest.streams) + (smallest.streams == 1 ? " slot" : " slots");
  if (limits.floor_context && LargestAdmissionContext(smallest, try_setting))
    layout.reason = "The memory fit stopped: " + slots + " fit only at a --ctx of " +
                    GroupDigits(uint64_t(limits.floor_context)) + " or less, and that turns YaRN off. The fit does not "
                    "change a YaRN layout to a plain one. Start with --ctx " +
                    GroupDigits(uint64_t(limits.floor_context)) + " or less to run without YaRN, or free memory.";
  else
    layout.reason = "The memory fit stopped: no smaller layout fits (" + slots + ", --ctx " +
                    GroupDigits(uint64_t(kAdmissionContextGrain)) + " or more, cache budget " +
                    std::to_string(floor >> 20) + " MiB). Free memory, or use a smaller pack.";
  return layout;
}

inline FitLayout FitServeLayout(const AdmissionMemory& memory, const AdmissionTry& try_setting, const FitLimits& limits) {
  FitLayout layout;
  layout.streams = memory.streams;
  layout.context = memory.context;
  layout.cache_budget = memory.cache_budget;
  const uint64_t available = std::min(memory.host_free, memory.device_free);
  layout.free_bytes = available;
  layout.need_bytes = SaturatedSum(SaturatedSum(kAdmissionReserve, memory.new_states_bytes), memory.cache_budget);
  if (AdmissionFits(available, memory.new_states_bytes, memory.cache_budget)) return layout;
  return FitWithMargin(memory, try_setting, limits, layout);
}

// After the --ctx step the engine measures again (the fit used a formula for the memory that a smaller slot 0 frees).
// The budget follows the measure, up to the request, with kFitMargin kept free. nullopt when the measure cannot hold
// min(request, kFitCacheFloor): the engine then keeps the fitted budget, and the check in ConfigureServeStreams decides.
inline std::optional<uint64_t> FitCacheAgain(uint64_t available, uint64_t new_states, uint64_t requested_cache) {
  const uint64_t floor = std::min(requested_cache, kFitCacheFloor);
  const auto cache = LargestAdmissionCache(available > kFitMargin ? available - kFitMargin : 0, new_states,
                                           requested_cache);
  if (cache && *cache >= floor) return cache;
  return std::nullopt;
}

// "4 streams at 262,144 context with a 9,700 MiB conversation cache"
inline std::string FitLayoutText(int streams, int context, uint64_t cache_budget) {
  std::string text = std::to_string(streams) + (streams == 1 ? " stream at " : " streams at ") +
                     GroupDigits(uint64_t(context)) + " context";
  if (cache_budget) text += " with a " + GroupDigits(cache_budget >> 20) + " MiB conversation cache";
  return text;
}
// The first line of a refusal at the second check after a fit: the refusal text below it names the fitted layout, so
// this line names the fit and the request. `measure_short`: after the --ctx step the new measure could not hold the
// fitted layout (FitCacheAgain gave nullopt), so the cause can be the fit's estimate, not a drop of free memory.
inline std::string FitRefusalNote(const FitLayout& layout, int streams, int context, uint64_t cache_budget,
                                  bool measure_short = false) {
  return std::string("The memory fit ran (step ") + FitStepName(layout.step) + "): the request was " +
         FitLayoutText(streams, context, cache_budget) + ", and the fit chose " +
         FitLayoutText(layout.streams, layout.context, layout.cache_budget) +
         (measure_short ? ". The measure after the fit did not fit that layout.\n"
                        : ". The free memory fell before the slots were allocated.\n");
}

// "Largest --ctx at --streams S: c." with "none" when no value fits.
inline std::string AdmissionContextText(const AdmissionMemory& memory, int largest_context) {
  return "Largest --ctx at --streams " + std::to_string(memory.streams) + ": " +
         (largest_context ? GroupDigits(uint64_t(largest_context)) : std::string("none")) + ".\n";
}
// "Largest --streams at --ctx C: n. Largest --ctx at --streams S: c."
inline std::string AdmissionFitsText(const AdmissionMemory& memory, int largest_streams, int largest_context) {
  std::string text;
  if (memory.streams > 1)
    text += "Largest --streams at --ctx " + GroupDigits(uint64_t(memory.context)) + ": " +
            (largest_streams ? GroupDigits(uint64_t(largest_streams)) : std::string("none")) + ". ";
  return text + AdmissionContextText(memory, largest_context);
}

// Empty when the setting fits. Otherwise the measured numbers and the one action that helps in that case:
// - free memory at or below the reserve: no stream count fits; free memory, or a lower --ctx, which frees the first
//   slot that was allocated at the requested --ctx before the check;
// - the reserve fits but the new state does not: the largest --streams and --ctx that the check accepts;
// - the state fits but the conversation cache budget does not: the largest --conversation-cache-mib.
// The text never changes a setting.
inline std::string BatchAdmissionError(const AdmissionMemory& memory, const AdmissionTry& try_setting) {
  const uint64_t available = std::min(memory.host_free, memory.device_free);
  if (AdmissionFits(available, memory.new_states_bytes, memory.cache_budget)) return {};
  std::string text = "batch memory does not fit: " + std::to_string(memory.streams) +
      (memory.streams == 1 ? " stream at " : " streams at ") + GroupDigits(uint64_t(memory.context)) + " context " +
      (memory.streams == 1 ? "needs " : "need ") + BytesText(memory.states_bytes) + " of state. ";
  text += memory.allocated_bytes == 0 ? "None of that state is allocated yet."
          : memory.allocated_bytes >= memory.states_bytes ? "All of that state is already allocated."
          : BytesText(memory.allocated_bytes) + " of it is already allocated.";
  text += "\n  free host memory: " + BytesText(memory.host_free);
  text += "\n  free device memory: " + BytesText(memory.device_free);
  text += "\n  start-time reserve: " + BytesText(kAdmissionReserve) + ": system " + GibText(kAdmissionOsReserve) +
          ", plans " + MibText(kAdmissionPlanReserve) + ", host " + MibText(kAdmissionHostReserve);
  text += "\n  conversation cache budget: " + BytesText(memory.cache_budget) + "\n";

  if (available <= kAdmissionReserve) {
    // The check needs free memory above the reserve, so no stream count fits. A lower --ctx does fit when its slot 0
    // frees enough. The check cannot tell which program holds the memory: the engine's own first slot is one of them.
    const uint64_t needed = SaturatedSum(SaturatedSum(kAdmissionReserve, memory.new_states_bytes), memory.cache_budget);
    const bool exclusive = memory.new_states_bytes == 0 && memory.cache_budget == 0;
    const int context = LargestAdmissionContext(memory, try_setting);
    text += "Free memory is at or below the reserve before any new state, so no stream count fits.\n";
    text += "The check needs " + std::string(exclusive ? "more than " : "at least ") + BytesText(needed) + " free now";
    if (needed > available && needed - available > (1ull << 20))
      text += ", " + GibText(needed - available) + " more than now";
    text += ".\n";
    text += context ? "Stop other memory-heavy programs, if any, or start with a lower --ctx.\n"
                    : "Stop other memory-heavy programs, then start again.\n";
    text += AdmissionContextText(memory, context);
    text += "These numbers are text only: Crucible did not change your settings.";
    return text;
  }
  const uint64_t room = available - kAdmissionReserve;
  const uint64_t states_room = memory.cache_budget < room ? room - memory.cache_budget : 0;
  if (memory.new_states_bytes > room) {
    // The reserve fits but the state alone does not. Name the settings that the check accepts, tried one by one.
    const int streams = LargestAdmissionStreams(memory, try_setting);
    const int context = LargestAdmissionContext(memory, try_setting);
    text += "  free for new state after the reserve and the cache: " + BytesText(states_room) + "\n";
    text += "The reserve fits, but the new state does not.\n";
    text += AdmissionFitsText(memory, streams, context);
    if (memory.cache_budget) text += "The fits above include the conversation cache budget.\n";
    text += "These numbers are text only: Crucible did not change your settings.";
    return text;
  }
  if (memory.cache_budget > room - memory.new_states_bytes) {
    const uint64_t cache_room = room - memory.new_states_bytes;
    text += "  free after the reserve and the state: " + BytesText(cache_room) + "\n";
    text += "The state fits, but the conversation cache budget does not. Reduce --conversation-cache-mib to " +
            std::to_string(cache_room / (1ull << 20)) + " or less.\n";
    text += "These numbers are text only: Crucible did not change your settings.";
    return text;
  }
  return {};
}
}  // namespace crucible
