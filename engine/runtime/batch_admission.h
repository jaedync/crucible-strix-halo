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

namespace crucible {
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

// The largest context on the admission grain below the request that the check accepts at the requested stream count,
// or 0 for none. The scan is exact; it does not assume that the charge grows with the context.
inline int LargestAdmissionContext(const AdmissionMemory& memory, const AdmissionTry& try_setting) {
  const uint64_t available = std::min(memory.host_free, memory.device_free);
  if (memory.streams < 1 || memory.context <= kAdmissionContextGrain) return 0;
  for (int context = (memory.context - 1) / kAdmissionContextGrain * kAdmissionContextGrain;
       context >= kAdmissionContextGrain; context -= kAdmissionContextGrain) {
    const AdmissionCharge charge = try_setting(memory.streams, context);
    if (!charge.allowed) continue;
    const uint64_t at = AdmissionAvailableAt(available, memory.first_slot_bytes, charge.first_slot_bytes);
    if (AdmissionFits(at, charge.new_states_bytes, memory.cache_budget)) return context;
  }
  return 0;
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
