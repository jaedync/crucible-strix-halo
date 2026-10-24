// Keep startup validation independent from device allocation.
#pragma once
#include "serve_settings.h"
#include <algorithm>
#include <string>
namespace crucible {
inline constexpr int kServeBatchSlots = 8;
struct BatchedServeSettings {
  int ctx=32768, streams=0, prefill_rows=kServePrefillAuto, cache_mib=0;
  int host_reserve_mib=kServeHostReserveMib;  // MemAvailable that each cache save keeps free
  // Slot classes (G4): the last short_streams slots hold short_ctx tokens.
  int short_streams=0, short_ctx=0;
  bool session_replay=false;  // G15 opt-in A/B; needs the conversation cache
  // With session replay: a turn that repeats the tail still on its slot resumes there, with no host restore.
  // 0 keeps the host restore (the A/B gate: outputs must be equal).
  bool slot_resume=true;
  // With session replay: in-slot checkpoints at a turn's start, last chunk boundary and prompt end
  // (slot_checkpoints.h). A turn that leaves the previous output resumes there. 0 turns them off (the A/B gate).
  bool slot_checkpoints=true;
  bool vision=false;          // load the vision tower and accept request images
  // A cache state that no later request used leaves first and is never a protected resume point (speed only).
  bool cache_probation=true;
  // A session state that no request used for this many minutes may leave the host cache (to disk with a state dir).
  int cache_idle_minutes=30;
  // The disk tier (state_disk.h): a directory inside $HOME for spilled states, empty for off, and its size limit.
  std::string state_dir;
  int state_disk_gib=64;
  bool state_disk_gib_set=false;
  // SSD wear guard: GiB that the tier may write in one local day (0: it only reads).
  int state_write_gib_per_day=300;
  bool state_write_set=false;
  int state_write_delay_ms=0;  // the same-second GPU test only (an earlier change): each state write waits this long
  bool state_write_delay_set=false;
  // The server's default MTP window (serve --mtp-window). It sizes each slot's MTP cache (draft_warmup.h MtpSlotRows,
  // an earlier change); 0 (the whole prompt) keeps the context. Requests still choose their own window.
  int mtp_window=kDefaultMtpPrefix;
  // serve --draft-blocks FILE: the block IDs of the served draft head; empty keeps the embedded 64K head.
  // The head changes only drafts, never an output.
  std::string draft_blocks;
};
inline BatchedServeSettings ParseBatchedServeSettings(std::span<const std::string_view> args) {
  if (args.size()%2) throw std::invalid_argument("batch options need a value after each switch");
  BatchedServeSettings result;
  std::set<std::string_view> seen;
  for (size_t index=0;index<args.size();index+=2) {
    const auto key=args[index],value=args[index+1];
    if (!seen.insert(key).second) throw std::invalid_argument("duplicate batch option");
    if (key=="--ctx") result.ctx=ServeNumber<int>(value);
    else if (key=="--streams") result.streams=ServeNumber<int>(value);
    else if (key=="--prefill") result.prefill_rows=ServeNumber<int>(value);
    else if (key=="--conversation-cache-mib") result.cache_mib=ServeNumber<int>(value);
    else if (key=="--host-reserve-mib") result.host_reserve_mib=ServeNumber<int>(value);
    else if (key=="--short-streams") result.short_streams=ServeNumber<int>(value);
    else if (key=="--short-ctx") result.short_ctx=ServeNumber<int>(value);
    else if (key=="--session-replay") {
      const int parsed=ServeNumber<int>(value);
      if (parsed<0 || parsed>1) throw std::invalid_argument("session-replay must be 0 or 1");
      result.session_replay=parsed!=0;
    }
    else if (key=="--slot-resume") {
      const int parsed=ServeNumber<int>(value);
      if (parsed<0 || parsed>1) throw std::invalid_argument("slot-resume must be 0 or 1");
      result.slot_resume=parsed!=0;
    }
    else if (key=="--slot-checkpoints") {
      const int parsed=ServeNumber<int>(value);
      if (parsed<0 || parsed>1) throw std::invalid_argument("slot-checkpoints must be 0 or 1");
      result.slot_checkpoints=parsed!=0;
    }
    else if (key=="--cache-probation") {
      const int parsed=ServeNumber<int>(value);
      if (parsed<0 || parsed>1) throw std::invalid_argument("cache-probation must be 0 or 1");
      result.cache_probation=parsed!=0;
    }
    else if (key=="--cache-idle-minutes") {
      result.cache_idle_minutes=ServeNumber<int>(value);
      if (result.cache_idle_minutes<1 || result.cache_idle_minutes>10080)
        throw std::invalid_argument("cache-idle-minutes must be in [1,10080]");
    }
    else if (key=="--state-dir") {
      if (value.empty() || value.front()!='/' || value.size()>4096 ||
          std::any_of(value.begin(),value.end(),[](char c){return static_cast<unsigned char>(c)<0x20 || c==0x7f;}))
        throw std::invalid_argument("state-dir must be an absolute directory path");
      result.state_dir=std::string(value);
    }
    else if (key=="--state-disk-gib") {
      result.state_disk_gib=ServeNumber<int>(value);
      result.state_disk_gib_set=true;
      if (result.state_disk_gib<1 || result.state_disk_gib>65536) throw std::invalid_argument("state-disk-gib must be in [1,65536]");
    }
    else if (key=="--state-disk-write-gib-per-day") {
      result.state_write_gib_per_day=ServeNumber<int>(value);
      result.state_write_set=true;
      if (result.state_write_gib_per_day<0 || result.state_write_gib_per_day>65536)
        throw std::invalid_argument("state-disk-write-gib-per-day must be in [0,65536]");
    }
    else if (key=="--state-disk-write-delay-ms") {
      result.state_write_delay_ms=ServeNumber<int>(value);
      result.state_write_delay_set=true;
      if (result.state_write_delay_ms<0 || result.state_write_delay_ms>60000)
        throw std::invalid_argument("state-disk-write-delay-ms must be in [0,60000]");
    }
    else if (key=="--mtp-window") {
      result.mtp_window=ServeNumber<int>(value);
      if (result.mtp_window<0 || result.mtp_window>262144) throw std::invalid_argument("mtp-window must be in [0,262144]");
    }
    else if (key=="--draft-blocks") {
      if (value.empty() || value.front()!='/' || value.size()>4096 ||
          std::any_of(value.begin(),value.end(),[](char c){return static_cast<unsigned char>(c)<0x20 || c==0x7f;}))
        throw std::invalid_argument("draft-blocks must be an absolute file path");
      result.draft_blocks=std::string(value);
    }
    else if (key=="--vision") {
      const int parsed=ServeNumber<int>(value);
      if (parsed<0 || parsed>1) throw std::invalid_argument("vision must be 0 or 1");
      result.vision=parsed!=0;
    }
    else throw std::invalid_argument("unsupported batch option");
  }
  if (result.ctx<1 || result.ctx>262144 || result.streams<1 || result.streams>kServeBatchSlots)
    throw std::invalid_argument("invalid batch context or stream count");
  if (result.prefill_rows<kServePrefillAuto || result.prefill_rows>kServeMaxPrefillRows)
    throw std::invalid_argument("batch prefill rows exceed the supported range");
  if (result.cache_mib<0 || result.cache_mib>kServeMaxCacheMib)
    throw std::invalid_argument("batch cache budget must fit 0..131072 MiB");
  CheckHostReserve(result.host_reserve_mib);
  if (result.session_replay && !result.cache_mib)
    throw std::invalid_argument("session replay needs --conversation-cache-mib");
  if (!result.state_dir.empty() && !result.session_replay)
    throw std::invalid_argument("state-dir needs --session-replay 1 and a conversation cache");
  if ((result.state_disk_gib_set || result.state_write_set || result.state_write_delay_set) && result.state_dir.empty())
    throw std::invalid_argument("state-disk-gib, state-disk-write-gib-per-day and state-disk-write-delay-ms need --state-dir");
  // Slot 0 keeps the full context, so every request that fits --ctx has a slot.
  if (result.short_streams<0 || result.short_streams>=result.streams ||
      (result.short_streams ? result.short_ctx<1 || result.short_ctx>=result.ctx : result.short_ctx!=0))
    throw std::invalid_argument("short streams need 1..streams-1 slots and a short context below --ctx");
  return result;
}
} // namespace crucible
