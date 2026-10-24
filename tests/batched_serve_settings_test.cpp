#include "../engine/serve/batched_serve_settings.h"
#include <cassert>
#include <vector>
int main() {
  using namespace crucible;
  const auto parse=[](std::initializer_list<std::string_view> args) {
    const std::vector<std::string_view> values(args); return ParseBatchedServeSettings(values);
  };
  assert(parse({"--streams","2"}).prefill_rows==kServePrefillAuto);
  assert(parse({"--streams","8","--prefill","4096"}).prefill_rows==4096);
  assert(parse({"--streams","1","--prefill","1024"}).prefill_rows==1024);
  assert(parse({"--streams","1","--prefill","0"}).prefill_rows==0);
  assert(parse({"--streams","1","--prefill","-1"}).prefill_rows==kServePrefillAuto);
  assert(parse({"--streams","2"}).cache_mib==0);
  assert(parse({"--streams","2","--conversation-cache-mib","4096"}).cache_mib==4096);
  const auto classed=parse({"--streams","8","--ctx","131072","--short-streams","5","--short-ctx","32768"});
  assert(classed.short_streams==5 && classed.short_ctx==32768);
  assert(parse({"--streams","8"}).short_streams==0 && parse({"--streams","8"}).short_ctx==0);
  assert(!parse({"--streams","8"}).session_replay);
  assert(parse({"--streams","8","--conversation-cache-mib","64","--session-replay","1"}).session_replay);
  // The cache limit is the host's memory; startup admission checks the real fit (after ple-offload).
  assert(parse({"--streams","4","--conversation-cache-mib","28672"}).cache_mib==28672);
  assert(parse({"--streams","4","--conversation-cache-mib","131072"}).cache_mib==131072);
  assert(parse({"--streams","8"}).slot_resume && !parse({"--streams","8","--slot-resume","0"}).slot_resume);
  assert(parse({"--streams","8"}).slot_checkpoints && !parse({"--streams","8","--slot-checkpoints","0"}).slot_checkpoints);
  // The free host memory that each cache save keeps (BatchConversationCache::Admit): 8 GiB unless set.
  assert(parse({"--streams","4"}).host_reserve_mib==8192);
  assert(parse({"--streams","4","--conversation-cache-mib","64","--host-reserve-mib","4096"}).host_reserve_mib==4096);
  for (const auto value:{"1023","131073","-1"}) {
    bool rejected=false;
    try {parse({"--streams","4","--conversation-cache-mib","64","--host-reserve-mib",value});}catch(const std::exception&){rejected=true;}
    assert(rejected);
  }
  // serve --draft-blocks FILE replaces the embedded draft head; the path is absolute, empty means the default.
  assert(parse({"--streams","4"}).draft_blocks.empty());
  assert(parse({"--streams","4","--draft-blocks","/x/heads/a.txt"}).draft_blocks=="/x/heads/a.txt");
  for (const char* bad : {"","heads/a.txt","/x/a\nb"}) {
    bool threw=false;
    try { parse({"--streams","4","--draft-blocks",bad}); } catch (const std::exception&) { threw=true; }
    assert(threw);
  }
  // the server's default MTP window sizes the slot MTP caches (draft_warmup.h MtpSlotRows); 256 unless set.
  assert(parse({"--streams","4"}).mtp_window==256);
  assert(parse({"--streams","4","--mtp-window","0"}).mtp_window==0);
  assert(parse({"--streams","4","--mtp-window","262144"}).mtp_window==262144);
  for (const char* bad : {"-1","262145","x"}) {
    bool threw=false;
    try { parse({"--streams","4","--mtp-window",bad}); } catch (const std::exception&) { threw=true; }
    assert(threw);
  }
  // Cache probation (batch_conversation_cache.h): on unless turned off.
  assert(parse({"--streams","4"}).cache_probation && !parse({"--streams","4","--cache-probation","0"}).cache_probation);
  assert(parse({"--streams","4"}).cache_idle_minutes==30 &&
         parse({"--streams","4","--cache-idle-minutes","5"}).cache_idle_minutes==5);
  for (const char* bad : {"0","10081","x"}) {
    bool threw=false;
    try { parse({"--streams","4","--cache-idle-minutes",bad}); } catch (const std::exception&) { threw=true; }
    assert(threw);
  }
  // The disk tier (state_disk.h): off unless --state-dir names an absolute directory; it needs session replay.
  assert(parse({"--streams","4"}).state_dir.empty());
  const auto disk=parse({"--streams","4","--conversation-cache-mib","64","--session-replay","1",
                         "--state-dir","/home/u/.cache/crucible/states","--state-disk-gib","200"});
  assert(disk.state_dir=="/home/u/.cache/crucible/states" && disk.state_disk_gib==200 && disk.state_write_gib_per_day==300);
  assert(parse({"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x",
                "--state-disk-write-gib-per-day","0"}).state_write_gib_per_day==0);
  assert(parse({"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x"}).state_disk_gib==64);
  // Fix 2 of an earlier change: the same-second GPU test delays each state write (default 0: no wait).
  assert(parse({"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x"}).state_write_delay_ms==0);
  assert(parse({"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x",
                "--state-disk-write-delay-ms","3000"}).state_write_delay_ms==3000);
  for (const auto args:std::vector<std::vector<std::string_view>>{
      {"--streams","4","--state-dir","/x"},
      {"--streams","4","--conversation-cache-mib","64","--state-dir","/x"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","x"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir",""},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x\n"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x","--state-disk-gib","0"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x","--state-disk-gib","65537"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-disk-gib","10"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-disk-write-gib-per-day","10"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-disk-write-delay-ms","10"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x",
       "--state-disk-write-delay-ms","60001"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x",
       "--state-disk-write-delay-ms","-1"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x",
       "--state-disk-write-gib-per-day","-1"},
      {"--streams","4","--conversation-cache-mib","64","--session-replay","1","--state-dir","/x",
       "--state-disk-write-gib-per-day","65537"}}) {
    bool rejected=false;try {ParseBatchedServeSettings(args);}catch(const std::exception&){rejected=true;}assert(rejected);
  }
  for (const auto value:{"-2","4097","1_0","1x"}) {
    bool rejected=false;try {parse({"--streams","1","--prefill",value});}catch(const std::exception&){rejected=true;}
    assert(rejected);
  }
  for (const auto args:std::vector<std::vector<std::string_view>>{{"--streams","0"},{"--streams","9"},{"--streams","1","--prefill"},{"--streams","1","--prefill","0","--prefill","1024"},{"--streams","1","--conversation-cache-mib","-1"},{"--streams","1","--conversation-cache-mib","131073"},{"--streams","1","--conversation-cache-mib","0","--conversation-cache-mib","0"},
      {"--streams","2","--short-streams","2","--short-ctx","64"},{"--streams","2","--short-streams","1"},
      {"--streams","2","--short-streams","1","--short-ctx","32768"},{"--streams","2","--short-ctx","64"},
      {"--streams","2","--short-streams","-1","--short-ctx","64"},{"--streams","2","--short-streams","1","--short-ctx","0"},
      {"--streams","2","--session-replay","1"},{"--streams","2","--conversation-cache-mib","64","--session-replay","2"}}) {
    bool rejected=false;try {ParseBatchedServeSettings(args);}catch(const std::exception&){rejected=true;}assert(rejected);
  }
}
