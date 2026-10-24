// NVMe tier for conversation states.
// A state that the host cache evicts, or that a stopping engine holds, goes to a state file and block files (an earlier change,
// two states with equal block bytes share the block files). A later request, also after
// a restart of the same binary, reads it back on a reader thread and restores the same bytes that Save wrote.
// The device worker never waits for the disk: a spill only queues, and a read runs before the request opens.
// Files hold state derived from prompts: directory 0700, files 0600, and logs carry counts and sizes only.
#pragma once
#include "conversation_resume.h"
#include "retained_prefix.h"
#include "sha256.h"
#include "snapshot_bytes.h"
#include "state_format.h"
#include "../switches.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace crucible {
// The process identity that a state belongs to ("Key"). Any change of the engine binary,
// the checkpoint metadata or an engine setting gives a new directory, so a new binary never reads old states.
struct StateDiskIdentity { std::string sha, directory; };
inline std::string StateSha(const std::string& text) {
  return StateSha256(reinterpret_cast<const uint8_t*>(text.data()), text.size());
}
inline StateDiskIdentity StateDiskIdentityFor(const std::string& binary_sha, const std::string& checkpoint_sha,
                                              std::vector<std::string> settings) {
  std::sort(settings.begin(), settings.end());
  std::string joined;
  for (const auto& item : settings) joined += item + "\n";
  const std::string settings_sha = StateSha(joined);
  const std::string text = "binary=" + binary_sha + "\ncheckpoint=" + checkpoint_sha + "\nsettings=" + settings_sha + "\n";
  return {StateSha(text), binary_sha.substr(0, 16) + "-" + checkpoint_sha.substr(0, 16) + "-" + settings_sha.substr(0, 16)};
}
// "" when the file cannot be read.
inline std::string FileSha256(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::vector<char> data((std::istreambuf_iterator<char>(in)), {});
  return StateSha256(reinterpret_cast<const uint8_t*>(data.data()), data.size());
}
// The checkpoint identity of the gate records (tools/gates.py checkpoint_identity).
inline std::string CheckpointSha256(const std::filesystem::path& checkpoint) {
  std::string text;
  for (const char* name : {"config.json", "quantization_config.json", "model.safetensors.index.json"}) {
    const std::string sha = FileSha256(checkpoint / name);
    text += std::string(name) + ":" + (sha.empty() ? "-" : sha) + "\n";
  }
  // The weight files themselves, cheaply: name, size and modification time (a changed file gives a new identity).
  std::vector<std::string> weights;
  std::error_code error;
  for (const auto& item : std::filesystem::directory_iterator(checkpoint, error)) {
    const std::string name = item.path().filename();
    struct stat info{};
    if (!name.ends_with(".safetensors") || ::stat(item.path().c_str(), &info) != 0) continue;
    weights.push_back(name + ":" + std::to_string(info.st_size) + ":" + std::to_string(info.st_mtim.tv_sec) + "." +
                      std::to_string(info.st_mtim.tv_nsec));
  }
  std::sort(weights.begin(), weights.end());
  for (const auto& item : weights) text += item + "\n";
  return StateSha(text);
}
// Every CRUCIBLE_* switch of this process (numerics A/Bs and layouts change state bytes), the content of an
// ablation file and of the MTP sidecar, and the serve options that change state.
inline std::vector<std::string> StateDiskSettings(char** environ_list, int prefill_rows, bool vision) {
  std::vector<std::string> settings;
  for (char** item = environ_list; item && *item; ++item) {
    const std::string entry = *item;
    if (entry.rfind("CRUCIBLE_", 0) != 0) continue;
    settings.push_back(entry);
    if (entry.rfind("CRUCIBLE_ABLATE=", 0) == 0)
      settings.push_back("ablate_sha256=" + FileSha256(entry.substr(16)));
    // The sidecar's content, not only its path: a file that changes in place gives other MTP rows and drafts.
    if (entry.rfind("CRUCIBLE_MTP_SIDECAR=", 0) == 0)
      settings.push_back("mtp_sidecar_sha256=" + FileSha256(entry.substr(21)));
    // The BF16 n-gram table (102 GB) is too large to hash at each start: size and modification time, as the
    // checkpoint identity names its weight files, plus inode and ctime. touch -r can set the mtime of a
    // same-size replacement back; a rename then gives a new inode and an in-place rewrite a new ctime.
    if (struct stat info{}; entry.rfind("CRUCIBLE_PLE_BF16=", 0) == 0)
      settings.push_back("ple_bf16_file=" + (::stat(entry.substr(18).c_str(), &info) != 0 ? std::string("-") :
                         std::to_string(info.st_size) + ":" + std::to_string(info.st_mtim.tv_sec) + "." +
                         std::to_string(info.st_mtim.tv_nsec) + ":" + std::to_string(info.st_ino) + ":" +
                         std::to_string(info.st_ctim.tv_sec) + "." + std::to_string(info.st_ctim.tv_nsec)));
  }
  settings.push_back("prefill_rows=" + std::to_string(prefill_rows));
  settings.push_back(std::string("vision=") + (vision ? "1" : "0"));
  std::sort(settings.begin(), settings.end());
  return settings;
}

struct StateDiskOptions {
  std::filesystem::path root;  // parent of the identity directories; must be inside $HOME
  std::string identity;        // StateDiskIdentity::sha
  std::string directory;       // StateDiskIdentity::directory
  uint64_t budget = uint64_t(64) << 30;        // bytes of state files on disk
  size_t min_tokens = 16384;                   // smaller states are cheaper to prefill than to write and read
  uint64_t queue_bytes = uint64_t(8) << 30;    // evicted bytes that wait for the writer (held in host memory)
  uint64_t free_reserve = uint64_t(16) << 30;  // filesystem space that a spill always leaves free
  int io_threads = 8;                          // parallel requests of one read or write (the test machine: 8 reach 6.9 GB/s)
  double stage_seconds = 60;                   // a read state that no request uses within this time leaves memory
  std::function<uint64_t()> free_space;        // tests; default statvfs of the directory
  std::function<void(const std::string&)> log; // default stderr
  std::function<void()> before_write;          // tests: runs on the writer thread before each write
  // The proof of the same-second case (serve --state-disk-write-delay-ms): each write waits this long
  // first, so a returning turn finds its state in the writer queue. 0: no wait (the default).
  unsigned write_delay_ms = 0;
  std::function<void()> before_chunk;          // tests: runs on an I/O thread before each payload chunk
  std::function<void()> before_read;           // tests: runs on the reader thread before each read
  // tests: an errno for the payload pwrite or pread on `buffer` (0: none), as the kernel gives EFAULT for direct I/O
  // on pages that it cannot pin (hipHostMalloc memory on the test machine).
  std::function<int(const void* buffer)> io_fault;
  // SSD wear guard: bytes that the tier may write in one local calendar day, across restarts (the ledger file
  // `write-ledger` in `root`). Above it every spill is refused, stop flushes too. 0: the tier only reads.
  uint64_t write_per_day = uint64_t(300) << 30;
  std::function<std::string()> today;          // tests; default StateToday
  // Block storage. The rows of one block (0: every state is one file), and the
  // snapshot geometry that the plan needs: the fixed-size state before the QSA layers and the stacks after them
  // (BatchDiskTier sets the engine's sizes; another size gives `plan=whole`).
  size_t block_rows = kStateBlockRows;
  uint64_t block_head_bytes = 0, block_tail_bytes = 0;
  size_t max_states = 1024;    // state files on disk (each holds a key and logits in the index and on disk)
  size_t keep_superseded = 2;  // older states of one branch that a new state keeps (a rewind restores them)
  bool verify_digest = false;  // tests: the reader also checks each block's SHA-256
  // tests: changes the snapshot ranges of each new block's write after the hash (a wrong gather).
  std::function<void(std::vector<StreamRange>&)> write_ranges;
  std::function<void()> after_blocks;  // tests: runs on the writer thread after the new blocks, before the state file
};
enum class StageState { None, Reading, Ready, Failed };
struct StageStatus { StageState state = StageState::None; size_t tokens = 0; std::string reason; };
// Counts and bytes since the start of this process (refused_daily: spills that the daily write budget refused,
// present: spills of a state whose file is on disk already, so nothing was written).
struct StateDiskStats {
  uint64_t written = 0, refused = 0, reads = 0, failed = 0, evicted = 0;
  uint64_t written_bytes = 0, read_bytes = 0, present = 0, refused_daily = 0;
  uint64_t blocks_written = 0, blocks_shared = 0, pruned = 0;
};
// The local calendar day, as YYYY-MM-DD.
inline std::string StateToday() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  char text[16];
  std::strftime(text, sizeof text, "%F", &local);
  return text;
}
inline const char* StageName(StageState state) {
  switch (state) {
    case StageState::Reading: return "reading";
    case StageState::Ready: return "ready";
    case StageState::Failed: return "failed";
    default: return "none";
  }
}

// An aligned 8 MiB buffer for each payload chunk of a direct write or read (allocated on first use).
class StateBounce {
 public:
  uint8_t* Get() {
    if (!data_) data_.reset(static_cast<uint8_t*>(std::aligned_alloc(kStateAlign, SnapshotChunkPool::kChunkBytes)));
    if (!data_) throw std::bad_alloc();
    return data_.get();
  }
 private:
  struct Free { void operator()(uint8_t* p) const { std::free(p); } };
  std::unique_ptr<uint8_t, Free> data_;
};
inline int StateWriteAll(int fd, const uint8_t* p, size_t n, uint64_t offset) {
  while (n) {
    const ssize_t done = ::pwrite(fd, p, n, off_t(offset));
    if (done < 0 && errno == EINTR) continue;
    if (done <= 0) return done < 0 ? errno : EIO;
    p += done, n -= size_t(done), offset += uint64_t(done);
  }
  return 0;
}
inline int StateReadAll(int fd, uint8_t* p, size_t n, uint64_t offset) {
  while (n) {
    const ssize_t done = ::pread(fd, p, n, off_t(offset));
    if (done < 0 && errno == EINTR) continue;
    if (done <= 0) return done < 0 ? errno : EIO;
    p += done, n -= size_t(done), offset += uint64_t(done);
  }
  return 0;
}
// O_DIRECT where the filesystem supports it: page cache copies would double the memory traffic. Payload I/O uses only
// StateBounce buffers (malloc pages), never the pinned chunks.
inline int StateOpen(const std::filesystem::path& path, int flags, mode_t mode = 0600) {
  const int fd = ::open(path.c_str(), flags | O_DIRECT | O_CLOEXEC, mode);
  if (fd >= 0 || errno != EINVAL) return fd;
  return ::open(path.c_str(), flags | O_CLOEXEC, mode);
}
// The reason of a failed write step: "full", or "io errno=N" (the number only, for diagnosis).
inline std::string StateIoReason(int error) {
  return error == ENOSPC || error == EDQUOT ? "full" : "io errno=" + std::to_string(error);
}
// Runs io(index, bounce) for every index on up to `threads` threads; returns the first error ("" when all pass).
inline std::string StateParallel(size_t count, int threads, const std::function<std::string(size_t, StateBounce&)>& io) {
  std::atomic<size_t> next = 0;
  std::mutex mutex;
  std::string error;
  const auto run = [&] {
    StateBounce bounce;
    for (size_t index; (index = next++) < count;) {
      {
        std::lock_guard lock(mutex);
        if (!error.empty()) return;
      }
      std::string failed;
      try { failed = io(index, bounce); } catch (const std::bad_alloc&) { failed = "memory"; }
      if (failed.empty()) continue;
      std::lock_guard lock(mutex);
      if (error.empty()) error = failed;
      return;
    }
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < std::min<int>(threads, int(count)); ++t) pool.emplace_back(run);
  run();
  for (auto& thread : pool) thread.join();
  return error;
}

// The KV layout of a snapshot's K and V rows (0 when the type has no layout field).
template<class T> int SnapshotKvLayout(const T& t) {
  if constexpr (requires { t.kv_layout; }) return int(t.kv_layout);
  else return 0;
}
// The KV code and layout of this process, as the state identity holds them ("code.layout"). Real 8-bit KV storage
// stopped (h9 GUARD FAIL), so every process stores fp16 rows: 0.0. The KV owner returns the resolved pair here.
inline std::string StateKvCode() { return "0.0"; }

template<class Snapshot>
class StateDisk {
 public:
  // A staged state that a request can restore.
  struct ReadyState { std::shared_ptr<const Snapshot> snapshot; StateMeta meta; };

  static_assert(StateFieldCount<Snapshot>() == 5 + StateKvFields<Snapshot>(),
                "a snapshot member changed: store it in BuildStateHeader and Read (state_disk.h)");
  explicit StateDisk(StateDiskOptions options) : options_(std::move(options)) {
    if (!options_.log) options_.log = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
    if (!options_.free_space) options_.free_space = [this] { return FreeSpace(); };
    if (!options_.today) options_.today = StateToday;
    options_.io_threads = std::clamp(options_.io_threads, 1, 16);
    options_.max_states = std::max<size_t>(options_.max_states, 1);
    if (!IdentityDirectory(options_.directory) || options_.identity.empty())
      throw std::invalid_argument("state disk identity is malformed");
    PrepareRoot();
    dir_ = options_.root / options_.directory;
    MakePrivateDirectory(dir_);
    lock_ = ::open((dir_ / ".lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (lock_ < 0 || ::flock(lock_, LOCK_EX | LOCK_NB) != 0) {
      Log("state_disk off reason=locked");
      return;
    }
    const size_t removed = RemoveOtherIdentities();
    const double start = Now();
    const auto rebuilt = Rebuild();
    Unlink();
    enabled_ = true;
    day_ = options_.today();
    day_written_ = LedgerAdd(0);
    char ms[32];
    std::snprintf(ms, sizeof ms, "%.1f", (Now() - start) * 1e3);
    Log("state_disk start entries=" + std::to_string(index_.size()) + " blocks=" + std::to_string(blocks_.size()) +
        " bytes=" + std::to_string(bytes_) + " orphans=" + std::to_string(rebuilt.orphans) + " broken=" +
        std::to_string(rebuilt.broken) + " rebuild_ms=" + ms + " removed_dirs=" + std::to_string(removed) +
        " day_written=" + std::to_string(day_written_) + " write_per_day=" + std::to_string(options_.write_per_day) +
        (options_.write_delay_ms ? " write_delay_ms=" + std::to_string(options_.write_delay_ms) : std::string()));
    writer_ = std::thread([this] { WriterLoop(); });
    reader_ = std::thread([this] { ReaderLoop(); });
  }
  ~StateDisk() {
    stopping_ = true;  // a block write in progress ends at its next chunk; its temporary file leaves
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    work_.notify_all();
    if (writer_.joinable()) writer_.join();
    if (reader_.joinable()) reader_.join();
    if (lock_ >= 0) ::close(lock_);
  }
  StateDisk(const StateDisk&) = delete;
  StateDisk& operator=(const StateDisk&) = delete;

  bool Enabled() const { return enabled_; }
  const std::filesystem::path& Directory() const { return dir_; }
  size_t MinTokens() const { return options_.min_tokens; }

  // Queues a state for the writer. `keep` holds the snapshot memory until the write ends. Returns "" when queued,
  // else why not: off, short, present (already on disk), queued, memory (the held bytes would pass `room`, the host
  // memory above the reserve that the evicting save leaves, an earlier change), busy (the states that wait for the writer fill the
  // queue). `flush` (a stopping engine) ignores the room and the queue limit, because those states are in memory
  // already.
  std::string Spill(std::span<const int> key, const ConversationPolicy& policy, bool compact, bool shared, bool guarded,
                    std::shared_ptr<const void> keep, const Snapshot& snapshot, bool flush = false,
                    uint64_t room = UINT64_MAX, bool queue_limit = true) {
    if (!enabled_) return "off";
    if (key.size() < options_.min_tokens) return "short";
    if (snapshot.hist.size() != key.size()) return "policy";
    StateMeta meta;
    meta.key.assign(key.begin(), key.end());
    meta.policy = policy;
    meta.compact = compact, meta.shared = shared, meta.guarded = guarded;
    meta.resume = snapshot.resume;
    meta.resume.logits.clear();
    meta.payload = snapshot.bytes.size();
    meta.name = Name(policy, compact, key);
    BlockPlan plan = Plan(snapshot);
    const char* refused = nullptr;
    std::vector<Job> dropped;  // destroyed after the lock: their bytes leave host memory here
    size_t place = SIZE_MAX;   // where the job joins the queue: the place of the first prefix that left (fix 3), else the back
    {
      std::lock_guard lock(mutex_);
      RollDay();
      meta.file_bytes = QueueEstimate(meta, snapshot, plan);
      if (const auto found = index_.find(meta.name); found != index_.end()) {
        found->second.used = ++tick_;
        ++stats_.present;
        refused = "present";
      } else if (queued_names_.contains(meta.name)) {
        refused = "queued";
      } else if (day_written_ + queued_file_bytes_ + FileEstimate(meta) > options_.write_per_day) {
        ++stats_.refused_daily;
        refused = "daily";
      } else if (!flush && meta.payload > room) {
        refused = "memory";
      } else if (!flush && queue_limit && QueueFullLocked(meta.payload) && !DropSupersededLocked(meta, dropped, place)) {
        refused = "busy";
      } else {
        queued_file_bytes_ += FileEstimate(meta);
        queued_bytes_ += meta.payload;
        queued_names_.insert(meta.name);
        Job job{std::move(meta), std::move(keep), &snapshot, std::move(plan), Now()};
        jobs_.insert(jobs_.begin() + std::ptrdiff_t(std::min(place, jobs_.size())), std::move(job));
      }
    }
    for (const auto& job : dropped)
      Log("state_disk queue_drop tokens=" + std::to_string(job.meta.key.size()) + " bytes=" +
          std::to_string(job.meta.payload) + " reason=superseded by=" + std::to_string(key.size()));
    if (refused) {
      const std::string reason = refused;
      if (reason == "busy" || reason == "daily" || reason == "memory")
        Log("state_disk spill tokens=" + std::to_string(key.size()) + " bytes=" + std::to_string(snapshot.bytes.size()) +
            " result=refused reason=" + reason + " day_written=" + std::to_string(DayWritten()) +
            " write_per_day=" + std::to_string(options_.write_per_day));
      return refused;
    }
    work_.notify_all();
    return {};
  }
  // The longest state on disk whose key starts `key`, with the same policy (so the same tenant) and storage mode,
  // that `accept` takes.
  template<class Accept>
  std::optional<StateMeta> Best(std::span<const int> key, const ConversationPolicy& policy, bool compact,
                                Accept accept) const {
    std::lock_guard lock(mutex_);
    const StateMeta* best = nullptr;
    for (const auto& [name, meta] : index_)
      if (Matches(meta, key, policy, compact) && (!best || meta.key.size() > best->key.size()) && accept(meta))
        best = &meta;
    if (!best) return std::nullopt;
    return *best;
  }
  // An earlier change (fix 2): the longest state that waits for the writer or that the writer writes now, as Best would
  // find it once written, if it was queued at most `max_age` seconds ago. Best reads only the written index, so a turn
  // that arrives while its own state is queued (a direct spill of its slot) waits for this write and then reads it.
  template<class Accept>
  std::optional<StateMeta> Pending(std::span<const int> key, const ConversationPolicy& policy, bool compact,
                                   Accept accept, double max_age) const {
    std::lock_guard lock(mutex_);
    const double now = Now();
    const StateMeta* best = nullptr;
    const auto consider = [&](const StateMeta& meta, double queued_at) {
      if (now - queued_at <= max_age && Matches(meta, key, policy, compact) &&
          (!best || meta.key.size() > best->key.size()) && accept(meta))
        best = &meta;
    };
    for (const auto& job : jobs_) consider(job.meta, job.queued_at);
    if (writing_meta_) consider(*writing_meta_, writing_at_);
    if (!best) return std::nullopt;
    return *best;
  }
  // Starts or reports the read of `meta`'s files into host memory. `room`: bytes that staged states may hold in all.
  // The read pins the state's blocks, so an eviction cannot remove them while it runs.
  StageStatus Stage(const StateMeta& meta, uint64_t room) {
    StageStatus status;
    {
      std::lock_guard lock(mutex_);
      Expire();
      status.tokens = meta.key.size();
      if (const auto found = staged_.find(meta.name); found != staged_.end()) {
        status.state = found->second.state;
        status.reason = found->second.reason;
        if (status.state == StageState::Failed) Unstage(found);
        // A poll shows that the request still waits (for a slot): its read state stays in memory.
        if (status.state == StageState::Ready) found->second.since = std::chrono::steady_clock::now();
        return status;
      }
      const auto indexed = index_.find(meta.name);
      if (indexed == index_.end()) return {StageState::Failed, status.tokens, "missing"};
      for (const auto& block : indexed->second.blocks) {
        const auto found = blocks_.find(block);
        if (found == blocks_.end() || found->second.bad) return {StageState::Failed, status.tokens, "damaged"};
      }
      if (staged_bytes_ + meta.payload > room) return {StageState::None, status.tokens, "memory"};
      indexed->second.used = ++tick_;
      Pin(indexed->second.blocks);
      Staged staged;
      staged.meta = indexed->second;
      staged.pinned = true;
      staged_[meta.name] = std::move(staged);
      staged_bytes_ += meta.payload;
      reads_.push_back(meta.name);
      status.state = StageState::Reading;
    }
    work_.notify_all();
    return status;
  }
  // An earlier change: the longest staged read (reading, ready or failed) that `key` starts with, as Stage reports it, so a turn
  // never starts a second read beside one that it holds. A read that the floor beats is reported (\"floor\") and not
  // kept alive. nullopt: no staged read fits.
  template<class Accept>
  std::optional<StageStatus> PollStaged(std::span<const int> key, const ConversationPolicy& policy, bool compact,
                                        size_t floor, Accept accept) {
    std::lock_guard lock(mutex_);
    Expire();
    auto best = staged_.end();
    for (auto it = staged_.begin(); it != staged_.end(); ++it)
      if (Matches(it->second.meta, key, policy, compact) &&
          (best == staged_.end() || it->second.meta.key.size() > best->second.meta.key.size()) &&
          accept(it->second.meta))
        best = it;
    if (best == staged_.end()) return std::nullopt;
    StageStatus status{best->second.state, best->second.meta.key.size(), best->second.reason};
    if (status.state == StageState::Failed) Unstage(best);  // reported once, as Stage does
    else if (status.tokens <= floor) return StageStatus{StageState::None, status.tokens, "floor"};
    else if (status.state == StageState::Ready) best->second.since = std::chrono::steady_clock::now();
    return status;
  }
  // The longest read state that `key` starts with (as Best). It stays staged until Consume.
  template<class Accept>
  ReadyState Ready(std::span<const int> key, const ConversationPolicy& policy, bool compact, Accept accept) {
    std::lock_guard lock(mutex_);
    Expire();
    const Staged* best = nullptr;
    for (const auto& [name, staged] : staged_)
      if (staged.state == StageState::Ready && Matches(staged.meta, key, policy, compact) &&
          (!best || staged.meta.key.size() > best->meta.key.size()) && accept(staged.meta))
        best = &staged;
    if (!best) return {};
    return {best->snapshot, best->meta};
  }
  // The request restored `ready`: its host copy leaves (the files stay on disk).
  void Consume(const ReadyState& ready) {
    std::lock_guard lock(mutex_);
    if (const auto found = staged_.find(ready.meta.name); found != staged_.end() && found->second.state == StageState::Ready)
      Unstage(found);
  }
  // Waits until every queued spill is written or refused. False when `seconds` pass first.
  bool Drain(double seconds) {
    std::unique_lock lock(mutex_);
    return idle_.wait_for(lock, std::chrono::duration<double>(seconds), [this] { return jobs_.empty() && !writing_; });
  }
  uint64_t StagedBytes() const { std::lock_guard lock(mutex_); return staged_bytes_; }
  // Bytes of reads that Stage reserved and the reader has not allocated yet: MemAvailable does not show them, so a
  // spill's room leaves them out.
  uint64_t UnallocatedReadBytes() const {
    std::lock_guard lock(mutex_);
    uint64_t bytes = 0;
    for (const auto& [name, staged] : staged_)
      if (staged.state == StageState::Reading && !staged.allocated) bytes += staged.meta.payload;
    return bytes;
  }
  // Bytes that the writer queue holds in host memory, and its limit.
  uint64_t QueuedBytes() const { std::lock_guard lock(mutex_); return queued_bytes_; }
  uint64_t QueueLimit() const { return options_.queue_bytes; }
  size_t Entries() const { std::lock_guard lock(mutex_); return index_.size(); }
  size_t Blocks() const { std::lock_guard lock(mutex_); return blocks_.size(); }
  // Bytes on disk: every state file and every block file once.
  uint64_t Bytes() const { std::lock_guard lock(mutex_); return bytes_; }
  StateDiskStats Stats() const { std::lock_guard lock(mutex_); return stats_; }
  // Bytes that the tier wrote today (all engines that share the state root), as the ledger last showed them.
  uint64_t DayWritten() {
    std::lock_guard lock(mutex_);
    RollDay();
    return day_written_;
  }
  // A file with this full key, policy (with its tenant) and storage mode is on disk: a spill would write nothing.
  bool Present(std::span<const int> key, const ConversationPolicy& policy, bool compact) const {
    const std::string name = Name(policy, compact, key);
    std::lock_guard lock(mutex_);
    return index_.contains(name);
  }
  // A state with this full key, policy (with its tenant) and storage mode waits for the writer (or the writer writes
  // it now).
  bool Waiting(std::span<const int> key, const ConversationPolicy& policy, bool compact) const {
    const std::string name = Name(policy, compact, key);
    std::lock_guard lock(mutex_);
    return queued_names_.contains(name);
  }
  // A spill would pass the daily write budget (the caller can skip the device copy of a stop flush).
  bool OverDailyBudget(size_t payload) {
    std::lock_guard lock(mutex_);
    RollDay();
    return day_written_ + queued_file_bytes_ + StateRoundUp(payload) + kStateAlign > options_.write_per_day;
  }
  // A spill of `payload` bytes would be refused as `busy` (the caller can skip the device copy of a direct spill).
  bool QueueFull(uint64_t payload) const { std::lock_guard lock(mutex_); return QueueFullLocked(payload); }
  // As Spill decides it for a state of this key (fix 3): the waiting prefixes that the state would supersede give
  // their place first (DropSupersededLocked).
  bool QueueFull(uint64_t payload, std::span<const int> key, const ConversationPolicy& policy, bool compact) const {
    StateMeta meta;
    meta.key.assign(key.begin(), key.end());
    meta.policy = policy, meta.compact = compact;  // the policy holds the tenant
    std::lock_guard lock(mutex_);
    return QueueFullLocked(payload) && QueueFullLocked(payload, SupersededBytesLocked(meta));
  }

 private:
  // The limit counts the states that wait, not the one that the writer writes: one state waits beside it, so a
  // spill during a long write is not lost as `busy`. The host memory of both is guarded by the `memory` check (the
  // room that the evicting save leaves) and by the direct spill's own room check.
  // `leaving`: waiting bytes that leave the queue before this state joins it (DropSupersededLocked).
  bool QueueFullLocked(uint64_t payload, uint64_t leaving = 0) const {
    const uint64_t waiting = queued_bytes_ - writing_bytes_ - leaving;
    return waiting && waiting + payload > options_.queue_bytes;
  }
  struct Job {
    StateMeta meta;
    std::shared_ptr<const void> keep;
    const Snapshot* snapshot;
    BlockPlan plan;
    double queued_at = 0;  // Now() at the queue (Pending)
  };
  // Fix 3: the waiting jobs that `meta` supersedes. These are the older states of its
  // own branch (`meta` extends their keys, as the disk eviction rule says): a slot's grid frontier that went to the
  // queue just before the slot's tail. Never a shared state (other sessions restore it) and never the job that the
  // writer writes now (it is not in jobs_). Holds mutex_.
  template<class Visit>
  void ForSupersededLocked(const StateMeta& meta, Visit visit) const {
    for (size_t i = 0; i < jobs_.size(); ++i)
      if (!jobs_[i].meta.shared && Extends(meta, jobs_[i].meta)) visit(i);
  }
  uint64_t SupersededBytesLocked(const StateMeta& meta) const {
    uint64_t bytes = 0;
    ForSupersededLocked(meta, [&](size_t i) { bytes += jobs_[i].meta.payload; });
    return bytes;
  }
  // Holds mutex_. The queue is full for `meta`: when the waiting jobs that it supersedes make the room, they leave the
  // queue (into `dropped`, which the caller destroys after the lock) and the newer state takes their place. Else
  // nothing leaves, and the caller refuses `meta` ("busy"). `place` is the queue place of the first job that left: the
  // newer state joins there, ahead of the jobs that were behind its prefixes. Before fix 3 the newer state was refused: the tail of
  // session 2 in G (50,341 tokens) was lost, and its turn 2 restored only its 49,152-token frontier.
  bool DropSupersededLocked(const StateMeta& meta, std::vector<Job>& dropped, size_t& place) {
    const uint64_t leaving = SupersededBytesLocked(meta);
    if (!leaving || QueueFullLocked(meta.payload, leaving)) return false;
    std::vector<size_t> drop;
    ForSupersededLocked(meta, [&](size_t i) { drop.push_back(i); });
    place = drop.front();
    dropped.reserve(drop.size());  // no allocation between the first and the last change of the counters
    for (auto it = drop.rbegin(); it != drop.rend(); ++it) {
      Job& job = jobs_[*it];
      queued_bytes_ -= job.meta.payload;
      queued_file_bytes_ -= FileEstimate(job.meta);
      queued_names_.erase(job.meta.name);
      dropped.push_back(std::move(job));
      jobs_.erase(jobs_.begin() + std::ptrdiff_t(*it));
    }
    std::reverse(dropped.begin(), dropped.end());  // queue order in the log
    return true;
  }
  struct Staged {
    StateMeta meta;
    StageState state = StageState::Reading;
    std::shared_ptr<const Snapshot> snapshot;
    std::string reason;
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    bool allocated = false;  // the reader holds its memory (UnallocatedReadBytes)
    bool pinned = false;     // the read pins the state's blocks until it ends
  };
  // One block file in the index. `refs`: the listings in state files (a state that lists a block twice counts twice);
  // `pins`: the reads and the write in progress that use it. A block leaves when both are 0. `bad`: a read found it
  // damaged; it leaves when its last pin goes, and no spill or read uses it before.
  struct BlockMeta { uint64_t payload = 0, file_bytes = 0; uint32_t refs = 0, pins = 0; bool bad = false; };
  struct RebuildCounts { size_t orphans = 0, broken = 0; };
  // Three groups of 16 lower-case hex digits joined by '-' (StateDiskIdentity::directory).
  static bool IdentityDirectory(const std::string& name) {
    if (name.size() != 50) return false;
    for (size_t i = 0; i < name.size(); ++i) {
      const char c = name[i];
      if (i % 17 == 16 ? c != '-' : !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
  }
  static double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void Log(const std::string& line) const { options_.log(line); }
  BlockPlan Plan(const Snapshot& snapshot) const {
    SnapshotGeometry geometry;
    geometry.layout = snapshot.qsa_layout;
    geometry.kvl = SnapshotKvLayout(snapshot);
    geometry.rows = snapshot.hist.size();
    geometry.mpos = size_t(std::max(snapshot.resume.mpos, 0));
    geometry.head = options_.block_head_bytes;
    geometry.tail = options_.block_tail_bytes;
    geometry.total = snapshot.bytes.size();
    return PlanBlocks(geometry, options_.block_rows);
  }
  // The state file header before it exists: the fixed part and the fields of BuildStateHeader, with room to spare.
  uint64_t HeaderEstimate(const StateMeta& meta, const Snapshot& snapshot, const BlockPlan& plan) const {
    size_t ranges = plan.rest.size();
    for (const auto& block : plan.blocks) ranges += block.ranges.size();
    return StateRoundUp(kStateFixedHeader + 512 + options_.identity.size() + 4 * (meta.key.size() + snapshot.hist.size()) +
                        4 * (snapshot.resume.logits.size() + snapshot.resume.pending_tokens.size()) +
                        8 * snapshot.qsa_layout.size() + 56 * plan.blocks.size() + 16 * ranges +
                        8 * StateChunks(RangeBytes(plan.rest)));
  }
  uint64_t BlockFileEstimate(uint64_t bytes) const {
    return StateRoundUp(kStateFixedHeader + 128 + options_.identity.size() + 8 * StateChunks(bytes)) + StateRoundUp(bytes);
  }
  // Holds mutex_. The bytes that a spill will write, before the hash: the state file, and the blocks
  // above the rows of the longest state on disk that this one extends (those blocks hold the same bytes in most
  // cases). An estimate only: the writer charges what it writes.
  uint64_t QueueEstimate(const StateMeta& meta, const Snapshot& snapshot, const BlockPlan& plan) const {
    size_t covered = 0;
    for (const auto& [name, other] : index_)
      if (other.key.size() > covered && Extends(meta, other)) covered = other.key.size();
    uint64_t bytes = HeaderEstimate(meta, snapshot, plan) + StateRoundUp(RangeBytes(plan.rest));
    for (const auto& block : plan.blocks)
      if ((block.index + 1) * options_.block_rows > covered) bytes += BlockFileEstimate(RangeBytes(block.ranges));
    return bytes;
  }
  // A queued job's estimate (StateMeta::file_bytes holds it until the write).
  static uint64_t FileEstimate(const StateMeta& meta) { return meta.file_bytes; }
  // Holds mutex_. A new local day starts the day's totals at zero.
  void RollDay() {
    const std::string today = options_.today();
    if (today == day_) return;
    day_ = today;
    day_written_ = day_read_ = 0;
  }
  // Adds `bytes` to today's line of the ledger that all engines on this root share, and returns today's total. The
  // ledger holds one line "YYYY-MM-DD BYTES", mode 0600. An I/O error keeps this process's own count.
  uint64_t LedgerAdd(uint64_t bytes) {
    const std::string today = options_.today();
    const int fd = ::open((options_.root / "write-ledger").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return bytes;
    struct Close { int fd; ~Close() { ::close(fd); } } closer{fd};
    if (::flock(fd, LOCK_EX) != 0) return bytes;
    char text[64] = {};
    const ssize_t got = ::pread(fd, text, sizeof text - 1, 0);
    uint64_t total = 0;
    char day[16] = {};
    unsigned long long stored = 0;
    if (got > 0 && std::sscanf(text, "%15s %llu", day, &stored) == 2 && today == day) total = stored;
    total += bytes;
    const std::string line = today + " " + std::to_string(total) + "\n";
    if (::ftruncate(fd, 0) != 0 || ::pwrite(fd, line.data(), line.size(), 0) != ssize_t(line.size())) return total;
    return total;
  }
  // Charges `bytes` that this process wrote: the shared ledger at once (a crash later does not lose them,
  // a review L2) and today's count.
  void Charge(uint64_t bytes) {
    const uint64_t day = LedgerAdd(bytes);
    std::lock_guard lock(mutex_);
    RollDay();
    day_written_ = std::max(day, day_written_ + bytes);
    stats_.written_bytes += bytes;
  }
  // The policy holds the tenant: a state matches only inside its tenant.
  static bool Matches(const StateMeta& meta, std::span<const int> key, const ConversationPolicy& policy, bool compact) {
    return meta.compact == compact && meta.policy == policy && meta.key.size() <= key.size() &&
           std::equal(meta.key.begin(), meta.key.end(), key.begin());
  }
  // WritePolicy writes the tenant first, where train 23 wrote it: the names of the local tenant's states stay.
  std::string Name(const ConversationPolicy& policy, bool compact, std::span<const int> key) const {
    StateBytesOut out;
    out.Str(options_.identity);
    WritePolicy(out, policy);
    out.U8(compact);
    out.Ints(key);
    return StateSha256(out.Bytes().data(), out.Bytes().size()) + ".state";
  }
  std::string BlockName(const StateTenant& tenant, const StateDigest& digest, uint64_t bytes) const {
    return BlockFileNameFor(options_.identity, tenant, digest, bytes);
  }
  // The root must sit inside the user's home. Components that this call creates get mode 0700.
  void PrepareRoot() {
    const char* home = crucible::SwitchText("HOME");
    if (!home || !*home || !options_.root.is_absolute()) throw std::invalid_argument("state disk needs HOME and an absolute directory");
    const auto base = std::filesystem::weakly_canonical(home);
    const auto root = std::filesystem::weakly_canonical(options_.root);
    const auto relative = root.lexically_relative(base);
    if (relative.empty() || relative == "." || *relative.begin() == "..")
      throw std::invalid_argument("the state directory must be inside the home directory");
    std::filesystem::path path = base;
    bool created = false;
    for (const auto& part : relative) {
      path /= part;
      created = ::mkdir(path.c_str(), 0700) == 0;
      if (!created && errno != EEXIST) throw std::runtime_error("cannot create the state directory");
    }
    // The root holds the ledger and every identity directory: this user's own directory, no symlink, mode 0700. An
    // existing root must be the tier's own folder (empty, or only the tier's files), so a chmod never changes a
    // shared folder such as ~/.cache.
    if (!created) {
      std::error_code error;
      for (const auto& item : std::filesystem::directory_iterator(root, error)) {
        const std::string name = item.path().filename();
        if (name != "write-ledger" && !(IdentityDirectory(name) && item.is_directory(error)))
          throw std::invalid_argument("the state directory must be a folder of its own (it holds other files)");
      }
      if (error) throw std::runtime_error("cannot read the state directory");
    }
    MakePrivateDirectory(root);
    options_.root = root;
  }
  static void MakePrivateDirectory(const std::filesystem::path& path) {
    if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) throw std::runtime_error("cannot create the state directory");
    struct stat info{};
    if (::lstat(path.c_str(), &info) != 0 || !S_ISDIR(info.st_mode) || info.st_uid != ::geteuid())
      throw std::runtime_error("the state directory is not a private directory of this user");
    if ((info.st_mode & 0777) != 0700 && ::chmod(path.c_str(), 0700) != 0)
      throw std::runtime_error("cannot set the state directory mode");
  }
  // Every name that this tier writes. Train 22 matches the same names, so a rollback to it
  // removes this directory in full.
  static bool StateFileName(const std::string& name) {
    return name.size() > 6 && (name.ends_with(".state") || name.find(".state.tmp.") != std::string::npos ||
                               name.find(".state.dead.") != std::string::npos || BlockFileName(name));
  }
  static bool BlockFileName(const std::string& name) {
    return name.size() > 6 && (name.ends_with(".block") || name.find(".block.tmp.") != std::string::npos ||
                               name.find(".block.dead.") != std::string::npos);
  }
  // Directories of other identities (another binary, checkpoint or setting) leave at start, unless a live engine
  // holds their lock. Only state and block files, temporary files and the lock are removed; anything else stays.
  size_t RemoveOtherIdentities() {
    size_t removed = 0;
    std::error_code error;
    for (const auto& item : std::filesystem::directory_iterator(options_.root, error)) {
      const std::string name = item.path().filename();
      if (name == options_.directory || !item.is_directory(error) || item.is_symlink(error) ||
          !IdentityDirectory(name)) continue;
      const int fd = ::open((item.path() / ".lock").c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
      if (fd >= 0 && ::flock(fd, LOCK_EX | LOCK_NB) != 0) { ::close(fd); continue; }
      for (const auto& file : std::filesystem::directory_iterator(item.path(), error))
        if (StateFileName(file.path().filename()) && file.is_regular_file(error))
          std::filesystem::remove(file.path(), error);
      std::filesystem::remove(item.path() / ".lock", error);
      if (fd >= 0) ::close(fd);
      removed += std::filesystem::remove(item.path(), error) ? 1 : 0;
    }
    return removed;
  }

  using Aligned = std::unique_ptr<uint8_t, void (*)(void*)>;
  static Aligned AlignedBytes(uint64_t bytes) {
    Aligned p(static_cast<uint8_t*>(std::aligned_alloc(kStateAlign, StateRoundUp(bytes))), std::free);
    if (!p) throw std::bad_alloc();
    return p;
  }
  static int WriteAligned(int fd, const std::vector<uint8_t>& bytes) {
    auto aligned = AlignedBytes(bytes.size());
    std::memcpy(aligned.get(), bytes.data(), bytes.size());
    return StateWriteAll(fd, aligned.get(), bytes.size(), 0);
  }
  // The header of a state or block file: the whole header, or (`index_only`) the pages up to the end of the index
  // section. The first page holds the header size (offset 16) and the index end (offset 48).
  static std::pair<Aligned, uint64_t> ReadFront(int fd, bool index_only) {
    auto first = AlignedBytes(kStateAlign);
    if (StateReadAll(fd, first.get(), kStateAlign, 0)) throw std::runtime_error("cannot read a header");
    uint64_t header_bytes, index_used;
    std::memcpy(&header_bytes, first.get() + 16, 8);
    std::memcpy(&index_used, first.get() + 48, 8);
    if (header_bytes < kStateAlign || header_bytes > kStateMaxHeader || header_bytes % kStateAlign ||
        index_used > header_bytes)
      throw std::runtime_error("header bounds");
    const uint64_t need = index_only ? std::max<uint64_t>(kStateAlign, StateRoundUp(index_used)) : header_bytes;
    if (need == kStateAlign) return {std::move(first), need};
    auto all = AlignedBytes(need);
    if (StateReadAll(fd, all.get(), need, 0)) throw std::runtime_error("cannot read a header");
    return {std::move(all), need};
  }
  static uint64_t FileSize(int fd) {
    struct stat info{};
    return ::fstat(fd, &info) == 0 ? uint64_t(info.st_size) : UINT64_MAX;
  }
  struct Fd {
    int fd = -1;
    explicit Fd(int f) : fd(f) {}
    Fd(Fd&& other) noexcept : fd(std::exchange(other.fd, -1)) {}
    ~Fd() { if (fd >= 0) ::close(fd); }
  };
  // A block file's header: it must name this identity, its own tenant, digest and size, and fill the file.
  BlockHeader OpenBlock(const std::string& name, int fd) const {
    const auto [bytes, size] = ReadFront(fd, false);
    auto header = ParseBlockHeader(bytes.get(), size);
    if (header.identity != options_.identity || BlockName(header.tenant, header.digest, header.payload) != name ||
        FileSize(fd) != header.FileBytes())
      throw std::runtime_error("foreign block file");
    return header;
  }
  // The block names of a state file's list, in order.
  std::vector<std::string> BlockNames(const StateHeader& header) const {
    std::vector<std::string> names;
    for (const auto& block : header.blocks) names.push_back(BlockName(header.meta.policy.tenant, block.digest, block.bytes));
    return names;
  }
  // The index comes from the file headers: every block header, then the
  // index section of every state header. The references are the state files' lists: no count file can disagree.
  // A state that lists a missing or other block, a block that no state lists, a damaged or foreign file and a
  // temporary file leave.
  RebuildCounts Rebuild() {
    RebuildCounts counts;
    std::error_code error;
    std::vector<std::filesystem::path> states;
    for (const auto& item : std::filesystem::directory_iterator(dir_, error)) {
      const std::string name = item.path().filename();
      if (!StateFileName(name) || !item.is_regular_file(error)) continue;
      if (name.ends_with(".state")) { states.push_back(item.path()); continue; }
      if (name.ends_with(".block")) {
        try {
          const Fd fd(StateOpen(item.path(), O_RDONLY));
          if (fd.fd < 0) throw std::runtime_error("cannot open a block file");
          const auto header = OpenBlock(name, fd.fd);
          blocks_[name] = {header.payload, header.FileBytes()};
          continue;
        } catch (const std::exception&) {
          ++counts.broken;
        }
      }
      std::filesystem::remove(item.path(), error);
    }
    std::vector<std::pair<std::filesystem::file_time_type, std::string>> order;
    for (const auto& path : states) {
      const std::string name = path.filename();
      try {
        const Fd fd(StateOpen(path, O_RDONLY));
        if (fd.fd < 0) throw std::runtime_error("cannot open a state file");
        const auto [bytes, size] = ReadFront(fd.fd, true);
        const auto header = ParseStateHeader(bytes.get(), size, false);
        if (header.identity != options_.identity || header.meta.file_bytes != FileSize(fd.fd) ||
            Name(header.meta.policy, header.meta.compact, header.meta.key) != name)
          throw std::runtime_error("foreign state file");
        StateMeta meta = header.meta;
        meta.name = name;
        meta.blocks = BlockNames(header);
        for (size_t i = 0; i < meta.blocks.size(); ++i) {
          const auto found = blocks_.find(meta.blocks[i]);
          if (found == blocks_.end() || found->second.payload != header.blocks[i].bytes)
            throw std::runtime_error("missing block");
        }
        for (const auto& block : meta.blocks) ++blocks_[block].refs;
        bytes_ += meta.file_bytes;
        Insert(std::move(meta));
        order.emplace_back(std::filesystem::last_write_time(path, error), name);
      } catch (const std::exception&) {
        ++counts.broken;
        std::filesystem::remove(path, error);
      }
    }
    for (auto it = blocks_.begin(); it != blocks_.end();) {
      if (it->second.refs) { bytes_ += it->second.file_bytes; ++it; continue; }
      ++counts.orphans;
      std::filesystem::remove(dir_ / it->first, error);
      it = blocks_.erase(it);
    }
    std::sort(order.begin(), order.end());
    for (const auto& [time, name] : order) index_[name].used = ++tick_;
    MakeRoom(0, nullptr);
    return counts;
  }
  uint64_t FreeSpace() const {
    struct statvfs info{};
    if (::statvfs(dir_.c_str(), &info) != 0) return 0;
    return uint64_t(info.f_bavail) * info.f_frsize;
  }
  static bool Extends(const StateMeta& longer, const StateMeta& meta) {
    return longer.key.size() > meta.key.size() && longer.policy == meta.policy &&
           longer.compact == meta.compact && std::equal(meta.key.rbegin(), meta.key.rend(),
                                                        longer.key.rbegin() + (longer.key.size() - meta.key.size()));
  }
  // Holds mutex_. The bytes that eviction can free for `incoming`: the other state files, and the
  // blocks that no read or write pins and that have no bad mark (the incoming state's own blocks are pinned).
  uint64_t Evictable(const StateMeta& incoming) const {
    uint64_t bytes = 0;
    for (const auto& [name, meta] : index_) if (name != incoming.name) bytes += meta.file_bytes;
    for (const auto& [name, block] : blocks_) if (!block.pins && !block.bad) bytes += block.file_bytes;
    return bytes;
  }
  // The bytes that MakeRoom would free for a write of `file_bytes` (0 when the budget has room).
  uint64_t EvictableBytes(uint64_t file_bytes, const StateMeta& incoming) {
    std::lock_guard lock(mutex_);
    if (bytes_ + file_bytes <= options_.budget) return 0;
    return std::min(Evictable(incoming), bytes_ + file_bytes - options_.budget);
  }
  // Disk eviction: a state that a longer state of its own branch extends leaves first (it cannot be the best
  // match any more, unless it is a shared frontier), then the least recently used. A state's blocks leave with it
  // when no other state lists them and no read pins them. `incoming`: the state that needs the room (it counts as a
  // longer state, and as one more state file). The superseded marks are kept up to date on each insert and removal
  // (Insert, Remove), so a choice is one pass. Holds mutex_.
  bool MakeRoom(uint64_t need, const StateMeta* incoming) {
    const auto over = [&] {
      return bytes_ + need > options_.budget || index_.size() + (incoming ? 1 : 0) > options_.max_states;
    };
    while (over()) {
      const StateMeta* victim = nullptr;
      bool victim_superseded = false;
      for (const auto& [name, meta] : index_) {
        if (incoming && name == incoming->name) continue;
        const bool superseded = !meta.shared && (meta.superseded || (incoming && Extends(*incoming, meta)));
        if (!victim || superseded > victim_superseded || (superseded == victim_superseded && meta.used < victim->used))
          victim = &meta, victim_superseded = superseded;
      }
      if (!victim) return false;
      Log("state_disk evict tokens=" + std::to_string(victim->key.size()) + " bytes=" + std::to_string(victim->file_bytes) +
          " superseded=" + (victim_superseded ? "1" : "0"));
      Remove(victim->name);
      ++stats_.evicted;
    }
    return true;
  }
  // Holds mutex_. The unlink of a large file can take tens of milliseconds, so a rename (metadata only) takes the
  // file out of the index's name space now, and Unlink removes it later without the lock.
  void Doom(const std::string& name) {
    const auto dead = dir_ / (name + ".dead." + std::to_string(++dead_serial_));
    if (::rename((dir_ / name).c_str(), dead.c_str()) == 0) doomed_.push_back(dead);
  }
  // Holds mutex_. A block leaves when no state lists it and no read or write pins it.
  void DropIfUnused(const std::string& name) {
    const auto found = blocks_.find(name);
    if (found == blocks_.end() || found->second.refs || found->second.pins) return;
    Doom(name);
    bytes_ -= found->second.file_bytes;
    blocks_.erase(found);
  }
  void Pin(const std::vector<std::string>& names) {
    for (const auto& name : names) ++blocks_.at(name).pins;
  }
  void Unpin(const std::vector<std::string>& names) {
    for (const auto& name : names) {
      const auto found = blocks_.find(name);
      if (found == blocks_.end()) continue;
      if (found->second.pins) --found->second.pins;
      DropIfUnused(name);
    }
  }
  // Holds mutex_. Takes a state out of the index (its file stays): its references go, and so do the superseded marks
  // that only it gave.
  void Forget(typename std::map<std::string, StateMeta>::iterator found) {
    bytes_ -= found->second.file_bytes;
    const StateMeta gone = std::move(found->second);
    index_.erase(found);
    for (const auto& name : gone.blocks)
      if (const auto block = blocks_.find(name); block != blocks_.end()) {
        if (block->second.refs) --block->second.refs;
        DropIfUnused(name);
      }
    for (auto& [other_name, meta] : index_)
      if (meta.superseded && Extends(gone, meta))
        meta.superseded = std::any_of(index_.begin(), index_.end(), [&](const auto& o) { return Extends(o.second, meta); });
  }
  // Holds mutex_. The state file goes first, then its blocks at count 0 (section 6: a lost block removal leaves an
  // orphan; a lost state removal leaves a state that lists a missing block; the next start removes either).
  void Remove(const std::string& name) {
    const auto found = index_.find(name);
    if (found == index_.end()) return;
    Doom(name);
    Forget(found);
  }
  // Holds mutex_. A read found block `name` damaged: every state that lists it leaves (each read would fail), and the
  // block gets the bad mark until its last pin goes, so no spill or read uses it.
  void MarkBad(const std::string& name) {
    const auto found = blocks_.find(name);
    if (found == blocks_.end()) return;
    found->second.bad = true;
    std::vector<std::string> states;
    for (const auto& [state, meta] : index_)
      if (std::find(meta.blocks.begin(), meta.blocks.end(), name) != meta.blocks.end()) states.push_back(state);
    for (const auto& state : states) Remove(state);
    DropIfUnused(name);
  }
  // Holds mutex_. Adds `meta` to the index and updates the superseded marks: one pass over the index.
  void Insert(StateMeta meta) {
    meta.superseded = false;
    for (auto& [other_name, other] : index_) {
      if (other_name == meta.name) continue;
      if (Extends(meta, other)) other.superseded = true;
      else if (Extends(other, meta)) meta.superseded = true;
    }
    const std::string name = meta.name;
    index_[name] = std::move(meta);
  }
  // Holds mutex_. The older states of `meta`'s branch: the newest `keep_superseded` stay (a rewind
  // restores them); the others cost only their state files, because their blocks belong to `meta` too. A shared
  // state and a staged state (a read in progress or a ready copy) stay.
  size_t Prune(const StateMeta& meta) {
    std::vector<const StateMeta*> older;
    for (const auto& [name, other] : index_)
      if (name != meta.name && !other.shared && !staged_.contains(name) && Extends(meta, other)) older.push_back(&other);
    if (older.size() <= options_.keep_superseded) return 0;
    std::sort(older.begin(), older.end(), [](const StateMeta* a, const StateMeta* b) { return a->used > b->used; });
    std::vector<std::string> names;
    for (size_t i = options_.keep_superseded; i < older.size(); ++i) names.push_back(older[i]->name);
    for (const auto& name : names) Remove(name);
    stats_.pruned += names.size();
    return names.size();
  }
  void Unlink() {
    std::vector<std::filesystem::path> doomed;
    {
      std::lock_guard lock(mutex_);
      doomed.swap(doomed_);
    }
    std::error_code error;
    for (const auto& path : doomed) std::filesystem::remove(path, error);
  }
  void Expire() {
    const auto now = std::chrono::steady_clock::now();
    for (auto it = staged_.begin(); it != staged_.end();)
      if (it->second.state == StageState::Ready &&
          std::chrono::duration<double>(now - it->second.since).count() > options_.stage_seconds) {
        // A read that no turn used (the checker counts these).
        Log("state_disk expire tokens=" + std::to_string(it->second.meta.key.size()) +
            " bytes=" + std::to_string(it->second.meta.payload) + " result=unused");
        it = Unstage(it);
      } else ++it;
  }
  typename std::map<std::string, Staged>::iterator Unstage(typename std::map<std::string, Staged>::iterator it) {
    if (it->second.pinned) Unpin(it->second.meta.blocks);
    staged_bytes_ -= it->second.meta.payload;
    return staged_.erase(it);
  }

  void WriterLoop() {
    for (;;) {
      Job job;
      {
        std::unique_lock lock(mutex_);
        work_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
        if (stop_) {
          for (const auto& dropped : jobs_) queued_bytes_ -= dropped.meta.payload;
          jobs_.clear();
          queued_names_.clear();
          idle_.notify_all();
          return;
        }
        job = std::move(jobs_.front());
        jobs_.pop_front();
        writing_ = true;
        writing_bytes_ = job.meta.payload;
        writing_meta_ = job.meta;
        writing_at_ = job.queued_at;
      }
      if (options_.before_write) options_.before_write();
      if (options_.write_delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(options_.write_delay_ms));
      std::string result;
      try { result = Write(job); } catch (const std::bad_alloc&) { result = "result=refused reason=memory"; }
      Unlink();  // the blocks that a refused write unpinned
      std::string line = "state_disk spill tokens=" + std::to_string(job.meta.key.size()) + " bytes=" +
                         std::to_string(job.meta.payload) + " " + result;
      {
        std::lock_guard lock(mutex_);
        queued_bytes_ -= job.meta.payload;
        writing_bytes_ = 0;
        writing_meta_.reset();
        queued_file_bytes_ -= FileEstimate(job.meta);
        queued_names_.erase(job.meta.name);
        ++(result.starts_with("ms=") ? stats_.written : stats_.refused);
        line += " day_written=" + std::to_string(day_written_) + " total_written=" + std::to_string(stats_.written_bytes);
      }
      job.keep.reset();
      Log(line);
      {
        std::lock_guard lock(mutex_);
        writing_ = false;
      }
      idle_.notify_all();
    }
  }
  // The exact size of a block file of `bytes` (its header holds one checksum per chunk).
  uint64_t BlockFileBytes(const StateTenant& tenant, const StateDigest& digest, uint64_t bytes) const {
    BlockHeader header{options_.identity, tenant, digest, bytes, 0, std::vector<uint64_t>(StateChunks(bytes))};
    return BuildBlockHeader(header).size() + StateRoundUp(bytes);
  }
  // Writes one new block file (write path step 6). The SHA-256 of the bytes that go to the file, in order, must be the
  // digest of the hash step. The ledger counts the file at its rename.
  std::string WriteBlock(const Snapshot& snapshot, const StateTenant& tenant, const StateBlockEntry& entry,
                         const std::string& name, StateBounce& bounce, std::vector<std::filesystem::path>& written,
                         std::mutex& written_mutex) {
    constexpr size_t kChunk = SnapshotChunkPool::kChunkBytes;
    std::vector<StreamRange> ranges = entry.ranges;
    if (options_.write_ranges) options_.write_ranges(ranges);
    BlockHeader header{options_.identity, tenant, entry.digest, entry.bytes, 0,
                       std::vector<uint64_t>(StateChunks(entry.bytes))};
    const uint64_t header_bytes = header.header_bytes = BuildBlockHeader(header).size();
    const auto temp = dir_ / (name + ".tmp." + std::to_string(::getpid()));
    const int fd = StateOpen(temp, O_WRONLY | O_CREAT | O_EXCL);
    if (fd < 0) return StateIoReason(errno);
    Sha256 hash;
    std::string failed;
    for (size_t index = 0; index < header.sums.size() && failed.empty(); ++index) {
      if (options_.before_chunk) options_.before_chunk();
      if (stopping_) { failed = "stopped"; break; }
      const size_t length = std::min<uint64_t>(kChunk, entry.bytes - index * kChunk), padded = StateRoundUp(length);
      uint8_t* source = bounce.Get();
      GatherRanges(snapshot.bytes, ranges, index * kChunk, length, source);
      std::memset(source + length, 0, padded - length);
      hash.Update(source, length);
      header.sums[index] = StateChecksum(source, length);
      int error = options_.io_fault ? options_.io_fault(source) : 0;
      if (!error) error = StateWriteAll(fd, source, padded, header_bytes + index * kChunk);
      if (error) failed = StateIoReason(error);
    }
    if (failed.empty() && hash.Digest() != entry.digest) failed = "digest";
    if (failed.empty())
      if (const int error = WriteAligned(fd, BuildBlockHeader(header))) failed = StateIoReason(error);
    if (failed.empty() && ::fsync(fd) != 0) failed = StateIoReason(errno);
    ::close(fd);
    if (failed.empty() && ::rename(temp.c_str(), (dir_ / name).c_str()) != 0) failed = StateIoReason(errno);
    if (!failed.empty()) {
      std::error_code error;
      std::filesystem::remove(temp, error);
      return failed;
    }
    {
      std::lock_guard lock(written_mutex);
      written.push_back(dir_ / name);
    }
    Charge(header.FileBytes());
    return "";
  }
  // Writes the state file (write path step 7): the rest bytes, then the header with their checksums. With new blocks on
  // disk (`stoppable` false) it does not stop at the stop flag: else those blocks become orphans.
  std::string WriteStateFile(const Snapshot& snapshot, const StateMeta& meta, const std::vector<StateBlockEntry>& entries,
                             std::span<const StreamRange> rest, bool stoppable) {
    constexpr size_t kChunk = SnapshotChunkPool::kChunkBytes;
    const uint64_t rest_bytes = RangeBytes(rest);
    const size_t chunks = StateChunks(rest_bytes);
    std::vector<uint64_t> sums(chunks);
    const uint64_t header_bytes = BuildStateHeader(options_.identity, meta, snapshot, options_.block_rows, entries, rest,
                                                   sums).size();
    const auto temp = dir_ / (meta.name + ".tmp." + std::to_string(::getpid()));
    const int fd = StateOpen(temp, O_WRONLY | O_CREAT | O_EXCL);
    if (fd < 0) return StateIoReason(errno);
    std::string failed = StateParallel(chunks, options_.io_threads, [&](size_t index, StateBounce& bounce) -> std::string {
      if (options_.before_chunk) options_.before_chunk();
      if (stoppable && stopping_) return "stopped";
      const size_t length = std::min<uint64_t>(kChunk, rest_bytes - index * kChunk), padded = StateRoundUp(length);
      // Every chunk goes through the bounce buffer: direct I/O from the pinned chunk pages fails with EFAULT on the test machine
      // (Oct 9: every spill was refused). The padding is zero: the file never holds bytes of another state.
      uint8_t* source = bounce.Get();
      GatherRanges(snapshot.bytes, rest, index * kChunk, length, source);
      std::memset(source + length, 0, padded - length);
      sums[index] = StateChecksum(source, length);
      int error = options_.io_fault ? options_.io_fault(source) : 0;
      if (!error) error = StateWriteAll(fd, source, padded, header_bytes + index * kChunk);
      return error ? StateIoReason(error) : "";
    });
    if (failed.empty()) {
      const auto header = BuildStateHeader(options_.identity, meta, snapshot, options_.block_rows, entries, rest, sums);
      if (const int error = WriteAligned(fd, header)) failed = StateIoReason(error);
    }
    if (failed.empty() && ::fsync(fd) != 0) failed = StateIoReason(errno);
    ::close(fd);
    if (failed.empty() && ::rename(temp.c_str(), (dir_ / meta.name).c_str()) != 0) failed = StateIoReason(errno);
    if (!failed.empty()) {
      std::error_code error;
      std::filesystem::remove(temp, error);
    }
    return failed;
  }
  void SyncDirectory() const {
    if (const int dir = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); dir >= 0) { ::fsync(dir); ::close(dir); }
  }
  // Returns "ms=<T> result=written ..." or "result=refused reason=<why>".
  std::string Write(const Job& job) {
    const double start = Now();
    const Snapshot& snapshot = *job.snapshot;
    const BlockPlan& plan = job.plan;
    // Step 2: the digests, without mutex_, from host memory (a composed snapshot gives the digest of its flat copy).
    std::vector<StateDigest> digests(plan.blocks.size());
    StateParallel(plan.blocks.size(), options_.io_threads, [&](size_t index, StateBounce&) -> std::string {
      digests[index] = DigestRanges(snapshot.bytes, plan.blocks[index].ranges);
      return "";
    });
    const double hash_ms = (Now() - start) * 1e3;
    StateMeta meta = job.meta;
    std::vector<StateBlockEntry> entries;
    meta.blocks.clear();
    for (size_t i = 0; i < plan.blocks.size(); ++i) {
      entries.push_back({digests[i], RangeBytes(plan.blocks[i].ranges), plan.blocks[i].ranges});
      meta.blocks.push_back(BlockName(meta.policy.tenant, digests[i], entries.back().bytes));
    }
    meta.file_bytes = BuildStateHeader(options_.identity, meta, snapshot, options_.block_rows, entries, plan.rest,
                                       std::vector<uint64_t>(StateChunks(RangeBytes(plan.rest)))).size() +
                      StateRoundUp(RangeBytes(plan.rest));
    // Step 3: lookup and pin in one hold. Equal digests in one state count once.
    std::vector<size_t> fresh;      // entries of the new blocks, one for each name
    std::vector<std::string> held;  // blocks on disk that this write pinned
    struct Release {
      StateDisk& disk;
      std::vector<std::string>& held;
      ~Release() { std::lock_guard lock(disk.mutex_); disk.Unpin(held); }
    } release{*this, held};
    uint64_t write_bytes = meta.file_bytes;
    {
      std::lock_guard lock(mutex_);
      std::set<std::string> seen;
      for (size_t i = 0; i < entries.size(); ++i) {
        if (!seen.insert(meta.blocks[i]).second) continue;
        const auto found = blocks_.find(meta.blocks[i]);
        if (found == blocks_.end()) {
          fresh.push_back(i);
          write_bytes += BlockFileBytes(meta.policy.tenant, entries[i].digest, entries[i].bytes);
        } else if (found->second.bad) {
          return "result=refused reason=damaged";
        } else {
          ++found->second.pins;
          held.push_back(meta.blocks[i]);
        }
      }
    }
    // Step 4: the checks that can refuse run before eviction, so a refused write evicts nothing. Other engines on this
    // root write to the same ledger: read it, so their writes count against the budget.
    const uint64_t shared = LedgerAdd(0);
    {
      std::lock_guard lock(mutex_);
      RollDay();
      day_written_ = std::max(day_written_, shared);
      if (day_written_ + write_bytes > options_.write_per_day) {
        ++stats_.refused_daily;
        return "result=refused reason=daily";
      }
      if (write_bytes > options_.budget) return "result=refused reason=budget";
    }
    // The space of files that eviction would free counts toward the free space (files of earlier evictions leave first).
    Unlink();
    if (options_.free_space() + EvictableBytes(write_bytes, meta) < write_bytes + options_.free_reserve)
      return "result=refused reason=full";
    if (stopping_) return "result=refused reason=stopped";
    {
      // Step 5. Pinned and bad blocks cannot leave, so the room must be there without them.
      std::lock_guard lock(mutex_);
      if (bytes_ + write_bytes > options_.budget && bytes_ - Evictable(meta) + write_bytes > options_.budget)
        return "result=refused reason=budget";
      if (!MakeRoom(write_bytes, &meta)) return "result=refused reason=budget";
    }
    Unlink();
    // Step 6: the new blocks, one thread for each block. They are durable before the state file names them.
    std::vector<std::filesystem::path> written;
    std::mutex written_mutex;
    const auto discard = [&](const std::string& reason) {
      std::error_code error;
      for (const auto& path : written) std::filesystem::remove(path, error);
      return "result=refused reason=" + reason;
    };
    std::string failed = StateParallel(fresh.size(), options_.io_threads, [&](size_t index, StateBounce& bounce) {
      const size_t entry = fresh[index];
      return WriteBlock(snapshot, meta.policy.tenant, entries[entry], meta.blocks[entry], bounce, written, written_mutex);
    });
    if (!failed.empty()) return discard(failed);
    if (!fresh.empty()) SyncDirectory();
    if (options_.after_blocks) options_.after_blocks();
    // Step 7: the state file. Its rename is the commit point of the spill.
    failed = WriteStateFile(snapshot, meta, entries, plan.rest, fresh.empty());
    if (!failed.empty()) return discard(failed);
    SyncDirectory();
    // Step 8: a read can have found a listed block damaged meanwhile.
    size_t pruned = 0;
    bool damaged = false;
    {
      std::lock_guard lock(mutex_);
      for (const auto& name : held)
        if (const auto found = blocks_.find(name); found == blocks_.end() || found->second.bad) damaged = true;
      for (const size_t i : fresh) {
        if (damaged) break;
        auto& block = blocks_[meta.blocks[i]];
        block.payload = entries[i].bytes;
        block.file_bytes = BlockFileBytes(meta.policy.tenant, entries[i].digest, entries[i].bytes);
        bytes_ += block.file_bytes;
      }
      if (!damaged) {
        for (const auto& name : meta.blocks) ++blocks_.at(name).refs;
        stats_.blocks_written += fresh.size();
        stats_.blocks_shared += held.size();
        Unpin(held);
        held.clear();
        if (const auto old = index_.find(meta.name); old != index_.end()) Forget(old);
        meta.used = ++tick_;
        bytes_ += meta.file_bytes;
        Insert(meta);
        pruned = Prune(meta);
      }
    }
    if (damaged) {
      written.push_back(dir_ / meta.name);
      return discard("damaged");
    }
    Charge(meta.file_bytes);
    Unlink();
    char text[224];
    std::snprintf(text, sizeof text, "ms=%.1f result=written plan=%s blocks=%zu new_blocks=%zu shared=%zu written=%llu "
                  "hash_ms=%.1f pruned=%zu", (Now() - start) * 1e3, plan.whole ? "whole" : "blocks", entries.size(),
                  fresh.size(), entries.size() - fresh.size(),
                  static_cast<unsigned long long>(write_bytes), hash_ms, pruned);
    return text;
  }

  void ReaderLoop() {
    for (;;) {
      std::string name;
      StateMeta meta;
      int fd = -1, open_error = 0;
      {
        std::unique_lock lock(mutex_);
        work_.wait(lock, [this] { return stop_ || !reads_.empty(); });
        if (stop_) return;
        name = reads_.front();
        reads_.pop_front();
        const auto found = staged_.find(name);
        if (found == staged_.end()) continue;
        meta = found->second.meta;
        // The read holds the state file from here: an eviction during the read renames it, and the read goes on.
        fd = StateOpen(dir_ / name, O_RDONLY);
        if (fd < 0) open_error = errno;
      }
      const Fd file(fd);
      if (options_.before_read) options_.before_read();
      const double start = Now();
      std::string reason, bad;
      size_t files = 0;
      std::shared_ptr<Snapshot> snapshot;
      // A missing file is gone (the state leaves); any other open error (EMFILE, EIO) keeps the state.
      if (fd < 0) reason = open_error == ENOENT ? "header" : "io errno=" + std::to_string(open_error);
      else try { snapshot = Read(meta, fd, reason, bad, files); } catch (const std::bad_alloc&) { reason = "memory"; }
      char text[256];
      std::snprintf(text, sizeof text, "state_disk read tokens=%zu bytes=%llu files=%zu ms=%.1f result=%s%s%s",
                    meta.key.size(), static_cast<unsigned long long>(meta.payload), files, (Now() - start) * 1e3,
                    snapshot ? "ok" : "failed", snapshot ? "" : " reason=", snapshot ? "" : reason.c_str());
      {
        std::lock_guard lock(mutex_);
        ++(snapshot ? stats_.reads : stats_.failed);
        RollDay();
        if (snapshot) stats_.read_bytes += meta.payload, day_read_ += meta.payload;
        if (snapshot) std::snprintf(text + std::strlen(text), sizeof text - std::strlen(text), " day_read=%llu",
                                    static_cast<unsigned long long>(day_read_));
        const auto found = staged_.find(name);
        if (found != staged_.end() && found->second.pinned) {
          Unpin(found->second.meta.blocks);
          found->second.pinned = false;
        }
        // A damaged block leaves with every state that lists it; a damaged or foreign state file leaves alone; a read
        // that only lacked memory or met an I/O error keeps the files.
        if (!bad.empty()) MarkBad(bad);
        else if (!snapshot && reason != "memory" && !reason.starts_with("io")) Remove(name);
        if (found != staged_.end()) {
          found->second.state = snapshot ? StageState::Ready : StageState::Failed;
          found->second.snapshot = std::move(snapshot);
          found->second.reason = reason;
          found->second.since = std::chrono::steady_clock::now();
          if (!found->second.snapshot) { staged_bytes_ -= found->second.meta.payload; found->second.meta.payload = 0; }
        }
      }
      Unlink();
      Log(text);
    }
  }
  // One chunk of a read: from file `fd` at `offset`, checked against `sum`, to the stream bytes [from, from + length)
  // of `ranges`. `block`: the block file's name ("" for the state file).
  struct ReadJob {
    int fd;
    uint64_t offset, sum, from, length;
    std::span<const StreamRange> ranges;
    const std::string* block;
  };
  // Reads and checks one state and its blocks. The state header must name this identity and the same tenant, key,
  // policy and storage mode; its ranges must cover the snapshot once; each block must be the indexed block; every
  // chunk must match its checksum. `bad`: a block that failed its checks.
  std::shared_ptr<Snapshot> Read(const StateMeta& meta, int fd, std::string& reason, std::string& bad, size_t& files) {
    constexpr size_t kChunk = SnapshotChunkPool::kChunkBytes;
    StateHeader header;
    try {
      const auto [bytes, size] = ReadFront(fd, false);
      header = ParseStateHeader(bytes.get(), size);
    } catch (const std::bad_alloc&) { throw; } catch (const std::exception&) {
      reason = "header";
      return nullptr;
    }
    if (header.identity != options_.identity || header.meta.key != meta.key || !(header.meta.policy == meta.policy) ||
        header.meta.compact != meta.compact || header.meta.payload != meta.payload ||
        FileSize(fd) != header.meta.file_bytes || BlockNames(header) != meta.blocks) {
      reason = "header";
      return nullptr;
    }
    {
      std::lock_guard lock(mutex_);
      for (size_t i = 0; i < meta.blocks.size(); ++i)
        if (const auto found = blocks_.find(meta.blocks[i]); found == blocks_.end() || found->second.payload != header.blocks[i].bytes) {
          reason = "header";
          return nullptr;
        }
    }
    auto snapshot = std::make_shared<Snapshot>();
    snapshot->hist = std::move(header.hist);
    if (!RetainedPrefix::FromFields(header.retained, snapshot->retained) || !SetSnapshotKvFields(header.kv, *snapshot)) {
      reason = "header";
      return nullptr;
    }
    snapshot->resume = std::move(header.resume);
    snapshot->qsa_layout = std::move(header.layout);
    snapshot->bytes = SnapshotBytes(size_t(meta.payload));
    {
      std::lock_guard lock(mutex_);
      if (const auto found = staged_.find(meta.name); found != staged_.end()) found->second.allocated = true;
    }
    // The block files: each once, with its header checked.
    std::map<std::string, std::pair<Fd, BlockHeader>> blocks;
    for (const auto& name : meta.blocks) {
      if (blocks.contains(name)) continue;
      Fd block(StateOpen(dir_ / name, O_RDONLY));
      if (block.fd < 0 && errno != ENOENT) { reason = "io errno=" + std::to_string(errno); return nullptr; }
      try {
        if (block.fd < 0) throw std::runtime_error("missing block");
        auto block_header = OpenBlock(name, block.fd);
        blocks.emplace(name, std::make_pair(std::move(block), std::move(block_header)));
      } catch (const std::bad_alloc&) { throw; } catch (const std::exception&) {
        reason = "block";
        bad = name;
        return nullptr;
      }
    }
    files = 1 + blocks.size();
    std::vector<ReadJob> jobs;
    const auto add = [&](int file, uint64_t header_bytes, const std::vector<uint64_t>& sums, uint64_t bytes,
                         std::span<const StreamRange> ranges, const std::string* block) {
      for (size_t index = 0; index < sums.size(); ++index)
        jobs.push_back({file, header_bytes + index * kChunk, sums[index], index * kChunk,
                        std::min<uint64_t>(kChunk, bytes - index * kChunk), ranges, block});
    };
    add(fd, header.header_bytes, header.sums, RangeBytes(header.rest), header.rest, nullptr);
    for (size_t i = 0; i < meta.blocks.size(); ++i) {
      const auto& [block, block_header] = blocks.at(meta.blocks[i]);
      add(block.fd, block_header.header_bytes, block_header.sums, block_header.payload, header.blocks[i].ranges,
          &meta.blocks[i]);
    }
    std::mutex bad_mutex;
    reason = StateParallel(jobs.size(), options_.io_threads, [&](size_t index, StateBounce& bounce) -> std::string {
      const ReadJob& job = jobs[index];
      // Through the bounce buffer, as the write: direct I/O into the pinned chunk pages fails with EFAULT.
      uint8_t* into = bounce.Get();
      int error = options_.io_fault ? options_.io_fault(into) : 0;
      if (!error) error = StateReadAll(job.fd, into, StateRoundUp(job.length), job.offset);
      if (error) return "io errno=" + std::to_string(error);
      if (StateChecksum(into, job.length) != job.sum) {
        std::lock_guard lock(bad_mutex);
        if (job.block && bad.empty()) bad = *job.block;
        return "checksum";
      }
      ScatterRanges(snapshot->bytes, job.ranges, job.from, job.length, into);
      return "";
    });
    if (!reason.empty()) {
      if (reason != "checksum") bad.clear();
      return nullptr;
    }
    // Tests: the digest of each block's bytes (a write path that names a block by the wrong bytes fails here).
    if (options_.verify_digest)
      for (size_t i = 0; i < header.blocks.size(); ++i)
        if (DigestRanges(snapshot->bytes, header.blocks[i].ranges) != header.blocks[i].digest) {
          reason = "digest";
          bad = meta.blocks[i];
          return nullptr;
        }
    return snapshot;
  }

  StateDiskOptions options_;
  std::filesystem::path dir_;
  int lock_ = -1;
  bool enabled_ = false;
  mutable std::mutex mutex_;
  std::condition_variable work_, idle_;
  bool stop_ = false, writing_ = false;
  std::atomic<bool> stopping_ = false;
  std::map<std::string, StateMeta> index_;
  std::map<std::string, BlockMeta> blocks_;
  uint64_t bytes_ = 0, tick_ = 0, dead_serial_ = 0;
  std::vector<std::filesystem::path> doomed_;
  std::deque<Job> jobs_;
  std::set<std::string> queued_names_;
  uint64_t queued_bytes_ = 0, queued_file_bytes_ = 0;
  uint64_t writing_bytes_ = 0;  // the part of queued_bytes_ that the writer writes now
  std::optional<StateMeta> writing_meta_;  // the state that the writer writes now (Pending)
  double writing_at_ = 0;
  std::string day_;
  uint64_t day_written_ = 0, day_read_ = 0;
  std::map<std::string, Staged> staged_;
  std::deque<std::string> reads_;
  uint64_t staged_bytes_ = 0;
  StateDiskStats stats_;
  std::thread writer_, reader_;
};
}  // namespace crucible
