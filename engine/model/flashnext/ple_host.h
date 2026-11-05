// PLE n-gram rows gathered on the CPU from the mmapped checkpoint instead of a
// device-resident copy (CRUCIBLE_PLE_HOST=1). The table is 32.6 GB (3.05),
// 39.0 GB (4.05) or 102.4 GB (the BF16 rows of CRUCIBLE_PLE_BF16, 320 B each),
// and each token reads only 16 rows, so the page cache holds the rows in use
// and the rest stays on disk. The rows are plain copies of the file bytes,
// which are the same bytes the resident path uploads. Nothing here depends on
// the row size: a row can cross a page edge, and the hints cover both pages.
// The file must stay unchanged while the engine runs: a truncated file makes a
// later row read fail with SIGBUS, and a rewritten file changes the rows.
//
// CRUCIBLE_PLE_FETCH (SetFetch, default 0): the batched read of a new prompt's rows
// A plain gather hints a page only the first time it
// sees it; a page that was hinted once and later evicted is then read by a
// synchronous fault, one at a time (62 us on the test machine NVMe, 110 us on the test machine), which is
// the 45 ms that a new 30 token prompt waits. Mode 1 hints every page of the gather
// (a hint on a resident page costs 0.4 us) in blocks, so the reads of the next block
// are in flight while the rows of this block are copied. Mode 2 also splits a gather
// of 160 rows or more over the 16 gather parts, so the reads and the copies run in
// parallel even when a hint does not queue a read. The copied bytes are the same.
#pragma once
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace crucible {

struct PleSegment {
  int64_t first_row, rows;
  uint64_t file_offset;
};

// Persistent workers: thread creation would cost more than a hot gather.
class GatherPool {
 public:
  explicit GatherPool(int workers) {
    for (int w = 0; w < workers; ++w) {
      threads_.emplace_back([this, w] { Loop(w + 1); });
      pthread_setname_np(threads_.back().native_handle(), "ple-gather");
    }
  }
  ~GatherPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
  }
  GatherPool(const GatherPool&) = delete;
  GatherPool& operator=(const GatherPool&) = delete;
  int Parts() const { return int(threads_.size()) + 1; }

  // Runs job(part) for every part, part 0 on the caller, and returns when all end.
  void Run(const std::function<void(int)>& job) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      job_ = &job;
      pending_ = int(threads_.size());
      ++generation_;
    }
    wake_.notify_all();
    job(0);
    std::unique_lock<std::mutex> lock(mu_);
    done_.wait(lock, [&] { return pending_ == 0; });
    job_ = nullptr;
  }

 private:
  void Loop(int part) {
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(int)>* job;
      {
        std::unique_lock<std::mutex> lock(mu_);
        wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        job = job_;
      }
      (*job)(part);
      std::lock_guard<std::mutex> lock(mu_);
      if (--pending_ == 0) done_.notify_one();
    }
  }
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable wake_, done_;
  const std::function<void(int)>* job_ = nullptr;
  uint64_t generation_ = 0;
  int pending_ = 0;
  bool stop_ = false;
};

class PleHostTable {
 public:
  // Measured on the test host (bench/ple_gather_bench.cpp): 16 parts gather a cold
  // 4096-row chunk in about 20 ms and a page-cache-hot one in about 0.3 ms.
  static constexpr int kParts = 16;
  static constexpr size_t kPoolRows = 1024;
  static constexpr size_t kPage = 4096;
  // Fetch modes: a part hints kFetchAhead blocks of kFetchBlock rows before it copies the first, then one block per
  // copied block. At most kParts x kFetchAhead x kFetchBlock = 2,048 rows (up to 2 pages each, for a row on a page
  // edge) are in flight at any time. A plain gather has every page of its slice in flight. Mode 2 splits gathers of
  // kFetchParMin rows (10 tokens) or more; decode and verify gathers (up to 8 tokens) stay on one thread.
#ifdef PLE_FETCH_BLOCK_FOR_BENCH  // bench/ple_cold_probe.cpp sweeps the block size at build time
  static constexpr size_t kFetchBlock = PLE_FETCH_BLOCK_FOR_BENCH;
#else
  static constexpr size_t kFetchBlock = 64;
#endif
  static constexpr size_t kFetchAhead = 2;
  static constexpr size_t kFetchParMin = 160;

  PleHostTable(const std::string& path, size_t row_bytes, std::vector<PleSegment> segments)
      : row_(row_bytes), path_(path) {
    if (row_bytes == 0 || segments.empty()) throw std::invalid_argument("PLE host table geometry is empty");
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("cannot open PLE table " + path + ": " + std::strerror(errno));
    struct stat sb{};
    if (fstat(fd, &sb) != 0 || !S_ISREG(sb.st_mode)) {
      close(fd);
      throw std::runtime_error("PLE table is not a regular file: " + path);
    }
    size_ = size_t(sb.st_size);
    for (const auto& s : segments) {
      const bool next_row = segs_.empty() ? s.first_row == 0 : s.first_row == segs_.back().first_row + segs_.back().rows;
      if (!next_row || s.rows <= 0 || s.file_offset > size_ || uint64_t(s.rows) > (size_ - s.file_offset) / row_) {
        close(fd);
        throw std::invalid_argument("PLE host table segment lies outside " + path);
      }
      // The 3.05 shards are back to back in one file; one segment keeps lookup a multiply.
      if (!segs_.empty() && segs_.back().file_offset + uint64_t(segs_.back().rows) * row_ == s.file_offset)
        segs_.back().rows += s.rows;
      else
        segs_.push_back(s);
    }
    rows_ = segs_.back().first_row + segs_.back().rows;
    // Read-only private pages are the page-cache pages; nothing is copied.
    void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    const int error = errno;
    close(fd);
    if (p == MAP_FAILED) throw std::runtime_error("cannot map PLE table " + path + ": " + std::strerror(error));
    base_ = static_cast<const uint8_t*>(p);
    // Fault readahead (read_ahead_kb is 4096 on the test host) would read megabytes per row.
    madvise(p, size_, MADV_RANDOM);
    const size_t pages = (size_ + kPage - 1) / kPage;
    seen_words_ = (pages + 63) / 64;
    seen_ = std::make_unique<std::atomic<uint64_t>[]>(seen_words_);
  }
  ~PleHostTable() {
    {
      std::lock_guard<std::mutex> lock(front_mu_);
      front_stop_ = true;
    }
    front_wake_.notify_all();
    if (front_.joinable()) front_.join();
    {
      std::lock_guard<std::mutex> lock(hint_mu_);
      hint_stop_ = true;
    }
    hint_wake_.notify_all();
    if (hint_.joinable()) hint_.join();
    munmap(const_cast<uint8_t*>(base_), size_);
  }
  PleHostTable(const PleHostTable&) = delete;
  PleHostTable& operator=(const PleHostTable&) = delete;

  // Diagnostic sums over all parts since construction (nanoseconds, pages).
  // max_ahead_rows: the most rows that a fetch part hinted and had not yet copied. major_faults: major faults of the
  // gather threads while they copy (a plain gather counts the copy phase only).
  struct Stats {
    std::atomic<uint64_t> hint_ns{0}, copy_ns{0}, fresh_pages{0}, major_faults{0}, max_ahead_rows{0};
  };
  Stats& stats() { return stats_; }

  // 0: plain gather (default); 1: hint every page in blocks; 2: also split gathers of kFetchParMin rows or more.
  void SetFetch(int mode) {
    if (mode < 0 || mode > 2) throw std::invalid_argument("PLE fetch mode must be 0, 1 or 2");
    fetch_.store(mode, std::memory_order_relaxed);
  }
  int Fetch() const { return fetch_.load(std::memory_order_relaxed); }

  // Tests and benches only: unmaps the file's pages from this process and drops the clean page-cache pages of
  // this one file (POSIX_FADV_DONTNEED), so the next read is cold. The engine never calls it.
  void EvictForTest() {
    madvise(const_cast<uint8_t*>(base_), size_, MADV_DONTNEED);
    const int fd = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
  }

  int64_t Rows() const { return rows_; }
  size_t Segments() const { return segs_.size(); }
  size_t RowBytes() const { return row_; }

  // Starts Gather on a background thread, so the reads overlap the device work
  // queued before the rows are needed. rows and out must stay valid until Wait.
  void Start(const int64_t* rows, size_t n, uint8_t* out) {
    std::lock_guard<std::mutex> lock(front_mu_);
    if (front_job_) throw std::logic_error("PLE gather started twice");
    StartFront();
    front_job_ = true;
    front_done_ = false;
    front_error_ = nullptr;
    front_args_ = {rows, n, out};
    front_wake_.notify_all();
  }

  // Queues read hints for rows that a later step needs (the next prefill
  // chunks), so their disk reads finish before that step. One thread at nice
  // 19 sends them, so the hints never delay a gather or the thread that
  // queues device work. A new call replaces the hints not yet sent. Invalid
  // rows are skipped: a hint changes no result.
  void Hint(std::vector<int64_t> rows) {
    std::lock_guard<std::mutex> lock(hint_mu_);
    if (!hint_.joinable()) {
      hint_ = std::thread([this] { HintLoop(); });
      pthread_setname_np(hint_.native_handle(), "ple-hint");
    }
    hints_ = std::move(rows);
    hint_pos_ = 0;
    hint_wake_.notify_all();
  }

  // Waits for the started gather and rethrows its error. No-op without one.
  void Wait() {
    if (auto error = Drain()) std::rethrow_exception(error);
  }

  // Wait for destructors: returns the gather's error instead of throwing it.
  // After it returns, the gather no longer touches its rows or destination.
  std::exception_ptr Drain() noexcept {
    std::unique_lock<std::mutex> lock(front_mu_);
    if (!front_job_) return nullptr;
    front_finished_.wait(lock, [&] { return front_done_; });
    front_job_ = false;
    return std::exchange(front_error_, nullptr);
  }

  // out[i] = row rows[i], row_bytes each. A null out only sends the read hints.
  void Gather(const int64_t* rows, size_t n, uint8_t* out) {
    for (size_t i = 0; i < n; ++i)
      if (rows[i] < 0 || rows[i] >= rows_) throw std::out_of_range("PLE row index is out of range");
    const int fetch = Fetch();
    if (fetch > 0) {
      GatherFetch(rows, n, out, fetch);
      return;
    }
    // A major fault means the kernel evicted a page that we had hinted once.
    // Forget all hints, so the next gathers queue their reads again.
    if (stale_.exchange(false, std::memory_order_relaxed))
      for (size_t w = 0; w < seen_words_; ++w) seen_[w].store(0, std::memory_order_relaxed);
    if (n < kPoolRows) {
      Part(rows, 0, n, out);
      return;
    }
    if (!pool_) pool_ = std::make_unique<GatherPool>(kParts - 1);
    const int parts = pool_->Parts();
    pool_->Run([&](int k) { Part(rows, n * k / parts, n * (k + 1) / parts, out); });
  }

 private:
  size_t Offset(int64_t r) const {
    if (segs_.size() == 1) return segs_[0].file_offset + size_t(r) * row_;
    const auto it = std::upper_bound(segs_.begin(), segs_.end(), r,
                                     [](int64_t v, const PleSegment& s) { return v < s.first_row; }) - 1;
    return it->file_offset + size_t(r - it->first_row) * row_;
  }

  void GatherFetch(const int64_t* rows, size_t n, uint8_t* out, int mode) {
    const bool split = n >= kPoolRows || (mode >= 2 && n >= kFetchParMin);
    if (!split) {
      FetchPart(rows, 0, n, out);
      return;
    }
    if (!pool_) pool_ = std::make_unique<GatherPool>(kParts - 1);
    const int parts = pool_->Parts();
    pool_->Run([&](int k) { FetchPart(rows, n * k / parts, n * (k + 1) / parts, out); });
  }

  void HintRow(int64_t r) const {
    const size_t o = Offset(r);
    const size_t p0 = o / kPage, p1 = (o + row_ - 1) / kPage;
    madvise(const_cast<uint8_t*>(base_) + p0 * kPage, (p1 - p0 + 1) * kPage, MADV_WILLNEED);
  }

  // Rows [b, e): every page is hinted (no seen filter), kFetchAhead blocks ahead of the copy. A page that is
  // resident, or already queued, costs one cheap call; a page that was evicted is read again, in flight with the
  // others, so the copy takes minor faults on pages whose read is queued. Without out, only the hints are sent.
  void FetchPart(const int64_t* rows, size_t b, size_t e, uint8_t* out) {
    const auto t0 = std::chrono::steady_clock::now();
    const long major0 = out ? MajorFaults() : 0;
    std::chrono::steady_clock::duration hint_time{};
    size_t hinted = b;
    auto hint_to = [&](size_t upto) {
      const auto h0 = std::chrono::steady_clock::now();
      for (; hinted < upto; ++hinted) HintRow(rows[hinted]);
      hint_time += std::chrono::steady_clock::now() - h0;
    };
    if (!out) {
      hint_to(e);
      stats_.fresh_pages += e - b;
      return;
    }
    uint64_t max_ahead = 0;
    for (size_t s = b; s < e; s += kFetchBlock) {
      hint_to(std::min(e, s + kFetchAhead * kFetchBlock));
      max_ahead = std::max<uint64_t>(max_ahead, hinted - s);
      const size_t end = std::min(e, s + kFetchBlock);
      for (size_t i = s; i < end; ++i) std::memcpy(out + i * row_, base_ + Offset(rows[i]), row_);
    }
    const auto t1 = std::chrono::steady_clock::now();
    stats_.major_faults += MajorFaults() - major0;
    stats_.hint_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(hint_time).count();
    stats_.copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0 - hint_time).count();
    stats_.fresh_pages += e - b;
    uint64_t seen = stats_.max_ahead_rows.load(std::memory_order_relaxed);
    while (max_ahead > seen && !stats_.max_ahead_rows.compare_exchange_weak(seen, max_ahead)) {}
  }

  // The first touch of a page queues its read with MADV_WILLNEED, so one thread
  // keeps many reads in flight. A hint on a resident page costs about 0.3 us,
  // so later touches skip it. Exactness never depends on the hint.
  void Part(const int64_t* rows, size_t b, size_t e, uint8_t* out) {
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t fresh_pages = 0;
    for (size_t i = b; i < e; ++i) {
      const size_t o = Offset(rows[i]);
      const size_t p0 = o / kPage, p1 = (o + row_ - 1) / kPage;
      bool fresh = false;
      for (size_t p = p0; p <= p1; ++p) {
        const uint64_t bit = uint64_t(1) << (p & 63);
        fresh |= !(seen_[p / 64].fetch_or(bit, std::memory_order_relaxed) & bit);
      }
      if (fresh) madvise(const_cast<uint8_t*>(base_) + p0 * kPage, (p1 - p0 + 1) * kPage, MADV_WILLNEED);
      fresh_pages += fresh;
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (!out) return;
    const long major = MajorFaults();
    for (size_t i = b; i < e; ++i) std::memcpy(out + i * row_, base_ + Offset(rows[i]), row_);
    if (const long faults = MajorFaults() - major) {
      stale_.store(true, std::memory_order_relaxed);
      stats_.major_faults += faults;
    }
    const auto t2 = std::chrono::steady_clock::now();
    stats_.hint_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    stats_.copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    stats_.fresh_pages += fresh_pages;
  }

  static long MajorFaults() {
    rusage ru{};
    getrusage(RUSAGE_THREAD, &ru);
    return ru.ru_majflt;
  }

  // Caller holds front_mu_.
  void StartFront() {
    if (front_.joinable()) return;
    front_ = std::thread([this] { FrontLoop(); });
    pthread_setname_np(front_.native_handle(), "ple-front");
  }

  void FrontLoop() {
    for (;;) {
      Args args;
      {
        std::unique_lock<std::mutex> lock(front_mu_);
        front_wake_.wait(lock, [&] { return front_stop_ || (front_job_ && !front_done_ && !front_running_); });
        if (front_stop_) return;
        front_running_ = true;
        args = front_args_;
      }
      std::exception_ptr error;
      try {
        Gather(args.rows, args.n, args.out);
      } catch (...) {
        error = std::current_exception();
      }
      std::lock_guard<std::mutex> lock(front_mu_);
      front_running_ = false;
      front_done_ = true;
      front_error_ = error;
      front_finished_.notify_all();
    }
  }

  void HintLoop() {
    setpriority(PRIO_PROCESS, 0, 19);  // this thread only (Linux thread id 0 = caller)
    for (;;) {
      std::vector<int64_t> piece;
      {
        std::unique_lock<std::mutex> lock(hint_mu_);
        hint_wake_.wait(lock, [&] { return hint_stop_ || hint_pos_ < hints_.size(); });
        if (hint_stop_) return;
        // Small pieces let a newer Hint replace the rest soon.
        const size_t n = std::min(kHintPiece, hints_.size() - hint_pos_);
        piece.assign(hints_.begin() + hint_pos_, hints_.begin() + hint_pos_ + n);
        hint_pos_ += n;
      }
      piece.erase(std::remove_if(piece.begin(), piece.end(), [&](int64_t r) { return r < 0 || r >= rows_; }),
                  piece.end());
      if (Fetch() > 0) FetchPart(piece.data(), 0, piece.size(), nullptr);
      else Part(piece.data(), 0, piece.size(), nullptr);
    }
  }

  struct Args {
    const int64_t* rows = nullptr;
    size_t n = 0;
    uint8_t* out = nullptr;
  };

  const size_t row_;
  std::string path_;
  std::atomic<int> fetch_{0};
  size_t size_ = 0;
  int64_t rows_ = 0;
  std::vector<PleSegment> segs_;
  const uint8_t* base_ = nullptr;
  std::unique_ptr<std::atomic<uint64_t>[]> seen_;
  size_t seen_words_ = 0;
  std::atomic<bool> stale_{false};
  std::thread front_;
  std::mutex front_mu_;
  std::condition_variable front_wake_, front_finished_;
  Args front_args_;
  static constexpr size_t kHintPiece = 256;
  std::thread hint_;
  std::mutex hint_mu_;
  std::condition_variable hint_wake_;
  std::vector<int64_t> hints_;
  size_t hint_pos_ = 0;
  bool hint_stop_ = false;
  std::exception_ptr front_error_;
  bool front_job_ = false, front_done_ = false, front_running_ = false, front_stop_ = false;
  std::unique_ptr<GatherPool> pool_;
  Stats stats_;
};

// Drains the owner's started gather when the owner is destroyed. Declare it
// after the rows and the destination of the gather, so it is destroyed first:
// the table (and its thread) can outlive them.
class PleDrainGuard {
 public:
  explicit PleDrainGuard(const std::shared_ptr<PleHostTable>* table) : table_(table) {}
  ~PleDrainGuard() {
    if (!*table_) return;
    if (auto error = (*table_)->Drain()) {
      try {
        std::rethrow_exception(error);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "PLE host gather failed during shutdown: %s\n", e.what());
      }
    }
  }
  PleDrainGuard(const PleDrainGuard&) = delete;
  PleDrainGuard& operator=(const PleDrainGuard&) = delete;

 private:
  const std::shared_ptr<PleHostTable>* table_;
};

}  // namespace crucible
