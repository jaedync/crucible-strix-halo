// Times ways to read the PLE rows of one NEW prompt (all rows on pages that are not in the page cache).
// Build: g++ -O2 -std=c++20 -pthread bench/ple_cold_probe.cpp -o build/bench/ple_cold_probe
// Run:   ple_cold_probe FILE DATA_OFFSET ROW_BYTES TOKENS TRIALS STRATEGY... [drop] [resident]
// It never writes FILE. Rows are random (16 per token), and only rows whose pages are
// absent from the page cache (mincore) are used, so every trial is cold. "drop" first sends
// POSIX_FADV_DONTNEED for the whole file (clean pages only; no root; other readers see misses).
// Strategies: f0 f1 f2 (PleHostTable::Gather, fetch mode 0, 1, 2)  fault  willneed  hot (second pass, rows resident)  readahead  fadvise  tfault:N  pread:N
// N = threads. Each result is compared with a pread of the same rows.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../engine/model/flashnext/ple_host.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr size_t kPage = 4096;
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }

struct Ctx {
  int fd = -1;
  const uint8_t* base = nullptr;
  size_t size = 0, row = 0;
  uint64_t off = 0;
  int64_t nrows = 0;
  size_t Off(int64_t r) const { return off + size_t(r) * row; }
};

long MajFlt() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_majflt;
}

void Fault(const Ctx& c, const std::vector<int64_t>& rows, uint8_t* out) {
  for (size_t i = 0; i < rows.size(); ++i) std::memcpy(out + i * c.row, c.base + c.Off(rows[i]), c.row);
}

void WillNeed(const Ctx& c, const std::vector<int64_t>& rows, uint8_t* out) {
  for (size_t i = 0; i < rows.size(); ++i) {
    const size_t o = c.Off(rows[i]);
    const size_t p0 = o / kPage, p1 = (o + c.row - 1) / kPage;
    madvise(const_cast<uint8_t*>(c.base) + p0 * kPage, (p1 - p0 + 1) * kPage, MADV_WILLNEED);
  }
  Fault(c, rows, out);
}

void ReadaheadSys(const Ctx& c, const std::vector<int64_t>& rows, bool fadvise, uint8_t* out) {
  for (size_t i = 0; i < rows.size(); ++i) {
    const size_t o = c.Off(rows[i]);
    const size_t p0 = o / kPage, p1 = (o + c.row - 1) / kPage;
    if (fadvise) posix_fadvise(c.fd, p0 * kPage, (p1 - p0 + 1) * kPage, POSIX_FADV_WILLNEED);
    else readahead(c.fd, p0 * kPage, (p1 - p0 + 1) * kPage);
  }
  Fault(c, rows, out);
}

void ThreadsFault(const Ctx& c, const std::vector<int64_t>& rows, int n, uint8_t* out) {
  std::vector<std::thread> t;
  for (int k = 0; k < n; ++k)
    t.emplace_back([&, k] {
      for (size_t i = k; i < rows.size(); i += n) std::memcpy(out + i * c.row, c.base + c.Off(rows[i]), c.row);
    });
  for (auto& x : t) x.join();
}

void ThreadsPread(const Ctx& c, const std::vector<int64_t>& rows, int n, uint8_t* out) {
  std::vector<std::thread> t;
  for (int k = 0; k < n; ++k)
    t.emplace_back([&, k] {
      for (size_t i = k; i < rows.size(); i += n)
        if (pread(c.fd, out + i * c.row, c.row, c.Off(rows[i])) != (ssize_t)c.row) std::abort();
    });
  for (auto& x : t) x.join();
}

bool Resident(const Ctx& c, const uint8_t* probe, int64_t r) {
  const size_t o = c.Off(r);
  const size_t p0 = o / kPage, p1 = (o + c.row - 1) / kPage;
  unsigned char v[2];
  mincore(const_cast<uint8_t*>(probe) + p0 * kPage, (p1 - p0 + 1) * kPage, v);
  for (size_t p = 0; p <= p1 - p0; ++p)
    if (v[p] & 1) return true;
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) return 2;
  Ctx c;
  const std::string path = argv[1];
  c.off = std::strtoull(argv[2], nullptr, 10);
  c.row = std::strtoull(argv[3], nullptr, 10);
  const int tokens = std::atoi(argv[4]), trials = std::atoi(argv[5]);
  std::vector<std::string> strategies;
  bool drop = false, resident_arg = false;
  for (int i = 6; i < argc; ++i) {
    if (std::string(argv[i]) == "drop") drop = true;
    else if (std::string(argv[i]) == "resident") resident_arg = true;
    else strategies.push_back(argv[i]);
  }
  c.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (c.fd < 0) { std::perror("open"); return 1; }
  struct stat sb{};
  fstat(c.fd, &sb);
  c.size = sb.st_size;
  c.nrows = (c.size - c.off) / c.row;
  if (drop) posix_fadvise(c.fd, 0, 0, POSIX_FADV_DONTNEED);
  void* p = mmap(nullptr, c.size, PROT_READ, MAP_PRIVATE, c.fd, 0);
  if (p == MAP_FAILED) { std::perror("mmap"); return 1; }
  c.base = static_cast<const uint8_t*>(p);
  madvise(p, c.size, MADV_RANDOM);
  // A second mapping only for mincore (never touched, so it adds no pages).
  void* q = mmap(nullptr, c.size, PROT_NONE, MAP_PRIVATE, c.fd, 0);
  // PLE_PROBE_RESIDENT=1: use only rows whose pages are in the page cache (minor-fault cost of a new process).
  const bool want_resident = resident_arg;
  std::mt19937_64 rng(std::random_device{}());
  const size_t n = size_t(tokens) * 16;
  std::vector<uint8_t> out(n * c.row), ref(n * c.row);
  std::unique_ptr<crucible::PleHostTable> table;
  for (const auto& s : strategies) {
    std::vector<double> ms;
    long faults = 0;
    for (int t = 0; t < trials; ++t) {
      std::vector<int64_t> rows;
      while (rows.size() < n) {
        const int64_t r = int64_t(rng() % uint64_t(c.nrows));
        if (Resident(c, static_cast<const uint8_t*>(q), r) == want_resident) rows.push_back(r);
      }
      const long f0 = MajFlt();
      const auto a = Clock::now();
      const auto colon = s.find(':');
      const std::string name = s.substr(0, colon);
      const int arg = colon == std::string::npos ? 0 : std::atoi(s.c_str() + colon + 1);
      if (name == "fault") Fault(c, rows, out.data());
      else if (name == "willneed") WillNeed(c, rows, out.data());
      else if (name == "hot") {  // the same rows a second time: all pages resident
        WillNeed(c, rows, out.data());
        const auto b = Clock::now();
        WillNeed(c, rows, out.data());
        ms.push_back(Ms(b));
        faults += MajFlt() - f0;
        continue;
      }
      else if (name == "readahead") ReadaheadSys(c, rows, false, out.data());
      else if (name == "fadvise") ReadaheadSys(c, rows, true, out.data());
      else if (name == "tfault") ThreadsFault(c, rows, arg, out.data());
      else if (name == "pread") ThreadsPread(c, rows, arg, out.data());
      else if (name == "f0" || name == "f1" || name == "f2") {  // ple_host.h Gather, fetch mode 0, 1, 2
        if (!table) table = std::make_unique<crucible::PleHostTable>(path, c.row, std::vector<crucible::PleSegment>{{0, c.nrows, (uint64_t)c.off}});
        table->SetFetch(name[1] - '0');
        table->Gather(rows.data(), rows.size(), out.data());
      }
      else { std::fprintf(stderr, "unknown strategy %s\n", s.c_str()); return 2; }
      ms.push_back(Ms(a));
      faults += MajFlt() - f0;
      for (size_t i = 0; i < n; ++i)
        if (pread(c.fd, ref.data() + i * c.row, c.row, c.Off(rows[i])) != (ssize_t)c.row ||
            std::memcmp(ref.data() + i * c.row, out.data() + i * c.row, c.row) != 0) {
          std::fprintf(stderr, "BYTES DIFFER in %s\n", s.c_str());
          return 1;
        }
    }
    std::sort(ms.begin(), ms.end());
    std::printf("%-12s tokens=%d rows=%zu trials=%d median %.2f ms  min %.2f  max %.2f  majflt/trial %.0f\n", s.c_str(),
                tokens, n, trials, ms[ms.size() / 2], ms.front(), ms.back(), double(faults) / trials);
    std::fflush(stdout);
  }
  return 0;
}
