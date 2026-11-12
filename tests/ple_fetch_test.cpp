// CRUCIBLE_PLE_FETCH (ple_host.h): the batched read of the PLE rows of a new prompt. The bytes must be those of the
// plain gather, the reads must be hinted before the copy (no synchronous major fault per page), and the reads in
// flight must stay under the stated bound. The fault checks need a file system that can drop clean pages (not tmpfs):
// the file goes to $PLE_TEST_DIR or /var/tmp, and the checks say SKIP when the pages stay.
#include "../engine/model/flashnext/ple_host.h"

#include <sys/resource.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(x)                                                                    \
  do {                                                                              \
    if (!(x)) {                                                                     \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x);    \
      std::exit(1);                                                                 \
    }                                                                               \
  } while (0)

namespace {
long MajorFaults() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_majflt;
}

// How many pages of the file are in the page cache.
size_t ResidentPages(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  struct stat sb{};
  fstat(fd, &sb);
  void* p = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
  const size_t pages = (sb.st_size + 4095) / 4096;
  std::vector<unsigned char> v(pages);
  mincore(p, sb.st_size, v.data());
  munmap(p, sb.st_size);
  close(fd);
  size_t n = 0;
  for (auto c : v) n += c & 1;
  return n;
}
}  // namespace

int main() {
  using namespace crucible;
  constexpr size_t kRow = 122;  // the 4.05 pack: (10 * 6 + 1) * 2 bytes
  constexpr uint64_t kHeader = 6328;
  constexpr int64_t kRows = 400000;  // 48.8 MB: 11,900 pages, so 500 random rows hit 500 pages
  std::vector<uint8_t> file(kHeader + kRows * kRow);
  std::mt19937_64 rng(11);
  for (auto& b : file) b = uint8_t(rng());
  const char* dir = std::getenv("PLE_TEST_DIR");
  std::string path = std::string(dir ? dir : "/var/tmp") + "/ple_fetch_testXXXXXX";
  {
    const int fd = mkstemp(path.data());
    CHECK(fd >= 0);
    CHECK(write(fd, file.data(), file.size()) == (ssize_t)file.size());
    fsync(fd);
    close(fd);
  }
  auto expect = [&](int64_t r) { return file.data() + kHeader + r * kRow; };
  auto make_rows = [&](size_t n) {
    std::vector<int64_t> rows(n);
    for (auto& r : rows) r = int64_t(rng() % kRows);
    if (n > 3) {  // edges, a repeat, and a row that crosses a page edge
      rows[0] = 0;
      rows[1] = kRows - 1;
      rows[2] = rows[3];
      for (int64_t r = 100; r < 200; ++r)
        if ((kHeader + r * kRow) / 4096 != (kHeader + r * kRow + kRow - 1) / 4096) { rows[n - 1] = r; break; }
    }
    return rows;
  };

  // 1. Same bytes in every mode, at decode, short-prompt, pool and block-edge sizes.
  for (int mode : {0, 1, 2}) {
    PleHostTable t(path, kRow, {{0, 150000, kHeader}, {150000, kRows - 150000, kHeader + 150000 * kRow}});
    t.SetFetch(mode);
    CHECK(t.Fetch() == mode);
    const size_t block = PleHostTable::kFetchBlock;
    for (size_t n : {size_t(1), size_t(16), size_t(63), size_t(64), size_t(65), block, 2 * block, 2 * block + 1,
                     size_t(159), size_t(160), size_t(500), size_t(1023), size_t(1024), size_t(5000), size_t(40000)}) {
      const auto rows = make_rows(n);
      std::vector<uint8_t> out(n * kRow, 0xEE);
      for (int pass = 0; pass < 2; ++pass) {  // cold, then resident
        t.Gather(rows.data(), n, out.data());
        for (size_t i = 0; i < n; ++i) CHECK(!std::memcmp(out.data() + i * kRow, expect(rows[i]), kRow));
      }
    }
    // The background gather and the hint path give the same bytes too.
    const auto rows = make_rows(800);
    std::vector<uint8_t> out(rows.size() * kRow, 0);
    t.Hint(rows);
    t.Start(rows.data(), rows.size(), out.data());
    t.Wait();
    for (size_t i = 0; i < rows.size(); ++i) CHECK(!std::memcmp(out.data() + i * kRow, expect(rows[i]), kRow));
    t.Hint({});
    // Bad rows still fail closed.
    int64_t bad[2] = {kRows - 1, kRows};
    uint8_t two[2 * kRow];
    bool threw = false;
    try { t.Gather(bad, 2, two); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
  }

  // 2. The reads in flight (rows hinted and not yet copied) stay under kFetchAhead blocks per part.
  {
    PleHostTable t(path, kRow, {{0, kRows, kHeader}});
    for (int mode : {1, 2}) {
      t.SetFetch(mode);
      t.stats().max_ahead_rows = 0;
      const auto rows = make_rows(40000);
      std::vector<uint8_t> out(rows.size() * kRow);
      t.Gather(rows.data(), rows.size(), out.data());
      CHECK(t.stats().max_ahead_rows.load() > 0);
      CHECK(t.stats().max_ahead_rows.load() <= PleHostTable::kFetchAhead * PleHostTable::kFetchBlock);
      // Pages in flight in the whole gather: at most parts x rows x 2 pages (a row can cross a page edge).
      CHECK(PleHostTable::kParts * PleHostTable::kFetchAhead * PleHostTable::kFetchBlock * 2 <= 4096);
    }
  }

  // 3. A stale gather (every page "seen", then evicted) faults once per page; the fetch modes do not.
  {
    PleHostTable t(path, kRow, {{0, kRows, kHeader}});
    const auto rows = make_rows(500);  // one new short prompt of 31 tokens
    std::vector<uint8_t> out(rows.size() * kRow);
    t.SetFetch(0);
    t.Gather(rows.data(), rows.size(), out.data());  // sets the seen bits
    t.EvictForTest();
    if (ResidentPages(path) > 100) {
      std::puts("ple_fetch_test: SKIP fault checks (this file system keeps the pages)");
    } else {
      const long f0 = MajorFaults();
      t.Gather(rows.data(), rows.size(), out.data());  // seen, evicted: no hint, one fault per page
      const long legacy = MajorFaults() - f0;
      CHECK(legacy >= 400);
      for (int mode : {1, 2}) {
        t.SetFetch(mode);
        t.EvictForTest();
        CHECK(ResidentPages(path) <= 100);
        const long f1 = MajorFaults();
        t.Gather(rows.data(), rows.size(), out.data());
        const long fetch = MajorFaults() - f1;
        std::printf("ple_fetch_test: mode %d major faults %ld (plain gather of stale pages: %ld)\n", mode, fetch, legacy);
        CHECK(fetch * 10 <= legacy);
        for (size_t i = 0; i < rows.size(); ++i) CHECK(!std::memcmp(out.data() + i * kRow, expect(rows[i]), kRow));
      }
    }
  }

  // 4. The mode is a per-table choice that can change between gathers.
  {
    PleHostTable t(path, kRow, {{0, kRows, kHeader}});
    CHECK(t.Fetch() == 0);
    bool threw = false;
    try { t.SetFetch(3); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    try { t.SetFetch(-1); threw = false; } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    CHECK(t.Fetch() == 0);
  }
  unlink(path.c_str());
  std::puts("ple_fetch_test: ok");
  return 0;
}
