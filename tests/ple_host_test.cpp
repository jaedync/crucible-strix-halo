#include "../engine/model/flashnext/ple_host.h"
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Always on, also with NDEBUG: several checks have side effects.
#define CHECK(x)                                                        \
  do {                                                                  \
    if (!(x)) {                                                         \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {
// TMPDIR first (the test machine /tmp is a small shared tmpfs that is often full), then /tmp.
std::string TempDir() {
  const char* dir = std::getenv("TMPDIR");
  return dir && *dir ? dir : "/tmp";
}

std::string WriteFile(const std::vector<uint8_t>& bytes) {
  std::string path = TempDir() + "/ple_host_testXXXXXX";
  const int fd = mkstemp(path.data());
  if (fd < 0 || write(fd, bytes.data(), bytes.size()) != (ssize_t)bytes.size()) {
    std::perror("ple_host_test setup");
    std::exit(1);
  }
  close(fd);
  return path;
}

bool Throws(const std::function<void()>& f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}
}  // namespace

int main() {
  using namespace crucible;
  constexpr size_t kRow = 102;
  constexpr uint64_t kHeader = 26736;  // not page aligned, like the 3.05 file
  constexpr int64_t kRows = 200000;    // spans many pages and both segment layouts
  std::vector<uint8_t> file(kHeader + kRows * kRow + 333);
  std::mt19937_64 rng(7);
  for (auto& b : file) b = uint8_t(rng());
  const std::string path = WriteFile(file);
  auto expect = [&](int64_t r) { return file.data() + kHeader + r * kRow; };

  // Contiguous shards merge into one segment; a gap keeps two.
  {
    PleHostTable t(path, kRow, {{0, 120000, kHeader}, {120000, kRows - 120000, kHeader + 120000 * kRow}});
    CHECK(t.Rows() == kRows && t.Segments() == 1);
  }
  const uint64_t gap = kHeader + 1000 * kRow;  // segment 1 starts 1000 rows into the payload
  PleHostTable split(path, kRow, {{0, 1000, kHeader}, {1000, 5000, gap + 4000 * kRow}});
  CHECK(split.Segments() == 2 && split.Rows() == 6000);
  {
    std::vector<int64_t> rows = {0, 999, 1000, 5999};
    std::vector<uint8_t> out(rows.size() * kRow);
    split.Gather(rows.data(), rows.size(), out.data());
    CHECK(!std::memcmp(out.data(), expect(0), kRow));
    CHECK(!std::memcmp(out.data() + kRow, expect(999), kRow));
    CHECK(!std::memcmp(out.data() + 2 * kRow, expect(5000), kRow));
    CHECK(!std::memcmp(out.data() + 3 * kRow, expect(9999), kRow));
  }

  // Decode-sized and prefill-sized gathers (the pool path) are byte-identical.
  PleHostTable t(path, kRow, {{0, kRows, kHeader}});
  for (size_t n : {size_t(16), size_t(128), size_t(4096 * 16)}) {
    std::vector<int64_t> rows(n);
    for (auto& r : rows) r = int64_t(rng() % kRows);
    rows[0] = 0;
    rows[n - 1] = kRows - 1;
    std::vector<uint8_t> out(n * kRow, 0);
    for (int pass = 0; pass < 2; ++pass) {  // first touch, then resident pages
      t.Gather(rows.data(), n, out.data());
      for (size_t i = 0; i < n; ++i) CHECK(!std::memcmp(out.data() + i * kRow, expect(rows[i]), kRow));
    }
  }

  // The background gather gives the same bytes, and the next Start reuses the thread.
  for (int pass = 0; pass < 3; ++pass) {
    const size_t n = pass == 1 ? 16 : 4096 * 16;
    std::vector<int64_t> rows(n);
    for (auto& r : rows) r = int64_t(rng() % kRows);
    std::vector<uint8_t> out(n * kRow, 0);
    t.Start(rows.data(), n, out.data());
    CHECK(Throws([&] { t.Start(rows.data(), n, out.data()); }));  // one gather at a time
    t.Wait();
    t.Wait();  // no gather: no-op
    for (size_t i = 0; i < n; ++i) CHECK(!std::memcmp(out.data() + i * kRow, expect(rows[i]), kRow));
  }
  {
    const int64_t bad = kRows;
    uint8_t row[kRow];
    t.Start(&bad, 1, row);
    CHECK(Throws([&] { t.Wait(); }));  // the error reaches the waiter
    t.Wait();
  }

  // Drain waits without throwing, so an owner's destructor can free the rows
  // and the destination of a started gather (ASan and TSan check the frees).
  {
    CHECK(!t.Drain());  // no gather: no-op
    const size_t n = 4096 * 16;
    auto rows = std::make_unique<int64_t[]>(n);
    for (size_t i = 0; i < n; ++i) rows[i] = int64_t(rng() % kRows);
    auto out = std::make_unique<uint8_t[]>(n * kRow);
    t.Start(rows.get(), n, out.get());
    CHECK(!t.Drain());
    for (size_t i = 0; i < n; ++i) CHECK(!std::memcmp(out.get() + i * kRow, expect(rows[i]), kRow));
    out.reset();
    rows.reset();
    const int64_t bad = kRows;
    uint8_t row[kRow];
    t.Start(&bad, 1, row);
    CHECK(t.Drain() != nullptr);  // the error is returned, not thrown
    CHECK(!t.Drain());
    t.Wait();
  }
  {
    // The guard drains a started gather before the rows and destination go.
    auto shared = std::make_shared<PleHostTable>(path, kRow, std::vector<PleSegment>{{0, kRows, kHeader}});
    const size_t n = 4096 * 16;
    auto rows = std::make_unique<int64_t[]>(n);
    for (size_t i = 0; i < n; ++i) rows[i] = int64_t(rng() % kRows);
    auto out = std::make_unique<uint8_t[]>(n * kRow);
    {
      PleDrainGuard guard(&shared);
      shared->Start(rows.get(), n, out.get());
    }
    for (size_t i = 0; i < n; ++i) CHECK(!std::memcmp(out.get() + i * kRow, expect(rows[i]), kRow));
    out.reset();
    rows.reset();
    shared->Wait();  // drained: no-op
    std::shared_ptr<PleHostTable> none;
    { PleDrainGuard idle(&none); }  // no table: no-op
  }

  // Hints (including invalid rows) never change the gathered bytes, and a
  // gather started while hints are queued still completes.
  {
    std::vector<int64_t> later(3000);
    for (auto& r : later) r = int64_t(rng() % kRows);
    later.push_back(-5);
    later.push_back(kRows + 7);
    t.Hint(later);
    std::vector<int64_t> rows(later.begin(), later.begin() + 2000);
    std::vector<uint8_t> out(rows.size() * kRow, 0);
    t.Start(rows.data(), rows.size(), out.data());
    t.Wait();
    for (size_t i = 0; i < rows.size(); ++i) CHECK(!std::memcmp(out.data() + i * kRow, expect(rows[i]), kRow));
    t.Hint({});
  }

  // Bad geometry and out-of-range rows fail closed.
  std::vector<uint8_t> one(kRow);
  for (int64_t bad : {int64_t(-1), kRows})
    CHECK(Throws([&] { t.Gather(&bad, 1, one.data()); }));
  CHECK(Throws([&] { PleHostTable(path, kRow, {{0, kRows + 10, kHeader}}); }));  // past end of file
  CHECK(Throws([&] { PleHostTable(path, kRow, {{0, 10, kHeader}, {11, 10, kHeader}}); }));  // row gap
  CHECK(Throws([&] { PleHostTable(path, 0, {{0, 10, kHeader}}); }));
  CHECK(Throws([&] { PleHostTable("/nonexistent/ple", kRow, {{0, 10, 0}}); }));
  unlink(path.c_str());
  std::puts("ple_host_test: ok");
  return 0;
}
