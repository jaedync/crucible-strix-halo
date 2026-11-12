// the serve JSON-line limit follows the context, and the reader takes a long line whole (serve_line.h).
#include "../engine/serve/serve_line.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <unistd.h>

namespace {
// Feeds `text` into a pipe from a second thread (a long line does not fit the pipe buffer) and reads it back.
struct Piped {
  int read_end = -1;
  std::thread writer;
  explicit Piped(std::string text) {
    int fds[2];
    assert(pipe(fds) == 0);
    read_end = fds[0];
    writer = std::thread([fd = fds[1], text = std::move(text)] {
      size_t done = 0;
      while (done < text.size()) {
        const ssize_t n = write(fd, text.data() + done, text.size() - done);
        assert(n > 0);
        done += size_t(n);
      }
      close(fd);
    });
  }
  ~Piped() { writer.join(); close(read_end); }
};
}  // namespace

int main() {
  using crucible::ServeLineBytes;
  constexpr size_t kMiB = size_t(1) << 20;
  // The same numbers as server/tests/test_engine_line.py LINE_BYTES: max(2 MiB, 8 bytes a token + 1 MiB).
  assert(ServeLineBytes(32768) == 2 * kMiB);
  assert(ServeLineBytes(262144) == 3 * kMiB);
  assert(ServeLineBytes(524288) == 5 * kMiB);
  assert(ServeLineBytes(1048576) == 9 * kMiB);
  // A prompt that fills the context with the widest IDs (6 digits and a comma) fits, with room for the other fields.
  for (const int ctx : {1, 4096, 262144, 524288, 1048576})
    assert(size_t(ctx) * 7 + 64 * 1024 < ServeLineBytes(ctx));

  const size_t limit = ServeLineBytes(524288);
  const std::string big(3 * kMiB + 17, '7');       // above the old 2 MiB, under the limit
  const std::string huge(limit + 4096, '9');        // above the limit
  {
    Piped input(big + "\n" + huge + "\nshort\n\nlast");
    crucible::ServeLineReader reader(input.read_end, limit);
    std::string line;
    bool oversized = true;
    assert(reader.Next(line, oversized) && !oversized && line == big);
    // An oversized line is consumed to its end, and the next line is read normally.
    assert(reader.Next(line, oversized) && oversized && line.size() == limit);
    assert(reader.Next(line, oversized) && !oversized && line == "short");
    assert(reader.Next(line, oversized) && !oversized && line.empty());
    // A last line with no newline is still a line (as std::cin.get before).
    assert(reader.Next(line, oversized) && !oversized && line == "last");
    assert(!reader.Next(line, oversized));
    assert(!reader.Next(line, oversized));
  }
  {
    // An oversized last line with no newline still reports the error.
    Piped input(huge);
    crucible::ServeLineReader reader(input.read_end, limit);
    std::string line;
    bool oversized = false;
    assert(reader.Next(line, oversized) && oversized);
    assert(!reader.Next(line, oversized));
  }
  {
    // A line of exactly the limit is not oversized; one byte more is.
    Piped input(std::string(limit, 'a') + "\n" + std::string(limit + 1, 'b') + "\n");
    crucible::ServeLineReader reader(input.read_end, limit);
    std::string line;
    bool oversized = true;
    assert(reader.Next(line, oversized) && !oversized && line.size() == limit);
    assert(reader.Next(line, oversized) && oversized);
    assert(!reader.Next(line, oversized));
  }
  {
    // Empty input: no line.
    Piped input("");
    crucible::ServeLineReader reader(input.read_end, limit);
    std::string line;
    bool oversized = false;
    assert(!reader.Next(line, oversized));
  }
  std::puts("serve_line_test: ok");
  return 0;
}
