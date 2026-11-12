// The serve JSON-line limit and reader. CPU-only, so tests/serve_line_test.cpp compiles it.
//
// The server sends a prompt as token IDs in one line. A line of the old fixed 2 MiB held about 300,000 to 400,000
// prompt IDs, so a 524,288-token YaRN server refused long prompts with HTTP 413. The limit now follows the context:
// 8 bytes a token (the widest ID has 6 digits and a comma) plus 1 MiB for the other fields, never less than 2 MiB.
// Image pixels go in private files, so one limit serves text and image lines. The engine
// reports the value in its ready event ("line_bytes"); the server refuses a longer request before it reaches the engine.
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

namespace crucible {
inline constexpr size_t kServeLineFloorBytes = size_t(2) << 20;
inline constexpr size_t kServeLineTokenBytes = 8;
inline constexpr size_t kServeLineMarginBytes = size_t(1) << 20;

inline size_t ServeLineBytes(int ctx) {
  return std::max(kServeLineFloorBytes, size_t(std::max(ctx, 0)) * kServeLineTokenBytes + kServeLineMarginBytes);
}

// Reads newline-ended lines from a file descriptor in blocks. The old reader took one character at a time through
// std::cin (about 120 ms for a 3.7 MB line on the test machine, in the time to the first token); this one scans each block with memchr.
// A line longer than the limit keeps its first `limit` bytes, sets `oversized`, and is read to its end, so the next
// line starts clean. A last line without a newline is still a line.
class ServeLineReader {
 public:
  ServeLineReader(int fd, size_t limit) : fd_(fd), limit_(limit), block_(size_t(1) << 16) {}
  size_t Limit() const { return limit_; }

  bool Next(std::string& line, bool& oversized) {
    line.clear();
    oversized = false;
    bool any = false;
    for (;;) {
      if (begin_ == end_ && !Fill()) return any;
      any = true;
      const char* start = block_.data() + begin_;
      const size_t avail = end_ - begin_;
      const void* found = std::memchr(start, '\n', avail);
      const size_t take = found ? size_t(static_cast<const char*>(found) - start) : avail;
      Append(line, oversized, start, take);
      begin_ += take;
      if (found) {
        ++begin_;
        return true;
      }
    }
  }

 private:
  void Append(std::string& line, bool& oversized, const char* data, size_t size) {
    const size_t room = limit_ - std::min(limit_, line.size());
    if (size > room) oversized = true;
    line.append(data, std::min(size, room));
  }
  bool Fill() {
    if (eof_) return false;
    for (;;) {
      const ssize_t n = read(fd_, block_.data(), block_.size());
      if (n > 0) {
        begin_ = 0;
        end_ = size_t(n);
        return true;
      }
      if (n < 0 && errno == EINTR) continue;
      eof_ = true;  // end of input, or a read error: both end the serve loop, as std::cin.get did
      return false;
    }
  }

  int fd_;
  size_t limit_;
  std::vector<char> block_;
  size_t begin_ = 0, end_ = 0;
  bool eof_ = false;
};
}  // namespace crucible
