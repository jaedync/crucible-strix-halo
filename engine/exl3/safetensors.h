// Bounded safetensors reader for sharded checkpoints (header parse + mmap).
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace st {

[[noreturn]] inline void Fail(const std::string& file, const std::string& tensor,
                             const std::string& reason) {
  std::fprintf(stderr, "checkpoint error: %s, tensor %s: %s; check or replace the checkpoint\n",
               file.c_str(), tensor.empty() ? "<header>" : tensor.c_str(), reason.c_str());
  std::exit(1);
}

struct TensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // absolute file offsets
  std::string file;
};

class Scanner {
 public:
  explicit Scanner(std::string_view s, std::string file = "JSON", bool recover = false)
      : s_(s), file_(file), recover_(recover) {}
  void Tensor(const std::string& name) { tensor_ = name; }
  void Ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' ||
                            s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
  }
  bool Peek(char c) { Ws(); return i_ < s_.size() && s_[i_] == c; }
  void Expect(char c) {
    Ws();
    if (i_ == s_.size() || s_[i_] != c) Error(std::string("expected '") + c + "'");
    ++i_;
  }
  void Separator(char close) {
    if (Peek(close)) return;
    Expect(',');
    if (Peek(close)) Error("trailing comma");
  }
  void Finish() { Ws(); if (i_ != s_.size()) Error("trailing JSON data"); }
  std::string Str() {
    Expect('"');
    std::string out;
    while (i_ < s_.size()) {
      const unsigned char c = s_[i_++];
      if (c == '"') return out;
      if (c < 32) Error("control character in string");
      if (c != '\\') { out += c; continue; }
      if (i_ == s_.size()) Error("unterminated string escape");
      const char escape = s_[i_++];
      switch (escape) {
        case '"': case '\\': case '/': out += escape; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': AppendUnicode(out); break;
        default: Error("invalid string escape");
      }
    }
    Error("unterminated string");
  }
  int64_t Int() {
    Ws();
    const size_t start = i_;
    Number(false);
    int64_t value;
    const auto result = std::from_chars(s_.data() + start, s_.data() + i_, value);
    if (result.ec != std::errc{}) Error("integer overflow");
    return value;
  }
  uint64_t Uint() {
    Ws();
    const size_t start = i_;
    Number(false);
    uint64_t value;
    const auto result = std::from_chars(s_.data() + start, s_.data() + i_, value);
    if (result.ec != std::errc{}) Error("unsigned integer overflow or negative value");
    return value;
  }
  double Real() {
    Ws();
    const size_t start = i_;
    Number(true);
    double value;
    const auto result = std::from_chars(s_.data() + start, s_.data() + i_, value);
    if (result.ec != std::errc{}) Error("number overflow or invalid number");
    return value;
  }
  std::vector<int64_t> IntArray() {
    std::vector<int64_t> values;
    Expect('[');
    while (!Peek(']')) { values.push_back(Int()); Separator(']'); }
    Expect(']');
    return values;
  }
  void Skip(unsigned depth = 0) {
    if (depth > 64) Error("JSON nesting exceeds 64 levels");
    Ws();
    if (i_ == s_.size()) Error("missing JSON value");
    const char c = s_[i_];
    if (c == '"') { Str(); return; }
    if (c == '{' || c == '[') {
      const char close = c == '{' ? '}' : ']';
      ++i_;
      while (!Peek(close)) {
        if (c == '{') { Str(); Expect(':'); }
        Skip(depth + 1);
        Separator(close);
      }
      Expect(close);
    } else if (c == 't' || c == 'f' || c == 'n') {
      const std::string_view literal = c == 't' ? "true" : c == 'f' ? "false" : "null";
      if (s_.substr(i_, literal.size()) != literal) Error("invalid JSON literal");
      i_ += literal.size();
    } else Number(true);
  }
  size_t pos() const { return i_; }

 private:
  [[noreturn]] void Error(const std::string& reason) const {
    const std::string message = reason + " at JSON byte " + std::to_string(i_);
    if (recover_) throw std::runtime_error(message);
    Fail(file_, tensor_, message);
  }
  bool Digit() const { return i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9'; }
  void Digits() { if (!Digit()) Error("expected digit"); while (Digit()) ++i_; }
  void Number(bool fractional) {
    if (i_ < s_.size() && s_[i_] == '-') ++i_;
    if (!Digit()) Error("expected integer or number");
    if (s_[i_] == '0') ++i_; else Digits();
    if (Digit()) Error("leading zero in number");
    if (!fractional) return;
    if (i_ < s_.size() && s_[i_] == '.') { ++i_; Digits(); }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      Digits();
    }
  }
  uint32_t Hex4() {
    uint32_t value = 0;
    for (int n = 0; n < 4; ++n) {
      if (i_ == s_.size()) Error("short unicode escape");
      const char c = s_[i_++];
      const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                    c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
      if (d < 0) Error("invalid unicode escape");
      value = value * 16 + d;
    }
    return value;
  }
  void AppendUnicode(std::string& out) {
    uint32_t u = Hex4();
    if (u >= 0xd800 && u <= 0xdbff) {
      if (s_.substr(i_, 2) != "\\u") Error("missing low surrogate");
      i_ += 2;
      const uint32_t low = Hex4();
      if (low < 0xdc00 || low > 0xdfff) Error("invalid low surrogate");
      u = 0x10000 + (u - 0xd800) * 1024 + low - 0xdc00;
    } else if (u >= 0xdc00 && u <= 0xdfff) Error("unpaired low surrogate");
    if (u < 128) out += char(u);
    else {
      const int n = u < 0x800 ? 2 : u < 0x10000 ? 3 : 4;
      out += char((n == 2 ? 0xc0 : n == 3 ? 0xe0 : 0xf0) | (u >> (6 * (n - 1))));
      for (int k = n - 2; k >= 0; --k) out += char(0x80 | ((u >> (6 * k)) & 63));
    }
  }
  std::string_view s_;
  std::string file_, tensor_;
  size_t i_ = 0;
  bool recover_ = false;
};

inline size_t DtypeBytes(const std::string& dtype) {
  if (dtype == "BOOL" || dtype == "I8" || dtype == "U8") return 1;
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  return 0;
}

class Checkpoint {
 public:
  explicit Checkpoint(const std::string& dir) : dir_(dir) {
    const std::string path = dir + "/model.safetensors.index.json";
    std::ifstream f(path);
    if (!f) Fail(path, "", "cannot open index: " + std::string(std::strerror(errno)));
    std::stringstream ss;
    ss << f.rdbuf();
    if (f.bad()) Fail(path, "", "cannot read index");
    const std::string s = ss.str();
    Scanner sc(s, path);
    sc.Expect('{');
    bool have_map = false;
    while (!sc.Peek('}')) {
      const std::string k = sc.Str();
      sc.Expect(':');
      if (k == "weight_map") {
        if (have_map) Fail(path, "", "duplicate weight_map");
        have_map = true;
        sc.Expect('{');
        while (!sc.Peek('}')) {
          const std::string name = sc.Str();
          sc.Tensor(name);
          sc.Expect(':');
          const std::string file = sc.Str();
          CheckFile(file);
          if (!shard_of_.emplace(name, file).second) Fail(path, name, "duplicate index tensor");
          sc.Separator('}');
        }
        sc.Expect('}');
        sc.Tensor("");
      } else sc.Skip();
      sc.Separator('}');
    }
    sc.Expect('}');
    sc.Finish();
    if (!have_map) Fail(path, "", "missing weight_map");
  }
  Checkpoint(const Checkpoint&) = delete;
  Checkpoint& operator=(const Checkpoint&) = delete;
  ~Checkpoint() {
    for (const auto& [file, m] : maps_) {
      munmap(const_cast<uint8_t*>(m.base), m.size);
      close(m.read_fd);
    }
  }

  void AddFile(const std::string& file) {
    CheckFile(file);
    LoadHeader(file);
    for (const auto& [n, t] : tensors_) if (t.file == file) shard_of_[n] = file;
  }
  bool Has(const std::string& name) const { return shard_of_.count(name) != 0; }
  // Index shards are plain names in dir_ (CheckFile); an override key is the sidecar's own path.
  std::string PathOf(const std::string& shard) const {
    return shard.find('/') == std::string::npos ? dir_ + "/" + shard : shard;
  }
  // An EXL3 trellis [in/16][out/16][16*K] may be requantized at another K; the engine's
  // linear kernels take K 2 to 6 (engine/exl3/decode_support.h).
  static bool SameOverrideShape(const std::string& name, const std::vector<int64_t>& a,
                                const std::vector<int64_t>& b) {
    if (a == b) return true;
    const std::string_view suffix = ".trellis";
    const bool trellis = name.size() > suffix.size() && name.ends_with(suffix);
    return trellis && a.size() == 3 && b.size() == 3 && a[0] == b[0] && a[1] == b[1] && b[2] % 16 == 0 &&
           b[2] >= 32 && b[2] <= 96;
  }
  // Replaces the data of existing tensors named prefix* with the tensors of an
  // external safetensors file (the MTP draft sidecar). Every tensor in the file
  // must replace one of the same dtype and shape (SameOverrideShape). Returns the replaced count.
  // A file with no tensors stops here, unless allow_empty: the caller then sees the count 0 and decides.
  size_t Override(const std::string& path, const std::string& prefix, bool allow_empty = false) {
    // A key with a slash cannot collide with an index shard name (CheckFile).
    const std::string key = path.find('/') == std::string::npos ? "./" + path : path;
    if (maps_.count(key)) Fail(path, "", "override file is already loaded");
    const auto found = ParseHeader(key, path);
    if (found.empty() && !allow_empty) Fail(path, "", "override file has no tensors");
    for (const auto& [name, ti] : found) {
      if (name.rfind(prefix, 0) != 0) Fail(path, name, "override tensor is outside " + prefix);
      if (!Has(name)) Fail(path, name, "override tensor is not in the checkpoint");
      const TensorInfo& old = Info(name);
      if (old.dtype != ti.dtype || !SameOverrideShape(name, old.shape, ti.shape))
        Fail(path, name, "override dtype or shape differs");
    }
    for (const auto& [name, ti] : found) {
      tensors_[name] = ti;
      shard_of_[name] = key;
    }
    return found.size();
  }
  [[noreturn]] void Error(const std::string& name, const std::string& reason) const {
    const auto it = shard_of_.find(name);
    Fail(it == shard_of_.end() ? dir_ + "/model.safetensors.index.json" : PathOf(it->second), name, reason);
  }
  const TensorInfo& Info(const std::string& name) {
    const auto it = shard_of_.find(name);
    if (it == shard_of_.end()) Error(name, "required tensor is missing from index");
    LoadHeader(it->second);
    const auto t = tensors_.find(name);
    if (t == tensors_.end() || t->second.file != it->second) Error(name, "tensor is missing from its shard");
    return t->second;
  }
  const TensorInfo& Require(const std::string& name, std::initializer_list<std::string_view> dtypes,
                            const std::vector<int64_t>& shape) {
    const auto& ti = Info(name);
    bool dtype_ok = false;
    std::string expected;
    for (const auto d : dtypes) { dtype_ok |= ti.dtype == d; expected += (expected.empty() ? "" : "/") + std::string(d); }
    if (!dtype_ok) Error(name, "expected dtype " + expected + ", got " + ti.dtype);
    if (ti.shape != shape) Error(name, "shape does not match the engine geometry");
    return ti;
  }
  const uint8_t* Data(const std::string& name, size_t* bytes = nullptr) {
    // A fault on the mapping reads up to read_ahead_kb (4 MB on the test machine) per small value;
    // validation reads small values with ReadBytes instead.
    if (no_mapped_reads_) Error(name, "mapped payload read during validation; use ReadBytes");
    const auto& t = Info(name);
    if (bytes) *bytes = t.end - t.begin;
    return maps_.at(t.file).base + t.begin;
  }
  // Reads the first n bytes of a tensor with pread on a descriptor without readahead,
  // or from the values that Prefetch already read.
  void ReadBytes(const std::string& name, void* dst, size_t n) {
    const auto& t = Info(name);
    if (n > t.end - t.begin) Error(name, "read past the end of the tensor");
    if (const auto held = small_.find(name); held != small_.end()) {
      if (n) std::memcpy(dst, held->second.data(), n);  // a 0-byte value has no data pointer
      return;
    }
    if (const int error = PreadAll(maps_.at(t.file).read_fd, dst, n, t.begin)) Error(name, ReadFailure(error));
  }
  std::vector<std::string> NamesEndingWith(std::string_view suffix) const {
    std::vector<std::string> names;
    for (const auto& [name, file] : shard_of_)
      if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
        names.push_back(name);
    std::sort(names.begin(), names.end());
    return names;
  }
  // Reads small tensors (at most max_bytes) on several threads for later ReadBytes calls.
  // Validation reads about 37,000 4-byte markers; one at a time on a cold NVMe this took ~5 s.
  // Returns how many values it read.
  size_t Prefetch(const std::vector<std::string>& names, int threads = 16, size_t max_bytes = 64) {
    struct Item { const std::string* name; int fd; uint64_t offset; size_t bytes; };
    std::vector<Item> items;
    for (const auto& name : names) {
      const auto& t = Info(name);  // loads headers here, before any thread starts
      if (t.end - t.begin <= max_bytes && !small_.count(name))
        items.push_back({&name, maps_.at(t.file).read_fd, t.begin, size_t(t.end - t.begin)});
    }
    std::vector<std::vector<uint8_t>> values(items.size());
    std::vector<int> errors(items.size(), 0);
    std::atomic<size_t> next{0};
    auto work = [&] {
      for (size_t i; (i = next++) < items.size();) {
        try {
          values[i].resize(items[i].bytes);
          errors[i] = PreadAll(items[i].fd, values[i].data(), items[i].bytes, items[i].offset);
        } catch (const std::bad_alloc&) {
          errors[i] = kNoMemory;  // reported with the tensor name after the joins
        }
      }
    };
    std::vector<std::thread> pool;
    pool.reserve(size_t(std::max(1, threads)));
    // A thread that cannot start only makes the reads slower: the started threads and
    // this thread take its share.
    try {
      for (int i = 1; i < threads; ++i) pool.emplace_back(work);
    } catch (const std::exception&) {  // std::system_error, or std::bad_alloc for the thread state
    }
    work();
    for (auto& thread : pool) thread.join();
    for (size_t i = 0; i < items.size(); ++i) {
      if (errors[i]) Error(*items[i].name, ReadFailure(errors[i]));
      small_.emplace(*items[i].name, std::move(values[i]));
    }
    return items.size();
  }

 private:
  friend class NoMappedReads;
  struct Map { const uint8_t* base; size_t size; int read_fd; };
  static constexpr int kShortRead = -1, kNoMemory = -2;
  static std::string ReadFailure(int error) {
    if (error == kShortRead) return "short read";
    if (error == kNoMemory) return "cannot allocate memory for the value";
    return "pread failed: " + std::string(std::strerror(error));
  }
  // 0 on success, errno for a failed read, kShortRead for an early end of file.
  static int PreadAll(int fd, void* dst, size_t n, uint64_t offset) {
    auto* out = static_cast<uint8_t*>(dst);
    for (size_t done = 0; done < n;) {
      const ssize_t got = pread(fd, out + done, n - done, off_t(offset + done));
      if (got < 0 && errno == EINTR) continue;
      if (got < 0) return errno;
      if (got == 0) return kShortRead;
      done += size_t(got);
    }
    return 0;
  }
  void CheckFile(const std::string& file) const {
    if (file.empty() || file == "." || file == ".." || file.find('/') != std::string::npos ||
        file.find('\\') != std::string::npos || file.find('\0') != std::string::npos)
      Fail(dir_ + "/model.safetensors.index.json", "", "shard name must be a plain filename");
  }
  void CheckTensor(const std::string& path, const std::string& name, const TensorInfo& ti,
                   const std::vector<int64_t>& offsets, uint64_t payload) {
    if (offsets.size() != 2) Fail(path, name, "data_offsets must contain two offsets");
    if (offsets[0] < 0 || offsets[1] < offsets[0] || uint64_t(offsets[1]) > payload)
      Fail(path, name, "data_offsets lie outside the mapped payload");
    uint64_t bytes = DtypeBytes(ti.dtype);
    if (!bytes) Fail(path, name, "unsupported dtype " + ti.dtype);
    for (const int64_t dim : ti.shape) {
      if (dim < 0) Fail(path, name, "negative shape dimension");
      if (dim && bytes > std::numeric_limits<uint64_t>::max() / uint64_t(dim))
        Fail(path, name, "shape byte product overflow");
      bytes *= uint64_t(dim);
    }
    if (bytes != uint64_t(offsets[1] - offsets[0])) Fail(path, name, "shape and dtype do not match payload size");
  }
  void LoadHeader(const std::string& file) {
    if (maps_.count(file)) return;
    for (auto& [name, ti] : ParseHeader(file, dir_ + "/" + file))
      if (!tensors_.emplace(name, ti).second) Fail(dir_ + "/" + file, name, "duplicate tensor name across checkpoint shards");
  }
  // Maps path under the key file and returns its tensors in header order.
  std::vector<std::pair<std::string, TensorInfo>> ParseHeader(const std::string& file, const std::string& path) {
    std::vector<std::pair<std::string, TensorInfo>> found;
    std::unordered_set<std::string> names;
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) Fail(path, "", "cannot open shard: " + std::string(std::strerror(errno)));
    struct stat sb{};
    if (fstat(fd, &sb) != 0) { const int e = errno; close(fd); Fail(path, "", "fstat failed: " + std::string(std::strerror(e))); }
    if (!S_ISREG(sb.st_mode) || sb.st_size < 8) { close(fd); Fail(path, "", "shard is not a regular file or is shorter than 8 bytes"); }
    if (uint64_t(sb.st_size) > std::numeric_limits<size_t>::max()) { close(fd); Fail(path, "", "shard exceeds host address space"); }
    void* p = mmap(nullptr, size_t(sb.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    const int map_error = errno;
    close(fd);
    if (p == MAP_FAILED) Fail(path, "", "mmap failed: " + std::string(std::strerror(map_error)));
    // A second descriptor for small reads: POSIX_FADV_RANDOM on it leaves the mapping's readahead as it is.
    const int read_fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (read_fd < 0) {
      const int open_error = errno;
      munmap(p, size_t(sb.st_size));
      Fail(path, "", "cannot open shard: " + std::string(std::strerror(open_error)));
    }
    posix_fadvise(read_fd, 0, 0, POSIX_FADV_RANDOM);
    const Map m{static_cast<const uint8_t*>(p), size_t(sb.st_size), read_fd};
    maps_.emplace(file, m);
    uint64_t hlen = 0;
    for (int b = 0; b < 8; ++b) hlen |= uint64_t(m.base[b]) << (8 * b);
    if (hlen > m.size - 8) Fail(path, "", "header length exceeds file size");
    const uint64_t data0 = 8 + hlen;
    Scanner sc(std::string_view(reinterpret_cast<const char*>(m.base + 8), size_t(hlen)), path);
    sc.Expect('{');
    while (!sc.Peek('}')) {
      const std::string name = sc.Str();
      sc.Tensor(name);
      sc.Expect(':');
      if (name == "__metadata__") sc.Skip();
      else {
        TensorInfo ti;
        ti.file = file;
        std::vector<int64_t> offsets;
        unsigned fields = 0;
        sc.Expect('{');
        while (!sc.Peek('}')) {
          const std::string key = sc.Str();
          sc.Expect(':');
          const unsigned bit = key == "dtype" ? 1 : key == "shape" ? 2 : key == "data_offsets" ? 4 : 0;
          if (bit && (fields & bit)) Fail(path, name, "duplicate tensor field " + key);
          fields |= bit;
          if (key == "dtype") ti.dtype = sc.Str();
          else if (key == "shape") ti.shape = sc.IntArray();
          else if (key == "data_offsets") offsets = sc.IntArray();
          else sc.Skip();
          sc.Separator('}');
        }
        sc.Expect('}');
        if (fields != 7) Fail(path, name, "missing dtype, shape, or data_offsets");
        CheckTensor(path, name, ti, offsets, m.size - data0);
        ti.begin = data0 + uint64_t(offsets[0]);
        ti.end = data0 + uint64_t(offsets[1]);
        if (!names.insert(name).second) Fail(path, name, "duplicate tensor name in one file");
        found.emplace_back(name, ti);
      }
      sc.Tensor("");
      sc.Separator('}');
    }
    sc.Expect('}');
    sc.Finish();
    return found;
  }
  std::string dir_;
  std::unordered_map<std::string, std::string> shard_of_;
  std::unordered_map<std::string, TensorInfo> tensors_;
  std::map<std::string, Map> maps_;
  bool no_mapped_reads_ = false;
  std::unordered_map<std::string, std::vector<uint8_t>> small_;
};

// Validation reads small values only; while this guard lives, Data() fails.
class NoMappedReads {
 public:
  explicit NoMappedReads(Checkpoint& ck) : ck_(ck), before_(ck.no_mapped_reads_) { ck_.no_mapped_reads_ = true; }
  ~NoMappedReads() { ck_.no_mapped_reads_ = before_; }
  NoMappedReads(const NoMappedReads&) = delete;
  NoMappedReads& operator=(const NoMappedReads&) = delete;

 private:
  Checkpoint& ck_;
  bool before_;
};

}  // namespace st
