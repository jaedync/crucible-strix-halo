// The xgrammar 0.2.9 backend of structured output (constraint_vocab.h). The only engine file that includes the
// library. Library messages never reach a log: XGRAMMAR_LOG_CUSTOMIZE routes them here, warnings are dropped, and a
// fatal library error becomes a ConstraintError with a fixed message (no schema or output text).
//
// The grammar compiler runs only in the helper process (constraint_helper.h):
// - The helper's first process (ConstraintHelperMain) builds the vocabulary and forks one child for each compile.
//   The child sets its limits, compiles, checks a matcher, and writes the grammar in the binary form below.
// - This process (the engine) loads the binary form and runs the per-token matcher. It never compiles.
// The binary form is the upstream JSON form of the grammar (Grammar::SerializeJSON) and the token mask cache as
// arrays: the upstream JSON form of the mask cache is 8 times the compile time (10 MB for a small schema at the
// 248,320-token vocabulary). Loading checks every index as upstream DeserializeJSONValue does
// (compiled_grammar.cc:528) and rebuilds the derived repeat bitsets (PopulateRepeatInteriorBitsets).
#include "constraint_vocab.h"
#include <xgrammar/xgrammar.h>
#include <picojson.h>
// xgrammar internals (third_party/xgrammar/cpp, not changed): the mask cache of a compiled grammar.
#include <compiled_grammar_impl.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <list>
#include <malloc.h>
#include <map>
#include <mutex>
#include <poll.h>
#include <spawn.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <variant>

#if !XGRAMMAR_LOG_CUSTOMIZE
#error "build the constraint library with -DXGRAMMAR_LOG_CUSTOMIZE=1 (Makefile XGR_DEFINES)"
#endif

namespace xgrammar {
[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
  throw std::runtime_error(message);
}
void LogMessageImpl(const std::string&, int, int, const std::string&) {}
}  // namespace xgrammar

namespace crucible {
namespace {
constexpr int kCallOpen = 248058, kCallClose = 248059;  // <tool_call>, </tool_call>: text inside a grammar
using HelperClock = std::chrono::steady_clock;

[[noreturn]] void Refuse(HelperStatus status) { throw ConstraintError(HelperMessage(status)); }

// ---- The binary form of a compiled grammar ----
constexpr uint32_t kGrammarMagic = 0x31475843;  // "CXG1"

void Put(std::string& out, const void* data, size_t bytes) { out.append(static_cast<const char*>(data), bytes); }
template <class T> void PutValue(std::string& out, T value) { Put(out, &value, sizeof(value)); }
void PutInts(std::string& out, const std::vector<int32_t>& values) {
  PutValue<uint64_t>(out, values.size());
  Put(out, values.data(), values.size() * sizeof(int32_t));
}

class Reader {
 public:
  explicit Reader(std::string_view data) : data_(data) {}
  void Bytes(void* out, size_t bytes) {
    if (bytes > data_.size() - at_) throw std::runtime_error("short grammar");
    std::memcpy(out, data_.data() + at_, bytes);
    at_ += bytes;
  }
  template <class T> T Value() {
    T value;
    Bytes(&value, sizeof(value));
    return value;
  }
  // A count of `unit`-byte items: never more than the bytes left, so a bad count allocates nothing.
  size_t Count(size_t unit) {
    const uint64_t count = Value<uint64_t>();
    if (count > (data_.size() - at_) / unit) throw std::runtime_error("bad grammar count");
    return size_t(count);
  }
  std::vector<int32_t> Ints() {
    std::vector<int32_t> values(Count(sizeof(int32_t)));
    Bytes(values.data(), values.size() * sizeof(int32_t));
    return values;
  }
  std::string Text() {
    std::string text(Count(1), '\0');
    Bytes(text.data(), text.size());
    return text;
  }
  bool Done() const { return at_ == data_.size(); }

 private:
  std::string_view data_;
  size_t at_ = 0;
};

std::string EncodeGrammar(const xgrammar::CompiledGrammar& compiled) {
  const auto* impl = compiled.ImplPtr();
  std::string out;
  PutValue(out, kGrammarMagic);
  const std::string grammar = impl->grammar.SerializeJSON();
  PutValue<uint64_t>(out, grammar.size());
  out += grammar;
  PutValue<uint64_t>(out, impl->adaptive_token_mask_cache.size());
  for (const auto& [state, mask] : impl->adaptive_token_mask_cache) {
    const int32_t fields[10] = {state.rule_id, state.sequence_id, state.element_id, state.rule_start_pos,
                                state.budget_deadline, state.sub_element_id, state.repeat_count,
                                state.partial_codepoint, state.active_temperature_rule_id, state.char_budget_deadline};
    Put(out, fields, sizeof(fields));
    PutValue<uint32_t>(out, uint32_t(mask.store_type));
    PutInts(out, mask.accepted_indices);
    PutInts(out, mask.rejected_indices);
    PutInts(out, mask.uncertain_indices);
    const int bits = mask.accepted_bitset.Size();
    std::vector<uint32_t> words(size_t(xgrammar::DynamicBitset::GetBufferSize(bits)), 0);
    if (bits > 0) {
      xgrammar::DynamicBitset view(bits, words.data());
      view |= mask.accepted_bitset;
    }
    PutValue<int32_t>(out, bits);
    Put(out, words.data(), words.size() * sizeof(uint32_t));
  }
  return out;
}

xgrammar::CompiledGrammar DecodeGrammar(std::string_view data, const xgrammar::TokenizerInfo& info) {
  Reader in(data);
  if (in.Value<uint32_t>() != kGrammarMagic) throw std::runtime_error("bad grammar magic");
  auto parsed = xgrammar::Grammar::DeserializeJSON(in.Text());
  if (!std::holds_alternative<xgrammar::Grammar>(parsed)) throw std::runtime_error("bad grammar");
  auto impl = std::make_shared<xgrammar::CompiledGrammar::Impl>();
  impl->grammar = std::get<xgrammar::Grammar>(std::move(parsed));
  if (impl->grammar.IsNull()) throw std::runtime_error("null grammar");
  impl->tokenizer_info = info;
  const int64_t sorted = int64_t(info.GetSortedDecodedVocab().size());
  const int rules = impl->grammar->NumRules();
  const auto check = [&](const std::vector<int32_t>& indices) {
    for (const int32_t index : indices)
      if (index < 0 || index >= sorted) throw std::runtime_error("bad mask index");
  };
  using StoreType = xgrammar::AdaptiveTokenMask::StoreType;
  const size_t masks = in.Count(10 * sizeof(int32_t));
  for (size_t m = 0; m < masks; ++m) {
    int32_t f[10];
    in.Bytes(f, sizeof(f));
    const xgrammar::ParserState state(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]);
    if (state.rule_id < 0 || state.rule_id >= rules) throw std::runtime_error("bad mask rule");
    xgrammar::AdaptiveTokenMask mask;
    const uint32_t store = in.Value<uint32_t>();
    if (store > uint32_t(StoreType::kAcceptedBitset)) throw std::runtime_error("bad mask store");
    mask.store_type = StoreType(store);
    mask.accepted_indices = in.Ints();
    mask.rejected_indices = in.Ints();
    mask.uncertain_indices = in.Ints();
    check(mask.accepted_indices);
    check(mask.rejected_indices);
    check(mask.uncertain_indices);
    const int32_t bits = in.Value<int32_t>();
    if (bits != 0 && bits != info.GetVocabSize()) throw std::runtime_error("bad mask bitset");
    if (mask.store_type == StoreType::kAcceptedBitset && bits != info.GetVocabSize())
      throw std::runtime_error("bad mask bitset");
    if (bits > 0) {
      std::vector<uint32_t> words(size_t(xgrammar::DynamicBitset::GetBufferSize(bits)));
      in.Bytes(words.data(), words.size() * sizeof(uint32_t));
      mask.accepted_bitset = xgrammar::DynamicBitset(bits);
      mask.accepted_bitset |= xgrammar::DynamicBitset(bits, words.data());
    }
    if (!impl->adaptive_token_mask_cache.emplace(state, std::move(mask)).second)
      throw std::runtime_error("duplicate mask state");
  }
  if (!in.Done()) throw std::runtime_error("grammar has extra bytes");
  xgrammar::PopulateRepeatInteriorBitsets(impl->grammar, info, &impl->adaptive_token_mask_cache);
  return xgrammar::CompiledGrammar(std::move(impl));
}

// ---- The matcher ----
class XgrammarState final : public GrammarState {
 public:
  XgrammarState(const xgrammar::CompiledGrammar& grammar, int rows, std::shared_ptr<const std::vector<uint8_t>> special)
      : matcher_(grammar, std::nullopt, false, kConstraintRollback), rows_(rows), special_(std::move(special)) {}
  bool Accept(int token) override {
    // The library logs and refuses these; refuse them here, before any library call.
    if (token < 0 || token >= rows_ || (*special_)[size_t(token)] || matcher_.IsTerminated()) return false;
    return matcher_.AcceptToken(token);
  }
  void Rollback(int tokens) override { if (tokens > 0) matcher_.Rollback(tokens); }
  void FillMask(std::span<uint32_t> words) override {
    const int32_t size = xgrammar::GetBitmaskSize(rows_);
    if (words.size() != size_t(size)) throw std::logic_error("constraint mask has the wrong size");
    int64_t shape[1] = {size};
    DLTensor tensor{};
    tensor.data = words.data();
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = shape;
    matcher_.FillNextTokenBitmask(&tensor);
  }
  bool Terminated() const override { return matcher_.IsTerminated(); }

 private:
  xgrammar::GrammarMatcher matcher_;
  int rows_;
  std::shared_ptr<const std::vector<uint8_t>> special_;
};

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw ConstraintError("constraint_vocab: the tokenizer file does not open");
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

std::vector<uint8_t> SpecialRows(const std::vector<std::string>& tokens, int rows, int stop) {
  std::vector<uint8_t> special(size_t(rows), 1);
  for (size_t i = 0; i < tokens.size() && i < size_t(rows); ++i) special[i] = tokens[i].empty();
  if (stop >= 0 && stop < rows) special[size_t(stop)] = 0;
  return special;
}
// The phase prefix of a reasoning request: the template writes up to two newlines after </think>.
xgrammar::Grammar ReasoningPrefix() { return xgrammar::Grammar::FromEBNF("root ::= [ \\t\\n]{0,4}"); }

uint32_t KindCode(const std::string& kind) {
  return kind == "json_schema" ? 0 : kind == "json_object" ? 1 : kind == "structural_tag" ? 2 : 3;
}

// The compile itself (only in a helper child). Throws ConstraintError with a fixed message.
xgrammar::CompiledGrammar Compile(xgrammar::GrammarCompiler& compiler, uint32_t kind, const std::string& spec,
                                  bool reasoning) {
  std::optional<xgrammar::Grammar> grammar;
  if (kind == 0 || kind == 1) {
    if (kind == 1 && !spec.empty()) Refuse(HelperStatus::kSpec);
    const std::string schema = kind == 1 ? std::string(R"({"type":"object"})") : spec;
    try {
      grammar = xgrammar::Grammar::FromJSONSchema(schema, true, std::nullopt, std::nullopt, true, kConstraintMaxWhitespace);
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception&) {
      Refuse(HelperStatus::kSchema);
    }
  } else if (kind == 2) {
    try {
      auto result = xgrammar::Grammar::FromStructuralTag(spec);
      if (std::holds_alternative<xgrammar::StructuralTagError>(result)) Refuse(HelperStatus::kTag);
      grammar = std::get<xgrammar::Grammar>(std::move(result));
    } catch (const ConstraintError&) {
      throw;
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception&) {
      Refuse(HelperStatus::kTag);
    }
  } else {
    Refuse(HelperStatus::kKind);
  }
  try {
    if (reasoning) grammar = xgrammar::Grammar::Concat({ReasoningPrefix(), *grammar});
    return compiler.CompileGrammar(*grammar);
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception&) {
    Refuse(kind == 2 ? HelperStatus::kTag : HelperStatus::kSchema);
  }
}

// The child's matcher check: the first mask, then up to kHelperWalkTokens legal tokens (the lowest legal ID that is
// not the stop, else the stop), then a rollback and the mask again. A fault here stays in the child.
void CheckMatcher(const xgrammar::CompiledGrammar& compiled, int rows, int stop, const std::vector<uint8_t>& special) {
  auto special_rows = std::make_shared<const std::vector<uint8_t>>(special);
  XgrammarState state(compiled, rows, special_rows);
  std::vector<uint32_t> words(size_t(xgrammar::GetBitmaskSize(rows)));
  int accepted = 0;
  for (int step = 0; step < kHelperWalkTokens && !state.Terminated(); ++step) {
    state.FillMask(words);
    int pick = -1;
    bool stop_legal = false;
    for (int t = 0; t < rows && pick < 0; ++t) {
      if (!(words[size_t(t) / 32] >> (t % 32) & 1u) || special[size_t(t)]) continue;
      if (t == stop) stop_legal = true;
      else pick = t;
    }
    if (pick < 0 && stop_legal) pick = stop;
    if (pick < 0) break;
    if (!state.Accept(pick)) throw std::runtime_error("matcher refused a legal token");
    ++accepted;
  }
  state.Rollback(std::min(accepted, kConstraintRollback));
  state.FillMask(words);
}

// ---- Socket and pipe IO ----
int MsLeft(HelperClock::time_point deadline) {
  const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - HelperClock::now()).count();
  return left <= 0 ? 0 : int(std::min<int64_t>(left, 1 << 30));
}
bool WaitFd(int fd, short events, HelperClock::time_point deadline) {
  for (;;) {
    pollfd p{fd, events, 0};
    const int left = MsLeft(deadline);
    const int rc = poll(&p, 1, left);
    if (rc > 0) return true;
    if (rc == 0) return false;
    if (errno != EINTR) return false;
  }
}
// With a deadline: the engine side (never blocks past it; MSG_NOSIGNAL, so a dead helper is no SIGPIPE).
bool SendAll(int fd, const void* data, size_t bytes, HelperClock::time_point deadline) {
  const char* at = static_cast<const char*>(data);
  while (bytes) {
    if (!WaitFd(fd, POLLOUT, deadline)) return false;
    const ssize_t sent = send(fd, at, bytes, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (sent < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (sent <= 0) return false;
    at += sent;
    bytes -= size_t(sent);
  }
  return true;
}
bool RecvAll(int fd, void* data, size_t bytes, HelperClock::time_point deadline) {
  char* at = static_cast<char*>(data);
  while (bytes) {
    if (!WaitFd(fd, POLLIN, deadline)) return false;
    const ssize_t got = read(fd, at, bytes);
    if (got < 0 && (errno == EINTR || errno == EAGAIN)) continue;
    if (got <= 0) return false;
    at += got;
    bytes -= size_t(got);
  }
  return true;
}
// Blocking: the helper side (the engine closes the socket to stop it).
bool WriteAll(int fd, const void* data, size_t bytes) {
  const char* at = static_cast<const char*>(data);
  while (bytes) {
    const ssize_t sent = write(fd, at, bytes);
    if (sent < 0 && errno == EINTR) continue;
    if (sent <= 0) return false;
    at += sent;
    bytes -= size_t(sent);
  }
  return true;
}
bool ReadAll(int fd, void* data, size_t bytes) {
  char* at = static_cast<char*>(data);
  while (bytes) {
    const ssize_t got = read(fd, at, bytes);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) return false;
    at += got;
    bytes -= size_t(got);
  }
  return true;
}
template <class T> bool ReadValue(int fd, T& value) { return ReadAll(fd, &value, sizeof(value)); }

constexpr uint32_t kInitMagic = 0x494758ab, kReadyMagic = 0x524758ab, kRequestMagic = 0x514758ab,
                   kReplyMagic = 0x504758ab;
constexpr uint64_t kMaxInitTokens = uint64_t(1) << 22, kMaxTokenBytes = uint64_t(1) << 16;
constexpr int kExitMemory = 3;

struct ReplyHeader {
  uint32_t magic = kReplyMagic;
  uint32_t status = 0;
  uint32_t wall_ms = 0;
  uint32_t pad = 0;
  uint64_t max_rss_kb = 0;
  uint64_t bytes = 0;
};

// ---- The helper child: one compile under its limits ----
uint64_t AddressSpaceBytes() {
  std::ifstream statm("/proc/self/statm");
  uint64_t pages = 0;
  statm >> pages;
  return pages * uint64_t(sysconf(_SC_PAGESIZE));
}

[[noreturn]] void CompileChild(const xgrammar::TokenizerInfo& info, int rows, int stop,
                               const std::vector<uint8_t>& special, uint32_t kind, bool reasoning,
                               const std::string& spec, int out) {
  prctl(PR_SET_PDEATHSIG, SIGKILL);
  prctl(PR_SET_DUMPABLE, 0);  // no core file, no ptrace
  rlimit core{0, 0};
  setrlimit(RLIMIT_CORE, &core);
  rlimit stack{};
  getrlimit(RLIMIT_STACK, &stack);
  if (stack.rlim_cur == RLIM_INFINITY || stack.rlim_cur > kHelperStackBytes) {
    stack.rlim_cur = kHelperStackBytes;
    setrlimit(RLIMIT_STACK, &stack);
  }
  rlimit cpu{rlim_t(kHelperCpuSeconds), rlim_t(kHelperCpuSeconds + 1)};
  setrlimit(RLIMIT_CPU, &cpu);
  mallopt(M_ARENA_MAX, 2);  // few arenas, so the address-space limit measures memory, not reservations
  rlimit memory{rlim_t(AddressSpaceBytes() + kHelperMemoryBytes), rlim_t(AddressSpaceBytes() + kHelperMemoryBytes)};
  setrlimit(RLIMIT_AS, &memory);
  uint32_t status = uint32_t(HelperStatus::kOk);
  std::string payload;
  try {
    xgrammar::GrammarCompiler compiler(info, 8, false);
    const auto compiled = Compile(compiler, kind, spec, reasoning);
    try {
      CheckMatcher(compiled, rows, stop, special);
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception&) {
      status = uint32_t(HelperStatus::kMatcher);
    }
    if (status == uint32_t(HelperStatus::kOk)) {
      payload = EncodeGrammar(compiled);
      if (payload.size() > kMaxGrammarBytes) {
        status = uint32_t(HelperStatus::kTooBig);
        payload.clear();
      }
    }
  } catch (const std::bad_alloc&) {
    _exit(kExitMemory);
  } catch (const ConstraintError& error) {
    status = uint32_t(HelperStatus::kSchema);
    for (uint32_t s = 1; s <= uint32_t(HelperStatus::kLoad); ++s)
      if (std::strcmp(error.what(), HelperMessage(HelperStatus(s))) == 0) status = s;
  } catch (const std::exception&) {
    status = uint32_t(kind == 2 ? HelperStatus::kTag : HelperStatus::kSchema);
  }
  const uint64_t bytes = payload.size();
  if (!WriteAll(out, &status, sizeof(status)) || !WriteAll(out, &bytes, sizeof(bytes)) ||
      !WriteAll(out, payload.data(), payload.size()))
    _exit(1);
  _exit(0);
}

// The helper's first process: runs one child, classifies its end, and reads its result within the deadline.
ReplyHeader RunChild(const xgrammar::TokenizerInfo& info, int rows, int stop, const std::vector<uint8_t>& special,
                     int engine_fd, uint32_t kind, bool reasoning, const std::string& spec, std::string& payload) {
  ReplyHeader reply;
  payload.clear();
  const auto start = HelperClock::now();
  const auto deadline = start + std::chrono::milliseconds(kHelperDeadlineMs);
  int pipe_fds[2];
  if (pipe2(pipe_fds, O_CLOEXEC) != 0) {
    reply.status = uint32_t(HelperStatus::kHelper);
    return reply;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    reply.status = uint32_t(HelperStatus::kHelper);
    return reply;
  }
  if (pid == 0) {
    close(engine_fd);
    close(pipe_fds[0]);
    CompileChild(info, rows, stop, special, kind, reasoning, spec, pipe_fds[1]);
  }
  close(pipe_fds[1]);
  // The child's frame: status, length, payload. Read it all before the child ends (a pipe holds 64 KiB).
  std::string frame;
  bool timed_out = false, too_big = false;
  char buffer[1 << 16];
  for (;;) {
    if (!WaitFd(pipe_fds[0], POLLIN, deadline)) { timed_out = true; break; }
    const ssize_t got = read(pipe_fds[0], buffer, sizeof(buffer));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) break;
    frame.append(buffer, size_t(got));
    if (frame.size() > kMaxGrammarBytes + 12) { too_big = true; break; }
  }
  close(pipe_fds[0]);
  if (timed_out || too_big) kill(pid, SIGKILL);
  int wait_status = 0;
  rusage usage{};
  while (wait4(pid, &wait_status, 0, &usage) < 0 && errno == EINTR) {}
  reply.wall_ms = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(HelperClock::now() - start).count());
  reply.max_rss_kb = uint64_t(usage.ru_maxrss);
  HelperStatus status = HelperStatus::kCrash;
  if (timed_out) status = HelperStatus::kTime;
  else if (too_big) status = HelperStatus::kTooBig;
  else if (WIFSIGNALED(wait_status))
    status = WTERMSIG(wait_status) == SIGXCPU || WTERMSIG(wait_status) == SIGKILL ? HelperStatus::kTime
                                                                                 : HelperStatus::kCrash;
  else if (WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == kExitMemory) status = HelperStatus::kMemory;
  else if (WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0 && frame.size() >= 12) {
    uint32_t child_status;
    uint64_t bytes;
    std::memcpy(&child_status, frame.data(), 4);
    std::memcpy(&bytes, frame.data() + 4, 8);
    if (bytes == frame.size() - 12 && child_status <= uint32_t(HelperStatus::kLoad)) {
      status = HelperStatus(child_status);
      if (status == HelperStatus::kOk) payload = frame.substr(12);
    }
  }
  if (status == HelperStatus::kCrash && usage.ru_maxrss > 0 &&
      uint64_t(usage.ru_maxrss) * 1024 > kHelperMemoryBytes - (kHelperMemoryBytes >> 4))
    status = HelperStatus::kMemory;  // an allocation failure on a worker thread ends in abort
  reply.status = uint32_t(status);
  reply.bytes = payload.size();
  return reply;
}

// ---- The engine side of the helper ----
int HelperImageFd() {
  static int fd = [] {
    const auto image = ConstraintHelperImage();
    if (image.empty()) return -1;
    const int memfd = memfd_create("crucible-grammar-helper", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (memfd < 0) return -1;
    if (!WriteAll(memfd, image.data(), image.size()) ||
        fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL) != 0) {
      close(memfd);
      return -1;
    }
    return memfd;
  }();
  return fd;
}

class HelperClient {
 public:
  explicit HelperClient(std::string init) : init_(std::move(init)) {}
  ~HelperClient() { Stop(); }
  HelperClient(const HelperClient&) = delete;
  HelperClient& operator=(const HelperClient&) = delete;

  // One compile. Returns the reply (status kHelper when the helper does not start or answer).
  ReplyHeader Compile(uint32_t kind, bool reasoning, const std::string& spec, std::string& payload) {
    std::lock_guard lock(mutex_);
    payload.clear();
    ReplyHeader failed;
    failed.status = uint32_t(HelperStatus::kHelper);
    if (fd_ < 0 && !Start()) return failed;
    const auto deadline = HelperClock::now() + std::chrono::milliseconds(kHelperDeadlineMs + 5000);
    const uint32_t head[3] = {kRequestMagic, kind, uint32_t(reasoning)};
    const uint64_t bytes = spec.size();
    ReplyHeader reply;
    if (!SendAll(fd_, head, sizeof(head), deadline) || !SendAll(fd_, &bytes, sizeof(bytes), deadline) ||
        !SendAll(fd_, spec.data(), spec.size(), deadline) || !RecvAll(fd_, &reply, sizeof(reply), deadline) ||
        reply.magic != kReplyMagic || reply.bytes > kMaxGrammarBytes ||
        (reply.bytes && reply.status != uint32_t(HelperStatus::kOk))) {
      Stop();
      return failed;
    }
    payload.resize(size_t(reply.bytes));
    if (!RecvAll(fd_, payload.data(), payload.size(), deadline)) {
      Stop();
      payload.clear();
      return failed;
    }
    return reply;
  }

 private:
  bool Start() {
    const int image = HelperImageFd();
    if (image < 0) return false;
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, pair) != 0) return false;
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    posix_spawn_file_actions_init(&actions);
    posix_spawnattr_init(&attr);
    // fd 0: the socket; 1 and 2: /dev/null (library text never reaches a log); 3: the image; nothing else.
    posix_spawn_file_actions_adddup2(&actions, pair[1], 0);
    posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, image, 3);
    posix_spawn_file_actions_addclosefrom_np(&actions, 4);
    sigset_t none, all;
    sigemptyset(&none);
    sigfillset(&all);
    posix_spawnattr_setsigmask(&attr, &none);
    posix_spawnattr_setsigdefault(&attr, &all);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
    char name[] = "crucible-grammar-helper";
    char* argv[] = {name, nullptr};
    char* envp[] = {nullptr};
    pid_t pid = -1;
    const int rc = posix_spawn(&pid, "/proc/self/fd/3", &actions, &attr, argv, envp);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    close(pair[1]);
    if (rc != 0) {
      close(pair[0]);
      return false;
    }
    fd_ = pair[0];
    pid_ = pid;
    const auto deadline = HelperClock::now() + std::chrono::milliseconds(kHelperStartMs);
    uint32_t ready[2] = {0, 1};
    if (!SendAll(fd_, init_.data(), init_.size(), deadline) || !RecvAll(fd_, ready, sizeof(ready), deadline) ||
        ready[0] != kReadyMagic || ready[1] != 0) {
      Stop();
      return false;
    }
    return true;
  }
  void Stop() {
    if (fd_ >= 0) close(fd_);
    fd_ = -1;
    if (pid_ > 0) {
      kill(pid_, SIGKILL);
      while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
    }
    pid_ = -1;
  }

  std::mutex mutex_;
  std::string init_;
  int fd_ = -1;
  pid_t pid_ = -1;
};

// The helper's start message: the vocabulary as the helper builds it (the same TokenizerInfo arguments).
std::string InitMessage(const std::vector<std::string>& tokens, bool byte_level, int rows, int stop) {
  std::string out;
  PutValue(out, kInitMagic);
  PutValue<uint32_t>(out, byte_level ? 1 : 0);
  PutValue<int32_t>(out, rows);
  PutValue<int32_t>(out, stop);
  PutValue<uint64_t>(out, tokens.size());
  for (const auto& token : tokens) {
    PutValue<uint64_t>(out, token.size());
    out += token;
  }
  return out;
}
}  // namespace

// ---- The helper's main loop ----
int ConstraintHelperMain(int fd) {
  prctl(PR_SET_PDEATHSIG, SIGKILL);
  signal(SIGPIPE, SIG_IGN);
  close(3);  // the image
  uint32_t magic = 0, byte_level = 0;
  int32_t rows = 0, stop = 0;
  uint64_t count = 0;
  if (!ReadValue(fd, magic) || magic != kInitMagic || !ReadValue(fd, byte_level) || !ReadValue(fd, rows) ||
      !ReadValue(fd, stop) || !ReadValue(fd, count) || count > kMaxInitTokens || rows < int64_t(count) || stop < 0 ||
      stop >= rows)
    return 1;
  std::vector<std::string> tokens(static_cast<size_t>(count));
  for (auto& token : tokens) {
    uint64_t bytes = 0;
    if (!ReadValue(fd, bytes) || bytes > kMaxTokenBytes) return 1;
    token.resize(size_t(bytes));
    if (!ReadAll(fd, token.data(), token.size())) return 1;
  }
  std::optional<xgrammar::TokenizerInfo> info;
  uint32_t ready[2] = {kReadyMagic, 0};
  try {
    info.emplace(tokens, byte_level ? xgrammar::VocabType::BYTE_LEVEL : xgrammar::VocabType::RAW, rows,
                 std::vector<int32_t>{stop}, false);
  } catch (const std::exception&) {
    ready[1] = 1;
  }
  if (!WriteAll(fd, ready, sizeof(ready)) || ready[1] != 0) return 1;
  const auto special = SpecialRows(tokens, rows, stop);
  tokens.clear();
  tokens.shrink_to_fit();
  std::string spec, payload;
  for (;;) {
    uint32_t head[3];
    uint64_t bytes = 0;
    if (!ReadAll(fd, head, sizeof(head)) || head[0] != kRequestMagic || !ReadValue(fd, bytes) || bytes > kMaxSpecBytes)
      return 0;
    spec.resize(size_t(bytes));
    if (!ReadAll(fd, spec.data(), spec.size())) return 0;
    const ReplyHeader reply = RunChild(*info, rows, stop, special, fd, head[1], head[2] != 0, spec, payload);
    if (!WriteAll(fd, &reply, sizeof(reply)) || !WriteAll(fd, payload.data(), payload.size())) return 0;
  }
}

// ---- The vocabulary, its cache and MakeConstraint ----
struct ConstraintVocab {
  ConstraintVocab(xgrammar::TokenizerInfo info, int rows, int stop, std::vector<uint8_t> special, std::string init)
      : info(std::move(info)), rows(rows), stop(stop),
        special(std::make_shared<const std::vector<uint8_t>>(std::move(special))), helper(std::move(init)) {}
  xgrammar::TokenizerInfo info;
  int rows, stop;
  // Tokens the grammar never takes (empty texts: added tokens and pad rows), except the stop token.
  std::shared_ptr<const std::vector<uint8_t>> special;
  HelperClient helper;
  mutable std::mutex mutex;  // the cache and the last run
  using Key = std::tuple<std::string, std::string, bool>;
  struct Entry {
    Key key;
    xgrammar::CompiledGrammar grammar;
    size_t bytes;
  };
  std::list<Entry> lru;
  std::map<Key, std::list<Entry>::iterator> index;
  size_t cache_bytes = 0;
  HelperRun last;

  std::optional<xgrammar::CompiledGrammar> Find(const Key& key) {
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    if (found == index.end()) return std::nullopt;
    lru.splice(lru.begin(), lru, found->second);
    return found->second->grammar;
  }
  // Keeps at most kGrammarCacheEntries grammars and kGrammarCacheBytes bytes (the newest always stays).
  void Insert(const Key& key, const xgrammar::CompiledGrammar& grammar) {
    std::lock_guard lock(mutex);
    if (index.contains(key)) return;
    const size_t bytes = grammar.MemorySizeBytes();
    lru.push_front(Entry{key, grammar, bytes});
    index[key] = lru.begin();
    cache_bytes += bytes;
    while (lru.size() > 1 && (lru.size() > kGrammarCacheEntries || cache_bytes > kGrammarCacheBytes)) {
      cache_bytes -= lru.back().bytes;
      index.erase(lru.back().key);
      lru.pop_back();
    }
  }
  // Compiles `spec` in the helper and caches the grammar. Throws ConstraintError.
  xgrammar::CompiledGrammar CompileAndCache(const ConstraintSpec& spec) {
    if (spec.spec.size() > kMaxSpecBytes) throw ConstraintError("constraint_spec: the spec exceeds kMaxSpecBytes = 1 MiB");
    const uint32_t kind = KindCode(spec.kind);
    if (kind == 3) Refuse(HelperStatus::kKind);
    std::string payload;
    const ReplyHeader reply = helper.Compile(kind, spec.reasoning, spec.spec, payload);
    HelperStatus status = HelperStatus(reply.status);
    std::optional<xgrammar::CompiledGrammar> grammar;
    if (status == HelperStatus::kOk) {
      try {
        grammar = DecodeGrammar(payload, info);
      } catch (const std::exception&) {
        status = HelperStatus::kLoad;
      }
    }
    {
      std::lock_guard lock(mutex);
      last = HelperRun{status, double(reply.wall_ms), reply.max_rss_kb};
    }
    if (status != HelperStatus::kOk) Refuse(status);
    Insert(Key{spec.kind, spec.spec, spec.reasoning}, *grammar);
    return *grammar;
  }
  std::shared_ptr<Constraint> Open(const xgrammar::CompiledGrammar& grammar, const ConstraintSpec& spec, int think_end) {
    std::unique_ptr<GrammarState> state;
    try {
      state = std::make_unique<XgrammarState>(grammar, rows, special);
    } catch (const std::exception&) {
      Refuse(HelperStatus::kMatcher);
    }
    return std::make_shared<Constraint>(std::move(state), rows, stop, spec.reasoning, think_end);
  }
};

namespace {
std::shared_ptr<ConstraintVocab> Build(const std::vector<std::string>& tokens, int rows, int stop, bool byte_level) {
  if (rows < int(tokens.size()) || stop < 0 || stop >= rows) throw ConstraintError("constraint_vocab: bad vocabulary size or stop token");
  try {
    xgrammar::TokenizerInfo info(tokens, byte_level ? xgrammar::VocabType::BYTE_LEVEL : xgrammar::VocabType::RAW,
                                 rows, std::vector<int32_t>{stop}, false);
    return std::make_shared<ConstraintVocab>(std::move(info), rows, stop, SpecialRows(tokens, rows, stop),
                                             InitMessage(tokens, byte_level, rows, stop));
  } catch (const ConstraintError&) {
    throw;
  } catch (const std::exception&) {
    throw ConstraintError("constraint_vocab: the library refused the vocabulary");
  }
}
}  // namespace

std::shared_ptr<ConstraintVocab> LoadConstraintVocab(const std::string& path, int rows, int stop) {
  return LoadConstraintVocab(path, ConstraintFamily{rows, stop, kConstraintThinkEnd, {kCallOpen, kCallClose}});
}

std::shared_ptr<ConstraintVocab> LoadConstraintVocab(const std::string& path, const ConstraintFamily& family) {
  const int rows = family.rows, stop = family.stop;
  const std::string text = ReadFile(path);
  picojson::value root;
  const std::string error = picojson::parse(root, text);
  if (!error.empty() || !root.is<picojson::object>()) throw ConstraintError("constraint_vocab: the tokenizer file is not JSON");
  const auto& object = root.get<picojson::object>();
  const auto model = object.find("model");
  if (model == object.end() || !model->second.is<picojson::object>()) throw ConstraintError("constraint_vocab: no model vocabulary");
  const auto& fields = model->second.get<picojson::object>();
  const auto vocab = fields.find("vocab");
  if (vocab == fields.end() || !vocab->second.is<picojson::object>()) throw ConstraintError("constraint_vocab: no model vocabulary");
  std::vector<std::string> tokens(static_cast<size_t>(rows));
  for (const auto& [token, id] : vocab->second.get<picojson::object>()) {
    if (!id.is<int64_t>() || id.get<int64_t>() < 0 || id.get<int64_t>() >= rows) throw ConstraintError("constraint_vocab: a token ID is out of range");
    tokens[size_t(id.get<int64_t>())] = token;
  }
  // Added tokens: only the family's text tokens (the call tags) are text; the stop token keeps its content (the
  // library marks it a stop).
  if (const auto added = object.find("added_tokens"); added != object.end() && added->second.is<picojson::array>())
    for (const auto& entry : added->second.get<picojson::array>()) {
      if (!entry.is<picojson::object>()) continue;
      const auto& fields = entry.get<picojson::object>();
      const auto id = fields.find("id"), content = fields.find("content");
      if (id == fields.end() || content == fields.end() || !id->second.is<int64_t>() || !content->second.is<std::string>()) continue;
      const int64_t value = id->second.get<int64_t>();
      if (value < 0 || value >= rows) throw ConstraintError("constraint_vocab: an added token ID is out of range");
      const bool text_token = std::find(family.text_tokens.begin(), family.text_tokens.end(), value) !=
                              family.text_tokens.end();
      tokens[size_t(value)] = text_token || value == stop ? content->second.get<std::string>() : "";
    }
  return Build(tokens, rows, stop, true);
}

std::shared_ptr<ConstraintVocab> MakeConstraintVocab(const std::vector<std::string>& tokens, int stop) {
  return Build(tokens, int(tokens.size()), stop, false);
}

int ConstraintVocabRows(const ConstraintVocab& vocab) { return vocab.rows; }

HelperRun LastHelperRun(const ConstraintVocab& vocab) {
  std::lock_guard lock(vocab.mutex);
  return vocab.last;
}
size_t ConstraintCacheBytes(const ConstraintVocab& vocab) {
  std::lock_guard lock(vocab.mutex);
  return vocab.cache_bytes;
}
size_t ConstraintCacheEntries(const ConstraintVocab& vocab) {
  std::lock_guard lock(vocab.mutex);
  return vocab.lru.size();
}

std::shared_ptr<Constraint> MakeConstraint(ConstraintVocab& vocab, const ConstraintSpec& spec, int think_end) {
  const auto start = HelperClock::now();
  const ConstraintVocab::Key key{spec.kind, spec.spec, spec.reasoning};
  int hit = 1;
  auto grammar = vocab.Find(key);
  if (!grammar) {
    hit = 0;
    grammar = vocab.CompileAndCache(spec);
  }
  auto constraint = vocab.Open(*grammar, spec, think_end);
  constraint->counts.cache_hit = hit;
  constraint->counts.compile_us = std::chrono::duration<double, std::micro>(HelperClock::now() - start).count();
  return constraint;
}

// ---- The serve path: compiles on a worker thread ----
class ConstraintService {
 public:
  explicit ConstraintService(std::function<std::shared_ptr<ConstraintVocab>()> load) : load_(std::move(load)) {}
  ~ConstraintService() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  CompileStatus Compile(const ConstraintSpec& spec) {
    const ConstraintVocab::Key key{spec.kind, spec.spec, spec.reasoning};
    std::lock_guard lock(mutex_);
    if (vocab_ && vocab_->Find(key)) return {CompileState::kReady, ""};
    if (const auto refused = refused_index_.find(key); refused != refused_index_.end())
      return {CompileState::kRefused, refused->second->second};
    if (std::find(queue_.begin(), queue_.end(), key) != queue_.end() || (running_ && *running_ == key))
      return {CompileState::kCompiling, ""};
    if (queue_.size() >= kCompileQueue)
      return {CompileState::kRefused, kBusyMessage};
    queue_.push_back(key);
    if (!worker_.joinable()) worker_ = std::thread([this] { Work(); });
    wake_.notify_all();
    return {CompileState::kCompiling, ""};
  }

  std::shared_ptr<Constraint> Open(const ConstraintSpec& spec, int think_end) {
    const ConstraintVocab::Key key{spec.kind, spec.spec, spec.reasoning};
    std::shared_ptr<ConstraintVocab> vocab;
    {
      std::lock_guard lock(mutex_);
      if (const auto refused = refused_index_.find(key); refused != refused_index_.end())
        throw ConstraintError(refused->second->second);
      vocab = vocab_;
    }
    const auto grammar = vocab ? vocab->Find(key) : std::nullopt;
    if (!grammar) throw ConstraintError(kPendingMessage);
    auto constraint = vocab->Open(*grammar, spec, think_end);
    constraint->counts.cache_hit = 1;
    return constraint;
  }

 private:
  void Work() {
    for (;;) {
      ConstraintVocab::Key key;
      std::shared_ptr<ConstraintVocab> vocab;
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (stopping_) return;
        key = queue_.front();
        queue_.pop_front();
        running_ = key;
        vocab = vocab_;
      }
      std::string refusal;
      try {
        if (!vocab) vocab = load_();  // the first compile loads the vocabulary here, off the serve loop
        vocab->CompileAndCache(ConstraintSpec{std::get<0>(key), std::get<1>(key), std::get<2>(key)});
      } catch (const ConstraintError& error) {
        refusal = error.what();
      } catch (const std::exception&) {
        refusal = HelperMessage(HelperStatus::kHelper);
      }
      std::lock_guard lock(mutex_);
      if (vocab && !vocab_) vocab_ = vocab;
      running_.reset();
      if (!refusal.empty() && !refused_index_.contains(key)) {
        refused_.emplace_front(key, refusal);
        refused_index_[key] = refused_.begin();
        if (refused_.size() > kRefusalCacheEntries) {
          refused_index_.erase(refused_.back().first);
          refused_.pop_back();
        }
      }
    }
  }

  std::function<std::shared_ptr<ConstraintVocab>()> load_;
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stopping_ = false;
  std::shared_ptr<ConstraintVocab> vocab_;
  std::deque<ConstraintVocab::Key> queue_;
  std::optional<ConstraintVocab::Key> running_;
  std::list<std::pair<ConstraintVocab::Key, std::string>> refused_;
  std::map<ConstraintVocab::Key, std::list<std::pair<ConstraintVocab::Key, std::string>>::iterator> refused_index_;
  std::thread worker_;
};

std::shared_ptr<ConstraintService> StartConstraintService(std::function<std::shared_ptr<ConstraintVocab>()> load) {
  return std::make_shared<ConstraintService>(std::move(load));
}
CompileStatus CompileConstraint(ConstraintService& service, const ConstraintSpec& spec) { return service.Compile(spec); }
std::shared_ptr<Constraint> OpenCompiledConstraint(ConstraintService& service, const ConstraintSpec& spec, int think_end) {
  return service.Open(spec, think_end);
}
ConstraintFactory ServeConstraintFactory(std::shared_ptr<ConstraintService> service, int think_end) {
  return [service = std::move(service), think_end](const ConstraintSpec& spec) {
    const auto status = service->Compile(spec);
    if (status.state == CompileState::kRefused) throw ConstraintError(status.message);
    if (status.state == CompileState::kCompiling) throw ConstraintError(kPendingMessage);
    // Ready now; an eviction between the two calls is pending again (the next open compiles it again).
    return service->Open(spec, think_end);
  };
}

std::string ConstraintLibraryVersion() { return "xgrammar 0.2.9 crucible-1"; }
}  // namespace crucible
