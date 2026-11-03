// BF16 vision tensors (issue 3): the BF16 to F16 conversion over all 65,536 bit patterns, and the
// vision validation on a synthetic sparse checkpoint (patch embedding and linear biases as F16 or BF16).
#include "../engine/model/flashnext/weight_validation.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

namespace fs = std::filesystem;
using namespace crucible;

namespace {

// Reference: the exact BF16 value as a double, rounded to F16 with nearbyint (round to nearest even)
// on the F16 grid. false: infinity, NaN or overflow. *value is the F16 value as a double.
bool RefHalf(uint16_t bf, uint16_t* bits, double* source) {
  const uint32_t w = uint32_t(bf) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  *source = f;
  if (!std::isfinite(f)) return false;
  const double v = std::fabs(double(f));
  const uint16_t sign = (bf & 0x8000u) ? 0x8000u : 0;
  if (v < std::ldexp(1.0, -14)) {  // F16 subnormals: multiples of 2^-24
    *bits = sign | uint16_t(std::nearbyint(std::ldexp(v, 24)));  // 1024 is the smallest normal
    return true;
  }
  int e = std::ilogb(v);
  double m = std::nearbyint(std::ldexp(v, 10 - e));  // 1024..2048
  if (m == 2048) { m = 1024; ++e; }
  if (e > 15) return false;
  *bits = sign | uint16_t(((e + 15) << 10) | (int(m) - 1024));
  return true;
}

double HalfValue(uint16_t bits) {
  const int e = (bits >> 10) & 31, m = bits & 1023;
  const double v = e ? std::ldexp(1024 + m, e - 25) : std::ldexp(m, -24);
  return (bits & 0x8000u) ? -v : v;
}

void ConversionOverAllPatterns() {
  size_t ok_count = 0, error_count = 0, inexact = 0;
  for (uint32_t i = 0; i < 65536; ++i) {
    const uint16_t bf = uint16_t(i);
    uint16_t want = 0, got = 0;
    double source = 0;
    bool exact = false;
    const bool want_ok = RefHalf(bf, &want, &source);
    const bool got_ok = Bf16ToHalf(bf, &got, &exact);
    if (want_ok != got_ok) std::fprintf(stderr, "pattern 0x%04x: ok %d, want %d\n", i, got_ok, want_ok);
    CHECK(want_ok == got_ok);
    if (!want_ok) {
      ++error_count;
      continue;
    }
    ++ok_count;
    if (want != got) std::fprintf(stderr, "pattern 0x%04x: got 0x%04x, want 0x%04x\n", i, got, want);
    CHECK(want == got);
    const bool want_exact = HalfValue(want) == source;
    CHECK(exact == want_exact);
    if (!exact) ++inexact;
    // The documented rule: every value with 2^-14 <= |x| <= 65504 is exact.
    if (std::fabs(source) >= std::ldexp(1.0, -14)) CHECK(exact);
    // Every inexact value is a F16 subnormal or zero.
    if (!exact) CHECK(std::fabs(source) < std::ldexp(1.0, -14));
  }
  // 256 patterns of exponent field 255 (inf, NaN) and 112 * 256 above 65504.
  CHECK(error_count == 256 + 112 * 256);
  CHECK(ok_count == 65536 - error_count);
  CHECK(inexact > 0);
  std::printf("bf16 to f16: %zu convert, %zu error, %zu inexact\n", ok_count, error_count, inexact);
}

// ---- synthetic checkpoint ------------------------------------------------------------------

struct Tensor {
  std::string name, dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;
};

struct Opts {
  std::string patch_w = "F16", patch_b = "F16", bias = "F16";
  // Edits the BF16 pattern of a touched tensor before it is stored.
  std::function<void(const std::string&, std::vector<uint16_t>&)> edit;
};

const std::string P = "model.visual.";

std::string TmpRoot() {
  const char* d = std::getenv("TMPDIR");
  return d && *d ? d : "/tmp";
}

// BF16 patterns whose F16 value is exact: exponent field 113..142 (2^-14 up to below 2^16), plus zeros.
std::vector<uint16_t> ExactPatterns(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint16_t> v(n);
  for (auto& x : v) {
    const uint32_t exp = 113 + rng() % 30;  // 113..142: |x| < 2^16, and 65280 is the largest below 65504
    x = uint16_t(((rng() & 1) << 15) | (exp << 7) | (rng() & 127));
    if (rng() % 97 == 0) x &= 0x8000;  // +0 and -0
  }
  return v;
}

// Writes the checkpoint as one sparse shard; only the touched tensors and the mul1 markers hold bytes.
// *touched receives the BF16 patterns of the three kinds in name order.
fs::path WriteVision(const Opts& o, std::vector<std::pair<std::string, std::vector<uint16_t>>>* touched = nullptr) {
  std::string tmpl = TmpRoot() + "/vision_bf16_testXXXXXX";
  CHECK(mkdtemp(tmpl.data()));
  const fs::path dir(tmpl);
  std::vector<Tensor> ts;
  uint64_t off = 0;
  auto add = [&](const std::string& n, const std::string& dt, std::vector<int64_t> shape) {
    size_t bytes = st::DtypeBytes(dt);
    for (const int64_t d : shape) bytes *= size_t(d);
    ts.push_back({n, dt, std::move(shape), off, off + bytes});
    off += bytes;
  };
  add(P + "patch_embed.proj.weight", o.patch_w, {1152, 3, 2, 16, 16});
  add(P + "patch_embed.proj.bias", o.patch_b, {1152});
  add(P + "pos_embed.weight", "BF16", {2304, 1152});
  auto lin = [&](const std::string& n, int64_t I, int64_t O) {
    add(n + ".trellis", "I16", {I / 16, O / 16, 48});
    add(n + ".suh", "F16", {I});
    add(n + ".svh", "F16", {O});
    add(n + ".mul1", "I32", {});
    add(n + ".bias", o.bias, {O});
  };
  for (int i = 0; i < 27; ++i) {
    const std::string B = P + "blocks." + std::to_string(i) + ".";
    for (const char* n : {"attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj"}) lin(B + n, 1152, 1152);
    lin(B + "mlp.linear_fc1", 1152, 4352);
    lin(B + "mlp.linear_fc2", 4352, 1152);
    for (const char* n : {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"}) add(B + n, "BF16", {1152});
  }
  lin(P + "merger.linear_fc1", 4608, 4608);
  lin(P + "merger.linear_fc2", 4608, kHidden);
  add(P + "merger.norm.weight", "F16", {1152});
  add(P + "merger.norm.bias", "F16", {1152});

  std::string header = "{", index = "{\"weight_map\":{";
  for (const Tensor& t : ts) {
    std::string shape;
    for (const int64_t d : t.shape) shape += (shape.empty() ? "" : ",") + std::to_string(d);
    const std::string sep = header.size() > 1 ? "," : "";
    header += sep + "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[" + shape +
              "],\"data_offsets\":[" + std::to_string(t.begin) + "," + std::to_string(t.end) + "]}";
    index += (index.size() > 15 ? "," : "") + std::string("\"") + t.name + "\":\"shard.safetensors\"";
  }
  header += "}";
  header.append((8 - header.size() % 8) % 8, ' ');
  index += "}}";
  std::ofstream(dir / "model.safetensors.index.json") << index;

  const std::string path = (dir / "shard.safetensors").string();
  const int fd = open(path.c_str(), O_CREAT | O_RDWR, 0600);
  CHECK(fd >= 0);
  const uint64_t hlen = header.size(), base = 8 + hlen;
  CHECK(ftruncate(fd, off_t(base + off)) == 0);
  uint8_t len[8];
  for (int b = 0; b < 8; ++b) len[b] = uint8_t(hlen >> (8 * b));
  CHECK(pwrite(fd, len, 8, 0) == 8);
  CHECK(pwrite(fd, header.data(), header.size(), 8) == ssize_t(header.size()));
  const uint32_t marker = 0x83DCD12D;
  uint32_t seed = 1;
  for (const Tensor& t : ts) {
    if (t.name.size() > 5 && t.name.compare(t.name.size() - 5, 5, ".mul1") == 0)
      CHECK(pwrite(fd, &marker, 4, off_t(base + t.begin)) == 4);
    const bool touched_kind = t.name == P + "patch_embed.proj.weight" || t.name == P + "patch_embed.proj.bias" ||
                              (t.name.size() > 5 && t.name.compare(t.name.size() - 5, 5, ".bias") == 0 &&
                               t.name.find(".norm") == std::string::npos && t.name.find("merger.norm") == std::string::npos);
    if (!touched_kind) continue;
    const size_t n = (t.end - t.begin) / 2;
    std::vector<uint16_t> bf = ExactPatterns(n, seed++);
    if (o.edit) o.edit(t.name, bf);
    if (touched) touched->push_back({t.name, bf});
    std::vector<uint16_t> out = bf;
    if (t.dtype == "F16") {
      for (size_t i = 0; i < n; ++i) {
        uint16_t h = 0;
        double src;
        if (!RefHalf(bf[i], &h, &src)) h = 0x7e00;  // an F16 file cannot hold the BF16 overflow
        out[i] = h;
      }
    }
    CHECK(pwrite(fd, out.data(), n * 2, off_t(base + t.begin)) == ssize_t(n * 2));
  }
  close(fd);
  return dir;
}

// Runs fn in a child (Fail exits with 1). Returns the exit status; *err receives the child's stderr.
int Child(const std::function<void()>& fn, std::string* err) {
  std::string path = TmpRoot() + "/vision_bf16_errXXXXXX";
  const int fd = mkstemp(path.data());
  CHECK(fd >= 0);
  close(fd);
  std::fflush(nullptr);
  const pid_t pid = fork();
  CHECK(pid >= 0);
  if (pid == 0) {
    if (!std::freopen(path.c_str(), "w", stderr)) _exit(3);
    fn();
    _exit(0);
  }
  int status = 0;
  CHECK(waitpid(pid, &status, 0) == pid);
  std::ifstream in(path);
  *err = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  fs::remove(path);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

bool Has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

int Validate(const fs::path& dir, std::string* err) {
  return Child([&] {
    st::Checkpoint ck(dir.string());
    ValidateVision(ck, P);
  }, err);
}

void CheckpointCases() {
  std::string err;
  // F16 checkpoint: unchanged, passes. An F16 value of 65504 is fine (no conversion).
  {
    const fs::path dir = WriteVision({});
    CHECK(Validate(dir, &err) == 0);
    fs::remove_all(dir);
  }
  // BF16 patch embedding and BF16 biases: pass.
  Opts bf;
  bf.patch_w = bf.patch_b = bf.bias = "BF16";
  std::vector<std::pair<std::string, std::vector<uint16_t>>> touched;
  const fs::path good = WriteVision(bf, &touched);
  CHECK(Validate(good, &err) == 0);
  CHECK(touched.size() == 2 + 164);
  // Each kind alone, and a mixed checkpoint.
  for (int kind = 0; kind < 3; ++kind) {
    Opts o;
    if (kind == 0) o.patch_w = "BF16";
    if (kind == 1) o.patch_b = "BF16";
    if (kind == 2) o.bias = "BF16";
    const fs::path dir = WriteVision(o);
    CHECK(Validate(dir, &err) == 0);
    fs::remove_all(dir);
  }
  // The values: BF16 with F16-exact values read back as the same F16 words that an F16 file holds.
  {
    const fs::path a = WriteVision({});
    st::Checkpoint ca(a.string()), cb(good.string());
    for (const auto& [name, bits] : touched) {
      HalfTensor b = ReadHalfTensor(cb, name);
      CHECK(b.bits.size() == bits.size());
      CHECK(b.inexact == 0);
      std::vector<uint16_t> f16(bits.size());
      ca.ReadBytes(name, f16.data(), f16.size() * 2);
      CHECK(b.bits == f16);
    }
    fs::remove_all(a);
  }
  // An inexact count: BF16 below 2^-14 that does not fit an F16 subnormal grid.
  {
    Opts o = bf;
    o.edit = [](const std::string& n, std::vector<uint16_t>& v) {
      if (n == P + "patch_embed.proj.bias") {
        v[0] = 0x3681;  // 2^-18 (1 + 2^-7): half a subnormal step, ties to even
        v[1] = 0x3680;  // 2^-18, exact
        v[2] = 0x0001;  // a BF16 subnormal: rounds to zero
        v[3] = 0x3880;  // 2^-14, exact
      }
    };
    const fs::path dir = WriteVision(o);
    st::Checkpoint ck(dir.string());
    HalfTensor h = ReadHalfTensor(ck, P + "patch_embed.proj.bias");
    CHECK(h.inexact == 2);
    CHECK(h.bits[0] == 0x0040 && h.bits[1] == 0x0040 && h.bits[2] == 0 && h.bits[3] == 0x0400);
    CHECK(Validate(dir, &err) == 0);
    fs::remove_all(dir);
  }
  // A value that does not fit in F16 names the tensor. 0x4780 = 65536, 0x7f80 = inf, 0x7fc0 = NaN.
  struct Bad { std::string tensor; size_t at; uint16_t pattern; const char* kind; };
  const std::vector<Bad> bad = {
      {P + "patch_embed.proj.weight", 12345, 0x4780, "patch weight 65536"},
      {P + "patch_embed.proj.weight", 0, 0xc780, "patch weight -65536"},
      {P + "patch_embed.proj.bias", 1151, 0x7f80, "patch bias inf"},
      {P + "blocks.3.attn.v_proj.bias", 7, 0x7fc0, "block bias NaN"},
      {P + "merger.linear_fc2.bias", 2559, 0x7f7f, "merger bias max BF16"},
  };
  for (const Bad& b : bad) {
    Opts o = bf;
    o.edit = [&](const std::string& n, std::vector<uint16_t>& v) {
      if (n == b.tensor) v[b.at] = b.pattern;
    };
    const fs::path dir = WriteVision(o);
    const int rc = Validate(dir, &err);
    if (rc != 1 || !Has(err, b.tensor + ":")) std::fprintf(stderr, "case %s: rc %d, stderr: %s\n", b.kind, rc, err.c_str());
    CHECK(rc == 1);
    CHECK(Has(err, "tensor " + b.tensor + ":"));
    CHECK(Has(err, "65504"));
    fs::remove_all(dir);
  }
  // The largest value that fits: 65280 (0x477f) passes.
  {
    Opts o = bf;
    o.edit = [](const std::string& n, std::vector<uint16_t>& v) {
      if (n == P + "patch_embed.proj.weight") v[5] = 0x477f;
    };
    const fs::path dir = WriteVision(o);
    CHECK(Validate(dir, &err) == 0);
    fs::remove_all(dir);
  }
  // Another dtype is still rejected.
  {
    Opts o;
    o.patch_w = "F32";
    const fs::path dir = WriteVision(o);
    CHECK(Validate(dir, &err) == 1);
    CHECK(Has(err, "tensor " + P + "patch_embed.proj.weight: expected dtype F16/BF16, got F32"));
    fs::remove_all(dir);
  }
  fs::remove_all(good);
}

}  // namespace

int main() {
  ConversionOverAllPatterns();
  CheckpointCases();
  std::puts("vision bf16 test passed");
}
