// A checkpoint whose vision tower is plain HF tensors (BF16 matrices, no EXL3 trellis), and the vision_k6.safetensors file
// that makes it run (issue 3). Also the BF16 PLE norms of such a checkpoint. CPU only, synthetic sparse files.
#include "../engine/model/flashnext/weight_validation.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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

const std::string P = "model.visual.";
const std::string kTextTensor = "model.language_model.test.weight";

struct Tensor {
  std::string name, dtype;
  std::vector<int64_t> shape;
};

std::string TmpRoot() {
  const char* d = std::getenv("TMPDIR");
  return d && *d ? d : "/tmp";
}

fs::path MakeDir() {
  std::string tmpl = TmpRoot() + "/vision_tower_testXXXXXX";
  CHECK(mkdtemp(tmpl.data()));
  return fs::path(tmpl);
}

bool EndsWith(const std::string& s, const std::string& tail) {
  return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// Writes one sparse safetensors file. Zero bytes everywhere except the ".mul1" markers that the EXL3 check reads.
void WriteShard(const fs::path& path, const std::vector<Tensor>& ts, const std::function<void(const Tensor&, std::vector<uint8_t>&)>& fill = {}) {
  std::string header = "{";
  uint64_t off = 0;
  std::vector<std::pair<uint64_t, uint64_t>> spans;
  for (const Tensor& t : ts) {
    size_t bytes = st::DtypeBytes(t.dtype);
    std::string shape;
    for (const int64_t d : t.shape) {
      bytes *= size_t(d);
      shape += (shape.empty() ? "" : ",") + std::to_string(d);
    }
    header += std::string(header.size() > 1 ? "," : "") + "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[" +
              shape + "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
    spans.push_back({off, off + bytes});
    off += bytes;
  }
  header += "}";
  header.append((8 - header.size() % 8) % 8, ' ');
  const int fd = open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
  CHECK(fd >= 0);
  const uint64_t hlen = header.size(), base = 8 + hlen;
  CHECK(ftruncate(fd, off_t(base + off)) == 0);
  uint8_t len[8];
  for (int b = 0; b < 8; ++b) len[b] = uint8_t(hlen >> (8 * b));
  CHECK(pwrite(fd, len, 8, 0) == 8);
  CHECK(pwrite(fd, header.data(), header.size(), 8) == ssize_t(header.size()));
  const uint32_t marker = 0x83DCD12D;
  for (size_t i = 0; i < ts.size(); ++i) {
    if (EndsWith(ts[i].name, ".mul1")) CHECK(pwrite(fd, &marker, 4, off_t(base + spans[i].first)) == 4);
    if (!fill) continue;
    std::vector<uint8_t> bytes(spans[i].second - spans[i].first, 0);
    fill(ts[i], bytes);
    if (!bytes.empty()) CHECK(pwrite(fd, bytes.data(), bytes.size(), off_t(base + spans[i].first)) == ssize_t(bytes.size()));
  }
  close(fd);
}

void WriteIndex(const fs::path& dir, const std::vector<Tensor>& ts, const std::string& shard) {
  std::string index = "{\"weight_map\":{";
  for (size_t i = 0; i < ts.size(); ++i)
    index += (i ? "," : "") + std::string("\"") + ts[i].name + "\":\"" + shard + "\"";
  std::ofstream(dir / "model.safetensors.index.json") << index << "}}";
}

// The EXL3 tower as vision_k6.safetensors holds it (all F16 small tensors).
std::vector<Tensor> ExlTower() {
  std::vector<Tensor> ts;
  ts.push_back({P + "patch_embed.proj.weight", "F16", {1152, 3, 2, 16, 16}});
  ts.push_back({P + "patch_embed.proj.bias", "F16", {1152}});
  ts.push_back({P + "pos_embed.weight", "BF16", {2304, 1152}});
  auto lin = [&](const std::string& n, int64_t I, int64_t O) {
    ts.push_back({n + ".trellis", "I16", {I / 16, O / 16, 48}});
    ts.push_back({n + ".suh", "F16", {I}});
    ts.push_back({n + ".svh", "F16", {O}});
    ts.push_back({n + ".mul1", "I32", {}});
    ts.push_back({n + ".bias", "F16", {O}});
  };
  for (int i = 0; i < 27; ++i) {
    const std::string B = P + "blocks." + std::to_string(i) + ".";
    for (const char* n : {"attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj"}) lin(B + n, 1152, 1152);
    lin(B + "mlp.linear_fc1", 1152, 4352);
    lin(B + "mlp.linear_fc2", 4352, 1152);
    for (const char* n : {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"}) ts.push_back({B + n, "BF16", {1152}});
  }
  lin(P + "merger.linear_fc1", 4608, 4608);
  lin(P + "merger.linear_fc2", 4608, kHidden);
  ts.push_back({P + "merger.norm.weight", "F16", {1152}});
  ts.push_back({P + "merger.norm.bias", "F16", {1152}});
  return ts;
}

// The same tower as the original model stores it: 333 BF16 tensors, fused qkv, no EXL3 tensors.
std::vector<Tensor> HfTower() {
  std::vector<Tensor> ts;
  ts.push_back({P + "patch_embed.proj.weight", "BF16", {1152, 3, 2, 16, 16}});
  ts.push_back({P + "patch_embed.proj.bias", "BF16", {1152}});
  ts.push_back({P + "pos_embed.weight", "BF16", {2304, 1152}});
  auto lin = [&](const std::string& n, int64_t I, int64_t O) {
    ts.push_back({n + ".weight", "BF16", {O, I}});
    ts.push_back({n + ".bias", "BF16", {O}});
  };
  for (int i = 0; i < 27; ++i) {
    const std::string B = P + "blocks." + std::to_string(i) + ".";
    lin(B + "attn.qkv", 1152, 3456);
    lin(B + "attn.proj", 1152, 1152);
    lin(B + "mlp.linear_fc1", 1152, 4352);
    lin(B + "mlp.linear_fc2", 4352, 1152);
    for (const char* n : {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"}) ts.push_back({B + n, "BF16", {1152}});
  }
  ts.push_back({P + "merger.norm.weight", "BF16", {1152}});
  ts.push_back({P + "merger.norm.bias", "BF16", {1152}});
  lin(P + "merger.linear_fc1", 4608, 4608);
  lin(P + "merger.linear_fc2", 4608, kHidden);
  return ts;
}

// Runs fn in a child (Fail exits with 1). Returns the exit status; *err receives the child's stderr.
int Child(const std::function<void()>& fn, std::string* err) {
  std::string path = TmpRoot() + "/vision_tower_errXXXXXX";
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

// The checkpoint the reporter has: the HF tower in a shard that the index lists, optionally one text tensor in
// the same shard, optionally the EXL3 tower file next to it (not in the index).
fs::path Make(bool with_k6, bool text_in_shard) {
  const fs::path dir = MakeDir();
  std::vector<Tensor> hf = HfTower();
  if (text_in_shard) hf.push_back({kTextTensor, "BF16", {4}});
  WriteShard(dir / "model-00009-of-00009.safetensors", hf);
  WriteIndex(dir, hf, "model-00009-of-00009.safetensors");
  if (with_k6) WriteShard(dir / "vision_k6.safetensors", ExlTower());
  return dir;
}

// What the Loader does for --vision: map vision_k6.safetensors over the index when it exists.
void OpenLikeLoader(st::Checkpoint& ck, const fs::path& dir) {
  if (fs::exists(dir / "vision_k6.safetensors")) ck.AddFile("vision_k6.safetensors", /*wins=*/true);
}

void MissingFile() {
  const fs::path dir = Make(false, false);
  const std::string fault = VisionTowerFault(dir.string());
  CHECK(!fault.empty());
  for (const char* part : {"vision_k6.safetensors", "turboderp/Qwen3.8-Flash-Next-exl3", "4.05bpw_h6_ng6",
                           "hf download turboderp/Qwen3.8-Flash-Next-exl3 vision_k6.safetensors --revision 4.05bpw_h6_ng6 --local-dir",
                           "only when the weights of the HF tower are the stock weights"}) {
    if (!Has(fault, part)) std::fprintf(stderr, "missing in message: %s\nmessage: %s\n", part, fault.c_str());
    CHECK(Has(fault, part));
  }
  // The load stops with the same text; the old dtype error does not come first.
  std::string err;
  const int rc = Child([&] {
    st::Checkpoint ck(dir.string());
    ValidateVision(ck, P);
  }, &err);
  if (rc != 1) std::fprintf(stderr, "rc %d: %s\n", rc, err.c_str());
  CHECK(rc == 1);
  CHECK(Has(err, "vision_k6.safetensors"));
  CHECK(Has(err, "hf download turboderp/Qwen3.8-Flash-Next-exl3 vision_k6.safetensors --revision 4.05bpw_h6_ng6"));
  CHECK(!Has(err, "expected dtype"));
  fs::remove_all(dir);
}

void WithFile() {
  std::string err;
  // The HF shard repeats patch_embed.proj.weight and every other tower name. The file wins and the shard header is never read.
  {
    const fs::path dir = Make(true, false);
    CHECK(VisionTowerFault(dir.string()).empty());
    // Proof of the lazy header: the shard is not a safetensors file any more, and the validation still passes.
    const fs::path shard = dir / "model-00009-of-00009.safetensors";
    std::ofstream(shard, std::ios::trunc) << "not a safetensors file";
    const int rc = Child([&] {
      st::Checkpoint ck(dir.string());
      OpenLikeLoader(ck, dir);
      CHECK(ck.Info(P + "patch_embed.proj.weight").dtype == "F16");  // the file's tensor, not the shard's BF16
      ValidateVision(ck, P);
    }, &err);
    if (rc != 0) std::fprintf(stderr, "rc %d: %s\n", rc, err.c_str());
    CHECK(rc == 0);
    fs::remove_all(dir);
  }
  // The shard also holds a text tensor that the text load reads: its header loads, repeats the tower names,
  // and must not fail with "duplicate tensor name across checkpoint shards".
  {
    const fs::path dir = Make(true, true);
    const int rc = Child([&] {
      st::Checkpoint ck(dir.string());
      OpenLikeLoader(ck, dir);
      CHECK(ck.Info(kTextTensor).dtype == "BF16");
      CHECK(ck.Info(P + "patch_embed.proj.weight").dtype == "F16");
      CHECK(ck.Info(P + "blocks.0.attn.q_proj.trellis").dtype == "I16");
      ValidateVision(ck, P);
    }, &err);
    if (rc != 0) std::fprintf(stderr, "rc %d: %s\n", rc, err.c_str());
    CHECK(rc == 0);
    fs::remove_all(dir);
  }
  // Two shards that repeat a name without the file are still an error.
  {
    const fs::path dir = MakeDir();
    const std::vector<Tensor> a = {{"x.a", "F16", {2}}, {"x.same", "F16", {2}}}, b = {{"x.same", "F16", {2}}};
    WriteShard(dir / "a.safetensors", a);
    WriteShard(dir / "b.safetensors", b);
    std::ofstream(dir / "model.safetensors.index.json")
        << "{\"weight_map\":{\"x.a\":\"a.safetensors\",\"x.same\":\"b.safetensors\"}}";
    const int rc = Child([&] {
      st::Checkpoint ck(dir.string());
      ck.Info("x.a");
      ck.Info("x.same");
    }, &err);
    CHECK(rc == 1 && Has(err, "duplicate tensor name"));
    fs::remove_all(dir);
  }
}

// The PLE norms of the HF text model are BF16 (the 4.05 pack stores F16). Both pass; F32 does not.
void PleNorms() {
  for (const char* dtype : {"F16", "BF16", "F32"}) {
    const fs::path dir = MakeDir();
    const std::string p = "ple.", n = p + "ple_embedding.ngram_embedding.";
    const std::vector<Tensor> ts = {
        {p + "key_proj.weight", "F16", {kWide, kHidden}},
        {p + "value_proj.weight", "F16", {kHidden, kHidden}},
        {p + "norm_key.weight", dtype, {kWide}},
        {p + "norm_query.weight", dtype, {kWide}},
        {p + "norm_conv.weight", dtype, {kWide}},
        {p + "conv1d.weight", "BF16", {kWide, 1, 4}},
        {n + "layer_multipliers", "I64", {3}},
        {n + "head_vocab_sizes", "I64", {16}},
        {n + "head_offsets", "I64", {16}},
        {n + "head_bias", "F16", {16, 160}},
        {n + "trellis", "I16", {4096, 61}},
    };
    WriteShard(dir / "s.safetensors", ts, [&](const Tensor& t, std::vector<uint8_t>& bytes) {
      auto* w = reinterpret_cast<int64_t*>(bytes.data());
      if (t.name == n + "head_vocab_sizes") for (int h = 0; h < 16; ++h) w[h] = 100;
      if (t.name == n + "head_offsets") for (int h = 0; h < 16; ++h) w[h] = 100 * h;
    });
    WriteIndex(dir, ts, "s.safetensors");
    std::string err;
    const int rc = Child([&] {
      st::Checkpoint ck(dir.string());
      ValidatePle(ck, p);
    }, &err);
    if (std::string(dtype) == "F32") {
      CHECK(rc == 1);
      CHECK(Has(err, "tensor ple.norm_key.weight: expected dtype F16/BF16, got F32"));
    } else {
      if (rc != 0) std::fprintf(stderr, "%s: rc %d: %s\n", dtype, rc, err.c_str());
      CHECK(rc == 0);
    }
    fs::remove_all(dir);
  }
}

}  // namespace

int main() {
  MissingFile();
  WithFile();
  PleNorms();
  std::puts("vision tower test passed");
}
