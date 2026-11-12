// Checkpoint::Override (MTP draft sidecar): replaces only existing tensors under a
// prefix, with the same dtype and shape, and stops on anything else.
#include "../engine/exl3/safetensors.h"
#include <cassert>
#include <cstring>
#include <filesystem>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

struct T {
  std::string name, dtype;
  std::vector<int64_t> shape;
  uint8_t fill;
};

static void Write(const fs::path& path, const std::vector<T>& ts) {
  std::string header = "{";
  std::string payload;
  for (const auto& t : ts) {
    size_t bytes = st::DtypeBytes(t.dtype);
    std::string shape;
    for (const int64_t d : t.shape) {
      bytes *= size_t(d);
      shape += (shape.empty() ? "" : ",") + std::to_string(d);
    }
    if (header.size() > 1) header += ",";
    header += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[" + shape + "],\"data_offsets\":[" +
              std::to_string(payload.size()) + "," + std::to_string(payload.size() + bytes) + "]}";
    payload.append(bytes, char(t.fill));
  }
  header += "}";
  std::ofstream file(path, std::ios::binary);
  const uint64_t size = header.size();
  for (int i = 0; i < 8; ++i) file.put(char(size >> (8 * i)));
  file << header << payload;
  file.close();
  assert(file);
}

// Runs fn in a child; returns its exit status (Fail exits with 1). The child's stderr goes to *err, so a test
// can require the exact reason: a mutant that removes one check often fails later with another message.
template <class F>
static int Child(F fn, std::string* err = nullptr) {
  char path[] = "/tmp/crucible-override-err-XXXXXX";
  const int fd = mkstemp(path);
  assert(fd >= 0);
  close(fd);
  std::fflush(nullptr);
  const pid_t pid = fork();
  assert(pid >= 0);
  if (pid == 0) {
    if (!std::freopen(path, "w", stderr)) _exit(3);
    fn();
    _exit(0);
  }
  int status = 0;
  assert(waitpid(pid, &status, 0) == pid);
  std::ifstream in(path);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (err) *err = text;
  fs::remove(path);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int main() {
  char dir[] = "/tmp/crucible-override-XXXXXX";
  assert(mkdtemp(dir));
  const fs::path root(dir);
  Write(root / "one.safetensors", {{"mtp.a", "F16", {4}, 1}, {"mtp.b", "BF16", {2, 2}, 2}, {"trunk.c", "F16", {4}, 3},
                                   {"mtp.q.trellis", "I16", {2, 2, 80}, 4}});
  {
    std::ofstream index(root / "model.safetensors.index.json");
    index << "{\"weight_map\":{\"mtp.a\":\"one.safetensors\",\"mtp.b\":\"one.safetensors\","
             "\"trunk.c\":\"one.safetensors\",\"mtp.q.trellis\":\"one.safetensors\"}}";
  }
  const std::string side = (root / "side.safetensors").string();
  Write(side, {{"mtp.a", "F16", {4}, 9}});
  {
    st::Checkpoint ck(root.string());
    assert(ck.Data("mtp.a")[0] == 1);
    assert(ck.Override(side, "mtp.") == 1);
    size_t bytes = 0;
    assert(ck.Data("mtp.a", &bytes)[0] == 9 && bytes == 8);
    assert(ck.Data("mtp.b")[0] == 2 && ck.Data("trunk.c")[0] == 3);
    assert(ck.Info("mtp.a").dtype == "F16");
  }
  {
    // Override before any original header is read.
    st::Checkpoint ck(root.string());
    assert(ck.Override(side, "mtp.") == 1);
    assert(ck.Data("mtp.a")[0] == 9 && ck.Data("mtp.b")[0] == 2);
  }
  {
    // A requantized EXL3 matrix may change its bit width: only the trellis word count differs.
    const std::string k6 = (root / "k6.safetensors").string();
    Write(k6, {{"mtp.q.trellis", "I16", {2, 2, 96}, 7}});
    st::Checkpoint ck(root.string());
    assert(ck.Override(k6, "mtp.") == 1);
    assert(ck.Info("mtp.q.trellis").shape[2] == 96 && ck.Data("mtp.q.trellis")[0] == 7);
  }
  // Each case must stop with exit 1 and name its own reason in the message.
  auto rejects = [&](const std::vector<T>& ts, const std::string& reason) {
    const std::string bad = (root / "bad.safetensors").string();
    Write(bad, ts);
    std::string err;
    const int code = Child([&] {
      st::Checkpoint ck(root.string());
      ck.Override(bad, "mtp.");
    }, &err);
    // The message names the sidecar file once, not dir_ + "//" + its absolute path.
    // Leak guard (M5): the message names the file, never the folder (a sidecar may sit in a private folder).
    return code == 1 && err.find(reason) != std::string::npos && err.find("//") == std::string::npos &&
           err.find("checkpoint error: bad.safetensors, tensor ") != std::string::npos &&
           err.find(root.string()) == std::string::npos && err.find(root.filename().string()) == std::string::npos;
  };
  const std::string shape = "override dtype or shape differs";
  assert(rejects({{"trunk.c", "F16", {4}, 9}}, "outside mtp."));
  assert(rejects({{"mtp.z", "F16", {4}, 9}}, "override tensor is not in the checkpoint"));
  assert(rejects({{"mtp.a", "BF16", {4}, 9}}, shape));           // dtype differs
  assert(rejects({{"mtp.b", "BF16", {4}, 9}}, shape));           // plain shape differs
  assert(rejects({}, "override file has no tensors"));
  {
    // The engine start asks for the empty case to be counted, not stopped here: it stops with exit 2 (mtp_sidecar.h).
    const std::string empty = (root / "empty.safetensors").string();
    Write(empty, {});
    st::Checkpoint ck(root.string());
    assert(ck.Override(empty, "mtp.", /*allow_empty=*/true) == 0);
    assert(ck.Data("mtp.a")[0] == 1);
  }
  assert(rejects({{"mtp.q.trellis", "I16", {2, 4, 96}, 7}}, shape));   // second tile dimension differs
  assert(rejects({{"mtp.q.trellis", "I16", {4, 2, 96}, 7}}, shape));   // first tile dimension differs
  assert(rejects({{"mtp.q.trellis", "I16", {2, 2, 112}, 7}}, shape));  // K7: the engine takes K 2 to 6
  assert(rejects({{"mtp.q.trellis", "I16", {2, 2, 120}, 7}}, shape));  // not a whole K
  assert(rejects({{"mtp.q.trellis", "I16", {2, 2, 40}, 7}}, shape));   // in range, not a whole K
  assert(rejects({{"mtp.q.trellis", "I16", {2, 2, 16}, 7}}, shape));   // K1
  assert(rejects({{"mtp.a", "F16", {8}, 7}}, shape));            // plain tensor shape differs
  {
    // A later geometry error on a replaced tensor names the sidecar file itself.
    std::string err;
    assert(Child([&] {
      st::Checkpoint ck(root.string());
      ck.Override(side, "mtp.");
      ck.Error("mtp.a", "geometry check");
    }, &err) == 1);
    assert(err.find("side.safetensors, tensor mtp.a: geometry check") != std::string::npos);
    assert(err.find(root.string()) == std::string::npos && err.find("//") == std::string::npos);
  }
  {
    // Leak guard (M5): no message prints a directory, also when the index or a shard is missing.
    std::string err;
    assert(Child([&] { st::Checkpoint ck(root.string()); ck.Override((root / "none").string(), "mtp."); }, &err) == 1);
    assert(err.find("none") != std::string::npos && err.find(root.string()) == std::string::npos);
    assert(Child([&] { st::Checkpoint ck(root.string()); ck.Error("not.a.tensor", "reason"); }, &err) == 1);
    assert(err.find("model.safetensors.index.json") != std::string::npos && err.find(root.string()) == std::string::npos);
    const std::string empty_dir = (root / "private-run-dir").string();
    fs::create_directory(empty_dir);
    assert(Child([&] { st::Checkpoint ck(empty_dir); }, &err) == 1);
    assert(err.find("private-run-dir") == std::string::npos && err.find(root.string()) == std::string::npos);
  }
  assert(Child([&] { st::Checkpoint ck(root.string()); ck.Override((root / "none").string(), "mtp."); }) == 1);
  fs::remove_all(root);
  std::puts("PASS: checkpoint override replaces only same-shape tensors under the prefix");
}
