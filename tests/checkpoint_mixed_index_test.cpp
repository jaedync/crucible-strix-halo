// Checkpoint with a mixed-pack index: when the index holds the key "crucible_mixed_pack", the index
// picks the shard of each tensor. Another index shard may hold a copy of the same name (a link to a whole shard of a
// second pack); that copy is skipped and counted. Without the key, a name in two shards still stops the load.
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

static void Index(const fs::path& dir, const std::string& extra, const std::string& map) {
  std::ofstream index(dir / "model.safetensors.index.json");
  index << "{" << extra << "\"weight_map\":{" << map << "}}";
}

template <class F>
static int Child(F fn, std::string* err = nullptr) {
  char path[] = "/tmp/crucible-mixidx-err-XXXXXX";
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
  char dir[] = "/tmp/crucible-mixidx-XXXXXX";
  assert(mkdtemp(dir));
  const fs::path root(dir);
  // Two "packs": each shard holds the same names with its own bytes (pack a fills 1/2/3, pack b fills 7/8/9).
  Write(root / "a-00001.safetensors", {{"l.x.trellis", "I16", {2, 2, 64}, 1}, {"l.y.trellis", "I16", {2, 2, 96}, 2},
                                       {"l.n.weight", "F16", {4}, 3}});
  Write(root / "b-00001.safetensors", {{"l.x.trellis", "I16", {2, 2, 80}, 7}, {"l.y.trellis", "I16", {2, 2, 80}, 8},
                                       {"l.n.weight", "F16", {4}, 9}});
  const std::string map = "\"l.x.trellis\":\"b-00001.safetensors\",\"l.y.trellis\":\"a-00001.safetensors\","
                          "\"l.n.weight\":\"a-00001.safetensors\"";
  Index(root, "\"crucible_mixed_pack\":{\"base\":\"a\",\"groups\":[\"x\"]},", map);
  {
    st::Checkpoint ck(root.string());
    assert(ck.mixed());
    // The index decides: x from pack b (K5 here), y and n from pack a.
    assert(ck.Data("l.x.trellis")[0] == 7 && ck.Info("l.x.trellis").shape[2] == 80);
    assert(ck.Data("l.y.trellis")[0] == 2 && ck.Info("l.y.trellis").shape[2] == 96);
    assert(ck.Data("l.n.weight")[0] == 3);
    // Each shard held one copy that the index did not pick: a has x, b has y and n.
    assert(ck.shadowed() == 3);
    assert(ck.NamesEndingWith(".trellis").size() == 2);
  }
  {
    // The other load order (pack a's shard first) gives the same tensors.
    st::Checkpoint ck(root.string());
    assert(ck.Data("l.y.trellis")[0] == 2);
    assert(ck.Data("l.x.trellis")[0] == 7);
    assert(ck.shadowed() == 3);
  }
  {
    // ReadBytes and Prefetch read the picked copy too.
    st::Checkpoint ck(root.string());
    ck.Prefetch({"l.n.weight", "l.x.trellis"}, 2, 64);
    uint8_t v = 0;
    ck.ReadBytes("l.n.weight", &v, 1);
    assert(v == 3);
    ck.ReadBytes("l.x.trellis", &v, 1);
    assert(v == 7);
  }
  // Without the key: the same shards stop the load with the old reason (the file name only, never the folder).
  Index(root, "", map);
  {
    std::string err;
    const int code = Child([&] {
      st::Checkpoint ck(root.string());
      assert(!ck.mixed());
      ck.Data("l.x.trellis");
      ck.Data("l.y.trellis");
    }, &err);
    assert(code == 1);
    assert(err.find("duplicate tensor name across checkpoint shards") != std::string::npos);
    assert(err.find(root.string()) == std::string::npos);
  }
  // With the key, a name that the index does not list is skipped in every shard (it could not be read: Info needs the
  // index), so two source packs may both hold it (the vision tower of 3.05 and 4.18).
  Write(root / "a-00001.safetensors", {{"l.x.trellis", "I16", {2, 2, 64}, 1}, {"l.y.trellis", "I16", {2, 2, 96}, 2},
                                       {"l.n.weight", "F16", {4}, 3}, {"l.extra", "F16", {2}, 4}});
  Write(root / "b-00001.safetensors", {{"l.x.trellis", "I16", {2, 2, 80}, 7}, {"l.y.trellis", "I16", {2, 2, 80}, 8},
                                       {"l.n.weight", "F16", {4}, 9}, {"l.extra", "F16", {2}, 5}});
  Index(root, "\"crucible_mixed_pack\":1,", map);
  {
    st::Checkpoint ck(root.string());
    assert(ck.Data("l.y.trellis")[0] == 2 && ck.Data("l.x.trellis")[0] == 7);
    assert(ck.shadowed() == 5 && !ck.Has("l.extra"));
  }
  // Without the key the same unlisted copies still stop the load.
  Index(root, "", map);
  {
    std::string err;
    const int code = Child([&] {
      st::Checkpoint ck(root.string());
      ck.Data("l.x.trellis");
      ck.Data("l.y.trellis");
    }, &err);
    assert(code == 1 && err.find("duplicate tensor name across checkpoint shards") != std::string::npos);
  }
  // An extra file (AddFile, the n-gram table and vision_k6) keeps its own rule: a wins file owns its names, and the
  // mixed key does not skip them when the file loads before the shards.
  Write(root / "b-00001.safetensors", {{"l.x.trellis", "I16", {2, 2, 80}, 7}, {"l.y.trellis", "I16", {2, 2, 80}, 8},
                                       {"l.n.weight", "F16", {4}, 9}});
  Write(root / "a-00001.safetensors", {{"l.x.trellis", "I16", {2, 2, 64}, 1}, {"l.y.trellis", "I16", {2, 2, 96}, 2},
                                       {"l.n.weight", "F16", {4}, 3}});
  Write(root / "vision.safetensors", {{"l.n.weight", "F16", {4}, 6}});
  Index(root, "\"crucible_mixed_pack\":1,", map);
  {
    st::Checkpoint ck(root.string());
    ck.AddFile("vision.safetensors", /*wins=*/true);
    assert(ck.Data("l.n.weight")[0] == 6);
    assert(ck.Data("l.y.trellis")[0] == 2 && ck.Data("l.x.trellis")[0] == 7);
  }
  fs::remove_all(root);
  std::puts("checkpoint_mixed_index_test PASS");
  return 0;
}
