// CPU-only allocation checks. This test does not open model tensors.
#include "../engine/exl3/static_layout.h"
#include <cassert>
#include <iostream>
#include <map>

struct Tensor {
  std::string dtype = "F16";
  std::vector<int64_t> shape = {512, 2560};
  uint64_t begin = 0, end = 256;
};
struct Checkpoint {
  std::map<std::string, Tensor> changed;
  Tensor normal;
  bool Has(const std::string& name) const { return Info(name).dtype != "MISSING"; }
  const Tensor& Info(const std::string& name) const {
    const auto it = changed.find(name);
    return it == changed.end() ? normal : it->second;
  }
};
template<class F> void Reject(F run) {
  bool rejected = false;
  try { run(); } catch (const std::exception&) { rejected = true; }
  assert(rejected);
}
int main() {
  const Checkpoint ck;
  // fp16 HC sites plan their fragment-order copies from the geometry (fn rows padded to 336 or 320), not
  // from the 256-byte test payloads: 96 + 2 inject sites x 13,434,880 + 2 mixers x 13,107,200 bytes.
  assert(crucible::ExpectedStaticBytes(ck, 48) == 1343279616);
  assert(crucible::ExpectedStaticBytes(ck, 1) == 79979520);
  assert(crucible::StaticArenaCapacity(523264) == 2097152);
  assert(crucible::StaticArenaCapacity(1) == 2097152);
  assert(crucible::StaticArenaCapacity(2097152) == 2097152);
  assert(crucible::StaticArenaCapacity(2097153) == 4194304);
  Reject([] { crucible::StaticArenaCapacity(0); });
  Reject([] { crucible::StaticArenaCapacity(SIZE_MAX); });
  const std::string norm = "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight";
  const Checkpoint odd{{{norm, Tensor{"F16", {}, 0, 3}}}, Tensor{}};
  Reject([&] { crucible::ExpectedStaticBytes(odd, 1); });
  const Checkpoint wrong{{{norm, Tensor{"F32", {}, 0, 256}}}, Tensor{}};
  Reject([&] { crucible::ExpectedStaticBytes(wrong, 1); });
  const Checkpoint large{{{norm, Tensor{"F16", {}, 0, SIZE_MAX - 1}}}, Tensor{}};
  Reject([&] { crucible::ExpectedStaticBytes(large, 1); });
  const Checkpoint absent{{{norm, Tensor{"MISSING", {}, 0, 0}}}, Tensor{}};
  Reject([&] { crucible::ExpectedStaticBytes(absent, 1); });
  const Checkpoint reversed{{{norm, Tensor{"F16", {}, 256, 0}}}, Tensor{}};
  Reject([&] { crucible::ExpectedStaticBytes(reversed, 1); });
  Reject([&] { crucible::ExpectedStaticBytes(ck, 0); });
  const Checkpoint padded{{{"lm_head.trellis", Tensor{"I16", {}, 0, 259}},
                           {"lm_head.suh", Tensor{"F16", {}, 0, 2}},
                           {"lm_head.svh", Tensor{"F16", {}, 0, 258}}}, Tensor{}};
  assert(crucible::ExpectedStaticBytes(padded, 48) == 1343279616 + 512);
  const std::string mlp = "model.language_model.layers.0.mlp.";
  const Checkpoint router{{{mlp + "gate.weight", Tensor{"F16", {}, 0, 2}},
                          {mlp + "shared_expert_gate.weight", Tensor{"F16", {}, 0, 2}}}, Tensor{}};
  assert(crucible::ExpectedStaticBytes(router, 48) == 1343279616 - 256);
  const Checkpoint ignored{{{"model.language_model.embed_tokens.weight", Tensor{"BF16", {}, 0, SIZE_MAX}},
                           {"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.trellis",
                            Tensor{"I16", {}, 0, SIZE_MAX}}}, Tensor{}};
  assert(crucible::ExpectedStaticBytes(ignored, 48) == 1343279616);
  // CRUCIBLE_HC_WMMA=2: 98 inject sites and 2 mixers hold int8 blocks (hc_int8.h) in place of fp16 fn and up.
  assert(crucible::ExpectedStaticBytes(ck, 48, true) == 703085056);
  Reject([&] { crucible::ExpectedStaticBytes(absent, 1, true); });
  std::cout << "static layout: 19 CPU checks PASS\n";
}
