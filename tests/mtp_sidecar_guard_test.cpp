// The engine start guard for CRUCIBLE_MTP_SIDECAR: a sidecar that replaces fewer tensors than the model needs, or
// that cannot be read, stops the start (exit 2). With exact verify a broken drafter passes every token gate and
// only the speed shows it, so the check is at the start.
#include "../engine/model/flashnext/mtp_sidecar.h"
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>

int main() {
  using namespace crucible;
  // The names that training/mtp_tune/export.py writes: 6 norms and gates, and 4 hyper-connection tensors
  // for each of 3 modules (the mixer has no block_inject).
  const auto& names = MtpSidecarTensorNames();
  assert(names.size() == 17);
  assert(std::set<std::string>(names.begin(), names.end()).size() == names.size());
  for (const auto& n : names) assert(n.rfind("mtp.", 0) == 0 && n.size() > 11 && n.ends_with(".weight"));

  // The expected count is the number of those names that the checkpoint holds.
  std::set<std::string> held(names.begin(), names.end());
  held.insert("mtp.fc_hidden.trellis");  // quantized tensors do not count
  const auto has = [&](const std::string& n) { return held.count(n) != 0; };
  assert(MtpSidecarExpected(has) == 17);
  held.erase(names[0]);
  assert(MtpSidecarExpected(has) == 16);
  assert(MtpSidecarExpected([](const std::string&) { return false; }) == 0);

  // The guard: ok when the sidecar replaces at least the expected count, a fault line otherwise.
  assert(MtpSidecarFault(17, 17, "side.safetensors (42 bytes)").empty());
  assert(MtpSidecarFault(18, 17, "side.safetensors (42 bytes)").empty());
  const std::string zero = MtpSidecarFault(0, 17, "side.safetensors (42 bytes)");
  assert(!zero.empty() && zero.find('\n') == std::string::npos);
  assert(zero.find("0 tensors") != std::string::npos && zero.find("17") != std::string::npos);
  const std::string few = MtpSidecarFault(5, 17, "side.safetensors (42 bytes)");
  assert(few.find("5 tensors") != std::string::npos && few.find("17") != std::string::npos);
  assert(!MtpSidecarFault(1, 17, "x").empty() && MtpSidecarFault(1, 1, "x").empty());
  // An expected count of 0 (a checkpoint with no MTP tensors) still stops an empty sidecar.
  assert(!MtpSidecarFault(0, 0, "x").empty());

  // The success line keeps the text that tools/gates.py SIDECAR_LINE reads; no fault line may match it.
  assert(MtpSidecarLine(17, "side.safetensors (42 bytes)") == "mtp sidecar: 17 tensors from side.safetensors (42 bytes)");
  assert(zero.rfind("mtp sidecar: ", 0) != 0 && few.rfind("mtp sidecar: ", 0) != 0);

  // A sidecar that is missing or cannot be read.
  char dir[] = "/tmp/crucible-sidecar-guard-XXXXXX";
  assert(mkdtemp(dir));
  const std::string ok = std::string(dir) + "/ok.safetensors";
  std::ofstream(ok, std::ios::binary) << "x";
  assert(MtpSidecarUnreadable(ok).empty());
  const std::string gone = std::string(dir) + "/gone.safetensors";
  const std::string missing = MtpSidecarUnreadable(gone);
  assert(!missing.empty() && missing.find('\n') == std::string::npos);
  assert(missing.find(dir) == std::string::npos);  // a path names the user
  assert(!MtpSidecarUnreadable(dir).empty());      // a directory is not a file
  std::filesystem::remove_all(dir);
  std::puts("PASS: sidecar start guard stops on a short, empty or unreadable sidecar");
}
