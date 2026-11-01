// Text and start checks for the opt-in MTP draft sidecar (CRUCIBLE_MTP_SIDECAR). The full path names the user's home
// directory, so logs and captured gate evidence carry only the file name and its size.
//
// A broken drafter fails silently: verify is exact, so every token gate passes and only the speed shows the fault.
// The start therefore stops (exit 2) when the sidecar cannot be read or replaces fewer tensors than the model needs.
#pragma once
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace crucible {
inline std::string MtpSidecarLabel(const std::string& path) {
  std::error_code error;
  const auto bytes = std::filesystem::file_size(path, error);
  const std::string name = std::filesystem::path(path).filename().string();
  return name + (error ? " (size unknown)" : " (" + std::to_string(bytes) + " bytes)");
}

// The plain (not EXL3) MTP tensors that a tuned head replaces: the table of training/mtp_tune/export.py (NAMES), which
// is the only writer of sidecars. The checkpoint holds many more mtp.* tensors (the quantized experts), but a sidecar
// never carries them, so a count of all mtp.* names would be wrong. The mixer has no block_inject tensor.
inline const std::vector<std::string>& MtpSidecarTensorNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> out = {"mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
                                    "mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight",
                                    "mtp.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.shared_expert_gate.weight"};
    for (const char* module : {"mtp.layers.0.attn_hyper_connection.", "mtp.layers.0.mlp_hyper_connection.",
                               "mtp.hyper_connection_mixer."}) {
      for (const char* part : {"hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"})
        out.push_back(std::string(module) + part);
      if (std::string_view(module).find("layers") != std::string_view::npos)
        out.push_back(std::string(module) + "block_inject_weight.weight");
    }
    return out;
  }();
  return names;
}

// How many sidecar tensors the model needs: the names of the table that the checkpoint holds. has(name) asks the checkpoint.
template <class Has>
inline size_t MtpSidecarExpected(Has has) {
  size_t n = 0;
  for (const auto& name : MtpSidecarTensorNames()) n += has(name) ? 1 : 0;
  return n;
}

// The success line; tools/gates.py SIDECAR_LINE reads it.
inline std::string MtpSidecarLine(size_t replaced, const std::string& label) {
  return "mtp sidecar: " + std::to_string(replaced) + " tensors from " + label;
}

// Empty when the sidecar replaced enough tensors, else the one stderr line that explains the stop. It starts with
// "mtp sidecar error" so that no gate takes it for the success line.
inline std::string MtpSidecarFault(size_t replaced, size_t expected, const std::string& label) {
  if (replaced > 0 && replaced >= expected) return "";
  return "mtp sidecar error: " + std::to_string(replaced) + " tensors from " + label + ", expected " +
         std::to_string(expected) + " or more; the draft head would run on the wrong weights; start stopped";
}

// Empty when the file opens for reading, else the one stderr line that explains the stop (file name only).
inline std::string MtpSidecarUnreadable(const std::string& path) {
  const std::string name = std::filesystem::path(path).filename().string();
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error))
    return "mtp sidecar error: " + name + " is not a readable file (" + (error ? error.message() : "missing or not a regular file") +
           "); start stopped";
  std::ifstream in(path, std::ios::binary);
  if (!in) return "mtp sidecar error: cannot read " + name + ": " + std::strerror(errno) + "; start stopped";
  return "";
}
}  // namespace crucible
