// Strict opt-in batch options are separate from the serial CLI and sampler.
#pragma once

#include <filesystem>
#include <charconv>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../runtime/bounds.h"
#include "../model/flashnext/geometry.h"

namespace crucible {
struct BatchOptions {
  int streams = 1, n = 32, chunk = 8, ctx = 32768, prefill_rows = 0;
  std::string dump;
  std::vector<std::vector<int>> prompts;
#if CRUCIBLE_DEBUG_NONFINITE & 4
  std::string inject;  // --inject KIND,LAYER,POS,STREAM (debug build only)
#endif
};

inline int BatchInteger(std::string_view text) {
  int value;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size())
    throw std::invalid_argument("batch integer option has an invalid value");
  return value;
}

inline std::string_view BatchTrim(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return {};
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

inline std::vector<int> BatchIds(std::string_view text) {
  std::vector<int> ids;
  while (true) {
    const auto comma = text.find(',');
    const auto item = BatchTrim(text.substr(0, comma));
    if (item.empty()) throw std::invalid_argument("batch token list has an empty field");
    ids.push_back(BatchInteger(item));
    if (comma == std::string_view::npos) return ids;
    text.remove_prefix(comma + 1);
  }
}

inline std::map<std::string, std::string> BatchPairs(int argc, char** argv) {
  if ((argc - 3) % 2 != 0) throw std::invalid_argument("batch options need values");
  std::map<std::string, std::string> options;
  for (int i = 3; i < argc; i += 2) {
    const std::string key = argv[i];
    if (key != "--streams" && key != "--batch-list" && key != "--ids" && key != "--ids-file" &&
        key != "--n" && key != "--chunk" && key != "--ctx" && key != "--dump" && key != "--spec" &&
        key != "--prefill"
#if CRUCIBLE_DEBUG_NONFINITE & 4
        && key != "--inject"
#endif
        )
      throw std::invalid_argument("unsupported batch option: " + key);
    if (!options.emplace(key, argv[i + 1]).second)
      throw std::invalid_argument("duplicate batch option: " + key);
  }
  return options;
}

inline std::string BatchReadFile(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::invalid_argument("cannot open batch input: " + std::filesystem::path(path).filename().string());
  std::stringstream text;
  text << file.rdbuf();
  if (file.bad()) throw std::invalid_argument("cannot read batch input: " + std::filesystem::path(path).filename().string());
  return text.str();
}

inline std::vector<std::vector<int>> BatchPrompts(const std::map<std::string, std::string>& args, int streams) {
  const int sources = args.contains("--batch-list") + args.contains("--ids") + args.contains("--ids-file");
  if (sources != 1 || (args.contains("--batch-list") && args.contains("--streams")))
    throw std::invalid_argument("use one prompt source; --batch-list sets the stream count");
  if (!args.contains("--batch-list")) {
    const std::string text = args.contains("--ids") ? args.at("--ids") : BatchReadFile(args.at("--ids-file"));
    return std::vector<std::vector<int>>(streams, BatchIds(text));
  }
  std::vector<std::vector<int>> prompts;
  std::stringstream lines(BatchReadFile(args.at("--batch-list")));
  for (std::string line; std::getline(lines, line);) {
    if (BatchTrim(line).empty()) continue;
    if (prompts.size() >= kMaxRows) throw std::invalid_argument("batch list has too many streams");
    prompts.push_back(BatchIds(line));
  }
  if (prompts.empty()) throw std::invalid_argument("batch list is empty");
  return prompts;
}

inline std::string ParseBatchOptions(int argc, char** argv, BatchOptions& out) {
  try {
    const auto args = BatchPairs(argc, argv);
    BatchOptions next;
    for (const auto& [key, value] : args) {
      if (key == "--n") next.n = BatchInteger(value);
      else if (key == "--ctx") next.ctx = BatchInteger(value);
      else if (key == "--chunk") next.chunk = BatchInteger(value);
      else if (key == "--streams") next.streams = BatchInteger(value);
      else if (key == "--prefill") next.prefill_rows = BatchInteger(value);
      else if (key == "--dump") next.dump = value;
#if CRUCIBLE_DEBUG_NONFINITE & 4
      else if (key == "--inject") next.inject = value;
#endif
      else if (key == "--spec" && BatchInteger(value) != 0)
        throw std::invalid_argument("batch speculation is not supported yet; use --spec 0");
    }
    if (next.streams < 1 || next.streams > kMaxRows || next.chunk < 1 || next.chunk > kMaxRows ||
        next.n < 1 || next.ctx < 1 || next.ctx > 1048576 || next.prefill_rows < 0 || next.prefill_rows > 4096)
      throw std::invalid_argument("invalid batch size, chunk, output count, or context (maximum 1048576)");
    next.prompts = BatchPrompts(args, next.streams);
    next.streams = (int)next.prompts.size();
    for (const auto& ids : next.prompts) {
      if (!ValidTokenIds(ids, kVocab)) throw std::invalid_argument("batch token IDs must be in [0, 248320)");
      if (!RequestFitsContext(ids.size(), next.n, next.ctx))
        throw std::invalid_argument("each batch prompt and output must fit --ctx");
    }
    out = std::move(next);
    return "";
  } catch (const std::invalid_argument& error) {
    return error.what();
  }
}
}  // namespace crucible
