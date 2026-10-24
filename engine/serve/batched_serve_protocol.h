// Commands are synchronous. One frontend worker owns all request state changes.
#pragma once
#include "serve_protocol.h"
#include <optional>

namespace crucible {
struct BatchedServeCommand {
  std::string op;
  uint64_t request_id = 0;
  std::vector<uint64_t> request_ids;
  std::optional<ServeRequest> request;
  int every = 0;  // interleaved prefill: a verify pass for request_ids after every `every` layers of the chunk
  // W1: the request's tenant field was bad. `request` is then empty: an open is refused (recoverable), a
  // stage reads nothing. The text is ServeTenantError's fixed message.
  std::optional<std::string> refusal;
  // stage: the turn's place among the server's waiting long turns, and whether it may start a read.
  size_t position = 0;
  bool read = true;
  uint64_t job = 0;  // the server's tag of the staging turn (log lines only; 0: an older server)
};

inline BatchedServeCommand ParseBatchCommand(const std::string& line, int context) {
  st::Scanner scanner(line, "batch command", true);
  BatchedServeCommand command;
  std::set<std::string> fields;
  scanner.Expect('{');
  while (!scanner.Peek('}')) {
    const auto key = scanner.Str();
    if (!fields.insert(key).second) throw std::runtime_error("duplicate batch command field");
    scanner.Expect(':');
    if (key == "op") command.op = scanner.Str();
    else if (key == "request_id") command.request_id = scanner.Uint();
    else if (key == "every") {
      const uint64_t every = scanner.Uint();
      if (every < 1 || every > 47) throw std::runtime_error("prefill interval must be 1 to 47 layers");
      command.every = int(every);
    }
    else if (key == "request") {
      const size_t start = scanner.pos();
      scanner.Skip();
      // Batch requests must state spec; a missing field keeps the invalid -1 and fails the policy check.
      DecodePolicy explicit_spec;
      explicit_spec.spec = -1;
      try {
        command.request = ParseServeRequest(line.substr(start, scanner.pos() - start), context, explicit_spec);
      } catch (const ServeTenantError& error) {
        command.refusal = error.what();
      }
    } else if (key == "position") {
      const uint64_t position = scanner.Uint();
      if (position > 4096) throw std::runtime_error("stage position must be at most 4096");
      command.position = size_t(position);
    } else if (key == "read") {
      const uint64_t read = scanner.Uint();  // 0 or 1 (the scanner reads no JSON literals)
      if (read > 1) throw std::runtime_error("stage read must be 0 or 1");
      command.read = read == 1;
    } else if (key == "job") {
      command.job = scanner.Uint();
    } else if (key == "request_ids") {
      scanner.Expect('[');
      while (!scanner.Peek(']')) {
        if (command.request_ids.size() >= kMaxRows) throw std::runtime_error("batch command exceeds eight rows");
        command.request_ids.push_back(scanner.Uint());
        scanner.Separator(']');
      }
      scanner.Expect(']');
    } else throw std::runtime_error("unknown batch command field");
    scanner.Separator('}');
  }
  scanner.Expect('}');
  scanner.Finish();
  if (command.op == "open") {
    if (fields != std::set<std::string>{"op", "request_id", "request"} || !command.request_id ||
        (!command.request && !command.refusal))
      throw std::runtime_error("open requires a positive request ID and an explicit spec request");
  } else if (command.op == "step" || command.op == "verify_step") {
    const auto& ids = command.request_ids;
    if (fields != std::set<std::string>{"op", "request_ids"} || ids.empty() ||
        std::set<uint64_t>(ids.begin(), ids.end()).size() != ids.size() ||
        std::find(ids.begin(), ids.end(), 0) != ids.end())
      throw std::runtime_error("step and verify_step require unique positive request IDs");
  } else if (command.op == "prefill" && fields.contains("request_ids")) {
    const auto& ids = command.request_ids;
    if (fields != std::set<std::string>{"op", "request_id", "request_ids", "every"} || !command.request_id ||
        ids.empty() || std::set<uint64_t>(ids.begin(), ids.end()).size() != ids.size() ||
        std::find(ids.begin(), ids.end(), 0) != ids.end() ||
        std::find(ids.begin(), ids.end(), command.request_id) != ids.end())
      throw std::runtime_error("interleaved prefill requires its request ID, other unique decoders, and every");
  } else if (command.op == "stage") {
    // The disk tier (state_disk.h): start the read of a disk state for this request before its open.
    // `position`, `read` and `job` are optional (an older server sends none: the next turn, may read, no tag).
    fields.erase("position");
    fields.erase("read");
    fields.erase("job");
    if (fields != std::set<std::string>{"op", "request"} || (!command.request && !command.refusal))
      throw std::runtime_error("stage requires a request and no request ID");
  } else if (command.op == "close" || command.op == "prefill" || command.op == "spec_step") {
    if (fields != std::set<std::string>{"op", "request_id"} || !command.request_id)
      throw std::runtime_error("close and prefill require a positive request ID");
  } else throw std::runtime_error("batch op must be open, prefill, step, verify_step, spec_step, stage, or close");
  return command;
}
}  // namespace crucible
