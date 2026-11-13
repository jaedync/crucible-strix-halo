// Batch protocol fixture with real host-cache admission and synthetic prefill state.
#include "../../../engine/serve/batch_conversation_cache.h"
#include "../../../engine/serve/batch_mtp_history.h"
#include "../../../engine/serve/batched_serve_protocol.h"
#include <cassert>
#include <iostream>

using namespace crucible;
struct Snapshot {
  std::vector<int> hist;
  ConversationResume resume;
};

int main(int argc, char** argv) {
  constexpr int context = 131072, rows = 4096;
  const bool stable = argc > 2 && std::string_view(argv[2]) == "stable";
  BatchConversationCache<Snapshot> cache(128, {}, 2);
  ConversationPolicy policy;
  policy.prefill_rows = rows;
  std::cout << R"({"event":"ready","context":131072,"streams":2,"prefill_rows":4096,)"
            << R"("slot_contexts":[131072,131072],"protocol":"batch-v1","line_bytes":2097152)"
            << (stable ? R"(,"stable_prefix":true)" : "") << "}\n" << std::flush;

  ServeRequest request;
  size_t generation = 0, cursor = 0, restored = 0;
  std::vector<CacheFrontier> points;
  for (std::string line; std::getline(std::cin, line);) {
    const auto command = ParseBatchCommand(line, context);
    const auto id = command.request_id;
    if (command.op == "open") {
      request = *command.request;
      assert(request.max_tokens == 1 && !request.spec);
      if (!stable) assert(!request.stable_prefix_tokens);
      points = BatchCacheFrontiers(request.ids.size(), request.cache_prefix_tokens, rows,
                                  request.shared_prefix_tokens, request.stable_prefix_tokens);
      for (const auto& point : points)
        if (point.stable) cache.KeepStable(std::span(request.ids).first(point.position), policy);
      const auto hit = request.cache_control == "auto"
          ? cache.Find(request.ids, policy, request.cache_prefix_tokens, BatchRestoreLimit(request), rows) : nullptr;
      cursor = restored = hit ? hit->snapshot->hist.size() : 0;
      std::cout << R"({"event":"opened","slot_context":131072,"cached_tokens":)" << restored
                << R"(,"request_id":)" << id << R"(,"slot_generation":)" << ++generation << "}\n";
    } else if (command.op == "prefill") {
      cursor = std::min(cursor + rows, request.ids.size());
      if (request.cache_control != "bypass") {
        for (const auto& point : points) {
          if (point.position != cursor) continue;
          // At this reserve even evicting both 64-byte entries cannot make room for the save.
          const auto available = generation == 4 ? cache.HostReserve() - 65 : cache.HostReserve() + 64;
          bool attempted = false;
          const bool saved = cache.Capture(request.ids, policy, point, 64, available, [&] {
            attempted = true;
            if (generation == 3 && point.stable) throw std::bad_alloc();
            return std::make_shared<Snapshot>(Snapshot{
                {request.ids.begin(), request.ids.begin() + cursor},
                BatchFrontierResume(cursor, request.cache_prefix_tokens)});
          });
          if (point.stable && (generation == 3 || generation == 4)) {
            assert(!saved && attempted == (generation == 3));
            assert(std::string_view(cache.LastRefusal()) == (attempted ? "memory" : "host"));
            std::cerr << "stable capture refused: " << cache.LastRefusal() << " at " << cursor << '\n';
          }
        }
      }
      if (cursor == request.ids.size()) {
        std::cout << R"({"event":"token","id":32,"request_id":)" << id << R"(,"slot_generation":)"
                  << generation << "}\n";
        std::cout << R"({"event":"done","finish_reason":"length","completion_tokens":1,"prompt_tokens":)"
                  << cursor << R"(,"cached_tokens":)" << restored << R"(,"prefill_tokens":)" << cursor - restored
                  << R"(,"request_id":)" << id << R"(,"slot_generation":)" << generation << "}\n";
      }
    }
    std::cout << R"({"event":"batch_end","op":)" << JsonQuote(command.op);
    if (command.op == "prefill") std::cout << R"(,"cursor":)" << cursor;
    std::cout << "}\n" << std::flush;
  }
}
