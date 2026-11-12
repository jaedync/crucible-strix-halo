#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../engine/runtime/draft_warmup.h"
#include "../engine/model/flashnext/prefill_launch.h"

// a bounded MTP slot holds more than one prefill launch (batched_serve.h FitMtp).
static_assert(crucible::kMtpSlotOutputRows > crucible::kMaxPrefillLaunchRows);

void Check(int length, int chunk, int limit) {
  const size_t start = crucible::DraftWarmupStart(length, limit);
  std::vector<std::pair<int, int>> actual, expected;
  int carry = -1;
  for (int i = 0; i < length; i += chunk) {
    const int rows = std::min(chunk, length - i);
    const int offset = crucible::DraftWarmupRow(i, rows, start);
    if (offset == rows) continue;
    const bool have_carry = carry >= 0;
    const int pairs = rows - 1 - offset + (have_carry ? 1 : 0);
    for (int p = 0; p < pairs; p += 8) {
      const int count = std::min(8, pairs - p);
      for (int j = 0; j < count; ++j) {
        const bool from_carry = p == 0 && j == 0 && have_carry;
        const int source = from_carry ? carry : i + offset + p + j - (have_carry ? 1 : 0);
        const int token = i + offset + p + j + (have_carry ? 0 : 1);
        actual.emplace_back(source, token);
      }
    }
    carry = i + rows - 1;
  }
  for (int i = (int)start; i + 1 < length; ++i) expected.emplace_back(i, i + 1);
  assert(actual == expected);
  assert(carry == length - 1);
}

int main() {
  for (int length : {1, 2, 7, 8, 9, 24, 33, 127, 128, 129, 4096, 16384})
    for (int chunk : {1, 2, 8, 24, 64, 128, 4096})
      for (int limit : {0, 1, 2, 7, 8, 64, 128, 256, length + 1})
        Check(length, chunk, limit);
  // a bounded window needs its warm-up, one client output (32,768 tokens) and draft slack; window 0 (the whole
  // prompt, earlier changes) needs the context.
  assert(crucible::kMtpSlotOutputRows == 33024);
  assert(crucible::MtpSlotRows(196608, 256) == 33280);
  assert(crucible::MtpSlotRows(106496, 256) == 33280);
  assert(crucible::MtpSlotRows(262144, 256) == 33280);
  assert(crucible::MtpSlotRows(32768, 256) == 32768);
  assert(crucible::MtpSlotRows(33280, 256) == 33280);
  assert(crucible::MtpSlotRows(8, 256) == 8);
  assert(crucible::MtpSlotRows(196608, 0) == 196608);
  assert(crucible::MtpSlotRows(262144, 0) == 262144);
  assert(crucible::MtpSlotRows(262144, 65536) == 65536 + 33024);
  assert(crucible::MtpSlotRows(262144, 262144) == 262144);
  // Y2: a long slot (YaRN, up to 1,048,576 positions) keeps the same bounded MTP cache; window 0 keeps its context.
  assert(crucible::MtpSlotRows(524288, 256) == 33280);
  assert(crucible::MtpSlotRows(524288, 0) == 524288);
  assert(crucible::MtpSlotRows(1048576, 0) == 1048576);
  assert(crucible::MtpSlotRows(524288, 262144) == 262144);
  for (int window : {1, 7, 255, 256, 4096})
    for (int ctx : {1, 128, 33279, 33280, 33281, 106496, 196608, 262144}) {
      const int rows = crucible::MtpSlotRows(ctx, window);
      assert(rows >= 1 && rows <= ctx);
      assert(rows == ctx || rows >= window + 32768 + 2 * 8);
    }
  for (const auto& [ctx, window] : {std::pair{0, 256}, {1048577, 256}, {1024, -1}, {1024, 262145}}) {
    bool threw = false;
    try { crucible::MtpSlotRows(ctx, window); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
  }
}
