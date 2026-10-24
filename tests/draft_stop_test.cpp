#include "../engine/runtime/draft_stop.h"
#include <cassert>
using namespace crucible;

int main() {
  const std::vector<int> stops{8, 9};
  // The room is the most drafts a step may carry: it commits drafts + 1 tokens at most.
  assert(DraftRoom(0) == 0 && DraftRoom(1) == 0 && DraftRoom(2) == 1 && DraftRoom(9) == 8);
  assert(DraftRoom(size_t(1) << 40) > 0);
  assert(IsStopToken(stops, 8) && IsStopToken(stops, 9) && !IsStopToken(stops, 7) && !IsStopToken({}, 8));

  // A copy keeps its first stop ID and drops everything after it.
  std::vector<int> copy{4, 5, 8, 6, 7};
  TrimDrafts(copy, stops, 5);
  assert((copy == std::vector<int>{4, 5, 8}));
  copy = {9, 5, 6};
  TrimDrafts(copy, stops, 3);
  assert((copy == std::vector<int>{9}));
  // The room cuts a copy without a stop ID, and a stop ID past the room is already cut.
  copy = {1, 2, 3, 4, 5};
  TrimDrafts(copy, stops, 3);
  assert((copy == std::vector<int>{1, 2, 3}));
  copy = {1, 2, 3, 8, 5};
  TrimDrafts(copy, stops, 2);
  assert((copy == std::vector<int>{1, 2}));
  // Nothing to cut, no room, and no stop IDs leave the right result.
  copy = {1, 2};
  TrimDrafts(copy, stops, 2);
  assert((copy == std::vector<int>{1, 2}));
  TrimDrafts(copy, stops, 0);
  assert(copy.empty());
  copy = {1, 8, 3};
  TrimDrafts(copy, {}, 3);
  assert((copy == std::vector<int>{1, 8, 3}));
  copy = {1, 2};
  TrimDrafts(copy, stops, -1);
  assert(copy.empty());

  // A chain grows only below the room and while its last draft is not a stop ID.
  assert(ChainMayGrow({}, stops, 1));
  assert(!ChainMayGrow({}, stops, 0));
  const std::vector<int> two{1, 2}, ending{1, 8};
  assert(ChainMayGrow(two, stops, 3) && !ChainMayGrow(two, stops, 2));
  assert(!ChainMayGrow(ending, stops, 5));
  // The chain and the trim agree: growing step by step gives the trimmed copy.
  const std::vector<int> source{4, 5, 6, 9, 7, 8};
  for (int room = 0; room <= 7; ++room) {
    std::vector<int> grown;
    while (ChainMayGrow(grown, stops, room) && grown.size() < source.size()) grown.push_back(source[grown.size()]);
    std::vector<int> trimmed = source;
    TrimDrafts(trimmed, stops, room);
    assert(grown == trimmed);
  }
  return 0;
}
