// Bulk prefill launches that merge grid chunks (prefill-moe lesson 1).
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace crucible {
// Most rows in one bulk prefill launch. Every layer's GDN and QSA kernels still
// run in kPrefillCoreRows parts; the GEMMs (MoE experts, dense projections)
// see all rows of the launch, so each decoded expert weight serves more rows.
constexpr int kMaxPrefillLaunchRows = 32768;
// Rows of one GDN or QSA core part inside a launch (engine.hip kPrefillCoreRows). A merge needs a grid of whole
// parts: the core scratch holds min(grid, parts) rows, and a launch cuts its parts from its own start.
constexpr int kPrefillLaunchCoreRows = 2048;
// Rows of a launch when CRUCIBLE_PF_LAUNCH_ROWS is unset: two chunks of the 4,096-row grid. Bitwise equal to one
// chunk a launch (1K to 128K prompts), and 55 ms (32K) to 78 ms (128K) less wall per 4,096 rows for 1,677 MiB of
// stage arena. CRUCIBLE_PF_LAUNCH_ROWS=4096 turns it off. Only the 4,096-row grid merges by default: no other grid was
// measured (a 2048-row grid keeps one chunk a launch unless the variable merges it).
constexpr int kDefaultPrefillLaunchRows = 8192;
// Join rows when CRUCIBLE_PF_LAUNCH_ROWS is unset (x4-tail): a cold prompt's last partial chunk
// of up to this many rows joins the last launch, so the default path is the daily path ("8192+128"). A set value
// without "+T" ("8192") has no join; "4096" turns the merge and the join off.
constexpr int kDefaultPrefillJoinRows = 128;
constexpr int kDefaultPrefillLaunchGrid = 4096;

// Rows of the bulk launch at prompt position `at` (prompt end `end`). The grid
// schedule takes min(step, end - at) rows a chunk. A launch merges whole grid
// chunks while it stays within `launch` rows and its end stays at or before
// `stop` (the first position that must end a chunk: a snapshot frontier or
// the MTP warm-up start). A partial chunk is always the last one of a
// launch, so a launch ends only on grid chunk ends and the GDN and QSA core
// cuts stay where the grid schedule puts them.
inline int PrefillLaunchRows(size_t at, size_t end, int step, int launch, size_t stop) {
  int rows = int(std::min<size_t>(size_t(step), end - at));
  while (rows % step == 0 && at + size_t(rows) < end) {
    const int next = int(std::min<size_t>(size_t(step), end - at - size_t(rows)));
    if (rows + next > launch || at + size_t(rows + next) > stop) break;
    rows += next;
  }
  return rows;
}

// An engine's launch rows; engines without merged launches take one grid chunk.
template <class E> int EngineLaunchRows(const E& engine) {
  if constexpr (requires { engine.PrefillLaunchRows(); }) return engine.PrefillLaunchRows();
  else return engine.PrefillRows();
}

// x4-tail: an engine's join rows; 0 for engines without joins.
template <class E> int EngineJoinRows(const E& engine) {
  if constexpr (requires { engine.PrefillJoinRows(); }) return engine.PrefillJoinRows();
  else return 0;
}

// The launch rows of CRUCIBLE_PF_LAUNCH_ROWS (`value`, nullptr when unset) for
// grid chunks of `grid` rows: unset gives kDefaultPrefillLaunchRows for the
// grid kDefaultPrefillLaunchGrid and the grid itself for every other grid; a
// set value must be a multiple of the grid in [grid, kMaxPrefillLaunchRows],
// and above the grid only for a grid of whole core parts. Returns -1 for an
// invalid value, and 0 without bulk prefill (grid 0).
// x4-tail: a value "L+T" also sets the join rows T (PrefillJoinSetting); L follows the
// rules above.
inline int PrefillLaunchSetting(const char* value, int grid) {
  if (grid <= 0) return 0;
  if (!value) return grid == kDefaultPrefillLaunchGrid ? kDefaultPrefillLaunchRows : grid;
  errno = 0;
  char* tail = nullptr;
  const long rows = std::strtol(value, &tail, 10);
  if (*tail == '+') {
    char* join_end = nullptr;
    const long join = std::strtol(tail + 1, &join_end, 10);
    // The join needs merged launches, and the launch with its join rows must fit kMaxPrefillLaunchRows.
    if (tail[1] < '0' || tail[1] > '9' || *join_end || join < 1 || join >= grid || rows <= grid ||
        rows + join > kMaxPrefillLaunchRows)
      return -1;
    tail = join_end;
  }
  if (!*value || *tail || errno || rows < grid || rows > kMaxPrefillLaunchRows || rows % grid) return -1;
  if (rows > grid && grid % kPrefillLaunchCoreRows) return -1;
  return int(rows);
}

// x4-tail: the join rows of CRUCIBLE_PF_LAUNCH_ROWS="L+T": T, or 0 when the value has no "+T". Unset gives
// kDefaultPrefillJoinRows for the grid kDefaultPrefillLaunchGrid, so the default launch is "8192+128". Only a valid
// setting has join rows (PrefillLaunchSetting).
inline int PrefillJoinSetting(const char* value, int grid) {
  if (!value) return grid == kDefaultPrefillLaunchGrid ? kDefaultPrefillJoinRows : 0;
  if (PrefillLaunchSetting(value, grid) <= 0) return 0;
  const char* plus = value;
  while (*plus && *plus != '+') ++plus;
  return *plus ? int(std::strtol(plus + 1, nullptr, 10)) : 0;
}

// The rows that the engine's launch buffers hold: the launch rows, and with a join the join rows too. Join 0 gives
// the launch rows (the bytes before x4-tail; CRUCIBLE_PF_LAUNCH_ROWS=8192 gives them back).
inline int PrefillRoomRows(int launch, int join) { return join > 0 ? launch + join : launch; }

// A tail of at most this many rows runs the small-row kernels alone (Engine::Prefill: fp32 x, PleMixKernel), and in
// a launch it would run the bulk kernels. That is not proved bitwise, so such a tail never joins (engine kMaxRows).
constexpr int kPrefillJoinSmallRows = 8;

// x4-tail: the plan of one bulk launch with joins. The grid schedule, the core
// cuts and every row's arithmetic stay; only the launch boundaries move.
struct PrefillJoinInput {
  size_t at = 0, end = 0;
  int step = 0, launch = 0;
  int join = 0;               // join rows: the prompt's last partial chunk joins when the launch stays in launch + join
  size_t first = SIZE_MAX;    // the next capture point after `at`
  size_t second = SIZE_MAX;   // the capture point after `first`
  bool cut = false;           // a cut buffer can save the state at `first` while the launch goes on
  size_t warm = SIZE_MAX;     // a stop that stays (the MTP warm-up start when it is not lifted); SIZE_MAX: none
};
struct PrefillJoin {
  int rows = 0;
  size_t cut = 0;  // the capture point inside the launch whose state the cut buffer saves; 0: none
};
inline PrefillJoin PrefillJoinPlan(const PrefillJoinInput& in) {
  if (in.step <= 0 || in.at >= in.end) return {};  // no rows: the caller checks a grid and rows left first
  // A cut is a core part end: `first` must be whole grid chunks from the launch start (a grid of whole core parts,
  // PrefillLaunchSetting) and before the stops that stay.
  const bool pass = in.cut && in.first != SIZE_MAX && in.first > in.at && in.first < in.end && in.first < in.warm &&
                    (in.first - in.at) % size_t(in.step) == 0;
  const size_t stop = std::min(pass ? in.second : in.first, in.warm);
  int rows = PrefillLaunchRows(in.at, in.end, in.step, in.launch, stop);
  const size_t next = in.at + size_t(rows), left = in.end - next;
  // The prompt's last partial chunk joins the launch before it (only at the prompt end: a later chunk keeps its
  // grid). PrefillLaunchRows has merged it already when it fits the launch rows.
  if (in.join > 0 && rows % in.step == 0 && left > size_t(kPrefillJoinSmallRows) && left < size_t(in.step) &&
      size_t(rows) + left <= size_t(in.launch) + size_t(in.join) && in.end <= stop)
    rows += int(left);
  return {rows, pass && in.at + size_t(rows) > in.first ? in.first : 0};
}

// The start of the last launch that the plan without joins runs before `to` (a stop of that plan or the prompt
// end), from `at`. `stop(cursor)` is that plan's stop at each cursor. The retained ledger of a launch records its
// start (RetainedPrefix::window_start), and the snapshot holds it, so a joined launch reports this start: the saved
// bytes stay those of the plan without joins.
template <class Stop>
size_t PrefillPlainStart(size_t at, size_t to, size_t end, int step, int launch, Stop stop) {
  while (true) {
    const int rows = step > 0 && at < end ? PrefillLaunchRows(at, end, step, launch, stop(at)) : 0;
    const size_t next = at + size_t(rows);
    if (rows <= 0 || next >= to || next >= end) return at;
    at = next;
  }
}

// x4-tail: the MTP warm-up of a joined launch runs once per grid chunk of the launch (offset from the launch start,
// rows), as the plan without joins runs it: those launches hold warm-up rows of one grid chunk at most.
inline std::vector<std::pair<int, int>> PrefillGridPieces(int rows, int step) {
  std::vector<std::pair<int, int>> pieces;
  if (step <= 0) return {{0, rows}};
  for (int offset = 0; offset < rows; offset += step) pieces.emplace_back(offset, std::min(step, rows - offset));
  return pieces;
}

// x4-tail: the device copies of one cut (Engine::SetPrefillJoin). Each source is a span of the active sequence
// (SlotCheckpointIO::Spans at the cut); the engine copies it once, when the span holds its value at the cut.
struct PrefillCutCopy {
  const void* source = nullptr;
  void* target = nullptr;
  size_t bytes = 0;
};
struct PrefillJoinSpec {
  int ledger = 0;  // rows from the launch start to the start of the plain plan's last launch (0: the launch start)
  int cut = 0;     // rows from the launch start to the cut (0: no cut)
  std::vector<PrefillCutCopy> copies;
};

// The checks of Engine::BeginPrefillJoin. The ledger is a grid point of the launch. A cut is a core part end inside the
// launch, at least 16 rows in (the PLE ring at the cut holds the launch's own rows), with copies; no cut, no copies.
// Each copy names a distinct span of `spans` (the sequence's state spans) with its size, and has a target.
inline bool PrefillJoinValid(const PrefillJoinSpec& spec, int rows, int grid,
                             const std::vector<std::pair<const void*, size_t>>& spans) {
  if (grid <= 0 || rows <= 0 || spec.ledger < 0 || spec.ledger >= rows || spec.ledger % grid) return false;
  if (!spec.cut) return spec.copies.empty();
  if (spec.cut < 16 || spec.cut >= rows || spec.cut % kPrefillLaunchCoreRows || spec.copies.empty()) return false;
  for (size_t i = 0; i < spec.copies.size(); ++i) {
    const auto& copy = spec.copies[i];
    if (!copy.target || !copy.source) return false;
    bool known = false;
    for (const auto& [pointer, bytes] : spans) known = known || (pointer == copy.source && bytes == copy.bytes);
    if (!known) return false;
    for (size_t j = 0; j < i; ++j)
      if (spec.copies[j].source == copy.source) return false;
  }
  return true;
}
}  // namespace crucible
