"""The engine's choice between a whole-launch PLE gather and the piece path, without a HIP context.

The production blocks of engine/engine.hip (the stage pick in the constructor, the stage rows, PleRowsUploaded,
FinishPleGather, GatherPle and WithPleStage) are compiled against a fake host table and a fake pinned stage. A daily
BF16 launch (16384 rows and the 128-row join) must start its gather at the row
upload and commit the stage once. The old piece path (FillTo once a piece while the device waits) fails these tests
(review prefill-stack-review MEDIUM 1, mutant B).
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def block(text, start, end):
    first = text.index(start)
    return text[first:text.index(end, first)]


PROGRAM = r"""
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace crucible;

struct Stats { std::atomic<long long> hint_ns{0}, copy_ns{0}, fresh_pages{0}, major_faults{0}; };
struct FakeHost {
  int starts = 0, gathers = 0;
  size_t started = 0;
  Stats st;
  void Start(const int64_t*, size_t n, uint8_t*) { ++starts; started = n; }
  void Wait() {}
  void Gather(const int64_t*, size_t, uint8_t*) { ++gathers; }
  Stats& stats() { return st; }
  int Fetch() { return 0; }
};
struct FakePle {
  FakeHost* host = nullptr;
  bool bf16 = false;
  int K = 6;
  int RowWords() const { return bf16 ? kPleRowDim : 10 * K + 1; }
};
struct FakeModel { FakePle ple; };

int begins = 0, commits = 0, fills = 0, stage_words = 0;
uint16_t scratch[16];
template <class T> T* Zeros(size_t) { return reinterpret_cast<T*>(scratch); }
// The interface of PinnedInputBuffer (engine/gfx1151/pinned_buffer.h) with no pinned memory.
template <typename T, size_t N>
struct PinnedInputBuffer {
  PinnedInputBuffer() { stage_words = (int)(N / ((size_t)kMaxPrefillLaunchRows * 16)); }
  T* Begin(size_t count) {
    if (count > N) { std::fprintf(stderr, "pinned input capacity exceeded\n"); std::abort(); }
    ++begins;
    return reinterpret_cast<T*>(scratch);
  }
  void Commit(T*, size_t) { ++commits; }
  template <typename Fill> void FillTo(T* device, size_t count, Fill fill) { ++fills; fill(Begin(count)); Commit(device, count); }
};

struct Engine {
  FakeModel m_;
  uint16_t* ple_packed_ = nullptr;
  std::vector<int64_t> ple_host_rows_;
  int ple_host_count_ = 0;
  std::chrono::steady_clock::time_point ple_gather_start_;
  const bool ple_trace_ = false;
@MEMBERS@
  void Construct(int capacity) {
@CTOR@
  }
@METHODS@
};

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const int bf16 = std::atoi(argv[1]), R = std::atoi(argv[2]);
  FakeHost host;
  Engine e;
  e.m_.ple.host = &host;
  e.m_.ple.bf16 = bf16 != 0;
  e.Construct(R);
  std::vector<int64_t> rows((size_t)R * 16, 0);
  e.PleRowsUploaded(rows.data(), R * 16);
  const int started = host.starts;
  e.GatherPle(R);
  std::printf("words=%d started=%d rows=%zu begins=%d commits=%d fills=%d gathers=%d\n", stage_words, started,
              host.started, begins, commits, fills, host.gathers);
  return 0;
}
"""


class PleStageWholeLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        engine = (ROOT / "engine/engine.hip").read_text()
        members = block(engine, "  static constexpr int kPleStageRows = ", "  std::vector<int64_t> ple_host_rows_;")
        ctor = block(engine, "      ple_packed_ = Zeros<uint16_t>(", "    ids_ = Zeros<int>(kPackedRows);")
        ctor = ctor[:ctor.rindex("}")]  # the close of `if (m_.ple.host) {`
        methods = block(engine, "  void PleRowsUploaded(", "  template <int K>\n  void LaunchPleGather(")
        program = (f'#include "{ROOT / "engine/model/flashnext/ple_stage.h"}"\n'
                   + PROGRAM.replace("@MEMBERS@", members).replace("@CTOR@", ctor).replace("@METHODS@", methods))
        (root / "stage.cpp").write_text(program)
        cls.binary = root / "stage"
        subprocess.run([*shlex.split(os.environ.get("CXX", "g++")), "-std=c++20", "-O1", "-g",
                        "-fsanitize=address,undefined", str(root / "stage.cpp"), "-o", str(cls.binary)],
                       check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def run_launch(self, bf16, rows):
        result = subprocess.run([str(self.binary), str(int(bf16)), str(rows)], check=True, capture_output=True,
                                text=True, timeout=60)
        return dict(item.split("=") for item in result.stdout.split())

    def test_daily_bf16_launch_is_gathered_whole_at_upload(self):
        # The daily room: 16384 launch rows and the 128-row join.
        out = self.run_launch(True, 16384 + 128)
        self.assertEqual(out["words"], "160")  # the BF16 stage, not a K-bit one
        self.assertEqual((out["started"], out["rows"]), ("1", str((16384 + 128) * 16)))
        self.assertEqual((out["begins"], out["commits"], out["fills"], out["gathers"]), ("1", "1", "0", "0"))

    def test_largest_bf16_launch_is_whole(self):
        out = self.run_launch(True, 32768)
        self.assertEqual((out["started"], out["commits"], out["fills"]), ("1", "1", "0"))

    def test_bf16_launch_above_the_old_4096_rows_is_whole(self):
        # The stage held 4,096 BF16 rows before prefill-plestage; 4,097 rows then took 2 pieces.
        out = self.run_launch(True, 4097)
        self.assertEqual((out["started"], out["commits"], out["fills"]), ("1", "1", "0"))

    def test_kbit_launch_is_whole(self):
        out = self.run_launch(False, 16384 + 128)
        self.assertEqual(out["words"], "61")
        self.assertEqual((out["started"], out["commits"], out["fills"]), ("1", "1", "0"))


if __name__ == "__main__":
    unittest.main()
