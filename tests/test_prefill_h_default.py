"""Check the production H allocation policy without a HIP context."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PrefillHDefaultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        # The pad decision and the H_ size are in the Engine constructor; InitPrefill sizes pf_H_.
        engine = (ROOT / "engine/engine.hip").read_text()
        start = engine.index('    const char* pad_h = crucible::SwitchText("CRUCIBLE_PF_PAD_H");')
        end = engine.index("    x_ = Scratch<float>", start)
        prefill = (ROOT / "engine/runtime/prefill_engine.inc").read_text()
        pf_start = prefill.index("  if (pf_pad_h_) {")
        pf_end = prefill.index("  // Fragment-order copies", pf_start)
        # Compile the exact production blocks, so a copied policy cannot hide a regression.
        program = (f'#include "{ROOT / "engine/switches.h"}"\n#include "{ROOT / "engine/runtime/prefill_rows.h"}"\n'
                   + """
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstddef>
constexpr int kWide=10240, kPackedRows=16;
size_t cells[2]={0,0};
int calls=0, scratch_calls=0, zeros_calls=0;
float storage=0;
template<class T> T* Allocate(size_t count) { cells[calls++]=count; return &storage; }
// Counted apart: an earlier change poisons Scratch<> with 0xFF, so H_ and pf_H_ must stay Scratch, not Zeros.
template<class T> T* Zeros(size_t count) { ++zeros_calls; return Allocate<T>(count); }
template<class T> T* Scratch(size_t count) { ++scratch_calls; return Allocate<T>(count); }
int main(int, char** argv) {
  int prefill_rows_=4096, pf_launch_rows_=std::atoi(argv[1]), pf_hstride_=kWide;
  // x4-tail: the row buffers take the room rows (launch rows plus the join rows; without "+T" the launch rows).
  int pf_room_rows_=argv[2] ? std::atoi(argv[2]) : pf_launch_rows_;
  const int capacity = std::max(kPackedRows, pf_launch_rows_);
  bool pf_pad_h_=false;
  float* H_=nullptr;
  float* pf_H_=nullptr;
""" + engine[start:end] + prefill[pf_start:pf_end] + """
  std::printf("%d %zu %zu %d %d %d\\n",pf_hstride_,cells[0],cells[1],pf_H_!=nullptr && H_!=nullptr,scratch_calls,
              zeros_calls);
}
""")
        (root / "policy.cpp").write_text(program)
        cls.binary = root / "policy"
        subprocess.run([*shlex.split(os.environ.get("CXX", "g++")), "-std=c++20",
                        "-O1", "-g", "-fsanitize=address,undefined", str(root / "policy.cpp"),
                        "-o", str(cls.binary)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def check_policy(self, value, expected, launch_rows=4096, room_rows=None):
        environment = {key: item for key, item in os.environ.items() if key != "CRUCIBLE_PF_PAD_H"}
        if value is not None:
            environment["CRUCIBLE_PF_PAD_H"] = value
        room = [] if room_rows is None else [str(room_rows)]
        result = subprocess.run([str(self.binary), str(launch_rows), *room], env=environment, check=True,
                                capture_output=True, text=True)
        # The last two numbers: Scratch<> calls (H_, and pf_H_ when padded) and Zeros<> calls (none).
        scratch = 2 if expected[3] else 1
        self.assertEqual(tuple(map(int, result.stdout.split())), (*expected, scratch, 0))

    # (pf_hstride_, H_ cells, pf_H_ cells, pf_H_ exists). With pf_H_, H_ holds kPackedRows rows.
    def test_unset_pads(self):
        self.check_policy(None, (10304, 16 * 10240, 4096 * 10304, 1))

    def test_one_pads(self):
        self.check_policy("1", (10304, 16 * 10240, 4096 * 10304, 1))

    def test_8192_row_launches_keep_h_at_one_packed_pass(self):
        self.check_policy(None, (10304, 16 * 10240, 8192 * 10304, 1), launch_rows=8192)

    def test_join_rows_size_pf_h_by_the_room_rows(self):
        # CRUCIBLE_PF_LAUNCH_ROWS="8192+128" (x4-tail): pf_H_ holds the joined launch of 8,320 rows.
        self.check_policy(None, (10304, 16 * 10240, 8320 * 10304, 1), launch_rows=8192, room_rows=8320)

    def test_zero_restores_tight(self):
        self.check_policy("0", (10240, 4096 * 10240, 0, 0))

    def test_other_values_keep_legacy_policy(self):
        for value in ("", "false", "01"):
            with self.subTest(value=value):
                self.check_policy(value, (10240, 4096 * 10240, 0, 0))


if __name__ == "__main__":
    unittest.main()
