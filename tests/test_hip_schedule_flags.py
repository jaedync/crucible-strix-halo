"""The engine never changes the HIP scheduling mode.

ROCm 7.2 CLR creates the signals of a queue in the mode of that time: with the default active wait they have no
interrupt. After hipSetDeviceFlags(hipDeviceScheduleBlockingSync), CLR no longer swaps in an interrupt signal for a
command that needs an async completion handler, and ROCr's hsa_amd_signal_async_handler rejects a signal without an
interrupt (HSA_STATUS_ERROR_INVALID_SIGNAL). The handler is then never set, and hipDeviceSynchronize waits forever on
a GPU that is idle (a hang during a long soak run: Engine::Prefill -> hipDeviceSynchronize -> HostQueue::finish ->
Event::awaitCompletion). A flag change is safe only before the first HIP call, and the engine makes none there.
Comments and strings do not count.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model_geometry import blank_lexemes, source_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Every program that runs engine code: the engine, its benches and its GPU tests.
SCANNED = {"engine", "bench", "tests"}
CALL = re.compile(r"\b(?:hipSetDeviceFlags|hipInitDevice)\s*\(")


def flag_calls(text):
    """Line numbers of hipSetDeviceFlags or hipInitDevice calls in C++ or HIP code (no comments, no strings)."""
    _, bare = blank_lexemes(text)
    return [bare.count("\n", 0, m.start()) + 1 for m in CALL.finditer(bare)]


class ScheduleFlagsTest(unittest.TestCase):
    def test_no_scheduling_flag_change(self):
        sites = {rel: lines for rel, path in source_files(ROOT, SCANNED)
                 if (lines := flag_calls(path.read_text(errors="replace")))}
        self.assertEqual(sites, {}, "the HIP scheduling mode must not change at run time (see this file's doc)")

    def test_scanner_sees_calls_and_skips_comments(self):
        self.assertEqual(flag_calls("int a;\n  HIP_OK(hipSetDeviceFlags (x));\n"), [2])
        self.assertEqual(flag_calls("// hipSetDeviceFlags(x)\n/* hipSetDeviceFlags(y) */ \"hipSetDeviceFlags(z)\"\n"), [])


if __name__ == "__main__":
    unittest.main()
