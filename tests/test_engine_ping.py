"""Engine ping (service watchdog): the ready event and the command loop in batched_serve.h.

batched_serve.h compiles only with hipcc, so its loop gets a source check here. The parser has a CPU test
(batched_serve_protocol_test.cpp). The reply is the usual batch_end event with op "ping" and no other event.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
BATCHED = (ROOT / "engine/serve/batched_serve.h").read_text()
READY = BATCHED[BATCHED.index('"{\\"event\\":\\"ready\\"'):]
READY = READY[:READY.index("std::fflush")]
LOOP = BATCHED[BATCHED.index("while (reader.Next(line, oversized))"):]  # the block line reader of the serve loop (serve_line.h)
LOOP = LOOP[:LOOP.index("// End of input")]


class EnginePingTest(unittest.TestCase):
    def test_ready_event_says_the_engine_answers_ping(self):
        self.assertIn('\\"ping\\":true', READY)

    def test_ping_ready_flag_does_not_depend_on_the_state_disk(self):
        flag = READY.index('\\"ping\\":true')
        self.assertLess(flag, READY.index("disk ?"))

    def test_loop_answers_ping_before_the_close_fallback(self):
        ping = LOOP.index('op == "ping"')
        self.assertLess(ping, LOOP.index("else session.Close("))

    def test_ping_branch_touches_no_session_state(self):
        match = re.search(r'else if \(op == "ping"\)[^\n]*\n', LOOP)
        self.assertTrue(match)
        self.assertNotIn("session.", match.group(0))
        self.assertNotIn("events", match.group(0))


if __name__ == "__main__":
    unittest.main()
