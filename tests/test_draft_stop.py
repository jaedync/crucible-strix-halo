"""Draft stop: every draft path ends its chain at a stop ID and at the token limit (engine/runtime/draft_stop.h).

batched_serve.h compiles only with hipcc and has no CPU engine, so its paths get a source check here. The helpers have
CPU tests (draft_stop_test.cpp), and the serial paths have behavior tests (generation_test.cpp,
sampling_generation_test.cpp, gen_cli_cpu_test.cpp).
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
BATCHED = (ROOT / "engine/serve/batched_serve.h").read_text()


def body(name, text=BATCHED):
    """One two-space-indented member function: from its signature line to the closing brace at that indent."""
    start = re.search(rf"^  [^ /][^\n]*\b{name}\(", text, re.M)
    assert start, name
    end = text.index("\n  }\n", start.end())
    return text[start.start():end]


class BatchedDraftStopTest(unittest.TestCase):
    def test_solo_propose_caps_greedy_drafts_by_the_token_limit(self):
        text = body("Propose")
        self.assertIn("crucible::DraftRoom(remaining)", text)
        self.assertIn("std::min({s.spec, limit, cap, room})", text)
        self.assertIn("std::min({std::max(s.spec, s.lookup), limit, cap, room})", text)

    def test_grouped_propose_stops_chains(self):
        text = body("ProposeAll")
        self.assertIn("remaining[i] <= 1", text)  # the last token is a solo step with no draft room
        self.assertIn("crucible::DraftRoom(remaining[i])", text)
        self.assertIn("std::min({s.spec, limit, cap[i], room})", text)
        self.assertIn("crucible::TrimDrafts(c.drafts, s.stops, c.copy_depth)", text)  # lookup copy
        self.assertIn("IsStopToken(s.stops, c.drafts.front())", text)  # first MTP draft
        self.assertIn("IsStopToken(s.stops, next)", text)  # each chain level

    def test_serial_greedy_and_sampled_chains_stop(self):
        greedy, sampled = body("GreedyDrafts"), body("SampledDrafts")
        for text in (greedy, sampled):
            self.assertIn("crucible::TrimDrafts(copied, s.stops, copy_depth)", text)
            self.assertIn("IsStopToken(s.stops, drafts.back())", text)
        self.assertIn("if (depth <= 0) return {};", greedy)

    def test_value_chain_stops(self):
        self.assertIn("IsStopToken(s.stops, token)", body("ValueChain"))

    def test_open_keeps_the_request_stop_ids(self):
        self.assertIn("state.stops = r.stop_ids", BATCHED)

    def test_serial_drivers_use_the_helpers(self):
        for path, calls in {
            "engine/runtime/generation.h": ("DraftRoom(", "ChainMayGrow(", "TrimDrafts("),
            "engine/runtime/sampling_generation.h": ("ChainMayGrow(", "TrimDrafts("),
            "engine/cli/gen_cli.h": ("DraftRoom(", "ChainMayGrow(", "TrimDrafts("),
        }.items():
            text = (ROOT / path).read_text()
            for call in calls:
                self.assertIn(call, text, f"{path} lost {call}")


if __name__ == "__main__":
    unittest.main()
