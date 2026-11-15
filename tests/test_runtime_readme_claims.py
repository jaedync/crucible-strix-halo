"""engine/runtime/README.md makes no claim that the docs removed (desk seq 24, Nov 13): a restored state is not always bitwise
the state of a fresh prefill (ARCHITECTURE.md exception: a state that holds decode rows), and the cache state can change
the output in that case. The public ARCHITECTURE.md lost the same two claims (docs lane 181894c)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RuntimeReadmeClaimsTest(unittest.TestCase):
    def test_no_bitwise_restore_and_no_cache_state_claim(self):
        text = " ".join((ROOT / "engine/runtime/README.md").read_text().split())
        self.assertNotIn("A restored state is bitwise", text)
        self.assertNotIn("cache state", text)
        self.assertIn("A request's output depends only on the request", text)


if __name__ == "__main__":
    unittest.main()
