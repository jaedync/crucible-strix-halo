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

    def test_the_private_architecture_page_and_the_server_guide_name_the_exception(self):
        # Release prep v0.38.0: the private ARCHITECTURE.md is the source of the public template, and the server guide
        # ships as it is. Neither states the rule without its session replay exception.
        arch = " ".join((ROOT / "docs/ARCHITECTURE.md").read_text().split())
        self.assertNotIn("Batching, cache state, load and speculation do not change it", arch)
        self.assertNotIn("is bitwise the state that a fresh prefill makes", arch)
        self.assertNotIn("A request's output depends only on the request. The server passes", arch)
        server = " ".join((ROOT / "server/README.md").read_text().split())
        self.assertNotIn("A request's output depends only on the request. The serial backend", server)
        self.assertIn("depends only on the request, except with session replay", server)


if __name__ == "__main__":
    unittest.main()
