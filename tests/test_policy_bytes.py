"""W1: every cache byte count uses kConversationPolicyBytes, the train 23 size of ConversationPolicy.

The tenant key made the struct 32 bytes larger. A count that still used sizeof would change the cache budget, the
eviction order and the done event's cache_bytes against train 23. serve.h and batched_serve.h compile only with hipcc,
so no CPU test runs those counts; this source check keeps them on the constant.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
DEFINITION = "engine/runtime/conversation_resume.h"
POLICY_SIZEOF = re.compile(r"sizeof\s*\(\s*(?:const\s+)?(?:crucible::)?(?:ConversationPolicy|policy)\s*\)")


class PolicyBytesTest(unittest.TestCase):
    def test_engine_counts_policy_bytes_with_the_constant(self):
        found = []
        for path in sorted((ROOT / "engine").rglob("*")):
            name = path.relative_to(ROOT).as_posix()
            if path.suffix not in {".h", ".hip", ".inc", ".cpp"} or name == DEFINITION:
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if POLICY_SIZEOF.search(line):
                    found.append(f"{name}:{number}")
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
