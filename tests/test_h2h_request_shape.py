"""The head-to-head tool sends the request shape that Pi sends: tools, and no tool_choice."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("h2h_decode", ROOT / "tools/h2h_decode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class H2hRequestShapeTest(unittest.TestCase):
    def test_no_workload_sends_tool_choice(self):
        h2h = load()
        session = h2h.Session({"messages": [{"role": "system", "content": "rules"}]}, {"path": "a.py", "function": "f"}, "model", {})
        for workload in h2h.WORKLOADS:
            with self.subTest(workload=workload):
                body = session.body(workload)
                self.assertNotIn("tool_choice", body)
                self.assertEqual(body["tools"], h2h.TOOLS)

    def test_strict_tools_flag(self):
        """--strict-tools (struct-core speed step): the same tools with strict true; the default stays unchanged."""
        h2h = load()
        base = {"messages": [{"role": "system", "content": "rules"}]}
        target = {"path": "a.py", "function": "f"}
        plain = h2h.Session(base, target, "model", {}).body("tool")
        strict = h2h.Session(base, target, "model", {}, strict_tools=True).body("tool")
        self.assertEqual(plain["tools"], h2h.TOOLS)
        self.assertTrue(all(t["function"]["strict"] is True for t in strict["tools"]))
        self.assertEqual([dict(t, function={k: v for k, v in t["function"].items() if k != "strict"})
                          for t in strict["tools"]], h2h.TOOLS)
        self.assertFalse(any("strict" in t["function"] for t in h2h.TOOLS))
        args = h2h.parser().parse_args(["run", "--prompts", "p", "--url", "u", "--engine", "e", "--out", "o",
                                        "--strict-tools"])
        self.assertTrue(args.strict_tools)


if __name__ == "__main__":
    unittest.main()
