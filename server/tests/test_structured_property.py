"""Property test of structured output: random schemas and random tool sets, built by
server/structured.py, decoded with fake logits through the engine's grammar (build/tests/constraint_property_test, the
vendored xgrammar and the real tokenizer). Every output that ends must parse with the strict JSON parser and match its
schema (server/tool_schema.matches), or parse as calls with server/tool_output.py and no fallback; every cut output
must be a legal prefix (the driver only takes legal tokens). It skips when the driver or the tokenizer is missing.

CRUCIBLE_PROPERTY_SCALE: 1 (default) runs 500 schemas and 200 tool sets over the three logit modes."""
import json
import os
import random
import subprocess
import types as pytypes
import unittest
from pathlib import Path
from server.api import ApiError
from server.tool_schema import matches
from server import structured

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "build" / "tests" / "constraint_property_test"
TOKENIZER = Path(os.environ.get("CRUCIBLE_TOKENIZER_JSON", str(Path.home() / ".cache/mtp-slot/ckpt305/tokenizer.json")))
SCALE = float(os.environ.get("CRUCIBLE_PROPERTY_SCALE", "1"))


def random_schema(rng, depth=0):
    kinds = ["string", "integer", "number", "boolean", "enum", "null", "union", "bounded"] + \
        (["object", "array", "ref"] if depth < 3 else [])
    kind = rng.choice(kinds)
    if kind == "enum":
        return {"type": "string", "enum": [f"v{rng.randrange(100)}_é{'中' * rng.randint(0, 2)}" for _ in range(rng.randint(1, 4))]}
    if kind == "object":
        props = {f"k{i}_{rng.randrange(1000)}": random_schema(rng, depth + 1) for i in range(rng.randint(1, 4))}
        return {"type": "object", "properties": props, "required": [p for p in props if rng.random() < 0.7],
                "additionalProperties": False}
    if kind == "array":
        return {"type": "array", "items": random_schema(rng, depth + 1), "minItems": rng.choice([0, 1]),
                "maxItems": rng.choice([2, 3])}
    if kind == "string":
        return {"type": "string", "maxLength": rng.choice([5, 20, 40]), "minLength": rng.choice([0, 1])}
    if kind == "union":
        return {"type": ["string", "null"]}
    if kind == "bounded":
        return {"type": "integer", "minimum": rng.randint(-5, 5), "maximum": rng.randint(6, 99)}
    if kind == "ref":
        return {"$ref": "#/$defs/item"}
    return {"type": kind}


def random_response_schema(rng):
    schema = {"type": "object", "properties": {"a": random_schema(rng), "b": random_schema(rng)},
              "required": ["a"], "additionalProperties": False,
              # The server refuses cyclic references (tool_schema.check_schema), so the chain is acyclic.
              "$defs": {"item": {"type": "object", "properties": {"n": {"type": "integer"}, "next": {
                  "anyOf": [{"$ref": "#/$defs/leaf"}, {"type": "null"}]}}, "required": ["n"],
                  "additionalProperties": False},
                  "leaf": {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"],
                           "additionalProperties": False}}}
    return schema


def random_parameter(rng):
    kind = rng.choice(["string", "string", "integer", "boolean", "enum", "array", "number", "union", "object"])
    if kind == "enum":
        return {"type": "string", "enum": [f"m{rng.randrange(50)}" for _ in range(rng.randint(1, 3))]}
    if kind == "array":
        return {"type": "array", "items": {"type": "integer"}, "maxItems": 3}
    if kind == "union":
        return {"type": ["string", "null"]}
    if kind == "object":
        return {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}
    return {"type": kind}


def random_tools(rng):
    tools = []
    for t in range(rng.randint(1, 4)):
        props = {f"p{j}": random_parameter(rng) for j in range(rng.randint(0, 3))}
        tools.append(dict(type="function", function=dict(
            name=f"tool_{t}_{rng.randrange(100)}", strict=True,
            parameters=dict(type="object", properties=props, required=[p for p in props if rng.random() < 0.6],
                            additionalProperties=False))))
    return tools


@unittest.skipUnless(DRIVER.exists() and TOKENIZER.exists(), "needs build/tests/constraint_property_test and a tokenizer")
class StructuredPropertyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))
        pieces = "\n</parameter>\n</function>\n</tool_call>}\""
        cls.end_ids = sorted(set(cls.tokenizer.encode(pieces, add_special_tokens=False).ids))

    def run_driver(self, lines):
        text = "".join(json.dumps(line) + "\n" for line in lines)
        result = subprocess.run([str(DRIVER), str(TOKENIZER), *map(str, self.end_ids)], input=text, text=True,
                                capture_output=True, timeout=1800)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        outputs = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(outputs), len(lines))
        for output in outputs:
            self.assertNotIn("error", output)
        return outputs

    def answer(self, ids, reasoning):
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        if reasoning:
            text = text.split("</think>", 1)[1]
        return text.removesuffix("<|im_end|>")

    def test_json_schemas(self):
        rng = random.Random(7)
        count = int(500 * SCALE)
        cases, lines = [], []
        for trial in range(count):
            schema = random_response_schema(rng)
            request = structured.constraint(dict(response_format=dict(type="json_schema", json_schema=dict(
                name="r", schema=schema))), "chat", (), "auto", True)
            reasoning = trial % 5 == 0
            cases.append((schema, reasoning))
            lines.append(dict(kind=request["kind"], spec=request["spec"], reasoning=int(reasoning), seed=trial,
                              mode=trial % 3, max=160))
        stats = dict(ended=0, cut=0)
        for (schema, reasoning), output in zip(cases, self.run_driver(lines)):
            if not output["ended"]:
                stats["cut"] += 1
                continue
            stats["ended"] += 1
            value = json.loads(self.answer(output["ids"], reasoning))  # strict: no raw control characters
            self.assertTrue(matches(value, schema, schema), (schema, value))
        self.assertGreater(stats["ended"], count // 2, stats)
        print(f"\nstructured property: {count} schemas, {stats['ended']} ended and validated, {stats['cut']} cut")

    def replay(self, request, text):
        """(accepted ids, ids of the text, ended) of `text` + the stop token through the request's grammar."""
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids + [248046]
        output = self.run_driver([dict(kind=request["kind"], spec=request["spec"], reasoning=0, seed=0, mode=3,
                                       max=0, ids=",".join(map(str, ids)))])[0]
        return output["ids"], ids, output["ended"]

    def test_auto_calls_end_the_output(self):
        """tool_output.py takes text before the calls only: the auto grammar must refuse text after or between calls."""
        tools = (dict(type="function", function=dict(name="f", strict=True, parameters=dict(
            type="object", properties=dict(a=dict(type="integer")), required=["a"], additionalProperties=False))),)
        call = "<tool_call>\n<function=f>\n<parameter=a>\n3\n</parameter>\n</function>\n</tool_call>"
        for parallel in (True, False):
            request = structured.constraint({}, "chat", tools, "auto", parallel)
            good = ["Hi.", "Hi.\n" + call] + (["Hi.\n" + call + "\n" + call] if parallel else [])
            bad = ["Hi.\n" + call + "\nbye", call + "x" + call] + ([] if parallel else [call + "\n" + call])
            for text in good:
                taken, ids, ended = self.replay(request, text)
                self.assertTrue(ended and taken == ids, (parallel, text))
            for text in bad:
                taken, ids, ended = self.replay(request, text)
                self.assertFalse(ended, (parallel, text))

    def test_tool_sets(self):
        from server.tool_output import ToolOutput
        rng = random.Random(11)
        count = int(200 * SCALE)
        cases, lines = [], []
        for trial in range(count):
            tools = random_tools(rng)
            choice = ["required", "auto", "named"][trial % 3]
            if choice == "named":
                choice = dict(type="function", function=dict(name=tools[0]["function"]["name"]))
            parallel = trial % 2 == 0
            request = structured.constraint({}, "chat", tuple(tools), choice, parallel)
            reasoning = trial % 4 == 0
            cases.append((tools, choice, parallel, reasoning))
            lines.append(dict(kind=request["kind"], spec=request["spec"], reasoning=int(reasoning), seed=trial,
                              mode=trial % 3, max=200))
        stats = dict(ended=0, calls=0, cut=0)
        for (tools, choice, parallel, reasoning), output in zip(cases, self.run_driver(lines)):
            if not output["ended"]:
                stats["cut"] += 1
                continue
            stats["ended"] += 1
            request = pytypes.SimpleNamespace(tools=tuple(tools), tool_choice=choice, parallel_tool_calls=parallel)
            parser = ToolOutput(request)
            parser.feed(self.answer(output["ids"], reasoning))
            message, _, reason = parser.finish("stop")
            self.assertFalse(parser.fallback, (parser.invalid, choice))
            calls = message.get("tool_calls") or []
            if choice != "auto":
                self.assertGreaterEqual(len(calls), 1)
            if not parallel:
                self.assertLessEqual(len(calls), 1)
            stats["calls"] += len(calls)
        self.assertGreater(stats["ended"], count // 3, stats)
        print(f"\nstructured property: {count} tool sets, {stats['ended']} ended and parsed, {stats['calls']} calls, "
              f"{stats['cut']} cut")


if __name__ == "__main__":
    unittest.main()
