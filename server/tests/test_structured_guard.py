"""Grammar compiler guard: input that faults the grammar compiler is a 400 at the
server boundary, before the engine sees it. Test 1 is the 40,000-level nested pattern of the grammar probe."""
import json
import unittest
from server.api import ApiError, validate_request
from server import grammar_limits as L
from server import schema_limits as S


def chat(**fields):
    body = dict(messages=[dict(role="user", content="Hi")], max_tokens=8) | fields
    return validate_request(body, "chat", "crucible")


def answer(schema):
    return chat(response_format=dict(type="json_schema", json_schema=dict(name="x", schema=schema)))


def strict_tool(parameters):
    return [dict(type="function", function=dict(name="f", strict=True, parameters=parameters))]


NESTED_40000 = "(" * 40000 + "a" + ")" * 40000


class NestedPatternTests(unittest.TestCase):
    def assert_limit(self, call):
        with self.assertRaises(ApiError) as caught:
            call()
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.code, "constraint_limit")
        # Friendly: it names the limit, never echoes the pattern.
        self.assertIn("pattern", str(caught.exception))
        self.assertNotIn("((((", str(caught.exception))
        return caught.exception

    def test_40000_levels_in_response_format(self):
        # 80 KB, under the 128 KiB schema limit: it passed the server and crashed the engine (SIGSEGV).
        self.assert_limit(lambda: answer({"type": "string", "pattern": NESTED_40000}))

    def test_40000_levels_in_an_object_property(self):
        self.assert_limit(lambda: answer({"type": "object", "properties": {"a": {"type": "string",
                                                                                   "pattern": NESTED_40000}}}))

    def test_40000_levels_in_a_strict_tool_parameter(self):
        parameters = {"type": "object", "properties": {"o": {"type": "object", "properties": {
            "s": {"type": "string", "pattern": NESTED_40000}}}}, "required": ["o"]}
        self.assert_limit(lambda: chat(tools=strict_tool(parameters), tool_choice="required"))


def props(count, required=0, schema=None):
    names = [f"p{i}" for i in range(count)]
    return {"type": "object", "properties": {name: dict(schema or {"type": "string"}) for name in names},
            "required": names[:required]}


def nested_groups(depth):
    return "(" * depth + "a" + ")" * depth


def string(**keywords):
    return {"type": "string"} | keywords


AT_EXPANSION = "(ab){128}" * 8  # 8 x 256 = MAX_PATTERN_EXPANSION
# name: (schema at the limit, schema one step above it). Each limit of server/grammar_limits.py and the schema
# limits of server/schema_limits.py that bound what the grammar compiler sees.
CASES = {
    "MAX_PATTERN_BYTES": (string(pattern="a" * L.MAX_PATTERN_BYTES), string(pattern="a" * (L.MAX_PATTERN_BYTES + 1))),
    "MAX_PATTERN_GROUP_DEPTH": (string(pattern=nested_groups(L.MAX_PATTERN_GROUP_DEPTH)),
                                string(pattern=nested_groups(L.MAX_PATTERN_GROUP_DEPTH + 1))),
    "MAX_PATTERN_GROUPS": (string(pattern="(a)" * L.MAX_PATTERN_GROUPS),
                           string(pattern="(a)" * L.MAX_PATTERN_GROUPS + "\\(")),
    "MAX_PATTERN_REPEAT": (string(pattern="[a-z]{1,%d}" % L.MAX_PATTERN_REPEAT),
                           string(pattern="[a-z]{1,%d}" % (L.MAX_PATTERN_REPEAT + 1))),
    "MAX_PATTERN_ALTERNATIVES": (string(pattern="|".join("a" * L.MAX_PATTERN_ALTERNATIVES)),
                                 string(pattern="|".join("a" * (L.MAX_PATTERN_ALTERNATIVES + 1)))),
    # (ab){128} is 256 (a repeat bound of 128 at most): 2,048 needs eight.
    "MAX_PATTERN_EXPANSION": (string(pattern=AT_EXPANSION), string(pattern=AT_EXPANSION + "a")),
    "MAX_PATTERNS": (props(L.MAX_PATTERNS, schema=string(pattern="a+")),
                     props(L.MAX_PATTERNS + 1, schema=string(pattern="a+"))),
    # 4 patterns of 2,048 = 8,192.
    "MAX_PATTERN_TOTAL_EXPANSION": (props(4, schema=string(pattern=AT_EXPANSION)),
                                    props(4, 4, schema=string(pattern=AT_EXPANSION))
                                    | {"properties": {**{f"p{i}": string(pattern=AT_EXPANSION) for i in range(4)},
                                                      "q": string(pattern="a")}}),
    "MAX_FORMATS": (props(L.MAX_FORMATS, schema=string(format="email")),
                    props(L.MAX_FORMATS + 1, schema=string(format="email"))),
    "MAX_LENGTH_BOUND": (string(maxLength=L.MAX_LENGTH_BOUND), string(maxLength=L.MAX_LENGTH_BOUND + 1)),
    "MAX_ITEMS_BOUND": ({"type": "array", "items": {"type": "integer"}, "minItems": L.MAX_ITEMS_BOUND},
                        {"type": "array", "items": {"type": "integer"}, "minItems": L.MAX_ITEMS_BOUND + 1}),
    "MAX_OPTIONAL_PROPERTIES": (props(L.MAX_OPTIONAL_PROPERTIES + 5, required=5),
                                props(L.MAX_OPTIONAL_PROPERTIES + 5, required=4)),
    # 4 objects of 128 optional properties = 512.
    "MAX_OPTIONAL_PROPERTIES_TOTAL": (
        {"type": "object", "properties": {f"o{j}": props(L.MAX_OPTIONAL_PROPERTIES) for j in range(4)},
         "required": ["o0", "o1", "o2", "o3"]},
        {"type": "object", "properties": {f"o{j}": props(L.MAX_OPTIONAL_PROPERTIES) for j in range(4)},
         "required": ["o0", "o1", "o2"]}),
    "MAX_ANYOF_BRANCHES": ({"anyOf": [{"const": i} for i in range(L.MAX_ANYOF_BRANCHES)]},
                           {"anyOf": [{"const": i} for i in range(L.MAX_ANYOF_BRANCHES + 1)]}),
    "MAX_ENUM_VALUES": ({"enum": list(range(L.MAX_ENUM_VALUES))}, {"enum": list(range(L.MAX_ENUM_VALUES + 1))}),
}


def array_depth(levels):
    schema = {"type": "integer"}
    for _ in range(levels):
        schema = {"type": "array", "items": schema}
    return schema


class LimitTests(unittest.TestCase):
    def test_each_limit_at_and_above(self):
        for name, (at, above) in CASES.items():
            with self.subTest(name):
                self.assertIsNotNone(answer(at).constraint)
                with self.assertRaises(ApiError) as caught:
                    answer(above)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(caught.exception.code, "constraint_limit")
                self.assertIn(f"{name} = {getattr(L, name)}", str(caught.exception))

    def test_every_limit_has_a_case(self):
        names = {name for name in dir(L) if name.startswith("MAX_")}
        self.assertEqual(names, set(CASES))

    def test_limits_hold_for_strict_tools(self):
        for name, (at, above) in CASES.items():
            # Inside an object: a top-level string parameter is raw text and refuses the string keywords.
            at_tool = props(1, 1, {"type": "object", "properties": {"v": at}, "required": ["v"]})
            above_tool = props(1, 1, {"type": "object", "properties": {"v": above}, "required": ["v"]})
            with self.subTest(name):
                self.assertIsNotNone(chat(tools=strict_tool(at_tool), tool_choice="required").constraint)
                with self.assertRaises(ApiError) as caught:
                    chat(tools=strict_tool(above_tool), tool_choice="required")
                self.assertEqual(caught.exception.code, "constraint_limit")

    def test_one_budget_for_the_answer_and_the_tools(self):
        half = props(L.MAX_FORMATS // 2, schema=string(format="uri"))
        wrap = lambda schema: props(1, 1, schema)
        self.assertIsNotNone(chat(response_format=dict(type="json_schema", json_schema=dict(name="x", schema=half)),
                                  tools=strict_tool(wrap(half)), tool_choice="auto").constraint)
        more = props(L.MAX_FORMATS // 2 + 1, schema=string(format="uri"))
        with self.assertRaises(ApiError) as caught:
            chat(response_format=dict(type="json_schema", json_schema=dict(name="x", schema=half)),
                 tools=strict_tool(wrap(more)), tool_choice="auto")
        self.assertEqual(caught.exception.code, "constraint_limit")

    def test_a_tool_that_is_not_strict_has_no_grammar_and_no_grammar_limit(self):
        loose = [dict(type="function", function=dict(name="f", parameters=props(1, 1, string(pattern=NESTED_40000))))]
        self.assertIsNone(chat(tools=loose, tool_choice="auto").constraint)

    def test_a_reference_target_outside_the_schema_keywords_counts(self):
        schema = {"x-hidden": string(pattern=nested_groups(L.MAX_PATTERN_GROUP_DEPTH + 1)),
                  "$ref": "#/x-hidden"}
        with self.assertRaises(ApiError) as caught:
            answer(schema)
        self.assertEqual(caught.exception.status, 400)

    def test_a_property_named_pattern_is_a_property(self):
        self.assertIsNotNone(answer({"type": "object", "properties": {"pattern": {"type": "string"},
                                                                      "format": {"type": "string"}}}).constraint)

    def test_schema_depth_at_and_above(self):
        # MAX_SCHEMA_DEPTH counts JSON levels below the root; the leaf's value is one more, so 63 array levels is the
        # deepest schema (deep_arr_1000, which the limit refuses, takes 114 s to compile).
        self.assertIsNotNone(answer(array_depth(S.MAX_SCHEMA_DEPTH - 1)).constraint)
        with self.assertRaises(ApiError) as caught:
            answer(array_depth(S.MAX_SCHEMA_DEPTH))
        self.assertEqual(caught.exception.status, 400)

    def test_schema_bytes_at_and_above(self):
        base = len(json.dumps(string(description=""), separators=(",", ":")))
        at = string(description="d" * (S.MAX_SCHEMA_BYTES - base))
        self.assertEqual(len(json.dumps(at, separators=(",", ":"))), S.MAX_SCHEMA_BYTES)
        self.assertIsNotNone(answer(at).constraint)
        with self.assertRaises(ApiError):
            answer(string(description="d" * (S.MAX_SCHEMA_BYTES - base + 1)))

    def test_reference_cycles_are_refused(self):
        for schema in ({"$ref": "#"}, {"type": "array", "items": {"$ref": "#"}},
                       {"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"}):
            with self.subTest(schema=schema):
                with self.assertRaises(ApiError) as caught:
                    answer(schema)
                self.assertEqual(caught.exception.status, 400)

    def test_only_the_three_kinds_reach_the_engine(self):
        # No client EBNF or regex kind: the engine line holds json_schema, json_object or structural_tag only.
        kinds = {answer(string(pattern="a+")).constraint["kind"],
                 chat(response_format=dict(type="json_object")).constraint["kind"],
                 chat(tools=strict_tool(props(1, 1)), tool_choice="required").constraint["kind"]}
        self.assertEqual(kinds, {"json_schema", "json_object", "structural_tag"})
        for kind in ("ebnf", "regex", "grammar", "lark", "structural_tag"):
            with self.assertRaises(ApiError):
                chat(response_format=dict(type=kind, grammar="root ::= \"a\""))


if __name__ == "__main__":
    unittest.main()
