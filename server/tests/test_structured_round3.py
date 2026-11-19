"""The three findings of review round 3, on the response_format path and the strict-tool path.

1. An integer literal (const or enum, at any depth) outside the 64-bit range: the converter writes it as a float64
   and the grammar takes another literal. It is a 400 constraint_keyword; a 64-bit integer keeps its literal.
2. A strict call whose argument holds `</tool_call>` (or a JSON value that holds `</parameter>`): the grammar takes
   it, so the parser reads the call by its structure and returns the call.
3. constraint_impossible only for a schema with no valid document. A keyword of another type is ignored, one
   productive alternative of a union is enough, a node with no value that a valid document avoids is a 400
   constraint_empty_branch, and a number range with real values but no grid decimal is a 400 constraint_range.
The engine parts skip when build/tests/constraint_text_check is not built."""
import json
import types as pytypes
import unittest
from server.api import ApiError
from server.tool_output import ToolOutput
from server.tool_schema import matches
from server.tests.test_structured_exact import answer_request, tool_request, wrap
from server.tests.test_structured_strict import CHECK, call, engine

BIG = 18446744073709551617
DEAD = {"type": "array", "minItems": 1, "items": {"type": "array", "minItems": 1, "items": False}}


class Refused(unittest.TestCase):
    def refused(self, build, schema, code, *words):
        with self.assertRaises(ApiError) as caught:
            build(schema)
        self.assertEqual((caught.exception.status, caught.exception.code), (400, code), str(caught.exception))
        for word in words:
            self.assertIn(word, str(caught.exception))


class IntegerLiteralTests(Refused):
    def test_the_review_repro_is_refused_on_both_paths(self):
        self.refused(answer_request, {"const": BIG}, "constraint_keyword", "const", "64-bit", "at #.")
        self.refused(tool_request, wrap({"const": BIG}), "constraint_keyword", "const", "#/properties/x")

    def test_every_depth_and_both_ends_are_refused(self):
        for x, keyword in (({"enum": [1, 2 ** 63]}, "enum"), ({"const": -(2 ** 63) - 1}, "const"),
                           ({"const": {"a": [1, {"b": 2 ** 64}]}}, "const"), ({"enum": [[1], [[BIG]]]}, "enum"),
                           ({"type": "array", "items": {"enum": [BIG]}}, "enum")):
            with self.subTest(x=x):
                where = "#/properties/x/items" if "items" in x else "#/properties/x"
                self.refused(answer_request, wrap(x), "constraint_keyword", keyword, where)
                self.refused(tool_request, wrap(x), "constraint_keyword", keyword, where)

    def test_64_bit_integers_and_floats_are_served(self):
        for x in ({"const": 2 ** 63 - 1}, {"const": -(2 ** 63)}, {"enum": [2 ** 53 + 1, 1.5, 1e300]},
                  {"const": {"a": [9007199254740993]}}):
            with self.subTest(x=x):
                answer_request(wrap(x))
                tool_request(wrap(x))

    @unittest.skipUnless(CHECK.exists(), "needs build/tests/constraint_text_check")
    def test_the_engine_keeps_a_64_bit_literal(self):
        # The converter keeps a 64-bit integer as an integer, also past 2^53: the exact literal only.
        constraint = answer_request({"const": 2 ** 53 + 1}).constraint
        self.assertEqual(engine(constraint, ["9007199254740993", "9007199254740992", "9.007199254740992e+15"]),
                         ["accept", "reject", "reject"])
        results = engine(tool_request(wrap({"const": 2 ** 63 - 1})).constraint,
                         [call(x=str(2 ** 63 - 1)), call(x="9.223372036854776e+18")])
        self.assertEqual(results[0], "accept")
        self.assertIn(results[1], ("reject", "prefix"))


def parsed(parameters, text, stream=False, parallel=True, strict=True):
    """The message of the parser for text, fed in pieces of 3 characters (as a stream would give it)."""
    tool = dict(type="function", function=dict(name="f", strict=strict, parameters=parameters))
    request = pytypes.SimpleNamespace(tools=(tool,), tool_choice="required", parallel_tool_calls=parallel)
    output = ToolOutput(request, stream=stream)
    deltas = []
    for start in range(0, len(text), 3):
        output.feed(text[start:start + 3])
        deltas += output.take_deltas()
    message, tail, reason = output.finish("stop")
    return output, message, deltas + tail, reason


def arguments(message):
    return [json.loads(item["function"]["arguments"]) for item in message.get("tool_calls", [])]


STRING_OR = ({"type": ["string", "null"]}, {}, {"anyOf": [{"type": "string"}, {"type": "integer"}]})
JSON_X = wrap({"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"],
               "additionalProperties": False})


class MarkerTests(unittest.TestCase):
    """A strict argument that holds the outer end tag: the call parses, in stream mode and not."""

    CASES = [
        (wrap({"type": "string", "const": "</tool_call>"}), dict(x="</tool_call>"), {"x": "</tool_call>"}),
        (wrap({"type": "string"}), dict(x="a</tool_call>b"), {"x": "a</tool_call>b"}),
        (wrap({"type": "string", "enum": ["</tool_call>", "b"]}), dict(x="</tool_call>"), {"x": "</tool_call>"}),
        (JSON_X, dict(x='{"s":"</tool_call>"}'), {"x": {"s": "</tool_call>"}}),
        (JSON_X, dict(x='{"s":"</parameter>\\n</function>"}'), {"x": {"s": "</parameter>\n</function>"}}),
        (wrap({"type": "array", "items": {"type": "string"}}), dict(x='["</tool_call>", "</parameter>"]'),
         {"x": ["</tool_call>", "</parameter>"]}),
    ] + [
        # Review round 4, finding 2: a JSON string of a type set with string, or of no type, holds the tag.
        (wrap(x), dict(x='"a</parameter>b"'), {"x": "a</parameter>b"}) for x in STRING_OR]

    def test_the_parser_returns_the_call(self):
        for parameters, args, expected in self.CASES:
            for stream in (False, True):
                with self.subTest(args=args, stream=stream):
                    output, message, deltas, reason = parsed(parameters, call(**args), stream)
                    self.assertFalse(output.fallback, output.invalid)
                    self.assertEqual((arguments(message), reason), ([expected], "tool_calls"))
                    if stream:
                        sent = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in deltas)
                        self.assertEqual(sent, message["tool_calls"][0]["function"]["arguments"])
                        # The stream reads a strict call as the parser does: it does not stop at the tag.
                        self.assertFalse(output.stream.stopped)

    def test_a_tool_that_is_not_strict_keeps_the_first_parameter_end(self):
        # Its grammar excludes the tag from the arguments: the value ends at the tag and the call is malformed.
        for x in STRING_OR:
            for stream in (False, True):
                with self.subTest(x=x, stream=stream):
                    text = call(x='"a</parameter>b"')
                    output, message, _, reason = parsed(wrap(x), text, stream, strict=False)
                    self.assertEqual((output.fallback, output.invalid), (True, "parameter_tag"))
                    self.assertEqual((message["content"], reason), (text, "length" if stream else "stop"))

    def test_a_strict_string_off_the_grammar_form_keeps_the_first_parameter_end(self):
        # Raw text (not a JSON string) for a type set with string: the value ends at the first tag, in both modes.
        for stream in (False, True):
            with self.subTest(stream=stream):
                output, message, deltas, _ = parsed(wrap({"type": ["string", "null"]}), call(x="a</parameter>b"),
                                                    stream)
                self.assertEqual((output.fallback, output.invalid), (True, "parameter_tag"))

    def test_parallel_calls_with_the_marker(self):
        parameters, args, expected = self.CASES[3]
        text = call(**args) + "\n" + call(x='{"s":"b</tool_call>"}')
        output, message, _, _ = parsed(parameters, text)
        self.assertFalse(output.fallback, output.invalid)
        self.assertEqual(arguments(message), [expected, {"x": {"s": "b</tool_call>"}}])

    def test_an_open_call_stays_open_past_an_argument_marker(self):
        tool = dict(type="function", function=dict(name="f", strict=True, parameters=JSON_X))
        request = pytypes.SimpleNamespace(tools=(tool,), tool_choice="required", parallel_tool_calls=True)
        output = ToolOutput(request)
        output.feed('<tool_call>\n<function=f>\n<parameter=x>\n{"s":"</tool_call>"}')
        self.assertTrue(output.open_call())
        output.feed("\n</parameter>\n</function>\n</tool_call>")
        self.assertFalse(output.open_call())

    def test_a_cut_call_is_still_incomplete(self):
        text = '<tool_call>\n<function=f>\n<parameter=x>\n{"s":"</tool_call>'
        output, message, _, _ = parsed(JSON_X, text)
        self.assertTrue(output.fallback)
        self.assertEqual(output.invalid, "incomplete_call")
        self.assertEqual(message["content"], text)

    def test_peers_keep_their_parse(self):
        # An ordinary strict call is unchanged. A tool that is not strict keeps the first end tag (its grammar
        # excludes the tag from the arguments), so its call still ends there.
        _, message, _, _ = parsed(JSON_X, call(x='{"s":"a"}'))
        self.assertEqual(arguments(message), [{"x": {"s": "a"}}])
        output, _, _, _ = parsed(JSON_X, call(x='{"s":"</tool_call>"}'), strict=False)
        self.assertTrue(output.fallback)

    def test_text_off_the_grammar_form_keeps_the_first_end_tag(self):
        from server.schema_limits import WorkBudget
        from server.tool_frame import call_end
        definitions = {"f": dict(name="f", strict=True, parameters=wrap({"type": "integer"}))}
        for value in ("<x", "1 2", '1"'):
            text = f"<tool_call>\n<function=f>\n<parameter=x>\n{value}</tool_call>tail</tool_call>"
            with self.subTest(value=value):
                self.assertEqual(call_end(text, 0, definitions, WorkBudget()), text.find("</tool_call>"))
        text = call(x="12") + "\n"
        self.assertEqual(call_end(text, 0, definitions, WorkBudget()), text.find("</tool_call>"))

    @unittest.skipUnless(CHECK.exists(), "needs build/tests/constraint_text_check")
    def test_the_engine_takes_each_call(self):
        for parameters, args, _ in self.CASES:
            with self.subTest(args=args):
                self.assertEqual(engine(tool_request(parameters).constraint, [call(**args)]), ["accept"])


class ImpossibleTests(Refused):
    def test_a_keyword_of_another_type_is_ignored(self):
        for x in ({"type": "integer", "minLength": 2, "maxLength": 1}, {"type": "string", "minItems": 2, "maxItems": 1},
                  {"type": "integer", "minItems": 3, "maxItems": 0}):
            with self.subTest(x=x):
                answer_request(wrap(x))
                tool_request(wrap(x))
        self.assertTrue(matches({"x": 1}, wrap({"type": "integer", "minLength": 2, "maxLength": 1}), {}))

    def test_one_productive_alternative_of_a_union_is_enough(self):
        # The review repro: the object type has no value, the array [] matches. The schema is not impossible; the
        # object type is an empty branch (its grammar would take {"a":1}, or have no end).
        for a in ({"const": 1, "minimum": 2}, DEAD, {"type": "number", "minimum": 2, "maximum": 1}):
            union = {"type": ["object", "array"], "properties": {"a": a}, "required": ["a"],
                     "additionalProperties": False, "items": {"type": "integer"}}
            with self.subTest(a=a):
                self.assertTrue(matches([], union, union))
                self.refused(answer_request, union, "constraint_empty_branch", "#/properties/a")
                self.refused(tool_request, wrap(union), "constraint_empty_branch", "#/properties/x/properties/a")
        # An array type with no value beside an object type with one: the same.
        union = {"type": ["object", "array"], "minItems": 2, "maxItems": 1, "items": {}, "properties": {}}
        self.refused(answer_request, union, "constraint_empty_branch", "minItems above maxItems", "at #.")

    def test_an_unused_definition_with_no_value_is_ignored(self):
        answer_request(wrap({"type": "integer"}) | {"$defs": {"D": DEAD}})
        tool_request(wrap({"type": "integer"}) | {"$defs": {"D": DEAD}})

    def test_an_empty_branch_of_a_valid_schema_has_its_own_refusal(self):
        optional = {"type": "object", "properties": {"x": {"const": 1, "minimum": 2}}, "additionalProperties": False}
        union = {"type": ["object", "array"], "properties": {"a": DEAD}, "required": ["a"],
                 "additionalProperties": False, "items": {"type": "integer"}}
        for schema, where in ((optional, "#/properties/x"), (union, "#/properties/a"),
                              (wrap({"anyOf": [{"type": "integer"}, DEAD]}), "#/properties/x/anyOf/1"),
                              (wrap({"type": ["string", "integer"], "minLength": 2, "maxLength": 1}),
                               "#/properties/x")):
            with self.subTest(schema=schema):
                self.assertTrue(any(matches(value, schema, schema) for value in ({}, [], {"x": 1})))
                self.refused(answer_request, schema, "constraint_empty_branch", where, "Remove")
                # A strict call is an object: the union goes one level down there.
                self.refused(tool_request, wrap(schema) if schema is union else schema, "constraint_empty_branch")

    def test_a_number_range_off_the_grid_has_its_own_refusal(self):
        for x in ({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1e-9},
                  {"type": "number", "minimum": 1e-7, "maximum": 1e-7},
                  {"type": ["number", "string"], "exclusiveMinimum": 0.1, "exclusiveMaximum": 0.1000001}):
            with self.subTest(x=x):
                self.refused(answer_request, wrap(x), "constraint_range", "#/properties/x", "6 digits")
                self.refused(tool_request, wrap(x), "constraint_range", "#/properties/x")

    def test_schemas_with_no_valid_document_stay_impossible(self):
        for x in ({"type": "number", "minimum": 5, "maximum": 3}, {"type": "number", "exclusiveMinimum": 1, "maximum": 1},
                  {"type": "string", "minLength": 2, "maxLength": 1}, DEAD, {"const": 1, "minimum": 2},
                  {"type": ["string", "array"], "minLength": 2, "maxLength": 1, "minItems": 2, "maxItems": 1,
                   "items": {}}):
            with self.subTest(x=x):
                self.refused(answer_request, wrap(x), "constraint_impossible", "#/properties/x")
                self.refused(tool_request, wrap(x), "constraint_impossible", "#/properties/x")

    def test_tool_refusals_that_have_valid_documents_are_not_impossible(self):
        # A raw string enum value with the parameter end tag, and a required name outside properties (the call
        # grammar writes only properties): each schema has valid calls, so each is a keyword refusal.
        self.refused(tool_request, wrap({"type": "string", "enum": ["a", "b</parameter>"]}), "constraint_keyword")
        self.refused(tool_request, {"type": "object", "properties": {}, "required": ["q"]}, "constraint_keyword")

    @unittest.skipUnless(CHECK.exists(), "needs build/tests/constraint_text_check")
    def test_the_engine_takes_the_served_schemas(self):
        schema = {"type": ["object", "array"], "properties": {"a": {"type": "integer"}}, "required": ["a"],
                  "additionalProperties": False, "items": {"type": "integer"}}
        texts = ["[]", "[1]", '{"a":1}', "{}", '{"a":"b"}']
        results = engine(answer_request(schema).constraint, texts)
        self.assertEqual(results[:3], ["accept"] * 3)
        self.assertTrue(all(result in ("reject", "prefix") for result in results[3:]), results)
        x = wrap({"type": "integer", "minLength": 2, "maxLength": 1})
        results = engine(tool_request(x).constraint, [call(x="1"), call(x='"ab"')])
        self.assertEqual(results[0], "accept")
        self.assertIn(results[1], ("reject", "prefix"))


if __name__ == "__main__":
    unittest.main()
