"""Strict tool grammars take only the arguments that the schema allows (review H2).

The server builds the call grammar (server/structured.py call_format); the engine grammar runs in
build/tests/constraint_text_check (xgrammar, a character vocabulary). Each root keyword of a strict tool's parameters is
either encoded (a call that the schema allows is taken, one that it forbids is refused) or refused with a 400 that
names it. The reference is the server's own validator (server/tool_schema.matches). The engine part skips when the
checker is not built (make build/tests/constraint_text_check)."""
from pathlib import Path
import subprocess
import unittest
from server.api import ApiError, validate_request
from server.tool_schema import matches

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "build" / "tests" / "constraint_text_check"
BASE = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"], "additionalProperties": False}
# Root keywords that the call grammar does not encode: a 400 that names the keyword.
REFUSED_ROOT = {
    "const": {"x": 1},
    "enum": [{"x": 1}],
    "anyOf": [{"properties": {"x": {"const": 1}}}],
    "oneOf": [{"properties": {"x": {"const": 1}}}],
    "allOf": [{"properties": {"x": {"const": 1}}}],
    "not": {"properties": {"x": {"const": 2}}},
    "if": {"properties": {"x": {"const": 2}}},
    "then": {"properties": {"x": {"const": 1}}},
    "else": {"properties": {"x": {"const": 1}}},
    "dependentRequired": {"x": ["y"]},
    "dependentSchemas": {"x": {"required": ["y"]}},
    "dependencies": {"x": ["y"]},
    "patternProperties": {"^x$": {"const": 1}},
    "propertyNames": {"maxLength": 0},
    "unevaluatedProperties": False,
    "minProperties": 2,
    "maxProperties": 0,
}


def call(**arguments):
    params = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items())
    return f"<tool_call>\n<function=f>\n{params}</function>\n</tool_call>"


def request(parameters):
    tool = dict(type="function", function=dict(name="f", strict=True, parameters=parameters))
    return validate_request(dict(messages=[dict(role="user", content="Hi")], max_tokens=64, tools=[tool],
                                 tool_choice="required"), "chat", "crucible")


def engine(constraint, texts):
    lines = "".join(f"{constraint['kind']}\t{constraint['spec']}\t{text.encode().hex()}\n" for text in texts)
    result = subprocess.run([str(CHECK)], input=lines, text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stderr[-500:]
    return result.stdout.splitlines()


class RefusalTests(unittest.TestCase):
    def refused(self, parameters, code, word):
        with self.assertRaises(ApiError) as caught:
            request(parameters)
        self.assertEqual((caught.exception.status, caught.exception.code), (400, code))
        self.assertIn(word, str(caught.exception))

    def test_the_review_repros_are_refused(self):
        # 1: x is forbidden (false) and required: no call matches.
        self.refused({"type": "object", "properties": {"x": False}, "required": ["x"], "additionalProperties": False},
                     "constraint_impossible", "false")
        # 2 and 3: root const and root anyOf are not encoded.
        self.refused(BASE | {"const": {"x": 1}}, "constraint_keyword", "const")
        self.refused(BASE | {"anyOf": [{"properties": {"x": {"const": 1}}}]}, "constraint_keyword", "anyOf")

    def test_every_root_keyword_that_is_not_encoded_is_refused(self):
        for keyword, value in REFUSED_ROOT.items():
            with self.subTest(keyword=keyword):
                self.refused(BASE | {keyword: value}, "constraint_keyword", keyword)

    def test_a_root_keyword_beside_a_root_ref_is_refused(self):
        self.refused({"$ref": "#/$defs/P", "$defs": {"P": BASE}, "const": {"x": 1}}, "constraint_keyword", "const")

    def test_a_root_that_is_not_an_object_schema_is_refused(self):
        self.refused({"$ref": "#/$defs/N", "$defs": {"N": False}}, "constraint_impossible", "false")
        self.refused({"$ref": "#/$defs/S", "$defs": {"S": {"type": "string"}}}, "constraint_impossible", "object")

    def test_a_false_parameter_schema_behind_a_ref_is_refused_when_required(self):
        self.refused({"type": "object", "properties": {"x": {"$ref": "#/$defs/N"}}, "required": ["x"],
                      "$defs": {"N": False}}, "constraint_impossible", "false")

    def test_a_raw_string_parameter_with_any_of_is_refused(self):
        x = {"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "string", "enum": ["b"]}]}
        self.refused({"type": "object", "properties": {"x": x}, "required": ["x"]}, "constraint_keyword", "anyOf")

    def test_a_ref_outside_the_definitions_is_refused(self):
        # In the value's own grammar "#/properties/w" would name x's w (a string), not the root's w (an integer).
        x = {"type": "object", "properties": {"w": {"type": "string"}, "z": {"$ref": "#/properties/w"}}}
        self.refused({"type": "object", "properties": {"w": {"type": "integer"}, "x": x}}, "constraint_keyword", "$ref")


@unittest.skipUnless(CHECK.exists(), "needs build/tests/constraint_text_check")
class EncodedTests(unittest.TestCase):
    def pairs(self, parameters, allowed, forbidden):
        """Each text of `allowed` is taken and each of `forbidden` is refused; the validator agrees on both."""
        texts = [text for text, _ in allowed] + [text for text, _ in forbidden]
        for (_, arguments), want in zip(allowed + forbidden, [True] * len(allowed) + [False] * len(forbidden)):
            if arguments is not None:
                self.assertEqual(matches(arguments, parameters, parameters), want, arguments)
        results = engine(request(parameters).constraint, texts)
        self.assertEqual(results[:len(allowed)], ["accept"] * len(allowed))
        for text, result in zip(texts[len(allowed):], results[len(allowed):]):
            self.assertIn(result, ("reject", "prefix"), text)

    def test_the_base_schema(self):
        self.pairs(BASE, [(call(x=2), {"x": 2})],
                   [(call(x="a"), {"x": "a"}), (call(), {}), (call(x=2, y=3), {"x": 2, "y": 3})])

    def test_a_false_parameter_that_is_not_required_is_never_emitted(self):
        parameters = {"type": "object", "properties": {"x": False, "y": {"type": "integer"}}}
        self.pairs(parameters, [(call(), {}), (call(y=1), {"y": 1})], [(call(x=2), {"x": 2}), (call(x=2, y=1), None)])

    def test_encoded_root_keywords(self):
        for extra in ({}, {"title": "t"}, {"description": "d"}, {"default": {"x": 1}}, {"examples": [{"x": 1}]},
                      {"$comment": "c"}, {"additionalProperties": True}, {"additionalProperties": {"type": "string"}},
                      {"$defs": {"unused": {"type": "string"}}}):
            with self.subTest(extra=extra):
                self.pairs(BASE | extra, [(call(x=2), {"x": 2})], [(call(x="a"), {"x": "a"}), (call(), {})])
        # Not required: the call may leave x out.
        optional = {"type": "object", "properties": {"x": {"type": "integer"}}}
        self.pairs(optional, [(call(), {}), (call(x=2), {"x": 2})], [(call(x="a"), {"x": "a"})])

    def test_a_root_ref(self):
        parameters = {"$ref": "#/$defs/P", "$defs": {"P": BASE}}
        self.pairs(parameters, [(call(x=2), {"x": 2})], [(call(x="a"), {"x": "a"}), (call(), {})])

    def test_parameter_keywords_go_to_the_value_grammar(self):
        x = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "const": {"n": 1}}
        self.pairs({"type": "object", "properties": {"x": x}, "required": ["x"]},
                   [(call(x='{"n":1}'), {"x": {"n": 1}})], [(call(x='{"n":2}'), {"x": {"n": 2}})])
        raw = {"type": "object", "properties": {"x": {"type": "string", "enum": ["a", "b"]}}, "required": ["x"]}
        self.pairs(raw, [(call(x="a"), {"x": "a"})], [(call(x="c"), {"x": "c"})])

    def test_a_nested_false_schema_fails_closed_in_the_engine(self):
        x = {"type": "object", "properties": {"y": False}}
        constraint = request({"type": "object", "properties": {"x": x}, "required": ["x"]}).constraint
        self.assertEqual(engine(constraint, [call(x='{"y":1}')]), ["refused constraint_compile"])


@unittest.skipUnless(CHECK.exists(), "needs build/tests/constraint_text_check")
class ResponseFormatTests(unittest.TestCase):
    """The json_schema answer goes through xgrammar's own schema converter: the same root keywords are enforced, and
    a false property schema does not compile (a 400 from the engine, no unchecked output)."""

    def answer(self, schema, texts):
        body = dict(messages=[dict(role="user", content="Hi")], max_tokens=64,
                    response_format=dict(type="json_schema", json_schema=dict(name="a", schema=schema)))
        return engine(validate_request(body, "chat", "crucible").constraint, texts)

    def test_root_keywords_are_enforced(self):
        for keyword in ("const", "enum"):
            with self.subTest(keyword=keyword):
                schema = BASE | {keyword: REFUSED_ROOT[keyword]}
                self.assertFalse(matches({"x": 2}, schema, schema))
                self.assertEqual(self.answer(schema, ['{"x":1}', '{"x":2}']), ["accept", "reject"])

    def test_a_root_any_of_beside_other_keywords_is_refused(self):
        # The converter reads anyOf only: it dropped required, and the grammar took {} (review H2 round 2).
        schema = BASE | {"anyOf": REFUSED_ROOT["anyOf"]}
        self.assertFalse(matches({}, schema, schema))
        with self.assertRaises(ApiError) as caught:
            self.answer(schema, ["{}"])
        self.assertEqual(caught.exception.code, "constraint_keyword")
        self.assertIn("beside anyOf", str(caught.exception))

    def test_a_false_property_does_not_compile(self):
        schema = {"type": "object", "properties": {"x": False}, "required": ["x"], "additionalProperties": False}
        # The server refuses it first now (schema_exact: the required x has no value), with a 400 at the same open.
        with self.assertRaises(ApiError) as caught:
            self.answer(schema, ['{"x":2}'])
        self.assertEqual(caught.exception.code, "constraint_impossible")
        optional = {"type": "object", "properties": {"x": False}, "additionalProperties": False}
        self.assertEqual(self.answer(optional, ['{"x":2}']), ["refused constraint_compile"])


if __name__ == "__main__":
    unittest.main()
