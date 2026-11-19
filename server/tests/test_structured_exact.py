"""The keyword allow-list of structured output (review H2 round 2, server/schema_exact.py).

Both paths (a response_format json_schema answer and a strict tool's parameters) refuse a schema whose engine grammar
would take a document that the schema forbids: a 400 constraint_keyword that names the keyword and the JSON pointer
of the node. A schema that is served is checked through the engine grammar (build/tests/constraint_text_check)."""
import json
import unittest
from server.api import ApiError, validate_request
from server.schema_exact import STRICTER
from server.tool_schema import matches
from server.tests.test_structured_strict import CHECK, call, engine


def answer_request(schema):
    body = dict(messages=[dict(role="user", content="Hi")], max_tokens=64,
                response_format=dict(type="json_schema", json_schema=dict(name="a", schema=schema)))
    return validate_request(body, "chat", "crucible")


def tool_request(parameters):
    tool = dict(type="function", function=dict(name="f", strict=True, parameters=parameters))
    return validate_request(dict(messages=[dict(role="user", content="Hi")], max_tokens=64, tools=[tool],
                                 tool_choice="required"), "chat", "crucible")


def wrap(x):
    """A schema with one required property x: the node under test sits at #/properties/x on both paths."""
    return {"type": "object", "properties": {"x": x}, "required": ["x"], "additionalProperties": False}


# Each node keeps a constraint that the converter's branch drops: the grammar would take a value that it forbids.
LOOSE = [
    ({"type": "integer", "enum": [1, 2], "minimum": 2}, "minimum"),
    ({"type": "string", "const": "a", "enum": ["a", "b"], "maxLength": 0}, "maxLength"),
    ({"enum": [1, "a"], "type": "integer"}, "type"),
    ({"enum": ["a", "bcd"], "maxLength": 2}, "maxLength"),
    ({"enum": [[1], [1, 2]], "maxItems": 1}, "maxItems"),
    ({"enum": [{"a": 1}, {}], "required": ["a"]}, "required"),
    ({"enum": ["a"], "pattern": "^a$"}, "pattern"),
    ({"enum": ["a"], "format": "date"}, "format"),
    ({"enum": [{"a": "x"}], "properties": {"a": {"pattern": "^y$", "type": "string"}}}, "pattern"),
    ({"$ref": "#/$defs/N", "minimum": 3}, "minimum"),
    ({"$ref": "#/$defs/N", "type": "string"}, "type"),
    ({"anyOf": [{"type": "integer"}], "minimum": 3}, "minimum"),
    ({"anyOf": [{"type": "integer"}, {"type": "string"}], "type": "integer"}, "type"),
    ({"minimum": 3}, "minimum"),
    ({"minLength": 3}, "minLength"),
    ({"required": ["a"]}, "required"),
    ({"additionalProperties": False}, "additionalProperties"),
    ({"items": {"type": "integer"}}, "items"),
    ({"format": "date"}, "format"),
    ({"type": "object", "required": ["q"]}, "required"),
    ({"type": "object", "properties": {"a": {}}, "required": ["a", "q"], "additionalProperties": True}, "required"),
    ({"type": "string", "pattern": "^a+$", "maxLength": 2}, "maxLength"),
    ({"type": "string", "format": "date", "minLength": 1}, "minLength"),
    ({"type": "string", "format": "date", "pattern": "^1"}, "pattern"),
    ({"type": "string", "pattern": '^a"b$'}, "pattern"),
    ({"type": "string", "pattern": "^a\\\\b$"}, "pattern"),
    ({"type": "string", "pattern": "^a\\nb$"}, "pattern"),
    ({"type": "string", "pattern": "^a\nb$"}, "pattern"),
    ({"type": "string", "pattern": "^a\\u005cb$"}, "pattern"),
    ({"type": "string", "pattern": "^a[\\x00-\\x1f]b$"}, "pattern"),
    ({"type": "array", "prefixItems": [{"type": "string"}]}, "prefixItems"),
    ({"type": "array", "maxItems": 2}, "items"),
    ({"type": ["array", "string"], "maxLength": 2}, "items"),
    ({"type": "object", "required": []}, "additionalProperties"),
    ({"type": "object", "x-extension": 1}, "additionalProperties"),
    ({"type": []}, "type"),
    ({"type": "object", "properties": {"a": {"$id": "x"}}}, "$id"),
    ({"type": "object", "properties": {"a": {"$anchor": "x"}}}, "$anchor"),
]


class LooseTests(unittest.TestCase):
    def refused(self, build, schema, keyword, where, code="constraint_keyword"):
        with self.assertRaises(ApiError) as caught:
            build(schema)
        self.assertEqual((caught.exception.status, caught.exception.code), (400, code), str(caught.exception))
        self.assertIn(keyword, str(caught.exception))
        self.assertIn(where, str(caught.exception))

    def test_every_loose_node_is_refused_on_both_paths(self):
        for x, keyword in LOOSE:
            with self.subTest(x=x):
                defs = {"$defs": {"N": {"type": "integer"}}}
                code = "constraint_impossible" if "const" in x else "constraint_keyword"
                self.refused(answer_request, wrap(x) | defs, keyword, "#/properties/x", code)
                self.refused(tool_request, wrap(x) | defs, keyword, "#/properties/x", code)

    def test_the_review_repros(self):
        x = {"type": "integer", "enum": [1, 2], "minimum": 2}
        self.refused(tool_request, {"type": "object", "properties": {"x": x}, "required": ["x"]}, "minimum",
                     "#/properties/x")
        self.refused(answer_request, x, "minimum", "at #.")
        # const "a" is not in the enum: no value matches the node.
        raw = {"type": "string", "const": "a", "enum": ["b"]}
        self.refused(tool_request, {"type": "object", "properties": {"x": raw}, "required": ["x"]}, "enum",
                     "#/properties/x", "constraint_impossible")

    def test_a_nested_definition_that_shadows_the_root_is_refused(self):
        parameters = {
            "type": "object", "$defs": {"N": {"type": "integer"}},
            "properties": {"x": {"type": "object", "$defs": {"N": {"type": "string"}},
                                 "properties": {"z": {"$ref": "#/$defs/N"}}, "required": ["z"],
                                 "additionalProperties": False}},
            "required": ["x"]}
        self.refused(tool_request, parameters, "$defs", "#/properties/x")
        self.refused(tool_request, {"$ref": "#/$defs/P", "$defs": {"P": parameters, "N": {"type": "integer"}}},
                     "$defs", "#/$defs/P")

    def test_a_const_that_fails_its_siblings_is_impossible(self):
        self.refused(answer_request, wrap({"const": 1, "minimum": 2}), "minimum", "#/properties/x",
                     "constraint_impossible")

    def test_a_node_whose_grammar_holds_nothing_is_refused(self):
        # The empty number grid took `{"x":}` and an empty answer before; it is refused at open now. The range holds
        # real numbers, so it is constraint_range, not constraint_impossible (review round 3).
        for x in ({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1e-9},
                  {"type": "number", "exclusiveMinimum": 0.1, "exclusiveMaximum": 0.1000001},
                  {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 0.000001}):
            with self.subTest(x=x):
                self.refused(answer_request, wrap(x), "", "#/properties/x", "constraint_range")
                self.refused(tool_request, wrap(x), "", "#/properties/x", "constraint_range")
        x = {"type": "array", "minItems": 1, "items": {"type": "array", "minItems": 1, "items": False}}
        self.refused(answer_request, wrap(x), "", "#/properties/x", "constraint_impossible")
        self.refused(tool_request, wrap(x), "", "#/properties/x", "constraint_impossible")
        self.refused(answer_request, wrap({"type": "number", "minimum": 1e20}), "minimum", "#/properties/x")
        # The grid holds a value: served.
        answer_request(wrap({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 0.000002}))
        answer_request(wrap({"type": "number", "minimum": 0.1, "maximum": 0.1000001}))

    def test_references(self):
        # A pointer must name a schema node of the document (the walk checks each node that a reference can reach).
        # The converter decodes "%25" ("a%b"); tool_schema.resolve does not ("a%25b"): the two would bind apart.
        for ref, why in (("#/properties", "does not name a schema"), ("#/$defs/a%25b", "percent")):
            with self.subTest(ref=ref):
                schema = wrap({"$ref": ref}) | {"$defs": {"a%25b": {"type": "string"}, "a%b": {"type": "integer"}}}
                self.refused(answer_request, schema, "$ref", why)
        with self.assertRaises(ApiError):  # tool_schema.check_schema refuses a remote reference first
            answer_request(wrap({"$ref": "https://example.com/s"}))

    def test_a_plain_array_or_object_as_a_ref_target_is_refused(self):
        # GenerateRef skips the converter's cache: strict mode closes the plain array there (it took only []).
        for target, keyword in (({"type": "array"}, "items"), ({"type": "object"}, "additionalProperties"),
                                ({"type": ["array", "null"]}, "items")):
            with self.subTest(target=target):
                schema = wrap({"$ref": "#/$defs/A"}) | {"$defs": {"A": target}}
                self.refused(answer_request, schema, keyword, "#/$defs/A")
                self.refused(tool_request, schema, keyword, "#/$defs/A")

    def test_a_root_keyword_beside_a_tool_root_ref_is_refused(self):
        base = wrap({"type": "integer"})
        self.refused(tool_request, {"$ref": "#/$defs/P", "$defs": {"P": base}, "properties": {}}, "properties",
                     "beside $ref")


SERVED = [
    {"type": "string", "enum": ["a", "b"]},
    {"type": ["string", "null"], "enum": ["a", None]},
    {"type": "integer", "enum": [2, 3], "minimum": 2},
    {"enum": [1, "a"], "description": "d", "title": "t"},
    {"const": {"n": 1}, "type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
    {"enum": ["a", "bc"], "maxLength": 2, "anyOf": [{"type": "string"}]},
    {"$ref": "#/$defs/N", "description": "d"},
    {"anyOf": [{"type": "integer", "minimum": 3}, {"type": "null"}], "default": None},
    {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
    {"properties": {"a": {"type": "integer"}}, "required": ["a"]},
    {"type": "object"},
    {"type": "array"},
    {"type": "array", "prefixItems": [{"type": "string"}], "items": {"type": "integer"}},
    {"type": "array", "prefixItems": [{"type": "string"}], "items": False},
    {"type": ["integer", "string"], "minimum": 2, "maxLength": 1},
    {"type": "integer", "minLength": 3},
    {"type": "string", "pattern": "^[a-z]+\\.\\d{2}$"},
    {"type": "string", "format": "date"},
    {"type": "string", "format": "x-custom", "minLength": 1},
    {"type": "string", "x-extension": 1, "nullable": True},
]


class ServedTests(unittest.TestCase):
    def test_served_nodes_pass_both_paths(self):
        for x in SERVED:
            with self.subTest(x=x):
                defs = {"$defs": {"N": {"type": "integer"}}}
                answer_request(wrap(x) | defs)
                # In an object, so that a string node is JSON and not a raw string parameter of the call.
                tool_request(wrap({"type": "object", "properties": {"v": x}}) | defs)

    def test_a_tool_root_ref_and_root_id_are_served(self):
        tool_request({"$ref": "#/$defs/P", "$defs": {"P": wrap({"type": "integer"})}, "$id": "urn:x"})
        answer_request({"$id": "urn:x"} | wrap({"$ref": "#/properties/y"}) | {"properties": {
            "x": {"$ref": "#/properties/y"}, "y": {"type": "integer"}}})


@unittest.skipUnless(CHECK.exists(), "needs build/tests/constraint_text_check")
class EngineTests(unittest.TestCase):
    """Accept and reject pairs of served schemas through the engine grammar; tool_schema.matches agrees on each."""

    def answer(self, schema, allowed, forbidden):
        constraint = answer_request(schema).constraint
        texts = [json.dumps(value, separators=(",", ":")) for value in allowed + forbidden]
        for value in allowed:
            self.assertTrue(matches(value, schema, schema), value)
        for value in forbidden:
            self.assertFalse(matches(value, schema, schema), value)
        results = engine(constraint, texts)
        self.assertEqual(results[:len(allowed)], ["accept"] * len(allowed), texts)
        for text, result in zip(texts[len(allowed):], results[len(allowed):]):
            self.assertIn(result, ("reject", "prefix"), text)

    def test_enum_with_checked_siblings(self):
        self.answer({"type": "integer", "enum": [2, 3], "minimum": 2}, [2, 3], [1, 4, "2"])
        self.answer({"type": ["string", "null"], "enum": ["a", None]}, ["a", None], ["b", 1])

    def test_the_stricter_policies(self):
        # Each named policy of schema_exact.STRICTER, with a value that matches the schema but not the grammar.
        cases = [({"type": "object", "properties": {"a": {"type": "integer"}}}, {"a": 1, "b": 2}),
                 ({"type": "string", "pattern": "b"}, "abc"),
                 ({"type": "string", "format": "date"}, "not a date"),
                 ({"type": "string", "maxLength": 3}, 'a"b'),
                 ({"type": "number", "exclusiveMinimum": 0, "maximum": 1}, 1e-7),
                 ({"type": "string", "format": "time"}, "10:30:00z")]
        self.assertEqual(len(cases), len(STRICTER))
        for schema, value in cases:
            with self.subTest(schema=schema):
                self.assertTrue(matches(value, schema, schema))
                self.assertEqual(engine(answer_request(schema).constraint, [json.dumps(value)]), ["reject"])

    def test_prefix_items_with_items(self):
        schema = {"type": "array", "prefixItems": [{"type": "string"}], "items": {"type": "integer"}}
        self.answer(schema, [[], ["a"], ["a", 1, 2]], [[1], ["a", "b"]])

    def test_the_raw_string_enum_of_a_tool(self):
        parameters = wrap({"type": "string", "enum": ["a", "b"], "maxLength": 1})
        results = engine(tool_request(parameters).constraint, [call(x="a"), call(x="c")])
        self.assertEqual(results[0], "accept")
        self.assertIn(results[1], ("reject", "prefix"))

    def test_the_shadow_repro_without_the_nested_definition(self):
        # The fixed form of the review's schema: x's reference binds to the root N, as the schema says.
        parameters = {"type": "object", "$defs": {"N": {"type": "integer"}},
                      "properties": {"x": {"type": "object", "properties": {"z": {"$ref": "#/$defs/N"}},
                                           "required": ["z"], "additionalProperties": False}}, "required": ["x"]}
        self.assertTrue(matches({"x": {"z": 1}}, parameters, parameters))
        self.assertFalse(matches({"x": {"z": "a"}}, parameters, parameters))
        results = engine(tool_request(parameters).constraint, [call(x='{"z":1}'), call(x='{"z":"a"}')])
        self.assertEqual(results[0], "accept")
        self.assertIn(results[1], ("reject", "prefix"))


if __name__ == "__main__":
    unittest.main()
