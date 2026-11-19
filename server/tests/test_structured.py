"""Structured output on the server (server/structured.py): the API fields, the 400 refusals, the engine payload."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from server.api import ApiError, validate_request
from server.backend import EngineBackend
from server import structured

FAKE = Path(__file__).parent / "fake_engine.py"
SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
          "required": ["city"], "additionalProperties": False}
TOOLS = [dict(type="function", function=dict(name="get_weather", strict=True, parameters=dict(
             type="object", properties=dict(city=dict(type="string"), unit=dict(type="string", enum=["c", "f"]),
                                            days=dict(type="integer")),
             required=["city", "unit"], additionalProperties=False))),
         dict(type="function", function=dict(name="search", parameters=dict(
             type="object", properties=dict(query=dict(type="string")), required=["query"])))]


def chat(**fields):
    body = dict(messages=[dict(role="user", content="Hi")], max_tokens=8) | fields
    return validate_request(body, "chat", "crucible")


def tag(request):
    assert request.constraint["kind"] == "structural_tag"
    return json.loads(request.constraint["spec"])


class ResponseFormatTests(unittest.TestCase):
    def test_shapes(self):
        self.assertIsNone(chat().constraint)
        self.assertIsNone(chat(response_format=dict(type="text")).constraint)
        self.assertEqual(chat(response_format=dict(type="json_object")).constraint, dict(kind="json_object", spec=""))
        for strict in (True, False, None):
            spec = dict(name="w", schema=SCHEMA) | ({} if strict is None else dict(strict=strict))
            # Enforced also when strict is false or absent.
            self.assertEqual(chat(response_format=dict(type="json_schema", json_schema=spec)).constraint,
                             dict(kind="json_schema", spec=json.dumps(SCHEMA, separators=(",", ":"))))

    def test_bad_shapes(self):
        for value in ("json", dict(type="xml"), dict(type="json_object", schema={}),
                      dict(type="json_schema"), dict(type="json_schema", json_schema=dict(schema=[])),
                      dict(type="json_schema", json_schema=dict(schema=SCHEMA, extra=1))):
            with self.assertRaises(ApiError) as caught:
                chat(response_format=value)
            self.assertEqual(caught.exception.status, 400)
        with self.assertRaises(ApiError):
            validate_request(dict(prompt="x", response_format=dict(type="json_object")), "completion", "crucible")

    def test_refused_keywords_name_the_keyword(self):
        for keyword, value in (("allOf", [{"type": "object"}]), ("oneOf", [{"type": "string"}]), ("not", {}),
                               ("if", {}), ("patternProperties", {}), ("uniqueItems", True), ("multipleOf", 2),
                               ("minProperties", 1), ("contains", {}), ("propertyNames", {})):
            schema = {"type": "object", "properties": {"a": {"type": "array", keyword: value}}}
            with self.assertRaises(ApiError) as caught:
                chat(response_format=dict(type="json_schema", json_schema=dict(name="x", schema=schema)))
            self.assertEqual(caught.exception.status, 400)
            self.assertIn(keyword, str(caught.exception))
            self.assertEqual(caught.exception.code, "constraint_keyword")

    def test_impossible_schemas(self):
        for schema in ({"type": "array", "minItems": 3, "maxItems": 1}, {"enum": []},
                       {"type": "string", "minLength": 4, "maxLength": 2}):
            with self.assertRaises(ApiError) as caught:
                chat(response_format=dict(type="json_schema", json_schema=dict(name="x", schema=schema)))
            self.assertEqual(caught.exception.status, 400)

    def test_logprobs_refused(self):
        with self.assertRaises(ApiError):
            validate_request(dict(prompt="x", max_tokens=1, logprobs=True, response_format=dict(type="json_object")),
                             "completion", "crucible")


class ToolTests(unittest.TestCase):
    def test_auto_without_strict_tools_has_no_constraint(self):
        # Tool-call syntax for every tool request stays off.
        tools = [TOOLS[1]]
        self.assertIsNone(chat(tools=tools).constraint)
        self.assertIsNone(chat(tools=TOOLS, tool_choice="none").constraint)

    def test_auto_with_a_strict_tool(self):
        # Text, then the calls: the first at the trigger, more after "\n" (parallel), nothing after them.
        whole = tag(chat(tools=TOOLS))["format"]
        self.assertEqual(whole["type"], "sequence")
        fmt, more = whole["elements"]
        self.assertEqual(more["type"], "star")
        self.assertEqual(fmt["type"], "triggered_tags")
        self.assertEqual(fmt["triggers"], ["<tool_call>"])
        self.assertTrue(fmt["stop_after_first"])
        self.assertEqual(tag(chat(tools=TOOLS, parallel_tool_calls=False))["format"]["type"], "triggered_tags")
        strict, loose = fmt["tags"]
        self.assertEqual(strict["begin"], "<tool_call>\n<function=get_weather>\n")
        self.assertEqual(strict["end"], "</function>\n</tool_call>")
        city, unit, days = strict["content"]["elements"]
        # The template's exact newlines: a raw string parameter, an enum, then an optional JSON parameter.
        self.assertEqual(city, dict(type="tag", begin="<parameter=city>\n", end="\n</parameter>\n",
                                    content=dict(type="any_text", excludes=["</parameter>"])))
        self.assertEqual(unit["elements"][1], dict(type="or", elements=[dict(type="const_string", value="c"),
                                                                         dict(type="const_string", value="f")]))
        self.assertEqual(days["type"], "optional")
        self.assertEqual(days["content"]["elements"][1]["type"], "json_schema")
        self.assertEqual(loose["type"], "tag")  # a tool that is not strict takes any parameter text
        self.assertTrue(tag(chat(tools=TOOLS, parallel_tool_calls=False))["format"]["stop_after_first"])

    def test_required_and_named(self):
        required = tag(chat(tools=TOOLS, tool_choice="required"))["format"]
        self.assertEqual(required["type"], "sequence")
        self.assertEqual(required["elements"][0]["type"], "or")
        one = tag(chat(tools=TOOLS, tool_choice="required", parallel_tool_calls=False))["format"]
        self.assertEqual(one["type"], "or")
        named = tag(chat(tools=TOOLS, tool_choice=dict(type="function", function=dict(name="get_weather"))))["format"]
        self.assertEqual(named["begin"], "<tool_call>\n<function=get_weather>\n")
        # Named and required are enforced also for a tool that is not strict.
        loose = tag(chat(tools=TOOLS, tool_choice=dict(type="function", function=dict(name="search"))))["format"]
        self.assertEqual(loose["begin"], "<tool_call>\n<function=search>\n")

    def test_response_format_with_tools(self):
        fmt = tag(chat(tools=TOOLS, response_format=dict(type="json_object")))["format"]
        self.assertEqual(fmt["type"], "or")
        self.assertEqual(fmt["elements"][0]["type"], "json_schema")

    def test_strict_string_keywords_refused(self):
        tools = [dict(type="function", function=dict(name="f", strict=True, parameters=dict(
            type="object", properties=dict(a=dict(type="string", maxLength=4)))))]
        with self.assertRaises(ApiError) as caught:
            chat(tools=tools)
        self.assertIn("maxLength", str(caught.exception))

    def test_refs_keep_definitions(self):
        tools = [dict(type="function", function=dict(name="f", strict=True, parameters={
            "type": "object", "$defs": {"p": {"type": "object", "properties": {"x": {"type": "integer"}}}},
            "properties": {"a": {"$ref": "#/$defs/p"}}, "required": ["a"]}))]
        value = tag(chat(tools=tools))["format"]["elements"][0]["tags"][0]["content"]["elements"][0]["elements"][1]
        self.assertEqual(value["json_schema"]["$defs"]["p"]["properties"]["x"]["type"], "integer")


class PayloadTests(unittest.TestCase):
    def setUp(self):
        popen = subprocess.Popen
        with patch("server.backend.subprocess.Popen",
                   side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            self.backend = EngineBackend("unused", "unused", context=32)

    def tearDown(self):
        self.backend.close()

    def payload(self, request, **options):
        job = self.backend.submit([1, 2, 3], request, **options)
        job.cancelled.set()
        return job.payload

    def test_no_constraint_payload_is_byte_identical(self):
        # Recorded on trunk 33d3484f: a chat request with a tool that is not strict and thinking on.
        request = chat(tools=[TOOLS[1]], enable_thinking=True, seed=5)
        self.assertEqual(json.dumps(self.payload(request), separators=(",", ":")), GOLDEN)

    def test_constraint_payload(self):
        with self.assertRaises(ApiError) as caught:
            self.payload(chat(response_format=dict(type="json_object")))
        self.assertEqual(caught.exception.status, 400)  # an engine without the ready field
        self.backend.constraint_ready = True
        off = self.payload(chat(response_format=dict(type="json_object"), enable_thinking=False))
        self.assertEqual(off["constraint"], dict(kind="json_object", spec="", reasoning=0))
        self.assertNotIn("constraint_from", off)
        on = self.payload(chat(response_format=dict(type="json_object"), enable_thinking=True), constraint_from=2)
        self.assertEqual(on["constraint"]["reasoning"], 1)
        self.assertEqual(on["constraint_from"], 2)
        self.assertNotIn("constraint", self.payload(chat(enable_thinking=True)))


class EngineRefusalTests(unittest.TestCase):
    def test_codes(self):
        self.assertEqual(structured.engine_refusal("constraint_compile: the schema does not compile to a grammar"),
                         structured.ENGINE_CODES["constraint_compile"])
        self.assertIsNone(structured.engine_refusal("all sequence slots are in use"))
        self.assertIsNone(structured.engine_refusal(None))


class StopStringTests(unittest.TestCase):
    def test_stop_strings_skip_constrained_answers(self):
        from server.response import CompletionResponse
        class Tokenizer:
            def stream(self): return None
        service = type("S", (), dict(model="crucible", tokenizer=Tokenizer()))()
        plain = CompletionResponse(None, service, chat(stop=["}"]), [1])
        constrained = CompletionResponse(None, service, chat(stop=["}"], response_format=dict(type="json_object")), [1])
        self.assertEqual(plain.stop_filter.feed("a}b"), "a")
        self.assertEqual(constrained.stop_filter.feed("a}b"), "a}b")


class TurnLineTests(unittest.TestCase):
    """The done event's constraint counters reach crucible_turn as struct_* numbers; other turns have none."""
    def fields(self, event):
        from server import metrics
        return json.loads(metrics.turn_line({}, event).split(" ", 1)[1])

    def test_struct_fields(self):
        got = self.fields(dict(event="done", completion_tokens=5, constraint=dict(
            tokens=5, refusals=2, cut_drafts=3, fallback_us=40, compile_us=900, cache_hit=1)))
        self.assertEqual({k: v for k, v in got.items() if k.startswith("struct_")},
                         dict(struct_tokens=5, struct_refusals=2, struct_cut_drafts=3, struct_fallback_us=40,
                              struct_compile_us=900, struct_cache_hit=1))

    def test_plain_turn_has_none(self):
        self.assertFalse([k for k in self.fields(dict(event="done", completion_tokens=5)) if k.startswith("struct_")])

    def test_bad_values_dropped(self):
        got = self.fields(dict(event="done", constraint=dict(tokens="x", refusals=-1, cut_drafts=True, compile_us=3)))
        self.assertEqual({k: v for k, v in got.items() if k.startswith("struct_")}, dict(struct_compile_us=3))
        self.assertFalse([k for k in self.fields(dict(event="done", constraint="text")) if k.startswith("struct_")])


class ViolationTests(unittest.TestCase):
    """A tool-call fallback where every callable tool is strict and calls are enforced is an engine fault."""
    def test_enforced(self):
        strict, loose = TOOLS[:1], TOOLS[1:]
        named = dict(type="function", function=dict(name="get_weather"))
        self.assertTrue(structured.violation(chat(tools=strict, tool_choice="required")))
        self.assertTrue(structured.violation(chat(tools=strict)))  # strict tools, auto: the grammar is on
        self.assertTrue(structured.violation(chat(tools=TOOLS, tool_choice=named)))
        self.assertFalse(structured.violation(chat(tools=loose, tool_choice="required")))  # free values
        self.assertFalse(structured.violation(chat(tools=TOOLS, tool_choice="required")))
        self.assertFalse(structured.violation(chat(tools=loose)))
        self.assertFalse(structured.violation(chat(tools=strict, tool_choice="none")))
        self.assertFalse(structured.violation(chat(response_format=dict(type="json_object"))))

    def test_fallback_line_names_it(self):
        from server import metrics
        line = json.loads(metrics.fallback_line("r", 4, False, violation=True).split(" ", 1)[1])
        self.assertEqual(line["constraint_violation"], 1)
        self.assertNotIn("constraint_violation", json.loads(metrics.fallback_line("r", 4, False).split(" ", 1)[1]))


class EngineWithoutStructuredOutputTests(unittest.TestCase):
    """An engine without the ready field: tool requests run as on trunk (no constraint), response_format is a 400."""
    def test_tools_run_unconstrained(self):
        for choice in ("required", "auto", dict(type="function", function=dict(name="get_weather"))):
            body = dict(messages=[dict(role="user", content="Hi")], tools=TOOLS[:1], tool_choice=choice)
            self.assertIsNone(validate_request(body, "chat", "crucible", structured=False).constraint)
            self.assertIsNotNone(validate_request(body, "chat", "crucible").constraint)

    def test_response_format_refused(self):
        body = dict(messages=[dict(role="user", content="Hi")], response_format=dict(type="json_object"))
        with self.assertRaises(ApiError) as caught:
            validate_request(body, "chat", "crucible", structured=False)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.param, "response_format")
        text = dict(messages=[dict(role="user", content="Hi")], response_format=dict(type="text"))
        self.assertIsNone(validate_request(text, "chat", "crucible", structured=False).constraint)


GOLDEN = ('{"ids":[1,2,3],"max_tokens":8,"spec":7,"stop_ids":[248046,248044],"temperature":0.0,"top_k":0,"top_p":1.0,'
          '"min_p":0.0,"seed":5,"cache_control":"auto","cache_prefix_tokens":0,"draft_q":0.3,"draft_head":"64k",'
          '"mtp_prefix":256,"lookup":7,"lookup_min":8,"cache_tail":"guard"}')

if __name__ == "__main__":
    unittest.main()
