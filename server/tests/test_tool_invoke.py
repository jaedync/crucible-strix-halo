"""Anthropic-style `<invoke name="...">` tool calls (by dineshr93).

The probe cases are the ones that the review of the pull request ran (four rounds, on its commits 7a5e83e to 33c68ae).
Each case runs in plain mode and in stream mode (`--stream-tool-calls on`), whole and in small pieces: both modes give
the same content, calls and fallback, and the joined stream deltas equal the final arguments.
The later classes are the port onto the strict-tool parser (tool_frame.py): an `<invoke>` block of a strict tool, the
`<invoke>` text inside a `<tool_call>` argument, prose that names the tag, and the end-of-turn guard (open_call).
"""
import contextlib
import io
import json
import random
import unittest
from server.api import validate_request
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, chat, chunks, invoke, joined_calls, token_events
from server.tests.test_tools import TOOLS
from server.tests.test_structured_strict import CHECK, engine
from server.tool_output import ToolOutput

INV = ('<invoke name="read">\n<parameter name="path">a.py</parameter>\n<parameter name="offset">4</parameter>\n'
       '<parameter name="flags">{"full":true}</parameter>\n</invoke>')
INV2 = ('<invoke name="read">\n<parameter name="path">c.py</parameter>\n<parameter name="offset">2</parameter>\n'
        '<parameter name="flags">{}</parameter>\n</invoke>')
TC = ('<tool_call>\n<function=read>\n<parameter=path>\nb.py\n</parameter>\n<parameter=offset>\n1\n</parameter>\n'
      '<parameter=flags>\n{}\n</parameter>\n</function>\n</tool_call>')
READ_A = 'read{"path":"a.py","offset":4,"flags":{"full":true}}'
READ_B = 'read{"path":"b.py","offset":1,"flags":{}}'
READ_C = 'read{"path":"c.py","offset":2,"flags":{}}'
STRICT = [dict(type="function", function=dict(name="f", strict=True, parameters=dict(
    type="object", properties=dict(s=dict(type="string"), o=dict(type="object", properties=dict(t=dict(type="string")),
                                                                 required=["t"], additionalProperties=False)),
    required=["s"], additionalProperties=False)))]


def result(raw, pieces=None, stream=False, tools=TOOLS, **options):
    """(finish, content, calls as name+arguments, fallback, invalid) and the stream check of one reply."""
    body = dict(messages=[dict(role="user", content="Read")], tools=tools, tool_choice="auto") | options
    output = ToolOutput(validate_request(body, "chat", "crucible"), stream=stream)
    pieces = [raw] if pieces is None else [raw[i:i + pieces] for i in range(0, len(raw), pieces)]
    streamed, deltas = "", []
    for piece in pieces:
        streamed += output.feed(piece)
        deltas += output.take_deltas()
    message, tail, finish = output.finish("stop")
    calls = [c["function"]["name"] + c["function"]["arguments"] for c in message.get("tool_calls") or []]
    if not output.fallback:
        joined = joined_calls([c for d in deltas + tail for c in d["tool_calls"]])
        assert [c["function"]["name"] + c["function"]["arguments"] for c in joined] == calls
        assert [c["id"] for c in joined] == [c["id"] for c in message.get("tool_calls") or []]
    assert (message["content"] or "").startswith(streamed)
    return finish, message["content"], calls, output.fallback, output.invalid


MODES = [dict(), dict(pieces=3), dict(pieces=1, stream=True), dict(pieces=3, stream=True), dict(stream=True)]

# (probe id, model output, finish, content, calls, fallback). The probe ids are the names in the review probes.
PROBES = [
    ("ends-with-lt", "Use a <", "stop", "Use a <", [], False),
    ("ends-with-<i", "The tag starts with <i", "stop", "The tag starts with <i", [], False),
    ("ends-<invo", "write <invo", "stop", "write <invo", [], False),
    ("prose-invoke", 'Claude writes <invoke name="read"> blocks. Done.', "stop",
     'Claude writes <invoke name="read"> blocks. Done.', [], False),
    ("prose-invoke-long", 'Claude writes <invoke name="read"> blocks; here is how they look. Done.', "stop",
     'Claude writes <invoke name="read"> blocks; here is how they look. Done.', [], False),
    ("invoke-call", "I will read.\n" + INV, "tool_calls", "I will read.\n", [READ_A], False),
    ("prose-then-tool_call", 'See <invoke name="x"> text.\n' + TC, "tool_calls", 'See <invoke name="x"> text.\n',
     [READ_B], False),
    ("unknown-fn-tc", "<tool_call>\n<function=rm_rf>\n<parameter=p>\nx\n</parameter>\n</function>\n</tool_call>",
     "tool_calls", None, ['rm_rf{"p":"x"}'], False),
    ("unknown-fn-invoke", '<invoke name="rm_rf">\n<parameter name="p">x</parameter>\n</invoke>', "tool_calls", None,
     ['rm_rf{"p":"x"}'], False),
    ("prose-then-invoke", 'See <invoke name="x"> text.\n' + INV2, "tool_calls", 'See <invoke name="x"> text.\n',
     [READ_C], False),
    ("prose-delete_file-then-invoke", 'See <invoke name="delete_file"> text.\n' + INV2, "tool_calls",
     'See <invoke name="delete_file"> text.\n', [READ_C], False),
    ("two-invokes", INV + "\n" + INV2, "tool_calls", None, [READ_A, READ_C], False),
    ("invoke-then-tool_call", INV + "\n" + TC, "tool_calls", None, [READ_A, READ_B], False),
    ("tool_call-then-prose-invoke", TC + '\nNote <invoke name="y"> here.', "stop", TC + '\nNote <invoke name="y"> here.',
     [], True),
    ("invoke-in-arg", TC.replace("\nb.py\n", '\n<invoke name="z">\n'), "tool_calls", None,
     ['read{"path":"<invoke name=\\"z\\">","offset":1,"flags":{}}'], False),
]


class ProbeTests(unittest.TestCase):
    def test_probe_cases_in_every_mode(self):
        for name, raw, finish, content, calls, fallback in PROBES:
            for mode in MODES:
                with self.subTest(probe=name, **mode):
                    got = result(raw, **mode)
                    # A fallback after a streamed delta ends with "length" (tool_output.finish).
                    started = fallback and mode.get("stream") and raw.startswith("<tool_call>")
                    self.assertEqual(got[:4], ("length" if started else finish, content, calls, fallback))


class ArgumentTextTests(unittest.TestCase):
    """`<invoke>` text inside a `<tool_call>` argument is argument text: the parser never rewrites it."""

    def test_an_argument_keeps_a_complete_invoke_block(self):
        for value in ('<invoke name="z"></invoke>', '<invoke name="z">text</invoke>', 'a <invoke name="z">\n</invoke> b'):
            for mode in MODES:
                with self.subTest(value=value, **mode):
                    got = result(TC.replace("\nb.py\n", "\n" + value + "\n"), **mode)
                    self.assertEqual(got[:4], ("tool_calls", None, ['read' + json.dumps(
                        dict(path=value, offset=1, flags={}), separators=(",", ":"))], False))

    def test_a_strict_argument_keeps_a_complete_invoke_block(self):
        value = 'see <invoke name="z">v</invoke>'
        raw = f"<tool_call>\n<function=f>\n<parameter=s>\n{value}\n</parameter>\n</function>\n</tool_call>"
        for mode in MODES:
            with self.subTest(**mode):
                self.assertEqual(result(raw, tools=STRICT, **mode)[:4],
                                 ("tool_calls", None, ['f' + json.dumps(dict(s=value), separators=(",", ":"))], False))


class StrictInvokeTests(unittest.TestCase):
    """An `<invoke>` block of a strict tool. In auto mode the grammar lets free text come before the first
    `<tool_call>` (structured.py), so the block is not under the grammar: the parser checks it strictly."""

    def test_the_request_has_the_call_grammar(self):
        body = dict(messages=[dict(role="user", content="Hi")], tools=STRICT, tool_choice="auto")
        self.assertEqual(validate_request(body, "chat", "crucible").constraint["kind"], "structural_tag")

    @unittest.skipUnless(CHECK.exists(), "needs build/tests/constraint_text_check")
    def test_the_grammar_takes_a_block_as_free_text(self):
        # The engine's own grammar: a block, then a call, is a whole reply; text after a call is not.
        body = dict(messages=[dict(role="user", content="Hi")], tools=STRICT, tool_choice="auto")
        block = '<invoke name="f">\n<parameter name="s">a</parameter>\n</invoke>'
        tool_call = "<tool_call>\n<function=f>\n<parameter=s>\na\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(engine(validate_request(body, "chat", "crucible").constraint,
                                [block, block + "\n" + tool_call, tool_call + "\n" + block]),
                         ["accept", "accept", "reject"])

    def test_a_json_value_can_hold_the_parameter_end_tag(self):
        raw = ('<invoke name="f">\n<parameter name="s">a</parameter>\n'
               '<parameter name="o">{"t":"x</parameter>y"}</parameter>\n</invoke>')
        for mode in MODES:
            with self.subTest(**mode):
                self.assertEqual(result(raw, tools=STRICT, **mode)[:4], (
                    "tool_calls", None, ['f{"s":"a","o":{"t":"x</parameter>y"}}'], False))

    def test_a_schema_failure_falls_back_as_for_a_tool_call(self):
        raw = '<invoke name="f">\n<parameter name="o">{"t":1}</parameter>\n</invoke>'
        for mode in MODES:
            with self.subTest(**mode):
                self.assertEqual(result(raw, tools=STRICT, **mode), ("stop", raw, [], True, "schema_mismatch"))

    def test_the_fallback_of_an_invoke_block_is_not_a_grammar_fault(self):
        # structured.violation marks a fallback under the grammar as an engine fault; the grammar never read this block.
        text = '<invoke name="f">\n<parameter name="o">{"t":1}</parameter>\n</invoke>'
        for stream in (False, True):
            log = io.StringIO()
            with self.subTest(stream=stream), contextlib.redirect_stderr(log):
                backend = ScriptBackend(token_events(list(text)))
                backend.constraint_ready = True  # the engine serves the grammar, so the request has it
                response = invoke(backend, PiecesTokenizer(list(text)),
                                  chat(tools=STRICT, stream=stream, max_tokens=len(text) + 1))
                content = ("".join(c["choices"][0]["delta"].get("content") or "" for c in chunks(response) if c["choices"])
                           if stream else json.loads(response.body)["choices"][0]["message"]["content"])
                self.assertEqual(content, text)
                lines = [json.loads(line.split(" ", 1)[1]) for line in log.getvalue().splitlines()
                         if line.startswith("crucible_error ")]
                self.assertEqual([(line["error"], line["invalid"]) for line in lines],
                                 [("tool_call_fallback", "schema_mismatch")])
                self.assertNotIn("constraint_violation", lines[0])

    def fallback_line(self, text, **options):
        """The crucible_error line of a reply under the strict call grammar (stream off)."""
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            backend = ScriptBackend(token_events(list(text)))
            backend.constraint_ready = True
            invoke(backend, PiecesTokenizer(list(text)), chat(tools=STRICT, max_tokens=len(text) + 1, **options))
        lines = [json.loads(line.split(" ", 1)[1]) for line in log.getvalue().splitlines()
                 if line.startswith("crucible_error ")]
        self.assertEqual(len(lines), 1)
        return lines[0]

    GOOD = "<tool_call>\n<function=f>\n<parameter=s>\na\n</parameter>\n</function>\n</tool_call>"
    BLOCK = '<invoke name="f">\n<parameter name="s">a</parameter>\n</invoke>'

    def test_a_bad_call_after_a_block_is_a_grammar_fault(self):
        # Review repro A: the grammar reads the <tool_call> after the free text, so its fault is an engine fault.
        bad = self.GOOD.replace("</parameter>\n</function>", '</parameter>\n<parameter=o>\n{"t":1}\n</parameter>\n</function>')
        line = self.fallback_line(self.BLOCK + "\n" + bad)
        self.assertEqual((line["invalid"], line.get("constraint_violation")), ("schema_mismatch", 1))

    def test_invoke_text_after_a_call_is_a_grammar_fault(self):
        # Review repro B: the grammar takes no text after a call.
        line = self.fallback_line(self.GOOD + '\n<invoke name="f"> oops')
        self.assertEqual((line["invalid"], line.get("constraint_violation")), ("unexpected_text", 1))

    def test_text_after_a_block_before_any_call_is_not_a_grammar_fault(self):
        line = self.fallback_line(self.BLOCK + " Done.")
        self.assertEqual(line["invalid"], "unexpected_text")
        self.assertNotIn("constraint_violation", line)

    def test_a_block_is_a_grammar_fault_when_the_grammar_has_no_free_text(self):
        # Review M1b: with tool_choice required or named, the grammar starts the reply with <tool_call> (structured.py).
        bad = '<invoke name="f">\n<parameter name="s">a</parameter>\n<parameter name="o">{"t":1}</parameter>\n</invoke>'
        line = self.fallback_line(bad, tool_choice="required")
        self.assertEqual((line["invalid"], line.get("constraint_violation")), ("schema_mismatch", 1))
        other = '<invoke name="g">\n<parameter name="s">a</parameter>\n</invoke>'
        line = self.fallback_line(other, tool_choice=dict(type="function", function=dict(name="f")))
        self.assertEqual((line["invalid"], line.get("constraint_violation")), ("tool_choice", 1))

    def test_a_policy_fault_that_counts_a_block_call_is_not_a_grammar_fault(self):
        line = self.fallback_line(self.BLOCK + "\n" + self.GOOD, parallel_tool_calls=False)
        self.assertEqual(line["invalid"], "parallel_calls")
        self.assertNotIn("constraint_violation", line)


class ProseTests(unittest.TestCase):
    """Prose that names `<invoke ...>` is content: it is not a call that the model has not closed."""

    def output(self, text, stream=False):
        body = dict(messages=[dict(role="user", content="Read")], tools=TOOLS, tool_choice="auto")
        output = ToolOutput(validate_request(body, "chat", "crucible"), stream=stream)
        return output, output.feed(text)

    def test_open_call_is_false_after_prose(self):
        # http.py keeps an end-of-turn token as text while open_call() is true: prose must not hold the reply open.
        for text in ('See <invoke name="x"> text.', 'Use <invoke>.', '<invoke name="read">\nhello',
                     INV + "\nDone.", '<invoke name="read">\n<parameter name="path">a</parameter> x'):
            with self.subTest(text=text):
                self.assertFalse(self.output(text)[0].open_call())

    def test_open_call_is_true_inside_a_block(self):
        for text in ('<invoke na', '<invoke name="re', '<invoke name="read">', '<invoke name="read">\n<par',
                     '<invoke name="read">\n<parameter name="path">a', INV[:-3], INV + "\n" + INV2[:30], TC[:40]):
            with self.subTest(text=text):
                self.assertTrue(self.output(text)[0].open_call())

    def test_invoke_text_after_a_tool_call_does_not_open_a_call(self):
        # Review L2: as in the base, only a <tool_call> start can follow a <tool_call> (the grammar takes no other text).
        for tail in ("<i", "<invo", "<invoke", '<invoke name="read">\n<parameter name="path">a'):
            with self.subTest(tail=tail):
                self.assertFalse(self.output(TC + "\n" + tail)[0].open_call())
                self.assertTrue(self.output(INV + "\n" + tail)[0].open_call())
        self.assertTrue(self.output(TC + "\n<tool")[0].open_call())

    def test_prose_after_the_tag_streams_before_the_reply_ends(self):
        for text in ('See <invoke name="x"> text. More text.', 'Use <invoke> and <invoke name="y"/>.'):
            with self.subTest(text=text):
                self.assertEqual(self.output(text)[1], text)

    def test_a_call_after_prose_streams_its_deltas(self):
        raw = 'See <invoke name="x"> text.\n' + TC
        output, streamed = self.output("", stream=True)
        deltas = []
        for index in range(0, len(raw), 3):
            streamed += output.feed(raw[index:index + 3])
            deltas += output.take_deltas()
        self.assertEqual(streamed, 'See <invoke name="x"> text.\n')
        self.assertTrue(deltas)
        message, tail, finish = output.finish("stop")
        self.assertEqual(joined_calls([c for d in deltas + tail for c in d["tool_calls"]])[0]["function"]["arguments"],
                         message["tool_calls"][0]["function"]["arguments"])

    def test_text_in_a_block_that_is_not_a_parameter_is_prose(self):
        raw = '<invoke name="read">\nhello\n<parameter name="path">a</parameter>\n</invoke>'
        for mode in MODES:
            with self.subTest(**mode):
                self.assertEqual(result(raw, **mode)[:4], ("stop", raw, [], False))


class MalformedTests(unittest.TestCase):
    """A block that leaves the form after a parameter tag is a malformed call, as a malformed `<tool_call>` is."""

    def test_text_after_a_parameter_falls_back(self):
        for raw in ('<invoke name="read">\n<parameter name="path">a</parameter> x\n</invoke>',
                    '<invoke name="read">\n<parameter name="path">a</parameter>\n</invoke' + "d",
                    'See <invoke name="read"><parameter name="path">a</parameter> and so on.\n' + INV2):
            for mode in MODES:
                with self.subTest(raw=raw, **mode):
                    self.assertEqual(result(raw, **mode), ("stop", raw, [], True, "parameter_tag"))

    def test_a_block_cut_after_a_parameter_tag_falls_back(self):
        for raw in ('<invoke name="read">\n<parameter name="path">a', '<invoke name="read">\n<parameter name="path">a'
                    '</parameter>\n<para', INV + "\n" + INV2[:50]):
            for mode in MODES:
                with self.subTest(raw=raw, **mode):
                    self.assertEqual(result(raw, **mode), ("stop", raw, [], True, "parameter_tag"))

    def test_a_block_cut_before_a_parameter_tag_is_prose(self):
        for raw in ('Use <invoke', 'Use <invoke name="read"', 'Use <invoke name="read">\n', 'Use <invoke name="read">\n<par'):
            for mode in MODES:
                with self.subTest(raw=raw, **mode):
                    self.assertEqual(result(raw, **mode), ("stop", raw, [], False, None))

    def test_a_long_open_block_is_read_a_few_times(self):
        # The prose check reads the whole block: while it grows, it runs only when the text has doubled.
        from unittest import mock
        from server import tool_output
        reads = []
        real = tool_output.block_end
        raw = '<invoke name="read">\n<parameter name="path">' + "x" * 200000 + "</parameter>\n</invoke>"
        with mock.patch.object(tool_output, "block_end", lambda *a, **k: reads.append(1) or real(*a, **k)):
            got = result(raw, pieces=4, stream=True)
        self.assertEqual(got[2], ["read" + json.dumps(dict(path="x" * 200000), separators=(",", ":"))])
        self.assertLess(len(reads), 120)


PARTS = ["Text. ", "<", "<i", "<invoke", '<invoke name="x">', "</invoke>", "\n", " ", "<tool_call>", "</tool_call>",
         "<parameter=path>", "</parameter>", '<parameter name="path">', "a.py", "{}", "<function=read>", "</function>"]


class RandomEqualityTests(unittest.TestCase):
    """Random replies of calls, tags and prose: stream mode and plain mode agree (content, calls, fallback, code)."""

    def reply(self, rng):
        blocks = []
        for _ in range(rng.randint(1, 4)):
            kind = rng.randint(0, 3)
            if kind == 0:
                blocks.append(rng.choice((INV, INV2, TC)))
            elif kind == 1:
                block = rng.choice((INV, INV2, TC))
                cut = rng.randint(0, len(block))
                blocks.append(block[:cut] + rng.choice(PARTS) + block[cut:])
            else:
                blocks.append("".join(rng.choice(PARTS) for _ in range(rng.randint(1, 5))))
        return rng.choice(("", "Answer. ")) + rng.choice(("\n", "")).join(blocks)

    def test_stream_and_plain_agree(self):
        rng = random.Random(15)
        for case in range(600):
            raw = self.reply(rng)
            plain = result(raw)
            for pieces in (1, rng.randint(2, 7)):
                with self.subTest(case=case, raw=raw, pieces=pieces):
                    finish, *rest = result(raw, pieces=pieces, stream=True)
                    self.assertEqual(rest, list(plain[1:]))
                    self.assertIn(finish, (plain[0], "length" if plain[3] else plain[0]))
