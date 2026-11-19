"""Session replay after a tool-call turn (rule E1).

The engine keeps the state after prompt + generated IDs. The next prompt resumes from it only when it repeats those
IDs. The template renders a returned tool call from its parsed arguments, so the next prompt left that tail inside a
long call when the model wrote another JSON form or a token split that the tokenizer does not make (Nov 16: 10
"session_replay diverged" turns, two of them 6,500 tokens into the previous output).

Rule E1: the prompt stays a function of the request only. At the end of a turn with tool calls, the server prefills
the returned message as the next request renders it, so the session tail is that render. A client that sends the
message back unchanged then extends the tail.

Every text here is synthetic. The generated IDs go through the real HTTP and parser path
(SequenceBackend, as test_ra8_replay does). The tests run with the synthetic tokenizer and, with CRUCIBLE_TEST_CKPT,
with the real checkpoint tokenizer.
"""
import json
import random
from pathlib import Path

import pytest

from server.tests.serve_fixtures import chunks, invoke, joined_calls
from server.tests.test_content_tokens import CHECKPOINT, synthetic_checkpoint
from server.tests.test_think_budget import SequenceBackend
from server.tokenization import ModelTokenizer

CHAT, MESSAGES, RESPONSES = "/v1/chat/completions", "/v1/messages", "/v1/responses"
TOOL_CALL, TOOL_CALL_END = 248058, 248059
STR = dict(type="string")
SCHEMAS = dict(
    read=dict(type="object", properties=dict(path=STR, offset=dict(type="number"), limit=dict(type="number")),
              required=["path"]),
    bash=dict(type="object", properties=dict(command=STR, timeout=dict(type="number")), required=["command"]),
    edit=dict(type="object", properties=dict(path=STR, edits=dict(type="array", items=dict(
        type="object", properties=dict(oldText=STR, newText=STR), required=["oldText", "newText"]))),
        required=["path", "edits"]),
    write=dict(type="object", properties=dict(path=STR, content=STR), required=["path", "content"]),
    # No declared type: the parser reads the value as JSON when it can (tool_output.argument_value).
    note=dict(type="object", properties=dict(data={}), required=["data"]))
TOOLS = [dict(type="function", function=dict(name=name, description=f"The {name} tool.", parameters=schema))
         for name, schema in SCHEMAS.items()]
ANTHROPIC_TOOLS = [dict(name=name, description=f"The {name} tool.", input_schema=schema) for name, schema in SCHEMAS.items()]
RESPONSES_TOOLS = [dict(type="function", name=name, description=f"The {name} tool.", parameters=schema)
                   for name, schema in SCHEMAS.items()]
REASONING = "The user wants the file. I will use the tool."
USER = [dict(role="system", content="You are a coding agent."), dict(role="user", content="Change src/app.py.")]


@pytest.fixture(scope="module", params=["synthetic", "checkpoint"])
def root(request, tmp_path_factory):
    if request.param == "checkpoint":
        if not CHECKPOINT:
            pytest.skip("Set CRUCIBLE_TEST_CKPT to test the real checkpoint tokenizer.")
        return Path(CHECKPOINT)
    return synthetic_checkpoint(tmp_path_factory.mktemp("tokenizer"))


def source(rng, lines):
    """Synthetic code with blank lines, indents, quotes, escapes and non-ASCII text."""
    words = ["value", "index", "total", "caf\u00e9", "\u00fcber", "items", "x"]
    out = []
    for number in range(lines):
        pad = "    " * rng.randint(0, 3)
        kind = rng.randint(0, 5)
        if kind == 0:
            out.append("")
        elif kind == 1:
            out.append(f"{pad}def f{number}({rng.choice(words)}, {rng.choice(words)}=None):")
        elif kind == 2:
            out.append(f'{pad}"""Return the {rng.choice(words)} \u2014 "quoted" and \\escaped."""')
        elif kind == 3:
            out.append(f"{pad}print(f\"{{{rng.choice(words)}!r}}\\t\\n\", end='')  # \U0001f642 note")
        elif kind == 4:
            out.append(f"{pad}\t{rng.choice(words)} = {{'a': [1, 2.5, -3e-2], \"b\": None}}   ")
        else:
            out.append(f"{pad}{rng.choice(words)} = {rng.choice(words)}.get({rng.choice(words)!r}, {rng.randint(0, 99)})")
    return "\n".join(out) + "\n"


FORMS = {
    "tojson": lambda value: json.dumps(value, ensure_ascii=False),
    "compact": lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":")),
    "ascii": lambda value: json.dumps(value),
    "indent": lambda value: json.dumps(value, ensure_ascii=False, indent=2),
}


def call_text(name, arguments, form=FORMS["tojson"]):
    """The checkpoint's tool-call form, as the model writes it."""
    parts = [f"<tool_call>\n<function={name}>\n"]
    for key, value in arguments.items():
        parts.append(f"<parameter={key}>\n{value if isinstance(value, str) else form(value)}\n</parameter>\n")
    return "".join(parts) + "</function>\n</tool_call>"


def output_text(calls, form=FORMS["tojson"], content="I will change the file.", thinking=True):
    text = (REASONING + "\n</think>\n\n") if thinking else ""
    text += (content + "\n\n") if content else ""
    return text + "\n".join(call_text(name, arguments, form) for name, arguments in calls) + "<|im_end|>"


def resplit(adapter, ids, rng, rate):
    """A model-like token split that the tokenizer does not make: some tokens inside the calls become two tokens of
    the same text. (The reasoning and the answer text are outside this node.)"""
    tokenizer, out, inside = adapter.tokenizer, [], False
    for token in ids:
        inside = (inside or token == TOOL_CALL) and token != TOOL_CALL_END
        piece = tokenizer.id_to_token(token)
        pair = None
        if inside and token < 248044 and piece and len(piece) > 1 and rng.random() < rate:
            pair = next(((tokenizer.token_to_id(piece[:cut]), tokenizer.token_to_id(piece[cut:]))
                         for cut in range(1, len(piece)) if tokenizer.token_to_id(piece[:cut]) is not None and
                         tokenizer.token_to_id(piece[cut:]) is not None), None)
        out += list(pair) if pair else [token]
    return out


def generated(adapter, text, split=0.0, seed=3):
    ids = adapter.tokenizer.encode(text, add_special_tokens=False).ids
    if split:
        ids = resplit(adapter, ids, random.Random(seed), split)
    assert adapter.decode(ids) == text.replace("<|im_end|>", "")
    return ids


REPLY = "Done."


def next_message_ids(adapter):
    """The tokens that the model writes after a closed assistant message: the template opens the next message (the
    tool results) with <|im_start|>user. The test assumes that the greedy extension job writes them."""
    return [adapter.tokenizer.token_to_id("<|im_start|>"), adapter.tokenizer.token_to_id("user")]


def serve(adapter, body, ids, path=CHAT, extend=True, extension_ids=None):
    """(backend, response) of one served turn. The backend has a script for the tail extension job too."""
    extension_ids = next_message_ids(adapter) if extension_ids is None else extension_ids
    backend = SequenceBackend((list(ids), "stop"), (list(extension_ids), "length"))
    backend.extension_ids = list(extension_ids)
    response = invoke(backend, adapter, body, path=path, service_options=dict(extend_tool_tails=extend))
    assert response.statuses == [200], response.body
    return backend, response


def engine_tail(prompt, generated):
    """The session tail that the engine keeps after a job, or None. The engine rule: no tail when the job generated
    fewer than 2 tokens (engine/runtime/conversation_resume.h:83-90, ConversationTailState; batched_serve.h:1034-1038
    logs `session_replay skip`), else prompt + generated[:-1] (engine/serve/batched_serve.h:1044-1046)."""
    return list(prompt) + list(generated[:-1]) if len(generated) >= 2 else None


def session_tail(backend, ids):
    """The tail that the next turn resumes from: the extension job's tail when the engine keeps one, else the
    turn's own tail. The engine generates at most max_tokens of the job's script."""
    tail = engine_tail(backend.submissions[0][0], ids)
    if len(backend.submissions) > 1:
        prompt, request, options = backend.submissions[-1]
        assert not request.stream and options.get("log_turn") is False and request.temperature == 0
        tail = engine_tail(prompt, backend.extension_ids[:request.max_tokens]) or tail
    return tail


def submitted(adapter, body, path=CHAT):
    """The prompt IDs that a request prefills."""
    reply = adapter.tokenizer.encode(REPLY, add_special_tokens=False).ids
    return serve(adapter, body, reply, path, extend=False)[0].submissions[0][0]


def chat_body(messages, **options):
    return dict(messages=messages, tools=TOOLS, max_tokens=1 << 16, enable_thinking=True) | options


def resent(message, client="openai", keep_reasoning=True):
    """The assistant message as an OpenAI client sends it back, and its tool results."""
    calls = []
    for call in message.get("tool_calls") or []:
        arguments = call["function"]["arguments"]
        if client == "pi":  # JSON.stringify(JSON.parse(...)): compact and key order kept; 1.0 becomes 1
            parsed = json.loads(arguments, parse_float=lambda text: int(float(text)) if float(text).is_integer() and
                                abs(float(text)) < 1e21 else float(text))
            arguments = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        elif client == "python":
            arguments = json.dumps(json.loads(arguments))
        elif client == "indent":
            arguments = json.dumps(json.loads(arguments), indent=1)
        calls.append(dict(id=call["id"], type="function", function=dict(name=call["function"]["name"],
                                                                        arguments=arguments)))
    reply = dict(role="assistant", content=message.get("content"))
    if calls:
        reply["tool_calls"] = calls
    if keep_reasoning and message.get("reasoning_content"):
        reply["reasoning_content"] = message["reasoning_content"]
    return [reply] + [dict(role="tool", tool_call_id=call["id"], content="ok") for call in calls]


def shared(tail, following):
    """Tokens that the next prompt shares with the tail (len(tail) when it repeats all of it)."""
    return next((index for index, (left, right) in enumerate(zip(tail, following)) if left != right),
                min(len(tail), len(following)))


def returned(response, stream):
    if not stream:
        return json.loads(response.body)["choices"][0]["message"]
    parts = chunks(response)
    deltas = [call for part in parts for choice in part["choices"] for call in choice["delta"].get("tool_calls", ())]
    text = lambda key: "".join(choice["delta"].get(key) or "" for part in parts for choice in part["choices"])
    return dict(content=text("content") or None, reasoning_content=text("reasoning_content"),
                tool_calls=joined_calls(deltas))


def round_trip(adapter, calls, form=FORMS["tojson"], split=0.0, client="openai", stream=False, edit=None):
    """(the generated tail, the session tail, the next prompt IDs, the turn's backend, the history sent back)."""
    ids = generated(adapter, output_text(calls, form), split)
    backend, response = serve(adapter, chat_body(USER, stream=stream), ids)
    history = resent(returned(response, stream), client)
    if edit:
        edit(history[0])
    following = submitted(adapter, chat_body(USER + history))
    return backend.submissions[0][0] + ids, session_tail(backend, ids), following, backend, history


def write_call(lines=60, seed=5):
    return ("write", dict(path="src/app.py", content=source(random.Random(seed), lines)))


def edit_call(seed=6):
    rng = random.Random(seed)
    return ("edit", dict(path="src/app.py", edits=[dict(oldText=source(rng, 12), newText=source(rng, 20)),
                                                    dict(oldText="x = 1\n", newText="x = 2\n")]))


@pytest.mark.parametrize("form", list(FORMS))
def test_model_json_forms_give_a_full_shared_tail(root, form):
    """The model may write a JSON value compact, indented or with \\u escapes: the template writes tojson."""
    adapter = ModelTokenizer(root)
    _, tail, following, backend, _ = round_trip(adapter, [("bash", dict(command="ls")), edit_call()], FORMS[form])
    assert shared(tail, following) == len(tail)
    assert len(backend.submissions) == (1 if form == "tojson" else 2)  # no extension job when nothing differs


def test_a_long_write_with_a_token_split_gives_a_full_shared_tail(root):
    """The 900-line synthetic write with 1% of its call tokens split another way than the tokenizer splits them."""
    adapter = ModelTokenizer(root)
    # The synthetic tokenizer has 5 merges, so it needs a higher rate for a split to occur.
    split = 0.01 if root == Path(CHECKPOINT or "/") else 0.3
    generated_tail, tail, following, backend, _ = round_trip(adapter, [write_call(900, 7)], split=split)
    prompt = backend.submissions[0][0]
    assert shared(generated_tail, following) < len(generated_tail)  # trunk: the next prompt leaves the tail
    assert shared(tail, following) == len(tail) and tail[:len(prompt)] == prompt
    assert shared(generated_tail, tail) >= len(prompt)  # the extension job resumes inside the previous output


@pytest.mark.parametrize("client", ["openai", "pi", "python", "indent"])
def test_client_serializations_give_a_full_shared_tail(root, client):
    """Clients send the arguments back compact, re-serialized or indented; JavaScript sends 1.0 back as 1."""
    adapter = ModelTokenizer(root)
    calls = [("read", dict(path="src/app.py", offset=1.0, limit=200)), edit_call(), write_call()]
    _, tail, following, _, _ = round_trip(adapter, calls, FORMS["compact"], 0.02, client)
    assert shared(tail, following) == len(tail)


def returned_arguments(adapter, arguments, extend=True):
    _, response = serve(adapter, chat_body(USER), generated(adapter, output_text([("note", arguments)])), extend=extend)
    return json.loads(response.body)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]


def test_integral_floats_are_returned_as_integers(root):
    """So that a JavaScript client's JSON.parse and JSON.stringify give back the same arguments. Only up to 2**53 in
    magnitude: above it a JavaScript integer loses digits, and a client could put it in a fixed-width integer."""
    adapter = ModelTokenizer(root)
    arguments = dict(data=dict(a=1.0, b=-0.0, c=2.5, d=2.0 ** 53, g=-(2.0 ** 53), h=2.0 ** 53 + 2, i=3e20, e=1e21,
                               f=[1.0, 7]))
    assert returned_arguments(adapter, arguments) == ('{"data":{"a":1,"b":0,"c":2.5,"d":9007199254740992,'
                                                      '"g":-9007199254740992,"h":9007199254740994.0,"i":3e+20,'
                                                      '"e":1e+21,"f":[1,7]}}')


def test_with_the_switch_off_integral_floats_stay_as_on_trunk(root):
    """--extend-tool-tails 0 (and --session-replay 0, which turns it off: test_the_switch_follows_session_replay)
    gives the reply of trunk, streamed and not streamed: 1.0 stays 1.0."""
    adapter = ModelTokenizer(root)
    arguments = dict(data=dict(a=1.0, b=-0.0, f=[1.0, 7]))
    assert returned_arguments(adapter, arguments, extend=False) == '{"data":{"a":1.0,"b":-0.0,"f":[1.0,7]}}'
    _, response = serve(adapter, chat_body(USER, stream=True), generated(adapter, output_text([("note", arguments)])),
                        extend=False)
    assert returned(response, True)["tool_calls"][0]["function"]["arguments"] == '{"data":{"a":1.0,"b":-0.0,"f":[1.0,7]}}'


def test_a_streamed_call_gives_a_full_shared_tail(root):
    adapter = ModelTokenizer(root)
    _, tail, following, _, _ = round_trip(adapter, [edit_call(), write_call()], FORMS["indent"], 0.02, "pi",
                                          stream=True)
    assert shared(tail, following) == len(tail)


def test_parse_mode_gives_a_full_shared_tail(root):
    adapter = ModelTokenizer(root, content_special_tokens="parse")
    _, tail, following, _, _ = round_trip(adapter, [edit_call(), write_call()], FORMS["ascii"], 0.02)
    assert shared(tail, following) == len(tail)


def change_content(message):
    call = message["tool_calls"][-1]["function"]
    arguments = json.loads(call["arguments"])
    arguments["content"] = arguments["content"].replace("value", "valve", 1)
    call["arguments"] = json.dumps(arguments)


def change_type(message):
    call = message["tool_calls"][0]["function"]
    call["arguments"] = json.dumps(dict(json.loads(call["arguments"]), limit=True))  # was 1: equal in Python, not JSON


def add_key(message):
    call = message["tool_calls"][0]["function"]
    call["arguments"] = json.dumps(dict(json.loads(call["arguments"]), extra=True))


@pytest.mark.parametrize("edit", [change_content, change_type, add_key])
def test_a_changed_call_renders_the_client_text(root, edit):
    adapter = ModelTokenizer(root)
    calls = [write_call()] if edit is change_content else [("read", dict(path="src/app.py", limit=1))]
    _, tail, following, _, history = round_trip(adapter, calls, FORMS["compact"],
                                                0.05 if edit is change_content else 0, edit=edit)
    assert following == submitted(ModelTokenizer(root), chat_body(USER + history))
    assert shared(tail, following) < len(tail)


def test_the_prompt_is_a_function_of_the_request_only(root):
    """A restarted server and a server that served other sessions (the same call with another split, other calls)
    give the same prompt IDs as the server that served the turn."""
    rng = random.Random(13)
    calls = [write_call(40), edit_call()]
    serving, busy = ModelTokenizer(root), ModelTokenizer(root)
    ids = generated(serving, output_text(calls, FORMS["compact"]), 0.05)
    _, response = serve(serving, chat_body(USER), ids)
    follow = chat_body(USER + resent(returned(response, False), "pi"))
    serve(busy, chat_body([dict(role="system", content="Another agent.")] + USER[1:]),
          generated(busy, output_text(calls, FORMS["compact"]), 0.05, seed=9))
    for _ in range(10):
        serve(busy, chat_body(USER), generated(busy, output_text([random_call(rng)], random_form(rng)), 0.03))
    restarted = ModelTokenizer(root)
    prompts = [submitted(adapter, follow) for adapter in (serving, busy, restarted, serving)]
    assert all(prompt == prompts[0] for prompt in prompts)


def test_no_extension_without_the_switch_or_without_calls(root):
    adapter = ModelTokenizer(root)
    ids = generated(adapter, output_text([edit_call()], FORMS["compact"]))
    backend, _ = serve(adapter, chat_body(USER), ids, extend=False)
    assert len(backend.submissions) == 1
    answer = generated(adapter, REASONING + "\n</think>\n\nThe answer.\n<|im_end|>")
    backend, _ = serve(adapter, chat_body(USER), answer)
    assert len(backend.submissions) == 1  # only a turn with tool calls is extended


def test_the_messages_api_round_trip_gives_a_full_shared_tail(root):
    adapter = ModelTokenizer(root)
    calls = [edit_call(), write_call()]
    ids = generated(adapter, output_text(calls, FORMS["compact"]), 0.02)
    body = dict(model="crucible", max_tokens=1 << 16, tools=ANTHROPIC_TOOLS, thinking=dict(type="enabled", budget_tokens=4096),
                system="You are a coding agent.", messages=[dict(role="user", content="Change src/app.py.")])
    backend, response = serve(adapter, body, ids, MESSAGES)
    content = json.loads(response.body)["content"]
    assert [block["type"] for block in content] == ["thinking", "text", "tool_use", "tool_use"]
    results = [dict(type="tool_result", tool_use_id=block["id"], content="ok") for block in content
               if block["type"] == "tool_use"]
    follow = dict(body, messages=body["messages"] + [dict(role="assistant", content=content),
                                                     dict(role="user", content=results)])
    tail = session_tail(backend, ids)
    assert len(backend.submissions) == 2 and shared(tail, submitted(adapter, follow, MESSAGES)) == len(tail)


def test_the_responses_api_round_trip_gives_a_full_shared_tail(root):
    adapter = ModelTokenizer(root)
    calls = [edit_call(), write_call()]
    ids = generated(adapter, output_text(calls, FORMS["indent"]), 0.02)
    body = dict(model="crucible", max_output_tokens=1 << 16, tools=RESPONSES_TOOLS, reasoning=dict(effort="high"),
                instructions="You are a coding agent.", input=[dict(role="user", content="Change src/app.py.")])
    backend, response = serve(adapter, body, ids, RESPONSES)
    output = json.loads(response.body)["output"]
    assert [item["type"] for item in output] == ["reasoning", "message", "function_call", "function_call"]
    results = [dict(type="function_call_output", call_id=item["call_id"], output="ok") for item in output
               if item["type"] == "function_call"]
    follow = dict(body, input=body["input"] + output + results)
    tail = session_tail(backend, ids)
    assert len(backend.submissions) == 2 and shared(tail, submitted(adapter, follow, RESPONSES)) == len(tail)


ALPHABET = "abcXYZ019 _-.,:;{}[]\"'\\/\n\t\r\x01<>=é€中\U0001f642"


def random_string(rng, longest=40):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, longest)))


def random_json(rng, depth=0):
    kind = rng.randint(0, 7 if depth < 3 else 4)
    if kind == 0:
        return random_string(rng)
    if kind == 1:
        return rng.choice([0, 1, -7, 200, 1 << 40, 1 << 60])
    if kind == 2:
        return rng.choice([1.0, 2.5, -0.5, 1e-7, 2.0 ** 52, 0.1])  # 3e20: above 2**53, not covered (L1)
    if kind == 3:
        return rng.choice([True, False, None])
    if kind == 4:
        return random_string(rng, 6)
    if kind <= 6:
        return {random_string(rng, 8) or "k": random_json(rng, depth + 1) for _ in range(rng.randint(0, 4))}
    return [random_json(rng, depth + 1) for _ in range(rng.randint(0, 4))]


def random_form(rng):
    separators = rng.choice([(",", ":"), (", ", ": "), (" , ", " : "), (",", ": ")])
    indent = rng.choice([None, None, 1, 2, "\t"])
    ascii_only = rng.random() < 0.5
    return lambda value: json.dumps(value, ensure_ascii=ascii_only, separators=separators, indent=indent)


def random_call(rng):
    kind = rng.randint(0, 2)
    if kind == 0:
        return "note", dict(data=random_json(rng))
    if kind == 1:
        return "edit", dict(path=random_string(rng, 10) or "a.py",
                            edits=[dict(oldText=random_string(rng, 80), newText=random_string(rng, 80))
                                   for _ in range(rng.randint(1, 3))])
    return "write", dict(path="src/app.py", content=source(rng, rng.randint(1, 30)))


def paths(value, path=()):
    yield path
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, item in items:
        yield from paths(item, path + (key,))


def mutate(rng, value):
    """A JSON value that the client really changed (never equal to value under the replay key)."""
    for _ in range(50):
        changed = json.loads(json.dumps(value))
        path = rng.choice(list(paths(changed)))
        replacement = rng.choice([random_json(rng), True, "1", 0.5, None, [], {}])
        if not path:
            changed = replacement
        else:
            parent = changed
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = replacement
        if json.dumps(changed) != json.dumps(value):
            return changed
    raise AssertionError("no change found")


def property_cases(root):
    return 80 if root.name != Path(CHECKPOINT or "/").name else 16


def test_random_json_arguments_give_a_full_shared_tail(root):
    """Random values, model forms (spacing, indent, \\u escapes), token splits and client serializations."""
    adapter, rng = ModelTokenizer(root), random.Random(11)
    for case in range(property_cases(root)):
        calls = [random_call(rng) for _ in range(rng.randint(1, 3))]
        client = rng.choice(["openai", "pi", "python", "indent"])
        _, tail, following, _, _ = round_trip(adapter, calls, random_form(rng), rng.choice([0, 0.03]), client)
        assert shared(tail, following) == len(tail), (case, client)


def test_random_changed_arguments_render_the_client_text(root):
    """A call that the client changed gets the IDs of a new server, and the next prompt leaves the session tail."""
    adapter, fresh, rng = ModelTokenizer(root), ModelTokenizer(root), random.Random(12)
    for case in range(property_cases(root)):
        name, arguments = random_call(rng)
        key = rng.choice(list(arguments))
        value = mutate(rng, arguments[key])

        def edit(message):
            call = message["tool_calls"][0]["function"]
            call["arguments"] = json.dumps(dict(json.loads(call["arguments"]), **{key: value}))

        _, _, following, _, history = round_trip(adapter, [(name, arguments)], random_form(rng),
                                                 rng.choice([0, 0.03]), rng.choice(["openai", "pi"]), edit=edit)
        assert following == submitted(fresh, chat_body(USER + history)), case


def test_the_extension_skips_requests_that_it_cannot_help(root):
    """Images, structured output, bypass and refresh requests get no extension job (refresh reads no saved state,
    so the job would prefill the whole conversation cold); a refused job writes one line of counts."""
    from dataclasses import replace
    from types import SimpleNamespace
    from server import tool_tail
    from server.api import ApiError, validate_request
    adapter = ModelTokenizer(root)
    ids = generated(adapter, output_text([edit_call()], FORMS["compact"]))
    _, response = serve(adapter, chat_body(USER), ids, extend=False)
    message = json.loads(response.body)["choices"][0]["message"]
    request = validate_request(chat_body(USER), "chat", "crucible")
    prompt = adapter.encode(request)
    assert tool_tail.extension(adapter, request, prompt, ids, message) is not None
    assert request.cache_control == "auto"
    for changed in (dict(images=(object(),)), dict(constraint=dict(kind="json_object")), dict(cache_control="bypass"),
                    dict(cache_control="refresh")):
        assert tool_tail.extension(adapter, replace(request, **changed), prompt, ids, message) is None
    assert tool_tail.extension(adapter, request, prompt, ids, dict(message, tool_calls=None)) is None

    def refuse(*args, **kwargs):
        # The real refusal: server/batch_backend.py:293 (the queue is full; no code).
        raise ApiError("The batch request queue is full. Try again later.", status=503)

    lines = []
    service = SimpleNamespace(tokenizer=adapter, backend=SimpleNamespace(submit=refuse), tail_extensions=tool_tail.Pending())
    original, tool_tail.metrics.emit = tool_tail.metrics.emit, lines.append
    try:
        assert tool_tail.extend(service, request, prompt, ids, message) is None
    finally:
        tool_tail.metrics.emit = original
    fields = json.loads(lines[0].partition(" ")[2])
    assert lines[0].startswith("crucible_tail_extend ") and fields["outcome"] == "refused" and fields["status"] == 503
    assert not service.tail_extensions.items


@pytest.mark.parametrize("place", ["render", "submit"])
def test_a_fault_in_the_extension_writes_one_count_line(root, place):
    """After the response, any exception of the extension is caught: one line of counts, no traceback."""
    from types import SimpleNamespace
    from server import tool_tail
    from server.api import validate_request
    adapter = ModelTokenizer(root)
    ids = generated(adapter, output_text([edit_call()], FORMS["compact"]))
    _, response = serve(adapter, chat_body(USER), ids, extend=False)
    message = json.loads(response.body)["choices"][0]["message"]
    request = validate_request(chat_body(USER), "chat", "crucible")

    def fault(*args, **kwargs):
        raise RuntimeError("synthetic fault")

    tokenizer = SimpleNamespace(render_prompt=fault, encode_prompt=fault) if place == "render" else adapter
    service = SimpleNamespace(tokenizer=tokenizer, backend=SimpleNamespace(submit=fault), tail_extensions=tool_tail.Pending())
    lines = []
    original, tool_tail.metrics.emit = tool_tail.metrics.emit, lines.append
    try:
        assert tool_tail.extend(service, request, adapter.encode(request), ids, message) is None
    finally:
        tool_tail.metrics.emit = original
    fields = json.loads(lines[0].partition(" ")[2])
    assert fields["outcome"] == "fault" and fields["error"] == "RuntimeError" and not service.tail_extensions.items


def test_the_extension_job_is_a_shape_that_the_engine_keeps_as_a_tail(root):
    """max_tokens 1 leaves no tail (conversation_resume.h:85). The job is greedy with max_tokens 2: the engine keeps
    its prompt + the first generated token, the template's <|im_start|> of the next message."""
    adapter = ModelTokenizer(root)
    ids = generated(adapter, output_text([edit_call()], FORMS["indent"]))
    backend, response = serve(adapter, chat_body(USER, temperature=0.7, top_p=0.9, seed=4), ids)
    prompt, request, options = backend.submissions[1]
    assert request.max_tokens == 2 and request.temperature == 0 and not request.stop
    following = submitted(adapter, chat_body(USER + resent(returned(response, False))))
    tail = session_tail(backend, ids)
    assert tail == prompt + next_message_ids(adapter)[:1] and shared(tail, following) == len(tail)
    # A job whose first token is not the next message's start leaves only its slot points (no full tail).
    other, _ = serve(adapter, chat_body(USER), ids, extension_ids=[adapter.tokenizer.token_to_id("x") or 0, 5])
    assert shared(session_tail(other, ids), following) < len(session_tail(other, ids))


def test_the_drain_line_has_counts_only():
    import queue
    import threading
    from types import SimpleNamespace
    from server import tool_tail
    events = queue.Queue()
    # The engine's done event (engine/serve/batched_serve.h:43-58), as server/batch_backend.py delivers it.
    done = dict(event="done", request_id=7, slot_generation=1, prompt_tokens=100, completion_tokens=2,
                finish_reason="length", prefill_seconds=0.5, decode_seconds=0.01, cache_hit=True, cached_tokens=90,
                prefill_tokens=10, cache_bytes=0, restore_seconds=0.0, capture_seconds=0.0, slot_context=1 << 18,
                cache_class="session")
    for event in (dict(event="token", request_id=7, slot_generation=1, id=5), done):
        events.put(event)
    job, lines = SimpleNamespace(events=events, cancelled=threading.Event()), []
    original, tool_tail.metrics.emit = tool_tail.metrics.emit, lines.append
    try:
        tool_tail.drain(job, dict(prompt_tokens=100, shared_tokens=80, previous_tokens=95))
    finally:
        tool_tail.metrics.emit = original
    fields = json.loads(lines[0].partition(" ")[2])
    assert fields["outcome"] == "done" and fields["cached_tokens"] == 90 and job.cancelled.is_set()
    assert set(fields) == {"prompt_tokens", "shared_tokens", "previous_tokens", "outcome", "cached_tokens", "seconds"}


@pytest.mark.parametrize("argv, expected", [([], False), (["--conversation-cache-mib", "32"], True),
                                            (["--conversation-cache-mib", "32", "--extend-tool-tails", "0"], False),
                                            (["--conversation-cache-mib", "32", "--session-replay", "0"], False)])
def test_the_switch_follows_session_replay(argv, expected):
    from unittest.mock import patch
    import server.__main__ as startup
    from server.tests.test_serial_flag import run_main
    with patch.object(startup, "Service") as service:
        run_main(argv)
    assert service.call_args.kwargs["extend_tool_tails"] is expected


def test_the_switch_needs_session_replay():
    from server.tests.test_serial_flag import refused
    assert "--extend-tool-tails needs session replay" in refused(["--extend-tool-tails", "1"])


def test_a_next_request_waits_for_the_running_extension():
    """The next prompt that extends a running extension job waits for its end; another prompt does not wait."""
    import threading
    import time
    from server import tool_tail
    pending = tool_tail.Pending()
    entry = pending.add([1, 2, 3])
    assert pending.wait([1, 2, 4, 5], timeout=5) is None
    threading.Timer(0.2, pending.finish, args=(entry,)).start()
    start = time.monotonic()
    seconds, done = pending.wait([1, 2, 3, 9], timeout=5)
    assert done and 0.1 < time.monotonic() - start < 4 and not pending.items
    late = pending.add([7])
    assert pending.wait([7, 8], timeout=0.05)[1] is False
    pending.finish(late)


def test_an_extension_with_no_end_writes_expired():
    import queue
    import threading
    from types import SimpleNamespace
    from server import tool_tail
    job, lines = SimpleNamespace(events=queue.Queue(), cancelled=threading.Event()), []
    original, tool_tail.metrics.emit = tool_tail.metrics.emit, lines.append
    seconds, tool_tail.EXTEND_SECONDS = tool_tail.EXTEND_SECONDS, 0.05
    try:
        tool_tail.drain(job, dict(prompt_tokens=100, shared_tokens=80, previous_tokens=95))
    finally:
        tool_tail.metrics.emit, tool_tail.EXTEND_SECONDS = original, seconds
    assert json.loads(lines[0].partition(" ")[2])["outcome"] == "expired" and job.cancelled.is_set()


def test_the_wait_runs_idle_between_polls():
    """The handler's idle() sends keepalives and sees a client that left; an exception from it ends the wait."""
    from server import tool_tail
    pending, calls = tool_tail.Pending(), []
    entry = pending.add([1, 2])

    def idle():
        calls.append(1)
        if len(calls) == 3:
            raise ConnectionResetError("The client closed the connection.")

    with pytest.raises(ConnectionResetError):
        pending.wait([1, 2, 3], timeout=30, idle=idle, poll=0.01)
    assert len(calls) == 3
    assert pending.wait([1, 2, 3], timeout=0.05, idle=lambda: None, poll=0.01)[1] is False
    pending.finish(entry)


class TimedWriter:
    """The stream writer of a handler test: the time of each flush, and the bytes written before it."""
    def __init__(self):
        import io
        import time
        self.buffer, self.flushes, self.clock, self.start = io.BytesIO(), [], time.monotonic, time.monotonic()

    def write(self, data):
        return self.buffer.write(data)

    def flush(self):
        self.flushes.append((self.clock() - self.start, self.buffer.getvalue()))

    def getvalue(self):
        return self.buffer.getvalue()


def running_extension(monkeypatch, adapter, body, finish_after, ended=None):
    """A Service whose Pending holds a running extension of the next prompt, which ends after finish_after s. The
    monotonic time of the end goes into the list `ended` just before the extension ends."""
    import threading
    import time
    from server import http, tool_tail
    pending = tool_tail.Pending()
    entry = pending.add(submitted(adapter, body)[:-8])
    monkeypatch.setattr(tool_tail, "Pending", lambda: pending)
    monkeypatch.setattr(http, "KEEPALIVE_SECONDS", 0.2)

    def finish():
        if ended is not None:
            ended.append(time.monotonic())
        pending.finish(entry)
    if finish_after is not None:
        timer = threading.Timer(finish_after, finish)
        timer.start()
    return pending, entry


def test_a_client_that_leaves_during_the_wait_is_seen(root, monkeypatch):
    import socket
    import time
    adapter = ModelTokenizer(root)
    body = chat_body(USER, stream=True)
    pending, entry = running_extension(monkeypatch, adapter, body, None)
    near, far = socket.socketpair()
    far.close()
    backend = SequenceBackend()
    start = time.monotonic()
    try:
        invoke(backend, adapter, body, writer=TimedWriter(), connection=near,
               service_options=dict(extend_tool_tails=True))
    finally:
        near.close()
        pending.finish(entry)
    assert time.monotonic() - start < 5 and not backend.submissions


def test_with_the_switch_off_a_request_does_not_wait(root, monkeypatch):
    import time
    adapter = ModelTokenizer(root)
    body = chat_body(USER, stream=True)
    pending, entry = running_extension(monkeypatch, adapter, body, None)
    reply = adapter.tokenizer.encode(REPLY, add_special_tokens=False).ids
    start = time.monotonic()
    try:
        response = invoke(SequenceBackend((reply, "stop")), adapter, body, service_options=dict(extend_tool_tails=False))
    finally:
        pending.finish(entry)
    assert response.statuses == [200] and time.monotonic() - start < 2
