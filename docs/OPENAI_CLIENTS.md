# Connect an OpenAI-compatible or Anthropic-compatible agent

Crucible supports text Chat Completions and function calls through a local OpenAI-compatible endpoint.
It also serves the OpenAI Responses API (`POST /v1/responses`) and the Anthropic Messages API (`POST /v1/messages`)
through the same chat code inside the server (see "OpenAI Responses API" and "Anthropic Messages API" below).
The official OpenAI SDK, the official Anthropic SDK and the Pi AI transport pass CPU compatibility tests.
These tests use recorded token scripts, not model quality or GPU performance tests.

## Connection settings

- Base URL: `http://127.0.0.1:8790/v1`
- Model: `crucible`, unless the server uses another `--model-id` name.
- API key: any dummy value, such as `unused`.
- API: Chat Completions, or the Responses API with no server-side state.
- Input: text. Images need a server started with `--vision` (base64 `data:` URLs only).

- Token count: `POST /v1/messages/count_tokens` takes an Anthropic Messages body and returns `{"input_tokens": N}`,
  the prompt size that Chat Completions prefills for the same conversation. It does no generation. The field
  mapping (system, text, thinking, `tool_use`, `tool_result`, tools, `tool_choice`, thinking and effort) is in
  `server/README.md`. It never applies the thinking skip, so its count can differ from what a served request
  prefills.
  Generation with an Anthropic body uses `POST /v1/messages` (see "Anthropic Messages API").

- Tokenize and template: `POST /tokenize` (also `/v1/tokenize`) and `POST /apply-template` use the OpenAI-compatible
  shapes. They return the prompt IDs or the prompt text of a request and do no generation. See `server/README.md`.

- Readiness: `GET /health` returns `{"status":"ok","sessions":N,"context":C}` when the engine is ready (the real slot
  count and `--ctx`), and HTTP 503 after it stops.

The server needs no key on loopback or a private LAN address. An unkeyed LAN start prints a warning.
Use `./crucible setup --api-key` to require a key, or `--no-api-key` to stop using it on a private address.
With `--api-key-file` it needs the key on every route but `/health`:
set the key as the API key of the client (see `server/README.md`, API key).
Keep it on loopback or behind a reverse proxy (see `docs/REMOTE_ACCESS.md`). Do not expose it to an untrusted network.
See `server/README.md` for server startup.

If the agent runs on another machine, use a foreground SSH tunnel:

```bash
ssh -N -L 8790:127.0.0.1:8790 <gpu-host>
```

Keep that terminal open while the agent uses the server.
Stop the tunnel with Ctrl-C after the test.

## Official OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8790/v1",
    api_key="unused",
)
response = client.chat.completions.create(
    model="crucible",
    messages=[{"role": "user", "content": "Write a short Fibonacci function."}],
    max_completion_tokens=256,
    temperature=0,
)
print(response.choices[0].message.content)
```

For streaming, add `stream=True` and `stream_options={"include_usage": True}`.
Read `choice.delta.content` for text and `choice.delta.tool_calls` for function calls.
The final usage chunk has no choices.
Chunks before it have `usage: null` when usage streaming is requested.

For functions, supply OpenAI `tools` definitions with JSON object parameters.
Use `tool_choice="auto"`, `"none"`, `"required"`, or a named function object.
A response can contain multiple calls when `parallel_tool_calls` is true.
Execute each function in the client, then send its result with the matching `tool_call_id`.
The server does not execute functions.

The server streams each call while the model writes it (`--stream-tool-calls on`, the default).
The call ID and name come first. An argument with a plain string schema streams as text.
Other arguments come when they pass the supported schema checks. The closing `}` comes after the whole call passes.
If a streamed call then fails a check, the stream ends with finish reason `length`.
The client then does not run the partial call. Pi sends an error result and the model writes the call again.
With `--stream-tool-calls off`, each call comes complete at the end of the output.

## Speed numbers (`timings`)

Each chat and completion response has a `timings` object with the OpenAI-compatible field names and units.
In a stream, the last chunk has it: the usage chunk when `include_usage` is true, else the chunk with the finish reason.
Benchmark tools read the prompt and decode speeds from it.

- `cache_n`: prompt tokens that came from the cache. `prompt_n`: the other prompt tokens.
- `prompt_ms`: the server time from the request submit to the first generated token. It includes a queue wait.
- `predicted_n`: the generated tokens (the same value as `usage.completion_tokens`).
- `predicted_ms`: the time from the first generated token to the end of the reply.
- `prompt_per_token_ms`, `prompt_per_second`, `predicted_per_token_ms`, `predicted_per_second`: the rates of
  these counts and times. A rate is `0` when its count or its time is `0`.

The field is always on. A client that does not know it ignores it.

## Token limits and stop sequences

Use `max_tokens` or `max_completion_tokens`, but not both.
A request that sends neither field takes the server default (see `server/README.md`, "Server defaults").
Send one when you need a specific limit.
The prompt and requested output must fit the server context.
The finish reason is `stop`, `length`, or `tool_calls`.

`stop` accepts one string or a list of at most four strings.
Each string must contain 1 to 4096 UTF-8 bytes.
An omitted stop, `null`, or an empty list disables text stops.
The returned text excludes the matched sequence and any later text in that token.
Matching works across token boundaries and streaming chunks.

The API ends the client response when a text stop completes.
The resident worker drains the canceled generation before it starts the next request.
This preserves the native protocol without changing engine arithmetic.
Usage counts complete accepted tokens through the token that contains the text stop.
Diagnostic token IDs can therefore contain text that the visible response excludes.

The server returns OpenAI error objects with `message`, `type`, `param`, and `code`.
Unsupported request features return HTTP 400 before queue submission.
An unknown model returns HTTP 404.
The default request body limit follows `--ctx`: 16 bytes per token, at least 1 MiB, or 4 MiB at 262,144 tokens.
With `--vision`, the server default is 256 MiB instead.
The launcher `./crucible` passes `--max-body-mib 16`, so its limit is 16 MiB.
A request body above the limit returns HTTP 413.
A client address above `--max-requests-per-client` returns HTTP 429. This limit is off by default.
A full or failed engine queue returns HTTP 503.
Invalid generated function calls come back as content with `usage.tool_call_fallback` (see server/README.md, "Tool calls").
Multiple output choices are not supported.
A JSON Schema keyword that the grammar cannot enforce returns HTTP 400 `constraint_keyword` with its JSON pointer.
The grammar is stricter than JSON Schema in six named cases.
A string `format` grammar took no value that python-jsonschema 4.26.0 refuses, in all tests and probes. Some take only
a part of the valid values, for example a lowercase `z` in `time`.
A schema with about 6 or more hostname fields can exceed the 5 s grammar compile limit (HTTP 400).
A `pattern` grammar matches the whole string. In a differential test of 10,000 random patterns, it took exactly the
ASCII strings that Python's `re.fullmatch` takes. The engine reads `\d`, `\w` and `\s` as ECMAScript does: `\d` and `\w`
are ASCII only.
Outside ASCII, `\D`, `\W` and `\S` take characters that Python's `re` refuses (for example `\W` takes `é`), and
`(?i)` folds only ASCII letters.
A `pattern` part that the engine cannot read as written returns HTTP 400 `constraint_keyword`: a lookahead or
lookbehind, an atomic group, a group flag other than a leading `(?i)`, a quantifier after a quantifier (for example
`a++`), a `{` that is not a repetition such as `{2,5}`, and an anchor `^` or `$` that is not at an end of the pattern.
An anchor `^` or `$` inside a group is served when no group around it has a quantifier, and only anchors of its kind
come before `^` or after `$`, for example `(^\d{5}$)|(^\d{5}-\d{4}$)`.
A schema that no value can match returns HTTP 400 `constraint_impossible`.
A schema node that has no value, in a schema that has other valid documents, returns HTTP 400 `constraint_empty_branch`.
A number range whose values all have more than 6 decimal places returns HTTP 400 `constraint_range`.
An integer `const` or `enum` value outside the 64-bit range returns HTTP 400 `constraint_keyword`.
A strict tool argument can hold `</tool_call>`: the server reads a strict call by its structure.
A grammar with no legal next token ends only that request with HTTP 503 (a stream gets an error event).
Images need `--vision`. Without it, an image part returns HTTP 400.

## Accepted and refused request fields

The sampling fields are `temperature`, `top_k`, `top_p`, `min_p`, and `seed`.
The thinking fields are `enable_thinking`, `chat_template_kwargs`, and `reasoning_effort`.
`chat_template_kwargs` takes `enable_thinking`, `preserve_thinking`, and `reasoning_effort` only.
`stream_options` takes `include_usage` only.
The thinking budget field is `max_thinking_tokens`.
`thinking_budget_tokens`, `thinking_budget`, and `thinking_token_budget` are aliases.

`server/README.md` "Supported request options" gives the accepted values and the thinking rules for these fields.
Its "Server defaults" subsection gives the flags that set a default.

The server refuses every unknown request field outside the allow-list of "Lenient request fields" with HTTP 400.
The error names that field. In the strict mode (`--lenient-fields off`), the server refuses every unknown field.
An unknown key inside `chat_template_kwargs` or `stream_options` also returns HTTP 400.
The error names the object and the keys that the object accepts.
The server refuses a field that it cannot apply instead of ignoring it, because the field could change the prompt.
It ignores only the allow-listed fields, which cannot change the tokens.
Other fields that another OpenAI-compatible server accepts can fail here.
The server applies `presence_penalty` and `frequency_penalty` (from `-2` to `2`) and `repetition_penalty`
(from `0.01` to `2`). Null is the default value. A value outside the range returns HTTP 400.
`server/README.md` gives the formula. With `response_format`, the penalties change the logits first and the grammar
mask applies after them, so a penalty never lets a token through that the grammar refuses.
`response_format` takes `text`, `json_object` and `json_schema` (chat completions only). `/v1/responses` and
`/v1/messages` use their own fields for it (see "Structured output on the adapters").
The server ignores `user` at any value (see "Lenient request fields"). With `--lenient-fields off`, the server refuses
`user`, so a client or a proxy must remove it before it sends the request.

### Lenient request fields (`--lenient-fields`, on by default)

Some clients send fields of other servers in every request (for example `user`, `cache_prompt` or `repeat_penalty`).
The server accepts these requests by default: the mode is on.
Start the server with `--lenient-fields off`, or set `CRUCIBLE_LENIENT_FIELDS=0`, to use the strict mode.
In the strict mode, every unknown field returns HTTP 400. The flag wins over the environment variable.
An empty `CRUCIBLE_LENIENT_FIELDS` is the same as no variable: the mode is on.
`./crucible start` has no option for the mode.
Without the user service, run `CRUCIBLE_LENIENT_FIELDS=0 ./crucible start`.
To use the strict mode with the launcher user service, add a systemd drop-in.
Run `systemctl --user edit crucible-server.service` and add these lines:

```ini
[Service]
Environment=CRUCIBLE_LENIENT_FIELDS=0
```

Then run `systemctl --user restart crucible-server.service`.

With the mode on, the server looks only at the top-level fields that it does not know. The mode is an allow-list:
it ignores only the fields below, which cannot change the tokens.

- Ignored at any value, because each one is an account, cache or routing hint:
  - `user` and `safety_identifier`: an end-user ID for the abuse checks of a hosted API.
  - `metadata`: the tags of a stored completion. Crucible stores no completion.
  - `service_tier`: the latency and price tier of a hosted API.
  - `prompt_cache_key` and `cache_prompt`: prompt-cache hints. Crucible finds a cached prefix by itself.
- Accepted only at the neutral value or `null`, because Crucible cannot apply another value:
  `repeat_penalty` 1, `typical_p` 1, `tfs_z` 1, `mirostat` 0, `dynatemp_range` 0, `xtc_probability` 0,
  `dry_multiplier` 0, `top_n_sigma` -1, `n_probs` 0, `best_of` 1, `logit_bias` `{}`, `ignore_eos` false, `echo`
  false, `store` false, `suffix` `""`, `modalities` `["text"]`. Another value returns HTTP 400 with the field name.
  The server never changes sampling silently.
- Accepted as a number or `null`, because each one is a setting of a sampler above that is off:
  `repeat_last_n` (`repeat_penalty`), `mirostat_tau` and `mirostat_eta` (`mirostat`), `dynatemp_exponent`
  (`dynatemp_range`), `xtc_threshold` (`xtc_probability`), `dry_base`, `dry_allowed_length` and
  `dry_penalty_last_n` (`dry_multiplier`).
- `repeat_last_n` must be `null` when the request sets `presence_penalty`, `frequency_penalty` or
  `repetition_penalty` to a value that is not neutral. Another value, also `0` or `-1`, returns HTTP 400 with the
  field name. In OpenAI-compatible, `repeat_last_n` is the window of all its penalties, and that window holds prompt tokens too.
  Crucible applies the penalties with no window, so it cannot apply a window that the client asks for.

Any other unknown field returns HTTP 400 with the field name, as in the strict mode. For example, the output controls
`allowed_token_ids`, `stop_token_ids`, `min_tokens`, `guided_regex`, `guided_json`, `grammar` and `json_schema`
return HTTP 400. `response_format` is a known field, so the mode does not change its checks.
A request never gets a reply with a constraint that the server did not apply.

Known fields keep all their checks. Nested objects (messages, `stream_options`, `chat_template_kwargs`) stay strict.
The ignored fields do not change the prompt or the sampling: the request is the same as without them.
The server log gets one `crucible_lenient` line for each request that had ignored fields. The line holds the field
names, never their values.
The startup log says when the mode is on.
The thinking on/off field has one accepted name: `enable_thinking`.
A different name, such as `think`, returns HTTP 400.

A history message takes `role`, `content`, and `reasoning_content`.
An assistant message also takes `tool_calls`. A tool message also takes `tool_call_id` and `name`.
`reasoning` is an alias of `reasoning_content`, because LiteLLM, Hermes, and other engines name the field `reasoning`.
Send one of the two, or the same text in both: the prompt is the same as for `reasoning_content` alone.
An empty string or `null` in either field counts as absent, as proxies send `null` for unset fields.
Two different non-empty texts return HTTP 400, because the server cannot know which text the client means.
A value that is not a string, `null`, or empty also returns HTTP 400.
A message field outside this list returns HTTP 400 with the message index, the role, and the field names, for example:
`messages[2] (assistant) has unsupported fields: foo. This role accepts: content, reasoning, reasoning_content, role, tool_calls.`

The server can drop thinking from a chat request that asks for it, so do not assume that `reasoning_content` is
present.
It does that when thinking is on, no thinking budget is set, and the room left for thinking is small.
Detect a drop with `usage.completion_tokens_details.reasoning_skipped_by`, which is `min_thinking_tokens`.
`server/README.md`, `--min-thinking-tokens`, gives the threshold, the exact rule, and the ways to turn it off.

## OpenAI Responses API

`POST /v1/responses` takes a Responses body. The server maps it to a chat request, so the prompt, the sampling and
the tokens are the same as for the equal `/v1/chat/completions` request. A CPU test checks that the prompt bytes
and the engine request are equal. The server keeps no state: send the whole conversation in `input` each time.

- `input`: text, or a list of items. A `message` item has the role `user`, `assistant`, `system` or `developer`, with
  text or a list of `input_text` (user, system), `input_image` (user, needs `--vision`) or `output_text` (assistant)
  parts. A `reasoning` item gives its `reasoning_text` content to the assistant `message` or `function_call` after
  it (`summary` and `encrypted_content` do not reach the prompt). When a `user`, `system` or `developer` item or the
  end of the input comes first, the reasoning is its own assistant turn with empty text, as a reasoning-only chat
  reply is. So the `output` of an `incomplete` response goes back into `input` unchanged. A `function_call` item
  joins the assistant turn before it, unless a reasoning item comes between them. A `function_call_output` item is
  a tool result. A reasoning item directly before a `function_call_output` returns HTTP 400.
- `instructions` and the `system` and `developer` messages before the first other item become one system message,
  joined with a blank line. A `system` or `developer` message after it returns HTTP 400.
- `max_output_tokens`, `temperature` and `top_p`. A field that the request leaves out gets the server default.
  `top_k`, `min_p` and `seed` are Crucible extensions.
- `tools` (function tools only), `tool_choice` (`auto`, `none`, `required` or a function by name) and
  `parallel_tool_calls`. A function tool with `strict: true` gets the strict call grammar of Chat Completions.
  A function tool with `strict: false` gets no grammar.
  A function tool without `strict` (or with `null`) gets the strict grammar when the strict checks accept its schema.
  If they do not accept it, the tool gets no grammar, and the reply's `tools` show `strict: false`. This is the
  OpenAI rule in <https://platform.openai.com/docs/guides/function-calling> ("Strict mode"): "If you omit
  `strict`, the default depends on the API: Responses requests will attempt to normalize your schema into strict mode
  when possible, and will fall back to non-strict, best-effort function calling if the schema cannot be made
  compatible with strict mode." On an engine with no structured output, the tool falls back too.
  Chat Completions keep no grammar for a tool without `strict`. The engine compiles
  the grammar after the server checks: a compile refusal stays HTTP 400 and does not fall back. The prompt never
  holds a `strict` flag that the server set (`server/responses_strict.py`).
- `reasoning.effort` uses the `reasoning_effort` levels of Chat Completions (`none` turns thinking off).
  `reasoning.summary` is accepted and has no effect: the reply gives the reasoning text itself.
- `text.format` type `text`, `json_schema` or `json_object`. A JSON format gets the grammar of the chat field
  `response_format` (see "Structured output on the adapters"). `include` accepts `reasoning.encrypted_content` only,
  and the reply has no encrypted content.
- `text.verbosity`: `medium` or absent only. `low` and `high` return HTTP 400, because they ask for a shorter or a
  longer answer, and the server cannot make that change. The official OpenAI SDK leaves out `text` by default, and
  the API default verbosity is `medium`.
- Ignored hints: `metadata`, `user`, `safety_identifier`, `prompt_cache_key`, `prompt_cache_retention`,
  `service_tier` and `stream_options`.
- Refused with HTTP 400 that names the field: `previous_response_id`, `conversation`, `prompt`, `store: true` and
  `background: true` (no conversation store), `truncation: auto`, `top_logprobs`, `max_tool_calls`, `response_format`
  (use `text.format`), tools that are not functions, items other than the four above (for example `item_reference`),
  and any other field.
- An error about a part of the input names its place and the supported values, for example
  `input[0].type must be message, reasoning, function_call, or function_call_output.` The `param` field holds the
  same place. The error never repeats the value that the client sent.

The reply has `output` items in this order: `reasoning` (with `reasoning_text` content and an empty `summary`),
`message` (`output_text`) and `function_call` items. `status` is `completed`, or `incomplete` with
`incomplete_details.reason` `max_output_tokens` when the token limit ended the reply. `usage.input_tokens` includes
the cached tokens (`input_tokens_details.cached_tokens`).

A stream (`"stream": true`) sends `response.created` and `response.in_progress`. For each output item it sends
`response.output_item.added`, the content events (`response.content_part.added`, `response.reasoning_text.delta`
or `response.output_text.delta`, their `.done` events and `response.content_part.done`; or
`response.function_call_arguments.delta` and `.done`) and `response.output_item.done`. The stream ends with
`response.completed` or `response.incomplete`, which hold the whole response and its usage. Each event has a
`sequence_number`. A failure after the start ends the stream with an `error` event. Errors before the start use the
OpenAI error object. Request size limits, the API key and the per-client limit apply as on Chat Completions.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8790/v1", api_key="unused")
response = client.responses.create(
    model="crucible",
    instructions="Answer briefly.",
    input="Write a short Fibonacci function.",
    max_output_tokens=256,
)
print(response.output_text)
```

## Structured output on the adapters

`/v1/responses` and `/v1/messages` map their answer format field to the chat field `response_format`
(`server/responses_api.py` `text_format`, `server/anthropic.py` `output_format`). So the request gets the same
grammar, the same refused JSON Schema keywords and the same limits as the equal chat request. A CPU test checks that
the engine constraint and the prompt are equal (`server/tests/test_structured_formats.py`).

| API field | Value | Chat field `response_format` |
|---|---|---|
| Responses `text.format` | absent, `null` or `{"type":"text"}` | absent (no grammar) |
| Responses `text.format` | `{"type":"json_object"}` | `{"type":"json_object"}` |
| Responses `text.format` | `{"type":"json_schema", "name", "schema", "description", "strict"}` | `{"type":"json_schema", "json_schema": {"name", "schema", "description", "strict"}}` |
| Anthropic `output_config.format` | absent or `null` | absent (no grammar) |
| Anthropic `output_config.format` | `{"type":"json_schema", "schema"}` | `{"type":"json_schema", "json_schema": {"schema"}}` |

- The server enforces the schema at every value of `strict`: `true`, `false`, `null` or absent. The OpenAI default is
  `false`. The constant `STRICT_DEFAULT` in `server/structured.py` holds the value that a missing `strict` takes
  (`true`). A Responses reply echoes the format in `text.format`, with `strict` at this value.
- A null `description` or `strict` in `text.format` counts as absent.
- The answer is the JSON text: on `/v1/responses` the `output_text` of the `message` item (and `output_text` in the
  SDK), on `/v1/messages` one `text` block. A stream sends it as `response.output_text.delta` or `text_delta` events.
- With thinking on, the reasoning comes first, then the JSON.
- With tools, the model writes the JSON answer or tool calls, as on Chat Completions with `response_format`.
- An error keeps the API's own shape. On `/v1/responses`, `param` is `text.format`. The message names the place and
  the supported values, never the client's value. A refused schema gets the same message as on Chat Completions.
- An engine without structured output returns HTTP 400 for a JSON format.
- `POST /v1/messages/count_tokens` accepts `output_config.format`. The format adds no prompt text, so the count does
  not change.
- `response_format` on `/v1/responses` returns HTTP 400 (the Responses field is `text.format`).

## Anthropic Messages API

`POST /v1/messages` takes an Anthropic Messages body. The server maps it to a chat request, so the prompt, the
sampling and the tokens are the same as for the equal `/v1/chat/completions` request. A CPU test checks that the
prompt bytes and the engine request are equal. The message, tool and thinking mapping is the table of
`server/README.md`, "Token count". A query string such as `?beta=true` is accepted.

- Base URL: `http://127.0.0.1:8790` (the SDK adds `/v1/messages`).
- API key: any dummy value. With `--api-key-file`, send the key as `x-api-key` or as a bearer token.
- `max_tokens` is required, as in the Anthropic API.
- Sampling: `temperature`, `top_p` and `top_k`. A field that the request leaves out gets the server default, as on
  Chat Completions (the server default temperature is 0, not the Anthropic default of 1). `seed` is a Crucible
  extension for repeatable sampling.
- `stop_sequences`: at most four strings. A match gives `stop_reason` `stop_sequence` and the matched string.
- `thinking`: `enabled` turns thinking on and `budget_tokens` is the thinking cap (`max_thinking_tokens`). `disabled`
  turns it off. `adaptive` turns it on with no cap. Without the field, the server default applies.
- Tools: custom tools with `input_schema`, `tool_choice`, `tool_use` and `tool_result` blocks. A tool with
  `strict: true` gets the strict call grammar of Chat Completions. A tool without `strict`, or with `strict: false`
  or `null`, gets no grammar for its arguments, as on Chat Completions. On an engine with no structured output, a
  strict tool runs with no grammar, as on Chat Completions. The prompt never holds the `strict` flag.
- `output_config.format` with `type: "json_schema"` and a `schema`: the answer is one `text` block that holds the
  JSON. It gets the grammar of the chat field `response_format` (see "Structured output on the adapters").
  `output_config.effort` sets the thinking level. The deprecated top-level `output_format` returns HTTP 400.
- Ignored hints: `metadata`, `service_tier`, `cache_control`, `inference_geo`, `speed`, `user_profile_id` and
  `workspace_id`.
- Refused with HTTP 400 that names the field: any other top-level field (for example `container`, `mcp_servers`
  or `context_management`), server tools, and content blocks other than text, image, `tool_use`, `tool_result`,
  thinking and `redacted_thinking` (for example `document` or `search_result`). The message names the block place,
  for example `messages[0].content[0].type`, and the supported types, never the type that the client sent.
  An image needs a server with `--vision`.
- `diagnostics` returns HTTP 400. It asks for prompt-cache diagnostic data (`diagnostics.previous_message_id`), and
  the server has none to give. The official Anthropic SDK leaves out `diagnostics` by default.
- An assistant turn with only a `thinking` block (for example a reply that `max_tokens` ended) goes back unchanged.
  The prompt is the same as for the chat history with the reasoning and an empty `content`.

The reply has a `thinking` block (the model's reasoning, with an empty `signature`), a `text` block and `tool_use`
blocks, in that order. Send the blocks back unchanged in the next request. `stop_reason` is `end_turn`,
`max_tokens`, `stop_sequence` or `tool_use`. `usage.input_tokens` leaves out the cached prompt tokens, which are
in `usage.cache_read_input_tokens`. `cache_creation_input_tokens` is always 0.

A stream (`"stream": true`) sends `message_start`, `ping`, then for each block `content_block_start`,
`content_block_delta` (`thinking_delta`, `text_delta` or `input_json_delta`) and `content_block_stop`, then
`message_delta` (stop reason and usage) and `message_stop`. A failure after the start ends the stream with an
`error` event. Errors use the Anthropic shape `{"type":"error","error":{"type":...,"message":...}}`.
Request size limits, the API key and the per-client limit apply as on Chat Completions.

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8790", api_key="unused")
message = client.messages.create(
    model="crucible",
    max_tokens=256,
    messages=[{"role": "user", "content": "Write a short Fibonacci function."}],
)
print(message.content[-1].text)
```

Newer Anthropic SDKs have no sampling arguments. Send them with `extra_body={"temperature": 0}`.

## Pi coding agent

Pi can use this endpoint through `models.json`; no provider extension is needed.
Keep your existing providers and add this provider entry:

```json
{
  "providers": {
    "crucible": {
      "baseUrl": "http://127.0.0.1:8790/v1",
      "api": "openai-completions",
      "apiKey": "unused",
      "models": [
        {
          "id": "crucible",
          "name": "Crucible",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 32768,
          "maxTokens": 8192,
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
          "compat": {
            "supportsStore": false,
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false,
            "supportsLongCacheRetention": false,
            "supportsStrictMode": false,
            "supportsUsageInStreaming": true,
            "supportsFinishReason": true,
            "maxTokensField": "max_completion_tokens"
          }
        }
      ]
    }
  }
}
```

Set `contextWindow` to the server's actual `--ctx` value or less.
Pi compacts the history when a request gets the HTTP 400 text "exceeds the context window".
For a server with `--vision`, set `"input": ["text", "image"]`.
Pi then sends images, including images from tool results, as `image_url` parts with base64 `data:` URLs.
This profile keeps reasoning off.
The flags prevent unsupported storage, cache-retention, developer-role, and reasoning fields.
They keep supported usage and finish-reason processing active.
The dummy key makes the model available in Pi; it is not an authentication secret.

For thinking, make these changes to the model entry:

1. Set `"reasoning": true`.
2. Add `"thinkingLevelMap": {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}`.
3. In `compat`, set `"supportsReasoningEffort": true`.
4. In `compat`, add `"thinkingFormat": "qwen"`.

Pi then sends `reasoning_effort`. The server returns the reasoning in `reasoning_content`.
Pi stores that reasoning and sends it back in later turns, as the chat template expects.
`server/tests/pi_thinking_client.mjs` tests this profile.

The server keeps conversation state only with `--conversation-cache-mib`.
Pi sends the complete history in each turn. A later turn is fast only when its early tokens do not change.
Keep the system prompt and the tool list the same during a session. See `docs/SESSION_REPLAY.md`.

Open `/model` to reload the model file, then select `crucible/crucible`.
Alternatively, start a session with `pi --model crucible/crucible --thinking off`.
The CPU check runs only the HTTP transport of Pi against a fixture. No agent tool runs.

References: the Pi documentation pages `models.md`, `custom-provider.md`, `configuration.md` and `cli.md`.
The Pi source file `pi-ai/dist/api/openai-completions.js` shows the request flags and the stream handling.

## CPU compatibility checks

Install the test SDK in a private environment:

```bash
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r server/tests/requirements-sdk.txt Jinja2 tokenizers
.venv-sdk/bin/python -m unittest server.tests.test_openai_sdk server.tests.test_openai_backend server.tests.test_stop_sequences
```

The HTTP tests use loopback port 8790.
Run only one such suite at a time.
The optional Pi transport check needs Node and `CRUCIBLE_PI_AI_DIR`:

```bash
CRUCIBLE_PI_AI_DIR=/path/to/pi-ai \
  .venv-sdk/bin/python -m unittest server.tests.test_openai_sdk
```

The Pi check also runs Pi's agent loop. It needs the `pi-agent-core` package beside `pi-ai`.
It checks that Pi does not run a streamed call that the server judges invalid, and that the loop continues.
Add `server.tests.test_thinking` to the command to check the thinking profile.

The tests cover text, streams, multiple calls, tool policies, tool results, token limits, stops, usage, models, errors, and cancellation.
They also check Unicode, repeated stop prefixes, and cross-request native protocol draining.

To run the same SDK tests against a running server, set `CRUCIBLE_LIVE_BASE_URL` to its root URL (for example
`http://127.0.0.1:8790`). The tests that need the CPU fake backend then skip, and `server/tests/test_sdk_live.py` runs
its live cases.
