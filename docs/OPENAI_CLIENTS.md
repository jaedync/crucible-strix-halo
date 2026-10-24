# Connect an OpenAI-compatible agent

Crucible supports text Chat Completions and function calls through a local OpenAI-compatible endpoint.
The official OpenAI SDK and the Pi AI transport pass CPU compatibility tests.
These tests use recorded token scripts, not model quality or GPU performance tests.

## Connection settings

- Base URL: `http://127.0.0.1:8790/v1`
- Model: `crucible`, unless the server uses another `--model-id` name.
- API key: any dummy value, such as `unused`.
- API: Chat Completions, not the Responses API.
- Input: text. Images need a server started with `--vision` (base64 `data:` URLs only).

- Token count: `POST /v1/messages/count_tokens` takes an Anthropic Messages body and returns `{"input_tokens": N}`,
  the prompt size that Chat Completions prefills for the same conversation. It does no generation. The field
  mapping (system, text, thinking, `tool_use`, `tool_result`, tools, `tool_choice`, thinking and effort) is in
  `server/README.md`. It never applies the thinking skip, so its count can differ from what a served request
  prefills.
  Generation stays on Chat Completions; `POST /v1/messages` is not supported.

- Tokenize and template: `POST /tokenize` (also `/v1/tokenize`) and `POST /apply-template` use the OpenAI-compatible
  shapes. They return the prompt IDs or the prompt text of a request and do no generation. See `server/README.md`.

- Readiness: `GET /health` returns `{"status":"ok"}` when the engine is ready, and HTTP 503 after it stops.

The server does not authenticate requests.
Keep it on loopback and do not expose it to an untrusted network.
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
A request body above `--max-body-mib` (default 1 MiB, or 32 MiB with `--vision`) returns HTTP 413.
A client address above `--max-requests-per-client` returns HTTP 429. This limit is off by default.
A full or failed engine queue returns HTTP 503.
Invalid generated function calls come back as content with `usage.tool_call_fallback` (see server/README.md, "Tool calls").
The Responses API, multiple output choices, and full JSON Schema constraints are not supported.
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

The server refuses an unknown request field with HTTP 400.
The error names that field.
An unknown key inside `chat_template_kwargs` or `stream_options` also returns HTTP 400.
The error names the object and the keys that the object accepts.
The server refuses a field that it cannot apply instead of ignoring it, because the field could change the prompt.
A field that another OpenAI-compatible server accepts can fail here.
`presence_penalty` and `frequency_penalty` do not follow this rule.
The server accepts them at `0` or null, and they have no effect.
Any other value returns HTTP 400.
The server refuses `response_format` and `user` at any value, so a client or a proxy must remove both before it
sends the request.
The thinking on/off field has one accepted name: `enable_thinking`.
A different name, such as `think`, returns HTTP 400.

The server can drop thinking from a chat request that asks for it, so do not assume that `reasoning_content` is
present.
It does that when thinking is on, no thinking budget is set, and the room left for thinking is small.
Detect a drop with `usage.completion_tokens_details.reasoning_skipped_by`, which is `min_thinking_tokens`.
`server/README.md`, `--min-thinking-tokens`, gives the threshold, the exact rule, and the ways to turn it off.

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
