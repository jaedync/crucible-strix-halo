# Crucible local server

This server provides a small OpenAI-compatible API.
It supports text chat and text completions with greedy or exact sampled decoding.
It binds to `127.0.0.1` by default, on port 8790 unless `--port` selects another port.
`--host ADDRESS` selects another bind address (see [Bind address](#bind-address)).
It has no authentication and must remain local.
It is not a persistent service.

## Agent clients

See [OpenAI and Pi connections](../docs/OPENAI_CLIENTS.md) for client settings and CPU compatibility checks.
Use Chat Completions with text input, streamed usage, and validated function calls.

## Run a local server

Follow the [build guide](../docs/BUILD.md) to install Python dependencies.
Run these commands from the repository root.
Use a complete checkpoint from the [download guide](../README.md#download-a-model).

```bash
make
CKPT="$PWD/models/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5"
python -m server "$CKPT" --engine build/engine/crucible --ctx 32768
```

These commands assume exclusive use of your own GPU.
On a shared GPU, start each command through `tools/gpu-run` with a timeout.
Stop the server with Ctrl-C after the test.
Do not run another full engine while this server uses the GPU.

GEMM prefill keeps a 1.29 GB copy of the HC weights for speed.
It makes prefill about 44 ms faster per 4K tokens. The results do not change.

Send a chat request from another terminal:

```bash
curl -sS -N http://127.0.0.1:8790/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"crucible","messages":[{"role":"user","content":"Say hello."}],
       "temperature":0,"max_tokens":32,"stream":true,
       "stream_options":{"include_usage":true}}'
```

Set `stream` to `false` for one JSON response.
Use `GET /v1/models` to get the loaded model ID.
Use `GET /v1/crucible/version` to get the Crucible version and the git commit. `GET /v1/models` also has
`crucible_version`.
The server tells you at start when a newer release exists (one request a day to GitHub). Turn it off with
`--no-update-check` or `CRUCIBLE_NO_UPDATE_CHECK=1`.
Use `GET /health` for a readiness check: it answers `{"status":"ok"}` (HTTP 503 if the engine has stopped).
Use `/v1/completions` with a `prompt` string for plain text.
A completion can return the log-probabilities of its one next token. Set `logprobs` to true or `top_logprobs` to an
integer from 0 to 20, with `max_tokens` 1, `temperature` 0 and `stream` false. The reply has `token_id`, `logprob` and
the top candidates in `choices[0].logprobs.content`. A request that breaks one of these limits gets HTTP 400.
A completion prompt can also contain one list of token IDs.
Batches are not supported.

### Behind a proxy (llama-swap)

The server accepts any `Host` header. A reverse proxy such as llama-swap, nginx or LiteLLM sends the `Host` header of
its own client, for example `myserver:8080`. The server answers it with no extra setting.
The server binds `127.0.0.1` by default, so the proxy must run on the same machine.
To reach it from another machine, use `--host` (see [Bind address](#bind-address)).

A llama-swap config (keys checked against llama-swap's `config-schema.json`):

```yaml
healthCheckTimeout: 600   # seconds to wait for a model to load; the default is 120
models:
  crucible:
    proxy: http://127.0.0.1:${PORT}
    env:
      - PYTHONPATH=/path/to/crucible-strix-halo
    cmd: >
      /path/to/venv/bin/python -m server /path/to/checkpoint
      --engine /path/to/crucible-strix-halo/build/engine/crucible --port ${PORT}
```

- llama-swap polls `checkEndpoint` (default `/health`) for HTTP 200. `GET /health` answers `{"status":"ok"}` once the
  engine is ready. The server binds its port only after that. `checkEndpoint: /v1/models` works too.
- llama-swap has no working-directory setting, so `python -m server` finds the `server` package through
  `PYTHONPATH`. Use absolute paths in `cmd`.
- The `Host` rule below applies to `/health` as it does to every other path.

The default accepts any `Host`, so a web page can reach the server through DNS rebinding if your browser can open it.
To prevent this, turn on strict mode. Then the server refuses (HTTP 403) every other `Host`.

```bash
export CRUCIBLE_ALLOWED_HOSTS='127.0.0.1,localhost,[::1],myserver'
# or: python -m server ... --allowed-host 127.0.0.1 --allowed-host localhost --allowed-host myserver
```

- Each entry is `HOST` or `HOST:PORT`. A `HOST` with no port matches any port. The match ignores case, leading zeros
  in a port, and one trailing dot in the host.
- `--allowed-host` (repeat it, or use commas) replaces `CRUCIBLE_ALLOWED_HOSTS` when you give both.
  A flag that names no host (`--allowed-host ""`) stops the start.
- In strict mode, a request with no `Host` header is refused (403), and a request with two `Host` headers is refused
  (400).
- Strict mode works through a proxy only if the proxy sends the client's `Host`. llama-swap does. nginx does not by
  default: it sends `Host $proxy_host`, so the server sees only the proxy's own address. Then the proxy must do the
  check, or you set `proxy_set_header Host $host;`.
- `CRUCIBLE_ALLOWED_HOSTS=127.0.0.1:<port>,localhost:<port>` gives the behavior of the versions before this change.
- The server prints `Host check: off (any Host)` or `Host check: only ...` at start.

### Bind address

- `--host ADDRESS` sets the bind address. The default is `127.0.0.1`, so nothing changes without the flag.
- `ADDRESS` is an IPv4 or IPv6 literal, for example `0.0.0.0`, `::`, or `::1`. A host name or a scope id (`%eth0`) stops the start (exit 2).
- The start line `Crucible is ready at ...` shows the real address.
- An address that is not loopback prints one `WARNING` line at start.
  The server has no authentication, so every machine that can reach the address can use it.
- Turn on strict Host mode (`--allowed-host`, above) and use a firewall that limits the port to the clients you trust.
- `::` accepts IPv4 clients too: the server sets `IPV6_V6ONLY` to 0 on every host.
- `--host` does not change the Host check rules.

### Tokenize and apply the template (`POST /tokenize`, `POST /v1/tokenize`, `POST /apply-template`)

These utility paths use the OpenAI-compatible request shapes. They use the CPU only: no engine command, no queue entry, no
slot, and no cache change. A success writes no log line; a refusal writes one `crucible_error` line, as on chat.

- `/tokenize` and `/v1/tokenize` with `{"content": "text"}` return `{"tokens": [...]}`: the IDs of a
  `/v1/completions` prompt with the same text (special-token text is parsed, as in a raw prompt). A list of token
  IDs comes back unchanged, and an empty text gives an empty list. `add_special` and `parse_special` are accepted
  and change nothing (the model has no BOS token). `with_pieces: true` gets HTTP 400, and so does a body with both
  `content` and `messages`.
- `/tokenize` with `{"messages": [...]}` and any chat field (`tools`, `tool_choice`, thinking fields, `model`)
  returns the IDs that `/v1/chat/completions` prefills for that request, with the server flags. This includes the
  thinking skip (`--min-thinking-tokens`), which uses the request's `max_tokens`.
- `/apply-template` with `{"messages": [...]}` and any chat field returns `{"prompt": "..."}`: the chat prompt text
  that the template renders for that request, with the thinking skip as above. An image part shows as one placeholder; the prefill expands it.

### Token count (`POST /v1/messages/count_tokens`)

Gateways that use the Anthropic Messages format ask for the input token count before each request. The server
answers `200 {"input_tokens": N}`. N is the number of prompt IDs that `/v1/chat/completions` prefills for the same
conversation with the same server flags (template, tools, `--default-thinking`, `--vision`). The count calls the same
prompt function as the chat path (`Service.prompt_ids` in `server/http.py`). It uses the CPU only: no engine command,
no queue entry, no slot, and no cache change. A query string such as `?beta=true` is accepted.
The count uses the thinking-on prompt. An Anthropic count body has no `max_tokens`, so the thinking skip (`--min-thinking-tokens`) does not apply.

The request body is an Anthropic Messages body. `server/anthropic.py` maps it to a chat request:

| Anthropic field | Chat request |
|---|---|
| `system` (text or text blocks), or a `system` message | the first `system` message (blocks become text parts) |
| user `text` and `image` blocks, or string content | one `user` message; an image needs `--vision`, as in chat |
| user `tool_result` blocks | one `tool` message each, before the user's text, with `tool_call_id` = `tool_use_id`; text blocks are joined; `is_error: true` adds the text `Error: ` at the start |
| assistant `text` blocks | `content` (joined) |
| assistant `thinking` blocks | `reasoning_content` (joined); `redacted_thinking` is ignored |
| assistant `tool_use` blocks | `tool_calls` (`id`, `name`, `arguments` = JSON of `input`) |
| `tools` (`name`, `description`, `input_schema`) | function tools with `parameters` = `input_schema`; server tools are refused |
| `tool_choice` `auto`, `none`, `any`, `{"type":"tool","name":...}` | `auto`, `none`, `required`, the named function |
| `disable_parallel_tool_use: true` | `parallel_tool_calls: false` |
| `thinking` `{"type":"disabled"}`; `enabled` or `adaptive` | thinking off; thinking on (`budget_tokens` does not change the prompt) |
| `output_config.effort` `low`, `medium`, `high`, `max` | `reasoning_effort` with the same name: template levels `low`, `medium`, `xhigh`, `xhigh` |
| no `thinking` and no effort | the server default (`--default-thinking`) |
| `max_tokens`, `stream`, `temperature`, `top_p`, `top_k`, `stop_sequences`, `metadata`, `service_tier`, `context_management`, `cache_control`, `container` | ignored (another unknown field gets HTTP 400, because it could change the prompt) |
| `cache_control` and `citations` on blocks and tools | ignored |

Errors are the same as on the chat path, with the server's error object: an unknown model gets HTTP 404, a bad
shape or an unknown field HTTP 400, and a body above `--max-body-mib` HTTP 413. A refusal writes one
`crucible_error` line with a reason code only. A successful count writes no log line.

The server protects itself against a bad client:

- It accepts at most 256 connections at one time. Above that, a new connection gets HTTP 503.
- A client must send the request head and body in 60 s. Then the server closes the connection.
- A request that does not fit the engine line limit (2 MiB) gets HTTP 413.
- With `--serial`, a stream that does not read its output for 30 s stops with an error.
- `--max-requests-per-client N` limits the open requests of one client address (HTTP 429). It is off by default,
  because all clients on the loopback address share one address.
- The log line `crucible_engine_error` holds only known engine messages. Another engine text appears as `other`, with
  its length and a sha256 prefix. `--log-engine-errors` also logs that text. Use it only with your own data, because
  the text could quote request content.
- The server passes only the allowed `CRUCIBLE_*` switches to the engine (`ENGINE_KEEP_SWITCHES` in
  `server/backend.py`): kernel and layout switches, and switches that print numbers only (timings, draft counts,
  kill-check counts). It removes every other `CRUCIBLE_*` name, for example the capture, dump and interleave check
  switches, and names the removed switches once on stderr.
  The `kServe` column of the switch registry `engine/switches.h` is the source of this set. A CPU test checks it.
  Serve never writes content-derived data to disk.
- Special-token strings in client content are text (RA-8). This applies to message content, tool results, tool call
  arguments, tool definitions and the text parts of multi-part content. For example, a tool result that contains
  `<|im_end|>` or `<tool_call>` encodes to the characters that it shows. Only the chat template writes control
  tokens. A prompt with no such string gets the same IDs as before, so the cache, session replay and slot resume do
  not change. The `/v1/completions` prompt is raw model input: its special-token strings stay control tokens.
  An added token that the model wrote itself (for example in a tool-call fallback) keeps its ID when the client
  sends the returned assistant text back unchanged, so session replay still hits. The server remembers only a hash
  and token offsets of such texts. A text that the model generated in two different ways stays text. Because of
  this memory, one fixed request body can give other prompt IDs after a restart, as with session replay (main's
  decision). The token count and the tokenize paths follow the same memory.
  `--content-special-tokens parse` sets the old rule. The startup line names the mode
  (`Crucible is ready at ... (content special tokens: text)`). `server/content_tokens.py` has the method.

## Supported request options

- `temperature`: a finite number from `0` through `2`, default `0`. The server flag `--temperature` changes the default.
- `top_k`: an integer from `0` through `248320`; `0` disables this filter. The flag `--top-k` changes the default.
- `top_p`: a finite number from `0` through `1`, default `1`. The flag `--top-p` changes the default.
- `min_p`: a finite number from `0` through `1`, default `0`. The flag `--min-p` changes the default.
  A field in the request always wins over the server default. See [Server defaults](#server-defaults).
- `seed`: an unsigned 64-bit integer. An omitted or null seed uses fresh system entropy.
  Every response and stream chunk reports the selected value as `seed`.
- `presence_penalty`, `frequency_penalty`: the server accepts these fields only when they are `0` or null. They have
  no effect, because the engine has no penalty sampler. Any other value gets HTTP 400.
- `max_tokens` or `max_completion_tokens`: a positive integer. The default is 128 when the request has neither field.
  The flag `--default-max-tokens N|max` changes the default (see [Server defaults](#server-defaults)).
  With thinking on, the server keeps room for the answer (see below). A thinking request with a `max_tokens` below 1280
  is served with thinking off. This includes a request with no count under the default of 128.
  See `--min-thinking-tokens` below. Send a large value, for example 8192, for real
  thinking.
- `stop`: one string or at most four strings, each with 1 to 4096 UTF-8 bytes.
  Null, an empty list, or an omitted field disables text stops.
- `stream`: `true` or `false`.
- `stream_options.include_usage`: include the final SSE usage event.
- `spec`: draft depth from `0` through `7`, default `7`; `0` selects serial decode.
- `draft_q`: the draft confidence limit from `0` through `1`, default `0.3`; `0` disables it.
- `draft_head`: `64k` by default; use `full` for the full vocabulary. In batched serve, the server flag
  `--draft-blocks FILE` (an absolute path) replaces the built-in 64K blocks for every request: a regular file of
  1 to 65536 bytes with sorted unique 128-token block IDs in `[0, 1940)`, comma-separated. The head changes only
  the drafts, not the output IDs. Memory: the head is a device copy of its blocks, one 128-row block per
  ID: 0.205 MB per block with the 5-bit head of the 3.05 pack, 0.246 MB with the 6-bit head of the 4.05 pack. The
  built-in 512 blocks take 105.0 MB or 126.0 MB; every block above 512 takes the same amount from the serve
  headroom (the full 1940 blocks: 397.8 MB or 477.3 MB, so 292.8 MB or 351.3 MB more). The engine prints
  `serve draft head: N blocks from --draft-blocks, X MB (the built-in head: 512 blocks, Y MB)` at startup.
- `mtp_prefix`: auxiliary warmup limit from `0` through `262144`; `0` warms the full prompt. The default comes from
  the server flag `--mtp-window N` (default `256`). Batched serve accepts `0` too. At an equal restore point, the
  window changes only the drafts, not the output IDs. In batched serve, the server window also sets the MTP
  cache size of each slot: window `W` above 0 gives `W + 33024` positions (at most the slot context), and `0`
  gives the slot context. A request that needs more MTP positions restarts its MTP at position 0. This changes only
  the drafts.
  A profile that uses the full MTP history (`mtp_prefix 0` on its requests) must start the server with
  `--mtp-window 0`. With the default window, such a request restarts its MTP above `256 + 33024` positions and saves
  no MTP snapshot, so its drafts lose the history.
- Draft vocabulary data is a server setting, off by default. `--draft-capture DIR` (an absolute path; the
  server makes it 0700) writes `DIR/draft-counts-YYYY-MM-DD.json` (0600) every 10 minutes and at stop: for each UTC
  day and turn class (`chat`, `chat+think`, `tool`, `tool+think`), how often each 128-token block occurs in the
  generated output. The file holds numbers only: no text, no prompt, no request ID, no time finer than the day.
  `--draft-capture-mode tokens` writes a token ID spool for the machine-only training pipeline instead. Only that
  exact value turns it on, and it needs the user's explicit approval. `--draft-capture-max-gib N` (tokens mode
  only) limits the spool. DIR must be the server user's own directory, closed to group and other, and not a
  symlink; one server at a time writes it (`DIR/.lock`); file times are set to 00:00 UTC of the day. A capture
  fault never fails a request. `--draft-stats` (batch mode, counts mode only, default off) also writes
  `DIR/draft-stats-YYYY-MM-DD.json`: per turn class, counts of draft tokens by position and by token ID, and of
  rejected target token IDs. Counts only. Run the server and the engine from the same commit: an older engine
  rejects the `draft_detail` request field.
  Export floor: the per-day files never leave the serving machine. Any export sums 7 or more days and drops each class with
  fewer than 20 sequences.
- `draft_adapt`: optional `0` or `1`; the default adapts the built-in 64K head.
  Value `1` requires the `64k` head.
- Prompt lookup is a server setting, not a request option. The server flag `--lookup N` sets the depth
  (`0` to `7`, default `7`; minimum match 8 tokens). A speculative request then also proposes earlier prompt or
  output tokens that follow the same 8 or more tokens. Use `--lookup 0` to turn it off. The output does not change.
- `enable_thinking`: `true` or `false`. The default comes from `--default-thinking` (`off` unless set).
- `chat_template_kwargs`: `enable_thinking`, `preserve_thinking`, and `reasoning_effort` only.
- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`.
  The server maps `minimal` to `low`, and `high` and `max` to `xhigh`.
  Without an explicit `enable_thinking`, a level turns thinking on and `none` turns it off.
- `n`: only `1`.
- `return_token_ids`: an optional diagnostic extension.

The diagnostic extension adds `token_ids` to a non-streamed choice.
It adds `token_id` to each streamed token choice.
It includes terminal stop IDs.

With thinking on, the server splits the output at `</think>`, as other engines and Halogen do:

- The text before the tag goes to `message.reasoning_content` (streamed as `delta.reasoning_content`).
- The text after the tag goes to `content`. Tool calls come only from this part.
- `usage.completion_tokens_details.reasoning_tokens` counts the tokens before and including the tag.
- If `max_tokens` stops the reasoning before the tag, all text is reasoning and `finish_reason` is `length`.

The server keeps room for the answer, with Halogen's rule (Halogen 0.11 to 0.15):

- The answer room is `max(1024, 15% of max_tokens)` tokens, after `--clamp-max-tokens`.
  `--thinking-answer-room N` sets it to N tokens. `0` turns it off.
- A thinking budget can close the block earlier: `max_thinking_tokens`, or the aliases `thinking_budget_tokens`,
  `thinking_budget`, and `thinking_token_budget` (two different values give HTTP 400).
  `--max-thinking-tokens N` sets a server default. The request value wins.
- The block closes after `max(1, max_tokens - room)` generated tokens, or after the budget if it is smaller.
  The server then writes Qwen's budget sentence, `</think>`, and two newlines. These tokens count against `max_tokens`.
- With the default room and a `max_tokens` of 1025 or less, the block would close after 1 token, and the reasoning would be only Qwen's budget sentence.
  This happens only with `--min-thinking-tokens 0` or an explicit thinking budget. See the next rule.
- `--min-thinking-tokens N` (default 256, range 0 to 262144) skips thinking when the room leaves too little.
  The rule applies to a chat request with thinking on and no explicit thinking budget (from the request or `--max-thinking-tokens`).
  It applies when `max_tokens - room` is below N. The server then serves the request with thinking off.
  The prompt is the `enable_thinking: false` render, and the answer gets all of `max_tokens`.
  An explicit thinking budget never skips. `--thinking-answer-room 0` never skips. `--min-thinking-tokens 0` turns the rule off.
  The limit of 256 is the point where the forced close sentence (about 25 tokens) is a large part of the thinking.
  Below it, real thinking has no room.
- With the default room, a `max_tokens` below 1280 skips thinking. This includes a request with no `max_tokens`, which gets the default of 128.
  To keep thinking, send a `max_tokens` of 1280 or more, or a thinking budget.
- With `--default-max-tokens max` or `--clamp-max-tokens`, the rule uses the final `max_tokens`. A skip counts the prompt again with thinking off.
- The server skips only when the request fits the context both with thinking on and with thinking off.
  Otherwise the request keeps thinking, and a prompt that does not fit gets the same HTTP 400 as before.
- A skipped request has `usage.completion_tokens_details.reasoning_skipped_by` set to `min_thinking_tokens`.
  It has no `reasoning_content`, as with thinking off.
- When the whole close text fits, `usage.completion_tokens_details` has `reasoning_closed_at` (the token count) and
  `reasoning_closed_by` (`answer_room` or `max_thinking_tokens`).
- The close position depends only on the request, so it lands on the same token with any decode mode, layout, or load.

The engine has no forced-token command. At the close, the server stops the engine job and starts a new one with the
prompt, the generated reasoning, and the close text. The new job restores the prompt's snapshot and prefills the
reasoning again (about 1 s for each 1,300 tokens). Only turns that reach the close pay this cost.
The continuation depends only on the request. It is not bit-identical to one job that decodes the forced tokens,
because the second job computes the reasoning with prefill arithmetic.
That turn writes one `crucible_turn` line with the counts that the client sees and the first job's times,
so a log collector can join it to its own request record. `think_closed_at`, `think_closed_by`, and `cont_*` fields describe the
second job. `crucible_metrics` in the reply describes the second job.
Every `crucible_turn` line also carries numbers about the engine, the decode passes, the stall and the token
classes.

Two stderr lines show a turn while it runs. They hold counts only, never token IDs or text.

- `crucible_prefill`: one line after each prompt chunk of the batch worker. Fields: `request_id`, `done` (prompt
  rows prefilled), `total` (prompt rows), `elapsed_s` (from the job's first device command). The continuation job
  of a closed think turn also writes `turn_request_id` (the first job's `request_id`), so a reader can pair it with
  the turn.
- `crucible_decode`: the draft counters of a speculative turn during decode.
  Fields: `request_id`, `completion_tokens`, `spec_rounds`, `draft_proposed`, `draft_accepted`, `elapsed_s` (from
  the first token). All counts are cumulative for the turn.
  The batch worker writes at most one line a second for each turn, and one more line when the turn ends.
  The last line has the values of the turn's `crucible_turn` line (`elapsed_s` is its `decode_s`).
  A turn without speculation writes no `crucible_decode` line. A turn without the engine's step counters
  (an older engine) writes no `crucible_decode` line. No line comes after the final line.
  The two jobs of a closed think turn write their lines with the first job's `request_id`, which is also the
  `request_id` of the `crucible_turn` line. Their counts are cumulative over both jobs, and `completion_tokens` is
  the count that the client sees.
The reply has no `prompt_tokens_details`: the first job stops before it reports its cache count, and the second
job's count describes only the snapshot that the first job wrote.

The template replays assistant `reasoning_content` inside `<think>` for every turn, unless
`chat_template_kwargs.preserve_thinking` is `false`. Then it keeps only the reasoning after the last user turn.

The server renders the checkpoint's real chat template.
It accepts system, user, assistant, and tool messages.
It also accepts text content blocks, assistant `reasoning_content`, and assistant `tool_calls`.
With `--vision`, it also accepts image content parts. See [Image input](#image-input).
It rejects video, unsupported sampling controls, and unknown fields.

Generation stops at `<|im_end|>` (`248046`) or `<|endoftext|>` (`248044`).
The response omits these markers from text.
Usage counts include the generated stop token.
The prompt and requested output must fit the configured context.
Otherwise the server returns HTTP 400 with a message that contains "exceeds the context window".
Start the server with `--clamp-max-tokens` to lower `max_tokens` to the remaining context instead.
Request bodies are limited to 1 MiB (HTTP 413). `--max-body-mib N` raises the limit to N MiB (at most 64, or 1024 with `--vision`).
A 256K-token conversation of code can need more than 1 MiB.

Without `--streams`, the server uses the batch worker with one stream (`--streams 1`).
It processes one request at a time. The remaining requests wait in a queue.
`--streams N` (1 to 8) runs up to N requests at once.
`--serial` selects the serial worker. Do not use `--serial` with `--streams`. The serial worker has a FIFO queue with a limit of 64 waiting requests.
The batch worker refuses `draft_head` `full`, `draft_adapt` 1 and `draft_q` 0 with HTTP 400. Only `--serial` accepts them.
A full queue returns HTTP 503.
A streamed request gets its HTTP 200 headers when the server accepts it, before the prefill starts.
While no token comes, the server sends the SSE comment `: keepalive` every 10 seconds.
SSE clients ignore comments, and the comments keep body timers (for example undici's 300 s timer) from firing.
An error after the headers ends the stream with an error chunk and `data: [DONE]`.
A connected client waits up to 3600 seconds for its first token, and up to 360 seconds between tokens.
The server checks the client connection every 0.25 seconds while no token comes.
If the client closed it, the server cancels the request and logs a `crucible_cancel` line.
The batch worker then stops the request before its next prefill chunk.
The serial worker (`--serial`) finishes a disconnected request internally before the next request starts.
This keeps the engine protocol synchronized.
Without `--conversation-cache-mib`, each request resets the sequence state and the server keeps no conversation state.
With it, the server keeps prompt snapshots in host memory, and session replay is on by default
([session replay](../docs/SESSION_REPLAY.md)). A restart clears the cache.

### Server defaults

Some clients cannot send sampling fields or `max_tokens`. Set the defaults on the server instead.
A field in the request always wins over a server default.

| Flag | Value | Default |
| --- | --- | --- |
| `--temperature` | a number from `0` through `2` | `0` (greedy) |
| `--top-k` | an integer from `0` through `248320` | `0` (off) |
| `--top-p` | a number from `0` through `1` | `1` (off) |
| `--min-p` | a number from `0` through `1` | `0` (off) |
| `--default-max-tokens` | an integer from `1` through `262144`, or `max` | `128` |

- The server checks each value at start. A bad value stops the start with exit code 2, before the engine loads.
- Without these flags, every request gives the same result as before.
- `--default-max-tokens max` gives a request that has no `max_tokens` and no `max_completion_tokens` all the
  context room after its prompt. The room is `context - prompt tokens + 1`, the same room that `--clamp-max-tokens`
  uses. The rule applies to that request only.
- A request that sends `max_tokens` keeps the old rules. The server rejects a count that does not fit the context,
  or lowers it when `--clamp-max-tokens` is on.
- With thinking on, the answer room and the think close position use the final count. For example, a context room of 30000
  tokens gives an answer room of 4500 tokens.
- A prompt that fills the context fails as before, with HTTP 400.
- A response that uses the whole room ends with `finish_reason` `length`. The usage counts show the tokens that
  the model made.
- A completion request with `logprobs` or `top_logprobs` needs temperature 0. If it sends no temperature, the
  server uses 0 for it and ignores `--temperature`. If it sends a temperature above 0, the server refuses it, as before.
  A scoring request also needs `max_tokens` 1, so `--default-max-tokens` does not apply to it.
- `--top-k` accepts 0 to 248320. If the value is larger than the vocabulary of the loaded model, the server uses
  the vocabulary size. This filters nothing.
- With `--default-max-tokens max`, a request that sends `"max_tokens": null` gets HTTP 400. Leave the field out.
- With `--short-streams`, a request that gets `max` needs a full-context slot, so it can wait longer. The output
  does not change.
- The start line `Request defaults: ...` names each value that is not the built-in default.

Qwen recommends sampling for thinking mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`
(model card of Qwen3.8-Flash-Next, section "Best Practices"). For non-thinking mode, Qwen recommends
`temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`.
The card also warns of endless repetition and suggests `presence_penalty`, which this server accepts only at `0`.
Qwen recommends sampling, not greedy decoding (the Crucible default), for thinking mode.
Example for thinking mode, with all the context room for the output:

```bash
python -m server "$CKPT" --engine build/engine/crucible --ctx 32768 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0 --default-max-tokens max
```

Greedy output is the same for every run of a request. Sampled output changes with the `seed` field.
A request that sends a `seed` still gives the same tokens each time.

## Image input

Start the server with `--vision` to load the vision tower (about 0.3 GB).
Without `--vision`, a request with an image part gets HTTP 400 with a message that names the flag.
`--vision` also raises the default body limit to 256 MiB (`--max-body-mib` allows up to 1024 with `--vision`).

Send each image as an OpenAI `image_url` part with a base64 `data:` URL:

```json
{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
  {"type": "text", "text": "What does the red button say?"}]}
```

- User and tool messages can contain images. System and assistant messages cannot.
- A request can contain at most 256 images (`--vision-max-images`). The context window is the real limit:
  an agent client can send every earlier screenshot again on each turn. When the image tokens of a request pass the
  context (`--ctx`), the server refuses the request (HTTP 400, "exceeds the context window") before it decodes the
  next image.
- The server gives the engine each resized image as a private file (mode 0600) in a private directory
  (mode 0700) under `/dev/shm`. The server deletes the files when the engine has finished with the request.
  The engine reads only the files of images that it encodes; a cached image is not read.
- The server does not fetch `http` or `https` URLs. Send the pixels in the request.
- The server resizes each image as Halogen does: both sides become multiples of 32,
  with at least 256x256 and at most 2560x1440 pixels. `--vision-max-pixels N` (65536 to 4194304) changes the maximum.
- An image of H x W pixels after the resize uses (H/32) x (W/32) prompt tokens.
  A 1280x800 screenshot uses 1,000 tokens.
- A bad image gets HTTP 400 with the reason: not base64, not decodable, aspect ratio above 200:1,
  or above four times the maximum pixel count before the resize.
- The conversation cache keys each image by a SHA-256 of its resized pixels.
  Two different images never share a cache entry. A cached image is not encoded again.
- Text-only requests give the same tokens with and without `--vision`.
- Video parts are not supported.

## Tool calls

Chat requests accept `tools` with OpenAI function definitions.
The default `tool_choice` is `auto`.
Use `none`, `required`, or `{"type":"function","function":{"name":"add"}}` to select another policy.
Use `parallel_tool_calls: false` to allow at most one call in the model response.
These fields are not available on `/v1/completions`.

The server uses the checkpoint template and its typed XML format.
It returns OpenAI `tool_calls` with JSON argument strings and generated call IDs.
Its finish reason is `tool_calls` when the response contains calls.
Streaming sends each call in indexed `delta.tool_calls` entries while the model writes it.
The id, type and name go out when the function tag closes and the name passes `tool_choice` and `parallel_tool_calls`.
A parameter whose schema is only `"type": "string"` streams as JSON-escaped argument text.
Other parameters go out when they close and pass validation.
The closing `}` goes out when the whole call passes validation.
For a valid call, the joined argument text is byte-equal to the non-streamed `arguments`.
Use `--stream-tool-calls off` to send each call whole when the output ends.
A complete, well-formed call that fails only the schema goes out as a normal tool call with `finish_reason`
`tool_calls` (design decision, Oct 8). Examples: an unknown or missing property, a wrong type or enum value, a value
that is not JSON for its type (it goes out as the raw string), or a function that is not in `tools` (if its name
matches `[A-Za-z0-9_-]{1,64}`). The server never drops or changes a parameter. The client checks the call against
its own schema and sends the model the exact error. A tool with `strict: true` keeps the strict check.
stderr then has one `crucible_tool_pass` line with the call count and the reason codes only (not an error line).
Model output that is not a well-formed call comes back as content, with `tool_call_fallback: true` and a fixed reason
code in `usage.tool_call_invalid` (for example `incomplete_call`, `parameter_tag`, `unexpected_text`,
`duplicate_parameter`, `tool_choice`, `parallel_calls`).
If deltas of a call already went out, the stream ends with `finish_reason` `length`, not `stop`.
Clients such as Pi then do not run the partial call. Pi sends the model an error result and asks it to issue the call again.

The client must execute each function and send one tool result for each call ID.
Include the assistant message and its `tool_calls` before these results.
Use `{"role":"tool","tool_call_id":"call_...","content":"42"}` for a result.
Send all pending results before the next normal turn.
The server never executes a function.

Arguments must be finite JSON objects with unique keys.
The parser validates common types, required fields, bounds, and local schema references.
It does not implement complete JSON Schema or constrained decoding.
It does not execute untrusted schema regular expressions.
Schema processing has a shared limit of 16,384 traversal steps per request or output.
Input schemas share limits of 4,096 JSON nodes, 128 KiB of compact UTF-8 JSON, and 512 reference entries.
Nesting cannot exceed 64 levels; each reference chain cannot exceed 32 links.
Cyclic references are not supported.
The validator checks each schema node once and caches output matches by schema node and argument path.
Over-limit input returns HTTP 400 before queue submission.
A model output that does not parse as valid calls (malformed, over a limit, or against `tool_choice` or
`parallel_tool_calls`) returns as content, as in other engines, with finish_reason `stop` (or `length` at `max_tokens`).
The usage then has `tool_call_fallback: true`, and stderr has one `crucible_error` line with counts only.
HTTP 502 stays for engine failures. Grammar-constrained calls would prevent malformed calls.
Raw string values can be ambiguous when they contain literal structural end tags.
Prefer simple parameter types over string/nonstring unions.

## Engine JSON-line protocol

Start `crucible serve CKPT [--ctx N]` with exclusive GPU access.
The resident server uses the same decode defaults as `gen`.
Use `--spec 0`, `--draft-q 0`, `--draft-head full`, `--mtp-prefix 0`, or `--draft-adapt 0` to disable each feature.
Raw request fields override these startup settings for one request only.
The engine loads once and writes a `ready` event.
Send one JSON object per stdin line:

```json
{"ids":[248045,846,198],"max_tokens":32,"spec":2,"stop_ids":[248046,248044]}
```

The engine writes accepted token IDs immediately:

```json
{"event":"token","id":123}
```

A `done` event gives prompt usage, completion usage, timing, and the finish reason.
A malformed request produces an `error` event without stopping the resident engine.
The input limit is 2 MiB per line.
With `--vision 1`, a request can add `"images"` (at most 256): a list of objects with `offset` (first image token),
`height`, `width`, `digest` (64 hex characters), and `path` (an absolute path to a regular file of
height x width x 3 RGB bytes, not a symbolic link). The digest is the SHA-256 of the little-endian `uint32` pair
(height, width) followed by the pixels. The engine reads a file only when the image needs an encoder pass,
and it refuses pixels that do not match the digest. The engine never deletes the file; the caller does.
In batch mode, a file that fails its check after the open ends only its request: the `prefill` command returns
`{"event":"error","request_id":...,"slot_generation":...,"error":...,"recoverable":true}`, and the slot is free.
Each image needs (height/32) x (width/32) image pad tokens (`248056`) at its offset.
In batch mode, the `opened` event of an image request has `vision_slices`: the next that many `prefill` commands
each run one encoder slice (about 0.1 s or less), so other sessions can decode between them.
In batch mode, the first `token` event of a speculative request in each `spec_step` or `verify_step` has the
request's cumulative `draft_proposed`, `draft_accepted` and `spec_steps` after that step (counters only). The
other `token` events and all events of plain requests do not change.
Only stdout carries protocol events.
The engine writes diagnostics to stderr.

Both modes use exact target verification.
Positive temperature applies top-k, top-p, and min-p in that order, then normalizes the probabilities.
Top-p retains the token that crosses its probability limit.
Min-p compares each probability with the largest probability.
Temperature zero ignores these filters and preserves greedy output.
Sampling uses a keyed draw: each draw depends only on the seed, the absolute token position, and the token ID.
Repeated identical requests with an explicit seed return identical token IDs.
Serial, speculative, and packed batch decoding give the same sampled token IDs for the same seed
(`engine/runtime/sampling.h`, keyed sampling law).

The CLI also supports optional draft policies and the same sampling parameters.
Use `--temperature`, `--top-k`, `--top-p`, `--min-p`, and `--seed`.
Positive-temperature CLI sampling does not support `--draft-overlap 1`.
The resident API uses exact target verification with the adaptive 64K proposal head by default.
It uses no lookup or asynchronous overlap.
Use `draft_head: "full"` to restore the full proposal vocabulary.
It emits verified IDs, never draft proposals.

## Tests

Run the CPU tests:

```bash
CRUCIBLE_TEST_CKPT="$CKPT" make test-cpu
```

Run these supplemental device tests only with exclusive GPU access.
On a shared GPU, start each command through `tools/gpu-run` with a 300-second timeout.

```bash
python -m server.tests.live_server_test "$CKPT"
python -m server.tests.live_tools_test "$CKPT"
```

These acceptance tests do not replace shared gate evidence.

It checks a real tool call, its result, streaming deltas, and exact greedy gen IDs.
It also checks explicit replay with the reported random seed.
It stops the temporary server and resident engine.

The original server test uses curl and stops its server in a cleanup block.
It checks exact dump files, exact output IDs, streaming, usage, queueing, and stop IDs.
It records readiness, time to first token, and decode speed under `results/`.

## Charter

- **Interface:** the local HTTP API (OpenAI and Anthropic shapes), the chat
  template, tokenization and tool-call checks. `EngineBackend` and
  `BatchedEngineBackend` talk to the engine's JSON-line protocol
  (`engine/serve/serve_protocol.h`, `engine/serve/batched_serve_protocol.h`).
- **Invariants:** binds to `127.0.0.1` unless `--host` says otherwise. A request's output depends only
  on the request. The serial backend is behind `--serial`.
- **Allowed dependencies:** `requirements-cpu.txt` and the engine binary.
  Model metadata comes from the engine's `ready` event, not from new
  literals.
- **Switches:** server flags. Engine switches that the server passes come
  from the registry.
- **Lever ledger (closed):** none recorded yet.
