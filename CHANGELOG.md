# Changelog

Releases are numbered `v0.<train>.<fix>`. This file lists the newest release first.
The same text is in the notes of each release on GitHub.

## v0.26.0 (2026-10-24)

Release v0.26.0 is the engine of train 26. It follows v0.22.4.
It adds new packs, a faster build, a conversation cache that explains its misses, and a way to update.

### New packs

- The loader reads the `h6_ng8` EXL3 packs from turboderp, with self-calibrated weights. The sizes are 2.18, 2.68,
  3.18, 3.68 and 4.18 bits per weight.
- Each of these packs loads and generates text.
- Weight widths go from 2 to 8 bits. Routed experts can also have half-bit widths (for example 3.5).
- The `h6_ng8` packs hold an 8-bit n-gram table. `CRUCIBLE_PLE_BF16` (off by default) reads the n-gram table at
  16 bits from a file instead. It needs `CRUCIBLE_PLE_HOST=1`.
- The packs 3.05 and 4.05 stay the qualified packs. The README tables describe them.

### Speed

- Several parts of decode and prefill are faster. Speculative decoding is still exact: the target model checks every draft.
- A draft chain now stops at a stop token and at the token limit.
- Long prompts prefill in launches of 8,192 rows. A short last part joins the last launch.

The tables compare v0.22.0 (A) with v0.26.0 (B). A is the engine of train 22. The releases v0.22.1 to v0.22.4 use the
same engine. Each cell is the median of 5 pairs. The v0.22.0 cell for 4 sessions on the 4.05 pack is the median of 4 pairs, because one pair gave no value. The ratio is B divided by A. For the rows in tok/s, a ratio above 1
means that v0.26.0 is faster. For the rows in seconds, a ratio below 1 means that it is faster.

Setup: Strix Halo with 128 GB. Each pack ran 5 pairs of A and B, in the order AB, BA, AB, BA, AB. A pair that failed a
rule (low GPU clock, high disk wait, or a missing value) was run again once. The 4.05 pack had 7 tries (14 runs) and the 3.05 pack had 5 tries (10 runs). The median GPU clock
was about 2,330 MHz in both builds. The server ran 4 sessions of 262,144 tokens, with MTP on and image input on. Only
the serial row has MTP off. The cold rows use prompts that no cache has seen.

**4.05 bpw pack**

| Row | A: v0.22.0 | B: v0.26.0 | B/A |
|---|---:|---:|---:|
| Serial decode, short context (tok/s) | 30.9 | 32.0 | 1.036 |
| Decode at 32K, 1 session, code (tok/s) | 66.1 | 64.9 | 0.982 |
| Decode at 32K, 1 session, prose (tok/s) | 49.0 | 49.3 | 1.005 |
| Decode at 32K, 4 sessions, total (tok/s) | 110.8 | 105.2 | 0.949 |
| Cold prefill 32K, time to first token (s) | 19.74 | 19.04 | 0.965 |
| Cold prefill 130K, time to first token (s) | 84.15 | 81.94 | 0.974 |
| Re-send of the same 32K prompt, time to first token (s) | 1.872 | 0.021 | 0.011 |

**3.05 bpw pack**

| Row | A: v0.22.0 | B: v0.26.0 | B/A |
|---|---:|---:|---:|
| Serial decode, short context (tok/s) | 35.4 | 36.5 | 1.029 |
| Decode at 32K, 1 session, code (tok/s) | 75.3 | 73.4 | 0.974 |
| Decode at 32K, 1 session, prose (tok/s) | 59.5 | 59.2 | 0.995 |
| Decode at 32K, 4 sessions, total (tok/s) | 129.6 | 129.2 | 0.997 |
| Cold prefill 32K, time to first token (s) | 18.75 | 18.78 | 1.001 |
| Cold prefill 130K, time to first token (s) | 80.39 | 80.77 | 1.005 |
| Re-send of the same 32K prompt, time to first token (s) | 1.728 | 0.021 | 0.012 |

What the tables show:

- A prompt that you send again gets its first token in about 0.02 s instead of 1.7 to 1.9 s.
- Cold prefill is 2.6% to 3.5% faster on the 4.05 pack. On the 3.05 pack it is equal within 0.5%.
- Serial decode is 3% to 4% faster.
- Decode with 4 sessions on the 4.05 pack is 5% slower: 110.8 to 105.2 tok/s (0.949). One of the 5 runs of v0.26.0
  was low (86.4 tok/s). In the 4.05 test, 3 of the 14 runs gave no value for this row: 2 of v0.22.0 and 1 of v0.26.0.
  This is a known fault of the measurement and is not new in this release. The 3.05 test had no failed run.
- The 3.05 pack shows no loss with 4 sessions (0.997).
- Decode at 32K with 1 session varies more between runs in v0.26.0 on the code prompt. The medians are within 3%.

### Conversation cache

- A prompt that you send again restores from the cache in more cases. This includes a prompt that the cache holds
  in full.
- A cache miss has a stated cause. The `crucible_turn` line of the server log now has `miss_reason`. On a miss it also
  has `closest_shared_tokens`, `closest_age_s` and `closest_outcome`.
- The slot, host and disk tiers count their own lookups and hits (`cache_slot_*`, `cache_host_*` and `cache_disk_*`
  on each turn line). Each host lookup writes one `cache_lookup tier=host` line.
- The newest state of a conversation stays in the cache.
- A slot can copy the state of another slot on the device when the host cache has a miss.
- The disk tier (`--state-dir`) stores equal blocks once. It reads a state back during a run. A state that the host
  cache refuses goes straight to disk.
- The disk files have a new format. At start, the server removes the state files of v0.22 from the state folder.
  The cache starts empty.

### Server

- `GET /health` answers `{"status":"ok"}` when the engine runs. Use it as a readiness check, for example in llama-swap.
  This is already in v0.22.1.
- The server accepts any `Host` header by default. Set `CRUCIBLE_ALLOWED_HOSTS` or `--allowed-host` for a strict list.
  This is already in v0.22.1.
- `GET /v1/crucible/version` gives the version and the git commit. `GET /v1/models` has `crucible_version`.
- The server flags `--temperature`, `--top-k`, `--top-p`, `--min-p` and `--default-max-tokens N|max` set defaults for
  requests that omit those fields. A field in the request always wins. Without a flag, nothing changes.
  A request for log-probabilities stays at temperature 0. This is already in v0.22.2.
- During a speculative turn, the server writes a `crucible_decode` line with draft counters. The line holds counts
  only. It never holds token IDs or text.
- A completion with `max_tokens` 1, temperature 0 and no stream accepts `top_logprobs` (0 to 20).
- With `--serial`, the engine stops a thinking turn at the point where the server closes the think block (the
  thinking budget, or the end of the answer room), not at `max_tokens`. Before, the engine decoded on to `max_tokens`
  after the close, and the answer waited behind it. In our GPU test of 2026-10-23, the wait after the forced close fell
  from 12.0 s to 0.3 s, and the engine made no extra tokens.

### Build

- The engine device code compiles in parallel parts (64 by default). An edit rebuilds only the parts that it changes.
  `make SPLIT=0` uses plain `hipcc`. `make isa-check` shows that the ISA is equal.
- The number of parallel compiles follows `make -j`. The default is 1.

### Updates

- The file `VERSION` and the file `CHANGELOG.md` give the release number and the notes.
- `build/engine/crucible --version` and the server start line show the version.
- The server asks GitHub once a day if a newer release exists, and prints a notice. It changes nothing. Turn the
  check off with `--no-update-check` or `CRUCIBLE_NO_UPDATE_CHECK=1`.
- `make update` moves your clone to the newest release, builds it, and runs a self-test. If a step fails, it restores
  the previous release. If a kill or a power loss stops an update, the next `make update` or `make rollback` restores the
  previous release first. `make rollback` undoes the last `make update` (once, with no local changes). It then holds the
  release that you left: `make update`, and the timer, skip it until a newer release exists.
- Automatic updates are off. "Updates" in the README shows how to turn on the timer.

### How to update from v0.22

1. Stop the server.
2. Run `git fetch --tags` and `git checkout v0.26.0` in your clone, then `make`.
3. Start the server. For later releases, run `make update`.

The cache starts empty after the update. Your pack files and checkpoints do not change.

Needs your action, if you come from v0.22.3 or earlier: a thinking request with `max_tokens` below 1280 and no
thinking budget now gets no thinking. `--min-thinking-tokens 0` restores the old behavior. If you use v0.22.4, you
have this change already.

## v0.22.4 (2026-10-23)

A server release for the train 22 engine. It improves thinking requests with a small `max_tokens`. It adds a flag for
the listen address. The engine did not change.

- New flag `--min-thinking-tokens N` (default 256, range 0 to 262144). With thinking on, the server keeps room for the
  answer. Before, a small `max_tokens` left thinking 1 token. Now, when thinking would get fewer than N tokens, the
  server serves the request with thinking off, and the answer gets all of `max_tokens`. The usage block then shows
  `completion_tokens_details.reasoning_skipped_by: "min_thinking_tokens"`.
- New flag `--host ADDRESS` (default `127.0.0.1`) sets the listen address, IPv4 or IPv6. For an address that is not
  loopback, the server writes a warning at start, because it has no authentication.
- `/tokenize` and `/apply-template` return the prompt that chat sends, also when the server skips thinking.

Needs your action:

- With the default settings, a thinking request with `max_tokens` below 1280 now gets no thinking. This includes a
  request with no `max_tokens` (the default is 128). To keep thinking, send `max_tokens` 1280 or more, or a thinking
  budget (`max_thinking_tokens` in the request, or `--max-thinking-tokens` on the server).
  `--min-thinking-tokens 0` restores the old behavior.
- If you serve other machines with `--host`, also turn on the strict Host check (`--allowed-host`) and use a firewall.

Requests with a larger `max_tokens` or with thinking off give the same tokens as in v0.22.3.

## v0.22.3 (2026-10-23)

A server release for the train 22 engine. The engine did not change.

- The server accepts `presence_penalty` and `frequency_penalty` when the value is `0` or null. Some clients, such as
  SillyTavern, send both in every request. The fields have no effect, because the engine has no penalty sampler.
  Any other value gets HTTP 400.

## v0.22.2 (2026-10-23)

A server release for the train 22 engine. It adds server-level request defaults. The engine did not change.

- New flags `--temperature`, `--top-k`, `--top-p` and `--min-p` set the value for each request that does not send the
  field. A field in the request always wins.
- New flag `--default-max-tokens N|max` sets the output tokens for a request that sends no `max_tokens` and no
  `max_completion_tokens`. With `max`, the request gets all the context room after its prompt.
- The server checks each value at start. A bad value stops the start with exit code 2.
- Without the new flags, the server works as before: greedy decoding and 128 output tokens when a request sends no
  count.
- The server README names the range of `--top-k` (0 to 248320), says that `"max_tokens": null` gets HTTP 400 with
  `--default-max-tokens max`, and says that `--short-streams` can make a `max` request wait longer.

## v0.22.1 (2026-10-23)

A fix for the Host check of the server.

- The server accepts any `Host` header by default, so a proxy or a container name works. `CRUCIBLE_ALLOWED_HOSTS` and
  `--allowed-host` turn on a strict list. A request with two `Host` headers gets HTTP 400 when the list is on.
- `GET /health` answers `{"status":"ok"}` when the engine runs. It gives HTTP 503 when the engine has stopped.
- The engine never inherits `CRUCIBLE_ALLOWED_HOSTS`.
- The server README has an example for llama-swap.

## v0.22.0 (2026-10-21)

The first numbered release. It is the engine of train 22.

- The engine runs EXL3 checkpoints of Qwen3.8-Flash-Next on one AMD Strix Halo APU (gfx1151).
- Speculative decoding is exact: the target model checks every draft.
- A local OpenAI-compatible server gives chat, streaming, tools and up to eight sessions on one engine.
- The conversation cache keeps session states in host memory. An optional disk tier keeps them across restarts.
- Image input is opt-in (`--vision`).
- Two packs are supported: `3.05bpw_h5_ng5` and `4.05bpw_h6_ng6`.
- The README states the measured speed and the distance from the 16-bit model.
