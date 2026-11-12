# Crucible

An inference engine for EXL3 checkpoints of Qwen3.8-Flash-Next on one AMD Strix Halo APU. It is built for fast long-context agent work.

Crucible runs **EXL3** checkpoints of **Qwen3.8-Flash-Next** on **AMD Strix Halo**: one **Ryzen AI Max+ 395** with a
**Radeon 8060S** (**gfx1151**) and 128 GB of unified memory. It uses its own **ROCm/HIP** kernels in C++20 and no other
engine. A local OpenAI-compatible server gives chat, streaming, function calls and exact speculative decoding. This is
research software for one GPU architecture. It is not a general model backend or a production service.

[How it got fast](docs/how-it-got-fast.md) · [Results](docs/RESULTS.md) · [Rivals](docs/RIVALS.md) ·
[Quality](docs/QUALITY.md) · [Changelog](CHANGELOG.md) · [Architecture](docs/ARCHITECTURE.md) · [Setup guide](docs/QUICKSTART.md) · [Build guide](docs/BUILD.md)

## Speed

Measured on the Ryzen AI Max+ 395 with the Radeon 8060S and 128 GB of memory, balanced power profile, with the public
files and the configuration that `./crucible start` uses: 4 sessions of 262,144 tokens, speculative decoding on.

| | 4.05 bpw pack | 3.05 bpw pack |
|---|---:|---:|
| Prefill, cold 32K prompt (tok/s) | 1,831 | 1,877 |
| Prefill, cold 130K prompt (tok/s) | 1,782 | 1,814 |
| First token, cold 32K prompt (s) | 17.1 | 16.6 |
| First token, cold 130K prompt (s) | 72.4 | 71.1 |
| First token, the same 32K prompt again (s) | 0.019 | 0.015 |
| Decode, 1 session, code (tok/s) | 71.2 | 76.8 |
| Decode, 1 session, prose (tok/s) | 58.4 | 60.7 |
| Decode, 4 sessions, total (tok/s) | 118.3 | 129.0 |
| Decode without speculation (tok/s) | 34.2 | 39.1 |

The table shows v0.34.0 (built with ROCm 7.2.4): median of 2 runs, Nov 7, 2026. It was not run again for later releases.
Decode speed changes with the text, because some text is easier to draft.

The method, the charts by context length and the tables of earlier releases are in [Results](docs/RESULTS.md).
The 512K configurations have their own numbers in [Long context](#long-context-opt-in).

## Long context (opt-in)

v0.37.0 can serve up to 524,288 tokens in one session with YaRN (factor 2). It is a start option of the same engine: a
context above 262,144 (`--ctx`) turns it on. A request that needs 262,144 positions or fewer runs with the plain rotary
positions, also in a long configuration. Setup offers three configurations by name:

| Name | Setup | Server flags | Sessions | Session state (engine formula) |
|---|---|---|---|---|
| default | `./crucible setup` | `--streams 4 --ctx 262144` | 4 of 262,144 | 28.28 GB |
| long | `./crucible setup --layout long` | `--streams 3 --ctx 524288 --short-streams 2 --short-ctx 262144` | 1 of 524,288 and 2 of 262,144 | 28.05 GB |
| two long | `./crucible setup --layout two-long` | `--streams 2 --ctx 524288` | 2 of 524,288 | 27.83 GB |

All three need about the same memory, so a long configuration fits where the default fits. The formula agrees with the
measured state of one session of 524,288 tokens (13.84 GB, Nov 10, 2026). `./crucible setup --layout default` goes back.

Measured on the release engine of v0.37.0, 4.05 pack, with the public files, in the configuration `long`: one run, power
profile balanced, Nov 11, 2026. One server start held both prompts, and both went through HTTP. The command is below the table.

| Row | 250,000 tokens (plain positions) | 490,000 tokens (YaRN) |
|---|---:|---:|
| Cold prompt: first token (s) | 149.0 | 340.4 |
| Cold prompt: prefill (tok/s) | 1,677.6 | 1,436.2 |
| Decode, code, with drafts (tok/s) | 67.7 | 59.8 |
| Decode, tool call, with drafts (tok/s) | 97.0 | 100.2 |
| Decode, all five workloads, with drafts (tok/s) | not measured: 3 of the 5 replies stopped after 12 tokens | 52.6 (server rate over all 2,366 tokens; code 59.8, tool 100.2, code sampled 46.1, think 50.5, prose 45.1) |

```sh
CRUCIBLE_PLE_HOST=1 CRUCIBLE_COMPACT_STATE=1 CRUCIBLE_PLE_FETCH=1 CRUCIBLE_DRAFT_HEAD=q2 CRUCIBLE_DRAFT_CHAIN=hc8,lead \
CRUCIBLE_PF_LAUNCH_ROWS=16384+128 python -m server models/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6 \
  --streams 3 --ctx 524288 --short-streams 2 --short-ctx 262144 --prefill 4096 --conversation-cache-mib 9700 \
  --host-reserve-mib 11264 --vision --clamp-max-tokens --default-thinking on --max-body-mib 16 --model-id crucible \
  --session-replay 1 --slot-resume 1 --slot-checkpoints 1 --draft-capture DIR --draft-stats
```

The words and the flags up to `--max-body-mib 16` are the ones that `./crucible setup --layout long` writes. The
measurement tool adds the rest (`--draft-capture` and `--draft-stats` write the draft counters).

What the long configurations cost:

| Item | Effect | Source |
|---|---|---|
| A short request in a long configuration | None: it runs with the plain positions, so it gets the tokens of the default configuration. | the server decides the mode of each request |
| YaRN forced on a short context (a test word, not a configuration) | KL against the BF16 reference +0.00586 (SE 0.00097), top-1 0.52 points lower | 7,168 positions, night of Nov 10 to 11, 2026 |
| A conversation that grows past 262,144 tokens | One full prefill of the conversation from its first token, in YaRN mode. Later turns use its YaRN states. | by design |
| Resume of a long conversation | Only from its session slot. The conversation cache in memory cannot hold a 524,288-token state, and a YaRN start has no disk tier (`--state-dir` is refused). | first version |
| Requests above 524,288 tokens | HTTP 400 `context_length` | |

## How it compares

Our own measurements of Halogen 0.18.14 and Gufo 0.22.0 on the same machine, with the same prompts and client, Oct 23 to 26,
2026. The GPU clock was similar; in the prefill rows Halogen ran about 3% higher. Each value is the median of 2 rounds. Each
Crucible value is the 4.05 pack and the Crucible build of that date (v0.26.0 or a build of Oct 25), older than v0.32.0. Bold marks the fastest.

| Row | Halogen 0.18.14 | Gufo 0.22.0 | Crucible, same run | Measured |
|---|---:|---:|---:|---|
| Serial decode, no speculation, short context (tok/s) | **42.2** | 29.3 | 32.2 | Oct 23 to 24 |
| Decode, mixed text, 1 session, speculation (tok/s) | **75.0** | 45.2 | 62.3 | Oct 23 to 24 |
| Decode, 4 sessions at 32K, total (tok/s) | 96.5 | 90.4 | **111.4** | Oct 23 to 24 |
| Cold prefill, 32K, balanced power profile (tok/s) | **1,938** | 1,693 | 1,614 | Oct 23 to 24 |
| Cold prefill, 32K, performance power profile (tok/s), Crucible build of Oct 25 | **2,121** | 1,843 | 1,870 | Oct 26 |

Halogen leads one-session decode and cold prefill at equal bits. It stores the layers that mix the hidden streams in 6 bits, and
its pack has 4.16 bits per weight. Crucible leads with 4 and 8 sessions. At equal bits (4.05), Halogen also leads decode with speculation on code (104.9 against 89.1 tok/s). The 3.05 pack has a lower KL divergence than the pack of Halogen (see below). All rows, dates, packs and the method are in [Rivals](docs/RIVALS.md). Newer versions exist (Halogen 0.18.18, Gufo v0.27.0, both Oct 29, 2026); we measured neither, and we did not measure the rival values again for v0.32.0. Thank you to the authors of Halogen and Gufo: their engines made this comparison possible.

## Why Crucible

- Batching, cache state and speculation do not change the tokens of a request. Speculative decoding changes the speed, not the tokens: the target model checks every draft token, and
  only checked tokens commit. Load and speculation depth do not change the tokens either. Each speed change must give
  byte-identical tokens on a fixed gate set before it merges ([exactness rules](docs/ARCHITECTURE.md#exactness-rules)).
  [Session replay](docs/SESSION_REPLAY.md) lists the cases in which a cache resume gives other tokens than a cold prompt.
- **Quality per bit.** KL divergence from the 16-bit (BF16) model, on 896 text positions, scored by us on Oct 24 to 25, 2026 (lower is closer). Gufo 0.22.0
  returned no log-probabilities, so it has no row (v0.27.0 serves them; not scored). [Quality](docs/QUALITY.md) has older tables.

| Engine and pack | Bits per weight | KL from BF16 | Same top token as BF16 |
|---|---:|---:|---:|
| Crucible 4.05 | 4.05 | **0.0230** | **94.3%** |
| Crucible 3.05 | 3.05 | 0.0513 | 91.7% |
| Halogen 0.18.14 | 4.16 | 0.0648 | 90.3% |

- **The conversation cache.** A later turn pays only for its new tokens. Sending the same 32K prompt again gives the first token
  after 0.021 s instead of 1.87 s in v0.22.0 (v0.26.0: 0.014 s; 4.05 pack). An optional disk tier keeps conversations
  across a restart.

## Quick start

**New machine?** Follow the [setup guide](docs/QUICKSTART.md): each step has a check, for a person or an LLM agent. It also has a
[one-command install](docs/QUICKSTART.md#one-command) and [Run as a service](docs/QUICKSTART.md#run-as-a-service).

### Build

```sh
make   # needs the software under "Requirements"; writes build/engine/crucible
# if make stops with "argument facts drift" on another ROCm, run: make SPLIT=0
```

### Download a model

Read the [model license note](#licenses) first. Keep every tokenizer file, weight shard, n-gram table and MTP mixer patch of the pack
([all packs](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)). The 3.05 payload alone is about 84.45 GB.

```sh
PACK=3.05bpw_h5_ng5   # or 4.05bpw_h6_ng6, the higher-quality pack
CKPT="$PWD/models/Qwen3.8-Flash-Next-exl3-$PACK"
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision "$PACK" --local-dir "$CKPT"
```

### Run

```sh
python -m server "$CKPT" --engine build/engine/crucible --ctx 32768
curl -sS http://127.0.0.1:8790/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"crucible","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":32}'
```

This line is a short test: it keeps the conversation cache off and sets none of the words of the speed table, so a long conversation prefills again on every turn. For the measured configuration, use step 6 of the [setup guide](docs/QUICKSTART.md) or the [Docker](docs/DOCKER.md) run line. With 4 sessions of 262K tokens and `--vision` on the 4.05 pack, no launch-row setting is needed. The default (`16384+128`) fits 4 sessions of 262K with `--vision` on the 4.05 pack (stage arena 2,174 MiB); `8192+128` uses 1,439 MiB and gives a slower cold prefill.

**Security.** The server binds to `127.0.0.1:8790` and needs no key there. A key is required for an address that is not loopback, except in the Docker image. Give the key as a file (`--api-key-file PATH`);
without it the server exits before it loads the model. With a key, every route but `GET /health` needs it.
There is no TLS: to reach the server from other machines, use a reverse proxy ([security](SECURITY.md), [remote access](docs/REMOTE_ACCESS.md)).
The Docker image listens on all container addresses with no key and no default Host list, and warns at start: publish its port to
`127.0.0.1` only, or use a key ([Docker](docs/DOCKER.md)). Do not use the server as a public service.

The API provides `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/tokenize`. Use `stream: true` for server-sent events
and OpenAI `tools` definitions for function calls ([server guide](server/README.md), [OpenAI clients](docs/OPENAI_CLIENTS.md)).

To watch the engine live (requests, rates, memory, engine switch), see [Kiln](https://github.com/jaedync/crucible-kiln), a dashboard and OpenAI-compatible gateway for Crucible.

### Requirements

Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB shared LPDDR5X-8000; Linux (tested: Arch Linux with kernel 7.2.7 and ROCm 10.1 and 7.2.4); a C++20 HIP compiler, GNU Make and POSIX threads; Python 3.12 or newer
(`requirements-cpu.txt`); disk space for one checkpoint and the build. Set `CRUCIBLE_PLE_HOST=1` to keep the n-gram table in host
memory (32.6 GB at 3.05, 39.0 GB at 4.05). To build from source, use the ROCm 7.2.4 compiler for now: an engine built with the ROCm
10.1 compiler does not pass our token gates yet. The release engine runs on ROCm 10.1 and on 7.2.4. The release notes name the ROCm of each number. The CPU tests run with `make -j2 test-cpu` and need no GPU ([all targets](docs/BUILD.md)).

## Features

- Exact speculative decoding with the model's draft head, and a keyed sampling law: a seeded request gives the same tokens at any speculation depth ([how](docs/how-it-got-fast.md#3-guess-ahead-and-check)).
- Local OpenAI-compatible server: chat, streaming, tools, up to 8 sessions on one engine ([server guide](server/README.md)).
- Conversation cache with an optional disk tier (`--state-dir`) ([session replay](docs/SESSION_REPLAY.md)).
- Image input, opt-in with `--vision`. No video.
- EXL3 packs from 2.18 to 4.18 bits per weight load. The qualified packs are 3.05 and 4.05 ([changelog](CHANGELOG.md)).
- Opt-in 16-bit n-gram table, built with one command ([NGRAM16](docs/NGRAM16.md)).
- Update by hand with `make update`, with a rollback; an optional timer ([Updates](#updates)).
- Hardware ceilings measured on the chip, and each speed claim names its ceiling ([calibration](docs/CALIBRATION.md)).

## Roadmap

In priority order. Each item moves only when a measured row moves.

1. More accepted tokens for each weight pass: a better draft head for prose and chat, and verify passes that cost close to a serial step.
2. Single-session decode: fewer bytes read for each token and fewer gaps between kernels. Serial decode uses 70% to 77% of the measured memory bandwidth (Oct 22 audit).
3. Cold prefill: the MoE matrix products, and cold prompts launched in pairs. Cold prefill is at 53% to 56% of the estimated WMMA ceiling (Oct 22 audit).
4. Quality per bit: more packs, mixed precision, the 16-bit n-gram table as a default if it pays.
5. Serving: cache memory, the disk tier, and a shared prompt start between sessions.

## Updates

Releases are numbered `v0.<train>.<fix>`; the notes are in [CHANGELOG.md](CHANGELOG.md) and on the Releases page. The server
shows its version at start. Once in 24 hours it asks GitHub if a newer release exists (one plain request, nothing else, no change
to your files). Turn this off with `CRUCIBLE_NO_UPDATE_CHECK=1` or `--no-update-check`. `make update` builds the newest release and runs a
short self-test; it restores the previous release when a step fails. `make rollback` undoes one update. Automatic updates are off
by default. The commands, the timer and the details are in [UPDATES](docs/UPDATES.md).

## Documentation

- [How it got fast](docs/how-it-got-fast.md): the six ideas behind the numbers, in plain words.
- [Results](docs/RESULTS.md): release data, method, hardware limits, history. [Benchmark suite](docs/BENCHMARKS.md).
- [Rivals](docs/RIVALS.md) and [Quality](docs/QUALITY.md). [Changelog](CHANGELOG.md).
- [Measurement details](docs/RESULTS.md#measurement-details) and [hardware limits](docs/CALIBRATION.md).
- [Architecture](docs/ARCHITECTURE.md), [forward specification](docs/FLASHNEXT_FORWARD.md), [EXL3 format](docs/EXL3_FORMAT.md), [build guide](docs/BUILD.md).
- [Third-party terms](docs/THIRD_PARTY.md). Progress report: [https://srv.jaedyn.me/strix-halo-engine-optimization/index.html](https://srv.jaedyn.me/strix-halo-engine-optimization/index.html).

Limits: only the named packs and gfx1151 are qualified. The server has no TLS and no tenant isolation, and one key is the only access control. The conversation
cache is opt-in and sits in host memory unless the disk tier is on. Generated tool arguments can be invalid: the server reports the
error and never runs a tool. A long-context timing is not a quality test.

## Licenses

The engine uses Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE); the ExLlamaV3 notice is in `NOTICE`). **The model weights use
other terms.** Both supported EXL3 packs use the Qwen Community License 1.0, and commercial use has additional conditions. Read
[the model license note](docs/THIRD_PARTY.md#model-weights) before you download or use a pack. This repository does not include the weights.
