# Long context

Long context is opt-in: the default configuration does not use it.
This page has the configurations, the measured rows, the costs and the limits. The short form is in the
[README](../README.md#configurations). The method of the other speed numbers is in [Results](RESULTS.md).

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
| Decode, all five workloads, with drafts (tok/s) | not measured: 3 of the 5 replies stopped after 12 tokens | 52.6 over 2,366 tokens (code 59.8, tool 100.2, code sampled 46.1, think 50.5, prose 45.1) |

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

More limits of this first version are in the [Changelog](../CHANGELOG.md) (v0.37.0, "Long context").
The setup steps are in the [setup guide](QUICKSTART.md#option-a-long-context-yarn).
