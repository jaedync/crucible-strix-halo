# Crucible

An inference engine for EXL3 checkpoints of Qwen3.8-Flash-Next on one AMD Strix Halo APU.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/crucible-hero-dark.png">
  <img src="docs/assets/crucible-hero-light.png" alt="A chart of the decode speed of one session of Crucible with the 3.05 bpw pack against the context length, the median of 8 or 9 runs from Oct 18 to 19, 2026. On code, one session decodes at 71 to 75 tok/s from 4K to 100K. On prose, it decodes at 55 to 58 tok/s. The first token of a cold 130K prompt comes after 80.6 s. When 4K new tokens follow the cached 130K prompt, the first token comes after 4.87 s. The table under the image gives each value.">
</picture>

Crucible runs **EXL3** checkpoints of **Qwen3.8-Flash-Next** on **AMD Strix Halo**:
one **Ryzen AI Max+ 395** with a **Radeon 8060S** (**gfx1151**) and 128 GB of unified memory. It uses its own
**ROCm/HIP** kernels in C++20. It does not use another engine for inference. A local OpenAI-compatible server
gives chat, streamed responses, function calls, and exact speculative decoding. This is research software for
one GPU architecture, not a general model backend or a production service.

[Progress report](https://srv.jaedyn.me/strix-halo-engine-optimization/index.html) · [Measurements](docs/BENCHMARKS.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Build guide](docs/BUILD.md)

## What it is like to use

The first table shows one session on the 3.05 bpw pack. It decodes faster with a smaller model
file. The second table shows the 4.05 bpw pack: one session
decodes at 57.6 tok/s at 32K, the cold prefill at 32K gives 1,561 tok/s,
and four sessions reach 109 tok/s in total. Its next-token accuracy is 0.0038 higher
(p 0.12, not significant at 0.05), and the published KL divergence from BF16 is 0.0066 against 0.0174 ([Quality](#quality)).
Speculative decoding (MTP) is on, as in daily use. A cold prompt has no part in the cache. The server
ran with four sessions and image input on, as in daily use.

The numbers in this section and in the later sections, except [v0.26.0 against v0.22.0](#v0260-against-v0220), are from the v0.22.0 engine, and each table names its dates.

**3.05 bpw pack (the faster option)**

*Engine v0.22.0, measured Oct 18 to 19, 2026.*

| Context | First token, cold prompt | Prefill | Decode, code | Decode, prose |
|---|---:|---:|---:|---:|
| 4K | not measured | not measured | 71 tok/s | 56 tok/s |
| 32K | 20.1 s | 1,638 tok/s | 72 tok/s | 58 tok/s |
| 100K | not measured | not measured | 75 tok/s | 55 tok/s |
| 130K | 80.6 s ¹ | 1,613 tok/s ¹ | not measured | not measured |

All values are the median of 9 runs (Oct 18 to 19, 2026), 8 for the 100K decode cells, with the same engine binary and prompts as the 4.05 runs. A run that did not measure a row does not count for that row. First token and prefill: the median of two cold requests in each run. Decode: one session, from the first to the last token of the code turn and of the prose turn. "not measured": those runs did not measure this cell.\
¹ A prompt of 130,079 tokens.

**4.05 bpw pack (the higher-quality option)**

*Engine v0.22.0, measured Oct 17 to 19, 2026.*

| Context | First token, cold prompt | Prefill | Decode, code | Decode, prose |
|---|---:|---:|---:|---:|
| 4K | not measured | not measured | 65 tok/s | 53 tok/s |
| 32K | 21.0 s | 1,561 tok/s | 58 tok/s | 50 tok/s |
| 100K | not measured | not measured | 58 tok/s | 50 tok/s |
| 130K | 83.9 s ¹ | 1,550 tok/s ¹ | not measured | not measured |

All values are the median of 5 runs (Oct 17 to 19, 2026) with the v0.22.0 build. A run that did not measure a row does not count for that row. First token and prefill: the median of two cold requests in each run. Decode: one session, from the first to the last token of the code turn and of the prose turn. "not measured": those runs did not measure this cell.\
¹ A prompt of 130,080 tokens.

**Four sessions at once:** 132 tok/s in total at 32K and 119 tok/s at 100K on
code, 106 tok/s and 102 tok/s on prose (median of 8 or 9 runs). Each session has its own
prompt start, as agents have. At 32K, each session decodes at 28 tok/s.

**Four sessions with one shared prompt start (the cost case):** all four sessions start with the same new 32K
prompt at the same time. The total is 128 tok/s at 32K, but each session decodes at only
13 tok/s (median). The sessions get their first token after 1.7 s to
65.9 s, because one slot holds the prompt start and each other session prefills it
again (median of 9 runs).

**A follow-up turn in a cached conversation:** when 4K new tokens follow a cached prompt, the first token comes
after 3.13 s at 32K and 4.87 s at 130K. An identical re-send gives the first token
after 0.21 s at 32K and 2.18 s at 130K (median of 9 runs). In real turns with an older build
and the 3.05 bpw pack (Oct 13 to 15), the median first token came after 0.34 s for up to 300 new
tokens, 0.50 s for 300 to 1,000, 1.14 s for 1,000 to 3,000, and
2.22 s for 3,000 to 10,000 new tokens. Eight sessions of that older build gave
148 tok/s in total at 32K (Oct 12).

- **The cache makes later turns fast.** A later turn pays only for its new tokens. In the real turns of the
  older build, that cost was about 0.19 s plus about 1 ms for each new
  token. A long context costs its cold prefill only once. An optional disk tier restored a conversation of
  87,867 tokens in 0.55 s (Oct 15, an older build).
- **Decode changes little with context.** One session on the 3.05 pack decodes at 71 to 75 tok/s on
  code from 4K to 100K, and one session on the 4.05 pack decodes at 58 to 65 tok/s. The rate
  changes little from 32K to 100K. The draft head predicts code better than prose, so prose decodes at
  55 to 58 tok/s on the 3.05 pack and 50 to 53 tok/s on the 4.05 pack.
- **The output is exact.** Speculative decoding changes the speed, not the tokens. Each speed change must pass
  bitwise-equal gates before it merges.
- **Real use.** On one day of agent work (Oct 12, 2,890 turns, an older build), the median time to
  the first token was 0.52 s. One session decoded at 74.9 tok/s (median). The median
  prefill was 1,076 tok/s for turns with 1,024 or more new tokens.

How each value was measured: [measurement details](#measurement-details).

**Repeat runs.** The tables and the image come from repeat runs on Oct 18 to 19, 2026 (3.05 pack) and on
Oct 17 to 19, 2026 (4.05 pack). Each Crucible value is the median of its runs. Values from an earlier run or
an older build name their date and their build.

## v0.26.0 against v0.22.0

Most rows are equal or faster in this release. Decode with 4 sessions on the 4.05 pack is 5% slower, and code decode with 1 session is 2% to 3% slower: see the notes under the tables. The tables below come
from one run of the two engines on one machine, one after the other. The tables in the other sections keep their dates
and are not measured again.

The tables compare v0.22.0 (A) with v0.26.0 (B). The releases v0.22.1 to v0.22.4 use the same engine as v0.22.0. Each cell is the median of 5 pairs. The v0.22.0 cell for 4 sessions on the 4.05 pack is the median of 4 pairs, because one pair gave no value. The ratio is B divided by A. For the rows in tok/s, a ratio above 1
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

How each value was measured: [measurement details](#measurement-details).

## Same-harness comparison with Halogen and Gufo

Crucible values are the median of repeat runs: 5 runs for the 4.05 pack (Oct 17 to 19, 2026) and 9 runs for
the 3.05 pack (Oct 18 to 19, 2026). Halogen 0.16.2 and Gufo 0.9.0 (MTP) ran once, on Oct 17, 2026. All
columns ran on one machine with one request client, one engine at a time. The 3.05 column used the same
engine binary and prompts as the 4.05 column. The Quality section shows the quality difference of each
engine and weight format. The Crucible columns are from the v0.22.0 engine, measured Oct 17 to 19, 2026 (4.05) and Oct 18 to 19, 2026 (3.05). They are not measured again
for v0.26.0.

Engines: Crucible 4.05 (4.05 bpw EXL3, median of 5 runs (Oct 17 to 19, 2026), v0.22.0 build, 4 sessions); Crucible 3.05 (3.05 bpw EXL3, median of 9 runs (Oct 18 to 19, 2026), same engine binary and prompts, 4 sessions, conversation cache 25,600 MiB against 9,700 MiB); Halogen (0.16.2, one run (Oct 17, 2026), 4 sessions); Gufo (0.9.0, one run (Oct 17, 2026), GGUF UD-Q4_K_XL, MTP).

**Prefill and first token.** Time to the first token of one request with 4 output tokens, the median of the runs. A re-send repeats the last cold prompt. An extension adds about 4K tokens of other source to it.

| | Crucible 4.05 | Crucible 3.05 | Halogen | Gufo |
|---|---:|---:|---:|---:|
| Cold 32K prompt | 21.04 s (1561 tok/s) | 20.06 s (1638 tok/s) | 18.08 s (1817 tok/s) | 24.19 s (1358 tok/s) |
| Re-send of the 32K prompt | 0.25 s (32768 of 32846 cached) | 0.21 s (32768 of 32847 cached) | 0.04 s (32845 of 32845 cached) | 0.02 s (32845 of 32845 cached) |
| 32K prompt + 4K new tokens | 3.31 s (32768 of 36625 cached) | 3.13 s (32768 of 36626 cached) | 2.65 s (32832 of 36624 cached) | 8.87 s (24576 of 36624 cached) |
| Cold 130K prompt | 83.90 s (1550 tok/s) | 80.63 s (1613 tok/s) | 73.50 s (1770 tok/s) | 98.39 s (1322 tok/s) |
| Re-send of the 130K prompt | 2.32 s (126976 of 130079 cached) | 2.18 s (126976 of 130080 cached) | 0.18 s (130079 of 130079 cached) | 0.05 s (130081 of 130081 cached) |
| 130K prompt + 4K new tokens | 5.15 s (126976 of 133858 cached) | 4.87 s (126976 of 133859 cached) | 3.06 s (130048 of 133858 cached) | 4.40 s (129024 of 133860 cached) |

**Decode, code turn.** Tokens per second: each session (median) / total while all sessions decode. Each session has its own prompt start, as agents have. The cost row is the case where all four sessions share one new long prompt start.

| | Crucible 4.05 | Crucible 3.05 | Halogen | Gufo |
|---|---:|---:|---:|---:|
| 4K context, 1 session | 64.9 / 64.9 | 70.7 / 70.7 | 57.2 / 57.2 | 34.8 / 34.8 |
| 4K context, 4 sessions | 25.6 / 109.0 | 29.0 / 128.2 | 19.7 / 78.5 | 19.1 / 77.3 |
| 32K context, 1 session | 57.6 / 57.6 | 72.0 / 72.0 | 54.8 / 54.8 | 35.8 / 35.8 |
| 32K context, 4 sessions | 24.1 / 109.0 | 28.5 / 131.8 | 19.5 / 78.9 | 16.9 / 66.7 |
| 32K context, 4 sessions, one shared prompt start (cost row) | 11.4 / 105.5; first token of each session: 1.9, 69.4, 23.9, 46.5 s | 12.7 / 128.4; first token of each session: 65.9, 1.7, 22.7, 44.3 s | 19.4 / 77.4; first token of each session: 0.34, 0.29, 0.23, 0.39 s (Oct 18) | 17.5 / 69.5; first token of each session: 0.04, 0.04, 0.04, 0.04 s (Oct 18) |
| 100K context, 1 session | 57.6 / 57.6 | 75.3 / 75.3 | 55.0 / 55.0 | 36.7 / 36.7 |
| 100K context, 4 sessions | 22.4 / 96.9 | 27.0 / 119.5 | 6.1 / 76.1 | not run |

**Decode, mean of four turns.** Tokens per second of each session: the mean of the code, sampled code, thinking and prose turns.

| | Crucible 4.05 | Crucible 3.05 | Halogen | Gufo |
|---|---:|---:|---:|---:|
| 4K context, 1 session | 61.4 | 66.3 | 50.8 | 35.3 |
| 4K context, 4 sessions | 25.1 | 29.4 | 19.7 | 17.7 |
| 32K context, 1 session | 57.5 | 63.9 | 50.4 | 37.2 |
| 32K context, 4 sessions | 24.1 | 28.3 | 19.0 | 16.7 |
| 100K context, 1 session | 56.2 | 67.6 | 50.7 | 35.6 |
| 100K context, 4 sessions | 22.0 | 26.3 | 10.9 | not run |

**Other.**

| | Crucible 4.05 | Crucible 3.05 | Halogen | Gufo |
|---|---:|---:|---:|---:|
| 4 sessions extend a 32K prompt: tokens / wall time | 2048 / 28.9 s = 70.8 tok/s | 2048 / 25.7 s = 79.6 tok/s | 2048 / 35.7 s = 57.4 tok/s | 2048 / 69.2 s = 29.6 tok/s |
| 4 sessions extend a 32K prompt: first token of the extension | 7.64 s | 7.17 s | 6.61 s | 40.18 s |
| First token after one 896 x 896 image | 0.93 s (846 prompt tokens) | 0.86 s (846 prompt tokens) | 1.45 s (846 prompt tokens) | not run |
| Mean GPU clock / package power while busy | 2454 MHz / 97.4 W | 2406 MHz / 98.0 W | 2498 MHz / 98.3 W | 2481 MHz / 96.8 W |

**Thank you to the authors of Halogen and of Gufo.** Their engines run on the same hardware with
their own weight formats, and they made this comparison possible.

Where each engine is faster in these runs:

- **Halogen** has the fastest cold prefill: 18.08 s at 32K and 73.50 s at 130K, against 20.06 s and 80.63 s
  for Crucible 3.05, 21.04 s and 83.90 s for Crucible 4.05, and 24.19 s and 98.39 s for Gufo. Halogen also
  gives the first token sooner than Crucible after a re-send, after the 32K and 130K extension of a cached
  prompt (2.65 s at 32K and 3.06 s at 130K, against 3.13 s and 4.87 s for Crucible 3.05, and 3.31 s and
  5.15 s for Crucible 4.05), and after the extension in the four-session test (6.61 s, against 7.17 s for
  Crucible 3.05 and 7.64 s for Crucible 4.05).
- **Gufo** gives the fastest re-send of a cached prompt. After an extension of the cached 130K prompt, Gufo
  gives the first token sooner than Crucible.
- **Halogen and Gufo** share one cached prompt among sessions. When four sessions start with the same new 32K
  prompt at the same time, each session gives its first token in 0.4 s or less, and then decodes at 19.4 tok/s
  (Halogen) or 17.5 tok/s (Gufo). Crucible keeps the cached prompt in one session slot, so the other three
  sessions prefill it again: Crucible 4.05 decodes at 11.4 tok/s and its last first token comes after 69.4 s;
  Crucible 3.05 decodes at 12.7 tok/s and its last first token comes after 65.9 s.
- **Crucible 3.05** has the highest decode rate in each decode row where each session has its own prompt
  start: 70.7 tok/s at 4K, 72.0 tok/s at 32K and 75.3 tok/s at 100K for one session, against 64.9, 57.6 and
  57.6 tok/s for Crucible 4.05. Crucible 3.05 also has the shortest wall time when four sessions extend a
  32K prompt (79.6 tok/s, against 70.8 tok/s for Crucible 4.05, 57.4 tok/s for Halogen and 29.6 tok/s for
  Gufo). After an image, it gives the first token sooner than the other columns (0.86 s, against 0.93 s for
  Crucible 4.05 and 1.45 s for Halogen). Gufo did not run the image test. Crucible 4.05 comes second in each
  of these decode rows, in this wall time and in the image test.

The headline tables above show both Crucible packs: the 3.05 pack, which decodes faster with a smaller
model file, and the 4.05 pack, the higher-quality option.

How the comparison was made: one machine, one request client for all four columns, one engine at a time,
a fresh engine for each block, and no other work on the GPU. Each Crucible value is the median of its runs;
Halogen and Gufo ran once. The prompts are Python standard-library source files in a coding-agent
conversation, and generated test images. They hold no user text. Each engine runs its own weights and the
image or the flags that its authors publish. The engine and the weight format differ, so a rate difference
is not an engine-only comparison. Halogen ran at 2% to 4% higher mean GPU clock than Crucible (2,498 MHz
against 2,454 MHz for 4.05 and 2,406 MHz for 3.05). Gufo runs one session in the prefill and one-session
blocks, and four sessions in the four-session blocks. The client of these runs is not part of this tree yet.
`tools/h2h_decode.py` is the client of the earlier runs.

## Quality

This table shows how much next-token quality each engine and weight format keeps on one fixed public
text set. All rows use the same text, the same tokenizer and the same scoring method, and each engine is
scored against the real text. Crucible 4.05 of 2026-10-10 is only the zero point of the differences. The
table after this one measures the distance from the original 16-bit (BF16) model.

The differences against Crucible 4.05 are small: all are under 0.6 percentage points of next-token
accuracy. Halogen 0.16.2 and Gufo 0.9.0 are lower with p 0.039. Crucible 3.05 is lower with p 0.12, which
is not significant at 0.05.

| Engine and weights | Date | Positions | Next-token accuracy (95%) | Accuracy difference (zero point: Crucible 4.05) | McNemar p | Matched NLL difference (zero point: Crucible 4.05) |
|---|---|---:|---|---|---:|---|
| Halogen 0.16.2 (.hgn v2, 4.16 bpw) | 2026-10-10 | 7,168 | 0.7095 (0.6609 to 0.7554) | -0.0050 (-0.0100 to -0.0010) | 0.039 | +0.017 (+0.006 to +0.030) |
| Crucible 3.05 (EXL3 3.05 bpw) | 2026-10-10 | 7,168 | 0.7108 (0.6599 to 0.7578) | -0.0038 (-0.0073 to +0.0000) | 0.12 | +0.015 (+0.007 to +0.022) |
| Crucible 4.05 (EXL3 4.05 bpw) | 2026-10-10 | 7,168 | 0.7146 (0.6646 to 0.7626) | zero point |  | zero point |
| Gufo 0.9.0 (GGUF UD-Q4_K_XL) | 2026-10-18 | 7,168 | 0.7102 (0.6610 to 0.7563) | -0.0043 (-0.0080 to -0.0006) | 0.039 | not available (no log-probabilities) |
| Crucible 4.05, repeat | 2026-10-18 | 7,168 | 0.7146 (0.6646 to 0.7626) | +0.0000 (+0.0000 to +0.0000) | 1 | +0.000 (+0.000 to +0.000) |

Method:

- Text: 16 chunks of 4,096 tokens. 8 chunks are prose from the wikitext-103 test set. 8 chunks are
  public source code files.
- Positions: every 8th token from token 512 on, 7,168 positions for each engine (3,584 prose, 3,584 code).
- Each position is one request. The prompt is the text before the position. The engine gives 1 greedy
  token (temperature 0). The prompt cache is on. Speculative decoding is off.
- Every request had a prompt token count equal to the position. A request with another count is not used.
- Next-token accuracy: the part of the positions where the greedy token is the real next token of the
  text.
- Matched NLL: the mean negative log-probability of the real next token, on the positions where both
  engines have the real token in their top 20 log-probabilities. Lower is better. The value is the row
  minus Crucible 4.05.
- Both values score each engine against the real text. Crucible 4.05 is only the zero point of the
  differences, not a standard for correct output.
- 95% intervals: 1,000 bootstrap resamples of the 16 chunks. McNemar p: an exact two-sided test on the
  positions where only one of the two engines is correct.

Sources for each row:

- Halogen 0.16.2 (.hgn v2, 4.16 bpw) (2026-10-10): image id 21131b0dc0de, the same as the release speed run; qwen38-flash-next-v2.hgn sha256 71246c6ab3fc1de2, the same as the release speed run; the n-gram and vision files were not hashed on 2026-10-10, but they have the same file dates (2026-09-29) as in the release speed run (n-gram sha256 3450acada94e19aa)
- Crucible 3.05 (EXL3 3.05 bpw) (2026-10-10): Crucible binary sha256 90cd28b0 (opt-in top_logprobs build, source 465a004e); checkpoint set sha256 c9084de3cdba083b (release speed run set; the files are dated 2026-10-01, before this run)
- Crucible 4.05 (EXL3 4.05 bpw) (2026-10-10): Crucible binary sha256 90cd28b0 (opt-in top_logprobs build, source 465a004e); checkpoint set sha256 5276585cc5c9139e (release speed run set; the files are dated 2026-10-01, before this run)
- Gufo 0.9.0 (GGUF UD-Q4_K_XL) (2026-10-18): image id 9e0a33461493, the same as the release speed run; model file set sha256 aaa547a30746994c (the 4 GGUF shards of the release speed run; no MTP file); speculation off (--speculative off), greedy (temperature 0)
- Crucible 4.05, repeat (2026-10-18): Crucible binary sha256 90cd28b0 (opt-in top_logprobs build, source 465a004e); checkpoint set sha256 5276585cc5c9139e

What the table can show:

- The difference in next-token quality between each row and Crucible 4.05, on this text set, with one
  harness.
- The repeat row shows if the harness gives the same numbers again on a different day.

What the table cannot show:

- Quality on long contexts (the prompts have 512 to 4,095 tokens), on chat or tool tasks, or with
  thinking.
- Gufo 0.9.0 does not return log-probabilities, so its row has no NLL value. Its accuracy compares the
  returned text with the real token. A position where the real token (or the Crucible 4.05 token) is not
  complete UTF-8 text is not used for that value.
- 16 chunks give wide absolute intervals. The paired differences against Crucible 4.05 are more precise.
- Speed. This table does not measure speed.

### Measured distance from the 16-bit model

This table measures the distance from the original 16-bit (BF16) model. Lower is closer. Measured on
2026-10-20. The reference is the BF16 model itself, run in a CPU forward pass written for this
measurement.

| Engine | Mean KL divergence (lower bound) | Median | p99 | Top-1 agreement with BF16 | Matched NLL |
|---|---:|---:|---:|---:|---:|
| Crucible 4.05 | 0.02291 | 0.00383 | 0.26894 | 94.34% | 0.8134 |
| Crucible 3.05 | 0.05015 | 0.00936 | 0.58872 | 92.05% | 0.8288 |
| Halogen 0.16.2 | 0.06158 | 0.01231 | 0.77951 | 90.96% | 0.8313 |
| Gufo 0.9.0 | - | - | - | 94.29% | - |
| BF16 reference | - | - | - | - | 0.8152 |

- Gufo 0.9.0 does not return log-probabilities, so its row gives top-1 agreement only, on 7,166 rows.
- Crucible 4.05 has a lower matched NLL than the BF16 reference (0.8134 against 0.8152). NLL measures
  how well a model predicts this text, not how close it is to BF16, so a small difference in either
  direction can come by chance.
- These values are not comparable with the published values below. The published values use other text,
  the full vocabulary of the model and another tool.

Method:

- Reference: a CPU forward pass of the BF16 model, with BF16 weights, fp32 activations and fp64
  log-sum-exp. It gives the reference probability of each token id.
- Positions: the same 7,168 positions as the table above, with the same tokenizer.
- KL(BF16 || engine) over 21 buckets for each position. The buckets are the engine's 20 best token ids
  and one rest bucket. A rest at or below zero takes 1e-10 and its row is counted. The log-sum
  inequality makes this value a lower bound of the full-vocabulary KL.
- Top-1 agreement: the part of the positions where the engine's best token id equals the BF16 best token
  id.
- Matched NLL: the mean negative log-probability of the real next token, on the 6,711 positions where all
  three engines that return log-probabilities give the real token. Lower is better. The BF16 reference
  gives 0.8152 on the same positions.

Source: our scoring run of 2026-10-20 against the BF16 reference (result sha256 e4c368f655a02fe4); Gufo has 7,166 rows because it returns no log-probabilities.

### Published distance from the 16-bit model

The KL divergence against the original 16-bit (BF16) model shows how far a weight file is from the
original. Lower is closer. The maker of the EXL3 files published one measurement with one tool and one
text for three of the four weight files in this comparison.

| Weight file | Used by | Mean KL divergence from BF16 |
|---|---|---:|
| EXL3 4.05 bpw H6 NG6 | Crucible 4.05 | 0.0066 |
| GGUF UD-Q4_K_XL | Gufo 0.9.0 | 0.0084 |
| EXL3 3.05 bpw H5 NG5 | Crucible 3.05 | 0.0174 |
| Halogen .hgn v2, 4.16 bpw | Halogen 0.16.2 | not published |

- Tool: exllamav3 qbench. Text: a self-generated in-domain trace, 14,496 input and 45,588 output tokens.
  Noise floor: 0.00262.
- Source: the model card of [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) (chart qb_kld), read on 2026-10-20.
- These values measure the weight files in the reference code of exllamav3, not in the engines of this
  comparison. An engine can add small differences of its own.
- The Halogen model card publishes no measurement against BF16.

## What it is

- One engine for one model family and one GPU architecture. No fallback path.
- **Exact speculative decoding.** The target verifies every draft. A rejected suffix never becomes
  committed state.
- **A keyed sampling law.** A seeded request gives the same IDs at any speculation depth, and in a
  packed batch pass.
- **Measured hardware ceilings.** Most speed claims name the ceiling and the percentage of it.
  Speculative decoding and batching have no single ceiling.
- A local OpenAI-compatible HTTP frontend with chat, streaming, tools, and up to eight sessions on
  one engine.

## Quick start

### 1. Build

```sh
make
```

The build needs the [software below](#hardware-and-software). It writes `build/engine/crucible`.

### 2. Download the model

Read the [model license note](#model-license) first.

```sh
PACK=3.05bpw_h5_ng5
CKPT="$PWD/models/Qwen3.8-Flash-Next-exl3-$PACK"
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision "$PACK" --local-dir "$CKPT"
```

The supported packs are in [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3).
Use revision `4.05bpw_h6_ng6` for the quality pack. The numbers at the top of this page come from that pack.
Keep every tokenizer file, weight shard, n-gram table, and MTP mixer patch.

### 3. Run

```sh
python -m server "$CKPT" --engine build/engine/crucible --ctx 32768
```

The server binds to `127.0.0.1:8790` by default. `--host ADDRESS` selects another address and prints a warning when it is
not loopback. It has no authentication. Do not expose it to a network.
Send a request from another terminal:

```sh
curl -sS http://127.0.0.1:8790/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"crucible","messages":[{"role":"user","content":"Say hello."}],
       "temperature":0,"max_tokens":32}'
```

The API provides `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/tokenize`.
Use `stream: true` for server-sent events. Add OpenAI `tools` definitions for function calls.
See [the server guide](server/README.md) for the options and the limits.

For direct generation without the server:

```sh
python tools/tok.py "$CKPT" encode "Write a Python function that adds two numbers." --chat > prompt.ids
build/engine/crucible gen "$CKPT" --ids-file prompt.ids --n 160 --spec 2 --ctx 4096
```

## Updates

Each release has a number `v0.<train>.<fix>`. The release notes are in [CHANGELOG.md](CHANGELOG.md) and on the
Releases page of this repository on GitHub. The file `VERSION` has the number of your
checkout. The server shows it at start, in `GET /v1/crucible/version`, and in `GET /v1/models`
(`crucible_version`). `build/engine/crucible --version` shows the number of the engine build.

**Watch for releases.** On GitHub, select Watch, then Custom, then Releases. GitHub sends you a message for each
new release.

**Start notice.** The server asks GitHub once in 24 hours if a newer release exists. It sends one plain request and
nothing else. If a newer release exists, the server prints two lines at start:

```text
Crucible v0.26.1 is available (you run v0.26.0). Update: make update. Release notes: <link>
Automatic updates are off unless you turned on the timer. See "Updates" in the README.
```

The notice never changes anything. A failed check prints nothing. To turn the check off, set
`CRUCIBLE_NO_UPDATE_CHECK=1` or start the server with `--no-update-check`.

**Update by hand.** Run this in your clone:

```sh
make update
```

The command gets the newest release tag, builds it, and runs a short self-test. It stops and changes nothing when
you have local changes, or when your checkout is not on the release line. If the build or the self-test fails, the
command restores the previous release by itself. It keeps the previous binary as `build/engine/crucible.prev`.

A kill or a power loss can stop an update half way. The next `make update` or `make rollback` finds the file
`build/update-running` and restores the previous release and its binary first, and it says so. It does not build again:
a build can be stopped again by the same cause, and a restore cannot. The linker writes a temporary file and renames it,
so the binary is never half written: a stopped build leaves the old binary or none. Until that next run, the checkout
can be on the new release and the binary on the old one. After a stopped update, run `make update` before you start
the server.

`make update` does not restart your server. Stop the server and start it again to run the new release.

`make update` leaves your clone on the release tag (a detached HEAD). A later `git pull` then fails. To get back on
a branch, run `git checkout main && git pull`. To follow the releases again, run `make update`.

To undo the last update, run:

```sh
make rollback
```

`make rollback` undoes one update, once, and needs a clean tree. It then holds the release that you left. `make update`
and the timer skip that release until a newer one exists. To install the held release anyway, delete the file
`build/update-hold` and run `make update`.

The self-test checks the CPU part only. To add a short GPU test (a few greedy tokens against the stored token IDs),
stop the server and run `make update CKPT="$CKPT"`. This release stores token IDs for the 3.05 and 4.05 packs. A pack
with no stored token IDs is skipped, and the command says so.

**Automatic updates (off by default).** Nothing in this repository turns them on. To turn them on, run these three
commands in your clone. They install a systemd user timer that runs `make update` once a day at about 04:00:

```sh
mkdir -p ~/.config/systemd/user
for f in crucible-update.service crucible-update.timer; do sed "s|@CHECKOUT@|$PWD|" deploy/user-systemd/$f > ~/.config/systemd/user/$f; done
systemctl --user enable --now crucible-update.timer
```

A user timer runs only while you are logged in. To let it run when you are logged out, run
`loginctl enable-linger "$USER"` once.

The build runs next to your loaded model. A build with one job peaked at 1.4 GB of memory in a test. The service
file limits the build to 4 GB (`MemoryMax`) and tells the kernel to stop the build first if memory runs out
(`OOMScoreAdjust`). In the same test the build took 16 minutes.

The 4 GB limit covers the split build, which is the default. If that build cannot run, the build falls back to the plain
build, which can need more memory. If the plain build hits the limit, the kernel stops it, `make update` restores the
previous release (`OOMPolicy=continue` in the service file lets it do so), and the timer tries again the next night.

To turn them off:

```sh
systemctl --user disable --now crucible-update.timer
```

After a good update, the timer job restarts the user unit `crucible-server.service` if that unit runs. A restart
stops the requests that are in progress. The folder `deploy/user-systemd/` has a template for that unit. In the
template, replace `/usr/bin/python3` with the `python` of your virtual environment if you installed
`requirements-cpu.txt` in one.
Use no restart if you start the server in another way: delete the `CRUCIBLE_RESTART_UNIT` line in the installed
service file. The timer job does not run the GPU self-test.

## Hardware and software

- Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB shared LPDDR5X-8000.
- Linux. The tested system is Arch Linux with kernel 7.2.7 and ROCm 7.2.4.
- A C++20 HIP compiler, GNU Make, and POSIX threads. The CPU test suite needs
  GCC 14 or newer, or Clang 18 or newer.
- Python 3.12 or newer. The CPU dependencies are in `requirements-cpu.txt`.
- Enough disk space for one complete checkpoint and the build output.

The 3.05 bits-per-weight payload alone is about 84.45 GB. Use one model instance. An ordinary
discrete desktop GPU does not have enough memory for it.
The 4.05 bits-per-weight pack needs more memory than the 3.05 pack.
Set `CRUCIBLE_PLE_HOST=1` to keep the PLE n-gram table in host memory, at 32.6 GB (3.05) or
39.0 GB (4.05).

## Hardware ceilings

The test system has a Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB shared LPDDR5X-8000.
Its measured ceilings are **245.5 GB/s cold reads** and **48.2 TFLOPS f16 WMMA**.
See [chip calibration](docs/CALIBRATION.md).

These rows are engine benchmarks measured on 2026-10-03 and 2026-10-04. They are not measured again for v0.26.0. They compare the engine with the hardware.
**Serial decode** makes one token for each forward pass, without speculative decoding. Daily use has
speculative decoding on: see [What it is like to use](#what-it-is-like-to-use).

| Workload | Measured rate | Hardware ceiling | Percent of ceiling | Runs | Date |
|---|---:|---:|---:|---:|---|
| Serial decode, 3.05 bpw | 38.79 tok/s | 53.19 tok/s | 72.9% | 2 | 2026-10-04 |
| Serial decode, 4.05 bpw | 33.71 tok/s | 45.84 tok/s | 73.5% | 2 | 2026-10-04 |
| MTP, two drafts, 3.05 bpw | 68.02 tok/s | acceptance-dependent | n/a | 1 | 2026-10-03 |
| 4,096-token prefill, 3.05 bpw | 1,566.2 tok/s | 2,310 tok/s | 67.8% | 2 | 2026-10-04 |
| 16,384-token prefill, 3.05 bpw | 1,526.9 tok/s | 2,310 tok/s | 66.1% | 2 | 2026-10-04 |
| 4,096-token prefill, 4.05 bpw | 1,511.3 tok/s | 2,289 tok/s | 66.0% | 2 | 2026-10-04 |
| 16,384-token prefill, 4.05 bpw | 1,493.1 tok/s | 2,289 tok/s | 65.2% | 2 | 2026-10-04 |
| Eight streams in total, 3.05 bpw | 104.24 tok/s | no single ceiling | n/a | 1 | 2026-10-03 |
| Eight streams in total, 4.05 bpw | 90.83 tok/s | no single ceiling | n/a | 1 | 2026-10-03 |

The serial ceiling is the weight-read ceiling: 4.615 GB of logical weight reads for each token at
3.05 bits per weight, and 5.356 GB at 4.05 bits per weight.
The prefill ceiling is an estimate for a 4,096-token prompt with EXL3 trellis decode in the current
data path.
The percentage is a model ratio. It is not a measured GPU utilization.
Prefill timing ends at the first target ID. It excludes model load, tokenization, queue time, and
HTTP transport.

## Measurement details

- **The tables at the top:** the median of 9 runs (3.05, Oct 18 to 19, 2026) and of 5 runs (4.05, Oct 17 to
  19, 2026) with the v0.22.0 build, and the daily server settings (four sessions, image input on).
  First token and prefill are the median of two cold requests in each run. Prefill tok/s is the prompt tokens
  divided by the time to the first token, so it includes the server path. Decode is one session in a
  five-turn agent conversation (code, tool, sampled code, thinking, prose). The code and prose columns are
  the tokens per second of those turns, from the first to the last token. The runs measured cold prefill at
  32K and 130K, and decode at 4K, 32K and 100K.
- **v0.26.0 against v0.22.0:** each run starts its own server with the server code of its own release, with 4 sessions
  of 262,144 tokens, image input on and MTP on (off for the serial row). The conversation cache is 9,700 MiB for the
  4.05 pack and 25,600 MiB for the 3.05 pack. The prompts are one synthetic coding conversation, with CPython library
  files as tool results. Both engines get the same text. Every long prompt has its own first line, so no cache has seen
  it. Serial decode: one code prompt, 257 tokens, the median of 2 runs. Cold prefill at 32K: the median of 5 cold
  prompts. Cold prefill at 130K: one cold prompt. Prefill speed is the prompt tokens divided by the time to the first
  token. Decode at 32K with 1 session: the code turn and the prose turn that follow the cold prompt, as the tokens
  minus 1, divided by the time from the first to the last chunk. Decode with 4 sessions: 4 sessions with different
  prompt starts, one code turn at the same time, in the window in which all 4 decode. Re-send: the same 32K request
  sent again after the cold one, the median of 3. Each pack ran 5 pairs. A pair is one run of each engine, one after
  the other. The pairs run in the order AB, BA, AB, BA, AB. A pair was run again when its GPU clock median was below
  2,200 MHz or its disk wait was high. Each cell is the median of the pairs.
- **Four sessions:** the total rate while all four sessions decode at the same time, and the median rate of each
  session.
- **Follow-up turns:** the runs send a re-send and a 4K extension of each cold prompt. The older values
  are medians of real turns in cached conversations, from the request log of the daily engine (2026-10-13 to
  15), by the number of new tokens. The fit of about 0.19 s plus about
  1 ms for each new token is for the engine prefill time of these turns.
- **Real use:** the turn statistics of the daily engine for 2026-10-12. They hold numbers only, no text.
- **Prefill in this source:** this source adds a faster MoE kernel pair for the 3.05 bpw pack. In A/B pairs on
  2026-10-14, with the engine profiler on, prefill went from 1,595 tok/s to 1,630 tok/s at
  32K, and from 1,547 tok/s to 1,580 tok/s at 128K. The A/B pairs measure this gain
  alone; the tables at the top include it with the other changes.
- **The 3.05 bpw pack** (less memory, fewer weight bytes for each token): the first table at the top gives the
  median of 9 runs. In the older decode test of 2026-10-13, with an older build, one session decoded at 77 tok/s to
  79 tok/s on code and 60 tok/s to 67 tok/s on prose, from 1K to
  128K. Four sessions at 32K gave 130 tok/s in total on code. That test used a single
  synthetic request and not an agent conversation, so its rates are not comparable with the table at the top.
  The draft head gave 3.3 to 3.6 accepted tokens for each forward pass on
  code, and 2.4 to 2.7 on prose. In the head-to-head run of 2026-10-14,
  the first token of a cold prompt came after 20.4 s at 32K and 82.8 s at
  130K.

**Run-to-run spread (Crucible).** Engine v0.22.0, measured Oct 17 to 19, 2026 (4.05) and Oct 18 to 19, 2026 (3.05). Each value is the median of the runs that measured the row. The range is
the lowest to the highest run, with the number of runs.

| Row | Crucible 4.05 | Crucible 3.05 |
|---|---|---|
| Cold 32K, first token (s) | 21.04 (20.67 to 21.29; n 5) | 20.06 (19.91 to 20.22; n 9) |
| Cold 32K, prefill (tok/s) | 1,561 (1,543 to 1,589; n 5) | 1,638 (1,625 to 1,650; n 9) |
| Cold 130K, first token (s) | 83.90 (83.69 to 84.66; n 5) | 80.63 (80.36 to 81.18; n 9) |
| Cold 130K, prefill (tok/s) | 1,550 (1,536 to 1,554; n 5) | 1,613 (1,602 to 1,619; n 9) |
| Decode 4K, 1 session, code (tok/s) | 64.9 (64.6 to 65.0; n 5) | 70.7 (70.6 to 70.8; n 9) |
| Decode 32K, 1 session, code (tok/s) | 57.6 (57.5 to 57.8; n 5) | 72.0 (71.7 to 72.1; n 9) |
| Decode 100K, 1 session, code (tok/s) | 57.6 (56.8 to 57.7; n 5) | 75.3 (74.9 to 75.5; n 8) |
| Decode 4K, 1 session, prose (tok/s) | 52.7 (52.4 to 52.7; n 5) | 56.5 (56.4 to 56.5; n 9) |
| Decode 32K, 1 session, prose (tok/s) | 50.2 (49.9 to 50.2; n 5) | 58.4 (58.3 to 58.5; n 9) |
| Decode 100K, 1 session, prose (tok/s) | 49.6 (49.1 to 49.8; n 5) | 55.0 (54.9 to 55.1; n 8) |
| Decode 4K, 4 sessions, each / total (tok/s) | 25.6 / 109.0 (24.5 to 26.5 / 107.6 to 112.1; n 5) | 29.0 / 128.2 (27.5 to 30.8 / 125.0 to 134.4; n 9) |
| Decode 32K, 4 sessions, each / total (tok/s) | 24.1 / 109.0 (23.9 to 25.1 / 106.2 to 113.4; n 5) | 28.5 / 131.8 (27.5 to 29.2 / 128.6 to 135.8; n 9) |
| Decode 100K, 4 sessions, each / total (tok/s) | 22.4 / 96.9 (21.7 to 23.1 / 95.5 to 101.4; n 5) | 27.0 / 119.5 (25.8 to 27.8 / 116.7 to 122.4; n 8) |
| 32K, 4 sessions, shared prompt start, each / total (tok/s) | 11.4 / 105.5 (11.3 to 11.6 / 101.5 to 110.0; n 5) | 12.7 / 128.4 (12.6 to 12.9 / 122.9 to 131.9; n 9) |
| 4 sessions extend a 32K prompt, tokens / wall (tok/s) | 70.8 (68.9 to 71.1; n 5) | 79.6 (76.4 to 82.7; n 9) |
| 4 sessions extend a 32K prompt, first token (s) | 7.64 (7.59 to 8.38; n 5) | 7.17 (7.12 to 7.92; n 9) |
| 32K + 4K new tokens, first token (s) | 3.31 (2.70 to 3.34; n 5) | 3.13 (2.53 to 3.22; n 9) |
| 130K + 4K new tokens, first token (s) | 5.15 (5.15 to 5.19; n 5) | 4.87 (4.84 to 4.92; n 9) |
| Image first, first token (s) | 0.93 (0.93 to 0.94; n 5) | 0.86 (0.86 to 0.87; n 9) |

At 32K + 4K new tokens, the Crucible runs fall into two groups, about 0.6 s apart. 4 of the 14 runs
(2 of 5 for Crucible 4.05 and 2 of 9 for Crucible 3.05) give the first token after about 2.5 to 2.7 s.
The other 10 runs (3 of 5 and 7 of 9) give it after about 3.1 to 3.3 s. The cause is the GPU clock. In
the 5 s pause between requests, the clock falls to about 600 MHz. At the next request, it rises to full
speed in 2 to 3 s, and in some requests the rise starts up to 0.7 s later. The cache use and the work
are the same in both groups. The clock behavior comes from the power settings of the machine, so it is
the same for all engines. Halogen and Gufo ran once, so their value can be in either group. The same
clock effect explains part of the spread of the cold 32K first token.

## Model license

The engine uses Apache-2.0. **The model weights use other terms.**
Both supported EXL3 packs use the Qwen Community License 1.0.
Commercial use has additional conditions.
Read [the model license note](docs/THIRD_PARTY.md#model-weights) before you download or use a pack.
The license covers the model artifacts. It does not cover this engine. This engine does not include
the weights.

## Tests

```sh
make -j2 test-cpu
```

The CPU suite builds the test and benchmark programs and runs the offline tests. It needs no GPU.
GPU checks are manual. They need the qualified machine, both checkpoints, and the gate tool
`tools/gates.py`.
See [all build and test targets](docs/BUILD.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [the Flash-Next forward specification](docs/FLASHNEXT_FORWARD.md)
- [EXL3 format](docs/EXL3_FORMAT.md) and [checkpoint metadata](docs/EXL3_CHECKPOINT_METADATA.json)
- [Build and test guide](docs/BUILD.md)
- [Calibration](docs/CALIBRATION.md) and [benchmark history](docs/BENCHMARKS.md)
- [Exactness rules and gates](docs/ARCHITECTURE.md#exactness-rules)
- [Server guide](server/README.md), [OpenAI clients](docs/OPENAI_CLIENTS.md) and [session replay](docs/SESSION_REPLAY.md)
- [Third-party terms](docs/THIRD_PARTY.md)
- Progress report: [https://srv.jaedyn.me/strix-halo-engine-optimization/index.html](https://srv.jaedyn.me/strix-halo-engine-optimization/index.html)

## Limitations

- Only the named model packs and gfx1151 are qualified.
- Image input is opt-in (`--vision`). Video is not supported. There is no general model loader.
- The HTTP frontend has no authentication, no TLS, and no tenant isolation. Keep it on loopback.
- The conversation cache is opt-in. It stays in host memory, and a restart clears it, unless the optional
  disk tier (`--state-dir`) keeps it.
- Long contexts need a large amount of memory. A long-context timing is not a quality test.
- Generated tool arguments can be invalid. The server reports the error. It never runs a tool.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
The ExLlamaV3 notice is in `NOTICE`.
