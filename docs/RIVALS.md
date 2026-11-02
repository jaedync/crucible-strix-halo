# Rivals

Our own measurements of Crucible, Halogen and Gufo on one machine. We did not copy any number from another project.
The README has a short table. This page has all rows, the packs, the dates and the method.
Where an engine leads, the table shows it.

**Dates and versions.** All rival values on this page were measured Oct 23 to 26, 2026, on Halogen 0.18.14 and Gufo 0.22.0.
Newer versions exist: Halogen 0.18.18 and Gufo v0.27.0 (both Oct 29, 2026). We did not measure them. We did not measure the
rivals again for Crucible v0.31.0. The Crucible values in these tables are from the builds named in each section (v0.26.0 or an
internal build of Oct 25), not from v0.31.0. Gufo v0.27.0 serves log-probabilities; Gufo 0.22.0 did not.

Contents: [the run of Oct 23 to 24](#the-run-of-oct-23-to-24-2026) · [the power profile run of Oct 26](#the-power-profile-run-of-oct-26-2026) ·
[the follow-up turn](#the-follow-up-turn-single-runs-oct-25-to-26-2026) · [the packs](#what-differs-between-the-engines) ·
[method](#method) · [older comparison](#history-the-comparison-of-oct-17-to-19-2026)

## The run of Oct 23 to 24, 2026

Halogen 0.18.14 (its own `.hgn` pack, 4.16 bits per weight), Gufo 0.22.0 (GGUF UD-Q4_K_XL), and Crucible with the 4.05 pack
(row D2b uses the 3.05 pack). Each value is the median of 2 rounds on the same machine, with the same prompts, the same
client and the balanced power profile. Bold marks the fastest engine in the row. Rates are tok/s.

| Row | Halogen 0.18.14 | Gufo 0.22.0 | Crucible | Crucible / Halogen |
|---|---:|---:|---:|---:|
| D1: decode, mixed text, 1 session, speculation | **75.0** | 45.2 | 62.3 | 0.83 |
| D2a: decode, code, speculation, Crucible 4.05 | **104.9** | 65.4 | 89.1 | 0.85 |
| D2b: decode, code, speculation, Crucible 3.05 † | 104.9 | 65.4 | **112.4** † | 1.07 † |
| D3: decode, serial, short context | **42.2** | 29.3 | 32.2 | 0.76 |
| D4: decode, 4 sessions at 32K, total | 96.5 | 90.4 | **111.4** | 1.15 |
| D5: decode, 8 sessions, mixed, total | 106.3 | 103.6 | **121.6** | 1.14 |
| P1: cold prefill, 32K | **1,938** | 1,693 | 1,614 | 0.83 |
| P2: cold prefill, 130K | **1,767** | 1,616 | 1,516 | 0.86 |
| P1c: prefill, 2K new tokens after a cached 32K prompt | **1,554** | 1,443 | 1,255 | 0.81 |
| P2c: prefill, 2K new tokens after a cached 130K prompt | 1,119 | 1,101 | **1,215** | 1.09 |

- † D2b ran with a draft-head file that is not part of the public pack, so you cannot repeat it with the public files. Do not compare it with D2a or quote it as a result of the pack.
- D2b is not the same weight file as D2a. The Crucible 3.05 pack is smaller than the pack of Halogen. At equal bits (4.05), Halogen leads in D2a.
- Crucible 4.05 and 3.05 ran on the same binary. The decode rows ran at a GPU clock of about 2.8 to 2.9 GHz and the prefill rows at about 2.35 to 2.43 GHz.
  Halogen ran the prefill rows at a clock about 3% higher than Crucible (2,422 against 2,347 MHz).
- The rates of P1c and P2c are new rows divided by the first-token time. The engines compute different numbers of rows
  (next section), so quote the first-token time and not the rate for these two rows.
- The Crucible build is an engine build of Oct 23 to 24. It is older than the current release.

## The power profile run of Oct 26, 2026

The machine has a performance power profile. Cold prefill, 2 rounds each, median, tok/s:

| Row | Halogen 0.18.14 | Gufo 0.22.0 | Crucible | Crucible / Halogen |
|---|---:|---:|---:|---:|
| P1: cold prefill, 32K, performance | **2,121** | 1,843 | 1,870 | 0.88 |
| P2: cold prefill, 130K, performance | **1,914** | 1,760 | 1,746 | 0.91 |
| P1: cold prefill, 32K, balanced (Oct 23 to 24) | **1,938** | 1,693 | 1,614 | 0.83 |
| P2: cold prefill, 130K, balanced (Oct 23 to 24) | **1,767** | 1,616 | 1,516 | 0.86 |

The Halogen and Gufo values are the same binaries in both profiles. Their gain from the performance profile is 8% to 9% (P1, P2).
The Crucible rows of the performance profile ran a later build (Oct 25) than the balanced rows, so the Crucible gain
of 15% to 16% is a gain of the profile and the build together. 

## The follow-up turn (single runs, Oct 25 to 26, 2026)

One run for each engine, so these values are less certain than the tables above. Time to the first token for 2K new tokens after a cached prompt
(Crucible 4.05 pack, an engine build of Oct 25, Halogen 0.18.14):

| Row | Halogen 0.18.14 | Crucible 4.05 |
|---|---:|---:|
| 2K new tokens after a cached 32K prompt (s) | 1.96 | **1.58** |
| 2K new tokens after a cached 130K prompt (s) | **1.84** | 2.32 |

In a separate run (Oct 25 to 26), at 32K, Halogen caches in blocks of 1,024 rows. It had 31,744 rows cached and computed 3,082 rows. Crucible caches in blocks of 32,768 rows and
computed 2,058 rows. Halogen computes each row faster; Crucible computes fewer rows. At 130K, Halogen has the shorter wait.

## What differs between the engines

- **Weight format.** Halogen uses its own pack at 4.16 bits per weight. Gufo uses GGUF UD-Q4_K_XL. Crucible uses the EXL3 packs at 3.05 and 4.05 bits per weight.
  A rate difference is not an engine-only difference. [Quality](QUALITY.md) shows the quality of each pack.
- **Quality.** KL from the BF16 model on 896 positions (Oct 24 to 25, 2026): Crucible 4.05 0.0230, Crucible 3.05 0.0513, Halogen 0.0648. Gufo 0.22.0 returned no log-probabilities.
- **Pack of Crucible.** In a separate run of Oct 25 to 26 (an Oct 25 engine build, plain server settings, not the daily settings; Halogen 1 round, Crucible 2 rounds),
  Crucible gave these rates with other packs. D1 is decode of mixed text with 1 session, D3 is serial decode, P1 is cold prefill at 32K:

| Row | Halogen | Crucible 3.05 | Crucible 3.68 | Crucible 4.05 |
|---|---:|---:|---:|---:|
| D1 (tok/s) | 75.2 | 74.6 | 63.1 | 65.9 |
| D3 (tok/s) | 42.6 | 38.1 | 30.4 | 33.4 |
| P1 (tok/s) | 1,940 | 1,705 | 1,640 | 1,693 |

## Method

- One machine (AMD Strix Halo, 128 GB), one request client for all engines, one engine at a time, and no other work on the GPU.
- Each engine ran its own weights, with the flags that its authors publish. Halogen ran its container image.
- The prompts are the same for all engines. They hold no user text.
- The rows D1 to D5 and P1 to P2c are the same rows for all engines.
- We did not measure other engines. A value of another project is not in this table.
- The rival values were not measured again for the current Crucible release. A same-run table for the current release is planned.

## History: the comparison of Oct 17 to 19, 2026

This is the table of the earlier README. The engines are older: Halogen 0.16.2 and Gufo 0.9.0. Crucible is the v0.22.0 engine.


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
