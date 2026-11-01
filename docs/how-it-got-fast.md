# How it got fast

This page explains six ideas behind the speed of Crucible. Each idea has a short text, one number, and a link.
The numbers are from [Results](RESULTS.md). The rival numbers are in [Rivals](RIVALS.md).

## 1. Read fewer bytes

Decode makes one token at a time. For each token, the GPU must read all active weights from memory.
Memory speed is the limit, not compute speed.

- At 3.05 bits per weight, one token needs 4.615 GB of weight reads.
- At 4.05 bits per weight, one token needs 5.356 GB.
- A 16-bit file would need about four times more than the 4.05 pack.

A smaller file gives a faster token. The 3.05 pack decodes faster than the 4.05 pack in every decode row of [Results](RESULTS.md) (serial decode without speculation: 39.4 against 34.3 tok/s).

## 2. Keep quality in each bit

A smaller file is worse if it loses quality. EXL3 is the weight format of ExLlamaV3 (turboderp).
It gives each matrix the number of bits that it needs, from 2 to 8 bits.
The mean is 3.05 or 4.05 bits per weight.

Quality is the KL divergence from the 16-bit (BF16) model. A lower value is closer.
Our own measurement of Oct 24 to 25, 2026 used 896 text positions:

| Pack | Bits per weight | KL from BF16 |
|---|---:|---:|
| Crucible 4.05 | 4.05 | 0.0230 |
| Crucible 3.05 | 3.05 | 0.0513 |
| Halogen 0.18.14 | 4.16 | 0.0648 |

The 3.05 pack has a lower KL than the pack of Halogen, with fewer bits. [Quality](QUALITY.md) has the method and the longer table.
The [EXL3 format note](EXL3_FORMAT.md) describes the files.

## 3. Guess ahead and check

The model has a small extra head. It proposes the next tokens before the main model computes them.
This is speculative decoding (MTP).

1. The draft head proposes a chain of tokens.
2. The main model reads its weights one time and checks the whole chain.
3. The engine keeps the tokens that the main model agrees with. It drops the rest.

One read of the weights then gives more than one token. In a test on code (Oct 25, 2026, one pair, 4.05 pack, the engine build of that date),
the engine accepted 4.69 tokens for each pass of the model.

The draft head predicts code better than prose. In the Oct 26 results (the build after v0.26.1), at 32K, one session decodes code at 60.5 tok/s and prose at 53.8 tok/s.
Without the draft head, decode runs at 34.1 tok/s (short context, 4.05 pack, [Results](RESULTS.md)).

Speculation changes the speed. It does not change the tokens (see idea 5).

## 4. Keep the work that you did

An agent session sends the same long prompt again and again, with a little new text at the end.
The server keeps the model state of the earlier turns. A later turn computes only the new tokens.

- A prompt that you send again gets its first token after 0.023 s (v0.26.0: 0.014 s). v0.22.0 needed 1.87 s (32K prompt, 4.05 pack).
- The cache also keeps the state at the end of each answer, so the next turn does not compute the answer again.
- An optional disk tier keeps conversations across a restart.

[Session replay](SESSION_REPLAY.md) explains the cache and lists the cases in which a cache hit gives other tokens than a cold prompt.
A cold prompt is still slow: the first 32K tokens need about 18 s. The cache helps only after the first turn.

## 5. Keep the output exact

A speed change is useful only if the answer stays the same. Crucible has three rules:

- The chunks of a prompt follow a fixed plan. Load and other sessions never change the plan.
- Rows of a packed batch give the same bits as one row alone.
- A cache hit gives the same bits as a miss.

A tool called `tools/gates.py` checks these rules on the GPU before a change merges:

- Gate 1 compares the prompt state with a named baseline, file by file, bit by bit.
- Gate 2 compares prompt chunks of one row and of eight rows.
- Gate 3 compares the complete output with speculation and without it.

Some changes give speed but are not bitwise equal. They are on by default.
`CRUCIBLE_EXACT=1` turns them off. A lossy change needs the approval of a maintainer
and must pass the KL, top-1 and long-context checks. See [Architecture](ARCHITECTURE.md#exactness-rules).

## 6. Fill the gaps

A GPU is idle between small kernels. Two changes use this time.

- **Prefetch.** Weights do not depend on the data. A small kernel on a second stream reads the next weights into the
  32 MiB cache of the chip (the MALL) while the main kernels run. A lab test on Oct 1, 2026 gave 1.15x to 1.21x on a chain
  with the layer sizes of this model. It gave no gain for chunks of 4 MiB or more (0.98x).
  The switch `CRUCIBLE_LOOKAHEAD` sets the prefetch.
- **Launch size.** Long prompts prefill in launches of 16,384 rows (the default of v0.30.0; v0.26.0 used 8,192). A short last part joins the last launch.
  There is no small launch at the end ([Changelog](../CHANGELOG.md), v0.26.0).

## What is not solved

Halogen is faster at one-session decode and at cold prefill ([Rivals](RIVALS.md)). Crucible has room left:
serial decode uses 70% to 77% of the measured memory bandwidth, and cold prefill is at 53% to 56% of the estimated
compute ceiling (Oct 22 audit). The roadmap in the [README](../README.md#roadmap) lists the next steps.
