# Quality

How much of the 16-bit model each weight file keeps. Speed is in [Results](RESULTS.md).

## Distance from the 16-bit model, Oct 24 to 25, 2026

KL divergence from the BF16 model on 896 text positions, scored by us. The reference is the BF16 model. Lower is closer.
The value is a lower bound of the full-vocabulary KL (21 buckets for each position: the 20 best token ids and one rest bucket).
Gufo 0.22.0 returned no log-probabilities, so it has no row. Gufo v0.27.0 (Oct 29, 2026) serves them; we did not score it.

| Engine and pack | Bits per weight | KL from BF16 | Same top token as BF16 |
|---|---:|---:|---:|
| Crucible 4.05 | 4.05 | 0.0230 | 94.3% |
| Crucible 3.68 | 3.68 | 0.0306 | 94.6% |
| Crucible 3.05 | 3.05 | 0.0513 | 91.7% |
| Halogen 0.18.14 | 4.16 | 0.0648 | 90.3% |

The positions are the same 896 for all rows. Halogen ran its own pack. The values of Crucible 4.05 were measured on Oct 25.

## History: the tables of Oct 17 to 20, 2026

### Quality

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

#### Measured distance from the 16-bit model

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

#### Published distance from the 16-bit model

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
