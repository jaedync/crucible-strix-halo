# Benchmarks

## Status and hardware

These results come from one test machine, a Framework Desktop with a Ryzen AI Max+ 395 and 128 GB of shared LPDDR5X memory.
The GPU is a Radeon 8060S, gfx1151, with 40 compute units.
[Calibration](CALIBRATION.md) records the measured limits and host setup.

Results from different branches do not describe one released binary.
The tables identify each source commit of the private development history. Those commits are not in this repository. A new suite corpus must not be compared directly with an older prompt set.

## Measured hardware ceilings

| Resource | Measured ceiling | Evidence |
|---|---:|---|
| Cold DRAM reads | 245.5 GB/s | 2 GiB stream, best measured shape |
| MALL rereads | 719 GB/s | 32 MiB working set |
| f16 to f32 WMMA | 48.2 TFLOPS | Sustained balanced-profile probe |
| Working f16 WMMA reference | about 49 TFLOPS | Rounded working value |

GB uses decimal bytes. TFLOPS uses decimal floating-point operations per second.
These are measured chip limits, not the theoretical rates on a product page.

## Published main results: October 3, 2026

Engine source: `origin/main` `c286bda`, with bulk prefill rows set to 4,096.
The engine and all 46 recorded source hashes match that published main commit.
The suite uses the 3.05 checkpoint and the fixed committed prompt IDs.

| Prompt tokens | Prefill tok/s | Compute TTFT s | Percent of 2,310 tok/s reference |
|---:|---:|---:|---:|
| 4,096 | 1,193.87 | 3.431 | 51.68% |
| 16,384 | 1,275.75 | 12.843 | 55.23% |
| 65,536 | 1,185.11 | 55.300 | 51.30% |

Each long run measures prefill through one target ID, with no warmup or decode loop.
The 64K small-row comparison is not run because it exceeds the 300-second execution limit.
The 2,310 tok/s reference includes EXL3 decode work. It does not model every long-context cost.

| Prompt | Serial tok/s | Serial ceiling % | Spec2 tok/s | Spec2 ceiling % | Accept % | Serial TTFT s | Spec2 TTFT s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Code Fibonacci | 34.31 | 64.50 | 65.12 | 62.86 | 91.96 | 0.144 | 0.155 |
| Chat planning | 34.28 | 64.45 | 49.35 | 64.66 | 55.26 | 0.154 | 0.167 |
| Prose story | 34.31 | 64.49 | 50.34 | 65.08 | 56.00 | 0.155 | 0.167 |
| Copy Python | 34.31 | 64.49 | 67.44 | 62.77 | 99.07 | 0.288 | 0.324 |

All four measured 160-ID speculative sequences match their serial sequences.
The short run uses one warmup and one measured repetition per mode.
The full-head spec2 ceilings are 103.59, 76.33, 77.35, and 107.43 tok/s, respectively.
They use the 27.41 ms R=3 weight-read floor and measured delivered IDs per forward.
These percentages are model ratios, not hardware utilization measurements.

Bulk chat IDs differ from the earlier `e0bf131` small-row chat IDs.
The other three short outputs remain equal. Do not claim cross-path bitwise equality or an exact chat speedup.
Raw main evidence (`bench/suite/results/2026-10-03-main-c286bda/README.md`) includes commands, hashes, complete IDs, and logs.

## Three-repetition spread: published main 093fc4c

These six sessions use the newest published main available when they entered the queue.
Main `a60dfe2` was published during the wait. Its post-push regression is separate.
Every workload has three genuine measurements. All six process exit codes are zero.
All 46 engine source hashes match the exact `093fc4c` archive.
All short serial/spec2 pairs and all repeated output sequences match exactly.

Values below use **min / median / max**.

| Prompt tokens | Prefill tok/s | Compute TTFT s | Percent of 2,310 tok/s reference |
|---:|---|---|---|
| 4,096 | 1,245.95 / 1,268.30 / 1,272.63 | 3.219 / 3.230 / 3.287 | 53.94 / 54.90 / 55.09 |
| 16,384 | 1,278.20 / 1,286.31 / 1,293.91 | 12.662 / 12.737 / 12.818 | 55.33 / 55.68 / 56.01 |
| 65,536 | 1,172.26 / 1,185.20 / 1,189.86 | 55.079 / 55.295 / 55.906 | 50.75 / 51.31 / 51.51 |

One long session serves all three lengths, with no warmup and one target ID per length.
The context allocation fits the 64K request for every long measurement.
The short sessions use one discarded warmup per mode and 160 measured output IDs.

| Prompt | Serial tok/s | Full-head spec2 tok/s | Serial TTFT s | Spec2 TTFT s |
|---|---|---|---|---|
| Code Fibonacci | 34.488 / 34.603 / 34.638 | 65.279 / 65.716 / 66.395 | 0.144 / 0.144 / 0.147 | 0.153 / 0.155 / 0.157 |
| Chat planning | 34.596 / 34.625 / 34.626 | 49.499 / 49.528 / 49.555 | 0.154 / 0.154 / 0.155 | 0.167 / 0.167 / 0.167 |
| Prose story | 34.611 / 34.614 / 34.621 | 50.475 / 50.519 / 50.533 | 0.154 / 0.155 / 0.155 | 0.167 / 0.167 / 0.179 |
| Copy Python | 34.612 / 34.617 / 34.622 | 67.628 / 67.636 / 67.694 | 0.288 / 0.288 / 0.288 | 0.324 / 0.325 / 0.344 |

Median serial ceiling ratios are 65.06%, 65.10%, 65.08%, and 65.08%, respectively.
Median spec2 ceiling ratios are 63.44%, 64.89%, 65.31%, and 62.96%, respectively.
The model still uses the 27.41 ms R=3 floor and representative expert union of 22.4.
These are model ratios, not hardware utilization measurements.
The short TTFT spread includes isolated increases above five percent. Do not treat one sample as stable performance.

Raw repeated evidence (`bench/suite/results/2026-10-03-main-093fc4c-three/README.md`, in the tag `archive/pre-reset-2026-10-22`) preserves every measurement and source hash.
The repaired spread tool rejects renamed copies as independent samples.
It also protects primary inputs and rechecks their hashes before publication.

## Three-repetition spread: frozen main 45c3f0e

These six sessions freeze published main `45c3f0e`. They are not current-main `c5dfa33` results.
All 61 source hashes match the archive in every report. Every workload has three genuine measurements.
All repeated output sequences match. Each short serial/spec2 pair has 160 equal IDs.
Values use **min / median / max**.

| Prompt tokens | Prefill tok/s | Compute TTFT s | Percent of 2,310 tok/s reference |
|---:|---|---|---|
| 4,096 | 1103.204 / 1314.965 / 1349.608 | 3.035 / 3.115 / 3.713 | 47.758 / 56.925 / 58.425 |
| 16,384 | 1267.617 / 1321.632 / 1331.127 | 12.308 / 12.397 / 12.925 | 54.875 / 57.213 / 57.625 |
| 65,536 | 1155.110 / 1201.228 / 1206.973 | 54.298 / 54.557 / 56.736 | 50.005 / 52.001 / 52.250 |

| Prompt | Serial tok/s | Full-head spec2 tok/s | Median serial ceiling % | Median spec2 ceiling % |
|---|---|---|---:|---:|
| Code Fibonacci | 35.919 / 36.380 / 36.606 | 68.725 / 69.517 / 69.982 | 68.393 | 67.107 |
| Chat planning | 36.032 / 36.521 / 36.596 | 51.315 / 51.829 / 52.029 | 68.658 | 67.901 |
| Prose story | 36.061 / 36.437 / 36.557 | 52.466 / 52.709 / 52.871 | 68.501 | 68.145 |
| Copy Python | 36.296 / 36.323 / 36.565 | 70.627 / 70.727 / 71.036 | 68.286 | 65.837 |

The 4K range is wide. Do not use one sample as a stable performance claim.
Percentages are analytical model ratios, not measured hardware utilization.
Raw frozen evidence (`bench/suite/results/2026-10-03-main-45c3f0e-three/README.md`, in the tag `archive/pre-reset-2026-10-22`) preserves all six sessions and source verification.
Fresh `c5dfa33` suite binaries are built. Their three-repetition device results remain pending.

## Post-push regression check

The command in [the suite guide](../bench/suite/README.md#post-push-regression) runs two short prompts and 4K prefill.
One checkpoint load serves serial/spec2 warmups, measured decode, and the first 4K target ID.
It compares complete IDs and rates with frozen published main reports.
The default threshold is five percent for throughput decreases and TTFT increases.
The tool returns nonzero on missing data, changed IDs, failed processes, or larger slowdowns.
A completed `c286bda` smoke passes all five comparisons with exact baseline IDs.
Its conservative admission estimate is 104.18 seconds.
The exact published `a60dfe2` regression also returns zero and passes all five comparisons.
Its two serial rates increase by 1.73% and 2.00%. Its two spec2 rates increase by 2.85% and 1.09%.
Its 4K prefill rate increases by 12.42%, with an 11.04% lower compute TTFT.
All five complete ID sequences match the frozen published baseline.
Raw regression evidence (`bench/suite/results/2026-10-03-regression-main-a60dfe2/README.md`, in the tag `archive/pre-reset-2026-10-22`) includes the outer exit code.
This performance check does not replace the mandatory correctness gates.

## Historical engine measurements

These runs predate `bench/suite`. Their prompt sets and timing paths differ.
The entries below preserve that distinction.

| Workload | Branch and commit | Measured result | Ceiling model | Percent |
|---|---|---:|---:|---:|
| Serial, 3.05, 160 outputs | speculative-decoding branch `44450ab` | 34.24 to 34.62 tok/s | 53.19 tok/s weight model | 64.4 to 65.1% |
| Code Fibonacci, adaptive speculation | speculative-decoding branch `72fcd5a` | 97.2 tok/s | Workload ceiling not measured | n/a |
| Copy, adaptive speculation and lookup | speculative-decoding branch `72fcd5a` | 130.6 tok/s | Estimated 235 to 260 tok/s | 50.2 to 55.6% |
| 4K prefill, v33 | prefill branch `5431bde` | 1,375.2 tok/s | 2,310 tok/s current dataflow | 59.5% |
| 16K prefill, v33 | prefill branch `5431bde` | 1,304.4 tok/s | Context-specific ceiling not established | n/a |

Evidence:

- The working notes of the two branches, October 2. They are not public.
- [Forward specification](FLASHNEXT_FORWARD.md#5-per-token-checkpoint-weight-byte-accounting), logical weight bytes.

The historical speculative runs use depth seven, a 32K adaptive draft head, and `--draft-q 0.3`.
The copy run also uses lookup depth seven.
Their percentages are estimates, not measured DRAM utilization.

The v33 prefill runs use a bulk path with different arithmetic from the small-row decode path.
The findings report oracle qualification and separate long-prompt checks.
Do not describe those prefill values as bitwise equality with serial recurrence.

## Fixed suite results: October 3, 2026

Engine source: speculative-decoding branch `44450ab`. Harness source: suite branch `933cd15`.
The run used the 3.05 checkpoint, one warmup, and one measured repetition.
It used the full draft head, depth two, and no lookup or adaptive policy.

| Prompt | Serial tok/s | Serial ceiling % | Spec tok/s | Acceptance % | Serial TTFT s | Spec TTFT s |
|---|---:|---:|---:|---:|---:|---:|
| Code Fibonacci | 34.11 | 64.13 | 65.90 | 91.96 | 0.385 | 0.389 |
| Chat planning | 34.16 | 64.21 | 50.49 | 57.43 | 0.451 | 0.463 |
| Prose story | 34.09 | 64.09 | 51.41 | 58.90 | 0.446 | 0.457 |
| Copy Python | 34.12 | 64.14 | 66.91 | 99.07 | 1.516 | 1.550 |

All four 160-ID speculative outputs match their serial outputs exactly.
One checkpoint load serves all 16 requests, including warmups.
Load plus plan preparation takes 33.97 seconds. The checkpoint loader itself reports 22.2 seconds.
The run uses `HSA_USERPTR_FOR_PAGED_MEM=0`.

Raw JSON, timings, and source hashes (`bench/suite/results/2026-10-03-spec444/results.json`, in the tag `archive/pre-reset-2026-10-22`)
provide the evidence. The adjacent Markdown table includes prefill rates.
These short-prefill rates are not the bulk prefill measurements above.
The separate ceiling annotation applies the representative R=3 route model to measured acceptance.
It preserves the original raw report and records its SHA-256.

| Prompt | Full-head spec2 ceiling tok/s | Measured percent of ceiling |
|---|---:|---:|
| Code Fibonacci | 103.59 | 63.62% |
| Chat planning | 78.39 | 64.41% |
| Prose story | 79.47 | 64.69% |
| Copy Python | 107.43 | 62.29% |

These are weight-read estimates, not measured hardware utilization.
The two full-head drafts and R=3 verification give a 27.41 ms step floor.
Ceiling evidence (`bench/suite/results/2026-10-03-spec444/ceiling-estimates.json`, in the tag `archive/pre-reset-2026-10-22`) includes every model input and source hash.

The published main table above contains the completed 4K, 16K, and 64K bulk measurements.
The 64K small-row case remains not run under the 300-second execution limit.

## Fixed suite method

The reproducible suite lives in [`bench/suite`](../bench/suite/README.md).
Its manifest commits all token IDs and their hashes.
The prompt classes are code, chat, prose, copy-heavy edits, and repeated-record long contexts.
The long inputs contain exactly 4K, 16K, and 64K IDs, where K means 1,024.

The default run uses four short prompts, one discarded warmup, and one measured repetition per mode.
It produces 160 output IDs with greedy serial decode and depth-two speculation.
There is no EOS stop and no reused prompt prefix.
This fixed output length measures throughput. It does not measure complete-answer latency or task quality.

Each harness process loads the checkpoint once and resets sequence state between requests.
It calls the selected engine's `RunGen` function.
It does not add event profiling or expert-count kernels to the timed path.
The runner starts each process through `gpu-run` (see `tools/gpu-run`) at priority three.
The helper holds the GPU lock only during the bounded resident process.
Each process has a 300-second timeout.

## Metrics

| Metric | Definition |
|---|---|
| Prefill tok/s | Prompt ID count divided by engine prefill seconds |
| Compute TTFT | Engine prefill time through the first target ID |
| Serial decode tok/s | Remaining `N - 1` IDs divided by serial decode seconds |
| Speculative decode tok/s | Remaining `N - 1` verified IDs divided by speculative decode seconds |
| Acceptance | Accepted drafts divided by proposed drafts |
| IDs per forward | Remaining `N - 1` IDs divided by target decode forwards |
| Percent of ceiling | Measured throughput divided by its stated ceiling, times 100 |

Compute TTFT excludes checkpoint load, request setup, tokenization, queue time, and network transport.
The load event reports checkpoint load and plan preparation separately.
Draft counters can include verified tokens beyond the requested final output length.

The JSON preserves individual repetitions and full output IDs.
The Markdown table does not hide variation behind a best-of-run selection.
For stable comparisons, run multiple repetitions and reverse the binary order.

## Ceiling models

### Serial decode

The 3.05 logical weight model uses 4,615,344,324 bytes per target token, including small gathers.
At 245.5 GB/s, its ideal time is 18.80 ms and its rate is 53.19 tok/s.
This model excludes recurrent-state traffic, context reads, activation traffic, and launch costs.
It assumes one read of each selected weight. It is not a complete memory-transaction measurement.

The runner applies this model only to the known 3.05 checkpoint name.
Use checkpoint-specific accounting for 4.05. Do not reuse the 3.05 byte count.

### Speculative decode

Acceptance alone does not determine the ceiling.
The model must also include draft cost, verification width, and the union of selected experts.
The serial ceiling is not a valid speculative denominator.

The historical copy estimate uses an R=8 verification floor of 31 to 34 ms.
At 7.95 emitted IDs per step, that gives about 235 to 260 tok/s.
Other prompts need their own route and draft accounting.
The fixed suite uses a representative route model supplied by the speculative-decoding branch on October 3.

| Verify rows R | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean distinct experts per layer U(R) | 10 | 16 | 22.4 | 26.2 | 32.0 | 37 | 41 | 44.1 |

```text
verify_GB(R) = 3.726 + 0.0894 * U(R)
step_floor_s = (verify_GB(R) + draft_depth * 0.500118692) / 245.5
ceiling_tok_s = delivered_IDs_per_forward / step_floor_s
```

For plain spec2, R=3 gives 5.72856 GB per verification pass.
Two full-head drafts raise the step floor to about 27.41 ms.
The suite uses its measured delivered IDs per forward to account for acceptance and final output truncation.
It applies this model only when every forward has the requested fixed draft depth.

The unions are representative, not measured on these suite prompts.
The copy workload can use more distinct experts; a separate R=8 measurement found 51.3 per layer.
Thus the reported percentage is an estimate against a weight-read model, not an actual bandwidth measurement.
It excludes KV/state reads, activation traffic, and launch gaps.
An explicit operator-supplied speculative ceiling replaces this estimate.

### Prefill

The historical 4K current-dataflow model gives 2,310 tok/s, including EXL3 decode work.
An estimated fused design gives about 3,866 tok/s.
The fused value is not a measured engine result.
Use `--prefill-ceiling 2310` for the documented current-dataflow reference.
The suite also reports a separate ratio against the 3,866 tok/s fused-design estimate.
The prefill branch confirms these per-token references across the tested lengths.
QSA index selection grows with context, so the reference does not model all long-context costs.

A tok/s ratio against these models is not a direct fraction of 49 TFLOPS.
Report WMMA utilization only when the operation count and measured kernel time support that calculation.

## Correctness and reproducibility

The suite checks every output ID between serial and speculative runs.
It rejects failed processes, incomplete records, duplicate records, wrong lengths, and unequal outputs.
It also checks binary and prompt hashes.
Its comparison mode requires matching request settings, checkpoint metadata, prompt IDs, and output IDs.

The suite records compiler details, source identity, source hashes, host memory state, HSA policy, and raw logs.
Memory fragmentation and cache state can move results by several percent.
The runner records `/proc/meminfo` and `/proc/buddyinfo` inside the GPU lock before model load.
It does not change host memory policy or clear caches.

The checkpoint record does not hash every weight shard.
Record the external checkpoint revision when copying a result outside this repository.
A corpus or runner test is not an end-to-end model-quality result.

The full shared gates remain required for engine changes.
The suite's token checks cover selected Gate 3 cases only.
Use the shared gate tool for complete engine qualification.
It requires fresh full-size dumps, byte equality, complete IDs, and all three successful gates.
Its bulk option also checks live decode and speculative continuation after prefill.
