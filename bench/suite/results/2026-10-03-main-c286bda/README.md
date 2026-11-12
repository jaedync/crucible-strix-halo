# Published main qualification

- Engine source: exact `git archive c286bda`, from `origin/main`.
- Full engine commit: `c286bdabe356fd859ddd94b4198b1f4f9551d05f`.
- Host: the test machine, October 3, 2026.
- Checkpoint: Qwen3.8-Flash-Next EXL3 3.05.
- Bulk prefill rows: 4,096.
- Binary SHA-256: `5dbee33289474fb41ca0322b1652bf197ad333b60aa449b5a17908792275fa29`.
- All 46 recorded engine source hashes match the archived main source.
- The harness and policy header hashes match the committed suite sources.
- Every run uses `gpu-run -p 3 -w 7200 -- timeout 300`.
- The effective host policy is `HSA_USERPTR_FOR_PAGED_MEM=0`.

`qualification.json` records the commands, source identity, report hashes, and limits.
Each subdirectory preserves `results.json`, `results.md`, `stderr.log`, `requests.txt`, and `host.json`.
The raw result log of each run is not kept. `results.json` holds every measured record.
The reports contain checkpoint metadata hashes. They do not hash every weight shard.

## Short requests

`short/` contains four prompt classes, serial decode, and depth-two speculation.
One discarded warmup precedes one measured repetition per mode.
All four measured 160-ID speculative sequences match their serial sequences.
The policy uses full head, Q zero, adaptation off, lookup off, and overlap off.
The resident process serves all 16 requests after one checkpoint load.

Bulk chat IDs differ from the earlier `e0bf131` small-row chat IDs.
The other three short prompt outputs remain equal to those earlier outputs.
These results do not establish bitwise equality between bulk and small-row prefill.
Do not use the unequal chat outputs for an exact speedup comparison.

## Long requests

Each long subdirectory contains one measured prefill request and one target ID.
No decode loop or speculative pass follows those long requests.
TTFT excludes model load, queue wait, request setup, and network transport.

| Prompt tokens | Prefill tok/s | TTFT seconds | Percent of 2,310 tok/s reference |
|---:|---:|---:|---:|
| 4,096 | 1,193.87 | 3.4309 | 51.68% |
| 16,384 | 1,275.75 | 12.8427 | 55.23% |
| 65,536 | 1,185.11 | 55.2995 | 51.30% |

The 2,310 tok/s value is a current-dataflow estimate, including EXL3 decode work.
It is not a context-specific hardware measurement.
The 64K small-row comparison is not run because it exceeds the 300-second execution limit.

## Gate limits

This task changes no engine behavior.
The earlier main qualification applies to the archived engine.
The short suite supplies additional Gate 3 token evidence only.
It does not replace dump, oracle, large-prefix, or restored-state gates.
A separate shared-tool smoke remains queued. No new complete gate PASS is claimed here.
