# engine/runtime: charter

Rank 4 (runtime).

## Interface

- Class `Engine` fragments (included inside or after the class in
  `engine.hip`): `batch_forward.h`, `batch_spec_forward.h`,
  `draft_overlap.h`, `prefill_engine.inc`, `conversation_snapshot.h`,
  `slot_checkpoint_io.h`.
- The one trunk layer loop: `layer_loop.h` (`RunTrunk`). The four
  forward passes call it; `tests/test_layer_loops.py` allows no other.
- Sequence state and request rules: `sequence_state.h`, `retained_prefix.h`,
  `bounds.h`, `prompt.h`, `decode_policy.h`.
- Small helpers: `dump_io.h` (checked residual dump output), `device_zero.h`
  (zero-span check for `gen --jobs` state).
- Decode, verification and sampling: `generation.h`, `spec_steps.h`,
  `sampling.h`, `sampling_penalty.h`, `sampling_generation.h`.
- Drafts: `draft_*.h` (`draft_accept.h`: the acceptance prior), `mtp_bulk.h`.
- Generation options: `gen_options.h` (`GenOptions`, `CliGenOptions`
  and the draft selection rules). The CLI parses them. Serve reads them.
- Prefill scheduling: `prefill_interleave.h`, `prefill_plan_budget.h`,
  `prefill_prune.h`, `prefill_front_check.h`, `stage_arena.h`, `prefill_rows.h`
  (the row limits and `kPrefillAuto`).
- Batch contracts: `batch_admission.h`, `batch_contract.h`,
  `batch_inputs.h`, `batch_spec_contract.h`, `packed_rows_option.h`,
  `pass_rows.h`.
- State caches: `conversation_*.h`, `slot_checkpoints.h`,
  `snapshot_bytes.h`, `state_disk.h`, `state_root.h` (where the state folder
  can be).
- `sha256.h`: a leaf. Any module may include it. It includes only the
  standard library.

## Invariants

- A request's output depends only on the request: batching, load and
  speculation do not change it.
  Session replay is the exception (docs/SESSION_REPLAY.md).
- Verification, never a proposal, commits tokens.

## Allowed includes

- Every lower module (`gfx1151`, `exl3`, `layers`, `model/flashnext`) and the
  leaves. Not `serve/` or `cli/`.

## Switches

The registry (`engine/switches.h`) is the truth for the names and the module of each switch.
Names have the prefix `CRUCIBLE_`, left out here.

- Numerics: `EXACT`.
- Exact-speed: `FUSE`, `LOOKAHEAD`, `LOOKAHEAD_MIB`, `PF_PRUNE`, `STATE_CACHE`, `STATIC_CACHE`, `TUNE_ROWS`, `WARMUP`.
- Layout: `COMPACT_STATE`, `PF_LAUNCH_ROWS`, `PF_PAD_H`, `PF_STAGE_SHARE`.
- Diagnostic: `INTERLEAVE_CHECK`, `ROUTE_DUMP`.

## Lever ledger (closed)

| Lever | Result | Evidence |
|---|---|---|
| Prompt lookup default | Landed: output identity and ABBA PASS | development notes (not public) |
| Tree verification (top-2 siblings) | Killed: runner-up right in 36% of rejections | development notes (not public) |
| 8K prefill chunks | About 1 percent; VGPR cap keeps 96-row units | development notes (not public) |
| Pipelining sub-chunks across layers | Per-row MoE cost +29 percent | development notes (not public) |
| A prefill megakernel | Launch cost is 7 ms of 2,979 ms per 4K | development notes (not public) |
| Draft or prefill work on the CPU | Same DRAM and one shared power limit | development notes (not public) |
| An exact greedy LM-head shortcut | No usable bound on the skipped logits | development notes (not public) |
