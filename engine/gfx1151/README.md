# engine/gfx1151: charter

Rank 0 (lowest) of the engine layers.

## Interface

- Wave32 device helpers: `HIP_OK`, wave and block sums, WHT stages
  (`common.h`).
- WMMA primitives with high/low expansion (`prefill_wmma.h`).
- Synchronization: compute-only barriers (`persistent_sync.h`) and the LDS
  stream handoff (`lds_stream_ring.h`).
- Host memory and queues: pinned buffers and allocator, page-migration
  warm-up and waits, SVM policy (`pinned_*.h`, `warmup.h`,
  `migration_wait.h`, `runtime_memory_policy.h`). The engine never calls
  `hipSetDeviceFlags`: a run-time change of the scheduling mode loses CLR
  completion handlers.

## Invariants

- No model, format or layer knowledge. A helper that names a model size or an
  EXL3 field belongs in a higher module.
- A helper keeps its arithmetic order. Callers depend on it for bitwise
  output.
- gfx1151 (gfx11) has only `s_barrier`. Split barriers are gfx12 only.

## Allowed includes

- System and HIP headers, and the leaves (`engine/model/flashnext/geometry.h`,
  `engine/switches.h`, `engine/runtime/sha256.h`).
- Nothing else in `engine/`. `tests/test_layering.py` checks this.

## Switches

- None. The registry gives `WARMUP` to the runtime module.

## Lever ledger (closed)

| Lever | Result | Evidence |
|---|---|---|
| Larger GPU pages for the expert gather | No change: 2 GB and 16 GB pools give the same rate | development notes (not public) |
| Persistent MALL residency for weights | 4.62 GB per token streams once; 32 MB helps only as staging | development notes (not public) |
| int8 or int4 WMMA for prefill | iu8 has the f16 rate; iu4 is lossy for mul1 weights | docs/CALIBRATION.md |
