# engine/model/flashnext: charter

Rank 3 (model). A sibling model changes the config here.
Another model family adds `engine/model/<name>/`.

## Interface

- Named geometry and the model config (`geometry.h`, a leaf: any module may
  include it; it includes nothing).
- Weights, tensor names and load (`model.h`), tensor checks
  (`weight_validation.h`).
- Model parts with their own rules: PLE rows on the host (`ple_host.h`), the
  MTP sidecar (`mtp_sidecar.h`), the vision encoder (`vision.h`), image
  request rules and mRoPE positions (`multimodal.h`).
- State layout (`fixed_state.h`) and the prefill row schedule
  (`prefill_launch.h`).

## Invariants

- Geometry is a named constant, not a new literal (structure rules). No
  runtime model descriptor and no dynamic shapes on the hot path.
- `docs/FLASHNEXT_FORWARD.md` defines the equations and shapes.
- The loader checks metadata before it uses a tensor.
- Vision patch embedding and linear biases may be BF16; the loader converts them to F16 once (`bf16_half.h`).
- `vision_k6.safetensors` owns its tensor names (`AddFile(file, wins)`): an index shard that repeats them, such as an HF
  vision tower, is ignored. An HF tower without that file stops `--vision` before the model loads (`VisionTowerFault`).

## Allowed includes

- `engine/gfx1151/`, `engine/exl3/`, `engine/layers/` (with their kernels) and
  the leaves.
- `multimodal.h` includes the leaf `runtime/sha256.h`.

## Switches

The registry (`engine/switches.h`) is the truth for the names and the module of each switch.
Names have the prefix `CRUCIBLE_`, left out here.

- Exact-speed: `MTP_SIDECAR`.
- Layout: `PLE_HOST`.
- Diagnostic: `ABLATE_ANY_CHECKPOINT`, `PLE_TRACE`.
- Profile: `ABLATE`.

Every served profile sets `PLE_HOST` and `MTP_SIDECAR`; their `decide_by` is train 20.

## Lever ledger (closed)

| Lever | Result | Evidence |
|---|---|---|
| MTP sidecar in the daily 3.05 profiles | Landed (speed only, about 1.14x live decode) | development notes (not public) |
