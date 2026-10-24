# update_selftest: charter

The self-test of `make update` (`tools/update.sh`).

## Interface

- `selftest.py cpu VERSION`: the VERSION file, the engine version header, the server imports and
  `build/engine/crucible --version`. No GPU.
- `selftest.py gpu CKPT`: `make update CKPT=<model folder>` runs it. It compares a few greedy tokens with
  `<pack>.json`, where the pack is the folder name after `Qwen3.8-Flash-Next-exl3-`. A pack without a file is skipped
  and reported, not failed. It loads the whole model: stop the server first.
- `selftest.py record CKPT`: writes `<pack>.json` from the current build. Run it on a qualified machine for each
  supported pack before a release, and commit the file in that release.

## Expected file

```json
{"pack": "3.05bpw_h5_ng5", "prompt_ids": [9707, 11, 1879, 0], "n": 16, "expected": [ ... ]}
```

## Invariants

- A release ships a file only for a pack it has run. A wrong file would roll back every update that follows.
- The test uses greedy decoding with the default options, so that a changed token is a changed output.

## Files in this release

- `3.05bpw_h5_ng5.json` and `4.05bpw_h6_ng6.json`: recorded with the v0.26.0 build on the GPU. A second load of each
  pack gave the same 16 token IDs. The files hold token IDs only.
- Another pack has no file, so the GPU part skips it.
