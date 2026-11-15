# Main baseline: fp64 oracle errors

## Source

- Source commit: `5b57ceb`.
- Date: 2026-10-01.
- Host: the test machine.
- Checkpoint: `~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5`.
- Prompt: `$WORK_FIB_IDS`.
- Prompt rows: 24.

`$WORK_DIR` and the `WORK_` variables come from `tools/work_paths.sh`.
Run `. tools/work_paths.sh` in a checkout before the commands.

The baseline binary was built from `git archive 5b57ceb`.
All 49 baseline dump files match the fused dump files bit for bit.
The oracle read the fused files. Thus, each error below is also a baseline error.
The printed oracle values have four significant digits.

## Commands

```sh
git archive 5b57ceb | tar -xf - -C baseline-source
/opt/rocm/bin/hipcc -O3 -std=c++20 --offload-arch=gfx1151 \
  --rocm-path=/opt/rocm -I/opt/rocm/include \
  -o build/engine/baseline baseline-source/engine/engine.hip -lpthread
flock -w 900 /tmp/crucible-gpu.lock timeout -k 10 300 build/engine/baseline gen \
  ~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 \
  --ids-file $WORK_FIB_IDS \
  --n 1 --chunk 8 --dump results/gate-baseline8
for f in results/gate-baseline8/h_*.f32; do
  cmp "$f" "results/gate-fused8/${f##*/}" || exit 1
done
for l in $(seq 0 47); do
  OPENBLAS_NUM_THREADS=8 timeout -k 10 300 \
    $WORK_REFENV_PYTHON ref/oracle.py \
    ~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 \
    results/gate-fused8 "$l" --pos 0,14,23 \
    --ids $WORK_FIB_IDS
done
```

## Results

- Mean relative L2 error: `1.560679514e-04`.
- Maximum error: `7.410e-4`.
- Maximum point: layer 31, position 14.
- Points above `3e-4`: 16 of 144.

| Layer | Position 0 | Position 14 | Position 23 |
|---:|---:|---:|---:|
| 0 | 2.703e-04 | 2.089e-04 | 3.231e-04 |
| 1 | 3.174e-04 | 8.681e-05 | 9.869e-05 |
| 2 | 3.511e-04 | 1.372e-04 | 1.351e-04 |
| 3 | 4.644e-05 | 1.777e-04 | 1.848e-04 |
| 4 | 2.554e-04 | 1.775e-04 | 1.834e-04 |
| 5 | 7.219e-06 | 1.671e-04 | 1.399e-04 |
| 6 | 2.545e-05 | 1.136e-04 | 1.407e-04 |
| 7 | 3.072e-05 | 2.037e-04 | 1.791e-04 |
| 8 | 1.771e-05 | 1.158e-04 | 6.094e-05 |
| 9 | 2.464e-05 | 8.837e-05 | 4.986e-05 |
| 10 | 3.861e-06 | 9.325e-05 | 5.071e-05 |
| 11 | 2.404e-05 | 1.245e-04 | 9.103e-05 |
| 12 | 5.642e-06 | 1.072e-04 | 1.261e-04 |
| 13 | 4.740e-06 | 1.032e-04 | 6.800e-05 |
| 14 | 5.273e-05 | 1.235e-04 | 9.932e-05 |
| 15 | 4.191e-05 | 3.130e-04 | 2.868e-04 |
| 16 | 5.897e-05 | 1.071e-04 | 1.026e-04 |
| 17 | 6.074e-06 | 1.231e-04 | 1.294e-04 |
| 18 | 6.157e-06 | 1.481e-04 | 1.867e-04 |
| 19 | 4.281e-05 | 1.858e-04 | 1.763e-04 |
| 20 | 1.054e-05 | 1.553e-04 | 1.785e-04 |
| 21 | 1.724e-05 | 1.505e-04 | 1.744e-04 |
| 22 | 8.676e-06 | 1.285e-04 | 2.651e-04 |
| 23 | 1.472e-05 | 1.866e-04 | 2.354e-04 |
| 24 | 1.741e-05 | 1.385e-04 | 1.915e-04 |
| 25 | 8.053e-05 | 1.368e-04 | 1.613e-04 |
| 26 | 1.396e-05 | 1.573e-04 | 1.538e-04 |
| 27 | 2.845e-05 | 2.705e-04 | 2.456e-04 |
| 28 | 1.587e-05 | 1.511e-04 | 2.066e-04 |
| 29 | 9.836e-06 | 1.726e-04 | 1.768e-04 |
| 30 | 1.723e-04 | 2.199e-04 | 2.212e-04 |
| 31 | 8.828e-05 | 7.410e-04 | 5.914e-04 |
| 32 | 5.255e-05 | 1.991e-04 | 2.075e-04 |
| 33 | 1.250e-05 | 1.974e-04 | 2.384e-04 |
| 34 | 1.260e-05 | 1.884e-04 | 2.283e-04 |
| 35 | 8.483e-05 | 3.997e-04 | 4.555e-04 |
| 36 | 1.965e-05 | 3.152e-04 | 3.123e-04 |
| 37 | 8.786e-05 | 1.666e-04 | 1.394e-04 |
| 38 | 2.211e-05 | 1.532e-04 | 1.645e-04 |
| 39 | 8.662e-05 | 4.535e-04 | 4.873e-04 |
| 40 | 2.922e-05 | 1.514e-04 | 1.218e-04 |
| 41 | 3.509e-05 | 1.594e-04 | 1.871e-04 |
| 42 | 6.992e-05 | 1.066e-04 | 1.632e-04 |
| 43 | 6.951e-05 | 2.432e-04 | 3.991e-04 |
| 44 | 2.232e-05 | 4.263e-04 | 3.633e-04 |
| 45 | 1.605e-04 | 2.696e-04 | 2.554e-04 |
| 46 | 1.189e-04 | 2.771e-04 | 3.782e-04 |
| 47 | 2.774e-04 | 1.668e-04 | 1.381e-04 |
