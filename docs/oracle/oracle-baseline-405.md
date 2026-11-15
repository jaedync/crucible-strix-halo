# Quality checkpoint: fp64 oracle baseline

## Result

All 144 points stay below `1.0e-3`.
The mean error is `1.564899931e-4`. The maximum is `7.307e-4` at layer 31, position 14.
The mean is 1.00265 times the corresponding 3.05 mean.

These are per-layer errors against each checkpoint's own decoded weights.
They are not errors against BF16 model weights.
The check uses each engine layer's input dump, not an accumulated fp64 forward pass.

## Source and method

- Host: the test machine, 2026-10-02.
- Engine base: `integrate` commit `9ee1158`, plus the `q405` format changes.
- Quality checkpoint: `~/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6`.
- Comparison checkpoint: `~/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5`.
- Prompt: `$WORK_FIB_IDS`, 24 tokens.
- Quality dumps: `$WORK_DIR/wt/q405/results/q405/q405-c1`.
- Comparison dumps: `$WORK_DIR/wt/q405/results/q405/base-c1`, from the unchanged base binary.
- Logs: `$WORK_DIR/wt/q405/results/q405/oracle{305,405}-L.log`.

`$WORK_DIR` and the `WORK_` variables come from `tools/work_paths.sh`.
Run `. tools/work_paths.sh` in a checkout before the commands.

The 3.05 dumps match the modified engine bitwise.
Both checkpoints also pass bitwise chunk-1 versus chunk-8 comparison across all 49 dump files.
The oracle infers the PLE width from each checkpoint and supports both table layouts.
The printed errors have four significant digits. Means below use those printed values.

The 3.05 mean is `1.560760764e-4`. Its maximum is `7.340e-4`, also at layer 31, position 14.
This differs slightly from the older `5b57ceb` baseline because `9ee1158` changes the QSA reduction order.

The parent approved a cross-checkpoint gate of errors below `1.0e-3` in the same range as the 3.05 values.
The quality errors pass this gate. Arithmetic changes were not needed.
Layer 32, position 0 has the largest point ratio: 2.03045.
Its absolute error is only `1.067e-4`; the 3.05 value is `5.255e-5`.
This is a comparison between different quantized weights, not a regression test on one checkpoint.

## Commands

Run from the checkout of the build under test.
Use an empty dump directory.

```sh
ck=~/models/exl3/Qwen3.8-Flash-Next-exl3-4.05bpw_h6_ng6
ids=$WORK_FIB_IDS
flock -w 900 /tmp/crucible-gpu.lock timeout -k 10 300 build/engine/crucible gen \
  "$ck" --ids-file "$ids" --n 1 --chunk 1 --dump results/q405/q405-c1
for layer in $(seq 0 47); do
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 nice -n 5 timeout -k 10 300 \
    $WORK_REFENV_PYTHON ref/oracle.py \
    "$ck" results/q405/q405-c1 "$layer" --pos 0,14,23 --ids "$ids"
done
```

The quality run used four CPU threads. The complete comparison run used eight threads.
The initial five comparison layers also used four threads.
The updated CPU etiquette was read after the quality run finished.
No oracle process used the GPU. Only one oracle process ran at a time.

## Relative L2 errors

| Layer | 3.05 pos 0 | 3.05 pos 14 | 3.05 pos 23 | 4.05 pos 0 | 4.05 pos 14 | 4.05 pos 23 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.703e-4 | 2.089e-4 | 3.231e-4 | 2.362e-4 | 3.313e-4 | 2.977e-4 |
| 1 | 3.174e-4 | 8.681e-5 | 9.869e-5 | 3.595e-4 | 8.187e-5 | 1.005e-4 |
| 2 | 3.511e-4 | 1.372e-4 | 1.351e-4 | 3.656e-4 | 1.248e-4 | 1.389e-4 |
| 3 | 4.644e-5 | 1.777e-4 | 1.848e-4 | 4.450e-5 | 1.559e-4 | 2.322e-4 |
| 4 | 2.554e-4 | 1.753e-4 | 1.822e-4 | 1.756e-4 | 1.707e-4 | 1.989e-4 |
| 5 | 7.219e-6 | 1.621e-4 | 1.339e-4 | 7.299e-6 | 1.690e-4 | 1.414e-4 |
| 6 | 2.545e-5 | 1.284e-4 | 1.377e-4 | 4.245e-5 | 1.251e-4 | 1.459e-4 |
| 7 | 3.072e-5 | 2.088e-4 | 1.847e-4 | 2.838e-5 | 2.005e-4 | 1.774e-4 |
| 8 | 1.771e-5 | 1.112e-4 | 6.356e-5 | 1.389e-5 | 1.165e-4 | 6.105e-5 |
| 9 | 2.464e-5 | 8.603e-5 | 5.177e-5 | 1.767e-5 | 8.380e-5 | 4.880e-5 |
| 10 | 3.861e-6 | 9.060e-5 | 5.072e-5 | 3.905e-6 | 8.760e-5 | 4.838e-5 |
| 11 | 2.404e-5 | 1.272e-4 | 8.790e-5 | 2.413e-5 | 1.248e-4 | 9.014e-5 |
| 12 | 5.642e-6 | 1.032e-4 | 1.328e-4 | 5.458e-6 | 1.137e-4 | 1.308e-4 |
| 13 | 4.740e-6 | 1.052e-4 | 7.077e-5 | 4.550e-6 | 9.309e-5 | 7.259e-5 |
| 14 | 5.273e-5 | 1.164e-4 | 9.810e-5 | 7.564e-5 | 1.099e-4 | 1.000e-4 |
| 15 | 4.191e-5 | 3.003e-4 | 2.757e-4 | 4.162e-5 | 2.981e-4 | 2.707e-4 |
| 16 | 5.897e-5 | 1.033e-4 | 1.005e-4 | 5.289e-5 | 9.923e-5 | 1.052e-4 |
| 17 | 6.074e-6 | 1.238e-4 | 1.285e-4 | 5.948e-6 | 1.234e-4 | 1.239e-4 |
| 18 | 6.157e-6 | 1.435e-4 | 1.792e-4 | 6.133e-6 | 1.406e-4 | 1.818e-4 |
| 19 | 4.281e-5 | 1.867e-4 | 1.784e-4 | 4.308e-5 | 1.738e-4 | 1.913e-4 |
| 20 | 1.054e-5 | 1.499e-4 | 1.927e-4 | 8.262e-6 | 1.424e-4 | 1.849e-4 |
| 21 | 1.724e-5 | 1.394e-4 | 1.813e-4 | 2.415e-5 | 1.460e-4 | 1.755e-4 |
| 22 | 8.676e-6 | 1.288e-4 | 2.485e-4 | 8.203e-6 | 1.348e-4 | 2.471e-4 |
| 23 | 1.472e-5 | 1.927e-4 | 2.407e-4 | 1.433e-5 | 1.763e-4 | 2.499e-4 |
| 24 | 1.741e-5 | 1.440e-4 | 1.839e-4 | 1.745e-5 | 1.369e-4 | 1.818e-4 |
| 25 | 8.053e-5 | 1.384e-4 | 1.661e-4 | 7.557e-5 | 1.345e-4 | 1.601e-4 |
| 26 | 1.396e-5 | 1.622e-4 | 1.510e-4 | 1.308e-5 | 1.534e-4 | 1.452e-4 |
| 27 | 2.845e-5 | 2.767e-4 | 2.434e-4 | 2.781e-5 | 2.721e-4 | 2.677e-4 |
| 28 | 1.587e-5 | 1.584e-4 | 2.045e-4 | 1.518e-5 | 1.546e-4 | 2.161e-4 |
| 29 | 9.836e-6 | 1.727e-4 | 1.670e-4 | 9.821e-6 | 1.739e-4 | 1.710e-4 |
| 30 | 1.723e-4 | 2.352e-4 | 2.325e-4 | 2.494e-4 | 2.137e-4 | 2.112e-4 |
| 31 | 8.828e-5 | 7.340e-4 | 5.999e-4 | 8.458e-5 | 7.307e-4 | 5.981e-4 |
| 32 | 5.255e-5 | 1.848e-4 | 2.032e-4 | 1.067e-4 | 1.968e-4 | 2.020e-4 |
| 33 | 1.250e-5 | 1.932e-4 | 2.200e-4 | 1.371e-5 | 1.942e-4 | 2.089e-4 |
| 34 | 1.260e-5 | 1.831e-4 | 2.377e-4 | 1.284e-5 | 1.763e-4 | 2.211e-4 |
| 35 | 8.483e-5 | 4.337e-4 | 4.630e-4 | 7.866e-5 | 3.898e-4 | 4.257e-4 |
| 36 | 1.965e-5 | 3.151e-4 | 3.025e-4 | 1.864e-5 | 3.099e-4 | 3.122e-4 |
| 37 | 8.786e-5 | 1.792e-4 | 1.433e-4 | 5.311e-5 | 1.802e-4 | 1.511e-4 |
| 38 | 2.211e-5 | 1.358e-4 | 1.626e-4 | 2.248e-5 | 1.422e-4 | 1.839e-4 |
| 39 | 8.662e-5 | 4.444e-4 | 5.177e-4 | 9.856e-5 | 4.535e-4 | 4.597e-4 |
| 40 | 2.922e-5 | 1.617e-4 | 1.262e-4 | 2.720e-5 | 1.581e-4 | 1.261e-4 |
| 41 | 3.509e-5 | 1.530e-4 | 1.992e-4 | 2.930e-5 | 1.774e-4 | 1.891e-4 |
| 42 | 6.992e-5 | 1.126e-4 | 1.693e-4 | 7.174e-5 | 1.176e-4 | 1.637e-4 |
| 43 | 6.951e-5 | 2.417e-4 | 4.050e-4 | 7.130e-5 | 2.378e-4 | 3.995e-4 |
| 44 | 2.232e-5 | 4.362e-4 | 3.523e-4 | 3.059e-5 | 4.064e-4 | 3.012e-4 |
| 45 | 1.605e-4 | 2.565e-4 | 2.670e-4 | 2.404e-4 | 2.595e-4 | 2.794e-4 |
| 46 | 1.189e-4 | 2.676e-4 | 3.598e-4 | 1.202e-4 | 2.698e-4 | 3.957e-4 |
| 47 | 2.774e-4 | 1.554e-4 | 1.348e-4 | 3.082e-4 | 1.633e-4 | 1.174e-4 |
