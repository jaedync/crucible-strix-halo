#!/usr/bin/env python3
"""Weight error of int8 HC (fn and up) for a few group sizes, on real HC tensors.

    hc8_qerr.py MODEL_DIR [layers...]          int8, groups 32 to 160
    HC8_BITS=6 hc8_qerr.py MODEL_DIR [layers...]   6 bits (qmax 31), fn G 64 and up G 32, plain and with a
                                                 64-point Hadamard rotation of each K group (rotated error)

fn = fp16(down_or_inject * (1 + hc_norm)), as the engine loader builds it.
Each group of G values along K gets s = fp16(absmax / 127) and
q = clip(rint(w / s), -127, 127); W' = fp16(q * s). Prints the relative
Frobenius error ||W' - W|| / ||W|| and the max abs error per matrix kind.
"""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor(model, index, name):
    shard = model / index[name]
    h, base = header(shard)
    info = h[name]
    a, b = info["data_offsets"]
    dt = {"F16": np.float16, "BF16": np.uint16, "F32": np.float32}[info["dtype"]]
    with open(shard, "rb") as f:
        f.seek(base + a)
        raw = np.frombuffer(f.read(b - a), dtype=dt).reshape(info["shape"])
    if info["dtype"] == "BF16":
        raw = (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32)


def quant(w, g, qmax=127):
    """w: fp16 [N][K]; returns W' fp16."""
    n, k = w.shape
    wg = w.astype(np.float32).reshape(n, k // g, g)
    amax = np.abs(wg).max(axis=2, keepdims=True)
    s = (amax / qmax).astype(np.float16).astype(np.float32)
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.rint(wg / s), -qmax, qmax)
    return (q * s).astype(np.float16).reshape(n, k)


def hadamard(n):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def quant_rot(w, g, qmax, rot=64):
    """Quantize w @ blockdiag(H) along K, rotate back: the error a rotated kernel (x rotated too) would see."""
    n, k = w.shape
    h = hadamard(rot)
    wr = (w.astype(np.float64).reshape(n, k // rot, rot) @ h).reshape(n, k).astype(np.float16)
    q = quant(wr, g, qmax).astype(np.float64).reshape(n, k // rot, rot) @ h.T
    return q.reshape(n, k)


def err(w, wq):
    d = wq.astype(np.float64) - w.astype(np.float64)
    return np.linalg.norm(d) / np.linalg.norm(w.astype(np.float64)), np.abs(d).max(), np.abs(w).max()


def main():
    model = Path(sys.argv[1])
    layers = [int(x) for x in sys.argv[2:]] or [0, 12, 24, 36, 47]
    index = json.load(open(model / "model.safetensors.index.json"))["weight_map"]
    P = "model.language_model.layers."
    for l in layers:
        for site in ("attn_hyper_connection.", "mlp_hyper_connection."):
            p = f"{P}{l}.{site}"
            nw = tensor(model, index, p + "hc_norm.weight")
            down = np.concatenate([tensor(model, index, p + "input_mix_weight_down.weight"),
                                   tensor(model, index, p + "block_inject_weight.weight")])
            fn = (down * (1 + nw)).astype(np.float16)
            up = tensor(model, index, p + "input_mix_weight_up.weight").astype(np.float16)
            line = [f"L{l:02d} {site[:4]}"]
            bits = int(os.environ.get("HC8_BITS", "8"))
            if bits != 8:
                qm = 2 ** (bits - 1) - 1
                for name, w, g in (("fn", fn, 64), ("up", up, 32)):
                    e8 = err(w, quant(w, g))[0]
                    e = err(w, quant(w, g, qm))[0]
                    er = err(w, quant_rot(w, g, qm))[0]
                    line.append(f"{name} g{g} int8 {e8:.2e} int{bits} {e:.2e} rot {er:.2e} ratio {e / e8:.2f}/{er / e8:.2f}")
                print(" | ".join(line), flush=True)
                continue
            for g in (32, 64, 128):
                e, m, a = err(fn, quant(fn, g))
                line.append(f"fn g{g} rel {e:.2e} max {m:.1e}/{a:.1e}")
            for g in (32, 80, 160):
                e, m, a = err(up, quant(up, g))
                line.append(f"up g{g} rel {e:.2e} max {m:.1e}/{a:.1e}")
            print(" | ".join(line), flush=True)


if __name__ == "__main__":
    main()
