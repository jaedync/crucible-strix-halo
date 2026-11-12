#!/usr/bin/env python3
"""Make a test checkpoint folder whose vision tensors hold BF16-exact values (issue 3).

  python3 tools/vision_bf16_ckpt.py SRC_CKPT OUT_DIR --dtype F16|BF16

SRC_CKPT is a pack folder with a `vision_k6.safetensors` file. OUT_DIR gets a link to every other file of the pack
and a new `vision_k6.safetensors` (about 560 MB). The tensors that the BF16 loader change converts are rewritten: the
patch embedding weight and bias, and the bias of every vision linear. Each F16 value is rounded to BF16
(nearest even) and stored as that value:
  --dtype F16   F16 words (the value is exact in F16, so the stored F16 word is the BF16 value)
  --dtype BF16  BF16 words (the same values; the loader converts them to F16 at load)
A value that BF16 rounding would push above 65504 is set to 65280 (the largest BF16 value below 65504).
Both outputs hold the same values, so an engine must give byte-identical vision embeddings from them.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

P = "model.visual."
CHUNK = 64 << 20


def touched(name, header):
    if name in (P + "patch_embed.proj.weight", P + "patch_embed.proj.bias"):
        return True
    return name.endswith(".bias") and name[:-5] + ".trellis" in header


def to_bf16_exact(raw):
    """F16 words (bytes) -> (BF16 words as uint16, F16 words as uint16, number clamped)."""
    f32 = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    bits = f32.view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    f32 = rounded.view(np.float32).copy()
    big = np.abs(f32) > 65504
    f32[big] = np.copysign(np.float32(65280), f32[big])
    half = f32.astype(np.float16)
    assert np.array_equal(half.astype(np.float32), f32), "a BF16 value is not exact in F16"
    return (f32.view(np.uint32) >> 16).astype(np.uint16), half.view(np.uint16), int(big.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--dtype", required=True, choices=["F16", "BF16"])
    args = ap.parse_args()
    src = os.path.join(args.src, "vision_k6.safetensors")
    os.makedirs(args.out, exist_ok=False)
    for name in sorted(os.listdir(args.src)):
        if name != "vision_k6.safetensors":
            os.symlink(os.path.realpath(os.path.join(args.src, name)), os.path.join(args.out, name))
    with open(src, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        new_header = json.loads(json.dumps(header))
        count = clamped = 0
        for name, info in header.items():
            if name != "__metadata__" and touched(name, header):
                if info["dtype"] != "F16":
                    sys.exit(f"{name}: source dtype {info['dtype']}, want F16")
                new_header[name]["dtype"] = args.dtype
                count += 1
        text = json.dumps(new_header, separators=(",", ":")).encode()
        text += b" " * ((8 - len(text) % 8) % 8)
        order = sorted((i["data_offsets"][0], n) for n, i in header.items() if n != "__metadata__")
        out_path = os.path.join(args.out, "vision_k6.safetensors")
        with open(out_path, "wb") as out:
            out.write(struct.pack("<Q", len(text)) + text)
            pos = 0
            for begin, name in order:
                end = header[name]["data_offsets"][1]
                if begin != pos:
                    sys.exit(f"{name}: tensors are not contiguous")
                f.seek(base + begin)
                if touched(name, header):
                    bf, half, nbig = to_bf16_exact(f.read(end - begin))
                    clamped += nbig
                    out.write((half if args.dtype == "F16" else bf).tobytes())
                else:
                    left = end - begin
                    while left:
                        data = f.read(min(left, CHUNK))
                        out.write(data)
                        left -= len(data)
                pos = end
    print(f"{out_path}: {count} tensors stored as {args.dtype}, {clamped} values clamped to 65280")


if __name__ == "__main__":
    main()
