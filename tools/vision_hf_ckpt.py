#!/usr/bin/env python3
"""Make a test checkpoint folder that stores the vision tower as plain HF tensors, like a third-party pack (issue 3).

  python3 tools/vision_hf_ckpt.py SRC_PACK OUT_DIR [--without-k6]

SRC_PACK is an EXL3 pack folder. OUT_DIR links every file of the pack, and its index also lists 333 BF16
`model.visual.*` tensors in a new shard `hf-tower.safetensors` (fused qkv, no EXL3 tensors). The shard is a sparse
file of zeros: an engine must not read it, so the check shows that the engine ignores it and uses vision_k6.safetensors.
  --without-k6  leave out the link to vision_k6.safetensors: `serve --vision 1` must stop with the missing-file message
"""
import argparse
import json
import os
import struct

P = "model.visual."
SHARD = "hf-tower.safetensors"
HIDDEN = 2560


def hf_tower():
    """(name, shape) of the 333 tensors of the HF vision tower, in the order of the original model."""
    out = [("patch_embed.proj.weight", [1152, 3, 2, 16, 16]), ("patch_embed.proj.bias", [1152]),
           ("pos_embed.weight", [2304, 1152])]
    for i in range(27):
        b = f"blocks.{i}."
        out += [(b + "attn.qkv.weight", [3456, 1152]), (b + "attn.qkv.bias", [3456]),
                (b + "attn.proj.weight", [1152, 1152]), (b + "attn.proj.bias", [1152]),
                (b + "mlp.linear_fc1.weight", [4352, 1152]), (b + "mlp.linear_fc1.bias", [4352]),
                (b + "mlp.linear_fc2.weight", [1152, 4352]), (b + "mlp.linear_fc2.bias", [1152])]
        out += [(b + n, [1152]) for n in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")]
    out += [("merger.norm.weight", [1152]), ("merger.norm.bias", [1152]),
            ("merger.linear_fc1.weight", [4608, 4608]), ("merger.linear_fc1.bias", [4608]),
            ("merger.linear_fc2.weight", [HIDDEN, 4608]), ("merger.linear_fc2.bias", [HIDDEN])]
    return [(P + n, s) for n, s in out]


def write_shard(path, tensors):
    header, off = {}, 0
    for name, shape in tensors:
        size = 2
        for d in shape:
            size *= d
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + size]}
        off += size
    text = json.dumps(header).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text)
        f.truncate(8 + len(text) + off)  # zeros, no disk used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--without-k6", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=False)
    skip = {"model.safetensors.index.json"} | ({"vision_k6.safetensors"} if args.without_k6 else set())
    for name in sorted(os.listdir(args.src)):
        if name not in skip:
            os.symlink(os.path.realpath(os.path.join(args.src, name)), os.path.join(args.out, name))
    tower = hf_tower()
    assert len(tower) == 333
    with open(os.path.join(args.src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    for name, _ in tower:
        assert name not in index["weight_map"], name
        index["weight_map"][name] = SHARD
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    write_shard(os.path.join(args.out, SHARD), tower)
    print(f"{args.out}: {len(tower)} HF tower tensors in {SHARD}" + (", no vision_k6.safetensors" if args.without_k6 else ""))


if __name__ == "__main__":
    main()
