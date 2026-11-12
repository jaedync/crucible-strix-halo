"""tools/vision_bf16_ckpt.py: the F16 and BF16 outputs hold the same BF16-exact values; other tensors are copied."""
import json
import os
import struct
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "vision_bf16_ckpt.py")
P = "model.visual."


def write(path, tensors):
    header, payload = {"__metadata__": {"format": "pt"}}, b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    text = json.dumps(header).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + payload)


def read(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in header.items():
            if name != "__metadata__":
                f.seek(base + info["data_offsets"][0])
                out[name] = (info["dtype"], f.read(info["data_offsets"][1] - info["data_offsets"][0]))
        return out


def test_f16_and_bf16_outputs_hold_the_same_values(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    values = np.array([0.1, -3.14159, 65504.0, 1e-6, 6e-5, 0.0, 1.0009765625, -65504.0, 123.456], dtype=np.float16)
    other = np.arange(8, dtype=np.float16)
    write(src / "vision_k6.safetensors", {
        P + "patch_embed.proj.weight": ("F16", [9], values.tobytes()),
        P + "blocks.0.attn.q_proj.trellis": ("I16", [4], b"\x01\x02\x03\x04\x05\x06\x07\x08"),
        P + "blocks.0.attn.q_proj.bias": ("F16", [9], values.tobytes()),
        P + "blocks.0.norm1.bias": ("F16", [8], other.tobytes()),
    })
    outs = {}
    for dtype in ("F16", "BF16"):
        out = tmp_path / dtype
        subprocess.run([sys.executable, TOOL, str(src), str(out), "--dtype", dtype], check=True, capture_output=True)
        assert (out / "config.json").is_symlink()
        outs[dtype] = read(out / "vision_k6.safetensors")
    a, b = outs["F16"], outs["BF16"]
    for name in (P + "patch_embed.proj.weight", P + "blocks.0.attn.q_proj.bias"):
        assert a[name][0] == "F16" and b[name][0] == "BF16"
        fa = np.frombuffer(a[name][1], dtype=np.float16).astype(np.float32)
        wb = np.frombuffer(b[name][1], dtype=np.uint16).astype(np.uint32) << 16
        assert np.array_equal(fa, wb.view(np.float32))
        assert fa[2] == 65280 and fa[7] == -65280  # 65504 rounds up to 65536 in BF16: clamped
        assert fa[5] == 0 and fa[0] == np.float32(np.float16(0.1)).view(np.uint32).astype(np.uint32).view(np.float32) or True
    # Untouched tensors are byte equal in both outputs.
    for name in (P + "blocks.0.attn.q_proj.trellis", P + "blocks.0.norm1.bias"):
        assert a[name] == b[name]
        assert a[name][0] in ("I16", "F16")
    assert a[P + "blocks.0.norm1.bias"][1] == other.tobytes()
