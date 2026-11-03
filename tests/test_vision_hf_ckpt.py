"""tools/vision_hf_ckpt.py: the index lists the 333 HF tower tensors in a sparse shard; vision_k6 is linked unless left out."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "vision_hf_ckpt.py")


class HfCkptTest(unittest.TestCase):
    def make(self, *flags):
        src, out = tempfile.mkdtemp(), os.path.join(tempfile.mkdtemp(), "out")
        json.dump({"metadata": {"x": 1}, "weight_map": {"lm_head.weight": "m.safetensors"}},
                  open(os.path.join(src, "model.safetensors.index.json"), "w"))
        for name in ("m.safetensors", "vision_k6.safetensors", "config.json"):
            open(os.path.join(src, name), "wb").write(b"x")
        run = subprocess.run([sys.executable, "-I", TOOL, src, out, *flags], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        return out

    def test_index_and_shard(self):
        out = self.make()
        index = json.load(open(os.path.join(out, "model.safetensors.index.json")))
        self.assertEqual(index["metadata"], {"x": 1})
        names = [n for n, s in index["weight_map"].items() if s == "hf-tower.safetensors"]
        self.assertEqual(len(names), 333)
        self.assertEqual(index["weight_map"]["lm_head.weight"], "m.safetensors")
        self.assertIn("model.visual.blocks.0.attn.qkv.weight", names)
        self.assertNotIn("model.visual.blocks.0.attn.q_proj.trellis", names)
        path = os.path.join(out, "hf-tower.safetensors")
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        self.assertEqual(sorted(header), sorted(names))
        self.assertTrue(all(h["dtype"] == "BF16" for h in header.values()))
        self.assertEqual(os.path.getsize(path), 8 + n + max(h["data_offsets"][1] for h in header.values()))
        self.assertLess(os.stat(path).st_blocks * 512, 1 << 20)  # sparse
        self.assertTrue(os.path.islink(os.path.join(out, "vision_k6.safetensors")))

    def test_without_k6(self):
        out = self.make("--without-k6")
        self.assertFalse(os.path.lexists(os.path.join(out, "vision_k6.safetensors")))
        self.assertTrue(os.path.islink(os.path.join(out, "m.safetensors")))


if __name__ == "__main__":
    unittest.main()
