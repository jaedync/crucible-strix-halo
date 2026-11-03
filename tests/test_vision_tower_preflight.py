"""server.vision.tower_fault: --vision on a checkpoint without the EXL3 vision tower stops at startup (issue 3)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server import vision  # noqa: E402

HF = "model.visual.blocks.0.attn.qkv.weight"
EXL = "model.visual.blocks.0.attn.q_proj.trellis"
COMMAND = "hf download turboderp/Qwen3.8-Flash-Next-exl3 vision_k6.safetensors --revision 4.05bpw_h6_ng6 --local-dir"


def make(names, k6=False):
    folder = tempfile.mkdtemp()
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {n: "model-00009-of-00009.safetensors" for n in names}}, f)
    if k6:
        open(os.path.join(folder, "vision_k6.safetensors"), "wb").close()
    return folder


class TowerFaultTest(unittest.TestCase):
    def test_hf_tower_without_file_names_the_file_the_repo_and_the_command(self):
        text = vision.tower_fault(make([HF, "model.visual.patch_embed.proj.weight"]))
        self.assertIn("vision_k6.safetensors", text)
        self.assertIn("turboderp/Qwen3.8-Flash-Next-exl3", text)
        self.assertIn(COMMAND, text)
        self.assertIn("only when the weights of the HF tower are the stock weights", text)

    def test_the_file_makes_it_run(self):
        self.assertIsNone(vision.tower_fault(make([HF], k6=True)))

    def test_exl3_tower_in_the_index_runs(self):
        self.assertIsNone(vision.tower_fault(make([EXL, "model.visual.patch_embed.proj.weight"])))

    def test_no_tower_at_all_is_left_to_the_engine(self):
        self.assertIsNone(vision.tower_fault(make(["model.language_model.embed_tokens.weight"])))

    def test_missing_or_bad_index_is_left_to_the_engine(self):
        self.assertIsNone(vision.tower_fault(tempfile.mkdtemp()))
        folder = tempfile.mkdtemp()
        open(os.path.join(folder, "model.safetensors.index.json"), "w").write("{")
        self.assertIsNone(vision.tower_fault(folder))

    def test_server_exits_before_the_engine_starts(self):
        folder = make([HF])
        run = subprocess.run([sys.executable, "-m", "server", folder, "--vision", "--engine", "/nonexistent/engine"],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 2)
        self.assertIn("vision_k6.safetensors", run.stderr)
        self.assertNotIn("Traceback", run.stderr)


if __name__ == "__main__":
    unittest.main()
