"""CPU malformed-checkpoint tests. Run with Python's unittest module."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def compiler(env):
    """The C++ compiler: CXX as make test-cpu passes it, else make's default."""
    return env.get("CXX") or "c++"


class CompilerChoiceTests(unittest.TestCase):
    def test_default_is_the_make_default(self):
        # make test-cpu passes CXX; a direct unittest run must use make's own default, c++.
        self.assertEqual(compiler({}), "c++")
        self.assertEqual(compiler({"CXX": "g++"}), "g++")
        self.assertEqual(compiler({"CXX": ""}), "c++")

class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.build.name) / "probe"
        cxx = compiler(os.environ)
        if shutil.which(cxx) is None:
            raise unittest.SkipTest(f"no C++ compiler {cxx!r}; set CXX to build the checkpoint probe")
        command = [cxx, "-std=c++20", "-O1", "-g",
                   "-fsanitize=address,undefined", str(ROOT / "tests/checkpoint_probe.cpp"),
                   "-o", str(cls.exe)]
        subprocess.run(command, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def probe(self, header=None, payload=b"\0\0", raw=None, linear=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                name: "shard.safetensors" for name in (header or {"x": 0})}}))
            encoded = json.dumps(header).encode()
            encoded += b" " * (-len(encoded) % 8)
            (root / "shard.safetensors").write_bytes(raw if raw is not None else
                                                    struct.pack("<Q", len(encoded)) + encoded + payload)
            return subprocess.run([str(self.exe), directory, "load"] + (["linear"] if linear else []),
                                  capture_output=True, text=True)

    def tensor(self, **changes):
        return dict(dtype="F16", shape=[1], data_offsets=[0, 2]) | changes

    def rejected(self, result, text):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(text, result.stderr)
        self.assertNotIn("AddressSanitizer", result.stderr)
        self.assertNotIn("runtime error:", result.stderr)

    def test_unterminated_string(self):
        result = subprocess.run([str(self.exe), '"'], capture_output=True, text=True)
        self.rejected(result, "unterminated string")

    def test_short_file(self):
        self.rejected(self.probe(raw=b"123"), "shorter than 8 bytes")

    def test_header_past_end(self):
        self.rejected(self.probe(raw=struct.pack("<Q", 200)), "header length")

    def test_offsets_past_end(self):
        self.rejected(self.probe({"x": self.tensor(data_offsets=[0, 3])}), "tensor x")

    def test_offset_length(self):
        self.rejected(self.probe({"x": self.tensor(data_offsets=[0])}), "two offsets")

    def test_shape_payload_mismatch(self):
        self.rejected(self.probe({"x": self.tensor(shape=[2])}), "payload size")

    def test_negative_offset(self):
        self.rejected(self.probe({"x": self.tensor(data_offsets=[-1, 1])}), "outside")

    def test_reversed_offset(self):
        self.rejected(self.probe({"x": self.tensor(data_offsets=[2, 0])}), "outside")

    def test_missing_shape(self):
        tensor = self.tensor()
        del tensor["shape"]
        self.rejected(self.probe({"x": tensor}), "missing dtype, shape")

    def test_unknown_dtype(self):
        self.rejected(self.probe({"x": self.tensor(dtype="INVALID")}), "unsupported dtype")

    def test_negative_shape(self):
        self.rejected(self.probe({"x": self.tensor(shape=[-1])}), "negative shape")

    def test_malformed_json(self):
        for header in (b'{"x":', b'{"x":{"dtype":"F16" "shape":[]}}',
                       b'{"x":{"shape":[1,]}}', b'{"x":{"shape":[--1]}}',
                       b'{"x":{"shape":[01]}}', b'{"x":{"shape":[9223372036854775808]}}',
                       b'{"x":{"data_offsets":[', b'{"x":{"dtype":"F16',
                       b'{"x":{"shape":nullX}}',
                       b'{"__metadata__":' + b'[' * 70 + b'0' + b']' * 70 + b'}'):
            raw = struct.pack("<Q", len(header)) + header
            result = self.probe(raw=raw)
            self.rejected(result, "checkpoint error")
            self.assertIn("shard.safetensors", result.stderr)

    def test_string_escapes(self):
        for string in ('"bad' + chr(92), r'"bad\q"', r'"\u123"', r'"\uD800"', r'"\uDC00"'):
            result = subprocess.run([str(self.exe), string], capture_output=True, text=True)
            self.rejected(result, "checkpoint error")
        for string in (r'"\u0061"', r'"\uD83D\uDE00"', r'"quote\""'):
            result = subprocess.run([str(self.exe), string], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_shape_overflow(self):
        self.rejected(self.probe({"x": self.tensor(shape=[2**62, 4])}), "overflow")

    def test_valid(self):
        result = self.probe({"x": self.tensor()})
        self.assertEqual(result.returncode, 0, result.stderr)

    def linear(self, width=48, dtype="F16", scale=128):
        tensors = {"x": self.tensor()}
        offset = 8
        for name, dt, shape, size in [
            ("trellis", "I16", [8, 8, width], 8*8*width*2),
            ("suh", dtype, [scale], scale*2), ("svh", "F16", [128], 256),
            ("mul1", "I32", [], 4)]:
            tensors["lin." + name] = dict(dtype=dt, shape=shape, data_offsets=[offset, offset+size])
            offset += size
        payload = bytes(offset-4) + struct.pack("<I", 0x83DCD12D)
        return self.probe(tensors, payload, linear=True)

    def test_wrong_dtype(self):
        self.rejected(self.linear(dtype="BF16"), "expected dtype F16")

    def test_unsupported_k(self):
        # K1, K9 and half widths: the probe's matrix is not a routed expert, so K3.5 (56) is rejected too.
        for width in (16, 24, 40, 56, 144):
            self.rejected(self.linear(width=width), "supported K")

    def test_scale_length(self):
        self.rejected(self.linear(scale=127), "shape")

    def test_valid_linear(self):
        # K2..K6, and K7/K8 (kWideDecodeK: the h6_ng8 packs' dense and shared matrices).
        for width in (32, 48, 64, 80, 96, 112, 128):
            result = self.linear(width=width)
            self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__":
    unittest.main()
