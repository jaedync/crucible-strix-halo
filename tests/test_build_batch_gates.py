import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tools import build_batch_gates as builder
from tools import batch_gate_report as report


class BatchBuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.source = self.root / "repo"
        for name in report.REQUIRED_SOURCES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.identity = dict(path=str(builder.COMPILER), hipcc_sha256="0"*64,
                             clang_sha256="1"*64, version="test compiler")

    def compile(self, command, **kwargs):
        self.assertEqual(kwargs["env"]["PATH"], "/opt/rocm/bin:/usr/bin:/bin")
        self.assertNotIn("HIPCC_COMPILE_FLAGS_APPEND", kwargs["env"])
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(output.name.encode())
        return subprocess.CompletedProcess(command, 0, b"", b"")

    def test_frozen_shared_source_and_fixed_flags(self):
        with patch.object(builder, "compiler_identity", return_value=self.identity), \
             patch.object(builder, "run_tree", side_effect=self.compile):
            result = builder.build(self.source, self.root / "build", "test-commit")
        candidate = json.loads(result["engine_record"].read_text())
        probe = json.loads(result["probe_record"].read_text())
        self.assertEqual(candidate["numeric_flags"], probe["numeric_flags"])
        self.assertEqual(candidate["source_hashes"], probe["source_hashes"])
        self.assertEqual(candidate["binary_sha256"], probe["candidate_binary_sha256"])
        report.validate_builds(result["engine"], result["probe"], result["engine_record"], result["probe_record"])
        (self.source / "engine/layers/ops.h").write_text("new working source")
        report.validate_builds(result["engine"], result["probe"], result["engine_record"], result["probe_record"])
        with self.assertRaises(ValueError): builder.build(self.source, self.root / "build", "another")

    def test_source_mutation_during_compilation_fails(self):
        def mutate(command, **kwargs):
            result = self.compile(command, **kwargs)
            (self.root / "build/source/engine/layers/ops.h").write_text("different arithmetic")
            return result
        with patch.object(builder, "compiler_identity", return_value=self.identity), \
             patch.object(builder, "run_tree", side_effect=mutate):
            with self.assertRaises(ValueError): builder.build(self.source, self.root / "build", "test-commit")
        self.assertFalse((self.root / "build/batch_gates.build.json").exists())

    def test_compilation_failure_cannot_write_qualified_record(self):
        with patch.object(builder, "compiler_identity", return_value=self.identity), \
             patch.object(builder, "run_tree", return_value=subprocess.CompletedProcess([], 1, b"", b"compile failed")):
            with self.assertRaises(ValueError): builder.build(self.source, self.root / "build", "test-commit")
        self.assertFalse((self.root / "build/crucible.build.json").exists())

if __name__ == "__main__": unittest.main()
