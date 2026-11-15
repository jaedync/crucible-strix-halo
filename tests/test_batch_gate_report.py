import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from tools import batch_gate_report as b


class BatchReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # The start time comes from the file clock, as the stamps it is compared with. time.time_ns() can
        # be ahead of a stamp that the kernel writes just after it (coarse timestamps, a clock step).
        marker = self.root / "started"
        marker.write_bytes(b"")
        self.started = min(marker.stat().st_mtime_ns, marker.stat().st_ctime_ns)
        self.width = patch.object(b, "WIDTH", 1)
        self.vocab = patch.object(b, "VOCAB", 4)
        self.width.start(); self.vocab.start()
        self.addCleanup(self.width.stop); self.addCleanup(self.vocab.stop)
        self.addCleanup(self.temp.cleanup)
        for stream in range(8):
            self.populate(self.root / "single" / f"stream_{stream}", False)
        for size in range(1, 9):
            for stream in range(size):
                self.populate(self.root / f"batch{size}" / f"stream_{stream}", True)

    def populate(self, path, batch):
        path.mkdir(parents=True)
        for layer in range(49): (path / f"h_{layer:02d}.f32").write_bytes(bytes(8 * 4))
        for step in range(8): (path / f"logits_{step}.f32").write_bytes(bytes(4 * 4))
        if batch:
            (path / "ids-dump.txt").write_text(",".join(["1"] * 8))
            (path / "ids-perf.txt").write_text(",".join(["1"] * 33))
        else:
            (path / "ids.txt").write_text(",".join(["1"] * 33))

    def test_valid_raw_files_and_full_hashes(self):
        result = b.validate_artifacts(self.root, self.started)
        self.assertEqual(len(result["batches"]), 8)
        self.assertEqual(len(result["batches"]["8"]["7"]["logits"]), 8)
        self.assertEqual(len(result["batches"]["8"]["7"]["ids_perf"]["ids"]), 33)
        self.assertEqual(len(result["batches"]["8"]["7"]["dumps"]), 49)

    def test_fixture_start_time_is_not_after_any_fixture_stamp(self):
        stamps = [min(path.stat().st_mtime_ns, path.stat().st_ctime_ns) for path in self.root.rglob("*") if path.is_file()]
        self.assertGreater(len(stamps), 400)
        self.assertLessEqual(self.started, min(stamps))

    def test_fixture_does_not_depend_on_the_wall_clock(self):
        # A wall clock ahead of the file clock (coarse stamps, a clock step) once flaked the valid test.
        ahead = time.time_ns() + 10**9
        with patch.object(time, "time_ns", return_value=ahead):
            self.setUp()
        b.validate_artifacts(self.root, self.started)

    def test_wrong_logits_fail(self):
        path = self.root / "batch8/stream_7/logits_4.f32"
        path.write_bytes(b"x" * 16)
        with self.assertRaises(ValueError): b.validate_artifacts(self.root, self.started)

    def test_wrong_ids_fail(self):
        (self.root / "batch2/stream_1/ids-perf.txt").write_text(",".join(["2"] * 33))
        with self.assertRaises(ValueError): b.validate_artifacts(self.root, self.started)

    def test_stale_missing_extra_and_symlink_fail(self):
        for kind in ("stale", "missing", "extra", "symlink"):
            with self.subTest(kind=kind):
                path = self.root / "batch1/stream_0/logits_0.f32"
                if kind == "stale":
                    import os
                    os.utime(path, ns=(1, 1))
                elif kind == "missing": path.unlink()
                elif kind == "extra": (path.parent / "extra").write_text("bad")
                else:
                    path.unlink(); path.symlink_to(self.root / "single/stream_0/logits_0.f32")
                with self.assertRaises(ValueError): b.validate_artifacts(self.root, self.started)
                if kind == "extra": (path.parent / "extra").unlink()
                elif path.exists() or path.is_symlink(): path.unlink()
                path.write_bytes(bytes(16))

    def prepare_run(self):
        engine = self.root / "engine"; engine.write_bytes(b"candidate")
        probe = self.root / "probe"; probe.write_bytes(b"batch-probe")
        record = self.root / "build.json"
        source = {name: "0" * 64 for name in b.REQUIRED_SOURCES}
        frozen = self.root / "source"
        for name in source:
            path = frozen / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        source = {name: b.digest(frozen / name) for name in source}
        import hashlib
        fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        common = dict(schema=1, builder="crucible-batch-v1", builder_sha256=b.digest(b.BUILDER),
                      source_hashes=source, source_root=str(frozen), source_commit="test-commit", numeric_flags=b.BUILD_FLAGS,
                      compiler=dict(path="/opt/rocm/bin/hipcc", hipcc_sha256="0"*64, clang_sha256="0"*64, version="test"))
        command = ["/opt/rocm/bin/hipcc", *b.BUILD_FLAGS, "-o", str(engine), str(frozen / "engine/engine.hip"), "-lpthread"]
        record.write_text(json.dumps(dict(**common, kind="candidate", binary_sha256=b.digest(engine), command=command)))
        probe_record = self.root / "probe.build.json"
        binding = [f'-DBATCHING_ENGINE_SHA="{b.digest(engine)}"', f'-DBATCHING_SOURCE_SHA="{fingerprint}"']
        command = ["/opt/rocm/bin/hipcc", *b.BUILD_FLAGS, *binding, "-o", str(probe), str(frozen / "bench/batch_gates.hip"), "-lpthread"]
        probe_record.write_text(json.dumps(dict(**common, kind="batch-probe", binary_sha256=b.digest(probe),
                                               candidate_binary_sha256=b.digest(engine), command=command)))
        prompts = self.root / "prompts.txt"
        paths = []
        for index in range(8):
            path = self.root / f"prompt{index}"; path.write_text("1,2,3"); paths.append(str(path))
        prompts.write_text("\n".join(paths))
        baseline = self.root / "baseline"; baseline.mkdir()
        for name in b.DUMPS: (baseline / name).write_bytes(b"baseline")
        kwargs = dict(checkpoint=self.root, engine=engine, probe=probe, build_record=record, probe_build_record=probe_record,
                      prompts=prompts, primary_ids=Path(paths[0]), output=self.root / "output", baseline_dump=baseline,
                      gpu_run="gpu-run")
        source_sha = b.hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return kwargs, dict(candidate_sha256=b.digest(engine), source_sha256=source_sha)

    def test_records_from_either_layout_verify(self):
        # a record built before the layout move keeps its keys.
        kwargs, _ = self.prepare_run()
        lay = b.layout.Layout(b.layout.load_map())
        for keys in ({lay.row(n).old if lay.row(n) else n for n in b.REQUIRED_SOURCES},
                     {lay.row(n).new if lay.row(n) else n for n in b.REQUIRED_SOURCES}):
            with patch.object(b, "REQUIRED_SOURCES", keys):
                b.validate_builds(kwargs["engine"], kwargs["probe"], kwargs["build_record"], kwargs["probe_build_record"])

    def test_frozen_source_and_flags_must_match(self):
        kwargs, binding = self.prepare_run()
        (self.root / "source/engine/layers/ops.h").write_text("wrong source")
        # The reason matters: a path that misses the frozen file fails as an incomplete manifest instead.
        with self.assertRaisesRegex(ValueError, "frozen engine source changed"): b.validate_builds(kwargs["engine"], kwargs["probe"], kwargs["build_record"], kwargs["probe_build_record"])
        (self.root / "source/engine/layers/ops.h").write_text("engine/layers/ops.h")
        record = json.loads(kwargs["probe_build_record"].read_text())
        record["numeric_flags"] = ["-ffast-math"]
        kwargs["probe_build_record"].write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "does not match the fixed dual builder"): b.validate_builds(kwargs["engine"], kwargs["probe"], kwargs["build_record"], kwargs["probe_build_record"])

    def test_wrong_binary_binding_fails_before_queue(self):
        import subprocess
        kwargs, binding = self.prepare_run()
        binding["candidate_sha256"] = "f" * 64
        with patch.object(b, "run_tree", return_value=subprocess.CompletedProcess([], 0, json.dumps(binding), "")) as run:
            with self.assertRaises(ValueError): b.run(**kwargs)
            self.assertEqual(run.call_count, 1)

    def test_queue_wait_reaches_the_gpu_run_command(self):
        import subprocess
        kwargs, binding = self.prepare_run()
        seen = []
        def execute(command, **options):
            if command[-1] == "--build-info":
                return subprocess.CompletedProcess(command, 0, json.dumps(binding), "")
            seen.append((command[3:5], options["timeout"]))
            return subprocess.CompletedProcess(command, 1, b"", b"failed")
        with patch.object(b, "run_tree", side_effect=execute), self.assertRaises(ValueError):
            b.run(**kwargs, queue_wait=21600)
        self.assertEqual(seen, [(["-w", "21600"], 21915)])

    def test_run_uses_file_clock_for_artifact_start(self):
        import subprocess
        kwargs, binding = self.prepare_run()
        def execute(command, **options):
            if command[-1] == "--build-info":
                return subprocess.CompletedProcess(command, 0, json.dumps(binding), "")
            root = Path(command[-2])
            for stream in range(8): self.populate(root / "single" / f"stream_{stream}", False)
            for size in range(1, 9):
                for stream in range(size): self.populate(root / f"batch{size}" / f"stream_{stream}", True)
            return subprocess.CompletedProcess(command, 0, b"stdout", b"")
        ahead = time.time_ns() + 10**9
        with patch.object(b.time, "time_ns", return_value=ahead), \
                patch.object(b, "run_tree", side_effect=execute):
            result = b.run(**kwargs)
        self.assertEqual(result["status"], "PASS")

    def test_run_command_and_report(self):
        import subprocess
        kwargs, binding = self.prepare_run()
        def execute(command, **options):
            if command[-1] == "--build-info":
                return subprocess.CompletedProcess(command, 0, json.dumps(binding), "")
            self.assertEqual(command[:9], ["gpu-run", "-p", "2", "-w", "7200", "--", "timeout", "-k", "10"])
            self.assertEqual(command[9], "300")
            root = Path(command[-2])
            for stream in range(8): self.populate(root / "single" / f"stream_{stream}", False)
            for size in range(1, 9):
                for stream in range(size): self.populate(root / f"batch{size}" / f"stream_{stream}", True)
            return subprocess.CompletedProcess(command, 0, b"stdout", b"")
        with patch.object(b, "run_tree", side_effect=execute): result = b.run(**kwargs)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue((kwargs["output"] / "report.json").is_file())
        self.assertEqual(result["build_binding"], binding)
        with self.assertRaises(ValueError): b.run(**kwargs)

    def run_with_sidecar(self, change=False):
        import os
        import subprocess
        kwargs, binding = self.prepare_run()
        sidecar = self.root / "side.safetensors"
        sidecar.write_bytes(b"sidecar bytes")
        def execute(command, **options):
            if command[-1] == "--build-info":
                return subprocess.CompletedProcess(command, 0, json.dumps(binding), "")
            root = Path(command[-2])
            for stream in range(8): self.populate(root / "single" / f"stream_{stream}", False)
            for size in range(1, 9):
                for stream in range(size): self.populate(root / f"batch{size}" / f"stream_{stream}", True)
            if change: sidecar.write_bytes(b"sidecar bytez")
            return subprocess.CompletedProcess(command, 0, b"stdout", b"")
        with patch.dict(os.environ, {"CRUCIBLE_MTP_SIDECAR": str(sidecar)}), \
                patch.object(b, "run_tree", side_effect=execute):
            return kwargs, b.run(**kwargs)

    def test_report_records_the_sidecar(self):
        kwargs, result = self.run_with_sidecar()
        expected = dict(bytes=13, sha256=b.hashlib.sha256(b"sidecar bytes").hexdigest())
        self.assertEqual(result["mtp_sidecar"], expected)
        self.assertEqual(json.loads((kwargs["output"] / "report.json").read_text())["mtp_sidecar"], expected)

    def test_sidecar_change_during_the_run_fails(self):
        with self.assertRaisesRegex(ValueError, "sidecar changed"):
            self.run_with_sidecar(change=True)

    def test_bad_sidecar_fails_before_any_subprocess(self):
        import os
        kwargs, _ = self.prepare_run()
        with patch.dict(os.environ, {"CRUCIBLE_MTP_SIDECAR": str(self.root)}), \
                patch.object(b, "run_tree") as run:
            with self.assertRaisesRegex(ValueError, "regular file"): b.run(**kwargs)
            self.assertEqual(run.call_count, 0)

    def test_first_prompt_must_match_primary_ids(self):
        import subprocess
        kwargs, binding = self.prepare_run()
        wrong = self.root / "wrong_ids"; wrong.write_text("1,2,4")
        kwargs["primary_ids"] = wrong
        with patch.object(b, "run_tree", return_value=subprocess.CompletedProcess([], 0, json.dumps(binding), "")) as execute:
            with self.assertRaises(ValueError): b.run(**kwargs)
            self.assertEqual(execute.call_count, 1)

    def test_failed_probe_cannot_pass(self):
        import subprocess
        kwargs, binding = self.prepare_run()
        with patch.object(b, "run_tree", side_effect=[subprocess.CompletedProcess([], 0, json.dumps(binding), ""),
                                                            subprocess.CompletedProcess([], 1, b"PASS", b"bad")]):
            with self.assertRaises(ValueError): b.run(**kwargs)
        self.assertEqual(json.loads((kwargs["output"] / "report.json").read_text())["status"], "FAIL")

    def test_environment_does_not_publish_secrets(self):
        hidden = "private-test-value"
        env = dict(CRUCIBLE_STATE_CACHE="1", CRUCIBLE_API_KEY=hidden, HSA_CREDENTIAL=hidden)
        names=('CRUCIBLE_PF_TILES','CRUCIBLE_FUSE','CRUCIBLE_DECODE_RING',
               'CRUCIBLE_WARMUP','CRUCIBLE_PERSISTENT_GDN',
               'CRUCIBLE_PF_EXACT_SPLIT', 'CRUCIBLE_COMPACT_STATE', 'CRUCIBLE_QSA_CAPTURE_INDEX_ONLY', 'CRUCIBLE_HC_WMMA', 'CRUCIBLE_PF_PRUNE')
        result = b.runtime_environment({**env,**dict.fromkeys(names,'1')})
        for name in names:self.assertEqual(result[name],"1")
        self.assertEqual(result["CRUCIBLE_STATE_CACHE"], "1")
        self.assertNotIn(hidden, json.dumps(result))
        self.assertNotIn("CRUCIBLE_API_KEY", result)
        self.assertNotIn("HSA_CREDENTIAL", result)

    def test_input_hash_recheck(self):
        path = self.root / "input"
        path.write_text("hello")
        tracked = {str(path): b.file_record(path)}
        b.recheck(tracked)
        path.write_text("changed")
        with self.assertRaises(ValueError): b.recheck(tracked)

if __name__ == "__main__": unittest.main()
