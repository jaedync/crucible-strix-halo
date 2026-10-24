"""Check the public build interface without ROCm or a checkpoint."""
import importlib.util
import os
import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATED_CPU_FIXTURES = ("tests/gen_cli_cpu_test.cpp", "tests/gen_jobs_cpu_test.cpp")
# Built by tools/split2048_probe.py --build. Its generated header is not in git.
TOOL_BUILT_PROBES = ("bench/split2048.hip",)
PINNED_REFERENCE_PROBES = (
    "bench/isa_hc_regs.hip", "bench/isa_hc_pitch.hip", "bench/isa_hc_bounds.hip",
)


SPEC_DIGEST_VARIABLE = "CRUCIBLE_BATCH_SPEC_SOURCE_SHA"


def gates_source_digest():
    # The digest that tools/build_batch_gates.py records for this tree.
    spec = importlib.util.spec_from_file_location("build_batch_gates", ROOT / "tools" / "build_batch_gates.py")
    gates = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gates)
    return gates.source_digest(gates.source_hashes(ROOT))


class BuildTests(unittest.TestCase):
    def make_dry_run(self, target, env=None, extra=()):
        return subprocess.run(
            ["make", "-n", "-B", target, *extra], cwd=ROOT, text=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )

    def dry_run(self, target, env=None):
        result = self.make_dry_run(target, env)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_ci_actions_use_complete_commit_ids(self):
        workflow = (ROOT / ".github/workflows/cpu.yml").read_text()
        references = re.findall(r"uses:\s+(\S+)", workflow)
        self.assertTrue(references)
        for reference in references:
            self.assertRegex(reference, r"^[^@]+@[0-9a-f]{40}$")

    def test_default_engine(self):
        commands = self.dry_run("all")
        self.assertIn("engine/engine.hip", commands)
        self.assertIn("--offload-arch=gfx1151", commands)
        self.assertIn("build/engine/crucible", commands)

    def test_cpu_target_never_builds_or_runs_hip(self):
        commands = self.dry_run("test-cpu")
        self.assertNotIn("hipcc", commands)
        self.assertNotIn("engine/engine.hip", commands)
        for directory in ("tests", "bench/suite"):
            self.assertIn("discover -s " + directory, commands)
        # serve-tests 44850fd runs the server suite with pytest (parametrized cases).
        # -n runs the files in parallel (pytest-xdist; make test-cpu and test-server).
        self.assertRegex(commands, r"pytest -q -n \d+ server/tests")
        self.assertIn("tests/data/draft/blocks65536.txt", commands)

    def test_every_binary_has_a_build_target(self):
        commands = self.dry_run("binaries")
        for directory in ("engine", "bench", "bench/suite", "tests"):
            for source in (ROOT / directory).iterdir():
                if source.suffix in (".hip", ".cpp", ".cc"):
                    relative = str(source.relative_to(ROOT))
                    if relative in GENERATED_CPU_FIXTURES or relative in TOOL_BUILT_PROBES:
                        self.assertNotIn(relative, commands)
                    elif relative in PINNED_REFERENCE_PROBES:
                        self.assertFalse(relative in commands, f"pinned probe selected by default: {relative}")
                        target = "build/" + str(pathlib.Path(relative).with_suffix(""))
                        self.assertIn(relative, self.dry_run(target))
                    else:
                        self.assertIn(relative, commands)

    def test_generated_driver_runs_through_python_only(self):
        commands = self.dry_run("cpu-tests")
        self.assertNotIn("tests/gen_cli_cpu_test.cpp", commands)
        self.assertIn("discover -s tests", self.dry_run("test-cpu"))

    def test_cpu_assertions_stay_enabled(self):
        commands = self.dry_run("cpu-tests")
        self.assertIn("-UNDEBUG", commands)

    def test_spec_probe_gets_the_gates_source_digest(self):
        env = {key: value for key, value in os.environ.items() if key != SPEC_DIGEST_VARIABLE}
        commands = self.dry_run("build/bench/batch_spec_probe", env)
        self.assertIn(f"-D{SPEC_DIGEST_VARIABLE}='\"{gates_source_digest()}\"'", commands)

    def test_spec_probe_digest_comes_from_the_caller_when_set(self):
        pinned = "ab" * 32
        commands = self.dry_run("build/bench/batch_spec_probe", dict(os.environ, **{SPEC_DIGEST_VARIABLE: pinned}))
        self.assertIn(f"-D{SPEC_DIGEST_VARIABLE}='\"{pinned}\"'", commands)
        self.assertNotIn(gates_source_digest(), commands)

    def test_spec_probe_digest_is_derived_when_caller_sets_it_empty(self):
        commands = self.dry_run("build/bench/batch_spec_probe", dict(os.environ, **{SPEC_DIGEST_VARIABLE: ""}))
        self.assertIn(f"-D{SPEC_DIGEST_VARIABLE}='\"{gates_source_digest()}\"'", commands)

    def test_spec_probe_stops_when_the_derived_digest_is_missing(self):
        env = {key: value for key, value in os.environ.items() if key != SPEC_DIGEST_VARIABLE}
        result = self.make_dry_run("build/bench/batch_spec_probe", env, ("PYTHON=/nonexistent",))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("64 lowercase hex", result.stderr)
        self.assertNotIn("hipcc", result.stdout)

    def test_spec_probe_stops_when_the_caller_digest_is_malformed(self):
        result = self.make_dry_run("build/bench/batch_spec_probe", dict(os.environ, **{SPEC_DIGEST_VARIABLE: "abc"}))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("64 lowercase hex", result.stderr)
        self.assertNotIn("hipcc", result.stdout)


if __name__ == "__main__":
    unittest.main()
