"""Compile the actual generation driver with a CPU engine and counted MTP calls."""
import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
# The asan+ubsan compile takes over 60 s when the host is loaded (load average 50 on the test machine). Neither step tests a hang, so
# the limits only stop a stuck process and are far above any loaded run.
COMPILE_TIMEOUT_S = 1800
RUN_TIMEOUT_S = 600


class GenCliCpuTests(unittest.TestCase):
    def build_and_run(self, test):
        source_path = ROOT / "engine/cli/gen_cli.h"
        source = source_path.read_text()
        # Batch and resident entry points need real device allocation. Keep the generation driver unchanged.
        core, separator, _ = source.partition('#include "batch_cli.h"')
        self.assertTrue(separator, "the CPU fixture needs the generation-driver boundary")
        # Resident and batch declarations now precede batch_cli.h. They are not part of RunGen.
        serve = re.search(r'^#include "(?:[^"]*/)?serve\.h"', core, re.M)
        core = core[:serve.start()] if serve else core
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "gen_cli_cpu.h").write_text(core)
            command = [os.environ.get("CXX", "g++"), "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined",
                       "-I", str(build), "-I", str(source_path.parent), "-I", str(ROOT / "engine"), str(ROOT / f"tests/{test}.cpp"),
                       "-o", str(build / "gen-test")]
            subprocess.run(command, check=True, timeout=COMPILE_TIMEOUT_S)
            subprocess.run([str(build / "gen-test")], check=True, timeout=RUN_TIMEOUT_S)

    def test_one_token_requests_do_not_warm_mtp(self):
        self.build_and_run("gen_cli_cpu_test")

    def test_jobs_parser_load_checks_and_state_reset(self):
        self.build_and_run("gen_jobs_cpu_test")


if __name__ == "__main__": unittest.main()
