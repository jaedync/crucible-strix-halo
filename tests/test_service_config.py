"""The service (./crucible start) serves the configuration that the public README speed table measured (v0.34.0, Track R).

The words below are the README words. tools/crucible_cli/measured.py is the one table that the service and the Docker
run line read. These tests pin the table, the service and docs/DOCKER.md to the README words, so that none of them drifts.
No GPU, no network."""
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from crucible_cli_fakes import ROOT  # noqa: E402

from crucible_cli import launch, measured, packs  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402

# The measured words (README, "Every row ..." paragraph).
README_ENV = {
    "CRUCIBLE_PLE_HOST": "1",
    "CRUCIBLE_COMPACT_STATE": "1",
    "CRUCIBLE_PLE_FETCH": "1",
    "CRUCIBLE_DRAFT_HEAD": "q2",
    "CRUCIBLE_DRAFT_CHAIN": "hc8,lead",
    "CRUCIBLE_PF_LAUNCH_ROWS": "16384+128",
}
README_FLAGS = {"--streams": "4", "--ctx": "262144", "--prefill": "4096", "--conversation-cache-mib": "9700",
                "--host-reserve-mib": "11264"}


def service_command(pack):
    s = settings_mod.Settings(repo=Path("/r"), pack=pack, models_dir=Path("/m"), state_dir=Path("/h/s"))
    return launch.server_command(s, "py")


def flag_values(argv):
    return {a: argv[i + 1] for i, a in enumerate(argv[:-1]) if a.startswith("--")}


def docker_run_words():
    text = (ROOT / "docs/DOCKER.md").read_text()
    block = re.search(r"## Run\n.*?```sh\n(.*?)```", text, re.S).group(1)
    words = shlex.split(block.replace("\\\n", " "))
    env = dict(w.split("=", 1) for i, w in enumerate(words) if i and words[i - 1] == "-e")
    return env, words[words.index("/models/pack"):]


class MeasuredTableTest(unittest.TestCase):
    def test_the_table_holds_the_readme_words(self):
        self.assertEqual(dict(measured.ENV), README_ENV)
        self.assertEqual(dict(measured.SERVER_FLAGS), README_FLAGS)

    def test_the_conversation_cache_value_is_one_number(self):
        self.assertEqual(measured.SERVER_FLAGS["--conversation-cache-mib"], "9700")


class ServiceMatchesReadmeTest(unittest.TestCase):
    def test_both_packs_get_every_readme_word(self):
        for pack in packs.PACKS:
            env, argv = service_command(pack)
            for key, value in README_ENV.items():
                self.assertEqual(env.get(key), value, f"{pack} {key}")

    def test_no_pack_pins_the_launch_rows(self):
        for spec in packs.PACKS.values():
            self.assertNotIn("CRUCIBLE_PF_LAUNCH_ROWS", spec.extra_env, spec.bpw)

    def test_the_service_argv_holds_the_readme_flags(self):
        for pack in packs.PACKS:
            env, argv = service_command(pack)
            values = flag_values(argv)
            for flag, value in README_FLAGS.items():
                self.assertEqual(values.get(flag), value, f"{pack} {flag}")
            self.assertIn("--vision", argv)

    def test_the_service_sets_no_other_engine_word(self):
        env, argv = service_command("4.05")
        self.assertEqual(set(env), set(README_ENV))


class DockerRunMatchesServiceTest(unittest.TestCase):
    def test_the_docker_run_line_sets_the_readme_words(self):
        env, args = docker_run_words()
        self.assertEqual(env, README_ENV)
        values = flag_values(args)
        for flag, value in README_FLAGS.items():
            self.assertEqual(values.get(flag), value, flag)
        self.assertIn("--vision", args)

    def test_the_docker_page_names_the_service(self):
        text = (ROOT / "docs/DOCKER.md").read_text()
        self.assertIn("./crucible start", text)


class ConversationCacheNoticeTest(unittest.TestCase):
    def test_off_cache_gets_one_line_that_names_the_flag_and_the_service_value(self):
        from server.__main__ import conversation_cache_notice
        line = conversation_cache_notice(0)
        self.assertIsNotNone(line)
        self.assertNotIn("\n", line)
        self.assertIn("--conversation-cache-mib", line)
        self.assertIn(measured.SERVER_FLAGS["--conversation-cache-mib"], line)
        self.assertIn(measured.SERVER_FLAGS["--host-reserve-mib"], line)

    def test_a_set_cache_gets_no_line(self):
        from server.__main__ import conversation_cache_notice
        self.assertIsNone(conversation_cache_notice(9700))



class UserEnvWinsTest(unittest.TestCase):
    """A word that the user sets in the environment beats the measured default (direct start and serve.sh)."""

    USER = {"CRUCIBLE_DRAFT_HEAD": "q4", "CRUCIBLE_DRAFT_CHAIN": "hc8", "CRUCIBLE_PLE_HOST": "0",
            "CRUCIBLE_COMPACT_STATE": "0", "CRUCIBLE_PLE_FETCH": "0", "CRUCIBLE_PF_LAUNCH_ROWS": "8192+128"}

    def settings(self, pack="4.05"):
        return settings_mod.Settings(repo=Path("/r"), pack=pack, models_dir=Path("/m"), state_dir=Path("/h/s"))

    def test_process_env_keeps_every_user_word(self):
        env, _ = launch.server_command(self.settings(), "py")
        merged = launch.process_env(dict(self.USER, PATH="/bin"), env)
        for key, value in self.USER.items():
            self.assertEqual(merged[key], value, key)

    def test_process_env_fills_the_words_that_are_unset(self):
        env, _ = launch.server_command(self.settings(), "py")
        merged = launch.process_env({"PATH": "/bin", "CRUCIBLE_DRAFT_HEAD": "exl3"}, env)
        self.assertEqual(merged["CRUCIBLE_DRAFT_HEAD"], "exl3")
        for key, value in README_ENV.items():
            if key != "CRUCIBLE_DRAFT_HEAD":
                self.assertEqual(merged[key], value, key)

    def test_an_empty_user_word_counts_as_set(self):
        env, _ = launch.server_command(self.settings(), "py")
        self.assertEqual(launch.process_env({"CRUCIBLE_DRAFT_CHAIN": ""}, env)["CRUCIBLE_DRAFT_CHAIN"], "")

    def test_the_sidecar_removal_still_works(self):
        self.assertNotIn("CRUCIBLE_MTP_SIDECAR", launch.process_env({"CRUCIBLE_MTP_SIDECAR": "/x"}, {"CRUCIBLE_MTP_SIDECAR": None}))

    def run_script(self, user):
        """Run serve.sh with a stand-in for the python program that prints the CRUCIBLE_ environment, and return it."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "py"
            fake.write_text("#!/bin/sh\nenv | grep '^CRUCIBLE_' | sort\n")
            fake.chmod(0o755)
            s = self.settings()
            s.repo = Path(tmp)
            script = Path(tmp) / "serve.sh"
            script.write_text(launch.serve_script(s, str(fake)))
            script.chmod(0o755)
            base = {k: v for k, v in os.environ.items() if not k.startswith("CRUCIBLE_")}
            out = subprocess.run([str(script)], env=dict(base, **user), capture_output=True, text=True, check=True).stdout
        return dict(line.split("=", 1) for line in out.splitlines())

    def test_serve_script_keeps_every_user_word(self):
        got = self.run_script(self.USER)
        for key, value in self.USER.items():
            self.assertEqual(got[key], value, key)

    def test_serve_script_sets_the_measured_words_when_unset(self):
        got = self.run_script({})
        for key, value in README_ENV.items():
            self.assertEqual(got[key], value, key)


if __name__ == "__main__":
    unittest.main()
