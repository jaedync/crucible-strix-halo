"""E3: the engine died (an out-of-memory kill on Nov 14), and the server stayed on its port without a model. The server
now exits 3 in every mode with one log line that names the cause: the engine exit code, or the signal of a kill."""
import os
import signal
import unittest

from server.tests.server_run import ServerRun, port_free


def stop_line(text):
    lines = [line for line in text.splitlines() if line.startswith("The engine stopped")]
    assert len(lines) == 1, text
    return lines[0]


class EngineDeathCauseTest(unittest.TestCase):
    def test_a_plain_server_names_the_exit_code_and_the_port(self):
        run = ServerRun(self, env={"FAKE_EXIT_AFTER_READY": "1"})
        run.wait_ready()
        self.assertEqual(run.wait_exit(20), 3, run.text())
        self.assertTrue(port_free(run.port))
        line = stop_line(run.text())
        self.assertIn("exit code 1", line)
        self.assertIn(f"frees port {run.port}", line)

    def test_a_killed_engine_is_named_with_its_signal_and_the_oom_hint(self):
        run = ServerRun(self)
        run.wait_ready()
        os.kill(run.engine_pid(), signal.SIGKILL)
        self.assertEqual(run.wait_exit(20), 3, run.text())
        line = stop_line(run.text())
        self.assertIn("signal 9, SIGKILL", line)
        self.assertIn("out-of-memory", line)


if __name__ == "__main__":
    unittest.main()
