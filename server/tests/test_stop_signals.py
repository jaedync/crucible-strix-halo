"""REPORT C an earlier change: a second SIGTERM or SIGINT during the state save ended the server early; the engine went on saving and
held its memory, so the next start failed. Now the server ignores further stop signals until the engine has exited."""
import os
import signal
import time
import unittest

from server.tests.server_run import ServerRun, alive


class StopSignalTest(unittest.TestCase):
    def saving_server(self):
        run = ServerRun(self, env={"FAKE_STATE_DISK": "1", "FAKE_IGNORE_TERM": "1", "FAKE_FLUSH_DELAY": "3"},
                        args=("--conversation-cache-mib", "64", "--state-dir", "STATE"))
        return run

    def check_waits_for_the_engine(self, second):
        run = self.saving_server()
        run.wait_ready()
        engine = run.engine_pid()
        os.kill(run.proc.pid, signal.SIGTERM)
        time.sleep(0.5)
        os.kill(run.proc.pid, second)
        time.sleep(0.3)
        os.kill(run.proc.pid, second)
        run.wait_exit(60)
        self.assertFalse(alive(engine), "the server exited while its engine still saved:\n" + run.text())
        self.assertNotIn("Traceback", run.text())

    def test_a_second_sigterm_during_the_save_waits_for_the_engine(self):
        self.check_waits_for_the_engine(signal.SIGTERM)

    def test_a_sigint_during_the_save_waits_for_the_engine(self):
        self.check_waits_for_the_engine(signal.SIGINT)


if __name__ == "__main__":
    unittest.main()
