"""a missing --streams means --streams 1 (the batched stack); --serial keeps the old serial stack
for one train (deleted in train 20)."""
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import server.__main__ as startup
from server.api import ApiError, validate_request
from server.batch_backend import BatchedEngineBackend
from server.batch_scheduler import INTERLEAVE_MIN_ROWS

FAKE = Path(__file__).parent / "fake_batch_engine.py"


def run_main(argv):
    """Run server.__main__.main with both backends mocked; return (serial mock, batched mock)."""
    listener = MagicMock()
    listener.serve_forever.side_effect = KeyboardInterrupt
    with patch.object(sys, "argv", ["crucible", "checkpoint", *argv]), patch.object(startup.signal, "signal"), \
            patch.object(startup, "ModelTokenizer"), patch.object(startup, "EngineBackend") as serial, \
            patch("server.batch_backend.BatchedEngineBackend") as batched, \
            patch.object(startup, "ThreadingHTTPServer", return_value=listener):
        assert startup.main() == 0
    return serial, batched


def refused(argv):
    listener = MagicMock()
    listener.serve_forever.side_effect = KeyboardInterrupt  # a missing refusal must fail, not block on a real port
    with patch.object(startup, "ThreadingHTTPServer", return_value=listener), patch.object(startup.signal, "signal"), \
            patch.object(sys, "argv", ["crucible", "checkpoint", *argv]), \
            patch.object(startup, "ModelTokenizer") as load, patch.object(startup, "EngineBackend") as serial, \
            patch("server.batch_backend.BatchedEngineBackend") as batched, \
            patch("sys.stderr", new_callable=io.StringIO) as err, \
            unittest.TestCase().assertRaises(SystemExit) as exit_info:
        startup.main()
    assert exit_info.exception.code == 2
    load.assert_not_called()
    serial.assert_not_called()
    batched.assert_not_called()
    return err.getvalue()


class DefaultStackTest(unittest.TestCase):
    def test_a_missing_streams_flag_starts_the_batched_stack_with_one_stream(self):
        serial, batched = run_main([])
        serial.assert_not_called()
        self.assertEqual(batched.call_args.kwargs["streams"], 1)

    def test_an_explicit_streams_flag_is_unchanged(self):
        for count in (1, 2, 8):
            serial, batched = run_main(["--streams", str(count)])
            serial.assert_not_called()
            self.assertEqual(batched.call_args.kwargs["streams"], count)

    def test_the_serial_flag_starts_the_old_stack(self):
        serial, batched = run_main(["--serial"])
        batched.assert_not_called()
        serial.assert_called_once()

    def test_the_serial_flag_keeps_the_old_options(self):
        serial, _ = run_main(["--serial", "--conversation-cache-mib", "32", "--prefill", "64", "--lookup", "0"])
        self.assertEqual(serial.call_args.kwargs["engine_options"], ("--session-replay", "1"))
        self.assertEqual(serial.call_args.kwargs["lookup"], 0)
        self.assertEqual(serial.call_args.args[3:5], (32, 64))

    def test_the_default_stack_gets_the_batched_options(self):
        _, batched = run_main(["--conversation-cache-mib", "32", "--prefill", "64", "--session-replay", "1"])
        self.assertEqual(batched.call_args.kwargs["conversation_cache_mib"], 32)
        self.assertEqual(batched.call_args.kwargs["prefill"], 64)
        self.assertIs(batched.call_args.kwargs["session_replay"], True)
        self.assertEqual(batched.call_args.kwargs["prefill_interleave"], 10)

    def test_the_batch_only_options_work_without_a_streams_flag(self):
        _, batched = run_main(["--conversation-cache-mib", "32", "--cache-probation", "1", "--cache-idle-minutes", "5"])
        self.assertEqual(batched.call_args.kwargs["streams"], 1)
        self.assertEqual(batched.call_args.kwargs["cache_idle_minutes"], 5)


class SerialFlagRefusalTest(unittest.TestCase):
    def test_serial_with_streams_is_an_error(self):
        for count in ("1", "2"):
            text = refused(["--serial", "--streams", count])
            self.assertIn("--serial", text)
            self.assertIn("--streams", text)

    def test_batch_only_options_are_refused_with_serial(self):
        for extra, name in ((["--short-streams", "1", "--short-ctx", "64"], "--short-streams"),
                            (["--conversation-cache-mib", "32", "--state-dir", "state"], "--state-dir"),
                            (["--conversation-cache-mib", "32", "--cache-idle-minutes", "5"], "--cache-idle-minutes"),
                            (["--conversation-cache-mib", "32", "--cache-probation", "1"], "--cache-probation")):
            text = refused(["--serial", *extra])
            self.assertIn(name, text)
            self.assertIn("--serial", text)

    def test_the_other_batch_options_are_refused_with_serial(self):
        for extra, name in ((["--slot-resume", "1"], "--slot-resume"), (["--slot-resume", "0"], "--slot-resume"),
                            (["--slot-checkpoints", "1"], "--slot-checkpoints"),
                            (["--packed-op", "step"], "--packed-op"), (["--packed-op", "verify_step"], "--packed-op"),
                            (["--prefill-interleave", "0"], "--prefill-interleave"),
                            (["--prefill-interleave", "10"], "--prefill-interleave"),
                            (["--interleave-min-rows", "0"], "--interleave-min-rows")):
            text = refused(["--serial", *extra])
            self.assertIn(name, text)
            self.assertIn("--serial", text)

    def test_the_batch_option_defaults_are_unchanged(self):
        _, batched = run_main([])
        kwargs = batched.call_args.kwargs
        self.assertEqual((kwargs["packed_op"], kwargs["prefill_interleave"]), ("verify_step", 10))
        self.assertEqual(kwargs["interleave_min_rows"], INTERLEAVE_MIN_ROWS)
        self.assertEqual((kwargs["slot_resume"], kwargs["slot_checkpoints"]), (None, None))

    def test_short_slots_still_need_two_streams_without_a_flag(self):
        refused(["--short-streams", "1", "--short-ctx", "64"])

    def test_the_streams_help_no_longer_says_spec_zero(self):
        out = io.StringIO()
        with redirect_stdout(out), patch.object(sys, "argv", ["server", "--help"]), self.assertRaises(SystemExit):
            startup.main()
        text = " ".join(out.getvalue().split())
        self.assertNotIn("Requests must use spec=0", text)
        self.assertIn("--serial", text)
        self.assertIn("Serve with the serial worker", text)
        self.assertNotIn("removed in train", text)


class SerialOnlyOptionTest(unittest.TestCase):
    """draft_head full, draft_adapt 1 and draft_q 0 stay serial only. The error names the field and --serial."""

    def setUp(self):
        popen = subprocess.Popen
        with patch("server.backend.subprocess.Popen",
                   side_effect=lambda *args, **kwargs: popen([sys.executable, str(FAKE)], **kwargs)):
            self.backend = BatchedEngineBackend("unused", "unused", context=128, streams=2)
        self.addCleanup(self.backend.close)

    def test_each_serial_only_option_is_refused_with_a_clear_error(self):
        for extra, field in ((dict(draft_head="full"), "draft_head"), (dict(draft_adapt=1), "draft_adapt"),
                             (dict(draft_q=0), "draft_q")):
            request = validate_request(dict(prompt="x", spec=7, **extra), "completion", "crucible")
            with self.assertRaises(ApiError) as caught:
                self.backend.submit([1], request)
            message = str(caught.exception)
            self.assertIn(field, message, extra)
            self.assertIn("--serial", message, extra)
            self.assertEqual(caught.exception.param, "spec")
        self.assertEqual(self.backend.pending_tokens, 0)

    def test_the_error_names_only_the_field_that_was_sent(self):
        request = validate_request(dict(prompt="x", spec=7, draft_adapt=1), "completion", "crucible")
        with self.assertRaises(ApiError) as caught:
            self.backend.submit([1], request)
        self.assertNotIn("draft_head", str(caught.exception))
        self.assertNotIn("draft_q", str(caught.exception))

    def test_spec_zero_requests_still_ignore_the_draft_fields(self):
        request = validate_request(dict(prompt="x", max_tokens=2, spec=0, draft_adapt=1), "completion", "crucible")
        self.backend.submit([1], request)


if __name__ == "__main__":
    unittest.main()
