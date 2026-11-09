"""sd_notify (server/sd_notify.py): the notify text, the abstract address, the environment strip, and the --systemd flags."""
import os
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import server.__main__ as startup
from server.backend import engine_environment
from server.sd_notify import ENV_NAMES, Notifier


def listener(address):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(address)
    sock.settimeout(2)
    return sock


class NotifierTest(unittest.TestCase):
    def test_sends_text_to_a_path_socket(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "notify")
            sock = listener(path)
            notifier = Notifier.from_environ({"NOTIFY_SOCKET": path})
            self.assertTrue(notifier.enabled)
            self.assertTrue(notifier.send("READY=1\nSTATUS=ready"))
            self.assertEqual(sock.recv(256), b"READY=1\nSTATUS=ready")
            notifier.close()
            sock.close()

    def test_abstract_address_with_an_at_sign(self):
        name = f"crucible-test-{os.getpid()}-{id(self)}"
        sock = listener("\0" + name)
        notifier = Notifier.from_environ({"NOTIFY_SOCKET": "@" + name})
        self.assertTrue(notifier.enabled)
        self.assertTrue(notifier.send("WATCHDOG=1"))
        self.assertEqual(sock.recv(256), b"WATCHDOG=1")
        sock.close()

    def test_refuses_an_address_that_is_not_a_path_or_abstract(self):
        for bad in ("vsock:1:2", "relative/path", "", "127.0.0.1:9"):
            notifier = Notifier.from_environ({"NOTIFY_SOCKET": bad})
            self.assertFalse(notifier.enabled, bad)
            self.assertFalse(notifier.send("READY=1"))

    def test_no_socket_means_disabled_and_silent(self):
        notifier = Notifier.from_environ({})
        self.assertFalse(notifier.enabled)
        self.assertFalse(notifier.send("READY=1"))

    def test_the_three_names_leave_the_environment(self):
        environ = {"NOTIFY_SOCKET": "/x", "WATCHDOG_USEC": "30000000", "WATCHDOG_PID": str(os.getpid()), "PATH": "/bin"}
        Notifier.from_environ(environ)
        self.assertEqual(environ, {"PATH": "/bin"})
        self.assertEqual(set(ENV_NAMES), {"NOTIFY_SOCKET", "WATCHDOG_USEC", "WATCHDOG_PID"})

    def test_watchdog_interval_from_usec_and_pid(self):
        self.assertEqual(Notifier.from_environ({"NOTIFY_SOCKET": "/x", "WATCHDOG_USEC": "30000000"}).watchdog_seconds, 30.0)
        mine = {"NOTIFY_SOCKET": "/x", "WATCHDOG_USEC": "3000000", "WATCHDOG_PID": str(os.getpid())}
        self.assertEqual(Notifier.from_environ(mine).watchdog_seconds, 3.0)
        other = {"NOTIFY_SOCKET": "/x", "WATCHDOG_USEC": "3000000", "WATCHDOG_PID": str(os.getpid() + 1)}
        self.assertIsNone(Notifier.from_environ(other).watchdog_seconds)
        for bad in ("0", "-5", "abc", ""):
            self.assertIsNone(Notifier.from_environ({"NOTIFY_SOCKET": "/x", "WATCHDOG_USEC": bad}).watchdog_seconds, bad)

    def test_a_send_failure_logs_one_line_and_never_raises(self):
        lines = []
        with tempfile.TemporaryDirectory() as folder:
            notifier = Notifier.from_environ({"NOTIFY_SOCKET": os.path.join(folder, "gone")}, log=lines.append)
            self.assertFalse(notifier.send("READY=1"))
            self.assertFalse(notifier.send("READY=1"))
        self.assertEqual(len(lines), 1)
        self.assertIn("notify", lines[0])


class EngineEnvironmentTest(unittest.TestCase):
    def test_the_engine_never_sees_the_service_names(self):
        environ = {"NOTIFY_SOCKET": "/x", "WATCHDOG_USEC": "1", "WATCHDOG_PID": "2", "PATH": "/bin", "FAKE_X": "1"}
        env, removed = engine_environment(environ)
        self.assertEqual(env, {"PATH": "/bin", "FAKE_X": "1"})
        self.assertEqual(removed, [])  # a service name is not an ignored engine switch: no note


class SystemdFlagsTest(unittest.TestCase):
    def run_main(self, argv, environ=None, serve=None):
        server = MagicMock()
        server.serve_forever.side_effect = serve or KeyboardInterrupt
        with patch.object(sys, "argv", ["crucible", "checkpoint", *argv]), \
                patch.dict(os.environ, environ or {}, clear=False), \
                patch.object(startup, "ModelTokenizer"), patch.object(startup, "ThreadingHTTPServer", return_value=server), \
                patch("server.batch_backend.BatchedEngineBackend") as backend:
            backend.return_value.fatal = None
            backend.return_value.last_event_at = 1e18
            server.http_alive_at = 1e18
            return startup.main(), backend, server

    def test_systemd_without_notify_socket_is_a_usage_error(self):
        with patch.dict(os.environ):
            os.environ.pop("NOTIFY_SOCKET", None)
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--systemd"]), self.assertRaises(SystemExit) as stop:
                startup.main()
        self.assertEqual(stop.exception.code, 2)

    def test_stall_flag_range_and_needs_systemd(self):
        for argv in (["--watchdog-stall-seconds", "50"], ["--systemd", "--watchdog-stall-seconds", "9"],
                     ["--systemd", "--watchdog-stall-seconds", "601"]):
            with patch.object(sys, "argv", ["crucible", "checkpoint", *argv]), patch.dict(os.environ, {"NOTIFY_SOCKET": "/x"}), \
                    self.assertRaises(SystemExit) as stop:
                startup.main()
            self.assertEqual(stop.exception.code, 2, argv)

    def test_ready_and_stopping_order_with_a_real_socket(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "notify")
            sock = listener(path)
            code, backend, server = self.run_main(["--systemd"], {"NOTIFY_SOCKET": path})
            messages = []
            sock.settimeout(.2)
            try:
                while True: messages.append(sock.recv(512).decode())
            except socket.timeout:
                pass
            sock.close()
        self.assertEqual(code, 0)
        self.assertTrue(messages[0].startswith("STATUS=loading the model"), messages)
        ready = next(i for i, m in enumerate(messages) if "READY=1" in m)
        stopping = next(i for i, m in enumerate(messages) if "STOPPING=1" in m)
        self.assertLess(ready, stopping)
        self.assertNotIn("NOTIFY_SOCKET", os.environ)

    def test_a_fatal_engine_gives_exit_3(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "notify")
            sock = listener(path)
            server = MagicMock()
            stop = threading.Event()
            server.serve_forever.side_effect = lambda: stop.wait(5)
            server.shutdown.side_effect = stop.set
            server.http_alive_at = 1e18
            with patch.object(sys, "argv", ["crucible", "checkpoint", "--systemd"]), patch.dict(os.environ, {"NOTIFY_SOCKET": path}), \
                    patch.object(startup, "ModelTokenizer"), patch.object(startup, "ThreadingHTTPServer", return_value=server), \
                    patch("server.batch_backend.BatchedEngineBackend") as backend:
                backend.return_value.fatal = "The engine stopped. Check its stderr log."
                backend.return_value.last_event_at = 1e18
                code = startup.main()
            sock.close()
        self.assertEqual(code, 3)
        backend.return_value.close.assert_called()

    def test_without_the_flag_nothing_is_sent_and_exit_is_0(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "notify")
            sock = listener(path)
            sock.settimeout(.1)
            code, backend, server = self.run_main([], {"NOTIFY_SOCKET": path})
            with self.assertRaises(socket.timeout): sock.recv(64)
            sock.close()
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
