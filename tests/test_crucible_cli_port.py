"""The start check must treat a port like the server does (the real fault test).

After a kill -9 the server side of every open connection sits in TIME_WAIT on the server port for 60 s. The server binds with
SO_REUSEADDR and starts. The doctor bound without it, called the port "in use", and ExecStartPre waited about 60 s (N3 and N5
healed in 106 s and 179 s) or, in N10, named the port first and hid the pack folder. Real sockets, loopback only."""
import json
import socket
import tempfile
import unittest
from pathlib import Path

from crucible_cli_fakes import make_repo, wire  # noqa: E402

from crucible_cli import doctor  # noqa: E402
from crucible_cli import host as hostmod  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402


def time_wait_port():
    """A free port that has a connection in TIME_WAIT on the server side, as a killed server leaves it."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    cli = socket.create_connection(("127.0.0.1", port))
    conn, _ = srv.accept()
    conn.close()   # the server side closes first: it keeps the TIME_WAIT entry
    cli.close()
    srv.close()
    return port


class PortFree(unittest.TestCase):
    def test_a_port_with_closed_connections_in_time_wait_is_free(self):
        port = time_wait_port()
        with socket.socket() as probe:   # the test must see the TIME_WAIT entry, or it proves nothing
            with self.assertRaises(OSError):
                probe.bind(("127.0.0.1", port))
        self.assertTrue(hostmod._port_free("127.0.0.1", port))

    def test_a_port_with_a_listening_server_is_not_free(self):
        with socket.socket() as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", 0))
            srv.listen()
            self.assertFalse(hostmod._port_free("127.0.0.1", srv.getsockname()[1]))

    def test_a_port_that_nothing_uses_is_free(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.assertTrue(hostmod._port_free("127.0.0.1", port))


class StartCheckOrder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.repo = make_repo(self.t)

    def test_a_bad_setting_is_named_before_a_busy_port(self):
        """N10: the status shows the first failed check. A port that may free itself must not hide the pack folder."""
        host = wire(self.t)
        s = settings_mod.build(self.repo, host)
        port = doctor.CheckResult("port", "Port 8791 is free", "fail", "127.0.0.1:8791 is in use", "Stop it.", ("start",))
        pack = doctor.CheckResult("pack", "Pack 4.05", "fail", "the pack folder /x is missing", "Run setup.", ("start",))
        orig = doctor.start_failures
        doctor.start_failures = lambda h, st: [port, pack]
        self.addCleanup(setattr, doctor, "start_failures", orig)
        check = self.t / "check.json"
        self.assertEqual(doctor.for_start(host, s, quiet=True, check_file=check), doctor.EX_CONFIG)
        ids = [f["id"] for f in json.loads(check.read_text())["failed"]]
        self.assertEqual(ids, ["pack", "port"])
        self.assertEqual(host.lines[0].split(":")[0], "pack")


if __name__ == "__main__":
    unittest.main()
