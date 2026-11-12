"""--host ADDRESS: the bind address (default 127.0.0.1), an IPv4 or IPv6 literal, with a warning off loopback."""
import ipaddress
import os
import socket
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler
from unittest.mock import patch

import server.__main__ as startup
from server.api_key import is_loopback
from server.tests.test_server_defaults import parse


def listening(args):
    """The start lines, and the mock of bind_server: main() calls it with (address, port, handler).

    An address that is not loopback needs a key file (design s3), so this makes one for such a host.
    """
    args = list(args)
    host = args[args.index("--host") + 1] if "--host" in args else "127.0.0.1"
    with tempfile.TemporaryDirectory() as folder, patch.object(startup, "bind_server") as bind:
        if "--api-key-file" not in args and not is_loopback(host):
            path = os.path.join(folder, "api-key")
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600), "w") as handle:
                handle.write("k" * 40 + "\n")
            args += ["--api-key-file", path]
        bind.return_value.serve_forever.side_effect = KeyboardInterrupt
        options, err = parse(args)
    return options, err, bind


def ipv6_available():
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
        return True
    except OSError:
        return False


class ParseTest(unittest.TestCase):
    def test_default_binds_127_0_0_1_with_no_warning(self):
        _, err, bind = listening([])
        self.assertEqual(bind.call_args.args[:2], (ipaddress.ip_address("127.0.0.1"), 8790))
        self.assertNotIn("not loopback", err)
        self.assertNotIn("API key", err)
        self.assertIn("ready at http://127.0.0.1:8790/v1", err)

    def test_ipv4_literals(self):
        for host in ("127.0.0.1", "127.0.0.2", "0.0.0.0", "192.0.2.7"):
            with self.subTest(host=host):
                _, err, bind = listening(["--host", host, "--port", "9001"])
                self.assertEqual(bind.call_args.args[:2], (ipaddress.ip_address(host), 9001))
                self.assertIn(f"ready at http://{host}:9001/v1", err)

    def test_ipv6_literals(self):
        for host in ("::1", "::", "2001:db8::7"):
            with self.subTest(host=host):
                _, err, bind = listening(["--host", host, "--port", "9001"])
                self.assertEqual(bind.call_args.args[:2], (ipaddress.ip_address(host), 9001))
                self.assertIn(f"ready at http://[{host}]:9001/v1", err)

    def test_bad_values_exit_2_before_loading(self):
        for host in ("localhost", "example.com", "", "256.0.0.1", "1.2.3", "::g", "[::1]", "127.0.0.1:80",
                     "fe80::1%eth0", " 127.0.0.1"):
            with self.subTest(host=host):
                code, _ = parse(["--host", host])
                self.assertEqual(code, 2)


class WarningTest(unittest.TestCase):
    def warnings(self, host):
        _, err, _ = listening(["--host", host])
        return [line for line in err.splitlines() if "which is not loopback" in line]

    def test_loopback_has_no_warning(self):
        for host in ("127.0.0.1", "127.9.9.9", "::1"):
            with self.subTest(host=host):
                self.assertEqual(self.warnings(host), [])

    def test_other_addresses_have_one_warning(self):
        for host in ("0.0.0.0", "::", "192.0.2.7", "2001:db8::7"):
            with self.subTest(host=host):
                (line,) = self.warnings(host)
                self.assertIn(host, line)
                self.assertIn("--allowed-host", line)
                self.assertIn("firewall", line)
                self.assertIn("API key", line)


class BindTest(unittest.TestCase):
    def test_the_class_follows_the_address_family(self):
        with patch.object(startup, "ThreadingHTTPServer") as v4, patch.object(startup, "Ipv6Server") as v6:
            startup.bind_server("192.0.2.7", 0, BaseHTTPRequestHandler)
            v6.assert_not_called()
            startup.bind_server("2001:db8::7", 0, BaseHTTPRequestHandler)
            v4.assert_called_once()
            v6.assert_called_once()
        self.assertEqual(startup.Ipv6Server.address_family, socket.AF_INET6)

    def serve(self, host):
        server = startup.bind_server(host, 0, BaseHTTPRequestHandler)
        self.addCleanup(server.server_close)
        return server

    def test_binds_127_0_0_1_on_port_0(self):
        server = self.serve("127.0.0.1")
        self.assertEqual(server.server_address[0], "127.0.0.1")
        self.assertEqual(server.socket.family, socket.AF_INET)
        self.assertGreater(server.server_address[1], 0)

    def test_the_bind_does_not_resolve_a_name(self):
        # HTTPServer.server_bind calls socket.getfqdn, which asks the resolver. The server needs no name.
        with patch("socket.getfqdn", side_effect=AssertionError("getfqdn called")):
            server = self.serve("127.0.0.1")
        self.assertEqual((server.server_name, server.server_port), ("127.0.0.1", server.server_address[1]))

    @unittest.skipUnless(ipv6_available(), "this host has no IPv6 loopback (::1 does not bind)")
    def test_ipv6_server_sets_v6only_to_0_before_it_binds(self):
        # The kernel reports 1 for a socket bound to ::1 whatever we set, and only "::" depends on the option. A test
        # must not bind "::" (every interface), so a spy records the option that server_bind sets.
        calls = []

        class Spy:
            def __init__(self, inner):
                self.inner = inner

            def setsockopt(self, *args):
                calls.append(("setsockopt", args))
                return self.inner.setsockopt(*args)

            def bind(self, *args):
                calls.append(("bind", args))
                return self.inner.bind(*args)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        server = startup.Ipv6Server(("::1", 0), BaseHTTPRequestHandler, bind_and_activate=False)
        self.addCleanup(server.server_close)
        server.socket = Spy(server.socket)
        server.server_bind()
        names = [name for name, _ in calls]
        self.assertEqual(calls[0], ("setsockopt", (socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)))
        self.assertLess(names.index("setsockopt"), names.index("bind"))

    @unittest.skipUnless(ipv6_available(), "this host has no IPv6 loopback (::1 does not bind)")
    def test_binds_ipv6_loopback_on_port_0(self):
        server = self.serve("::1")
        self.assertEqual(server.server_address[0], "::1")
        self.assertEqual(server.socket.family, socket.AF_INET6)
        self.assertGreater(server.server_address[1], 0)


if __name__ == "__main__":
    unittest.main()
