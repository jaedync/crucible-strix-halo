"""The stop path probes /health on the server port (r373 security review H1).

/health needs no key. The probe must never send the API key: on an IPv6 bind, or on a saved address that this machine
no longer owns, the listener can be another machine. The port check must also see IPv6 binds and foreign addresses
correctly, so that the probe goes only to a listener on this machine. Real sockets on loopback; no other address is
contacted (192.0.2.0/24 is TEST-NET-1 and the bind attempt fails before any packet)."""
import http.server
import os
import socket
import threading

import pytest

from crucible_cli_fakes import make_repo, wire
from crucible_cli import host as hostmod
from crucible_cli import http as httpmod
from crucible_cli import processes, settings, steps


def _has_ipv6_loopback():
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


needs_ipv6 = pytest.mark.skipif(not _has_ipv6_loopback(), reason="no IPv6 loopback on this host")


class _Recorder(http.server.BaseHTTPRequestHandler):
    seen = None

    def do_GET(self):
        type(self).seen.append(dict(self.headers))
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _listener(family=socket.AF_INET, addr="127.0.0.1"):
    class Server(http.server.HTTPServer):
        address_family = family
    handler = type("Rec", (_Recorder,), {"seen": []})
    srv = Server((addr, 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, handler


def _keyed_settings(tmp_path, host, bind, port):
    repo = make_repo(tmp_path)
    key = tmp_path / "api-key"
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, b"k" * 40 + b"\n")
    os.close(fd)
    return settings.build(repo, host, bind=bind, port=port, api_key_file=key)


def test_the_stop_probe_sends_no_authorization_header(tmp_path):
    srv, handler = _listener()
    try:
        port = srv.server_address[1]
        host = wire(tmp_path, http=httpmod.Http())
        host.port_free = hostmod._port_free
        host.port_in_use = hostmod._port_in_use
        s = _keyed_settings(tmp_path, host, "127.0.0.1", port)
        assert s.api_key_file is not None
        assert steps.stop(host, s) == 1   # a listener answers: it is named, not called "nothing to stop"
    finally:
        srv.shutdown()
        srv.server_close()
    assert handler.seen, "the probe did not reach the listener"
    assert all("Authorization" not in h and "authorization" not in h for h in handler.seen)


def test_the_probe_passes_no_key_file_to_the_client(tmp_path):
    host = wire(tmp_path, busy_ports=(8790,))
    s = _keyed_settings(tmp_path, host, "127.0.0.1", 8790)
    host.port_in_use = lambda a, p: True
    assert processes.remaining_server(host, s) is not None
    assert host.http.gets and host.http.key_files == [None] * len(host.http.gets)


def test_a_saved_address_that_this_machine_does_not_own_is_not_probed(tmp_path):
    host = wire(tmp_path)
    host.port_free = hostmod._port_free
    host.port_in_use = hostmod._port_in_use
    s = _keyed_settings(tmp_path, host, "192.0.2.55", 8790)
    assert processes.remaining_server(host, s) is None
    assert host.http.gets == []


@pytest.mark.parametrize("addr", ["192.0.2.55", "10.99.99.99", "fd00::1", "2001:db8::7"])
def test_a_foreign_address_is_not_in_use(addr):
    assert hostmod._port_in_use(addr, 8790) is False


def test_an_ipv4_listener_is_in_use_and_not_free():
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        port = srv.getsockname()[1]
        assert hostmod._port_in_use("127.0.0.1", port) is True
        assert hostmod._port_free("127.0.0.1", port) is False


@needs_ipv6
def test_an_ipv6_port_that_nothing_uses_is_free():
    with socket.socket(socket.AF_INET6) as s:
        s.bind(("::1", 0))
        port = s.getsockname()[1]
    assert hostmod._port_free("::1", port) is True
    assert hostmod._port_in_use("::1", port) is False


@needs_ipv6
def test_an_ipv6_listener_is_in_use_and_not_free():
    with socket.socket(socket.AF_INET6) as srv:
        srv.bind(("::1", 0))
        srv.listen()
        port = srv.getsockname()[1]
        assert hostmod._port_in_use("::1", port) is True
        assert hostmod._port_free("::1", port) is False


@needs_ipv6
def test_the_ipv6_stop_probe_sends_no_authorization_header(tmp_path):
    srv, handler = _listener(socket.AF_INET6, "::1")
    try:
        port = srv.server_address[1]
        host = wire(tmp_path, http=httpmod.Http())
        host.port_free = hostmod._port_free
        host.port_in_use = hostmod._port_in_use
        s = _keyed_settings(tmp_path, host, "::1", port)
        assert steps.stop(host, s) == 1
    finally:
        srv.shutdown()
        srv.server_close()
    assert handler.seen
    assert all("Authorization" not in h and "authorization" not in h for h in handler.seen)
