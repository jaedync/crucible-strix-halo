"""Server-side security findings 6, 7, 8, 10, 11, 16, 17: one test per
finding, plus checks that hold. A test named test_f<N>_* states the fixed behavior; a test named test_ok_* checks a rule that held.

Run from the repo root (or set CRUCIBLE_ROOT): python3 -m pytest -q -p no:cacheprovider tests/test_sec_server.py
No network, no GPU, no systemd, no sudo. Synthetic data only.
"""
import contextlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _root():
    here = Path(__file__).resolve().parent
    for cand in (here.parent, Path(os.environ.get("CRUCIBLE_ROOT") or "."), Path.cwd()):
        if (cand / "server/api_key.py").is_file() and (cand / "install.sh").is_file():
            return cand.resolve()
    raise RuntimeError("run from the repo root or set CRUCIBLE_ROOT")


ROOT = _root()
for p in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("CRUCIBLE_NO_UPDATE_CHECK", "1")

import server.__main__ as startup  # noqa: E402
from server import http as http_module  # noqa: E402
from server.api_key import bind_error  # noqa: E402
from server.backend import engine_environment  # noqa: E402
from server.request_id import HeaderNameError, resolve_client_request_id_header  # noqa: E402
from server.sd_notify import Notifier  # noqa: E402
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, token_events  # noqa: E402
from server.tests.test_api_key import KEY, gate  # noqa: E402
from crucible_cli import launch, service, settings  # noqa: E402

CANARY = "crucible-CANARYCANARYCANARYCANARYCANARYCANARY0123"


class Crash(BaseException):
    """A kill: not an Exception, so no handler of the update catches it."""


# ---- scope 1: API key ------------------------------------------------------------------------------------------------

def _serve(test, key=True, **options):
    service_ = http_module.Service(ScriptBackend(token_events(["hi"])), PiecesTokenizer(["hi"]),
                                   api_key=gate() if key else None, **options)
    srv = startup.bind_server("127.0.0.1", 0, http_module.handler_for(service_))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    test.addCleanup(lambda: (srv.shutdown(), srv.server_close()))
    return srv.server_address[1]


def _get(port, path, headers=None, timeout=5):
    try:
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {}),
                                    timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def _raw(port, data):
    """Send raw bytes, return the whole answer (the server closes after each error answer)."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(data if isinstance(data, bytes) else data.encode())
        out = b""
        try:
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                out += chunk
        except OSError:
            pass
    return out


def _status(answer):
    return answer.split(b"\r\n", 1)[0].decode()


class ApiKeyTest(unittest.TestCase):
    def test_ok_bind_rule_matrix(self):
        """Loopback only without a key: 127/8 and ::1. Mapped, ::, 0.0.0.0, link-local and LAN need it; scope ids and names are refused."""
        for text, needs in (("127.0.0.1", False), ("127.9.9.9", False), ("::1", False), ("0:0:0:0:0:0:0:1", False),
                            ("::ffff:127.0.0.1", True), ("::127.0.0.1", True), ("0.0.0.0", True), ("::", True),
                            ("::ffff:0.0.0.0", True), ("fe80::1", True), ("169.254.1.1", True), ("192.0.2.7", True)):
            with self.subTest(text=text):
                self.assertEqual(bind_error(startup.listen_address(text), False, {}) is not None, needs)
        for text in ("fe80::1%lo", "::1%1", "localhost", "127.1", "0", "2130706433", "[::1]", " 127.0.0.1", "127.0.0.1\n"):
            with self.subTest(text=text), self.assertRaises(Exception):
                startup.listen_address(text)

    def test_ok_odd_paths_and_methods_give_no_data_without_the_key(self):
        port = _serve(self)
        for path in ("/v1/models/", "/v1/models?x=1", "//v1/models", "/v1/../v1/models", "/v1/crucible/trace?since=0",
                     "/v1/crucible/version", "/tokenize", "/health/", "/HEALTH"):
            with self.subTest(path=path):
                self.assertIn(_get(port, path), (401, 404, 405, 501))
        self.assertIn(" 401 ", _status(_raw(port, f"GET http://127.0.0.1:{port}/v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")) + " ")
        for method in ("HEAD", "PUT", "OPTIONS", "DELETE"):
            with self.subTest(method=method):
                self.assertIn(" 501 ", _status(_raw(port, f"{method} /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")) + " ")
        self.assertEqual(_get(port, "/health"), 200)
        self.assertEqual(_get(port, "/v1/models", {"Authorization": "Bearer " + KEY}), 200)

    def test_ok_leak_paths_2_to_6_hold(self):
        """A key in an oversized header (431), in a query string, in a path, in a refused method, next to a bad header byte:
        neither the answers nor stderr (access log and crucible_error lines) hold it."""
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            port = _serve(self)
            answers = [_raw(port, f"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {CANARY}{'x' * 70000}\r\n\r\n"),
                       _raw(port, f"GET /v1/models?api_key={CANARY} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"),
                       _raw(port, f"GET /{CANARY} HTTP/1.1\r\nHost: 127.0.0.1\r\nx-api-key: {CANARY}\r\n\r\n"),
                       _raw(port, f"BREW /v1/models?k={CANARY} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"),
                       _raw(port, f"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {CANARY}\r\n".encode()
                            + b"Bad\xff: x\r\n\r\n")]
            time.sleep(0.3)
        self.assertEqual([_status(a).split()[1] for a in answers], ["431", "401", "401", "501", "401"])
        for answer in answers:
            self.assertNotIn(CANARY.encode(), answer)
        self.assertNotIn(CANARY, err.getvalue())

    def test_ok_notify_names_never_reach_the_engine(self):
        env = {"NOTIFY_SOCKET": "/run/x", "WATCHDOG_USEC": "30000000", "WATCHDOG_PID": "1", "PATH": "/usr/bin"}
        Notifier.from_environ(env)
        self.assertEqual(env, {"PATH": "/usr/bin"})
        engine_env, _ = engine_environment({"NOTIFY_SOCKET": "/run/x", "WATCHDOG_USEC": "1", "WATCHDOG_PID": "1", "PATH": "/usr/bin"})
        self.assertEqual(set(engine_env), {"PATH"})

    def test_f6_idle_connections_without_a_key_do_not_lock_out_the_key_holder(self):
        """http.py:39-41,305-317: one shared pool of MAX_CONNECTIONS (256) slots, taken before the key check and held up to
        READ_DEADLINE_SECONDS (60 s). Here the pool is 8: 8 half-sent requests from a client with no key give 503 to a client
        with the key. Measured with the real 256: 256 sockets gave 503 for 24 s, and the attacker reconnects."""
        with mock.patch.object(http_module, "MAX_CONNECTIONS", 8):
            port = _serve(self)
        idle = []
        self.addCleanup(lambda: [s.close() for s in idle])
        for _ in range(8):
            s = socket.create_connection(("127.0.0.1", port))
            s.sendall(b"GET /v1/models HTTP/1.1\r\n")
            idle.append(s)
        time.sleep(0.3)
        self.assertEqual(_get(port, "/v1/models", {"Authorization": "Bearer " + KEY}), 200,
                         "a client with no key took every connection slot")

    def test_f7_the_gateway_id_header_cannot_be_an_auth_header(self):
        """Leak path 1. request_id.py:15-25: --client-request-id-header or CRUCIBLE_CLIENT_REQUEST_ID_HEADER may name x-api-key or
        Authorization. Then metrics logs the header value: each 401 line holds the key that was tried
        (crucible_error {..."reason": "auth", "client_request_id": "crucible-WRONG..."}), and jobs carry the valid key."""
        accepted = []
        for name in ("x-api-key", "X-API-Key", "Authorization", "Proxy-Authorization", "Cookie"):
            for flag, environ in ((name, {}), (None, {"CRUCIBLE_CLIENT_REQUEST_ID_HEADER": name})):
                try:
                    resolve_client_request_id_header(flag, environ)
                    accepted.append((name, "flag" if flag else "env"))
                except HeaderNameError:
                    pass
        self.assertEqual(accepted, [])

    def test_f17_the_service_on_loopback_refuses_a_foreign_host_header(self):
        """DNS rebinding. hosts.py:1-3 accepts any Host by default and launch.py:58-68 passes no --allowed-host, so the unit
        that runs at boot on 127.0.0.1 with no key answers a page whose name now points at 127.0.0.1 (chat 200, usage with
        cached_tokens). Other local servers fixed the same class (CVE-2024-28224)."""
        port = _serve(self, key=False)
        answer = _raw(port, f"GET /v1/models HTTP/1.1\r\nHost: rebind.attacker.example:{port}\r\n\r\n")
        self.assertEqual(_status(answer).split()[1], "200")  # the server itself has no default list (by design, hosts.py)
        env, argv = launch.server_command(settings.Settings(repo=Path("/opt/u/crucible")), "/opt/u/crucible/.venv/bin/python")
        self.assertTrue("--allowed-host" in argv or "CRUCIBLE_ALLOWED_HOSTS" in env,
                        "the service command has no Host list: any web page can use the server by DNS rebinding")

    @unittest.skipIf(Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(), "runs in a container")
    def test_f10_the_container_switch_does_nothing_outside_a_container(self):
        """api_key.py:144-150: CRUCIBLE_CONTAINER_OPEN_BIND=1 in any process environment (for example a user manager
        environment that a login shell imported) opens 0.0.0.0 and :: with no key on a normal host too."""
        opened = [h for h in ("0.0.0.0", "::")
                  if bind_error(startup.listen_address(h), False, {"CRUCIBLE_CONTAINER_OPEN_BIND": "1"}) is None]
        self.assertEqual(opened, [])


# ---- scope 3: the unit -----------------------------------------------------------------------------------------------

def _unit(**kw):
    base = dict(repo="/opt/u/crucible", venv_py="/opt/u/crucible/.venv/bin/python", exec_start="/opt/u/crucible/.crucible/serve.sh",
                bind="127.0.0.1", port=8790, start_timeout=600, version="0.34.0")
    base.update(kw)
    return service.render_unit(**base)


class UnitTest(unittest.TestCase):
    def test_ok_specifiers_quotes_and_escapes_are_refused(self):
        for bad in ("/opt/u/a%h", "/opt/u/$HOME", "/opt/u/${X}", '/opt/u/a"b', "/opt/u/a'b", "/opt/u/a\\b", "/opt/u/a\nb", "/opt/u/a;b"):
            with self.subTest(bad=bad), self.assertRaises(service.UnitPathError):
                _unit(repo=bad, venv_py=bad + "/.venv/bin/python", exec_start=bad + "/.crucible/serve.sh")
        text = _unit(repo="/opt/u/my dir", venv_py="/opt/u/my dir/.venv/bin/python", exec_start="/opt/u/my dir/.crucible/serve.sh")
        self.assertIn('ExecStart="/opt/u/my dir/.crucible/serve.sh" --systemd', text)

    def test_f9_a_value_with_a_trailing_newline_is_refused(self):
        """service.py:15-16,65,78: re.match with $ accepts one trailing newline. A bind from config.json, a version or a timeout
        then breaks its line (systemd-analyze verify: "Missing '=', ignoring line"). No directive can follow: the next text is fixed."""
        accepted = []
        for kw in (dict(bind="127.0.0.1\n"), dict(version="0.34.0\n"), dict(start_timeout="600s\n"), dict(port="8790\n")):
            try:
                _unit(**kw)
                accepted.append(kw)
            except service.UnitPathError:
                pass
        self.assertEqual(accepted, [])


# ---- scope 5: the release engine -------------------------------------------------------------------------------------

FAKE_HIPCC = '#!/bin/bash\nout=""\nwhile [ $# -gt 0 ]; do [ "$1" = "-o" ] && { out="$2"; shift; }; shift; done\nprintf NEW > "$out"\n'


class ReleaseTest(unittest.TestCase):
    def test_f14_the_build_info_names_the_compile_flags(self):
        """Makefile:73-87, release_engine.py:189-205: build-info holds commit, dirty and hash only. HIPFLAGS or CPPFLAGS from the
        environment or the make line change the release engine, and `release` still accepts it as a clean build of the tag."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tree = Path(tmp.name) / "tree"
        for rel in ("Makefile", "VERSION", "tools/hipcc_checked.sh", "tools/release_engine.py", "tools/devsplit.py",
                    "tools/devsplit_ir.py", "tools/devsplit_groups.tsv", "tools/devsplit_facts.tsv"):
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, tree / rel)
        (tree / "engine").mkdir()
        (tree / "engine/engine.hip").write_text("// engine\n")
        fake = Path(tmp.name) / "hipcc"
        fake.write_text(FAKE_HIPCC)
        fake.chmod(0o755)
        git = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=test-user"]
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "t"]):
            subprocess.run([*git, *args], check=True, capture_output=True)
        env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS")}
        r = subprocess.run(["make", "build/engine/crucible", "SPLIT=0", f"HIPCC=bash {fake}", "PYTHON=" + sys.executable, "STATE_EPOCH=0",
                            "HIPFLAGS=-O0 -DLOCAL_EXPERIMENT"], cwd=tree, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        info = (tree / "build/engine/crucible.build-info").read_text()
        self.assertIn("LOCAL_EXPERIMENT", info, "the release check cannot see that the engine was built with other flags")


# ---- scope 6: the fault driver ---------------------------------------------------------------------------------------

class FaultDriverTest(unittest.TestCase):
    def test_f15_the_n9_server_has_a_life_limit_of_its_own(self):
        """service_faulttest.py:1025-1033: N9 starts the server with start_new_session=True, outside the unit and the guard timer.
        gpu-run's session kill skips it, and a SIGKILL of the driver leaves it on the GPU and on port 8791."""
        import service_faulttest as ft
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        t = Path(tmp.name)
        cfg = ft.build_config(ft.parser().parse_args(["--fake", "--work", str(t / "work"), "--state-dir", str(t / "states"),
                                                      "--repo", str(ROOT)]))
        cfg.work.mkdir(parents=True)
        seen = {}

        def popen(argv, **kw):
            seen.update(argv=argv, **kw)
            return SimpleNamespace(poll=lambda: 1, pid=0)
        with mock.patch.object(ft.subprocess, "Popen", popen):
            ft.RealMachine(cfg).direct_start()
        bounded = Path(seen["argv"][0]).name in ("timeout", "systemd-run")
        self.assertTrue(bounded or not seen.get("start_new_session"), seen["argv"][:4])


# ---- findings 6, 7, 10, 11 and 8: tests of the fixes ----------------------------------------

def _hold(test, port, data=b"GET /v1/models HTTP/1.1\r\n", count=1, source=None):
    """Open COUNT connections that send DATA and stay open."""
    sockets = []
    test.addCleanup(lambda: [s.close() for s in sockets])
    for _ in range(count):
        s = socket.create_connection(("127.0.0.1", port), timeout=5, source_address=source)
        s.sendall(data)
        sockets.append(s)
    return sockets


def _closed(sock, wait):
    """True when the server closed this connection within WAIT seconds."""
    sock.settimeout(wait)
    try:
        return sock.recv(4096) == b""
    except socket.timeout:
        return False
    except OSError:
        return True


def _answer(sock):
    """The status line of the next answer on SOCK (the whole answer is read)."""
    sock.settimeout(5)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(65536)
        if not chunk:
            return ""
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    length = int(next((line.split(b":")[1] for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")), b"0"))
    while len(body) < length:
        body += sock.recv(65536)
    return head.split(b"\r\n")[0].decode()


def _parallel(count, call):
    results = [None] * count

    def run(i):
        results[i] = call(i)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return results


def _patched(test, **values):
    """Patch http constants for the whole test (the handler reads them when a connection arrives)."""
    for name, value in values.items():
        patcher = mock.patch.object(http_module, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)


class ConnectionLimitTest(unittest.TestCase):
    """Finding 6: a deadline for the request head, and a limit on connections that have not passed the key check."""

    def test_a_loopback_server_with_no_key_keeps_its_behavior_for_many_connections(self):
        """The daily engine pattern: one gateway opens many connections from 127.0.0.1. No key: no per-address limit."""
        port = _serve(self, key=False)
        self.assertEqual(_parallel(64, lambda i: _get(port, "/v1/models")), [200] * 64)
        held = _hold(self, port, count=40)  # 40 slow heads from one address stay open (below the pool, under the head deadline)
        time.sleep(0.3)
        self.assertEqual(_get(port, "/v1/models"), 200)
        self.assertFalse(any(_closed(s, 0.01) for s in held))

    def test_a_server_with_a_key_serves_a_burst_of_parallel_keyed_requests(self):
        """The grace time: a client sends its first request at once, so 64 connections that arrive together all pass."""
        port = _serve(self)
        head = {"Authorization": "Bearer " + KEY}
        self.assertEqual(_parallel(64, lambda i: _get(port, "/v1/models", head)), [200] * 64)

    def test_the_request_head_has_its_own_short_deadline(self):
        _patched(self, HEAD_DEADLINE_SECONDS=0.4, WATCH_SECONDS=0.05)
        port = _serve(self, key=False)
        (slow,) = _hold(self, port)
        self.assertTrue(_closed(slow, 3), "a half-sent head was not closed at the head deadline")

    def test_the_defaults(self):
        self.assertEqual((http_module.HEAD_DEADLINE_SECONDS, http_module.UNKEYED_PER_ADDRESS), (10, 16))
        self.assertGreater(http_module.READ_DEADLINE_SECONDS, http_module.HEAD_DEADLINE_SECONDS)

    def test_a_body_that_arrives_is_not_cut_by_the_head_deadline(self):
        _patched(self, HEAD_DEADLINE_SECONDS=0.3, WATCH_SECONDS=0.05)
        port = _serve(self, key=False)
        body = json.dumps({"model": "crucible", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode()
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.addCleanup(s.close)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Content-Length: %d\r\n\r\n" % len(body) + body[:10])
        time.sleep(1.0)  # longer than the head deadline: the head is in, the body is slow
        s.sendall(body[10:])
        self.assertIn(" 200 ", _answer(s) + " ")

    def test_a_flood_from_one_address_cannot_hold_more_than_the_limit(self):
        _patched(self, UNKEYED_PER_ADDRESS=4, UNKEYED_GRACE_SECONDS=0.0)
        port = _serve(self)
        held = _hold(self, port, count=12)
        time.sleep(0.5)
        closed = sum(_closed(s, 0.05) for s in held)
        self.assertEqual(closed, 8, "the server kept more than 4 half-sent connections of one address")
        self.assertEqual(_get(port, "/v1/models", {"Authorization": "Bearer " + KEY}), 200)

    def test_a_connection_that_passed_the_key_check_is_not_counted_or_dropped(self):
        """Six requests with the key send their head and half of the body. A flood of heads follows. The six finish."""
        _patched(self, UNKEYED_PER_ADDRESS=2, UNKEYED_GRACE_SECONDS=0.5)  # the grace time lets the six pass the key check first
        port = _serve(self)
        body = json.dumps({"model": "crucible", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode()
        head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {KEY}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode()
        keyed = _hold(self, port, head + body[:10], count=6)
        time.sleep(0.2)
        flood = _hold(self, port, count=10)  # heads with no key
        time.sleep(0.8)  # the flood is older than the grace time now
        _hold(self, port)  # one more arrival: the oldest connections above the limit go
        time.sleep(0.2)
        self.assertEqual(sum(_closed(s, 0.02) for s in flood), 9)
        for s in keyed:
            s.sendall(body[10:])
            self.assertIn(" 200 ", _answer(s) + " ")

    def test_the_registry_of_connections_without_a_key(self):
        _patched(self, UNKEYED_PER_ADDRESS=2, UNKEYED_GRACE_SECONDS=1.0)
        u = http_module.Unkeyed()
        for name, at in (("a1", 0), ("a2", 0.1), ("a3", 0.2)):
            self.assertEqual(u.add(name, "A", now=at), [], "young connections are not counted")
        self.assertEqual(u.add("a4", "A", now=5), ["a1", "a2"], "the oldest connections above the limit go")
        self.assertEqual(u.add("a5", "A", now=5.1), [])
        self.assertEqual(u.add("b1", "B", now=5.2), [], "another address is not touched")
        u.keyed("a4")
        self.assertEqual(u.for_room(), "a3", "a full pool closes the oldest connection of the busiest address")
        self.assertEqual(u.for_room(), "a5")
        self.assertEqual(u.for_room(), "b1")
        self.assertIsNone(u.for_room())


class UnitHostTest(unittest.TestCase):
    """Finding 7: the service command passes a Host list. A user can add names for a reverse proxy."""

    def command(self, **kw):
        return launch.server_command(settings.Settings(repo=Path("/opt/u/crucible"), **kw), "/opt/u/crucible/.venv/bin/python")[1]

    def listed(self, argv):
        i = argv.index("--allowed-host")
        return [h for h in argv[i + 1].split(",")]

    def test_the_default_list_has_the_loopback_names(self):
        self.assertEqual(self.listed(self.command()), ["localhost", "127.0.0.1", "[::1]"])

    def test_the_bind_address_is_in_the_list(self):
        self.assertEqual(self.listed(self.command(bind="127.0.0.2")), ["localhost", "127.0.0.1", "[::1]", "127.0.0.2"])
        self.assertEqual(self.listed(self.command(bind="::1")), ["localhost", "127.0.0.1", "[::1]"])

    def test_user_names_are_added(self):
        argv = self.command(allowed_hosts=("llm.example.org", "proxy.lan:8443"))
        self.assertEqual(self.listed(argv), ["localhost", "127.0.0.1", "[::1]", "llm.example.org", "proxy.lan:8443"])

    def test_the_server_with_that_list_refuses_a_rebinding_host_and_accepts_the_local_names(self):
        from server.hosts import resolve_allowed_hosts
        argv = self.command(allowed_hosts=("llm.example.org",))
        allowed = resolve_allowed_hosts([argv[argv.index("--allowed-host") + 1]], {})
        port = _serve(self, key=False, allowed_hosts=allowed)
        for host, code in ((f"rebind.attacker.example:{port}", "403"), (f"localhost:{port}", "200"), (f"127.0.0.1:{port}", "200"),
                           (f"[::1]:{port}", "200"), ("llm.example.org", "200"), ("localhost.attacker.example", "403")):
            with self.subTest(host=host):
                answer = _raw(port, f"GET /v1/models HTTP/1.1\r\nHost: {host}\r\n\r\n")
                self.assertEqual(_status(answer).split()[1], code)

    def test_a_bind_that_is_not_loopback_keeps_the_key_as_its_protection_and_adds_no_list_by_itself(self):
        self.assertNotIn("--allowed-host", self.command(bind="0.0.0.0", api_key_file=Path("/opt/u/.config/crucible/api-key")))
        argv = self.command(bind="192.0.2.7", api_key_file=Path("/k"), allowed_hosts=("llm.example.org",))
        self.assertEqual(self.listed(argv), ["localhost", "127.0.0.1", "[::1]", "192.0.2.7", "llm.example.org"])

    def test_the_names_are_saved_in_the_config_and_the_serve_script_has_them(self):
        s = settings.Settings(repo=Path("/opt/u/crucible"), allowed_hosts=("llm.example.org",))
        self.assertEqual(s.to_json()["allowed_hosts"], ["llm.example.org"])
        self.assertIn("--allowed-host", launch.serve_script(s, "/opt/u/crucible/.venv/bin/python"))
        self.assertIn("llm.example.org", launch.serve_script(s, "/opt/u/crucible/.venv/bin/python"))

    def test_the_flag_takes_names_and_refuses_a_bad_name(self):
        from crucible_cli import cli
        args = cli.parser().parse_args(["setup", "--allowed-host", "llm.example.org", "--allowed-host", "[::1]:8443"])
        self.assertEqual(args.allowed_host, ["llm.example.org", "[::1]:8443"])
        for bad in ("a b", "http://x", "x/y", "*", "a,b", "", "x:99999", "x\n"):
            with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parser().parse_args(["setup", "--allowed-host", bad])

    def test_the_proxy_guide_says_how(self):
        text = " ".join((ROOT / "docs/REMOTE_ACCESS.md").read_text().split())
        self.assertIn("--allowed-host", text)
        self.assertRegex(text, r"(?i)DNS rebinding")


class RequestIdTest(unittest.TestCase):
    def test_other_headers_stay_allowed(self):
        for name in ("X-Request-Id", "X-Client-Request-Id", "Cf-Ray", "Traceparent"):
            self.assertEqual(resolve_client_request_id_header(name, {}), name)

    def test_the_refusal_names_the_header_and_the_reason(self):
        with self.assertRaises(HeaderNameError) as caught:
            resolve_client_request_id_header("x-api-key", {})
        text = str(caught.exception)
        self.assertIn("x-api-key", text)
        self.assertRegex(text, r"(?i)key|credential")

    def test_the_server_start_exits_2_with_that_message(self):
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["crucible", "ckpt", "--client-request-id-header", "Authorization"]), \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            startup.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("Authorization", err.getvalue())


class FullmatchTest(unittest.TestCase):
    def test_no_regex_in_the_cli_modules_of_this_worker_uses_match_for_a_whole_string_rule(self):
        from crucible_cli import release
        for text in ("v1.2.3\n", "v1.2.3 "):
            self.assertFalse(release.TAG.fullmatch(text))
        for module in ("service.py", "release.py", "cli.py"):
            source = (ROOT / "tools/crucible_cli" / module).read_text()
            self.assertNotRegex(source, r"\.match\(", module)

    def test_release_and_the_tag_flag_refuse_a_trailing_newline(self):
        from crucible_cli import cli
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parser().parse_args(["update", "--tag", "v0.34.0\n"])

    def test_a_trailing_newline_in_each_unit_value(self):
        for kw in (dict(repo="/opt/u/a\n", venv_py="/opt/u/a/p", exec_start="/opt/u/a/s"), dict(bind="::1\n")):
            with self.subTest(kw=kw), self.assertRaises(service.UnitPathError):
                _unit(**kw)


class ContainerMarkerTest(unittest.TestCase):
    """Finding 11: the image switch needs a container marker file. The switch keeps working in the image (a container needs the wildcard address)."""

    def test_the_switch_with_a_marker_file_opens_the_wildcard_addresses(self):
        from server import api_key
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / ".dockerenv"
            marker.write_text("")
            on = {"CRUCIBLE_CONTAINER_OPEN_BIND": "1"}
            with mock.patch.object(api_key, "CONTAINER_MARKERS", (str(Path(folder) / "none"), str(marker))):
                for host in ("0.0.0.0", "::"):
                    self.assertIsNone(bind_error(startup.listen_address(host), False, on))
                self.assertIsNotNone(bind_error(startup.listen_address("192.0.2.7"), False, on))
                self.assertIsNotNone(bind_error(startup.listen_address("0.0.0.0"), False, {}))
            marker.unlink()
            with mock.patch.object(api_key, "CONTAINER_MARKERS", (str(marker),)):
                self.assertIsNotNone(bind_error(startup.listen_address("0.0.0.0"), False, on))

    def test_the_marker_names_are_the_docker_and_podman_files(self):
        from server import api_key
        self.assertEqual(api_key.CONTAINER_MARKERS, ("/.dockerenv", "/run/.containerenv"))


if __name__ == "__main__":
    unittest.main()
