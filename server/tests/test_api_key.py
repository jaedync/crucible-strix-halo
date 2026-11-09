"""The API key (design s3). Each test that covers a point of the security checklist (s7) names its number: item_N.

Items 10 (NOTIFY_SOCKET) and 11 (unit file) belong to the notify and service parts; their tests here run as soon as
those parts are in the tree and skip before.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

import server.__main__ as startup
from server import api_key, http as http_module
from server.api_key import (AUTH_TEXT, DUPLICATE_TEXT, AuthError, KeyFileError, KeyGate, authorize, bind_error,
                            digest_of, is_loopback, load_digest)
from server.hosts import parse_allowed_hosts
from server.tests.serve_fixtures import ScriptBackend, PiecesTokenizer, chat, invoke, token_events

KEY = "crucible-" + "Ab1_-" * 8 + "xyz"           # 52 characters of the allowed set
OTHER = "crucible-" + "Zz9_-" * 8 + "xyz"
CANARY = "crucible-CANARYCANARYCANARYCANARYCANARYCANARY0123"
SERVER_DIR = Path(api_key.__file__).parent


def gate(key=KEY):
    return KeyGate(digest_of(key))


def message(*pairs):
    headers = Message()
    for name, value in pairs:
        headers[name] = value
    return headers


def request(headers=(), *, path="/v1/chat/completions", method="POST", key=KEY, strict=None, raw=None):
    """One request through the real handler. `headers` is added to Host, Content-Length and Content-Type."""
    payload = raw if raw is not None else json.dumps(chat()).encode()
    base = [("Host", "127.0.0.1:8790"), ("Content-Length", str(len(payload))), ("Content-Type", "application/json")]
    backend = ScriptBackend(token_events(["hi"]))
    options = dict(api_key=None if key is None else gate(key))
    if strict is not None:
        options["allowed_hosts"] = strict
    if method == "GET":
        response = invoke(backend, PiecesTokenizer(["hi"]), path=path, method="GET", raw=b"",
                          headers=base[:1] + list(headers), service_options=options)
    else:
        response = invoke(backend, PiecesTokenizer(["hi"]), path=path, raw=payload, headers=base + list(headers),
                          service_options=options)
    response.backend = backend
    return response


def status(response):
    return response.statuses[0]


def bearer(key=KEY):
    return ("Authorization", f"Bearer {key}")


def key_file(folder, text=KEY + "\n", mode=0o600):
    path = Path(folder) / "api-key"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.chmod(path, mode)
    return path


class KeyFileTest(unittest.TestCase):
    """Item 7: O_NOFOLLOW, regular file, owner, mode 0600, size limit, character set. A refusal shows no content."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "api-key"

    def write(self, text=KEY + "\n", mode=0o600):
        return key_file(self.dir.name, text, mode)

    def refused(self, **kw):
        with self.assertRaises(KeyFileError) as caught:
            load_digest(self.path, **kw)
        text = str(caught.exception)
        self.assertIn(str(self.path), text)
        self.assertNotIn(KEY, text)
        return text

    def test_item_7_a_good_file_gives_the_digest_and_not_the_text(self):
        self.write()
        self.assertEqual(load_digest(self.path), hashlib.sha256(KEY.encode()).digest())

    def test_item_7_one_trailing_newline_or_none_is_fine(self):
        for text in (KEY, KEY + "\n", KEY + "\r\n", KEY + "  \n"):
            with self.subTest(text=repr(text[-3:])):
                self.write(text)
                self.assertEqual(load_digest(self.path), digest_of(KEY))

    def test_item_7_a_missing_file_is_refused_with_the_path(self):
        self.refused()

    def test_item_7_a_symlink_is_refused(self):
        target = Path(self.dir.name) / "real"
        target.write_text(KEY)
        target.chmod(0o600)
        self.path.symlink_to(target)
        self.assertIn("symbolic link", self.refused())

    def test_item_7_a_folder_and_a_fifo_are_refused(self):
        self.path.mkdir()
        self.assertIn("regular file", self.refused())
        self.path.rmdir()
        os.mkfifo(self.path, 0o600)  # a read of a FIFO would block for ever; the open must not
        self.assertIn("regular file", self.refused())

    def test_item_7_a_mode_that_others_can_use_is_refused(self):
        for mode in (0o640, 0o604, 0o644, 0o660, 0o666, 0o601, 0o610):
            with self.subTest(mode=oct(mode)):
                self.write(mode=mode)
                self.assertIn("chmod 600", self.refused())

    def test_item_7_a_stricter_mode_is_fine(self):
        self.write(mode=0o400)
        self.assertEqual(load_digest(self.path), digest_of(KEY))

    def test_item_7_a_file_of_another_owner_is_refused(self):
        self.write()
        self.assertIn("owner", self.refused(uid=os.geteuid() + 1))

    def test_item_7_a_large_file_is_refused(self):
        self.write("a" * 5000)
        self.assertIn("1024", self.refused())

    def test_item_7_the_character_set_and_length(self):
        bad = ["", "\n", "short-key", "a" * 31, "a" * 257, KEY + " " + KEY, "crucible-" + "é" * 40, KEY + "\nsecond",
               "a b" * 20, KEY[:-1] + "$", KEY[:-1] + "'", KEY[:-1] + "\x00", "\ufeff" + KEY]
        for text in bad:
            with self.subTest(text=repr(text[:12])):
                self.write(text)
                shown = self.refused()
                self.assertNotIn(text.strip()[:20] or "\0", shown)

    def test_item_7_the_limits_of_the_length_are_in(self):
        for length in (32, 256):
            self.write("k" * length)
            self.assertEqual(load_digest(self.path), digest_of("k" * length))

    def test_item_7_a_refusal_never_holds_the_content(self):
        self.write(CANARY + " with a space", mode=0o644)
        self.assertNotIn("CANARY", self.refused())
        self.write(CANARY + " with a space")
        self.assertNotIn("CANARY", self.refused())


class AuthorizeTest(unittest.TestCase):
    """Items 1, 2 and 5: the compare and the header rules."""
    digest = digest_of(KEY)

    def check(self, *pairs):
        try:
            authorize(message(*pairs), self.digest)
        except AuthError as error:
            return error
        return None

    def test_bearer_and_x_api_key_both_work(self):
        self.assertIsNone(self.check(bearer()))
        self.assertIsNone(self.check(("x-api-key", KEY)))
        self.assertIsNone(self.check(("X-API-KEY", KEY)))

    def test_the_scheme_is_case_insensitive(self):
        for scheme in ("bearer", "BEARER", "BeArEr"):
            self.assertIsNone(self.check(("Authorization", f"{scheme} {KEY}")))

    def test_item_5_both_headers_present_must_both_match(self):
        self.assertIsNone(self.check(bearer(), ("x-api-key", KEY)))
        for pairs in ((bearer(), ("x-api-key", OTHER)), (bearer(OTHER), ("x-api-key", KEY)),
                      (bearer(OTHER), ("x-api-key", OTHER))):
            with self.subTest(pairs=pairs):
                self.assertEqual(self.check(*pairs).status, 401)

    def test_item_5_a_duplicate_header_is_refused_with_400(self):
        for pairs in ((bearer(), bearer()), (("x-api-key", KEY), ("x-api-key", KEY)),
                      (bearer(OTHER), bearer(KEY)), (("x-api-key", OTHER), ("X-Api-Key", KEY))):
            with self.subTest(pairs=pairs):
                error = self.check(*pairs)
                self.assertEqual((error.status, str(error)), (400, DUPLICATE_TEXT))

    def test_item_2_missing_and_wrong_give_the_same_error(self):
        wrong = [(), (bearer(OTHER),), (("x-api-key", OTHER),), (("Authorization", f"Basic {KEY}"),),
                 (("Authorization", KEY),), (("Authorization", "Bearer"),), (("Authorization", "Bearer "),),
                 (("Authorization", f"Bearer  {KEY[:-1]}"),), (("x-api-key", ""),), (("x-api-key", KEY + "x"),),
                 (("Authorization", f"Bearer {KEY} extra"),), (("Authorization", f"Token {KEY}"),),
                 (("x-api-key", KEY[:-1]),), (("x-api-key", " ".join([KEY, KEY])),)]
        seen = set()
        for pairs in wrong:
            with self.subTest(pairs=pairs):
                error = self.check(*pairs)
                self.assertEqual((error.status, str(error)), (401, AUTH_TEXT))
                seen.add((error.status, str(error), error.param, error.code))
        self.assertEqual(len(seen), 1)

    def test_a_wrong_scheme_with_the_right_key_text_does_not_pass(self):
        self.assertEqual(self.check(("Authorization", f"Basic {KEY}")).status, 401)
        self.assertEqual(self.check(bearer(), ("Authorization", f"Basic {KEY}")).status, 400)

    def test_a_long_or_odd_value_is_refused_not_raised(self):
        for value in ("x" * 60000, "\xff" * 100, "\ud800", "é" * 40, "\x00"):
            with self.subTest(length=len(value)):
                self.assertEqual(self.check(("x-api-key", value)).status, 401)
                self.assertEqual(self.check(("Authorization", "Bearer " + value)).status, 401)

    def test_item_1_the_compare_uses_two_digests_and_no_early_return(self):
        calls = []
        real = api_key.hmac.compare_digest

        def spy(a, b):
            calls.append((len(a), len(b)))
            return real(a, b)

        with patch.object(api_key.hmac, "compare_digest", spy):
            self.check(bearer(OTHER), ("x-api-key", "short"))
            self.check(bearer(), ("x-api-key", OTHER))
        # One compare for each present header, also after a first mismatch, and always on 32-byte digests.
        self.assertEqual(calls, [(32, 32)] * 4)

    def test_item_1_no_equals_compare_on_key_text_anywhere(self):
        """Grep test: in the code that handles the key, no == or != has a name that can hold key text."""
        names = re.compile(r"key|secret|digest|given|bearer|candidate|credential|authorization", re.I)
        narrow = re.compile(r"api_key|digest|given|bearer|credential|authorization|secret", re.I)  # http.py has other "key"s
        allowed = {"api_key", "api_key_file", "api_key_digest"}  # a None test or a path, not key text
        for module in ("api_key.py", "http.py", "__main__.py"):
            tree = ast.parse((SERVER_DIR / module).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare) or not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                    continue
                identifiers = [part.id if isinstance(part, ast.Name) else part.attr
                               for operand in [node.left, *node.comparators] for part in ast.walk(operand)
                               if isinstance(part, (ast.Name, ast.Attribute))]
                pattern = names if module == "api_key.py" else narrow
                bad = [name for name in identifiers if pattern.search(name) and name not in allowed]
                self.assertEqual(bad, [], f"{module}:{node.lineno} compares {bad} with ==")
        self.assertIn("compare_digest", (SERVER_DIR / "api_key.py").read_text())


class HandlerTest(unittest.TestCase):
    """Items 2, 4 and 12 through the real handler."""

    def test_a_good_key_reaches_the_engine_both_ways(self):
        for headers in ([bearer()], [("x-api-key", KEY)], [bearer(), ("x-api-key", KEY)]):
            response = request(headers)
            self.assertEqual(status(response), 200)
            self.assertEqual(len(response.backend.submissions), 1)

    def test_with_no_key_configured_nothing_changes(self):
        self.assertEqual(status(request(key=None)), 200)
        self.assertEqual(status(request(key=None, path="/v1/models", method="GET")), 200)
        self.assertEqual(status(request([bearer(OTHER)], key=None)), 200)  # a header is ignored, not checked

    def test_item_2_the_401_has_one_text_one_shape_and_the_challenge_header(self):
        responses = [request([]), request([bearer(OTHER)]), request([("x-api-key", OTHER)]),
                     request([("Authorization", "Basic abc")]), request([("x-api-key", "")])]
        for response in responses:
            self.assertEqual(status(response), 401)
            self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")
            body = json.loads(response.body)
            self.assertEqual(body["error"]["message"], AUTH_TEXT)
            self.assertEqual(body["error"]["type"], "authentication_error")
        self.assertEqual(len({response.body for response in responses}), 1, "missing and wrong give equal bodies")
        self.assertEqual(len({response.headers["Content-Length"] for response in responses}), 1)

    def test_the_401_fits_openai_and_anthropic_clients(self):
        body = json.loads(request([], path=http_module.COUNT_TOKENS_PATH).body)
        self.assertEqual(body["type"], "error")                       # Anthropic: {"type":"error","error":{...}}
        self.assertEqual(body["error"]["type"], "authentication_error")
        self.assertEqual(body["error"]["message"], AUTH_TEXT)         # OpenAI: {"error":{"message":...}}

    def test_a_duplicate_header_gives_400_through_the_handler(self):
        response = request([bearer(), bearer()])
        self.assertEqual(status(response), 400)
        self.assertEqual(json.loads(response.body)["error"]["message"], DUPLICATE_TEXT)
        self.assertEqual(response.backend.submissions, [])

    def test_item_4_the_check_runs_before_the_body_is_read_or_parsed(self):
        bodies = [b"not json", b"", json.dumps(chat(max_tokens=-1)).encode(), b"{" * 100000]
        for raw in bodies:
            with self.subTest(raw=raw[:8]):
                response = request([], raw=raw)
                self.assertEqual(status(response), 401)
                self.assertEqual(response.backend.submissions, [])
        # A bad Content-Type or a chunked body is no reason to answer before the key check.
        for headers in ([("Content-Type", "text/plain")], [("Transfer-Encoding", "chunked")]):
            self.assertEqual(status(request(headers)), 401)

    def test_item_4_no_tokenizer_or_engine_work_before_the_check(self):
        tokenizer = MagicMock()
        backend = ScriptBackend(token_events(["hi"]))
        body = json.dumps(chat()).encode()
        for path in ("/v1/chat/completions", "/v1/completions", "/tokenize", "/v1/tokenize", "/apply-template",
                     http_module.COUNT_TOKENS_PATH):
            response = invoke(backend, tokenizer, path=path, raw=body, service_options=dict(api_key=gate()))
            self.assertEqual(status(response), 401, path)
        self.assertEqual(tokenizer.mock_calls, [])
        self.assertEqual(backend.submissions, [])

    def test_item_12_the_host_check_runs_first(self):
        strict = parse_allowed_hosts("localhost")  # the test Host header is 127.0.0.1:8790
        for method, path in (("POST", "/v1/chat/completions"), ("GET", "/v1/models")):
            response = request([], method=method, path=path, strict=strict)
            self.assertEqual(status(response), 403, "a wrong Host with no key is 403, not 401")
        listed = parse_allowed_hosts("127.0.0.1")
        self.assertEqual(status(request([], method="GET", path="/v1/models", strict=listed)), 401)
        self.assertEqual(status(request([bearer()], method="GET", path="/v1/models", strict=listed)), 200)

    def test_item_12_a_key_works_from_a_reverse_proxy_on_the_same_host(self):
        # Caddy and nginx forward Authorization and the client's own Host header. The default accepts any Host.
        for host in ("127.0.0.1:8790", "ai.example.org", "localhost"):
            response = invoke(ScriptBackend([]), PiecesTokenizer([]), path="/v1/models", method="GET", raw=b"",
                              headers=[("Host", host), *[bearer()]], service_options=dict(api_key=gate()))
            self.assertEqual(status(response), 200, host)

    def test_the_401_is_one_client_error_line_with_the_reason_auth(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            request([bearer(CANARY)])
        lines = [line for line in err.getvalue().splitlines() if line.startswith("crucible_error")]
        self.assertEqual(len(lines), 1)
        fields = json.loads(lines[0].split(" ", 1)[1])
        self.assertEqual((fields["status"], fields["reason"]), (401, "auth"))
        self.assertNotIn("CANARY", err.getvalue())


class RouteTableTest(unittest.TestCase):
    """Item 3: every route needs the key except /health. A new route with no entry in this table fails the test."""
    TABLE = {
        "/v1/chat/completions": "key", "/v1/completions": "key", "/v1/models": "key",
        "/v1/crucible/trace": "key", "/v1/crucible/version": "key", http_module.COUNT_TOKENS_PATH: "key",
        "/tokenize": "key", "/v1/tokenize": "key", "/apply-template": "key",
        "/health": "open",
    }

    def test_item_3_every_route_in_the_source_has_an_entry(self):
        """Every path literal in server/http.py is in the table. A new route adds a literal, so this fails."""
        tree = ast.parse((SERVER_DIR / "http.py").read_text())
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and node.body
                      and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
        literals = {node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
                    and re.fullmatch(r"/[a-z][a-z0-9/_-]*", node.value)}
        self.assertEqual(sorted(literals - set(self.TABLE)), [], "a route has no entry in RouteTableTest.TABLE")
        self.assertEqual(sorted(set(self.TABLE) - literals), [], "the table names a route that http.py lacks")
        self.assertEqual(set(http_module.LOGGED_PATHS) - set(self.TABLE), set())

    def test_item_3_without_the_key_every_route_gives_401_except_health(self):
        for path, rule in self.TABLE.items():
            for method in ("GET", "POST"):
                with self.subTest(path=path, method=method):
                    expected = 200 if rule == "open" and method == "GET" else 401
                    self.assertEqual(status(request([], path=path, method=method)), expected)

    def test_item_3_with_a_wrong_key_every_route_gives_401_except_health(self):
        for path, rule in self.TABLE.items():
            for method in ("GET", "POST"):
                with self.subTest(path=path, method=method):
                    expected = 200 if rule == "open" and method == "GET" else 401
                    self.assertEqual(status(request([bearer(OTHER)], path=path, method=method)), expected)

    def test_item_3_with_the_key_no_route_gives_401(self):
        for path in self.TABLE:
            for method in ("GET", "POST"):
                with self.subTest(path=path, method=method):
                    try:
                        code = status(request([bearer()], path=path, method=method))
                    except Exception:  # the fixture tokenizer lacks a method: the request got past the key check
                        continue
                    self.assertNotEqual(code, 401)

    def test_item_3_an_unknown_path_needs_the_key_before_it_says_404(self):
        for path in ("/", "/v1", "/v1/chat/completions/", "/health/", "/v1/models/x", "/../etc/passwd", "/%68ealth"):
            for method in ("GET", "POST"):
                with self.subTest(path=path, method=method):
                    self.assertEqual(status(request([], path=path, method=method)), 401)
                    self.assertEqual(status(request([bearer()], path=path, method=method)), 404)

    def test_health_shows_only_ok(self):
        response = request([], path="/health", method="GET")
        self.assertEqual(json.loads(response.body), {"status": "ok"})
        self.assertEqual(status(request([], path="/health?x=1", method="GET")), 200)


class CanaryTest(unittest.TestCase):
    """Item 6: a canary key in a wrong header never appears in stderr, metrics, trace, exceptions or repr."""
    # X-Client-Request-Id is not here: the metrics line logs it by design (a gateway id, never a key).
    HEADERS = [("Authorization", f"Basic {CANARY}"), ("x-api-key", CANARY), ("Authorization", f"Bearer {CANARY}"),
               ("Proxy-Authorization", f"Bearer {CANARY}"), ("Cookie", f"k={CANARY}"),
               ("Authorization", CANARY)]

    def run_with(self, pairs, **kw):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            response = request(pairs, **kw)
        return response, err.getvalue()

    def test_item_6_no_output_holds_the_canary(self):
        h = self.HEADERS
        for pairs in ([h[0]], [h[1]], [h[2]], [h[3]], [h[4]], [h[5]], h[:2], h[1:3]):
            for method, path in (("POST", "/v1/chat/completions"), ("GET", "/v1/models"), ("GET", "/health"),
                                 ("POST", "/tokenize"), ("GET", "/v1/crucible/trace")):
                with self.subTest(pairs=[name for name, _ in pairs], path=path):
                    response, err = self.run_with(pairs, method=method, path=path)
                    self.assertNotIn("CANARY", err)
                    self.assertNotIn("CANARY", response.body.decode())
                    self.assertNotIn("CANARY", json.dumps(response.headers))
                    self.assertNotIn("CANARY", json.dumps(response.backend.trace.since(0)))

    def test_item_6_the_error_object_holds_no_header_value(self):
        for pairs in ([("x-api-key", CANARY)], [("Authorization", f"Basic {CANARY}")], [bearer(CANARY)],
                      [bearer(CANARY), bearer(CANARY)]):
            with self.assertRaises(AuthError) as caught:
                authorize(message(*pairs), digest_of(KEY))
            error = caught.exception
            for text in (str(error), repr(error), repr(error.args), str(error.param), str(error.code)):
                self.assertNotIn("CANARY", text)
            self.assertIsNone(error.__cause__)

    def test_item_6_the_gate_and_the_service_do_not_show_the_key_or_its_digest(self):
        key_gate = gate(CANARY)
        service = http_module.Service(ScriptBackend([]), PiecesTokenizer([]), api_key=key_gate)
        digest = digest_of(CANARY)
        for text in (repr(key_gate), str(key_gate), repr(service), repr(vars(service))):
            self.assertNotIn("CANARY", text)
            self.assertNotIn(digest.hex(), text)
        self.assertNotIn(repr(digest), repr(vars(service)))
        self.assertFalse(hasattr(key_gate, "__dict__"), "a slot class has no dict to dump")

    def test_item_6_the_access_log_line_has_no_header_or_query(self):
        err = io.StringIO()
        handler_class = http_module.handler_for(http_module.Service(ScriptBackend([]), PiecesTokenizer([]),
                                                                    api_key=gate()))
        handler = handler_class.__new__(handler_class)
        handler.command, handler.request_version, handler.path = "GET", "HTTP/1.1", f"/v1/models?key={CANARY}"
        handler.client_address = ("127.0.0.1", 1)
        with contextlib.redirect_stderr(err):
            handler.log_request(401, "-")
        self.assertNotIn("CANARY", err.getvalue())

    def test_item_6_the_key_is_not_in_argv_or_the_environment_of_the_server(self):
        with tempfile.TemporaryDirectory() as folder:
            path = key_file(folder, CANARY + "\n")
            before = dict(os.environ)
            err = io.StringIO()
            listener = MagicMock()
            listener.serve_forever.side_effect = KeyboardInterrupt
            argv = ["crucible", "checkpoint", "--host", "0.0.0.0", "--api-key-file", str(path)]
            with patch.object(sys, "argv", argv), patch.object(startup.signal, "signal"), \
                    patch.object(startup, "ModelTokenizer"), patch("server.batch_backend.BatchedEngineBackend"), \
                    patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
                    patch.object(startup, "Service") as service, contextlib.redirect_stderr(err):
                self.assertEqual(startup.main(), 0)
            self.assertEqual(dict(os.environ), before)
            self.assertNotIn("CANARY", err.getvalue())
            self.assertNotIn("CANARY", repr(service.call_args))
            self.assertNotIn("CANARY", " ".join(argv))
            sent = service.call_args.kwargs["api_key"]
            self.assertIsInstance(sent, KeyGate)
            self.assertNotIn("CANARY", repr(sent))

    def test_item_6_the_module_never_prints_or_logs(self):
        self.assertNotRegex((SERVER_DIR / "api_key.py").read_text(), r"print\(|logging|sys\.stderr")


class BindRuleTest(unittest.TestCase):
    """Item 9: no non-loopback address without the key, also for ::, 0.0.0.0 and IPv4-mapped IPv6."""
    LOOPBACK = ("127.0.0.1", "127.0.0.2", "127.255.255.254", "::1")
    NEEDS_KEY = ("0.0.0.0", "::", "192.0.2.7", "198.51.100.9", "169.254.1.1", "2001:db8::7", "fe80::1",
                 "::ffff:127.0.0.1", "::ffff:0.0.0.0", "::ffff:192.0.2.7", "0:0:0:0:0:ffff:7f00:1", "::2",
                 "128.0.0.1", "126.255.255.255", "224.0.0.1", "255.255.255.255", "1.1.1.1")

    def test_item_9_loopback_needs_no_key(self):
        for host in self.LOOPBACK:
            with self.subTest(host=host):
                self.assertTrue(is_loopback(host))
                self.assertIsNone(bind_error(host, has_key=False))

    def test_item_9_every_other_address_needs_the_key(self):
        for host in self.NEEDS_KEY:
            with self.subTest(host=host):
                self.assertFalse(is_loopback(host))
                text = bind_error(host, has_key=False)
                self.assertIn(host, text)
                self.assertIn("--api-key-file", text)
                self.assertIn("reverse proxy", text)
                self.assertIsNone(bind_error(host, has_key=True))

    def test_item_9_the_server_refuses_to_start_with_exit_2_before_anything_loads(self):
        for host in self.NEEDS_KEY:
            with self.subTest(host=host):
                code, err, loaded = self.start(["--host", host])
                self.assertEqual(code, 2)
                self.assertIn("--api-key-file", err)
                self.assertFalse(loaded)

    def test_item_9_loopback_starts_with_no_key(self):
        for host in self.LOOPBACK:
            with self.subTest(host=host):
                code, err, loaded = self.start(["--host", host])
                self.assertEqual(code, 0, err)
                self.assertTrue(loaded)
                self.assertNotIn("API key", err)

    def test_item_9_a_non_loopback_address_starts_with_the_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = key_file(folder)
            for host in ("0.0.0.0", "::", "::ffff:192.0.2.7", "192.0.2.7"):
                with self.subTest(host=host):
                    code, err, _ = self.start(["--host", host, "--api-key-file", str(path)])
                    self.assertEqual(code, 0, err)
                    self.assertIn("API key required", err)
                    self.assertNotIn("no authentication", err)

    def test_the_key_also_works_on_loopback(self):
        with tempfile.TemporaryDirectory() as folder:
            code, err, _ = self.start(["--api-key-file", str(key_file(folder))])
            self.assertEqual(code, 0, err)
            self.assertIn("API key required", err)

    def test_a_bad_key_file_exits_78_before_the_model_loads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = key_file(folder, CANARY + "\n", mode=0o644)
            for arguments in (["--api-key-file", str(path)], ["--host", "0.0.0.0", "--api-key-file", str(path)],
                              ["--api-key-file", str(Path(folder) / "missing")]):
                with self.subTest(arguments=arguments):
                    code, err, loaded = self.start(arguments)
                    self.assertEqual(code, 78)
                    self.assertIn(folder, err)
                    self.assertNotIn("CANARY", err)
                    self.assertFalse(loaded)

    def test_the_flag_needs_an_absolute_path(self):
        code, _, loaded = self.start(["--api-key-file", "relative/key"])
        self.assertEqual(code, 2)
        self.assertFalse(loaded)

    @staticmethod
    def start(arguments):
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        err = io.StringIO()
        with patch.object(sys, "argv", ["crucible", "checkpoint", *arguments]), patch.object(startup.signal, "signal"), \
                patch.object(startup, "ModelTokenizer") as load, patch("server.batch_backend.BatchedEngineBackend"), \
                patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
                patch.object(startup, "Ipv6Server", return_value=listener), \
                patch.object(startup, "Service"), contextlib.redirect_stderr(err):
            try:
                code = startup.main()
            except SystemExit as exit_:
                code = exit_.code
        return code, err.getvalue(), load.called


class ContainerOpenBindTest(unittest.TestCase):
    """Only the Docker image turns on one exception to the bind rule. The image sets
    CRUCIBLE_CONTAINER_OPEN_BIND=1. Then 0.0.0.0 and :: start with no key. Nothing else changes."""
    NAME = "CRUCIBLE_CONTAINER_OPEN_BIND"
    ROOT = Path(__file__).resolve().parents[2]

    def setUp(self):
        """A test host is not a container: a fake marker file stands in for /.dockerenv (the switch needs a marker)."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.marker = Path(folder.name) / ".dockerenv"
        self.marker.write_text("")
        patcher = patch.object(api_key, "CONTAINER_MARKERS", (str(Path(folder.name) / "no-such-file"), str(self.marker)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_switch_does_nothing_without_a_container_marker_file(self):
        self.marker.unlink()
        for host in ("0.0.0.0", "::"):
            with self.subTest(host=host):
                code, err, loaded = self.start(["--host", host], {self.NAME: "1"})
                self.assertEqual(code, 2)
                self.assertIn("--api-key-file", err)
                self.assertFalse(loaded)

    def start(self, arguments, env=None):
        environ = {key: value for key, value in os.environ.items() if key != self.NAME}
        environ.update(env or {})
        with patch.dict(os.environ, environ, clear=True):
            return BindRuleTest.start(arguments)

    def serving_key(self, arguments, env):
        """The api_key argument that main gives to Service."""
        environ = {key: value for key, value in os.environ.items() if key != self.NAME}
        environ.update(env)
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        with patch.dict(os.environ, environ, clear=True), \
                patch.object(sys, "argv", ["crucible", "checkpoint", *arguments]), patch.object(startup.signal, "signal"), \
                patch.object(startup, "ModelTokenizer"), patch("server.batch_backend.BatchedEngineBackend"), \
                patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
                patch.object(startup, "Ipv6Server", return_value=listener), \
                patch.object(startup, "Service") as service, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(startup.main(), 0)
        return service.call_args.kwargs["api_key"]

    def test_a_normal_start_on_all_addresses_with_no_key_still_exits_2(self):
        for host in ("0.0.0.0", "::"):
            with self.subTest(host=host):
                code, err, loaded = self.start(["--host", host])
                self.assertEqual(code, 2)
                self.assertIn("--api-key-file", err)
                self.assertFalse(loaded)

    def test_with_the_switch_and_no_key_all_addresses_start(self):
        for host in ("0.0.0.0", "::"):
            with self.subTest(host=host):
                code, err, loaded = self.start(["--host", host], {self.NAME: "1"})
                self.assertEqual(code, 0, err)
                self.assertTrue(loaded)
                self.assertIsNone(self.serving_key(["--host", host], {self.NAME: "1"}))

    def test_with_the_switch_and_a_key_file_the_key_is_on(self):
        with tempfile.TemporaryDirectory() as folder:
            path = key_file(folder)
            for host in ("0.0.0.0", "::"):
                with self.subTest(host=host):
                    arguments = ["--host", host, "--api-key-file", str(path)]
                    code, err, _ = self.start(arguments, {self.NAME: "1"})
                    self.assertEqual(code, 0, err)
                    self.assertIn("API key required", err)
                    gate_ = self.serving_key(arguments, {self.NAME: "1"})
                    self.assertIsInstance(gate_, KeyGate)
                    with self.assertRaises(AuthError):
                        gate_.check(message())
                    gate_.check(message(("Authorization", "Bearer " + KEY)))

    def test_with_the_switch_and_a_bad_key_file_the_server_still_exits_78(self):
        with tempfile.TemporaryDirectory() as folder:
            code, _, loaded = self.start(["--host", "0.0.0.0", "--api-key-file", str(Path(folder) / "missing")],
                                         {self.NAME: "1"})
            self.assertEqual(code, 78)
            self.assertFalse(loaded)

    def test_only_the_value_1_turns_it_on(self):
        for value in ("", "0", "true", "yes", "on", "2", " 1", "1 ", "01", "1\n", "TRUE", "enabled"):
            with self.subTest(value=value):
                code, _, loaded = self.start(["--host", "0.0.0.0"], {self.NAME: value})
                self.assertEqual(code, 2)
                self.assertFalse(loaded)

    def test_the_switch_is_for_all_addresses_only(self):
        for host in ("192.0.2.7", "2001:db8::7", "::ffff:0.0.0.0", "::ffff:192.0.2.7", "198.51.100.9", "0.0.0.1"):
            with self.subTest(host=host):
                code, _, loaded = self.start(["--host", host], {self.NAME: "1"})
                self.assertEqual(code, 2)
                self.assertFalse(loaded)

    def test_the_switch_changes_nothing_for_loopback(self):
        for host in ("127.0.0.1", "::1"):
            with self.subTest(host=host):
                code, err, _ = self.start(["--host", host], {self.NAME: "1"})
                self.assertEqual(code, 0, err)
                self.assertNotIn("WARNING", err)
                self.assertIsNone(self.serving_key(["--host", host], {self.NAME: "1"}))

    def test_bind_error_takes_the_switch_from_its_argument_not_from_os_environ(self):
        on = {self.NAME: "1"}
        with patch.dict(os.environ, on):
            self.assertIn("0.0.0.0", bind_error("0.0.0.0", has_key=False))
        self.assertIsNone(bind_error("0.0.0.0", has_key=False, environ=on))
        self.assertIsNone(bind_error("::", has_key=False, environ=on))
        self.assertIn("192.0.2.7", bind_error("192.0.2.7", has_key=False, environ=on))
        self.assertIn("0.0.0.0", bind_error("0.0.0.0", has_key=False, environ={self.NAME: "true"}))

    def test_the_container_start_prints_one_warning_line_that_names_the_risk(self):
        for host in ("0.0.0.0", "::"):
            with self.subTest(host=host):
                _, err, _ = self.start(["--host", host], {self.NAME: "1"})
                lines = [line for line in err.splitlines() if line.startswith("WARNING")]
                self.assertEqual(len(lines), 1, err)
                line = lines[0]
                self.assertIn("no API key", line)
                self.assertIn("open server", line)
                self.assertIn("127.0.0.1", line)
                self.assertIn("--api-key-file", line)
                self.assertNotIn(self.NAME, line)

    def test_the_container_start_with_a_key_keeps_the_normal_warning(self):
        with tempfile.TemporaryDirectory() as folder:
            _, err, _ = self.start(["--host", "0.0.0.0", "--api-key-file", str(key_file(folder))], {self.NAME: "1"})
            lines = [line for line in err.splitlines() if line.startswith("WARNING")]
            self.assertEqual(len(lines), 1, err)
            self.assertIn("API key is its only protection", lines[0])

    def test_the_dockerfile_sets_the_switch(self):
        lines = [line.strip() for line in (self.ROOT / "Dockerfile").read_text().splitlines()]
        self.assertIn(f"ENV {self.NAME}=1", lines)
        final_stage = lines[max(i for i, line in enumerate(lines) if line.startswith("FROM ")):]
        self.assertIn(f"ENV {self.NAME}=1", final_stage, "the switch is in the final image stage")
        self.assertNotIn("--api-key-file", "\n".join(line for line in lines if line.startswith("ENTRYPOINT")))

    def test_the_docker_page_names_the_switch_and_no_other_guide_does(self):
        self.assertIn(self.NAME, (self.ROOT / "docs/DOCKER.md").read_text())
        for name in ("README.md", "CHANGELOG.md", "SECURITY.md", "server/README.md",
                     "docs/OPENAI_CLIENTS.md", "docs/ARCHITECTURE.md"):
            path = self.ROOT / name
            if path.exists():
                with self.subTest(name=name):
                    self.assertNotIn(self.NAME, path.read_text())

    def test_the_guides_state_the_rule_with_the_docker_exception(self):
        for name in ("README.md", "CHANGELOG.md", "SECURITY.md", "docs/DOCKER.md"):
            with self.subTest(name=name):
                text = " ".join((self.ROOT / name).read_text().split())
                self.assertIn("A key is required for an address that is not loopback, except in the Docker image", text)
        docker = (self.ROOT / "docs/DOCKER.md").read_text()
        self.assertIn("-p 127.0.0.1:8790:8790", docker)
        self.assertIn("--api-key-file", docker)
        run = docker[docker.index("## Run"):]
        self.assertRegex(run, r"(?s)^.*?-p 127\.0\.0\.1:8790:8790", "the first example publishes to 127.0.0.1 only")
        self.assertLess(run.index("-p 127.0.0.1:8790:8790"), run.index("-p 8790:8790"), "the open publish comes after, with a key")
        self.assertLess(run.index("-p 8790:8790"), run.index("--api-key-file /run/crucible-api-key"))


class ReachableTest(unittest.TestCase):
    """Item 9 end to end on a real socket: the 401 comes from the real server stack."""

    def test_a_real_request_over_a_socket(self):
        service = http_module.Service(ScriptBackend(token_events(["hi"])), PiecesTokenizer(["hi"]), api_key=gate())
        server = startup.bind_server("127.0.0.1", 0, http_module.handler_for(service))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def get(path, headers=None):
            try:
                with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers or {}), timeout=10) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as error:
                return error.code, error.read()

        self.assertEqual(get("/health")[0], 200)
        code, body = get("/v1/models")
        self.assertEqual(code, 401)
        self.assertEqual(json.loads(body)["error"]["message"], AUTH_TEXT)
        self.assertEqual(get("/v1/models", {"Authorization": "Bearer " + OTHER})[0], 401)
        self.assertEqual(get("/v1/models", {"Authorization": "Bearer " + KEY})[0], 200)
        self.assertEqual(get("/v1/models", {"x-api-key": KEY})[0], 200)


class NeighbourPartsTest(unittest.TestCase):
    """Items 10 and 11 are code of the notify part and the service part. These checks start when that code exists."""

    @unittest.skipUnless((SERVER_DIR / "sd_notify.py").exists(), "server/sd_notify.py is not in this tree yet")
    def test_item_10_notify_and_watchdog_names_leave_the_environment_before_the_engine_starts(self):
        from server.sd_notify import Notifier
        env = {"NOTIFY_SOCKET": "/run/x", "WATCHDOG_USEC": "1000000", "WATCHDOG_PID": "2", "PATH": "/bin"}
        Notifier.from_environ(env)
        self.assertEqual(sorted(env), ["PATH"])

    def test_item_11_the_unit_file_has_no_key(self):
        service = Path(__file__).resolve().parents[2] / "tools/crucible_cli/service.py"
        if not service.exists():
            self.skipTest("tools/crucible_cli/service.py is not in this tree yet")
        sys.path.insert(0, str(service.parent.parent))
        try:
            from crucible_cli import service as units
        finally:
            sys.path.pop(0)
        for bind in ("127.0.0.1", "0.0.0.0"):
            text = units.render_unit(repo="/srv/crucible", venv_py="/srv/crucible/.venv/bin/python",
                                     exec_start="/srv/crucible/.crucible/serve.sh", bind=bind, port=8790,
                                     start_timeout=600, version="0.34.0")
            self.assertNotRegex(text, r"crucible-[A-Za-z0-9_-]{20,}")
            self.assertNotRegex(text, r"(?i)Environment=.*(key|token|secret)")
            self.assertNotIn("--api-key-file", text, "the path is in serve.sh, which the unit calls")
        template = Path(__file__).resolve().parents[2] / "deploy/user-systemd/crucible-server.service"
        if template.exists():
            self.assertNotRegex(template.read_text(), r"(?i)crucible-[A-Za-z0-9_-]{20,}|Environment=.*key")


if __name__ == "__main__":
    unittest.main()
