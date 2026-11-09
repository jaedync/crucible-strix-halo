"""Host header rule: any Host by default, a strict list with CRUCIBLE_ALLOWED_HOSTS or --allowed-host."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import server.__main__ as startup

from server.hosts import ALLOWED_HOSTS_ENV, HostError, parse_allowed_hosts, resolve_allowed_hosts
from server.tests.serve_fixtures import ScriptBackend, PiecesTokenizer, chat, invoke, token_events

LOOPBACK = "127.0.0.1,localhost,[::1]"


def post(host, allowed=None, method="POST", path="/v1/chat/completions"):
    """One request through the real handler with the given Host header (None: no header)."""
    body = json.dumps(chat()).encode()
    headers = [("Content-Length", str(len(body))), ("Content-Type", "application/json")]
    if host is not None:
        headers.insert(0, ("Host", host))
    backend = ScriptBackend(token_events(["hi"]))
    options = None if allowed is None else dict(allowed_hosts=allowed)
    if method == "GET":
        return invoke(backend, PiecesTokenizer(["hi"]), path="/v1/models", method="GET", raw=b"", headers=headers,
                      service_options=options)
    return invoke(backend, PiecesTokenizer(["hi"]), path=path, raw=body, headers=headers, service_options=options)


def status(response):
    return response.statuses[0]


class DefaultOpenTest(unittest.TestCase):
    def test_proxy_host_is_accepted_by_default(self):
        # llama-swap keeps the client's Host header on the upstream request.
        for method in ("POST", "GET"):
            for host in ("myserver:8080", "llama-swap.lan", "192.0.2.5:8080", "[fe80::1]:8080", "127.0.0.1:8790"):
                with self.subTest(method=method, host=host):
                    self.assertEqual(status(post(host, method=method)), 200)

    def test_missing_host_is_accepted_by_default(self):
        self.assertEqual(status(post(None)), 200)
        self.assertEqual(status(post(None, method="GET")), 200)

    def test_utility_endpoint_follows_the_same_rule(self):
        # read_body() checks the Host for count_tokens, tokenize and apply-template too.
        for allowed, expected in ((None, 200), (parse_allowed_hosts("localhost"), 403)):
            with self.subTest(allowed=allowed):
                self.assertEqual(status(post("proxy:80", allowed, path="/tokenize")), expected)


class StrictTest(unittest.TestCase):
    strict = parse_allowed_hosts(LOOPBACK)

    def test_unlisted_host_is_refused_with_text_that_names_the_switches(self):
        for method in ("POST", "GET"):
            with self.subTest(method=method):
                response = post("myserver:8080", self.strict, method=method)
                self.assertEqual(status(response), 403)
                message = json.loads(response.body)["error"]["message"]
                self.assertIn("CRUCIBLE_ALLOWED_HOSTS", message)
                self.assertIn("--allowed-host", message)
                self.assertNotIn("myserver", message)  # client text is not echoed

    def test_listed_hosts_are_accepted_on_any_port(self):
        for host in ("127.0.0.1:8790", "127.0.0.1:1234", "127.0.0.1", "localhost:8790", "[::1]:8790", "[::1]"):
            with self.subTest(host=host):
                self.assertEqual(status(post(host, self.strict)), 200)
                self.assertEqual(status(post(host, self.strict, method="GET")), 200)

    def test_match_ignores_case(self):
        allowed = parse_allowed_hosts("MyServer:8080")
        for host in ("myserver:8080", "MYSERVER:8080", "MyServer:8080"):
            with self.subTest(host=host):
                self.assertEqual(status(post(host, allowed)), 200)

    def test_host_with_a_port_matches_only_that_port(self):
        allowed = parse_allowed_hosts("myserver:8080")
        self.assertEqual(status(post("myserver:8080", allowed)), 200)
        self.assertEqual(status(post("myserver:9090", allowed)), 403)
        self.assertEqual(status(post("myserver", allowed)), 403)

    def test_host_without_a_port_matches_any_port(self):
        allowed = parse_allowed_hosts("myserver")
        for host in ("myserver:8080", "myserver:1", "myserver"):
            with self.subTest(host=host):
                self.assertEqual(status(post(host, allowed)), 200)

    def test_look_alike_hosts_are_refused(self):
        for host in ("evil.com", "localhost.evil.com:8790", "127.0.0.1.evil.com", "xlocalhost", "[::2]:8790",
                     "localhost@evil", "127.0.0.1:8790, evil.com", "", " "):
            with self.subTest(host=host):
                self.assertEqual(status(post(host, self.strict)), 403)

    def test_missing_host_is_refused(self):
        self.assertEqual(status(post(None, self.strict)), 403)
        self.assertEqual(status(post(None, self.strict, method="GET")), 403)

    def test_refused_request_never_reaches_the_engine(self):
        backend = ScriptBackend(token_events(["hi"]))
        invoke(backend, PiecesTokenizer(["hi"]), chat(), headers=[("Host", "evil.com")],
               service_options=dict(allowed_hosts=self.strict))
        self.assertEqual(backend.submissions, [])


class StrictEdgeTest(unittest.TestCase):
    strict = parse_allowed_hosts(LOOPBACK)

    def test_leading_zeros_in_a_port_do_not_change_it(self):
        self.assertEqual(status(post("127.0.0.1:08790", parse_allowed_hosts("127.0.0.1:8790"))), 200)
        self.assertEqual(status(post("127.0.0.1:8790", parse_allowed_hosts("127.0.0.1:08790"))), 200)
        self.assertEqual(status(post("127.0.0.1:08791", parse_allowed_hosts("127.0.0.1:8790"))), 403)

    def test_one_trailing_dot_names_the_same_host(self):
        for host in ("localhost.", "localhost.:8790", "127.0.0.1."):
            with self.subTest(host=host):
                self.assertEqual(status(post(host, self.strict)), 200)
        self.assertEqual(status(post("localhost", parse_allowed_hosts("localhost."))), 200)
        self.assertEqual(status(post("localhost..", self.strict)), 403)
        self.assertEqual(status(post(".", self.strict)), 403)

    def test_two_host_headers_are_refused_in_strict_mode(self):
        body = json.dumps(chat()).encode()
        headers = [("Host", "127.0.0.1:8790"), ("Host", "evil.com"), ("Content-Length", str(len(body)))]
        backend = ScriptBackend(token_events(["hi"]))
        response = invoke(backend, PiecesTokenizer(["hi"]), chat(), headers=headers,
                          service_options=dict(allowed_hosts=self.strict))
        self.assertEqual(status(response), 400)
        self.assertEqual(backend.submissions, [])
        response = invoke(backend, PiecesTokenizer(["hi"]), path="/v1/models", method="GET", raw=b"",
                          headers=headers, service_options=dict(allowed_hosts=self.strict))
        self.assertEqual(status(response), 400)

    def test_two_host_headers_are_accepted_by_default(self):
        body = json.dumps(chat()).encode()
        headers = [("Host", "a"), ("Host", "b"), ("Content-Length", str(len(body))),
                   ("Content-Type", "application/json")]
        response = invoke(ScriptBackend(token_events(["hi"])), PiecesTokenizer(["hi"]), chat(), headers=headers)
        self.assertEqual(status(response), 200)


POST_PATHS = ("/v1/chat/completions", "/v1/completions", "/tokenize", "/v1/tokenize", "/apply-template",
              "/v1/messages/count_tokens")
GET_PATHS = ("/v1/models", "/v1/crucible/trace", "/health")


class TemplateTokenizer(PiecesTokenizer):
    def render_chat(self, request):
        return "prompt"


def call(path, host, allowed):
    """One request on any path with the given Host header. The body fits each POST path."""
    body = chat() if path != "/v1/completions" else dict(prompt="hi", max_tokens=1)
    if path in ("/tokenize", "/v1/tokenize"):
        body = dict(content="hi")
    backend = ScriptBackend(token_events(["hi"]))
    headers = [("Host", host)]
    if path in GET_PATHS:
        return invoke(backend, TemplateTokenizer(["hi"]), path=path, method="GET", raw=b"", headers=headers,
                      service_options=dict(allowed_hosts=allowed))
    raw = json.dumps(body).encode()
    return invoke(backend, TemplateTokenizer(["hi"]), path=path, raw=raw,
                  headers=headers + [("Content-Length", str(len(raw))), ("Content-Type", "application/json")],
                  service_options=dict(allowed_hosts=allowed))


class EveryPathTest(unittest.TestCase):
    strict = parse_allowed_hosts(LOOPBACK)

    def test_strict_mode_refuses_an_unlisted_host_on_every_path(self):
        for path in POST_PATHS + GET_PATHS:
            with self.subTest(path=path):
                self.assertEqual(status(call(path, "proxy.example", self.strict)), 403)

    def test_strict_mode_serves_a_listed_host_on_every_path(self):
        for path in POST_PATHS + GET_PATHS:
            with self.subTest(path=path):
                self.assertNotEqual(status(call(path, "127.0.0.1:8790", self.strict)), 403)

    def test_the_default_serves_any_host_on_every_path(self):
        for path in POST_PATHS + GET_PATHS:
            with self.subTest(path=path):
                self.assertNotEqual(status(call(path, "proxy.example", None)), 403)


class HealthTest(unittest.TestCase):
    def test_health_answers_200_with_a_small_body(self):
        response = call("/health", "proxy.example", None)
        self.assertEqual(status(response), 200)
        self.assertEqual(json.loads(response.body), dict(status="ok"))

    def test_health_answers_503_when_the_engine_has_stopped(self):
        backend = ScriptBackend(token_events(["hi"]))
        backend.fatal = "The engine stopped."
        response = invoke(backend, PiecesTokenizer(["hi"]), path="/health", method="GET", raw=b"")
        self.assertEqual(status(response), 503)


class ParseTest(unittest.TestCase):
    def test_comma_list_and_blanks(self):
        self.assertEqual(parse_allowed_hosts(" a , B:80 ,,"), (("a", None), ("b", "80")))

    def test_ipv6(self):
        self.assertEqual(parse_allowed_hosts("[::1],[FE80::1]:9"), (("[::1]", None), ("[fe80::1]", "9")))

    def test_empty_means_not_set(self):
        for text in ("", "  ", ",", None):
            with self.subTest(text=text):
                self.assertIsNone(parse_allowed_hosts(text))

    def test_bad_entries_are_errors(self):
        for text in (".", "a b", "http://a", "a/b", "a:", "a:x", "a:99999", ":80", "a:80:90", "[::1", "::1", "*"):
            with self.subTest(text=text), self.assertRaises(HostError):
                parse_allowed_hosts(text)


class ResolveTest(unittest.TestCase):
    def test_default_is_open(self):
        self.assertIsNone(resolve_allowed_hosts([], {}))

    def test_env_turns_strict_mode_on(self):
        self.assertEqual(resolve_allowed_hosts([], {ALLOWED_HOSTS_ENV: "a,b"}), (("a", None), ("b", None)))

    def test_flag_beats_env(self):
        self.assertEqual(resolve_allowed_hosts(["x:1"], {ALLOWED_HOSTS_ENV: "a,b"}), (("x", "1"),))

    def test_flag_is_repeatable_and_takes_a_comma_list(self):
        self.assertEqual(resolve_allowed_hosts(["a", "b:2,c"], {}), (("a", None), ("b", "2"), ("c", None)))

    def test_bad_env_is_an_error_that_names_the_variable(self):
        with self.assertRaises(HostError) as caught:
            resolve_allowed_hosts([], {ALLOWED_HOSTS_ENV: "a b"})
        self.assertIn(ALLOWED_HOSTS_ENV, str(caught.exception))


    def test_a_flag_that_names_no_host_is_an_error_and_keeps_the_env_list(self):
        for flags in ([""], [","], [" "], ["", " "]):
            with self.subTest(flags=flags), self.assertRaises(HostError) as caught:
                resolve_allowed_hosts(flags, {ALLOWED_HOSTS_ENV: "a,b"})
            self.assertIn("--allowed-host", str(caught.exception))


class CommandLineTest(unittest.TestCase):
    def run_server(self, *arguments, env=None):
        return subprocess.run([sys.executable, "-m", "server", "/nonexistent-checkpoint", *arguments],
                              capture_output=True, text=True, timeout=60, env=env)

    def test_bad_flag_value_stops_the_start(self):
        result = self.run_server("--allowed-host", "a b")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--allowed-host", result.stderr)

    def test_an_empty_flag_stops_the_start_even_with_the_env_list_set(self):
        for value in ("", ","):
            with self.subTest(value=value):
                result = self.run_server("--allowed-host", value, env={**os.environ, ALLOWED_HOSTS_ENV: "localhost"})
                self.assertEqual(result.returncode, 2)
                self.assertIn("--allowed-host names no host", result.stderr)

    def test_bad_env_value_stops_the_start(self):
        result = self.run_server(env={**os.environ, ALLOWED_HOSTS_ENV: "http://x"})
        self.assertEqual(result.returncode, 2)
        self.assertIn(ALLOWED_HOSTS_ENV, result.stderr)

    def test_help_names_the_flag_and_the_risk(self):
        result = self.run_server("--help")
        self.assertIn("--allowed-host", result.stdout)
        self.assertIn("rebinding", result.stdout)


class EngineEnvironmentTest(unittest.TestCase):
    def test_the_host_list_is_not_an_engine_switch_and_is_not_reported_as_one(self):
        from server.backend import engine_environment
        env, removed = engine_environment({ALLOWED_HOSTS_ENV: "a", "CRUCIBLE_FOO": "1", "PATH": "/bin"})
        self.assertEqual(removed, ["CRUCIBLE_FOO"])
        self.assertNotIn(ALLOWED_HOSTS_ENV, env)


class EngineLaunchTest(unittest.TestCase):
    """The engine must not see CRUCIBLE_ALLOWED_HOSTS. The state-disk identity hashes every CRUCIBLE_* name it sees."""

    def launched_environment(self, backend_class, extra, **options):
        seen = []
        popen = subprocess.Popen
        engine = "fake_batch_engine.py" if backend_class.__name__ == "BatchedEngineBackend" else "fake_engine.py"

        def launch(arguments, **keywords):
            seen.append(keywords.get("env"))
            return popen([sys.executable, str(Path(__file__).parent / engine)], **{**keywords, "env": keywords.get("env")})

        with patch.dict(os.environ, {ALLOWED_HOSTS_ENV: "localhost", "FAKE_CONTEXT": "64", "FAKE_SLOT_CONTEXTS": "64", **extra}), \
                patch("server.backend.subprocess.Popen", side_effect=launch):
            backend = backend_class("unused", "unused", context=64, **options)
        backend.close()
        return seen

    def test_the_serial_engine_does_not_inherit_the_variable_when_it_is_the_only_one(self):
        from server.backend import EngineBackend
        seen = self.launched_environment(EngineBackend, {})
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0], "Popen got no env, so the engine inherits every variable")
        self.assertNotIn(ALLOWED_HOSTS_ENV, seen[0])

    def test_the_batch_engine_does_not_inherit_the_variable_when_it_is_the_only_one(self):
        from server.batch_backend import BatchedEngineBackend
        seen = self.launched_environment(BatchedEngineBackend, {}, streams=1)
        self.assertIsNotNone(seen[0])
        self.assertNotIn(ALLOWED_HOSTS_ENV, seen[0])

    def test_other_variables_still_reach_the_engine(self):
        from server.backend import EngineBackend
        seen = self.launched_environment(EngineBackend, {"CRUCIBLE_STATE_CACHE": "1", "UNRELATED": "x"})
        self.assertEqual((seen[0].get("CRUCIBLE_STATE_CACHE"), seen[0].get("UNRELATED")), ("1", "x"))


class StartLineTest(unittest.TestCase):
    def start(self, *arguments):
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(sys, "argv", ["crucible", "checkpoint", "--serial", *arguments]), \
                patch.object(startup.signal, "signal"), patch.object(startup, "ModelTokenizer"), \
                patch.object(startup, "EngineBackend"), patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
                patch.dict(os.environ, {}, clear=False), patch("sys.stderr") as err:
            os.environ.pop(ALLOWED_HOSTS_ENV, None)
            self.assertEqual(startup.main(), 0)
        return "".join(call.args[0] for call in err.write.call_args_list)

    def test_the_open_default_says_so(self):
        text = self.start()
        self.assertIn("Host check: off (any Host)", text)
        self.assertIn("Crucible is ready at http://127.0.0.1:8790/v1 (content special tokens: text)", text)

    def test_strict_mode_lists_the_hosts(self):
        text = self.start("--allowed-host", "Localhost,127.0.0.1:08790")
        self.assertIn("Host check: only localhost, 127.0.0.1:8790", text)
        self.assertNotIn("Host check: off", text)  # the conversation-cache notice also holds the word "off"


if __name__ == "__main__":
    unittest.main()


class LongPortTest(unittest.TestCase):
    def test_a_port_longer_than_the_int_limit_is_a_host_error(self):
        from server.hosts import HostError, parse_allowed_hosts
        with self.assertRaises(HostError):
            parse_allowed_hosts("localhost:" + "9" * 5000)

