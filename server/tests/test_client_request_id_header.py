"""The header that carries the gateway's request id has a configurable name (--client-request-id-header or
CRUCIBLE_CLIENT_REQUEST_ID_HEADER). Default X-Client-Request-Id. The value rules do not change."""
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

from server import metrics
from server.request_id import (CLIENT_REQUEST_ID_HEADER_ENV, DEFAULT_CLIENT_REQUEST_ID_HEADER, HeaderNameError,
                               resolve_client_request_id_header)
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, chat, invoke, token_events

ID = "0123456789abcdef0123456789abcdef"
CUSTOM_NAME = "X-Gateway-Request-Id"


def served(headers, header_name=None):
    body = json.dumps(chat()).encode()
    base = [("Host", "127.0.0.1:8790"), ("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
    options = None if header_name is None else dict(client_request_id_header=header_name)
    backend = ScriptBackend(token_events(["hi"]))
    response = invoke(backend, PiecesTokenizer(["hi"]), raw=body, headers=base + headers, service_options=options)
    return backend, response


class ResolveTest(unittest.TestCase):
    def test_default_name(self):
        self.assertEqual(DEFAULT_CLIENT_REQUEST_ID_HEADER, "X-Client-Request-Id")
        self.assertEqual(resolve_client_request_id_header(None, {}), DEFAULT_CLIENT_REQUEST_ID_HEADER)

    def test_env_sets_the_name_and_the_flag_beats_it(self):
        self.assertEqual(resolve_client_request_id_header(None, {CLIENT_REQUEST_ID_HEADER_ENV: "X-Gw-Id"}), "X-Gw-Id")
        self.assertEqual(resolve_client_request_id_header("X-Flag", {CLIENT_REQUEST_ID_HEADER_ENV: "X-Gw-Id"}), "X-Flag")

    def test_an_empty_env_value_is_the_default(self):
        self.assertEqual(resolve_client_request_id_header(None, {CLIENT_REQUEST_ID_HEADER_ENV: ""}),
                         DEFAULT_CLIENT_REQUEST_ID_HEADER)

    def test_a_bad_name_is_an_error_that_says_where_it_came_from(self):
        for bad in ("", "a b", "a:b", "a/b", "x" * 65, "id\u00e9", "a\nb", "a;b"):
            with self.subTest(bad=bad), self.assertRaises(HeaderNameError) as caught:
                resolve_client_request_id_header(bad, {})
            self.assertIn("--client-request-id-header", str(caught.exception))
        with self.assertRaises(HeaderNameError) as caught:
            resolve_client_request_id_header(None, {CLIENT_REQUEST_ID_HEADER_ENV: "a b"})
        self.assertIn(CLIENT_REQUEST_ID_HEADER_ENV, str(caught.exception))


class HeaderTest(unittest.TestCase):
    def test_the_default_name_is_read_in_any_case(self):
        for name in ("X-Client-Request-Id", "x-client-request-id", "X-CLIENT-REQUEST-ID"):
            backend, response = served([(name, ID)])
            self.assertEqual([options.get("client_request_id") for _, _, options in backend.submissions], [ID], name)
            self.assertEqual(response.statuses, [200])
            self.assertNotIn(ID.encode(), response.body)

    def test_a_configured_name_is_read_and_the_default_name_is_not(self):
        backend, _ = served([(CUSTOM_NAME, ID)], header_name=CUSTOM_NAME)
        self.assertEqual(backend.submissions[0][2].get("client_request_id"), ID)
        backend, _ = served([("x-" + CUSTOM_NAME[2:].lower(), ID)], header_name=CUSTOM_NAME)  # name match ignores case
        self.assertEqual(backend.submissions[0][2].get("client_request_id"), ID)
        backend, _ = served([("X-Client-Request-Id", ID)], header_name=CUSTOM_NAME)
        self.assertNotIn("client_request_id", backend.submissions[0][2])

    def test_a_custom_name_is_not_read_by_default(self):
        backend, _ = served([(CUSTOM_NAME, ID)])
        self.assertNotIn("client_request_id", backend.submissions[0][2])

    def test_an_invalid_value_is_dropped_and_an_absent_header_changes_nothing(self):
        for header in ([("X-Client-Request-Id", "bad id")], [("X-Client-Request-Id", "a" * 65)], []):
            backend, response = served(header)
            self.assertNotIn("client_request_id", backend.submissions[0][2])
            self.assertEqual(response.statuses, [200])

    def test_the_log_field_keeps_its_name(self):
        lines = []
        body = json.dumps(chat(max_tokens=-1)).encode()  # a refused request writes a crucible_error line
        headers = [("Host", "127.0.0.1:8790"), ("Content-Type", "application/json"),
                   ("Content-Length", str(len(body))), (CUSTOM_NAME, ID)]
        with mock.patch.object(metrics, "emit", lines.append):
            invoke(ScriptBackend([]), PiecesTokenizer(["hi"]), raw=body, headers=headers,
                   service_options=dict(client_request_id_header=CUSTOM_NAME))
        errors = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("crucible_error ")]
        self.assertEqual([line.get("client_request_id") for line in errors], [ID])


class CommandLineTest(unittest.TestCase):
    def run_server(self, *arguments, env=None):
        return subprocess.run([sys.executable, "-m", "server", "/nonexistent-checkpoint", *arguments],
                              capture_output=True, text=True, timeout=60, env=env)

    def test_a_bad_flag_stops_the_start(self):
        result = self.run_server("--client-request-id-header", "a b")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--client-request-id-header", result.stderr)

    def test_a_bad_env_value_stops_the_start(self):
        result = self.run_server(env={**os.environ, CLIENT_REQUEST_ID_HEADER_ENV: "a b"})
        self.assertEqual(result.returncode, 2)
        self.assertIn(CLIENT_REQUEST_ID_HEADER_ENV, result.stderr)

    def test_help_names_the_flag(self):
        self.assertIn("--client-request-id-header", self.run_server("--help").stdout)


class EngineEnvironmentTest(unittest.TestCase):
    def test_the_setting_is_a_server_setting_not_an_engine_switch(self):
        from server.backend import engine_environment
        env, removed = engine_environment({CLIENT_REQUEST_ID_HEADER_ENV: CUSTOM_NAME, "PATH": "/bin"})
        self.assertEqual(removed, [])
        self.assertNotIn(CLIENT_REQUEST_ID_HEADER_ENV, env)


class StartTest(unittest.TestCase):
    """main() hands the resolved name to the Service that the handler reads."""

    def start(self, arguments, environ):
        import server.__main__ as startup
        listener = mock.MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        real = startup.Service
        seen = []

        def service(*args, **kwargs):
            seen.append(kwargs["client_request_id_header"])
            return real(*args, **kwargs)

        clean = {key: value for key, value in os.environ.items() if key != CLIENT_REQUEST_ID_HEADER_ENV}
        with mock.patch.object(sys, "argv", ["crucible", "checkpoint", "--serial", *arguments]), \
                mock.patch.object(startup.signal, "signal"), mock.patch.object(startup, "ModelTokenizer"), \
                mock.patch.object(startup, "EngineBackend"), mock.patch.object(startup, "Service", service), \
                mock.patch.object(startup, "ThreadingHTTPServer", return_value=listener), \
                mock.patch.dict(os.environ, {**clean, **environ}, clear=True), mock.patch("sys.stderr"):
            self.assertEqual(startup.main(), 0)
        return seen[0]

    def test_default_env_and_flag(self):
        self.assertEqual(self.start([], {}), DEFAULT_CLIENT_REQUEST_ID_HEADER)
        self.assertEqual(self.start([], {CLIENT_REQUEST_ID_HEADER_ENV: CUSTOM_NAME}), CUSTOM_NAME)
        self.assertEqual(self.start(["--client-request-id-header", "X-Flag-Id"], {CLIENT_REQUEST_ID_HEADER_ENV: CUSTOM_NAME}),
                         "X-Flag-Id")
