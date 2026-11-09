"""The API key in ./crucible (design s3): the key file, --bind, the settings, the server command, the HTTP client and
the doctor check. No GPU, no network except a loopback socket, no sudo."""
import io
import json
import os
import re
import stat
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from crucible_cli_fakes import FakeHttp, make_repo, wire  # noqa: E402

from crucible_cli import cli, doctor, keyfile, launch  # noqa: E402
from crucible_cli import settings as settings_mod  # noqa: E402
from crucible_cli.http import Http  # noqa: E402

KEY_TEXT = re.compile(r"crucible-[A-Za-z0-9_-]{43}")
CANARY = "crucible-CANARYCANARYCANARYCANARYCANARYCANARY0123"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.t = Path(self.tmp.name)
        self.repo = make_repo(self.t)
        self.host = wire(self.t)
        self.path = Path(self.host.home) / ".config/crucible/api-key"

    def write_key(self, text=CANARY + "\n", mode=0o600):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.path.write_text(text, encoding="utf-8")
        self.path.chmod(mode)
        return self.path


class DefaultPathTest(Base):
    def test_the_default_is_under_the_config_folder_of_the_home(self):
        self.assertEqual(keyfile.default_path(self.host), Path(self.host.home) / ".config/crucible/api-key")

    def test_xdg_config_home_wins_when_it_is_absolute(self):
        self.host.env["XDG_CONFIG_HOME"] = str(self.t / "xdg")
        self.assertEqual(keyfile.default_path(self.host), self.t / "xdg/crucible/api-key")

    def test_a_relative_xdg_value_is_ignored(self):
        self.host.env["XDG_CONFIG_HOME"] = "relative/dir"
        self.assertEqual(keyfile.default_path(self.host), Path(self.host.home) / ".config/crucible/api-key")


class LoopbackTest(unittest.TestCase):
    def test_loopback_is_127_0_0_0_8_and_the_v6_loopback_only(self):
        for text in ("127.0.0.1", "127.9.9.9", "::1"):
            self.assertTrue(keyfile.is_loopback(text), text)
        for text in ("0.0.0.0", "::", "192.0.2.7", "::ffff:127.0.0.1", "0:0:0:0:0:ffff:7f00:1", "2001:db8::7", "126.0.0.1"):
            self.assertFalse(keyfile.is_loopback(text), text)

    def test_the_cli_and_the_server_agree(self):
        from server.api_key import is_loopback
        for text in ("127.0.0.1", "127.1.2.3", "::1", "0.0.0.0", "::", "::ffff:127.0.0.1", "198.51.100.9", "fe80::1"):
            self.assertEqual(keyfile.is_loopback(text), is_loopback(text), text)

    def test_not_an_address_is_an_error(self):
        for text in ("localhost", "", "256.0.0.1", "[::1]", "::g", "127.0.0.1:80", "fe80::1%eth0", " 127.0.0.1"):
            with self.assertRaises(ValueError, msg=text):
                keyfile.is_loopback(text)


class EnsureKeyTest(Base):
    def test_item_8_it_makes_the_folder_0700_and_the_file_0600_with_a_good_key(self):
        self.assertTrue(keyfile.ensure_key(self.host, self.path))
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)
        text = self.path.read_text()
        self.assertRegex(text, r"\Acrucible-[A-Za-z0-9_-]{43}\n\Z")  # 256 bits of token_urlsafe(32)
        self.assertIsNone(keyfile.check_key_file(self.path))

    def test_item_8_it_never_overwrites(self):
        self.write_key()
        self.assertFalse(keyfile.ensure_key(self.host, self.path))
        self.assertEqual(self.path.read_text(), CANARY + "\n")

    def test_it_never_overwrites_a_file_that_the_server_would_refuse(self):
        self.write_key("short\n", mode=0o644)
        self.assertFalse(keyfile.ensure_key(self.host, self.path))
        self.assertEqual(self.path.read_text(), "short\n")

    def test_a_symlink_is_not_followed_and_not_replaced(self):
        target = self.t / "elsewhere"
        self.path.parent.mkdir(parents=True)
        self.path.symlink_to(target)
        self.assertFalse(keyfile.ensure_key(self.host, self.path))
        self.assertFalse(target.exists())
        self.assertTrue(self.path.is_symlink())

    def test_item_8_two_keys_differ_and_come_from_secrets(self):
        other = self.t / "k2"
        keyfile.ensure_key(self.host, self.path)
        keyfile.ensure_key(self.host, other)
        self.assertNotEqual(self.path.read_text(), other.read_text())

    def test_item_6_it_does_not_print(self):
        out = io.StringIO()
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", out):
            keyfile.ensure_key(self.host, self.path)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(self.host.lines, [])

    def test_item_8_the_modes_do_not_depend_on_the_umask(self):
        for mask in (0o000, 0o022, 0o077):
            old = os.umask(mask)
            try:
                path = self.t / f"u{mask}/crucible/api-key"
                keyfile.ensure_key(self.host, path)
            finally:
                os.umask(old)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, oct(mask))
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700, oct(mask))

    def test_item_8_the_key_comes_from_secrets(self):
        with patch.object(keyfile.secrets, "token_urlsafe", return_value="T" * 43) as made:
            keyfile.ensure_key(self.host, self.path)
        made.assert_called_once_with(32)
        self.assertEqual(self.path.read_text(), "crucible-" + "T" * 43 + "\n")

    def test_item_8_the_file_is_made_with_mode_0600_from_the_start(self):
        modes = []
        real = os.open

        def spy(path, flags, mode=0o777, **kw):
            modes.append((flags, mode))
            return real(path, flags, mode, **kw)

        with patch.object(os, "open", spy):
            keyfile.ensure_key(self.host, self.path)
        flags, mode = modes[-1]
        self.assertEqual(mode, 0o600)
        for flag in (os.O_CREAT, os.O_EXCL, os.O_NOFOLLOW):
            self.assertTrue(flags & flag)


class CheckKeyFileTest(Base):
    def test_a_good_file_is_fine(self):
        self.write_key()
        self.assertIsNone(keyfile.check_key_file(self.path))

    def test_each_problem_gives_a_text_with_the_path_and_never_the_content(self):
        cases = {
            "missing": None,
            "mode": ("a" * 40 + "\n", 0o644),
            "size": ("a" * 5000, 0o600),
            "chars": ("a b " * 20, 0o600),
            "short": ("abc\n", 0o600),
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                if case:
                    self.write_key(*case)
                problem = keyfile.check_key_file(self.path)
                self.assertIn(str(self.path), problem)
                self.assertNotIn("aaaa", problem)
                self.assertNotIn("a b", problem)
        self.write_key(CANARY + " x\n")
        self.assertNotIn("CANARY", keyfile.check_key_file(self.path))

    def test_the_mode_problem_names_chmod_600(self):
        self.write_key(mode=0o640)
        self.assertIn(f"chmod 600 {self.path}", keyfile.check_key_file(self.path))

    def test_a_symlink_and_a_folder_are_problems(self):
        self.write_key()
        link = self.t / "link"
        link.symlink_to(self.path)
        self.assertIn("symbolic link", keyfile.check_key_file(link))
        self.assertIn("regular file", keyfile.check_key_file(self.t))

    def test_the_rules_are_the_ones_of_the_server(self):
        """The same inputs give the same verdict in the CLI and in the server."""
        from server.api_key import KeyFileError, read_key
        texts = ["", "\n", "short", "a" * 31, "a" * 32, "a" * 256, "a" * 257, "a" * 40 + " " + "a" * 40, "é" * 40,
                 CANARY, CANARY + "\r\n", CANARY + "$", "\ufeff" + CANARY]
        for text in texts:
            for mode in (0o600, 0o400, 0o640, 0o666):
                with self.subTest(text=repr(text[:10]), mode=oct(mode)):
                    self.write_key(text, mode)
                    try:
                        read_key(self.path)
                        server_ok = True
                    except KeyFileError:
                        server_ok = False
                    self.assertEqual(keyfile.check_key_file(self.path) is None, server_ok)

    def test_read_key_gives_the_text_or_none(self):
        self.assertIsNone(keyfile.read_key(self.path))
        self.write_key()
        self.assertEqual(keyfile.read_key(self.path), CANARY)
        self.write_key(mode=0o644)
        self.assertIsNone(keyfile.read_key(self.path))


class SettingsTest(Base):
    def build(self, **kw):
        return settings_mod.build(self.repo, self.host, **kw)

    def test_the_default_is_loopback_and_no_key(self):
        s = self.build()
        self.assertEqual((s.bind, s.api_key_file), ("127.0.0.1", None))

    def test_a_non_loopback_bind_gets_the_default_key_path(self):
        for bind in ("0.0.0.0", "::", "192.0.2.7", "::ffff:127.0.0.1"):
            with self.subTest(bind=bind):
                s = self.build(bind=bind)
                self.assertEqual(s.api_key_file, keyfile.default_path(self.host))

    def test_a_loopback_bind_needs_no_key(self):
        for bind in ("127.0.0.1", "127.0.0.2", "::1"):
            self.assertIsNone(self.build(bind=bind).api_key_file)

    def test_api_key_on_loopback_gives_the_default_path(self):
        self.assertEqual(self.build(api_key=True).api_key_file, keyfile.default_path(self.host))

    def test_the_path_is_saved_and_the_key_is_not(self):
        s = self.build(bind="0.0.0.0")
        data = s.to_json()
        self.assertEqual(data["api_key_file"], str(keyfile.default_path(self.host)))
        self.assertIn("api_key_file", settings_mod.SAVED)
        keyfile.ensure_key(self.host, s.api_key_file)
        self.assertNotRegex(json.dumps(data), KEY_TEXT.pattern)

    def test_a_saved_path_is_used_again_and_a_saved_bind_keeps_the_key(self):
        s = self.build(bind="0.0.0.0")
        (self.repo / ".crucible").mkdir(exist_ok=True)
        (self.repo / ".crucible/config.json").write_text(json.dumps(s.to_json()))
        again = self.build()
        self.assertEqual((again.bind, again.api_key_file), ("0.0.0.0", s.api_key_file))
        back = self.build(bind="127.0.0.1")
        self.assertEqual(back.api_key_file, s.api_key_file, "the key stays on when the bind goes back to loopback")

    def test_the_client_url_never_uses_a_wildcard(self):
        self.assertEqual(self.build(bind="0.0.0.0").base_url, "http://127.0.0.1:8790")
        self.assertEqual(self.build(bind="::").base_url, "http://[::1]:8790")
        self.assertEqual(self.build(bind="::1").base_url, "http://[::1]:8790")
        self.assertEqual(self.build(bind="192.0.2.7").base_url, "http://192.0.2.7:8790")
        self.assertEqual(self.build(bind="2001:db8::7", port=9001).base_url, "http://[2001:db8::7]:9001")


class ServerCommandTest(Base):
    def command(self, **kw):
        s = settings_mod.build(self.repo, self.host, **kw)
        return s, launch.server_command(s, "/venv/bin/python")

    def test_a_key_adds_the_flag_with_the_path_only(self):
        s, (env, argv) = self.command(bind="0.0.0.0")
        self.assertEqual(argv[argv.index("--api-key-file") + 1], str(s.api_key_file))
        self.assertEqual(argv[argv.index("--host") + 1], "0.0.0.0")

    def test_loopback_has_no_flag(self):
        _, (_, argv) = self.command()
        self.assertNotIn("--api-key-file", argv)

    def test_item_6_the_key_text_is_in_no_argument_variable_or_script(self):
        s, (env, argv) = self.command(bind="0.0.0.0")
        keyfile.ensure_key(self.host, s.api_key_file)
        key = s.api_key_file.read_text().strip()
        script = launch.serve_script(s, "/venv/bin/python")
        for text in (" ".join(argv), json.dumps(env), script, json.dumps(s.to_json())):
            self.assertNotIn(key, text)
        self.assertIn(str(s.api_key_file), script)

    def test_the_server_accepts_what_the_command_says(self):
        """The command that the CLI builds passes the argument parser of the server (a dry run of the flags)."""
        import server.__main__ as startup
        s, (_, argv) = self.command(bind="0.0.0.0")
        keyfile.ensure_key(self.host, s.api_key_file)
        with patch.object(sys, "argv", ["server", *argv[3:]]), patch.object(startup, "ModelTokenizer"), \
                patch("server.batch_backend.BatchedEngineBackend"), patch.object(startup, "bind_server") as bind, \
                patch.object(startup.signal, "signal"), patch.object(startup, "Service") as service, \
                patch("sys.stderr", io.StringIO()):
            bind.return_value.serve_forever.side_effect = KeyboardInterrupt
            self.assertEqual(startup.main(), 0)
        self.assertIsNotNone(service.call_args.kwargs["api_key"])


class BindFlagTest(Base):
    def run_setup(self, *argv):
        code = cli.main(["setup", "--dry-run", "--yes", *argv], host=self.host, repo=self.repo)
        return code, "\n".join(self.host.lines)

    def test_bind_is_a_flag_of_setup_and_doctor(self):
        parsed = cli.parser().parse_args(["setup", "--bind", "0.0.0.0"])
        self.assertEqual(parsed.bind, "0.0.0.0")
        self.assertEqual(cli.parser().parse_args(["doctor", "--bind", "::"]).bind, "::")
        self.assertIsNone(cli.parser().parse_args(["setup"]).bind)

    def test_a_bad_address_exits_2_and_says_what_is_allowed(self):
        for bad in ("localhost", "example.com", "256.1.1.1", "[::1]", "::g", "127.0.0.1:80", ""):
            with self.subTest(bad=bad), patch("sys.stderr", io.StringIO()) as err:
                with self.assertRaises(SystemExit) as caught:
                    cli.parser().parse_args(["setup", "--bind", bad])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("IPv4 or IPv6", err.getvalue())

    def test_setup_with_a_non_loopback_bind_uses_the_key_path_in_its_settings(self):
        args = cli.parser().parse_args(["setup", "--bind", "0.0.0.0", "--yes"])
        s = cli._settings(args, self.host, self.repo)
        self.assertEqual((s.bind, s.api_key_file), ("0.0.0.0", keyfile.default_path(self.host)))

    def test_api_key_flag_asks_for_the_key_on_loopback(self):
        args = cli.parser().parse_args(["setup", "--api-key", "--yes"])
        s = cli._settings(args, self.host, self.repo)
        self.assertEqual((s.bind, s.api_key_file), ("127.0.0.1", keyfile.default_path(self.host)))

    def test_the_dry_run_names_the_key_file_and_not_the_key(self):
        code, text = self.run_setup("--bind", "0.0.0.0")
        self.assertEqual(code, 0, text)
        self.assertFalse(self.path.exists(), "a dry run changes nothing")
        self.assertNotRegex(text, KEY_TEXT.pattern)

    def test_doctor_help_and_setup_help_say_what_the_flags_do(self):
        text = cli.parser().format_help() + cli.parser()._subparsers._group_actions[0].choices["setup"].format_help()
        self.assertIn("--bind", text)
        self.assertIn("--api-key", text)
        self.assertNotRegex(text, "—")


class HttpKeyTest(Base):
    """Item 6 for the client: the key goes only in the Authorization header, only to the URL that the caller gave,
    and a redirect never carries it elsewhere."""

    def serve(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        return f"http://127.0.0.1:{server.server_address[1]}"

    def recorder(self, status=200, location=None):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization"), self.headers.get("x-api-key")))
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Type", "text/event-stream" if self.path.startswith("/s") else "text/plain")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                seen.append((self.path, self.headers.get("Authorization"), self.headers.get("x-api-key")))
                body = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        return Handler, seen

    def test_get_sends_the_key_as_a_bearer_token_when_a_file_is_given(self):
        handler, seen = self.recorder()
        base = self.serve(handler)
        self.write_key()
        self.assertEqual(Http().get(base + "/v1/models", key_file=self.path)[0], 200)
        self.assertEqual(seen, [("/v1/models", f"Bearer {CANARY}", None)])

    def test_no_file_means_no_header(self):
        handler, seen = self.recorder()
        base = self.serve(handler)
        Http().get(base + "/health")
        Http().get(base + "/health", key_file=None)
        self.assertEqual(seen, [("/health", None, None)] * 2)

    def test_a_bad_file_means_no_header_and_no_crash(self):
        handler, seen = self.recorder()
        base = self.serve(handler)
        for case in (None, ("short\n", 0o600), (CANARY + "\n", 0o644)):
            if case:
                self.write_key(*case)
            self.assertEqual(Http().get(base + "/health", key_file=self.path)[0], 200)
        self.assertEqual([entry[1] for entry in seen], [None] * 3)

    def test_chat_stream_sends_the_key_too(self):
        handler, seen = self.recorder()
        base = self.serve(handler)
        self.write_key()
        res = Http().chat_stream(base + "/v1/chat/completions", {"stream": True}, key_file=self.path)
        self.assertTrue(res["ok"], res)
        self.assertEqual(seen, [("/v1/chat/completions", f"Bearer {CANARY}", None)])

    def test_item_6_a_redirect_is_not_followed_so_the_key_goes_nowhere_else(self):
        target, hits = self.recorder()
        other = self.serve(target)
        redirecting, seen = self.recorder(status=302, location=other + "/stolen")
        base = self.serve(redirecting)
        self.write_key()
        code, _ = Http().get(base + "/v1/models", key_file=self.path)
        self.assertEqual(code, 302)
        self.assertEqual(hits, [])

    def test_item_6_an_error_result_never_holds_the_key(self):
        self.write_key()
        res = Http().chat_stream("http://127.0.0.1:9/v1/chat/completions", {}, key_file=self.path)
        self.assertFalse(res["ok"])
        self.assertNotIn("CANARY", json.dumps(res))
        self.assertEqual(Http().get("http://127.0.0.1:9/x", key_file=self.path), (0, ""))

    def test_the_fake_takes_the_same_keyword(self):
        fake = FakeHttp()
        fake.get("http://x/health", key_file=None)
        fake.chat_stream("http://x/v1", {}, key_file=None)
        self.assertEqual((len(fake.gets), len(fake.posts)), (1, 1))

    def test_the_key_is_not_a_url_part(self):
        seen = []

        def fake_open(request, timeout=None):
            seen.append(request.full_url)
            raise OSError("no")

        self.write_key()
        with patch.object(urllib.request.OpenerDirector, "open", lambda self_, r, data=None, timeout=None: fake_open(r)):
            Http().get("http://127.0.0.1:9/x", key_file=self.path)
        self.assertEqual(seen, ["http://127.0.0.1:9/x"])


class DoctorKeyTest(Base):
    def check(self, **kw):
        s = settings_mod.build(self.repo, self.host, **kw)
        results = {r.id: r for r in doctor.run_checks(self.host, s)}
        return results["key-file"], results

    def test_it_is_the_last_check_and_has_the_id_key_file(self):
        s = settings_mod.build(self.repo, self.host)
        self.assertEqual(doctor.run_checks(self.host, s)[-1].id, "key-file")

    def test_loopback_with_no_key_is_a_skip(self):
        result, _ = self.check()
        self.assertEqual(result.status, "skip")

    def test_a_good_file_passes_and_shows_the_path_only(self):
        self.write_key()
        result, _ = self.check(bind="0.0.0.0")
        self.assertEqual(result.status, "pass")
        self.assertIn(str(self.path), result.detail)
        self.assertNotIn("CANARY", json.dumps(result.as_dict()))

    def test_a_missing_file_before_setup_is_a_skip_because_setup_makes_it(self):
        result, _ = self.check(bind="0.0.0.0")
        self.assertEqual(result.status, "skip")
        self.assertIn("setup makes it", result.detail)

    def test_a_missing_file_after_setup_fails_and_blocks_start_with_the_fix(self):
        (self.repo / ".crucible").mkdir()
        (self.repo / ".crucible/config.json").write_text("{}")
        result, _ = self.check(bind="0.0.0.0")
        self.assertEqual(result.status, "fail")
        self.assertIn("start", result.blocks)
        self.assertIn("./crucible setup --bind 0.0.0.0", result.fix)

    def test_a_file_that_others_can_read_fails_with_the_chmod_text(self):
        self.write_key(mode=0o644)
        result, _ = self.check(bind="0.0.0.0")
        self.assertEqual(result.status, "fail")
        self.assertIn("start", result.blocks)
        self.assertIn(f"chmod 600 {self.path}", result.fix)
        self.assertNotIn("CANARY", json.dumps(result.as_dict()))

    def test_a_file_with_a_bad_key_fails_and_never_shows_it(self):
        self.write_key(CANARY + " with space\n")
        result, _ = self.check(bind="0.0.0.0")
        self.assertEqual(result.status, "fail")
        self.assertNotIn("CANARY", json.dumps(result.as_dict()))

    def test_item_6_doctor_output_text_and_json_never_hold_the_key(self):
        self.write_key()
        s = settings_mod.build(self.repo, self.host, bind="0.0.0.0")
        for as_json in (False, True):
            self.host.lines.clear()
            doctor.main(self.host, s, as_json=as_json)
            self.assertNotIn("CANARY", "\n".join(self.host.lines))

    def test_the_other_checks_are_the_same_with_the_key(self):
        _, with_key = self.check(bind="0.0.0.0")
        s = settings_mod.build(self.repo, self.host)
        base = {r.id for r in doctor.run_checks(self.host, s)}
        self.assertEqual(set(with_key), base)


class ClientWordsTest(Base):
    def test_without_a_key_the_words_say_any_key(self):
        s = settings_mod.build(self.repo, self.host)
        self.assertEqual(keyfile.client_words(s), 'Use the model name "crucible" and any API key.')

    def test_with_a_key_the_words_name_the_file_and_never_the_key(self):
        s = settings_mod.build(self.repo, self.host, bind="0.0.0.0")
        keyfile.ensure_key(self.host, s.api_key_file)
        words = keyfile.client_words(s)
        self.assertIn(str(s.api_key_file), words)
        self.assertNotRegex(words, KEY_TEXT.pattern)
        self.assertNotIn("any API key", words)


class SetupFlowTest(Base):
    """./crucible setup with fakes: the key file is made, used for the test request and kept out of every output."""

    def setup_run(self, *argv):
        code = cli.main(["setup", "--yes", *argv], host=self.host, repo=self.repo)
        return code, "\n".join(self.host.lines)

    def test_a_non_loopback_bind_makes_the_key_and_uses_it(self):
        code, out = self.setup_run("--bind", "0.0.0.0")
        self.assertEqual(code, 0, out)
        self.assertIsNone(keyfile.check_key_file(self.path))
        key = self.path.read_text().strip()
        self.assertNotIn(key, out)
        self.assertIn(str(self.path), out)
        argv = self.host.runner.spawned[0]["argv"]
        self.assertEqual(argv[argv.index("--api-key-file") + 1], str(self.path))
        self.assertNotIn(key, " ".join(argv))
        self.assertNotIn(key, json.dumps(self.host.runner.spawned[0].get("env", {})))
        self.assertEqual(self.host.http.key_files[-1], self.path, "the test request carries the key file")
        config = (self.repo / ".crucible/config.json").read_text()
        self.assertIn(str(self.path), config)
        self.assertNotIn(key, config)
        self.assertNotIn(key, (self.repo / ".crucible/serve.sh").read_text())
        self.assertNotIn("any API key", out)
        self.assertIn("http://127.0.0.1:8790/v1", out, "a wildcard bind is reached by loopback")

    def test_the_default_run_has_no_key_file_and_no_flag(self):
        code, out = self.setup_run()
        self.assertEqual(code, 0, out)
        self.assertFalse(self.path.exists())
        self.assertNotIn("--api-key-file", self.host.runner.spawned[0]["argv"])
        self.assertEqual(set(self.host.http.key_files), {None})
        self.assertIn("any API key", out)

    def test_a_second_run_keeps_the_key(self):
        self.setup_run("--bind", "0.0.0.0")
        before = self.path.read_text()
        self.host.lines.clear()
        code, out = self.setup_run("--bind", "0.0.0.0")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.path.read_text(), before)

    def test_api_key_flag_on_loopback(self):
        code, out = self.setup_run("--api-key")
        self.assertEqual(code, 0, out)
        self.assertIsNone(keyfile.check_key_file(self.path))
        self.assertIn("--api-key-file", self.host.runner.spawned[0]["argv"])

    def test_the_sandbox_allows_only_the_key_folder_outside_the_repo(self):
        from crucible_cli import steps
        s = settings_mod.build(self.repo, self.host, bind="0.0.0.0")
        run = steps.Setup(self.host, s)
        run.box.check(s.api_key_file)
        from crucible_cli.files import OutsideRoots
        with self.assertRaises(OutsideRoots):
            run.box.check(self.path.parent.parent / "other")

    def test_a_dry_run_makes_nothing(self):
        code, out = self.setup_run("--bind", "0.0.0.0", "--dry-run")
        self.assertEqual(code, 0, out)
        self.assertFalse(self.path.exists())
        self.assertFalse((self.repo / ".crucible/config.json").exists())


class NoKeyInFilesTest(Base):
    def test_item_6_config_and_serve_script_hold_the_path_and_never_the_key(self):
        from crucible_cli import steps
        s = settings_mod.build(self.repo, self.host, bind="0.0.0.0")
        keyfile.ensure_key(self.host, s.api_key_file)
        key = s.api_key_file.read_text().strip()
        (self.repo / ".crucible").mkdir(exist_ok=True)
        (self.repo / ".crucible/config.json").write_text(json.dumps(s.to_json()))
        for path in (self.repo / ".crucible/config.json",):
            self.assertNotIn(key, path.read_text())
        self.assertNotIn(key, launch.serve_script(s, "/venv/bin/python"))
        self.assertTrue(hasattr(steps, "Setup"))


if __name__ == "__main__":
    unittest.main()
