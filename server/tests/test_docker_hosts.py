"""Docker image and the Host list (choice B): the image sets no default list. The page
docs/DOCKER.md warns and gives two opt-in fixes. The documented env value must turn the list on in the image start path."""
import contextlib
import io
import json
import os
import re
import shlex
import sys
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import server.__main__ as startup
from server import api_key
from server.hosts import ALLOWED_HOSTS_ENV
from server.tests.serve_fixtures import PiecesTokenizer, ScriptBackend, chat, token_events

ROOT = Path(__file__).resolve().parents[2]
SWITCH = "CRUCIBLE_CONTAINER_OPEN_BIND"
DOCS_ENV_LINE = "-e CRUCIBLE_ALLOWED_HOSTS=localhost,127.0.0.1,[::1]"


def docker_page():
    return (ROOT / "docs/DOCKER.md").read_text()


def image_arguments():
    """The arguments after the image name in the first docker run line of docs/DOCKER.md (the entry point is python -m server).
    The line is wrapped with backslashes: join it, then split it as a shell does."""
    run = docker_page().split("## Run", 1)[1]
    block = run.split("```sh", 1)[1].split("```", 1)[0]
    words = shlex.split(block.replace("\\\n", " "))
    image = next(i for i, word in enumerate(words) if re.fullmatch(r"crucible:v\d+\.\d+\.\d+", word))
    return words[image + 1:]


def documented_env_value():
    match = re.search(r"-e CRUCIBLE_ALLOWED_HOSTS=([^\s`]+)", docker_page())
    return match.group(1) if match else None


class RunBlockParserTest(unittest.TestCase):
    """The parser reads the whole documented run block (it is wrapped with backslashes), not its last line."""

    def test_the_image_arguments_are_every_word_after_the_image_name(self):
        run = docker_page().split("## Run", 1)[1].split("```sh", 1)[1].split("```", 1)[0]
        words = shlex.split(run.replace("\\\n", " "))
        image = next(i for i, word in enumerate(words) if re.fullmatch(r"crucible:v\d+\.\d+\.\d+", word))
        self.assertEqual(image_arguments(), words[image + 1:])
        self.assertEqual(image_arguments()[0], "/models/pack")
        self.assertIn("--vision", image_arguments())


class ImageStartPathTest(unittest.TestCase):
    """main() with the arguments of the docker run line and the image environment. Fake marker file for /.dockerenv."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        marker = Path(folder.name) / ".dockerenv"
        marker.write_text("")
        patcher = patch.object(api_key, "CONTAINER_MARKERS", (str(marker),))
        patcher.start()
        self.addCleanup(patcher.stop)

    def start(self, extra_env):
        """Start the server as the image does. Returns (handler class, stderr text)."""
        environ = {k: v for k, v in os.environ.items() if k not in (ALLOWED_HOSTS_ENV, SWITCH)}
        environ[SWITCH] = "1"  # the Dockerfile sets this
        environ.update(extra_env)
        listener = MagicMock()
        listener.serve_forever.side_effect = KeyboardInterrupt
        captured = []
        backend = ScriptBackend(token_events(["hi"]))
        backend.close = lambda: None
        err = io.StringIO()
        with patch.dict(os.environ, environ, clear=True), \
                patch.object(sys, "argv", ["crucible", *image_arguments()]), patch.object(startup.signal, "signal"), \
                patch.object(startup, "ModelTokenizer", return_value=PiecesTokenizer(["hi"])), \
                patch("server.batch_backend.BatchedEngineBackend", return_value=backend), \
                patch.object(startup, "bind_server", side_effect=lambda host, port, handler: captured.append(handler) or listener), \
                contextlib.redirect_stderr(err):
            self.assertEqual(startup.main(), 0, err.getvalue())
        return captured[0], err.getvalue()

    @staticmethod
    def status_for(handler_class, host):
        body = json.dumps(chat()).encode()
        handler = handler_class.__new__(handler_class)
        handler.path = "/v1/chat/completions"
        handler.headers = Message()
        for name, value in (("Host", host), ("Content-Length", str(len(body))), ("Content-Type", "application/json")):
            handler.headers[name] = value
        handler.server = SimpleNamespace(server_address=("0.0.0.0", 8790))
        handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
        statuses = []
        handler.send_response = statuses.append
        handler.send_header = lambda name, value: None
        handler.end_headers = lambda: None
        handler.do_POST()
        return statuses[0]

    def test_the_image_arguments_are_the_documented_ones(self):
        arguments = image_arguments()
        self.assertEqual(arguments[:1], ["/models/pack"])
        self.assertIn("--host", arguments)
        self.assertIn("0.0.0.0", arguments)

    def test_with_no_env_the_image_accepts_any_host(self):
        handler, err = self.start({})
        self.assertIn("Host check: off (any Host)", err)
        for host in ("127.0.0.1:8790", "evil.example"):
            with self.subTest(host=host):
                self.assertEqual(self.status_for(handler, host), 200)

    def test_the_documented_env_value_turns_the_host_list_on(self):
        value = documented_env_value()
        self.assertIsNotNone(value, "docs/DOCKER.md must show: " + DOCS_ENV_LINE)
        handler, err = self.start({ALLOWED_HOSTS_ENV: value})
        self.assertIn("Host check: only localhost, 127.0.0.1, [::1]", err)
        self.assertEqual(self.status_for(handler, "evil.example"), 403)
        self.assertEqual(self.status_for(handler, "127.0.0.1:8790"), 200)
        self.assertEqual(self.status_for(handler, "localhost:8790"), 200)
        self.assertEqual(self.status_for(handler, "[::1]:8790"), 200)

    def test_the_documented_env_value_is_the_one_of_the_owner_answer(self):
        self.assertEqual(documented_env_value(), "localhost,127.0.0.1,[::1]")


class DockerfileTest(unittest.TestCase):
    def test_the_dockerfile_sets_no_host_list(self):
        text = (ROOT / "Dockerfile").read_text()
        self.assertNotIn(ALLOWED_HOSTS_ENV, text)
        self.assertNotIn("--allowed-host", text)
        self.assertNotIn("allowed_hosts", text.lower())


class DocsTest(unittest.TestCase):
    def test_the_docker_page_warns_about_dns_rebinding_with_no_key(self):
        text = docker_page()
        self.assertIn("DNS rebinding", text)
        warning = text[text.index("DNS rebinding") - 600:text.index("DNS rebinding") + 900]
        self.assertIn("no key", warning)
        self.assertIn("--api-key-file", warning)
        self.assertIn(DOCS_ENV_LINE, warning)

    def test_the_docker_page_keeps_the_loopback_publish_in_the_default_example(self):
        run = docker_page().split("## Run", 1)[1].split("```sh", 1)[1].split("```", 1)[0]
        self.assertIn("-p 127.0.0.1:8790:8790", run)
        self.assertNotIn(ALLOWED_HOSTS_ENV, run)

    def test_the_docker_page_has_no_em_dash(self):
        self.assertNotIn("\u2014", docker_page())

    def test_security_says_the_image_has_no_host_list_by_default(self):
        text = (ROOT / "SECURITY.md").read_text()
        docker = text.split("## Docker", 1)[1]
        self.assertIn("no Host list", docker)
        self.assertIn(ALLOWED_HOSTS_ENV, docker)
        rebinding = text.split("## Host names and DNS rebinding", 1)[1].split("\n## ", 1)[0]
        self.assertIn("Docker image", rebinding)

    def test_the_changelog_v034_part_says_it_too(self):
        text = (ROOT / "CHANGELOG.md").read_text()
        part = re.split(r"## v0\.34\.0 \([^)]*\)", text, maxsplit=1)[1].split("\n## v", 1)[0]
        self.assertRegex(part, r"(?s)Docker.{0,900}no Host list")
        self.assertIn(DOCS_ENV_LINE.replace("-e ", ""), part)


if __name__ == "__main__":
    unittest.main()
