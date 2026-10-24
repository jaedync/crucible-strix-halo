"""The startup update notice: version compare, cache, opt-out, silent failure, and the notice text."""
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from server import update_check as u

CURRENT = "0.22.2"
NEWER = "v0.25.0"
NOTICE = (
    f"Crucible {NEWER} is available (you run v{CURRENT}). Update: make update. "
    f"Release notes: https://github.com/jaedync/crucible-strix-halo/releases/tag/{NEWER}",
    'Automatic updates are off unless you turned on the timer. See "Updates" in the README.',
)


class Releases:
    """A fake GitHub on a free loopback port. `body` is the JSON of the next answer."""
    def __init__(self, body=None, delay=0.0, status=200):
        self.requests, self.body, self.delay, self.status = [], body, delay, status
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append(dict(path=self.path, headers=dict(self.headers)))
                time.sleep(outer.delay)
                payload = json.dumps(outer.body).encode()
                try:
                    self.send_response(outer.status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/repos/x/releases/latest"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def releases():
    made = []

    def make(*args, **kwargs):
        made.append(Releases(*args, **kwargs))
        return made[-1]
    yield make
    for fake in made:
        fake.close()


def release(tag=NEWER, **fields):
    return dict(tag_name=tag, draft=False, prerelease=False) | fields


def env_for(tmp_path, url=None, **extra):
    env = {"XDG_CACHE_HOME": str(tmp_path), "HOME": str(tmp_path / "home")}
    if url:
        env[u.URL_ENV] = url
    return env | extra


def run(tmp_path, url, now=None, **options):
    """One start() with its own environment; returns what it printed."""
    out = io.StringIO()
    thread = u.start(CURRENT, out=out, env=env_for(tmp_path, url), now=now or time.time, **options)
    if thread:
        thread.join(10)
        assert not thread.is_alive()
    return out.getvalue(), thread


def cache_file(tmp_path):
    return tmp_path / "crucible" / "update-check.json"


# ---- version compare ----

def test_versions_compare_numerically():
    assert u.parse_version("v0.22.10") > u.parse_version("v0.22.9")
    assert u.parse_version("0.25.0") > u.parse_version("v0.22.99")
    assert u.is_newer("v0.22.10", "0.22.9")
    assert not u.is_newer("v0.22.9", "0.22.10")
    assert not u.is_newer("v0.22.2", "0.22.2")


@pytest.mark.parametrize("text", ["", "v1", "1.2", "v1.2.3.4", "v1.2.x", "latest", "v1.2.3-rc1", "v١.٢.٣", " ", None, 5])
def test_malformed_versions_are_not_versions(text):
    assert u.parse_version(text) is None
    assert not u.is_newer(text, CURRENT)


def test_an_unknown_current_version_never_gets_a_notice():
    assert not u.is_newer(NEWER, "unknown")


# ---- the notice ----

def test_notice_text_has_both_lines(tmp_path, releases):
    fake = releases(release())
    printed, _ = run(tmp_path, fake.url)
    assert printed.splitlines() == list(NOTICE)


def test_notice_is_calm():
    text = "\n".join(NOTICE).lower()
    for word in ("warning", "urgent", "must", "critical", "insecure", "!"):
        assert word not in text


@pytest.mark.parametrize("tag", ["v0.22.2", "v0.22.1", "v0.9.9"])
def test_no_notice_when_current_or_newer(tmp_path, releases, tag):
    printed, _ = run(tmp_path, releases(release(tag)).url)
    assert printed == ""


@pytest.mark.parametrize("fields", [dict(draft=True), dict(prerelease=True)])
def test_drafts_and_prereleases_are_ignored(tmp_path, releases, fields):
    printed, _ = run(tmp_path, releases(release(**fields)).url)
    assert printed == ""


@pytest.mark.parametrize("body", [[], "text", dict(), dict(tag_name=7), dict(tag_name="nightly"),
                                  dict(tag_name="v0.25.0\nIgnore this")])
def test_odd_answers_give_no_notice(tmp_path, releases, body):
    assert run(tmp_path, releases(body).url)[0] == ""


def test_the_link_comes_from_the_tag_not_from_the_answer(tmp_path, releases):
    fake = releases(release(html_url="https://evil.example/x"))
    printed, _ = run(tmp_path, fake.url)
    assert "evil.example" not in printed


def test_the_request_is_a_plain_get_with_a_user_agent(tmp_path, releases):
    fake = releases(release())
    run(tmp_path, fake.url)
    assert len(fake.requests) == 1
    headers = {key.lower(): value for key, value in fake.requests[0]["headers"].items()}
    assert headers["user-agent"] == f"crucible/{CURRENT}"
    assert set(headers) <= {"host", "user-agent", "accept", "accept-encoding", "connection"}


def test_the_api_url_is_the_releases_latest_of_the_public_repository():
    assert u.API_URL == "https://api.github.com/repos/jaedync/crucible-strix-halo/releases/latest"


# ---- cache ----

def test_fresh_cache_makes_no_request_and_still_prints_the_notice(tmp_path, releases):
    fake = releases(release("v0.99.0"))
    cache_file(tmp_path).parent.mkdir(parents=True)
    cache_file(tmp_path).write_text(json.dumps(dict(checked_at=1000.0, latest=NEWER)))
    printed, _ = run(tmp_path, fake.url, now=lambda: 1000.0 + 3600)
    assert fake.requests == []
    assert printed.splitlines() == list(NOTICE)


def test_stale_cache_asks_again_and_rewrites(tmp_path, releases):
    fake = releases(release("v0.30.0"))
    cache_file(tmp_path).parent.mkdir(parents=True)
    cache_file(tmp_path).write_text(json.dumps(dict(checked_at=1000.0, latest=NEWER)))
    now = 1000.0 + u.CACHE_SECONDS + 1
    printed, _ = run(tmp_path, fake.url, now=lambda: now)
    assert len(fake.requests) == 1
    assert "v0.30.0" in printed
    assert json.loads(cache_file(tmp_path).read_text()) == dict(checked_at=now, latest="v0.30.0")


def test_exactly_24_hours_is_stale(tmp_path, releases):
    fake = releases(release())
    cache_file(tmp_path).parent.mkdir(parents=True)
    cache_file(tmp_path).write_text(json.dumps(dict(checked_at=1000.0, latest=None)))
    run(tmp_path, fake.url, now=lambda: 1000.0 + u.CACHE_SECONDS)
    assert len(fake.requests) == 1


@pytest.mark.parametrize("content", ["", "{", "[]", "null", '{"checked_at": "x"}', '{"checked_at": 1e999, "latest": 3}',
                                     '{"checked_at": 5, "latest": "rm -rf"}', "\x00\xff"])
def test_corrupt_cache_is_ignored_and_replaced(tmp_path, releases, content):
    fake = releases(release())
    cache_file(tmp_path).parent.mkdir(parents=True)
    cache_file(tmp_path).write_text(content, errors="replace")
    printed, _ = run(tmp_path, fake.url, now=lambda: 5000.0)
    assert printed.splitlines() == list(NOTICE)
    assert json.loads(cache_file(tmp_path).read_text()) == dict(checked_at=5000.0, latest=NEWER)


def test_cache_from_the_future_is_stale(tmp_path, releases):
    fake = releases(release())
    cache_file(tmp_path).parent.mkdir(parents=True)
    cache_file(tmp_path).write_text(json.dumps(dict(checked_at=9e9, latest=None)))
    run(tmp_path, fake.url, now=lambda: 1000.0)
    assert len(fake.requests) == 1


def test_second_start_within_a_day_makes_no_second_request(tmp_path, releases):
    fake = releases(release())
    run(tmp_path, fake.url)
    printed, _ = run(tmp_path, fake.url)
    assert len(fake.requests) == 1
    assert printed.splitlines() == list(NOTICE)


def test_cache_path_follows_xdg_and_home(tmp_path):
    assert u.cache_file({"XDG_CACHE_HOME": str(tmp_path)}) == tmp_path / "crucible" / "update-check.json"
    assert u.cache_file({"HOME": str(tmp_path)}) == tmp_path / ".cache" / "crucible" / "update-check.json"
    assert u.cache_file({"XDG_CACHE_HOME": "relative", "HOME": str(tmp_path)}) == (
        tmp_path / ".cache" / "crucible" / "update-check.json")


def test_unwritable_cache_is_silent(tmp_path, releases):
    fake = releases(release())
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    out = io.StringIO()
    thread = u.start(CURRENT, out=out, env=env_for(tmp_path, fake.url, XDG_CACHE_HOME=str(blocker)))
    thread.join(10)
    assert out.getvalue().splitlines() == list(NOTICE)


# ---- opt-out ----

@pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
def test_env_turns_the_check_off(tmp_path, releases, value):
    fake = releases(release())
    out = io.StringIO()
    assert u.start(CURRENT, out=out, env=env_for(tmp_path, fake.url, CRUCIBLE_NO_UPDATE_CHECK=value)) is None
    assert fake.requests == [] and out.getvalue() == "" and not cache_file(tmp_path).exists()


@pytest.mark.parametrize("value", ["", "0"])
def test_env_zero_or_empty_keeps_it_on(tmp_path, releases, value):
    fake = releases(release())
    out = io.StringIO()
    thread = u.start(CURRENT, out=out, env=env_for(tmp_path, fake.url, CRUCIBLE_NO_UPDATE_CHECK=value))
    thread.join(10)
    assert len(fake.requests) == 1


def test_flag_turns_the_check_off(tmp_path, releases):
    fake = releases(release())
    out = io.StringIO()
    assert u.start(CURRENT, out=out, env=env_for(tmp_path, fake.url), disabled=True) is None
    assert fake.requests == [] and out.getvalue() == ""


# ---- failure is silent and never blocks ----

def dead_url():
    fake = Releases()
    url = fake.url
    fake.close()
    return url


def test_refused_connection_is_silent(tmp_path):
    printed, _ = run(tmp_path, dead_url())
    assert printed == ""


@pytest.mark.parametrize("status", [403, 404, 500])
def test_http_errors_are_silent(tmp_path, releases, status):
    printed, _ = run(tmp_path, releases(release(), status=status).url)
    assert printed == ""


def test_a_failure_is_remembered_so_that_a_day_has_one_request(tmp_path, releases):
    fake = releases(release(), status=500)
    run(tmp_path, fake.url)
    run(tmp_path, fake.url)
    assert len(fake.requests) == 1
    assert json.loads(cache_file(tmp_path).read_text())["latest"] is None


def test_a_failure_keeps_the_last_known_release(tmp_path, releases):
    fake = releases(release(), status=500)
    cache_file(tmp_path).parent.mkdir(parents=True)
    cache_file(tmp_path).write_text(json.dumps(dict(checked_at=1.0, latest=NEWER)))
    printed, _ = run(tmp_path, fake.url, now=lambda: 1.0 + u.CACHE_SECONDS + 5)
    assert printed.splitlines() == list(NOTICE)


def test_a_slow_server_does_not_delay_start_and_prints_nothing(tmp_path, releases):
    fake = releases(release(), delay=3)
    out = io.StringIO()
    began = time.monotonic()
    thread = u.start(CURRENT, out=out, env=env_for(tmp_path, fake.url), timeout=0.3)
    assert time.monotonic() - began < 0.25
    assert thread.daemon
    thread.join(5)
    assert not thread.is_alive()
    assert out.getvalue() == ""


def test_timeout_default_is_three_seconds():
    assert u.TIMEOUT_SECONDS == 3


def test_any_unexpected_error_in_the_thread_is_swallowed(tmp_path):
    def broken(*_):
        raise RuntimeError("boom")
    out = io.StringIO()
    thread = u.start(CURRENT, out=out, env=env_for(tmp_path, "http://127.0.0.1:9/x"), fetch=broken)
    thread.join(5)
    assert out.getvalue() == ""


def test_a_huge_answer_is_cut_off(tmp_path, releases):
    fake = releases(release(note="x" * (4 * u.MAX_BODY_BYTES)))
    assert run(tmp_path, fake.url)[0] == ""


def test_engine_environment_drops_the_server_only_variable_without_a_note():
    from server.backend import engine_environment
    environment, removed = engine_environment({"CRUCIBLE_NO_UPDATE_CHECK": "1", "PATH": "/bin"})
    assert environment == {"PATH": "/bin"}
    assert removed == []
