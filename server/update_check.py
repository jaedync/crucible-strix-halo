"""The startup notice about a newer release. It informs only: it never downloads or installs anything.

One plain GET of the public releases API at most once in 24 hours, in a background thread with a short timeout.
The answer is cached in ${XDG_CACHE_HOME:-~/.cache}/crucible/update-check.json. A network error prints nothing.
`CRUCIBLE_NO_UPDATE_CHECK=1` or `--no-update-check` turns the check off."""
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

REPOSITORY = "jaedync/crucible-strix-halo"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASE_URL = f"https://github.com/{REPOSITORY}/releases/tag/"
# Tests only: a fake release server on a loopback port.
URL_ENV = "CRUCIBLE_UPDATE_API_URL"
OFF_ENV = "CRUCIBLE_NO_UPDATE_CHECK"
CACHE_SECONDS = 24 * 3600
TIMEOUT_SECONDS = 3
# The answer for one release is a few KB. A larger body is not a release answer.
MAX_BODY_BYTES = 256 * 1024
VERSION_PATTERN = re.compile(r"v?(\d{1,6})\.(\d{1,6})\.(\d{1,6})", re.ASCII)
log = logging.getLogger("crucible.update")


def parse_version(text):
    """(0, 22, 9) for 'v0.22.9' or '0.22.9'; None for anything else. Tuples compare numerically."""
    if not isinstance(text, str):
        return None
    found = VERSION_PATTERN.fullmatch(text.strip())
    return tuple(int(part) for part in found.groups()) if found else None


def is_newer(latest, current):
    latest, current = parse_version(latest), parse_version(current)
    return latest is not None and current is not None and latest > current


def check_disabled(env, flag=False):
    return bool(flag) or env.get(OFF_ENV, "") not in ("", "0")


def cache_file(env):
    base = env.get("XDG_CACHE_HOME", "")
    if not base or not os.path.isabs(base):
        base = os.path.join(env.get("HOME") or os.path.expanduser("~"), ".cache")
    return Path(base) / "crucible" / "update-check.json"


def read_cache(path):
    """{checked_at, latest} or None when the file is missing or not what this module wrote."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or set(data) != {"checked_at", "latest"}:
        return None
    checked, latest = data["checked_at"], data["latest"]
    if isinstance(checked, bool) or not isinstance(checked, (int, float)) or checked != checked or abs(checked) == float("inf"):
        return None
    if latest is not None and parse_version(latest) is None:
        return None
    return dict(checked_at=float(checked), latest=latest)


def write_cache(path, checked_at, latest):
    """A failed write is not an error: the next start asks again."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
            json.dump(dict(checked_at=checked_at, latest=latest), handle)
        os.replace(handle.name, path)
    except OSError as error:
        log.debug("update check: cache not written: %s", error)


def fetch_latest(url, user_agent, timeout):
    """The tag of the latest published release, or None when the answer names none. Raises on a network error."""
    if not url.startswith(("https://", "http://")):
        raise ValueError("unsupported URL scheme")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=timeout) as answer:
        body = answer.read(MAX_BODY_BYTES + 1)
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("answer too large")
    data = json.loads(body)
    if not isinstance(data, dict) or data.get("draft") is not False or data.get("prerelease") is not False:
        return None
    tag = data.get("tag_name")
    return tag if parse_version(tag) is not None and tag == tag.strip() else None


def latest_tag(current, *, env, now, fetch, timeout):
    """The newest release tag: from a cache younger than 24 h, else from one request. None when unknown."""
    path, moment = cache_file(env), now()
    cached = read_cache(path)
    if cached is not None and 0 <= moment - cached["checked_at"] < CACHE_SECONDS:
        return cached["latest"]
    known = cached["latest"] if cached else None
    try:
        tag = fetch(env.get(URL_ENV) or API_URL, f"crucible/{current}", timeout)
    except Exception as error:  # a network failure must never reach the server; keep the last answer
        log.debug("update check failed: %s", error)
        tag = known
    # A failed request is recorded too, so that a day holds at most one request.
    write_cache(path, moment, tag)
    return tag


def notice_lines(current, tag):
    if not is_newer(tag, current):
        return []
    return [f"Crucible {tag} is available (you run v{current}). Update: make update. Release notes: {RELEASE_URL}{tag}",
            'Automatic updates are off unless you turned on the timer. See "Updates" in the README.']


def start(current, *, disabled=False, out=None, env=None, now=time.time, fetch=fetch_latest, timeout=TIMEOUT_SECONDS):
    """Check in a daemon thread and print the notice if a newer release exists. Returns the thread, or None when off."""
    env = os.environ if env is None else env
    if check_disabled(env, disabled):
        return None

    def work():
        try:
            lines = notice_lines(current, latest_tag(current, env=env, now=now, fetch=fetch, timeout=timeout))
            if lines:
                print("\n".join(lines), file=out or sys.stderr, flush=True)
        except Exception as error:
            log.debug("update check stopped: %s", error)

    thread = threading.Thread(target=work, name="update-check", daemon=True)
    thread.start()
    return thread
