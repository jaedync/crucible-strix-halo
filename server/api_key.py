"""The API key of the HTTP server (service design, section 3).

The key lives in a file that only the server user can read. The server reads it once at start, keeps only its
sha256 digest and drops the text. A request must carry the key in Authorization: Bearer or in x-api-key.

This module never prints or logs. Its errors name the file or the rule, never a key or a header value.
"""
import errno
import hashlib
import hmac
import ipaddress
import os
import re
import stat

from .api import ApiError

# The characters of token_urlsafe plus a few that other tools write. The CLI makes "crucible-" + 43 characters.
KEY_PATTERN = re.compile(r"[A-Za-z0-9._~+/=-]{32,256}")
MAX_FILE_BYTES = 1024
EXIT_BAD_KEY_FILE = 78  # EX_CONFIG: a setting is wrong. The service unit does not restart on it.

AUTH_TEXT = "A valid API key is required. Send it as Authorization: Bearer <key> or x-api-key: <key>."
DUPLICATE_TEXT = "Send each of the Authorization and x-api-key headers at most once."
KEY_RULE = ("It must hold one key of 32 to 256 characters (A-Z a-z 0-9 . _ ~ + / = -) and a new line. "
            "Make a new one with: ./crucible setup --bind ADDRESS")


class KeyFileError(Exception):
    """A key file that the server refuses. The text names the path and the rule, never the content."""


class AuthError(ApiError):
    """401 for a missing or wrong key (one text for both), 400 for a repeated header. Never holds a header value."""

    def __init__(self, message, status=401):
        super().__init__(message, status=status)
        self.kind = "authentication_error" if status == 401 else None
        self.headers = {"WWW-Authenticate": "Bearer"} if status == 401 else {}


def digest_of(text):
    """sha256 of a key or a header value. A lone surrogate becomes "?", so no header value can raise here."""
    return hashlib.sha256(text.encode("utf-8", "replace")).digest()


def read_key(path, uid=None):
    """The key text of a key file. Raises KeyFileError. Only the server start and the CLI check call this."""
    uid = os.geteuid() if uid is None else uid
    shown = str(path)
    try:
        # O_NOFOLLOW: a link is refused. O_NONBLOCK: a FIFO does not block the open. The checks below use the open
        # descriptor, so the file cannot change between the check and the read.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise KeyFileError(f"The API key file {shown} is a symbolic link. Use the file itself.") from None
        raise KeyFileError(f"Cannot open the API key file {shown}: {error.strerror}.") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise KeyFileError(f"The API key file {shown} is not a regular file.")
        if info.st_uid != uid:
            raise KeyFileError(f"The owner of the API key file {shown} is not the user that runs the server. "
                               "Make the file again as that user.")
        if info.st_mode & 0o077:
            raise KeyFileError(f"The API key file {shown} can be read by other users. Run: chmod 600 {shown}")
        if info.st_size > MAX_FILE_BYTES:
            raise KeyFileError(f"The API key file {shown} is larger than {MAX_FILE_BYTES} bytes. {KEY_RULE}")
        raw = os.read(fd, MAX_FILE_BYTES + 1)
    except OSError as error:
        raise KeyFileError(f"Cannot read the API key file {shown}: {error.strerror}.") from None
    finally:
        os.close(fd)
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        text = ""
    if len(raw) > MAX_FILE_BYTES or not KEY_PATTERN.fullmatch(text):
        raise KeyFileError(f"The API key file {shown} does not hold a valid key. {KEY_RULE}")
    return text


def load_digest(path, uid=None):
    """The sha256 digest of the key in the file. The key text is not kept."""
    return digest_of(read_key(path, uid))


def text_of(header):
    """A header value as str. A value with bytes outside ASCII can arrive as an email Header object. A value that
    cannot be turned into text becomes "", which never matches."""
    try:
        return str(header)
    except UnicodeError:
        return ""


def authorize(headers, digest):
    """Return when the request carries the key. Raise AuthError (401, or 400 for a repeated header) when not.

    Every header that is present must match. Each compare is hmac.compare_digest on two sha256 digests, and none
    returns early, so the time depends on neither the length nor the first wrong byte.
    """
    bearers = [text_of(header) for header in headers.get_all("Authorization", [])]
    api_keys = [text_of(header) for header in headers.get_all("x-api-key", [])]
    if len(bearers) > 1 or len(api_keys) > 1:
        raise AuthError(DUPLICATE_TEXT, status=400)
    passed = int(bool(bearers or api_keys))
    for header in bearers:
        scheme, _, given = header.partition(" ")
        same = hmac.compare_digest(digest_of(given.strip()), digest)
        passed &= int(same) & int(scheme.lower() == "bearer")
    for header in api_keys:
        passed &= int(hmac.compare_digest(digest_of(header.strip()), digest))
    if not passed:
        raise AuthError(AUTH_TEXT)


class KeyGate:
    """What the HTTP service holds: the digest only. It has no text to show and no dict to dump."""
    __slots__ = ("_digest",)

    def __init__(self, digest):
        self._digest = digest

    def check(self, headers):
        authorize(headers, self._digest)

    def __repr__(self):
        return "KeyGate(<redacted>)"


def is_loopback(address):
    """127.0.0.0/8 and ::1 only. An IPv4-mapped IPv6 address (::ffff:127.0.0.1) is not loopback here: a dual-stack
    socket could take it from anywhere, so it needs the key like any other address."""
    ip = ipaddress.ip_address(address) if isinstance(address, str) else address
    if ip.version == 4:
        return ip in ipaddress.IPv4Network("127.0.0.0/8")
    return ip == ipaddress.IPv6Address("::1")


# Exception for the Docker image: the image sets this variable. Then 0.0.0.0 and :: start with no key,
# because Docker publishes the port to the container address. Only the value "1" turns it on, and only in a container:
# Docker writes /.dockerenv and Podman writes /run/.containerenv. A variable that leaked into a normal shell (for
# example a user manager environment) opens nothing. Docker with --network host still opens the host network: the
# marker file is there, and so is the switch. The Docker page says so.
CONTAINER_SWITCH = "CRUCIBLE_CONTAINER_OPEN_BIND"
CONTAINER_MARKERS = ("/.dockerenv", "/run/.containerenv")
ALL_ADDRESSES = (ipaddress.IPv4Address("0.0.0.0"), ipaddress.IPv6Address("::"))


def container_open_bind(host, environ):
    """True when the image switch is set to exactly "1", a container marker file exists, and the address is 0.0.0.0 or ::."""
    return (environ.get(CONTAINER_SWITCH) == "1" and ipaddress.ip_address(host) in ALL_ADDRESSES
            and any(os.path.exists(marker) for marker in CONTAINER_MARKERS))


def bind_error(host, has_key, environ=None):
    """The refusal text when this bind address needs a key and none is set, else None.

    environ is the environment that holds the image switch. It is not read from os.environ here."""
    if has_key or is_loopback(host) or container_open_bind(host, environ or {}):
        return None
    return (f"Address {host} is not a loopback address. Add --api-key-file (./crucible setup --bind {host} makes "
            "the key), or use 127.0.0.1 and a reverse proxy.")
