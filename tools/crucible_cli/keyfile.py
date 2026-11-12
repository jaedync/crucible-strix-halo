"""The API key file (service design, section 3).

The key is in a file with mode 0600, in a folder with mode 0700. config.json, serve.sh and the unit file hold only the
path. Nothing in this module prints, and no error text holds the key. The rules of check_key_file are the rules of
server/api_key.py: a test runs both on the same inputs.
"""
import errno
import ipaddress
import os
import re
import secrets
import stat
from pathlib import Path

KEY_PATTERN = re.compile(r"[A-Za-z0-9._~+/=-]{32,256}")
MAX_FILE_BYTES = 1024
KEY_RULE = "It must hold one key of 32 to 256 characters (A-Z a-z 0-9 . _ ~ + / = -) and a new line."


def default_path(host):
    """$XDG_CONFIG_HOME/crucible/api-key. A relative XDG value is not valid (XDG spec) and is ignored."""
    xdg = host.env.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg and os.path.isabs(xdg) else Path(host.home) / ".config"
    return base / "crucible" / "api-key"


def client_words(s):
    """The sentence that tells a user which model name and API key a client needs."""
    if s.api_key_file is None:
        return 'Use the model name "crucible" and any API key.'
    return (f'Use the model name "crucible". The API key is the text in {s.api_key_file}. '
            "Send it as Authorization: Bearer KEY, or as x-api-key: KEY.")


def address(text):
    """The ip_address object of a bind address. Only an IPv4 or IPv6 literal. Raises ValueError."""
    if not isinstance(text, str) or text != text.strip() or "%" in text:  # a zone id (fe80::1%eth0) is not accepted
        raise ValueError(f"{text!r} is not an IPv4 or IPv6 address")
    return ipaddress.ip_address(text)


def is_loopback(text):
    """127.0.0.0/8 and ::1 only. An IPv4-mapped IPv6 address (::ffff:127.0.0.1) is not loopback: the same rule as the
    server, which then needs the key for it."""
    ip = address(text)
    if ip.version == 4:
        return ip in ipaddress.IPv4Network("127.0.0.0/8")
    return ip == ipaddress.IPv6Address("::1")


def ensure_key(host, path):
    """Make the key file when it does not exist. Return True when made, False when a file or link is there already.

    An existing file is never changed, also when the server would refuse it: doctor says what is wrong. The folder is
    made with mode 0700 when it is missing. The file is made with O_EXCL and O_NOFOLLOW and mode 0600 from the start (never wider, whatever the umask).
    """
    path = Path(path)
    folder = path.parent
    try:
        made = not folder.exists()
        folder.mkdir(parents=True, exist_ok=True)
        if made:
            os.chmod(folder, 0o700)  # mkdir's mode argument is cut by the umask; the chmod is not
    except OSError:
        pass  # the open below fails with a clear error
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    except FileExistsError:
        return False
    except OSError as error:
        if error.errno == errno.ELOOP:
            return False
        raise
    with os.fdopen(fd, "w") as handle:
        os.fchmod(fd, 0o600)  # the create mode is cut by the umask and is never wider than 0600; this sets 0600 exactly
        handle.write("crucible-" + secrets.token_urlsafe(32) + "\n")
    return True


def _read(path):
    """(text, problem). Same order of checks as the server: open without following links, then check the open file."""
    shown = str(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as error:
        if error.errno == errno.ELOOP:
            return None, f"The API key file {shown} is a symbolic link. Use the file itself."
        return None, f"Cannot open the API key file {shown}: {error.strerror}."
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None, f"The API key file {shown} is not a regular file."
        if info.st_uid != os.geteuid():
            return None, (f"The owner of the API key file {shown} is not the user that runs the server. "
                          "Make the file again as that user.")
        if info.st_mode & 0o077:
            return None, f"The API key file {shown} can be read by other users. Run: chmod 600 {shown}"
        if info.st_size > MAX_FILE_BYTES:
            return None, f"The API key file {shown} is larger than {MAX_FILE_BYTES} bytes. {KEY_RULE}"
        raw = os.read(fd, MAX_FILE_BYTES + 1)
    except OSError as error:
        return None, f"Cannot read the API key file {shown}: {error.strerror}."
    finally:
        os.close(fd)
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        text = ""
    if len(raw) > MAX_FILE_BYTES or not KEY_PATTERN.fullmatch(text):
        return None, f"The API key file {shown} does not hold a valid key. {KEY_RULE}"
    return text, None


def check_key_file(path):
    """None when the server accepts the file, else one sentence about the problem. Never the content."""
    return _read(path)[1]


def read_key(path):
    """The key text, or None when the file is missing or the server would refuse it. For the HTTP client only."""
    return _read(path)[0]
