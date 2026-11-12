"""Plain HTTP to the local server, with urllib."""
import json
import time
import urllib.error
import urllib.request

from . import keyfile


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """The server never redirects. A redirect must not carry the key to another address, so it is not followed."""

    def redirect_request(self, *args, **kwargs):
        return None


# No proxy: the key goes to the address that the caller gave and nowhere else.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def _request(url, key_file, data=None, headers=None):
    """The Request. The key goes only in the Authorization header, and only when the file is a good key file."""
    request = urllib.request.Request(url, data=data, headers=dict(headers or {}))
    key = keyfile.read_key(key_file) if key_file is not None else None
    if key is not None:
        request.add_header("Authorization", f"Bearer {key}")
    return request


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    """A download may follow a redirect, but only to https."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# The proxy settings of the environment count here: a download goes to a public host, and no key is sent.
_DOWNLOAD_OPENER = urllib.request.build_opener(_HttpsOnly)


class Http:
    def download(self, url, out, start=None, end=None, timeout=600):
        """Write the bytes START..END (inclusive) of URL to OUT, a binary file that the caller opened (through its Sandbox).
        Returns (status, bytes written); (0, 0) when no server answered."""
        headers = {"User-Agent": "crucible-setup"}
        if start is not None:
            headers["Range"] = f"bytes={start}-{end}"
        written = 0
        try:
            with _DOWNLOAD_OPENER.open(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                return resp.status, written
        except urllib.error.HTTPError as exc:
            return exc.code, 0
        except (OSError, ValueError):
            return (0, 0) if written == 0 else (206, written)

    def get(self, url, timeout=5, key_file=None):
        try:
            with _OPENER.open(_request(url, key_file), timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, ""
        except (OSError, ValueError):
            return 0, ""

    def chat_stream(self, url, body, timeout=900, key_file=None):
        """Send a streaming chat request. Return the time to the first token and the text."""
        req = _request(url, key_file, json.dumps(body).encode(), {"Content-Type": "application/json"})
        start = time.time()
        first = None
        text, tokens = [], 0
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    try:
                        delta = json.loads(line[5:])["choices"][0].get("delta", {})
                    except (ValueError, KeyError, IndexError):
                        continue
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if piece:
                        tokens += 1
                        if first is None:
                            first = time.time() - start
                        if delta.get("content"):
                            text.append(delta["content"])
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        total = time.time() - start
        return {"ok": first is not None, "first_token_s": first, "total_s": total, "text": "".join(text),
                "tokens": tokens}
