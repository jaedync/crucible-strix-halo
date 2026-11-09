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


class Http:
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
