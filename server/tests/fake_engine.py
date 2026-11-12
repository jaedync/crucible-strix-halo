"""A small JSON-line engine for backend ownership tests. It does not use HIP."""
import hashlib
import json
import os
import struct
import sys
import time

mode = os.environ.get("FAKE_ENGINE_MODE", "")
context = int(os.environ.get("FAKE_CONTEXT", "32"))
delay = float(os.environ.get("FAKE_DELAY", "0.002"))
log = os.environ.get("FAKE_REQUEST_LOG")
pixel_log = os.environ.get("FAKE_PIXEL_LOG")

def pixels_ok(image):
    """As the engine: the file holds height * width * 3 bytes whose digest matches."""
    try:
        with open(image["path"], "rb") as file: data = file.read()
    except OSError:
        return False
    shape = struct.pack("<II", image["height"], image["width"])
    return hashlib.sha256(shape + data).hexdigest() == image["digest"]


def log_pixels(ok):
    if pixel_log:
        with open(pixel_log, "a") as file: file.write(("ok" if ok else "bad") + "\n")
if mode in ("invalid-json", "oversized", "exit"):
    if mode == "invalid-json": print("not JSON", flush=True)
    if mode == "oversized": print("x" * 65537, flush=True)
    sys.exit(0)
ready = dict(event="ready", context=context + (mode == "wrong-context"))
if not os.environ.get("FAKE_NO_CACHE_TAIL"): ready["cache_tail"] = True
# The model fields as engine/serve/serve_protocol.h ReadyModelJson writes them.
ready["model"] = dict(arch="qwen4_exp", vocab=248320, eos=248044, layers=48, hidden=2560)
# FAKE_ABLATE_REQUEST=1: the engine was started in request mode (CRUCIBLE_ABLATE_MODE=request).
if os.environ.get("FAKE_ABLATE_REQUEST") == "1": ready["ablate_request"] = 1
line_limit = int(os.environ.get("FAKE_LINE_LIMIT", "0"))
# As engine/serve/serve_line.h: the engine reports its line limit (ServeLineBytes of its context). FAKE_NO_LINE_BYTES=1
# copies an older engine, which enforces the limit but does not report it.
if line_limit and os.environ.get("FAKE_NO_LINE_BYTES") != "1": ready["line_bytes"] = line_limit
print(json.dumps(ready), flush=True)
for line in sys.stdin:
    if line_limit and len(line) - 1 > line_limit:
        print(json.dumps(dict(event="error", error="request exceeds the serve JSON-line limit")), flush=True)
        continue
    request = json.loads(line)
    if log:
        with open(log, "a") as file: file.write(json.dumps(request) + "\n")
    changed = request["ids"][-1] == int(os.environ.get("FAKE_CHANGE_TOKEN", "-1"))
    for image in request.get("images", [])[:1] if changed else ():
        with open(image["path"], "ab") as file: file.write(b"x")  # the file changes before the engine reads it
    checks = [pixels_ok(image) for image in request.get("images", [])]
    for ok in checks: log_pixels(ok)
    if not all(checks):
        print(json.dumps(dict(event="error", error="an image file does not match its digest")), flush=True)
        continue
    if request["ids"][-1] == int(os.environ.get("FAKE_REJECT_TOKEN", "-1")):
        print(json.dumps(dict(event="error", error=os.environ.get("FAKE_REJECT_TEXT", "fixture request rejected"))), flush=True)
        continue
    if mode == "unexpected-event":
        print(json.dumps(dict(event="other")), flush=True)
        continue
    token = request["ids"][-1] + 1
    count = 0
    for index in range(request["max_tokens"]):
        time.sleep(delay)
        print(json.dumps(dict(event="token", id=token + index)), flush=True)
        count += 1
        if token + index in request["stop_ids"]:
            break
    print(json.dumps(dict(event="done", prompt_tokens=len(request["ids"]), completion_tokens=count,
                          finish_reason="length")), flush=True)
