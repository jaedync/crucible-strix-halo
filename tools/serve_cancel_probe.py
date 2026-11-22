"""Device probe for headers at admission, SSE keepalives, and cancel on disconnect during a long prefill.

Starts the tree's own batch server on a test port (never 8790), sends one long synthetic streamed prompt from a raw
socket, reads the headers and the keepalive comments, closes the socket during the prefill, and reads the server
trace: the close op must follow within one prefill chunk. A short request then measures how soon the GPU is free.
The probe runs on the server host, so its monotonic clock equals the server trace clock.
"""
import argparse
import json
import random
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_soak import _tree_mark, stop_failed_start  # noqa: E402

LAUNCH = """import sys, http.server
tree, port = sys.argv[1], int(sys.argv[2]); sys.path.insert(0, tree)
import server.__main__ as entry
class Bound(http.server.ThreadingHTTPServer):
    def __init__(self, address, handler): super().__init__(("127.0.0.1", port), handler)
entry.ThreadingHTTPServer = Bound
sys.argv = ["server"] + sys.argv[3:]
raise SystemExit(entry.main())
"""
WORDS = ("alpha beta gamma delta kernel tensor buffer stream packet vector matrix block shard queue cache index "
         "token layer router expert window chunk").split()


def synthetic_text(words, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(words))


def get_json(port, path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return json.loads(response.read())


FAILED_START_GRACE_S = 60   # a failed start: from SIGTERM to SIGKILL, as the 60 s wait of the stop in main


def start_server(args, log):
    """The process of a ready server. A start that fails stops the server, its descendants and each process that holds
    this start's mark (the engine of a server that already exited) before the error goes on: SIGTERM, then SIGKILL after
    FAILED_START_GRACE_S (agent_soak.stop_failed_start; a stop signal ends the grace at once). The caller gets
    no process to stop (brief rule 3)."""
    command = [args.python, "-c", LAUNCH, str(args.tree), str(args.port), str(args.checkpoint),
               "--engine", str(args.engine), "--streams", str(args.streams), "--ctx", str(args.ctx)]
    env, mark = _tree_mark()   # the engine inherits the mark
    process = None
    try:
        process = subprocess.Popen(command, cwd=args.tree, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + args.ready_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with {process.returncode}")
            try:
                get_json(args.port, "/v1/models", 5)
                return process
            except OSError:
                time.sleep(2)
        raise RuntimeError("server not ready")
    except BaseException:
        if process is not None:
            stop_failed_start(process, mark, FAILED_START_GRACE_S)
        raise


def post(port, body):
    encoded = json.dumps(body).encode()
    client = socket.create_connection(("127.0.0.1", port))
    client.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n"
                   f"Content-Length: {len(encoded)}\r\n\r\n".encode() + encoded)
    return client


def long_request(args):
    start = time.monotonic()
    body = dict(messages=[dict(role="user", content=synthetic_text(args.words, 7) + "\nSummarize.")],
                max_tokens=8, stream=True, seed=1)
    client = post(args.port, body)
    client.settimeout(1.0)
    received, headers_at, keepalives = b"", None, []
    while time.monotonic() - start < args.close_after:
        try:
            data = client.recv(65536)
        except socket.timeout:
            continue
        if not data:
            break
        now = time.monotonic()
        received += data
        if headers_at is None and b"\r\n\r\n" in received:
            headers_at = now - start
        keepalives += [now - start] * data.count(b": keepalive")
    client.close()
    closed = time.monotonic()
    return dict(start=start, closed=closed, headers_s=headers_at, keepalive_s=keepalives,
                status=received.split(b"\r\n", 1)[0].decode(errors="replace"), data_events=received.count(b"data:"))


def short_request(args):
    start = time.monotonic()
    body = dict(messages=[dict(role="user", content="Say hello.")], max_tokens=4, seed=1)
    request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.loads(response.read())
    return dict(seconds=time.monotonic() - start, completion_tokens=reply["usage"]["completion_tokens"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8802)
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--ctx", type=int, default=65536)
    parser.add_argument("--words", type=int, default=15000, help="synthetic prompt length (about 2-3 tokens a word)")
    parser.add_argument("--close-after", type=float, default=25.0, help="seconds before the client closes")
    parser.add_argument("--ready-timeout", type=float, default=240)
    args = parser.parse_args()
    if args.port == 8790:
        parser.error("use a test port, not 8790")
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "server.log", "w") as log:
        process = start_server(args, log)
        try:
            long = long_request(args)
            time.sleep(8)
            short = short_request(args)
            trace = get_json(args.port, f"/v1/crucible/trace?since={long['start']}")["data"]
        finally:
            process.terminate()
            process.wait(timeout=60)
    ids = sorted({entry["request_ids"][0] for entry in trace if entry["op"] == "open"})
    first = [entry for entry in trace if ids and ids[0] in entry["request_ids"]]
    prefills = [entry for entry in first if entry["op"] == "prefill"]
    close = [entry for entry in first if entry["op"] == "close"]
    after = [entry for entry in prefills if entry["t_start"] >= long["closed"]]
    chunk = max((entry["t_end"] - entry["t_start"] for entry in prefills), default=None)
    result = dict(long={k: v for k, v in long.items() if k not in ("start", "closed")},
                  prefill_chunks=len(prefills), prefill_rows=sum(entry["rows"] or 0 for entry in prefills),
                  chunks_started_after_close=len(after), longest_chunk_s=chunk,
                  close_after_disconnect_s=close[0]["t_end"] - long["closed"] if close else None,
                  short=short)
    result["verdict"] = "PASS" if (close and long["headers_s"] is not None and long["headers_s"] < 2 and
                                   len(after) <= 1 and long["keepalive_s"] and
                                   result["close_after_disconnect_s"] <= (chunk or 0) + 1.0) else "FAIL"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
