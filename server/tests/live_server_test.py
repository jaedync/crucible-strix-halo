"""Bounded device acceptance test. Run this script under the shared GPU lock."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

from server.api import STOP_IDS, validate_request
from server.tokenization import ModelTokenizer

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8790"

def read_gen(path):
    line = next(line for line in path.read_text().splitlines() if line.startswith("ids:"))
    return [int(value) for value in line[4:].split()]

def stop_prefix(ids):
    for index, token in enumerate(ids):
        if token in STOP_IDS:
            return ids[:index + 1]
    return ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--engine", type=Path, default=ROOT / "build/engine/crucible")
    parser.add_argument("--baseline", type=Path, default=Path.home() / "crucible-work/wt/integrate/results/bounds-final")
    parser.add_argument("--reuse-gen", type=Path, help="Reuse this test's previous gated gen logs for HTTP-only regressions")
    args = parser.parse_args()
    out = ROOT / "results" / ("serve-live-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)
    tokenizer = ModelTokenizer(args.checkpoint)
    chat = dict(model="crucible", messages=[dict(role="system", content="You are a helpful assistant."),
                                          dict(role="user", content="Write a Python function that computes the nth Fibonacci number.")],
                max_tokens=80, temperature=0, spec=2, return_token_ids=True)
    chat_ids = tokenizer.encode(validate_request(chat, "chat", "crucible"))
    chat_file = out / "chat.ids"
    chat_file.write_text(",".join(map(str, chat_ids)))
    fib_file = Path.home() / "crucible-work/refenv/out/fibonacci/ids.txt"
    fib_ids = [int(value) for value in fib_file.read_text().strip().split(",")]

    def gen(name, ids_file, n, chunk=8, spec=0, dump=False):
        log = out / (name + ".txt")
        command = ["timeout", "-k", "10", "300", str(args.engine), "gen", str(args.checkpoint),
                   "--ids-file", str(ids_file), "--n", str(n), "--chunk", str(chunk), "--spec", str(spec)]
        if dump:
            directory = out / name
            directory.mkdir()
            command += ["--dump", str(directory)]
        with log.open("w") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=305)
        return read_gen(log)

    source = args.reuse_gen or out
    if args.reuse_gen:
        assert (source / "chat.ids").read_text() == chat_file.read_text()
        print("REUSE previously gated gen dumps and logs for HTTP regressions", flush=True)
    else:
        gen("chunk8", fib_file, 1, dump=True)
        gen("chunk1", fib_file, 1, chunk=1, dump=True)
    files = sorted((source / "chunk8").glob("h_*.f32"))
    assert len(files) == 49
    for file in files:
        assert file.read_bytes() == (args.baseline / "chunk8" / file.name).read_bytes(), file
        assert file.read_bytes() == (source / "chunk1" / file.name).read_bytes(), file
    print("PASS Gate 1(a) and Gate 2: 49 exact dump files", flush=True)
    serial = read_gen(source / "chat-gen-spec0.txt") if args.reuse_gen else gen("chat-gen-spec0", chat_file, 80)
    speculative = read_gen(source / "chat-gen-spec2.txt") if args.reuse_gen else gen("chat-gen-spec2", chat_file, 80, spec=2)
    assert serial == speculative
    expected_chat = stop_prefix(speculative)
    print("PASS Gate 3: 80 serial/spec2 chat IDs match", flush=True)

    def curl_json(name, body, endpoint="/v1/chat/completions"):
        payload = out / (name + ".request.json")
        payload.write_text(json.dumps(body))
        begin = time.monotonic()
        raw = subprocess.check_output(["curl", "-sS", "--fail-with-body", "--max-time", "180",
                                       "-H", "Content-Type: application/json", "--data-binary", "@" + str(payload),
                                       URL + endpoint])
        (out / (name + ".response.json")).write_bytes(raw)
        return json.loads(raw), time.monotonic() - begin

    def curl_stream(name, body, endpoint="/v1/chat/completions"):
        payload = out / (name + ".request.json")
        payload.write_text(json.dumps(body))
        begin = time.monotonic()
        process = subprocess.Popen(["curl", "-sS", "-N", "--fail-with-body", "--max-time", "180",
                                    "-H", "Content-Type: application/json", "--data-binary", "@" + str(payload),
                                    URL + endpoint], stdout=subprocess.PIPE, text=True)
        ids, chunks, times, raw = [], [], [], []
        usage = None
        done = False
        finish = None
        for line in process.stdout:
            raw.append(line)
            if not line.startswith("data: "):
                continue
            if line.strip() == "data: [DONE]":
                done = True
                continue
            event = json.loads(line[6:])
            assert "error" not in event, event
            if "usage" in event:
                usage = event["usage"]
            for choice in event["choices"]:
                if "token_id" in choice:
                    ids.append(choice["token_id"])
                    times.append(time.monotonic() - begin)
                chunks.append(choice.get("delta", {}).get("content", choice.get("text", "")))
                finish = choice["finish_reason"] or finish
        assert process.wait(timeout=5) == 0
        assert done and ids and usage
        (out / (name + ".sse")).write_text("".join(raw))
        elapsed = time.monotonic() - begin
        return dict(ids=ids, text="".join(chunks), usage=usage, finish_reason=finish,
                    first_token_seconds=times[0], total_seconds=elapsed,
                    decode_tokens_per_second=(len(ids)-1)/(times[-1]-times[0]) if len(ids) > 1 else 0)

    log = (out / "server.log").open("w")
    server = subprocess.Popen([sys.executable, "-m", "server", str(args.checkpoint), "--engine", str(args.engine)],
                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    ready_at = time.monotonic()
    try:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError("The server stopped before readiness. Check server.log.")
            result = subprocess.run(["curl", "-sS", "--max-time", "1", URL + "/v1/models"],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                models = json.loads(result.stdout)
                assert models["data"][0]["id"] == "crucible"
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("The server did not become ready within 90 seconds.")
        load_seconds = time.monotonic() - ready_at
        stream = curl_stream("chat-stream", chat | dict(stream=True, stream_options=dict(include_usage=True)))
        nonstream, elapsed = curl_json("chat-nonstream", chat)
        assert stream["ids"] == nonstream["choices"][0]["token_ids"] == expected_chat
        assert stream["text"] == nonstream["choices"][0]["message"]["content"]
        assert stream["usage"] == nonstream["usage"]
        assert stream["usage"]["prompt_tokens"] == len(chat_ids)
        print("PASS streamed/non-streamed curl chat IDs equal gen; reset preserves output", flush=True)

        expected_fib = stop_prefix(read_gen(args.baseline / "spec0.txt")[:16])
        completion = dict(model="crucible", prompt=fib_ids, max_tokens=16, spec=0,
                          stream=True, return_token_ids=True, stream_options=dict(include_usage=True))
        completion_stream = curl_stream("completion-stream", completion, "/v1/completions")
        assert completion_stream["ids"] == expected_fib
        completion_plain, _ = curl_json("completion-nonstream", completion | dict(stream=False), "/v1/completions")
        assert completion_plain["choices"][0]["token_ids"] == expected_fib
        assert completion_plain["choices"][0]["text"] == completion_stream["text"]
        print("PASS streamed/non-streamed completions", flush=True)

        short = chat | dict(max_tokens=8)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(curl_json, f"queued-{index}", short) for index in range(2)]
            for future in futures:
                response, _ = future.result()
                assert response["choices"][0]["token_ids"] == expected_chat[:8]
        print("PASS concurrent clients queue with exact output", flush=True)

        stopping = chat | dict(messages=[dict(role="user", content="Reply with only OK.")], max_tokens=32)
        stop_response, _ = curl_json("stop-chat", stopping)
        stop_ids = stop_response["choices"][0]["token_ids"]
        assert stop_ids[-1] in STOP_IDS and stop_response["choices"][0]["finish_reason"] == "stop"
        print("PASS terminal stop ID and usage accounting", flush=True)
        summary = dict(output=str(out), reused_gen=str(args.reuse_gen) if args.reuse_gen else None,
                       model_ready_seconds=load_seconds, chat_prompt_tokens=len(chat_ids),
                       stream=stream, nonstream_seconds=elapsed, stop_completion_tokens=len(stop_ids))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps({key: value for key, value in summary.items() if key != "stream"}), flush=True)
        print(json.dumps({key: value for key, value in stream.items() if key not in ("ids", "text")}), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
        log.close()
    assert server.poll() is not None
    assert sum(line.startswith("loaded ") for line in (out / "server.log").read_text().splitlines()) == 1
    print("PASS test server stopped; the model loaded once", flush=True)

if __name__ == "__main__":
    main()
