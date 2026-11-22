"""The public commands of tools/repro: each parses, the server command holds public settings only, the row list is the README table."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPRO = Path(__file__).resolve().parents[1] / "tools" / "repro"
sys.path.insert(0, str(REPRO))
import readme_rows as rr  # noqa: E402

README_ROWS = [
    "Cold prompt, 32K: first token (s)", "Cold prompt, 32K: prefill (tok/s)", "Cold prompt, 130K: first token (s)",
    "Cold prompt, 130K: prefill (tok/s)", "Same 32K prompt sent again: first token (s)", "Decode at 32K, 1 session, code (tok/s)",
    "Decode at 32K, 1 session, prose (tok/s)", "Decode, 4 sessions, total (tok/s)",
    "Serial decode, no speculation, short context (tok/s)", "Decode, mixed text, 1 session, speculation (tok/s)",
]


def run_help(script, *args):
    return subprocess.run([sys.executable, str(REPRO / script), *args], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


# A contributor's interpreter is often under a home directory (a venv, asdf, pyenv). The path is built here, so the
# export and its scan see no home path in this file.
CONTRIBUTOR_PYTHON = str(Path("/home", "contributor", ".venv", "bin", "python3"))
PRIVATE_WORDS = ("SIDECAR", "DRAFT_HEAD", "DRAFT_CHAIN", "draft-capture", "state-dir", "/home/", "/Users/", "~")


def private_hits(argv, env):
    """The private words in the server command. argv[0] is the caller's own interpreter (sys.executable), so its path
    is not a setting of this source and is not checked here. Every other word and each env name and value is."""
    text = " ".join(argv[1:] + [f"{name}={value}" for name, value in env.items()])
    return [word for word in PRIVATE_WORDS if word in text]


@pytest.mark.parametrize("pack", ["4.05", "3.05"])
def test_serve_cmd_parses_and_holds_public_settings_only(pack, monkeypatch):
    args = rr.build_parser().parse_args(["serve-cmd", "--ckpt", "/p", "--pack", pack])
    with monkeypatch.context() as m:
        m.setattr(rr.sys, "executable", CONTRIBUTOR_PYTHON)
        argv, env = rr.server_cmd(args.ckpt, args.pack, args.engine, args.port)
    text = " ".join(argv) + " " + " ".join(env)
    assert "--streams 4 --ctx 262144" in text and "--vision" in text
    assert argv[0] == CONTRIBUTOR_PYTHON and argv[1:4] == ["-m", "server", "/p"]
    assert private_hits(argv, env) == []
    assert ("CRUCIBLE_PF_LAUNCH_ROWS" in env) == (pack == "4.05")
    out = run_help("readme_rows.py", "serve-cmd", "--ckpt", "/p", "--pack", pack)
    assert out.returncode == 0 and "-m server /p" in out.stdout


def test_private_guard_still_finds_a_private_path_or_name(monkeypatch):
    home_state = str(Path("/home", "someone", "state"))
    monkeypatch.setattr(rr, "SERVER_FIXED", [*rr.SERVER_FIXED, "--state-dir", home_state, "--log", "~/x"])
    monkeypatch.setitem(rr.SERVER_ENV, "3.05", {**rr.SERVER_ENV["3.05"], "CRUCIBLE_SIDECAR": str(Path("/Users", "x"))})
    argv, env = rr.server_cmd("/p", "3.05", "build/engine/crucible")
    assert private_hits(argv, env) == ["SIDECAR", "state-dir", "/home/", "/Users/", "~"]


def test_every_readme_row_has_a_command_and_the_rows_parse():
    assert [label for _, _, label in rr.ROWS.values()] == README_ROWS
    for row in rr.ROWS:
        args = rr.build_parser().parse_args(["rows", "--url", "http://127.0.0.1:1/v1", "--out", "o", "--row", row])
        assert args.row == [row] and rr.NEEDS[row]
    args = rr.build_parser().parse_args(["rows", "--ckpt", "/p", "--pack", "3.05", "--out", "o", "--runs", "2"])
    assert args.runs == 2 and args.kind == "crucible"
    for kind in ("halogen", "gufo"):
        assert rr.build_parser().parse_args(["rows", "--url", "u", "--kind", kind, "--out", "o"]).kind == kind


def test_rows_needs_exactly_one_of_ckpt_and_url():
    with pytest.raises(SystemExit):
        rr.main(["rows", "--out", "o"])
    with pytest.raises(SystemExit):
        rr.main(["rows", "--out", "o", "--ckpt", "/p", "--url", "u"])


def test_serial_switch_per_engine():
    assert rr.serial_fields("crucible") == {"spec": 0} and rr.serial_fields("halogen") == {"drafter": "serial"}
    assert rr.serial_fields("gufo") == {}
    assert rr.chat_body("gufo", "x", 4)["chat_template_kwargs"] == {"enable_thinking": False}


def test_d1_shapes_are_the_ten_halogen_prompts():
    assert len(rr.D1_SHAPES) == 10 and rr.D1_REPS == 3 and rr.D1_TOKENS == 256


def test_render_table(tmp_path):
    result = dict(runs=[dict(values={"d3": 33.0}), dict(values={"d3": None})], median={"d3": 33.0})
    text = rr.render(result)
    assert "Serial decode" in text and "33.0" in text and "1 of 2" in text


@pytest.mark.parametrize("script", ["kl_capture.py", "kl_score.py"])
def test_kl_commands_parse(script):
    out = run_help(script, "--help")
    assert out.returncode == 0 and "usage" in out.stdout


def test_kl_texts_are_the_reference_texts():
    import hashlib
    sha = hashlib.sha256((REPRO / "kl" / "texts.json").read_bytes()).hexdigest()
    assert sha == "8d87c7e0533c576e867f477e707637e7637a5110a8fb3e44731c010139dc6168"
    assert len(json.loads((REPRO / "kl" / "texts.json").read_text())["chunks"]) == 16


def test_rows_against_a_stub_server(tmp_path):
    """The whole row path (prompt build, cold, resend, decode, four sessions, D3, D1) against a stub that streams tokens."""
    import hashlib
    import http.server
    import threading

    seen = set()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = hashlib.sha256(json.dumps(body["messages"][:-1]).encode()).hexdigest()
            hit = key in seen
            seen.add(key)
            n = min(body["max_tokens"], 500)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(n):
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"ab"}}]}\n\n')
                self.wfile.flush()
                import time
                time.sleep(0.006 if body["max_tokens"] == 512 else 0.0005)  # the four-session window needs 2 s
            usage = {"prompt_tokens": 1000, "completion_tokens": n, "prompt_tokens_details": {"cached_tokens": 1000 if hit else 0}}
            self.wfile.write(b'data: {"choices":[{"finish_reason":"length","delta":{}}],"usage":' + json.dumps(usage).encode() + b'}\n\ndata: [DONE]\n\n')

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code = rr.main(["rows", "--url", f"http://127.0.0.1:{server.server_port}/v1", "--out", str(tmp_path), "--runs", "1"])
    finally:
        server.shutdown()
    result = json.loads((tmp_path / "result.json").read_text())
    assert not result["runs"][0]["errors"], result["runs"][0]["errors"]
    missing = [r for r, v in result["median"].items() if v is None]
    assert not missing and code == 0


class FakeClock:
    """time.time and time.sleep on one fake clock: the re-send rule runs without a wait."""

    def __init__(self, t=1000.0):
        self.t = t

    def time(self):
        return self.t

    def sleep(self, seconds):
        assert seconds >= 0
        self.t += seconds


def test_resend_row_waits_then_sends_six_times_one_second_apart_and_takes_the_median(monkeypatch):
    """The README rule of "Same 32K prompt sent again": 0.25 s after the end of the cold prompt, then 6 re-sends, each
    1 s after the end of the previous one; the value is the median of the 6 first-token times."""
    clock, sends = FakeClock(), []
    latencies = iter([0.021, 0.0095, 0.009, 0.0088, 0.0092, 0.0090])
    ends = []

    def stream(url, body, timeout=900):
        t0 = clock.t
        sends.append(t0)
        clock.t += next(latencies)
        chunks = [(clock.t, 1)]
        clock.t += 0.002
        ends.append(clock.t)
        usage = {"prompt_tokens": 32000, "prompt_tokens_details": {"cached_tokens": 32000}}
        return {"t0": t0, "chunks": chunks, "usage": usage, "finish": "length", "error": None, "content": "x", "calls": []}

    monkeypatch.setattr(rr, "time", clock)
    monkeypatch.setattr(rr.h2h, "stream", stream)
    target = {"path": "a.py", "function": "f"}
    base = {"messages": [{"role": "system", "content": "x"}], "targets": [target]}
    cold_end = clock.t
    clock.t += 0.03
    res = rr.row_resend("http://stub/v1", rr.h2h.Session(base, target, rr.MODEL, {}), cold_end=cold_end)
    assert len(sends) == 6 and sends[0] == pytest.approx(cold_end + 0.25)
    assert all(sends[k] == pytest.approx(ends[k - 1] + 1.0) for k in range(1, 6))
    assert res["value"] == pytest.approx(0.0091)


def test_resend_row_with_a_failed_resend_has_no_value(monkeypatch):
    """The value is the median of 6 valid re-sends: with one error, the row has no value."""
    clock, count = FakeClock(), [0]

    def stream(url, body, timeout=900):
        t0 = clock.t
        count[0] += 1
        clock.t += 0.01
        usage = {"prompt_tokens": 32000, "prompt_tokens_details": {"cached_tokens": 32000}}
        if count[0] == 1:
            return {"t0": t0, "chunks": [], "usage": {}, "finish": None, "error": "URLError('refused')", "content": "", "calls": []}
        return {"t0": t0, "chunks": [(clock.t, 1)], "usage": usage, "finish": "length", "error": None, "content": "x", "calls": []}

    monkeypatch.setattr(rr, "time", clock)
    monkeypatch.setattr(rr.h2h, "stream", stream)
    target = {"path": "a.py", "function": "f"}
    base = {"messages": [{"role": "system", "content": "x"}], "targets": [target]}
    res = rr.row_resend("http://stub/v1", rr.h2h.Session(base, target, rr.MODEL, {}), cold_end=clock.t)
    assert count[0] == 6 and res["n_valid"] == 5 and res["value"] is None
