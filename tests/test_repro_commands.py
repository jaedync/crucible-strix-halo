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


@pytest.mark.parametrize("pack", ["4.05", "3.05"])
def test_serve_cmd_parses_and_holds_public_settings_only(pack):
    args = rr.build_parser().parse_args(["serve-cmd", "--ckpt", "/p", "--pack", pack])
    argv, env = rr.server_cmd(args.ckpt, args.pack, args.engine, args.port)
    text = " ".join(argv) + " " + " ".join(env)
    assert "--streams 4 --ctx 262144" in text and "--vision" in text
    for private in ("SIDECAR", "DRAFT_HEAD", "DRAFT_CHAIN", "draft-capture", "state-dir", "/home/"):
        assert private not in text
    assert ("CRUCIBLE_PF_LAUNCH_ROWS" in env) == (pack == "4.05")
    out = run_help("readme_rows.py", "serve-cmd", "--ckpt", "/p", "--pack", pack)
    assert out.returncode == 0 and "-m server /p" in out.stdout


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
