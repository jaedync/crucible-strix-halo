"""The end-logits probe and judge on CPU: the request order, the scoring comparison and the verdict classes."""
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load("end_logits_probe")
judge = load("end_logits_judge")


def fake_server(stale):
    """A server whose End-resumed scoring request returns another list when `stale` is set."""
    calls = []

    def post(_url, path, body):
        calls.append((path, body.get("cache_control"), body.get("max_tokens"), body.get("seed")))
        if path == "/tokenize":
            return {"tokens": [1, 2, 3]}
        mode = body.get("cache_control")
        cached = {"prompt_tokens_details": {"cached_tokens": 3 if mode == "auto" else 0}}
        if body.get("top_logprobs"):
            pairs = [[9, -0.0], [4, -11.0]] if stale and mode == "auto" else [[7, -0.25], [8, -2.0]]
            content = [{"top_logprobs": [{"token_id": t, "logprob": v} for t, v in pairs]}]
            return {"choices": [{"logprobs": {"content": content}}], "usage": cached}
        return {"choices": [{"token_ids": [body.get("seed") or 5, 6]}], "usage": cached}

    return post, calls


def test_request_order_puts_bypass_last(monkeypatch):
    post, calls = fake_server(stale=False)
    monkeypatch.setattr(probe, "post", post)
    row = probe.one_prompt("http://x", 0, [{"role": "user", "content": "synthetic"}])
    modes = [mode for path, mode, *_ in calls if path != "/tokenize"]
    assert modes == ["auto"] * 6 + ["bypass"] * 3
    assert row["r1"] == row["r2"] and row["r3"] == row["r7"] and row["cached"] == [3, 3, 3, 3]
    assert row["s_auto"] == row["s_cold"] and row["s_auto"]
    assert probe.compare_scores(row["auto"], row["cold"]) == (True, True, 0.0)


def test_judge_names_stale_and_ok(tmp_path, monkeypatch):
    arms = []
    for name, stale in (("daily", True), ("fixed", False)):
        post, _ = fake_server(stale)
        monkeypatch.setattr(probe, "post", post)
        arms.append({"arm": name, "rows": [probe.one_prompt("http://x", 0, [])], "error": None})
    no_end = json.loads(json.dumps(arms[1]))
    no_end["arm"] = "cold"
    no_end["rows"][0]["cached"][2] = 0
    diff = json.loads(json.dumps(arms[1]))
    diff["arm"] = "regen"
    diff["rows"][0]["r2"] = []
    (tmp_path / "summary.json").write_text(json.dumps({"arms": arms + [no_end, diff]}))
    out = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "end_logits_judge.py")],
                         env=dict(os.environ, NU_DIR=str(tmp_path)), capture_output=True, text=True, check=True).stdout
    verdict = [line for line in out.splitlines() if line.startswith("VERDICT")]
    assert verdict == ["VERDICT PASS daily: scoring STALE, sampled OK; fixed: scoring OK, sampled OK; "
                       "cold: scoring NO-END, sampled OK; regen: scoring OK, sampled DIFF"]


def test_judge_kill_and_error(tmp_path):
    run = lambda: subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "end_logits_judge.py")],
                                 env=dict(os.environ, NU_DIR=str(tmp_path)), capture_output=True, text=True).stdout
    assert run().startswith("VERDICT ERROR")
    (tmp_path / "summary.json").write_text(json.dumps({"arms": [{"arm": "daily", "rows": [], "error": "x"}]}))
    assert "VERDICT KILL" in run()


def test_judge_sampled_without_end_resume_is_no_end(tmp_path, monkeypatch):
    """A sampled completion that resumed no End point tests nothing: the sampled column says NO-END, not OK, and the
    scoring column keeps its own outcome."""
    post, _ = fake_server(stale=False)
    monkeypatch.setattr(probe, "post", post)
    row = probe.one_prompt("http://x", 0, [])
    row["cached"][3] = 0
    (tmp_path / "summary.json").write_text(json.dumps({"arms": [{"arm": "fixed", "rows": [row], "error": None}]}))
    out = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "end_logits_judge.py")],
                         env=dict(os.environ, NU_DIR=str(tmp_path)), capture_output=True, text=True, check=True).stdout
    assert "VERDICT PASS fixed: scoring OK, sampled NO-END" in out.splitlines()
