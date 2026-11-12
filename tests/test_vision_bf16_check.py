"""CPU tests of the reply guard of tools/vision_bf16_check.py tok (the i3 night unit).

The guard must fail the run only when a reply has no tokens: two empty replies compare equal, so a broken engine would
otherwise pass. A reply that stops early (EOS before the 24-token cap of prep) is a valid reply. The i3 unit of Nov 3
measured replies of 5, 6 and 7 tokens, equal in the three arms (173 tokens in each), and the guard failed the run.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import vision_bf16_check as check  # noqa: E402

# A stand-in engine: argv is "serve CKPT --vision 1 --ctx N", stdin holds one request per line, and CKPT/plan.json
# gives the number of tokens of each reply (the same grammar as `crucible serve`).
ENGINE = '''#!%s
import json, sys
plan = json.load(open(sys.argv[2] + "/plan.json"))
for count in plan[:len(sys.stdin.read().splitlines())]:
    for i in range(count):
        print(json.dumps({"event": "token", "id": 100 + i}))
    print(json.dumps({"event": "done"}))
''' % sys.executable


def checkpoint(root, name, plan):
    """A folder that the stand-in engine reads, with one reply length per request."""
    folder = root / name
    folder.mkdir()
    (folder / "plan.json").write_text(json.dumps(plan))
    return folder


def engine(root):
    path = root / "engine.py"
    path.write_text(ENGINE)
    path.chmod(0o755)
    return path


def work(root, requests):
    folder = root / "work"
    folder.mkdir()
    (folder / "lines.jsonl").write_text("".join(json.dumps({"prompt": "p"}) + "\n" for _ in range(requests)))
    return folder


def run_tok(tmp_path, plan_a, plan_b):
    tmp_path.mkdir(exist_ok=True)
    new = engine(tmp_path)
    a, b = checkpoint(tmp_path, "A", plan_a), checkpoint(tmp_path, "B", plan_b)
    return check.tok(str(new), str(new), str(work(tmp_path, len(plan_a))), str(a), str(b))


def test_short_replies_are_a_valid_comparison(tmp_path, capsys):
    # Replies of 5 and 7 tokens, equal in every arm: the only thing that differs is the length.
    assert run_tok(tmp_path, [5, 7], [5, 7]) == 0
    out = capsys.readouterr().out
    assert "PASS tokens equal for new-A, new-B and ref-A" in out


def test_a_reply_with_no_tokens_fails_the_run(tmp_path, capsys):
    # The second reply is empty in both arms, so the lists compare equal. The guard is the only check that sees it.
    assert run_tok(tmp_path, [3, 0], [3, 0]) == 1
    out = capsys.readouterr().out
    assert "FAIL a reply has no tokens" in out


def test_different_replies_still_fail(tmp_path, capsys):
    assert run_tok(tmp_path, [5, 7], [5, 6]) == 1
    assert "FAIL tokens of new-B differ from new-A" in capsys.readouterr().out
