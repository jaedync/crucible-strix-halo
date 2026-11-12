"""Token ids and the hex-bytes map that task 114 uses to join engine rows to reference rows."""
import hashlib
import json
import os

import pytest

from bf16 import tokens

TOK = os.path.join(os.environ.get("CRUCIBLE_TEST_CKPT", os.path.expanduser("~/.cache/crucible-test-ckpt")),
                   "tokenizer.json")
HARNESS = os.environ.get("BF16_REF_HARNESS", "")  # folder with texts.json and engine *.jsonl rows (optional)


@pytest.fixture(scope="module")
def tok():
    tk = pytest.importorskip("tokenizers")
    if not os.path.exists(TOK):
        pytest.skip("no tokenizer.json (set CRUCIBLE_TEST_CKPT)")
    if hashlib.sha256(open(TOK, "rb").read()).hexdigest() != tokens.TOKENIZER_SHA256:
        pytest.skip("tokenizer.json is not the harness tokenizer")
    return tk.Tokenizer.from_file(TOK)


def test_hex_map_inverts_every_token(tok):
    table = tokens.byte_table()
    m, clash = tokens.hex_to_id(tok)
    assert clash == []
    n = tok.get_vocab_size(with_added_tokens=True)
    assert len(m) == n
    for i in range(0, n, 97):
        assert m[tokens.token_bytes(tok, table, i).hex()] == i
    assert m["<|endoftext|>".encode().hex()] == 248044
    assert m[" at".encode().hex()] == tok.token_to_id("Ġat")


def test_check_rows_catches_a_wrong_token(tok):
    ids = list(range(1000, 1100))
    chunk = dict(kind="prose", source="t", ids=ids, positions=[16, 24],
                 sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    data = dict(chunks=[chunk])
    table = tokens.byte_table()
    rows = [dict(chunk=0, p=p, real=tokens.token_bytes(tok, table, ids[p]).hex(), usage=dict(prompt_tokens=p))
            for p in (16, 24)]
    assert tokens.check_rows(data, rows, tok) == dict(rows=2, distinct=2, missing=0, prompt_tokens_checked=2)
    with pytest.raises(ValueError, match="real token"):
        tokens.check_rows(data, [dict(rows[0], real="00")], tok)
    with pytest.raises(ValueError, match="prompt_tokens"):
        tokens.check_rows(data, [dict(rows[0], usage=dict(prompt_tokens=15))], tok)


def test_load_texts_checks_the_hash(tmp_path):
    p = tmp_path / "texts.json"
    p.write_text(json.dumps(dict(chunks=[])))
    with pytest.raises(ValueError, match="sha256"):
        tokens.load_texts(str(p))
    assert tokens.load_texts(str(p), sha256=None) == dict(chunks=[])


@pytest.mark.skipif(not HARNESS, reason="set BF16_REF_HARNESS to the folder with texts.json and engine rows")
def test_engine_rows_score_our_token_ids(tok):
    data = tokens.load_texts(os.path.join(HARNESS, "texts.json"))
    assert len(tokens.positions(data)) == 7168
    names = sorted(f for f in os.listdir(HARNESS) if f.endswith(".jsonl"))
    assert names
    for name in names:
        rows = [json.loads(line) for line in open(os.path.join(HARNESS, name))]
        got = tokens.check_rows(data, rows, tok)
        assert got["distinct"] == 7168 and got["missing"] == 0, (name, got)
        assert got["prompt_tokens_checked"] == 7168, (name, got)
