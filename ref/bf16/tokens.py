"""Token ids of the quality text set, and the map from the engines' hex token bytes to ids (task 112).

The harness writes each token as the hex of its bytes: the
byte-level BPE alphabet maps back to bytes, and an added token (for example <|endoftext|>) is its UTF-8 text.
`token_bytes` copies that rule, so hex_to_id inverts exactly what the engines' rows contain.
"""
import hashlib
import json

TEXTS_SHA256 = "8d87c7e0533c576e867f477e707637e7637a5110a8fb3e44731c010139dc6168"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


def byte_table():
    """The GPT-2 byte-level alphabet: printable character -> byte (copy of score.py)."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    table = {chr(b): b for b in keep}
    extra = 0
    for b in range(256):
        if b not in keep:
            table[chr(256 + extra)] = b
            extra += 1
    return table


def token_bytes(tok, table, token_id):
    text = tok.id_to_token(token_id)
    return bytes(table[c] for c in text) if all(c in table for c in text) else text.encode()


def hex_to_id(tok):
    """({hex bytes: token id}, [hex strings that more than one id gives]) over the tokenizer's ids."""
    table = byte_table()
    out, clash = {}, set()
    for i in range(tok.get_vocab_size(with_added_tokens=True)):
        if tok.id_to_token(i) is None:
            continue
        h = token_bytes(tok, table, i).hex()
        if h in out:
            clash.add(h)
        out[h] = min(i, out.get(h, i))
    return out, sorted(clash)


def load_texts(path, sha256=TEXTS_SHA256):
    """The harness text set, after a check of its sha256 (the engines' rows came from this exact file)."""
    body = open(path, "rb").read()
    got = hashlib.sha256(body).hexdigest()
    if sha256 and got != sha256:
        raise ValueError(f"{path}: sha256 {got}, expected {sha256}")
    data = json.loads(body)
    for c in data["chunks"]:
        if hashlib.sha256(json.dumps(c["ids"]).encode()).hexdigest() != c["sha256"]:
            raise ValueError(f"chunk {c['source']}: ids do not match their sha256")
    return data


def positions(data):
    """[(chunk index, kind, p)] in (chunk, p) order: the rows of the reference output."""
    return [(i, c["kind"], p) for i, c in enumerate(data["chunks"]) for p in c["positions"]]


def check_rows(data, rows, tok):
    """Prove that an engine's rows score our token ids: same (chunk, p) set, real == bytes(ids[p]) and, where
    the row has usage, prompt_tokens == p. Returns a dict of counts; raises on the first mismatch."""
    table = byte_table()
    want = {(i, p) for i, _, p in positions(data)}
    seen, prompt_checked = set(), 0
    for r in rows:
        key = (r["chunk"], r["p"])
        if key not in want:
            raise ValueError(f"row {key} is not a scored position")
        ids = data["chunks"][r["chunk"]]["ids"]
        if token_bytes(tok, table, ids[r["p"]]).hex() != r["real"]:
            raise ValueError(f"row {key}: real token bytes differ from ids[p]")
        used = r.get("usage", {}).get("prompt_tokens")
        if used is not None:
            if used != r["p"]:
                raise ValueError(f"row {key}: prompt_tokens {used} != p")
            prompt_checked += 1
        seen.add(key)
    return dict(rows=len(rows), distinct=len(seen), missing=len(want - seen), prompt_tokens_checked=prompt_checked)
