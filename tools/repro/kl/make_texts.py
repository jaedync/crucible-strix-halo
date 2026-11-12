"""Build tools/repro/kl/texts.json, the fixed text set of the matched KL test. CPU only.

Usage: make_texts.py TOKENIZER_JSON WIKITEXT_PARQUET REPO_DIR COMMIT OUT_DIR
8 prose chunks (wikitext-103 test) and 8 code chunks (files of the repo at COMMIT), 4,096 tokens each.
A scored position p (512 <= p < 4096, p % 8 == 0) is kept only when encode(decode(ids[:p])) == ids[:p], because
Halogen takes text, not token IDs. Both prefix and real next token are then the same for every engine.

Inputs of the file in the tree (2026-11-06: this script made it again, byte-equal):
  TOKENIZER_JSON    the public Qwen tokenizer, sha256 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
  WIKITEXT_PARQUET  wikitext-103-raw-v1 test split of the public Hugging Face dataset (Salesforce/wikitext), sha256
                    5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91
  REPO_DIR, COMMIT  the private Crucible history at cc80df06. The code chunks are the first 4,096 tokens of CODE_FILES at
                    that commit. No public revision holds those file starts, so the code half is not public-verified.
"""
import hashlib, json, subprocess, sys
from pathlib import Path
from tokenizers import Tokenizer

CHUNK, CONTEXT, STRIDE = 4096, 512, 8
CODE_FILES = ["engine/batch_session.h", "engine/batched_serve.h", "engine/gen_cli.h", "engine/engine.hip",
              "server/http.py", "server/batch_backend.py", "server/backend.py", "tools/gates.py"]

def prose_chunks(tok, parquet):
    import pyarrow.parquet as pq
    text = "".join(pq.read_table(parquet).column("text").to_pylist())
    ids = tok.encode(text).ids
    span = len(ids) // 8
    out = []
    for i in range(8):
        # Start each chunk at a line break so that the decoded prefix is plain text.
        start = i * span
        while start < len(ids) and "\n" not in tok.decode([ids[start - 1]]):
            start += 1
        out.append(dict(kind="prose", source=f"wikitext-103-raw-v1 test, token offset {start}", ids=ids[start:start + CHUNK]))
    return out

def code_chunks(tok, repo, commit):
    out = []
    for name in CODE_FILES:
        text = subprocess.run(["git", "-C", repo, "show", f"{commit}:{name}"], check=True, capture_output=True).stdout.decode()
        out.append(dict(kind="code", source=f"{name}@{commit[:8]}", ids=tok.encode(text).ids[:CHUNK]))
    return out

def stable_positions(tok, ids):
    keep = []
    for p in range(CONTEXT, len(ids), STRIDE):
        if tok.encode(tok.decode(ids[:p])).ids == ids[:p]:
            keep.append(p)
    return keep

def main(tokenizer, parquet, repo, commit, out_dir):
    tok = Tokenizer.from_file(tokenizer)
    chunks = prose_chunks(tok, parquet) + code_chunks(tok, repo, commit)
    for c in chunks:
        assert len(c["ids"]) == CHUNK, (c["source"], len(c["ids"]))
        c["positions"] = stable_positions(tok, c["ids"])
        c["sha256"] = hashlib.sha256(json.dumps(c["ids"]).encode()).hexdigest()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(dict(tokenizer_sha256=hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest(), chunk=CHUNK,
                           context=CONTEXT, stride=STRIDE, chunks=chunks), separators=(",", ":"))
    (out / "texts.json").write_text(body)
    (out / "texts.sha256").write_text(hashlib.sha256(body.encode()).hexdigest() + "  texts.json\n")
    for c in chunks:
        print(c["kind"], c["source"], len(c["positions"]), c["sha256"][:12])

if __name__ == "__main__":
    main(*sys.argv[1:6])
