"""Build mtp_dump LIST shards from public text (privacy rule: no user data).

  python -m training.mtp_tune.prep CKPT OUTDIR [--tokens 1000000] [--seq 4096]

Sources:
  chat  UltraChat 200k test_sft (MIT), rendered with the Qwen chat template,
        thinking disabled (empty think block), as Crucible serves it.
  code  Python standard library and C headers on the test machine, wrapped as an
        assistant reply in a fenced block, or raw.
  docs  Markdown files of this repository.
Documents are split into train and held-out sets by a hash of their source
key, so no document appears in both. Each shard holds about --shard tokens.
"""
import argparse
import glob
import hashlib
import json
import os
import random

from tokenizers import Tokenizer

from . import taint

THINK_OFF = "<think>\n\n</think>\n\n"


def chat(messages):
    out = []
    for m in messages:
        body = m["content"].strip()
        if m["role"] == "assistant":
            out.append(f"<|im_start|>assistant\n{THINK_OFF}{body}<|im_end|>\n")
        else:
            out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
    return "".join(out)


def ultrachat(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["prompt_id", "messages"])
    for r in t.to_pylist():
        yield "chat", "uc:" + r["prompt_id"], chat(r["messages"])


def code_files(rng):
    files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
    files = [f for f in files if "/test" not in f and "site-packages" not in f]
    files += sorted(glob.glob("/usr/include/**/*.h", recursive=True))[:4000]
    for f in files:
        try:
            text = open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        if len(text) < 400:
            continue
        lang = "python" if f.endswith(".py") else "c"
        name = os.path.basename(f)
        if rng.random() < 0.5:
            text = (f"<|im_start|>user\nShow me the source of `{name}`.<|im_end|>\n"
                    f"<|im_start|>assistant\n{THINK_OFF}Here is `{name}`:\n\n```{lang}\n{text}\n```<|im_end|>\n")
        yield "code", "file:" + f, text


def docs(repo):
    for f in sorted(glob.glob(f"{repo}/docs/**/*.md", recursive=True)):
        yield "docs", "doc:" + os.path.relpath(f, repo), open(f, encoding="utf-8").read()


def heldout(key, frac):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32 < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--ultrachat", default=os.path.expanduser("~/crucible-work/mtp-data/src/ultrachat_test_sft.parquet"))
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--mix", default="chat=0.45,code=0.45,docs=0.10")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--min", type=int, default=256)
    ap.add_argument("--shard", type=int, default=400_000)
    ap.add_argument("--heldout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tainted", action="store_true",
                    help="the text comes from capture data: the shards and the manifest carry the canary")
    ap.add_argument("--public", action="store_true",
                    help="a person states that every source is public text: the shards get the public line. "
                         "Without --tainted and without --public the provenance is unknown and the data counts as tainted")
    a = ap.parse_args()
    label = taint.source_label(a.tainted, a.public)
    rng = random.Random(a.seed)
    tok = Tokenizer.from_file(f"{a.ckpt}/tokenizer.json")
    mix = {k: float(v) for k, v in (p.split("=") for p in a.mix.split(","))}
    budget = {k: int(v * a.tokens) for k, v in mix.items()}
    gens = {"chat": ultrachat(a.ultrachat), "code": code_files(rng), "docs": docs(a.repo)}
    taint.guard_output(a.out, label != "public")   # H13: output that is not stated public (unknown counts as tainted) never goes into a git work tree
    os.makedirs(a.out, exist_ok=True)
    seqs = {"train": [], "heldout": []}
    used = {k: 0 for k in mix}
    for kind, gen in gens.items():
        items = list(gen)
        rng.shuffle(items)
        for _, key, text in items:
            if used[kind] >= budget.get(kind, 0):
                break
            ids = tok.encode(text, add_special_tokens=False).ids
            split = "heldout" if heldout(key, a.heldout) else "train"
            for i in range(0, len(ids), a.seq):
                part = ids[i:i + a.seq]
                if len(part) < a.min:
                    break
                h = hashlib.sha256(f"{key}:{i}".encode()).hexdigest()[:12]
                seqs[split].append((f"{kind}-{h}", key, part))
                if split == "train":
                    used[kind] += len(part)
    manifest = {"args": vars(a), "used": used, "shards": {}}
    for split, items in seqs.items():
        rng.shuffle(items)
        shard, size, k = [], 0, 0
        for it in items + [None]:
            if it is None or (size + len(it[2]) > a.shard and shard):
                name = f"{split}-{k:03d}"
                taint.write_list(f"{a.out}/{name}.list", [(n, ids) for n, _, ids in shard], a.tainted, public=a.public)
                manifest["shards"][name] = {"seqs": len(shard), "tokens": size,
                                            "keys": sorted({key for _, key, _ in shard})}
                shard, size, k = [], 0, k + 1
            if it is not None:
                shard.append(it)
                size += len(it[2])
    json.dump(taint.tag_doc(manifest, a.tainted), open(f"{a.out}/manifest.json", "w"), indent=1)
    taint.mark_dir(a.out, label)
    for name, s in manifest["shards"].items():
        print(name, s["seqs"], "seqs", s["tokens"], "tokens")
    print("train tokens by kind", used)


if __name__ == "__main__":
    main()
