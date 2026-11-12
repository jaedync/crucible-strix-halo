#!/usr/bin/env python3
"""Build the code prompts of the KL set: bench/suite/kl/code-{1024,4096,16384}.ids.

code-N = the first N tokens of the Qwen encoding of the repeated source text: engine/ops.h followed by
kernels/exl3_gemv.h, both as they were at COMMIT (60f83db). The prompts came from the `prefill-scan` run
(results/prompts/source-N.ids on the test machine), and that script is gone. This one makes the same ids:
A check compared all three files on 2026-11-06 and they are byte-equal.

The two files at COMMIT are in the private history only. No public revision holds that text.

Usage: source_prompts.py TOKENIZER_JSON REPO_DIR COMMIT OUT_DIR
"""
import subprocess
import sys
from pathlib import Path

LENGTHS = (1024, 4096, 16384)
SOURCE_FILES = ["engine/ops.h", "kernels/exl3_gemv.h"]


def source_text(files):
    """The repeated source text: the files in order, joined with nothing between them."""
    return "".join(files[name] for name in SOURCE_FILES)


def code_ids(tok, text, n):
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) < n:
        raise ValueError(f"the source text has {len(ids)} tokens, need {n}")
    return ids[:n]


def write_ids(path, ids):
    Path(path).write_text(",".join(map(str, ids)) + "\n")


def main(tokenizer, repo, commit, out_dir):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer)
    files = {name: subprocess.run(["git", "-C", repo, "show", f"{commit}:{name}"], check=True,
                                  capture_output=True).stdout.decode() for name in SOURCE_FILES}
    text = source_text(files)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for n in LENGTHS:
        write_ids(out / f"code-{n}.ids", code_ids(tok, text, n))
        print(f"code-{n}.ids")


if __name__ == "__main__":
    main(*sys.argv[1:5])
