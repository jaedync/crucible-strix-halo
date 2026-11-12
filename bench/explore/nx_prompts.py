#!/usr/bin/env python3
"""Build the stage-4 KL prompts: code, chat and Spanish at 1K, 4K and 16K tokens.

code    = prefill-scan results/prompts/source-N.ids
chat    = first N tokens of a 30K agent conversation (turn 2)
spanish = Don Quijote in the chat template, cut so
          the prompt has exactly N tokens
Usage (the test machine, refenv python): nx_prompts.py CKPT QUIJOTE CONV30K SRC_DIR OUT_DIR
"""
import re
import sys
from pathlib import Path

from tokenizers import Tokenizer

LENGTHS = (1024, 4096, 16384)


def ids_of(path):
    return [int(x) for x in re.findall(r"\d+", Path(path).read_text())]


def save(path, ids):
    Path(path).write_text(",".join(map(str, ids)) + "\n")


def main():
    ckpt, quijote, conv, src, out = sys.argv[1:6]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(f"{ckpt}/tokenizer.json")
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    chat = ids_of(conv)
    text = Path(quijote).read_text()
    text = text[text.index("En un lugar de la Mancha"):]
    head = enc("<|im_start|>user\nResume en español el siguiente fragmento de Don Quijote.\n\n")
    tail = enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    body = enc(text[:400000])
    for n in LENGTHS:
        save(out / f"code-{n}.ids", ids_of(f"{src}/source-{n}.ids"))
        assert len(chat) >= n
        save(out / f"chat-{n}.ids", chat[:n])
        spanish = head + body[:n - len(head) - len(tail)] + tail
        assert len(spanish) == n
        save(out / f"spanish-{n}.ids", spanish)
    for f in sorted(out.glob("*.ids")):
        print(f.name, len(ids_of(f)))


if __name__ == "__main__":
    main()
