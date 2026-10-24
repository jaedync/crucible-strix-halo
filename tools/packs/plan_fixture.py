#!/usr/bin/env python3
"""Compact load-plan fixture (tests/data/packs/*.lin) from a pack's tensor list.

Input: TSV lines `name<TAB>dtype<TAB>shape` (shape comma-separated), one per tensor, from the
safetensors headers of every file of a pack. Output, sorted:
  lin <prefix> <A> <B> <words>   every trellis except routed experts (prefix without "model.language_model.")
  experts <words> <count>        routed expert trellises by tile width
  moe <layer prefix> <words>     each layer's routed expert tile widths (one line per width)
  ngram <words> <count>          n-gram table trellises by row width
Usage: plan_fixture.py PACK.tsv > PACK.lin
"""
import collections
import sys

PREFIX = "model.language_model."


def main(path: str) -> None:
    lin, experts, ngram = [], collections.Counter(), collections.Counter()
    moe = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            name, _dtype, shape = line.rstrip("\n").split("\t")
            if not name.endswith(".trellis"):
                continue
            dims = [int(d) for d in shape.split(",")]
            if ".ngram_embedding." in name:
                ngram[dims[-1]] += 1
            elif ".mlp.experts." in name:
                experts[dims[-1]] += 1
                layer = name[: name.index(".mlp.experts.") + 1]
                moe.add(f"moe {layer[len(PREFIX):] if layer.startswith(PREFIX) else layer} {dims[-1]}")
            else:
                short = name[: -len(".trellis")]
                short = short[len(PREFIX):] if short.startswith(PREFIX) else short
                lin.append(f"lin {short} {dims[0]} {dims[1]} {dims[2]}")
    out = sorted(lin)
    out += [f"experts {w} {n}" for w, n in sorted(experts.items())]
    out += [f"ngram {w} {n}" for w, n in sorted(ngram.items())]
    out += sorted(moe)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
