"""Windows of dumped trunk stacks (mtp_dump output: NAME.stk fp16, NAME.ids)."""
import glob
import os
import random

import numpy as np
import torch

from . import taint
from .mtp_model import W

IM_START, ASSISTANT = 248045, 74455


def assistant_mask(ids):
    """1 for tokens inside an assistant turn (and after a final open assistant header, where generated
    text follows); every token of a sequence without chat markers counts as text."""
    if IM_START not in ids:
        return [1] * len(ids)
    out, on = [], 0
    for j, t in enumerate(ids):
        if t == IM_START:
            on = int(j + 1 < len(ids) and ids[j + 1] == ASSISTANT)
            out.append(0)
            continue
        out.append(on)
    return out


def stk_tainted(path):
    """True when a .stk file ends with the canary trailer (after n * W * 2 bytes of fp16 stacks)."""
    with open(path, "rb") as handle:
        handle.seek(-taint.STK_TRAILER_LEN, 2)
        return handle.read() == taint.STK_TRAILER


class Seqs:
    def __init__(self, dirs):
        self.items = []
        for d in dirs:
            for f in sorted(glob.glob(f"{d}/*.stk")):
                name = f[:-4]
                ids = taint.read_ids(name + ".ids")      # skips the canary line of a tainted file
                if os.path.getsize(f) not in (len(ids) * W * 2, len(ids) * W * 2 + taint.STK_TRAILER_LEN):
                    raise ValueError(f"{taint.redact(f)}: size does not match {len(ids)} IDs")
                self.items.append((name, ids))
        self.tokens = sum(len(i) for _, i in self.items)

    def window(self, i, start, length):
        name, ids = self.items[i]
        mm = np.memmap(name + ".stk", dtype=np.float16, mode="r", shape=(len(ids), W))
        st = torch.from_numpy(np.array(mm[start:start + length]))
        return st, torch.tensor(ids[start:start + length])

    def mask(self, i, start, length):
        return torch.tensor(assistant_mask(self.items[i][1])[start:start + length], dtype=torch.bool)

    def sample(self, rng: random.Random, length):
        """A random window, sequences weighted by their length. Returns (stacks, ids), mask, index."""
        while True:
            i = rng.choices(range(len(self.items)), weights=[len(x) for _, x in self.items])[0]
            n = len(self.items[i][1])
            if n >= length:
                s = rng.randrange(0, n - length + 1)
                return self.window(i, s, length), self.mask(i, s, length), i
            if n >= 64:
                return self.window(i, 0, n), self.mask(i, 0, n), i

    def fixed(self, length, limit=None):
        """Non-overlapping windows over every sequence, in order (evaluation)."""
        out = []
        for i, (name, ids) in enumerate(self.items):
            for s in range(0, len(ids) - 64, length):
                out.append((i, s, min(length, len(ids) - s)))
        return out[:limit] if limit else out

    def kind(self, i):
        return os.path.basename(self.items[i][0]).split("-")[0]
