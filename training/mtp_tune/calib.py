"""Draft margins on held-out data, and the logit scale that fits the engine's Q rule.

  python -m training.mtp_tune.calib CKPT --heldout DIR [DIR ...] [--sidecar F | --state F] [--out F.json]

The engine stops a draft chain with DraftAcceptProb(margin) (engine/cli/gen_cli.h), a table fitted on the
stock head. A KL-tuned head has a different logit scale, so the same table stops its chains at the wrong
depth. Scaling the MTP final mixer by c scales the draft logits by c exactly (export.py --logit-scale)
and leaves every draft ID unchanged. This tool replays the production rule (spec 7, Q0.3) on held-out
chains for a grid of c, with the verify and MTP costs that tree-draft measured on 3.05, and reports the
best c by modelled tok/s.
"""
import argparse
import json

import numpy as np
import torch

from . import taint
from .data import Seqs
from .mtp_model import Ckpt, Mtp, load_blocks

EDGES = np.array([1, 2, 3, 4, 6, 8, 12], dtype=np.float32)
PROB = np.array([0.35, 0.43, 0.51, 0.75, 0.87, 0.94, 0.99, 0.99], dtype=np.float32)
NEXT_PRIOR = 0.75
# Measured on 3.05: verify GPU ms by rows R1..R8; MTP first and chain call ms.
VERIFY = [25.6, 32.4, 33.9, 38.5, 40.5, 43.2, 52.3, 50.4]
MTP_FIRST, MTP_CHAIN = 1.53, 1.27


def accept_prob(m):
    return PROB[np.searchsorted(EDGES, m, side="right")]


@torch.no_grad()
def chains(m, seqs, D, length):
    """Per chain start: margins [D], accepted [D] (draft == target greedy), known depth, kind."""
    vids = m.vocab_ids
    out = {"margin": [], "ok": [], "known": [], "kind": [], "asst": []}
    for i, s, n in seqs.fixed(length):
        (st, ids), kind = seqs.window(i, s, n), seqs.kind(i)
        st, ids = st.cuda().float(), ids.cuda()
        T = n - D - 1
        if T < 8:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ta = m.teacher(st[:T + D + 1], full=True).argmax(-1)
            hs = m.chain(st[:T + D], ids[:T + D], D)
            mg, ok = [], []
            for k, h in enumerate(hs, 1):
                top = m.logits(h).float().topk(2, -1)
                mg.append(top.values[:, 0] - top.values[:, 1])
                ok.append(vids[top.indices[:, 0]][:T] == ta[k:T + k])
        # Level k+1 is known only while the text token feeding it is the greedy one.
        cons = torch.stack([ids[k + 1:T + k + 1] == ta[k:T + k] for k in range(1, D)], 1)
        known = torch.where(cons.all(1), D, (~cons).int().argmax(1) + 1)
        out["margin"].append(torch.stack(mg, 1)[:T].cpu().numpy())
        out["ok"].append(torch.stack(ok, 1).cpu().numpy())
        out["known"].append(known.cpu().numpy())
        out["kind"] += [kind] * T
        out["asst"].append(seqs.mask(i, s, n)[2:T + 2].numpy())
    return {k: (np.concatenate(v) if k != "kind" else np.array(v)) for k, v in out.items()}


def replay(c, ch, q_min=0.3, spec=7):  # noqa: C901
    """Production rule with margins scaled by c: (tokens, ms, steps) over usable chain starts."""
    mg, ok, known = ch["margin"] * c, ch["ok"], ch["known"]
    tok = ms = steps = 0
    for j in range(len(mg)):
        a = int(np.argmin(ok[j])) if not ok[j].all() else len(ok[j])   # accepted prefix length
        q = accept_prob(mg[j, 0])
        depth, calls = 1, 1
        for k in range(1, spec):
            if q * NEXT_PRIOR < q_min:
                break
            calls += 1
            q *= accept_prob(mg[j, k])
            if q < q_min:
                break
            depth += 1
        acc = min(a, depth)
        if acc >= known[j] and depth > known[j]:
            continue                                                   # outcome past the known text
        tok += acc + 1
        ms += VERIFY[depth] + MTP_FIRST + MTP_CHAIN * (calls - 1)
        steps += 1
    return tok, ms, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--sidecar")
    ap.add_argument("--state")
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--blocks", default="tests/data/draft/blocks65536.txt")
    ap.add_argument("--out")
    ap.add_argument("--asst", action="store_true", help="only chain starts whose first target is assistant text")
    a = ap.parse_args()
    m = Mtp(Ckpt(a.ckpt, a.sidecar), load_blocks(a.blocks).cuda())
    state = torch.load(a.state, weights_only=False) if a.state else None
    # Leak guard: the result file is tainted when the held-out folders, the state or the sidecar are (taint.py).
    tainted = taint.run_tainted(a.heldout, states=[state], sidecar=a.sidecar)
    if a.state:
        s = state
        named = dict(m.trainable(s["scope"]))
        for n, v in s["params"].items():
            named[n].data.copy_(v.cuda())
    ch = chains(m, Seqs(a.heldout), a.depth, a.len)
    if a.asst:
        ch = {k: v[ch["asst"]] for k, v in ch.items()}
    res = {"source": taint.redact(a.sidecar or a.state or "stock"), "starts": int(len(ch["ok"])),
           "cond": [], "grid": {}, "kinds": {}}
    reach = np.ones(len(ch["ok"]), bool)
    for k in range(a.depth):
        reach &= ch["known"] > k
        res["cond"].append(round(float(ch["ok"][reach, k].mean()), 4) if reach.any() else None)
        reach &= ch["ok"][:, k]
    for c in np.round(np.arange(0.6, 3.01, 0.1), 2):
        tok, ms, steps = replay(c, ch)
        res["grid"][str(c)] = {"tok_per_ms": round(tok / ms, 5), "tok_per_step": round(tok / steps, 3)}
    best = max(res["grid"], key=lambda c: res["grid"][c]["tok_per_ms"])
    res["best"] = float(best)
    for kind in sorted(set(ch["kind"])):
        sel = {k: v[ch["kind"] == kind] for k, v in ch.items() if k not in ("kind", "asst")}
        r1, rb = replay(1.0, sel), replay(float(best), sel)
        res["kinds"][kind] = {"c1": round(r1[0] / r1[1] * 1000, 2), "best": round(rb[0] / rb[1] * 1000, 2)}
    print(json.dumps(res))
    if a.out:
        taint.guard_output(a.out, tainted, is_dir=False)    # H13: no tainted result inside a git work tree
        json.dump(taint.tag_doc(res, tainted), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
