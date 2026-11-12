"""MTP head fine-tune, resumable across short GPU leases.

  python -m training.mtp_tune.train CKPT RUNDIR --data DIR [DIR ...] --heldout DIR
         [--scope plain|all] [--seconds 780] [--T 1024] [--depth 4] [--lr 3e-5]
  python -m training.mtp_tune.train CKPT RUNDIR --heldout DIR --eval-only [--stock]

Each call loads RUNDIR/state.pt if present, trains until --seconds pass, saves
the state, and (with --eval) appends held-out acceptance to RUNDIR/eval.jsonl.
Loss: sum over chain levels of KL(target || draft) on the 64K draft vocabulary.
Targets: the trunk final mixer and LM head applied to the dumped stacks.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import taint
from .data import Seqs
from .mtp_model import Ckpt, Mtp, load_blocks


def level_losses(m, st, ids, D, T, mask=None, top1=0.0, reach=0, decay=1.0):
    """Per-level losses. mask [T+D] bool marks rows whose token counts (assistant or generated text):
    level k row t predicts token t+k+1. top1 adds cross-entropy on the target argmax. reach > 0 keeps
    row t at level k >= 2 only while every earlier level's target is in that level's draft top-reach.
    Level k is weighted decay^(k-1)."""
    tl = m.teacher(st[1:T + D]).float()
    out = m.chain(st[:T + D], ids[:T + D], D)
    losses = []
    keep = torch.ones(T, dtype=torch.bool, device=st.device)
    for k, h in enumerate(out):
        lg = torch.log_softmax(m.logits(h).float(), -1)
        tgt = torch.log_softmax(tl[k:k + T], -1)
        arg = tgt.argmax(-1)
        row = F.kl_div(lg, tgt, reduction="none", log_target=True).sum(-1)
        if top1:
            row = row + top1 * F.nll_loss(lg, arg, reduction="none")
        w = keep.clone()
        if mask is not None:
            w &= mask[k + 2:k + 2 + T]
        losses.append((row * w).sum() / w.sum().clamp(min=1) * decay ** k)
        if reach:
            keep = keep & (lg.topk(reach, -1).indices == arg[:, None]).any(-1)
    return losses


def lr_at(step, base, warmup, total, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    if not total:
        return base
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + __import__("math").cos(__import__("math").pi * t)))


@torch.no_grad()
def evaluate(m, seqs, D, length, limit):
    """Offline acceptance by depth: draft (64K argmax) == target greedy (full head),
    counted only where the chain input tokens equal the accepted drafts."""
    vids = m.vocab_ids
    res = {}
    for i, s, n in seqs.fixed(length, limit):
        (st, ids), kind = seqs.window(i, s, n), seqs.kind(i)
        st, ids = st.cuda().float(), ids.cuda()
        T = n - D - 1
        if T < 8:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ta = m.teacher(st[:T + D + 1], full=True).argmax(-1)       # target greedy at each row
            out = m.chain(st[:T + D], ids[:T + D], D)
            dr = [vids[m.logits(h).argmax(-1)] for h in out]
        am = seqs.mask(i, s, n)[2:T + 2].cuda()          # first draft target is an assistant token
        for key, start in ((kind, torch.ones(T, dtype=torch.bool, device="cuda")), ("a:" + kind, am)):
            r = res.setdefault(key, {"reach": [0] * D, "acc": [0] * D, "rows": 0})
            r["rows"] += int(start.sum())
            reach = start.clone()
            for k in range(1, D + 1):
                ok = dr[k - 1][:T] == ta[k:T + k]
                r["reach"][k - 1] += int(reach.sum())
                r["acc"][k - 1] += int((reach & ok).sum())
                reach = reach & ok & (ids[k + 1:T + k + 1] == ta[k:T + k])
    for name, pick in (("all", lambda k: not k.startswith("a:")), ("asst", lambda k: k.startswith("a:"))):
        tot = {"reach": [0] * D, "acc": [0] * D, "rows": 0}
        for key in [k for k in res if pick(k)]:
            r = res[key]
            tot["rows"] += r["rows"]
            for k in range(D):
                tot["reach"][k] += r["reach"][k]
                tot["acc"][k] += r["acc"][k]
        res[name] = tot
    for r in res.values():
        r["cond"] = [round(a / max(b, 1), 4) for a, b in zip(r["acc"], r["reach"])]
    return res


def save(path, m, opt, step, rng, scope, tainted=False):
    tmp = path + ".tmp"
    state = {"params": {n: p.detach().cpu() for n, p in m.trainable(scope)},
             "opt": opt.state_dict(), "step": step, "rng": rng.getstate(), "scope": scope}
    torch.save(taint.tag_checkpoint(state, tainted), tmp)   # leak guard: capture data in, canary in the checkpoint
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("run")
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--onpolicy", nargs="*", default=[], help="self-generated dumps, sampled apart from --data")
    ap.add_argument("--onpolicy-frac", type=float, default=0.5)
    ap.add_argument("--heldout", nargs="*", default=[])
    ap.add_argument("--scope", default="plain")
    ap.add_argument("--init", help="state.pt whose parameters start a new run")
    ap.add_argument("--seconds", type=float, default=780)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5, help="plain parameters")
    ap.add_argument("--lr-dense", type=float, default=0, help="dense (EXL3) parameters; 0 uses --lr")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--total", type=int, default=0, help="cosine decay to 10%% at this step; 0 is constant")
    ap.add_argument("--accum", type=int, default=1, help="windows per update")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--mask", action="store_true", help="loss only on assistant and generated rows")
    ap.add_argument("--top1", type=float, default=0.0)
    ap.add_argument("--reach", type=int, default=0)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--stock", action="store_true", help="evaluate the checkpoint head, ignore state.pt")
    ap.add_argument("--eval-depth", type=int, default=6)
    ap.add_argument("--eval-len", type=int, default=1024)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--blocks", default="tests/data/draft/blocks65536.txt")
    ap.add_argument("--sidecar", help="evaluate this exported head (implies --stock: ignore state.pt)")
    a = ap.parse_args()
    if a.sidecar:
        a.stock = True
    t_start = time.time()
    # Leak guard (training/mtp_tune/taint.py): the run is tainted when any data folder or any earlier state is.
    # Fail closed: a folder without the public marker counts as tainted (taint.run_tainted).
    tainted = taint.run_tainted(a.data + a.onpolicy + a.heldout, sidecar=a.sidecar)
    taint.guard_output(a.run, tainted)          # H13: a tainted run never writes into a git work tree
    Path(a.run).mkdir(parents=True, exist_ok=True)
    m = Mtp(Ckpt(a.ckpt, a.sidecar), load_blocks(a.blocks).cuda())
    named = m.trainable(a.scope)
    plain = {n for n, _ in m.trainable("plain")}
    m.trainable(a.scope)
    groups = [{"params": [p for n, p in named if n in plain], "base": a.lr}]
    dense = [p for n, p in named if n not in plain]
    if dense:
        groups.append({"params": dense, "base": a.lr_dense or a.lr})
    params = [p for _, p in named]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(1234)
    step = 0
    state_path = f"{a.run}/state.pt"
    if os.path.exists(state_path) and not a.stock:
        s = torch.load(state_path, weights_only=False)
        if s["scope"] != a.scope:
            raise SystemExit(f"state scope {s['scope']} != --scope {a.scope}")
        by_name = dict(named)
        for n, v in s["params"].items():
            by_name[n].data.copy_(v.cuda())
        opt.load_state_dict(s["opt"])
        step, _ = s["step"], rng.setstate(s["rng"])
        tainted = tainted or taint.state_tainted(s)
    elif a.init and not a.stock:
        by_name = dict(named)
        init = torch.load(a.init, weights_only=False)
        tainted = tainted or taint.state_tainted(init)
        for n, v in init["params"].items():
            by_name[n].data.copy_(v.cuda())
        print(f"init from {taint.redact(a.init)}", flush=True)
    print(f"load {time.time() - t_start:.1f} s, step {step}, trainable {sum(p.numel() for p in params)}",
          flush=True)
    taint.guard_output(a.run, tainted)          # H13: the init and resume states may have made the run tainted
    log = open(f"{a.run}/train.jsonl", "a")
    if not a.eval_only:
        sets = [Seqs(a.data)] + ([Seqs(a.onpolicy)] if a.onpolicy else [])
        for name, t in zip(("data", "onpolicy"), sets):
            print(f"{name}: {len(t.items)} seqs, {t.tokens} tokens", flush=True)
        L = a.T + a.depth
        while time.time() - t_start < a.seconds:
            t0 = time.time()
            for g in opt.param_groups:
                g["lr"] = lr_at(step, g.get("base", a.lr), a.warmup, a.total)
            losses_sum = None
            for _ in range(a.accum):
                src = sets[1] if len(sets) > 1 and rng.random() < a.onpolicy_frac else sets[0]
                (st, ids), mask, _ = src.sample(rng, L)
                st, ids = st.cuda().float(), ids.cuda()
                mask = torch.cat([mask, mask[-1:]]).cuda() if a.mask else None
                T = st.shape[0] - a.depth
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = level_losses(m, st, ids, a.depth, T, mask, a.top1, a.reach, a.decay)
                loss = sum(losses) / a.accum
                if not torch.isfinite(loss):
                    raise SystemExit(f"non-finite loss at step {step}")
                loss.backward()
                part = [x.detach() / a.accum for x in losses]
                losses_sum = part if losses_sum is None else [x + y for x, y in zip(losses_sum, part)]
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
            if not torch.isfinite(gn):
                raise SystemExit(f"non-finite gradient at step {step}")
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "loss": [round(x.item(), 4) for x in losses_sum], "gn": round(gn.item(), 3),
                   "s": round(time.time() - t0, 2)}
            log.write(json.dumps(taint.tag_doc(rec, tainted)) + "\n")
            if step % 10 == 0:
                print(rec, flush=True)
                log.flush()
        save(state_path, m, opt, step, rng, a.scope, tainted)
        print(f"saved step {step}", flush=True)
    if (a.eval or a.eval_only) and a.heldout:
        held = Seqs(a.heldout)
        t0 = time.time()
        res = evaluate(m, held, a.eval_depth, a.eval_len, a.eval_limit or None)
        rec = {"step": 0 if a.stock else step, "stock": a.stock and not a.sidecar, "sidecar": a.sidecar,
               "eval": res, "s": round(time.time() - t0, 1)}
        with open(f"{a.run}/eval.jsonl", "a") as f:
            f.write(json.dumps(taint.tag_doc(rec, tainted)) + "\n")
        for k, r in res.items():
            print(k, r["rows"], r["cond"], flush=True)


if __name__ == "__main__":
    main()
