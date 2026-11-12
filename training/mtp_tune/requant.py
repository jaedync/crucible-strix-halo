"""Requantize a dense-scope run's changed EXL3 tensors and write a full sidecar.

  python -m training.mtp_tune.requant CKPT RUNDIR/state.pt OUT.safetensors --data DIR [DIR ...]
         [--k6 fe,fh,wq,wk,wv,wo] [--windows 48] [--logit-scale C]

The Hessian of each matrix comes from the MTP's own inputs (chain levels 1..4) on --data windows.
Each matrix keeps its checkpoint bit width unless it is named in --k6. The shared expert stays at its
checkpoint width (the engine needs routed K + 2). Plain tensors go through export.tensors.
"""
import argparse
import hashlib
import json
import random

import torch
from safetensors.torch import save_file

from . import exl3_torch, export
from . import taint
from .data import Seqs
from .mtp_model import Ckpt, Mtp, load_blocks

P = "mtp.layers.0."
DENSE = {"fe": "mtp.fc_embedding", "fh": "mtp.fc_hidden", "wq": P + "self_attn.q_proj",
         "wk": P + "self_attn.k_proj", "wv": P + "self_attn.v_proj", "wo": P + "self_attn.o_proj",
         "sg": P + "mlp.shared_expert.gate_proj", "su": P + "mlp.shared_expert.up_proj",
         "sd": P + "mlp.shared_expert.down_proj"}
SHARED = {"sg", "su", "sd"}


@torch.no_grad()
def capture(m, seqs, windows, length, depth, seed=5):
    m.cap = {}
    rng = random.Random(seed)
    for _ in range(windows):
        (st, ids), _, _ = seqs.sample(rng, length + depth)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.chain(st.cuda().float(), ids.cuda(), depth)
    cap, m.cap = m.cap, None
    return cap


def quantize(W, H, count, K, seed):
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    H_data = {"H": H.clone(), "count": count, "finalized": False, "device": W.device, "first_key": "mtp"}
    args = {"seed": seed, "K": K, "devices": [W.device.index or 0], "apply_out_scales": None, "mul1": True}
    wq, err, out = quantize_exl3(W.float().contiguous(), H_data, args, True)
    return wq, err, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--k6", default="")
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--logit-scale", type=float, default=1.0)
    ap.add_argument("--blocks", default="tests/data/draft/blocks65536.txt")
    a = ap.parse_args()
    ck = Ckpt(a.ckpt)
    s = torch.load(a.state, weights_only=False)
    m = Mtp(ck, load_blocks(a.blocks).cuda())
    named = dict(m.trainable(s["scope"]))
    for n, v in s["params"].items():
        named[n].data.copy_(v.cuda())
    cap = capture(m, Seqs(a.data), a.windows, a.len, 4)
    k6 = {x for x in a.k6.split(",") if x}
    if k6 & SHARED:
        raise SystemExit("the shared expert must keep its checkpoint width")
    tensors, report = {}, {}
    for i, (n, prefix) in enumerate(DENSE.items()):
        if n not in s["params"]:
            continue
        stock = ck.lin(prefix)
        trained = getattr(m, n).detach().float()
        rel = ((trained - stock).norm() / stock.norm()).item()
        native = ck.t(prefix + ".trellis").shape[-1] // 16
        K = 6 if n in k6 else native
        H, count = cap[n]
        wq, err, out = quantize(trained, H, count, K, seed=i)
        dec = exl3_torch.dense({**out, "mul1": ck.t(prefix + ".mul1")})
        report[n] = {"K": K, "native": native, "delta": round(rel, 5), "proxy_err": round(float(err), 6),
                     "q_vs_trained": round(((dec - trained).norm() / trained.norm()).item(), 5),
                     "stock_q_vs_trained": round(((stock - trained).norm() / trained.norm()).item(), 5),
                     "decode_vs_wq": round(((dec - wq.float()).norm() / wq.float().norm()).item(), 6)}
        print(n, report[n], flush=True)
        for part in ("trellis", "suh", "svh"):
            tensors[f"{prefix}.{part}"] = out[part].contiguous().cpu()
        tensors[f"{prefix}.mul1"] = ck.t(prefix + ".mul1")
    plain = {n: v for n, v in s["params"].items() if n not in DENSE}
    tensors.update(export.tensors(ck, plain, a.logit_scale))
    meta = {"step": str(s["step"]), "scope": s["scope"], "k6": ",".join(sorted(k6)),
            "logit_scale": str(a.logit_scale), "source": "crucible mtp_tune requant"}
    # Leak guard: the Hessians come from the data folders and the weights from the state (taint.py).
    tainted = taint.run_tainted(a.data, states=[s])
    taint.guard_output(a.out, tainted, is_dir=False)    # H13: no tainted head inside a git work tree
    meta = taint.tag_meta(meta, tainted)
    save_file(tensors, a.out, metadata=meta)
    digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(json.dumps({"tensors": len(tensors), "sha256": digest, "report": report, **meta}))


if __name__ == "__main__":
    main()
