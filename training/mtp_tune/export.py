"""Write a trained run's plain MTP tensors as an engine sidecar (CRUCIBLE_MTP_SIDECAR).

  python -m training.mtp_tune.export CKPT RUNDIR/state.pt OUT.safetensors

Only plain (non-EXL3) tensors are written, under their checkpoint names and in
their checkpoint dtypes, so the engine's draft call cost does not change.
"""
import argparse
import hashlib
import json

import torch
from safetensors.torch import save_file

from . import taint
from .mtp_model import Ckpt

P = "mtp.layers.0."
HC = {"norm": "hc_norm.weight", "down": "input_mix_weight_down.weight",
      "up": "input_mix_weight_up.weight", "inject": "block_inject_weight.weight"}
NAMES = {"ne": "mtp.pre_fc_norm_embedding.weight", "nh": "mtp.pre_fc_norm_hidden.weight",
         "qn": P + "self_attn.q_norm.weight", "kn": P + "self_attn.k_norm.weight",
         "router": P + "mlp.gate.weight", "sgate": P + "mlp.shared_expert_gate.weight"}
for mod, pre in (("hca", P + "attn_hyper_connection."), ("hcm", P + "mlp_hyper_connection."),
                 ("mixer", "mtp.hyper_connection_mixer.")):
    for k, v in HC.items():
        NAMES[f"{mod}.{k}"] = pre + v


def scale_mixer(params, ck, c):
    """Draft logits times c: (1 + norm) times c and down divided by c keep the mixer gates equal."""
    params = dict(params)
    for k, name in (("mixer.norm", NAMES["mixer.norm"]), ("mixer.down", NAMES["mixer.down"])):
        if k not in params:
            params[k] = ck.t(name).float()
    params["mixer.norm"] = (1 + params["mixer.norm"].float()) * c - 1
    params["mixer.down"] = params["mixer.down"].float() / c
    return params


def tensors(ck, params, scale=1.0):
    """Checkpoint-named tensors in the checkpoint dtype; rejects EXL3 parameters."""
    if scale != 1.0:
        params = scale_mixer(params, ck, scale)
    out = {}
    for n, v in params.items():
        if n not in NAMES:
            raise SystemExit(f"{n} is not a plain MTP tensor; it needs EXL3 requantization")
        name = NAMES[n]
        old = ck.t(name)
        if tuple(old.shape) != tuple(v.shape):
            raise SystemExit(f"{name}: shape {tuple(v.shape)} != checkpoint {tuple(old.shape)}")
        out[name] = v.to(old.dtype).contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("--logit-scale", type=float, default=1.0, help="calib.py best c")
    a = ap.parse_args()
    if a.state.endswith(".safetensors"):
        # Rescale an earlier sidecar: map its checkpoint names back to parameter names.
        from safetensors.torch import load_file
        meta = __import__("safetensors").safe_open(a.state, "pt").metadata() or {}
        back = {v: k for k, v in NAMES.items()}
        s = {"params": {back[n]: v.float() for n, v in load_file(a.state).items()},
             "step": meta.get("step", "?"), "scope": meta.get("scope", "plain"),
             "canary": meta.get(taint.META_KEY),
             "provenance": meta.get(taint.PUBLIC_META_KEY)}
        if float(meta.get("logit_scale", 1)) != 1.0:
            raise SystemExit("the input sidecar is already scaled")
    else:
        s = torch.load(a.state, weights_only=False, map_location="cpu")
    ck = Ckpt(a.ckpt)
    out = tensors(ck, s["params"], a.logit_scale)
    changed = sum(int(not torch.equal(v, ck.t(n))) for n, v in out.items())
    meta = {"step": str(s["step"]), "scope": s["scope"], "logit_scale": str(a.logit_scale),
            "source": "crucible mtp_tune"}
    # Leak guard: a head trained on capture data carries the canary in its metadata (training/mtp_tune/taint.py).
    tainted = taint.state_tainted(s)    # a state without the public marker counts as tainted
    taint.guard_output(a.out, tainted, is_dir=False)    # H13: no tainted head inside a git work tree
    meta = taint.tag_meta(meta, tainted)
    save_file(out, a.out, metadata=meta)
    digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(json.dumps({"tensors": len(out), "changed": changed, "sha256": digest, **meta}))


if __name__ == "__main__":
    main()
