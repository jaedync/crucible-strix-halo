"""Independent CPU YaRN reference. No engine or GPU dependency.

Check exact float32 values against transformers==4.57.1. The engine's HIP
powf is not a CPU PyTorch oracle. This module makes no HIP bitwise claim.
"""
import argparse
import json
import math


def frequency_table(config, *, enabled=False):
    """Return (CPU float32 inverse frequencies, cos/sin multiplier).

    Accept legacy rope_scaling or modern rope_parameters under text_config.
    The explicit enabled flag defaults to false. This is a reference, not a
    sequence-length policy. Callers must keep native sequences on plain RoPE.
    """
    import torch

    text = config.get("text_config", config)
    params = text.get("rope_parameters", {})
    scaling = text.get("rope_scaling") or params
    base = float(params.get("rope_theta", text.get("rope_theta", 10000000.0)))
    head_dim = text.get("head_dim", text.get("hidden_size", 2560) // text.get("num_attention_heads", 24))
    dim = int(head_dim * params.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25)))
    if dim <= 0 or dim % 2 or not math.isfinite(base) or base <= 1:
        raise ValueError("invalid rotary geometry")
    powers = torch.arange(0, dim, 2, device="cpu", dtype=torch.float32) / dim
    pos_freqs = base ** powers
    plain = 1.0 / pos_freqs
    if not enabled:
        return plain, 1.0
    factor = float(scaling.get("factor", 4.0))
    original = int(scaling.get("original_max_position_embeddings", text.get("max_position_embeddings", 262144)))
    fast = float(scaling.get("beta_fast", 32.0))
    slow = float(scaling.get("beta_slow", 1.0))
    if not math.isfinite(factor) or factor < 1 or original <= 0 or not (0 < slow <= fast < float("inf")):
        raise ValueError("invalid YaRN scaling")
    scale = scaling.get("attention_factor")
    if scale is None:
        def mscale(multiplier):
            return 1.0 if factor <= 1 else 1.0 + 0.1 * multiplier * math.log(factor)
        if scaling.get("mscale") and scaling.get("mscale_all_dim"):
            scale = mscale(scaling["mscale"]) / mscale(scaling["mscale_all_dim"])
        else:
            scale = mscale(1.0)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid attention factor")
    low = dim * math.log(original / (fast * 2 * math.pi)) / (2 * math.log(base))
    high = dim * math.log(original / (slow * 2 * math.pi)) / (2 * math.log(base))
    if scaling.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrapolation = 1 - ramp
    interpolation = 1.0 / (factor * pos_freqs)
    return interpolation * (1 - extrapolation) + plain * extrapolation, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--enable-yarn", action="store_true")
    args = parser.parse_args()
    with open(args.config) as file:
        config = json.load(file)
    inv, scale = frequency_table(config, enabled=args.enable_yarn)
    print(json.dumps(dict(inv_freq=inv.tolist(), attention_factor=scale, enabled=args.enable_yarn), indent=2))


if __name__ == "__main__":
    main()
