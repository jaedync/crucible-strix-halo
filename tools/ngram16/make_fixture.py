"""Write fixtures/codec_ref.npz from the torch reference codec (not a test).

Usage: make_fixture.py PATH/TO/exllamav3/modules/quant/exl3_lib/ngram_codec.py OUT.npz
Needs torch. Loads the codec file by path, so the exllamav3 package (and its
GPU extension) is not imported. CPU only.
"""
import importlib.util
import sys

import numpy as np
import torch

KS = (3, 5, 6, 8)
ROWS = 24


def main(codec_path: str, out: str) -> None:
    spec = importlib.util.spec_from_file_location("ngram_codec_ref", codec_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    g = torch.Generator().manual_seed(20261018)
    cb = ref.mul1_codebook("cpu")
    arrays = {"codebook": cb.view(torch.int16).numpy().view(np.uint16)}
    for K in KS:
        states = torch.randint(0, 65536, (ROWS, 160), generator=g, dtype=torch.int64)
        scales = (torch.rand(ROWS, generator=g) * 0.05 + 1e-3).to(torch.float16)
        bias = (torch.randn(ROWS, 160, generator=g) * 0.01).to(torch.float16)
        packed = ref.pack_rows(states, scales, K)
        st, sc = ref.unpack_rows(packed, K)
        deq = ref.dequant_rows(packed, K, cb, bias)
        arrays[f"k{K}_packed"] = packed.numpy()
        arrays[f"k{K}_states"] = st.numpy().astype(np.uint16)
        arrays[f"k{K}_scales"] = sc.view(torch.int16).numpy().view(np.uint16)
        arrays[f"k{K}_bias"] = bias.view(torch.int16).numpy().view(np.uint16)
        arrays[f"k{K}_deq"] = deq.numpy()
    np.savez_compressed(out, **arrays)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
