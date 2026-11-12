# Reproduce the README tables from this tree

Every command here uses only this tree and files that anyone can download. None uses a private sidecar, a tuned draft head file
or a private measurement tool.

## Downloads

| What | Where |
|---|---|
| Crucible 4.05 pack | Hugging Face `turboderp/Qwen3.8-Flash-Next-exl3`, revision `4.05bpw_h6_ng6` |
| Crucible 3.05 pack | same repository, revision `3.05bpw_h5_ng5` |
| Halogen pack | Hugging Face `peonist-ai/halogen-qwen3.8-flash-next`, file `qwen38-flash-next-v2.hgn` and the `mtp`, `ngram` and `vision` files; image `ghcr.io/peonist-ai/halogen-flash-server:0.18.22` |
| Gufo model | Hugging Face `unsloth/Qwen3.8-Flash-Next-GGUF`, folder `UD-Q4_K_XL` (4 files) and `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`; Gufo v0.27.0 |
| BF16 reference | the folder with `logprobs.f32.npy`, `reference.npz` and `meta.json` that was built from `kl/texts.json` |

## Speed rows

`readme_rows.py` starts `python -m server` with the settings of the README (4 sessions of 262,144 tokens, image input and MTP on,
`CRUCIBLE_PF_LAUNCH_ROWS=8192+128` on the 4.05 pack), runs the rows, stops the server, and prints the median of the runs.

    python3 tools/repro/readme_rows.py serve-cmd --ckpt PACK_DIR --pack 4.05        # print the server command only
    python3 tools/repro/readme_rows.py rows --ckpt PACK_DIR --pack 4.05 --out out/405 --runs 2 --profile balanced
    python3 tools/repro/readme_rows.py rows --ckpt PACK_DIR --pack 3.05 --out out/305 --runs 2 --profile balanced
    python3 tools/repro/readme_rows.py rows --url http://127.0.0.1:PORT/v1 --kind halogen --out out/halo   # a running rival server
    python3 tools/repro/readme_rows.py rows --ckpt PACK_DIR --pack 4.05 --out out/d3 --row d3             # one row

One `--row` name per README row: `cold32_ttft`, `cold32_prefill`, `cold130_ttft`, `cold130_prefill`, `resend32`, `dec32_code`,
`dec32_prose`, `dec4`, `d3`, `d1_mixed`. Stop other GPU work first. A run with a platform profile other than `--profile` stops.

## Quality table

Capture the top-20 log-probabilities of each engine on the reference texts, then score them against the BF16 reference.

    python3 tools/repro/kl_capture.py --engine crucible --url http://127.0.0.1:PORT --texts tools/repro/kl/texts.json \
        --tokenizer PACK_DIR/tokenizer.json --stride 8 --passes 4 --out rows-c405.jsonl
    python3 tools/repro/kl_capture.py --engine gufo --gufo-logprobs ...      # Gufo v0.27.0 and later
    python3 tools/repro/kl_score.py --ref REF_DIR --tokenizer PACK_DIR/tokenizer.json --ci 2000 \
        --engine c405=rows-c405.jsonl --engine halogen=rows-halogen.jsonl --out-md quality.md

Start the Crucible server with `--streams 2 --ctx 8192 --prefill 4096` on a pack, and send `spec` 0. The score uses the positions
that every engine file has. The KL is a lower bound of the full-vocabulary KL (21 buckets: the top 20 and a rest bucket).
