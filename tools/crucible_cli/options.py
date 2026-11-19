"""The optional parts of ./crucible setup, each with its measured trade-off (all options in one place).

Every number in a text here is a measured row of the public guide that the text names. No new claim goes here."""

# --ngram16: docs/NGRAM16.md (sizes: tools/ngram16/fetch.py; rows: the guide's table, the 4.05 pack, 5 A/B pairs).
NGRAM16_FOLDER = "ngram16"                       # inside the models folder
NGRAM16_FILE = "ngram_embedding_bf16.safetensors"
NGRAM16_BYTES = 102_400_496_832
NGRAM16_SHA256 = "71205584be130551594a3e1f43949f47c576a3cc65486f1e51f4e7ba45fa42be"
NGRAM16_PEAK_GB = 207                            # the 104.3 GB download and the 102.4 GB file, before the download is deleted
NGRAM16_ROWS = (
    "KL against the BF16 reference: 5.7% lower (95% interval 1.2 to 9.9%)",
    "4 streams at 32K, decode: 1.3% slower",
    "code with speculation: 2.2% faster",
    "serial decode: flat (0.1% lower); cold 32K prefill: flat (0.4% higher)",
)
NGRAM16_TEXT = (
    "16-bit n-gram table (optional; default no). Setup downloads 104.3 GB of the original Qwen/Qwen3.8-Flash-Next "
    "weights (33 files, each checked by sha256) and builds a 102.4 GB file next to the pack. The disk needs about "
    f"{NGRAM16_PEAK_GB} GB at the peak and 102.4 GB after the build (setup deletes the download). The build took 26 minutes "
    "on our machine. The engine then reads the rows of this file from the SSD during decode.\n"
    "Measured on the 4.05 pack (docs/NGRAM16.md, the same binary, 5 A/B pairs for the speed rows):\n"
    + "".join(f"  - {row}\n" for row in NGRAM16_ROWS)
    + "Not measured: the 3.05 pack, and the speed with the setup defaults (4 sessions of 262,144 tokens, drafts on)."
)
NGRAM16_QUESTION = "Build the 16-bit n-gram table now? (y/N)"

# --abliterated: docs/ABLITERATED.md (the pinned public sidecar; the rows of its table "Known problems on the 4.05 pack").
ABL_REPO = "andrevp/Qwen3.8-Flash-Next-exl3-3.05bpw-heretic"
ABL_REVISION = "da2b870f01da36c6601557bc386d0d45c51b36fc"
ABL_FILE = "abliteration.safetensors"
ABL_LICENSE = "Qwen Community License 1.0"
# The LICENSE file at ABL_REVISION. Condition 1 of the license: the notice goes with every copy, so setup keeps it next to
# the sidecar and next to the 4.05 copy.
ABL_LICENSE_FILE = "LICENSE"
ABL_LICENSE_SHA256 = "a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08"
ABL_SOURCE_FOLDER = "andrevp-heretic-da2b870"
ABL_405_FOLDER = "andrevp-heretic-405"
ABL_405_FILE = "heretic-andrevp-4.05bpw_h6_ng6.safetensors"
# The sha256 that the engine pins (kAblationPins, engine/layers/ablation.h) for each pack: the download is the 3.05 file,
# and tools/ablation_pack.py repackage makes the 4.05 file (the same tensor bytes, a header that names the 4.05 pack).
ABL_SHA256 = {
    "3.05": "05f77ae18082de58b31fb645f52ec5689dce22820add8bb81439e06f01998e5f",
    "4.05": "1bb0d4132a5798d2a7da7820b65af1e5d37e39a0cf5726f1986be924f2678baa",
}
ABL_SELECT = ('A client selects the mode for one request with the model name crucible-abliterated, or with the body field '
              '"crucible_ablate": true. Every other request gets the stock model.')
ABL_ROWS = (
    "refusals, 25 harmful prompts: stock 25 of 25, abliterated 0 of 25",
    "refusals, 10 benign prompts: stock 6 of 10, abliterated 0 of 10",
    "KL from the 16-bit model, 896 positions: stock 0.0206, abliterated 0.0248",
    "top-1 agreement with the 16-bit model: stock 94.87%, abliterated 94.42%",
    "decode: 2.7% slower (2 interleaved pairs); cold prefill: 1.3% slower at 32K tokens",
    "tool calls: 2 malformed calls in 2,337 turns with the mode on, none in 10,721 stock turns",
)
ABL_TEXT = (
    "Abliterated mode (optional; default no). The engine can also serve the model with its refusal behavior removed, "
    f"with a public sidecar file ({ABL_REPO} at revision {ABL_REVISION[:7]}, {ABL_LICENSE}). Setup downloads the file, "
    "checks its sha256, and for the 4.05 pack makes a copy with the header of that pack. The stock model stays the "
    "default of every request. " + ABL_SELECT + "\n"
    "Measured on the 4.05 pack, 26 to 29 October 2026 (docs/ABLITERATED.md; the sidecar was tuned on the 3.05 pack):\n"
    + "".join(f"  - {row}\n" for row in ABL_ROWS)
    + "Responsibility: this mode makes the model answer requests that the stock model refuses. Use it only on your own "
    "machine. Add your own controls before other people use it."
)
ABL_QUESTION = "Turn on the abliterated mode (stock stays the default)? (y/N)"
