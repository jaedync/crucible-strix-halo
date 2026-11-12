# Abliterated mode (opt-in)

Crucible can run Qwen3.8-Flash-Next with its refusal behavior removed.
The mode works on the 3.05 bpw pack and on the 4.05 bpw pack.
It is off by default. Read [Responsibility](#responsibility) before you turn it on.

Crucible hosts no sidecar file. You download one public file and, for the
4.05 pack, make a copy of it with a tool in this repository.

## What it does

The mode applies a published "Heretic" sidecar at run time:
[andrevp/Qwen3.8-Flash-Next-exl3-3.05bpw-heretic](https://huggingface.co/andrevp/Qwen3.8-Flash-Next-exl3-3.05bpw-heretic),
revision `da2b870f01da36c6601557bc386d0d45c51b36fc`, file
`abliteration.safetensors`. The license is the Qwen Community License 1.0,
the same as the base model.

For 36 sublayers (attention output, layers 30 to 39; MoE output, layers 17
to 42), the engine changes the sublayer output `y` before the
hyper-connection write-back:

```text
y' = y - b * dot(y, a)
```

The EXL3 weights do not change. The MTP draft layer does not change, so
speculative output stays exact against the ablated model: the ablated model
checks every draft token.

## The sidecar was tuned on the 3.05 pack

The author of the file tuned the directions on the 3.05 bpw checkpoint. The
4.05 pack uses the same vectors: its copy has every tensor value byte for
byte, and only the header names the other pack. Nobody tuned or checked the
directions against the 4.05 weights. The measurements below are the only
evidence that the file works on 4.05. They cover one small prompt set, and
they show costs as well as the effect.

## Turn it on (3.05 pack)

1. Download the sidecar at the pinned revision:

   ```sh
   hf download andrevp/Qwen3.8-Flash-Next-exl3-3.05bpw-heretic abliteration.safetensors \
     --revision da2b870f01da36c6601557bc386d0d45c51b36fc --local-dir ~/models/exl3/ablation/andrevp-heretic-da2b870
   ```

2. Set the environment variable for `crucible gen`, `crucible serve` or
   `python -m server`:

   ```sh
   CRUCIBLE_ABLATE=~/models/exl3/ablation/andrevp-heretic-da2b870/abliteration.safetensors
   ```

3. Find this line in the engine log:

   ```text
   ablation: 36 sublayers from ... (sha256 05f77ae1..., andrevp/...)
   ```

## Turn it on (4.05 pack)

The engine finds a sidecar by its sha256 and ties it to one checkpoint
geometry. The downloaded file is pinned for the 3.05 pack only. The 4.05 pack
needs the same tensors under a header that names the 4.05 pack.

1. Download the sidecar as above. Its sha256 must be `05f77ae18082de58b31fb645f52ec5689dce22820add8bb81439e06f01998e5f`.

2. Make the 4.05 copy. The output directory must be outside the repository:

   ```sh
   python3 tools/ablation_pack.py repackage \
     ~/models/exl3/ablation/andrevp-heretic-da2b870/abliteration.safetensors \
     ~/models/exl3/ablation/andrevp-heretic-405 --pack 405
   ```

   The tool keeps every tensor value byte for byte and checks them after the
   copy. It writes the file only when the sha256 of the copy is
   `1bb0d4132a5798d2a7da7820b65af1e5d37e39a0cf5726f1986be924f2678baa`, the
   value that the engine pins for the 4.05 pack. A different sha256 gives
   exit code 1 and no file.

3. Set the environment variable:

   ```sh
   CRUCIBLE_ABLATE=~/models/exl3/ablation/andrevp-heretic-405/heretic-andrevp-4.05bpw_h6_ng6.safetensors
   ```

4. Find this line in the engine log:

   ```text
   ablation: 36 sublayers from ... (sha256 1bb0d413..., andrevp/...)
   ```

## Known problems on the 4.05 pack

We measured the mode on the 4.05 pack on 26 to 29 October 2026, against the
stock 4.05 pack on the same engine.

| Measure | Stock 4.05 | Abliterated 4.05 |
|---|---|---|
| Refusals, 25 harmful prompts | 25 of 25 | 0 of 25 |
| Refusals, 6 harm categories | 6 of 6 | 0 of 6 |
| Refusals, 10 benign prompts | 6 of 10 | 0 of 10 |
| KL from the 16-bit model, 896 positions | 0.0206 | 0.0248 |
| Top-1 agreement with the 16-bit model | 94.87% | 94.42% |

- **Quality cost.** The KL grows by 0.0042 (21%) and the top-1 agreement falls
  by 0.45 points. The 896 positions are a small sample, so the KL values are a
  lower bound. The test covers next-token quality only. We did not measure
  long answers, code or tool use on the 4.05 pack against a reference.
- **Refusal test.** The set has 41 prompts. A reply counts as a refusal when
  the first 400 characters say so. When the reply ends inside the reasoning,
  the reasoning decides. All 6 stock refusals of benign prompts were decided
  in the reasoning, where the rule can count too many. The ablated run had no
  reply that the reasoning decided.
- **Decode speed.** Decode is 2.7% slower (64.64 against 66.41 tok/s,
  2 interleaved pairs, GPU clock 2,887 to 2,895 MHz). The cause is a lower acceptance of the
  draft head: 2.79 against 2.84 accepted tokens per verify step. The cost of
  a verify step is the same.
- **Prefill speed.** Cold prefill is 1.3% slower at 32K tokens (1,692.90
  against 1,715.75 tok/s, 4 pairs, all 4 slower) and 1.2% slower at 130K tokens.
- **Tool calls.** In 2,337 served turns with the mode on, 2 tool calls came
  back as text with a malformed function tag (`function_tag`). In 10,721 served
  turns on stock builds, none did. Two cases are too few to give a rate, but
  the tag error never occurred on stock builds.
- **Longer answers.** We measured this only on the 3.05 pack, in a test of
  October 5: the mean answer was 704 output tokens against 459 for stock, and
  100 of 126 replies reached the 768-token cap against 30. The author of the
  sidecar saw the same effect. We have not measured it on the 4.05 pack.

Speed on the 3.05 pack: one smoke run showed no cost (74.55 against 73.42
tok/s). We did not run an interleaved comparison there.

## Safety checks

The engine stops with exit code 2 when:

- the file has a sha256 that is not in `kAblationPins` (`engine/layers/ablation.h`);
- a tensor name, dtype, shape or value is not valid;
- the checkpoint does not match the geometry of the pin: expert K 3 and head K
  5 for the 3.05 pack, expert K 4 and head K 6 for the 4.05 pack.
  `CRUCIBLE_ABLATE_ANY_CHECKPOINT=1` skips this check for measurements only.

An empty `CRUCIBLE_ABLATE` is the same as no variable.

## Responsibility

This mode makes the model answer requests that the stock model refuses.
Use it only on your own machine. Add your own controls before other people
use it.
