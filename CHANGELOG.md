# Changelog

Releases are numbered `v0.<train>.<fix>`. This file lists the newest release first.
The same text is in the notes of each release on GitHub.

## v0.33.1 (2026-11-05)

Release v0.33.1 is a patch release of v0.33.0. It has two fixes and one server option that is on by default.
The fixes are the end-of-turn guard and the top-N log-probabilities of a scoring request that the cache
answers in full. Thanks @m-max for both reports and for the pull request.

### Fixes

- End-of-turn guard (thanks @m-max). A model can write an end-of-turn token (`<|im_end|>` or
  `<|endoftext|>`) inside its reasoning or inside a tool call, for example when it quotes a chat template. The reply
  then ended at once, with finish reason `stop`, no content and no tool call. The server now keeps that token as plain
  text and the model goes on. The limit is 16 times for each reply in the reasoning and 4 times for each reply in a
  tool call. After the limit the token ends the reply as before, and an end-of-turn token in the answer itself always
  ends the reply. The guard acts on chat requests only, not on `/v1/completions`, and not when the kept text would pass
  the thinking cap or leave no room for the reply. Each kept token starts a new engine job, so on a busy server a kept
  token can add a short delay (not measured). The option is `--end-of-turn-guard on|off` (default on); `off` gives the old behavior. The count of kept tokens is in
  `usage.completion_tokens_details.end_of_turn_kept`, and each kept token writes one `crucible_end_of_turn_kept` line
  to stderr with no model text. Two fixes to the pull request are in this release. A tool call whose argument holds a
  call tag, such as a path or a quoted template, no longer looks open after it closes. A `stop` string of the client
  that the kept text matches now ends the reply.
- Scoring log-probabilities on a cache hit (thanks @m-max). A `/v1/completions` request with `logprobs` or
  `top_logprobs` (`max_tokens` 1, temperature 0) whose whole prompt the cache answered returned the top-N list of a
  later decode step, not of the end of the prompt. The token was right, the list and the value of the chosen token
  were wrong. The engine now reads the saved row of the prompt end. Every release since v0.26.0 has this fault.
  A request that the cache does not answer in full, a chat request and a sampled request were not affected.
  On an older release, send `cache_control` `bypass` with the scoring request. This release fixes the top-N list only.
  The token difference after a session replay stays a documented limit (`docs/SESSION_REPLAY.md`).
- The `Dockerfile` default release tag and the build lines in `docs/DOCKER.md` are v0.33.1.

### Speed

- This release has no new speed numbers. The numbers in the [README](README.md) tables and charts are the v0.33.0
  numbers. The one engine change reads a saved row on the path of a temperature 0 request that asks for
  log-probabilities, and no other request runs it. The guard runs in the server and acts only on an end-of-turn token.
  This release changes no prefill or decode arithmetic. We did not run a new speed measurement for it.

### Checks

- A token gate of a test build with the scoring fix, against the engine of daily use, gave byte-identical greedy tokens on
  both packs (4.05 and 3.05). The same test build gave the same top-N list for a cache hit and for a `bypass` request
  on two synthetic prompts. The engine source of that test build is the same as in this release, except the version
  string. The numbers are results of that test binary, not of the build of this release.
- `make test-cpu` passes on the export tree and `make test-fast` on the private tree. The export scan with gitleaks
  reports no finding.

## v0.33.0 (2026-11-04)

Release v0.33.0 is the engine of train 33. It follows v0.32.0.
It has one speed option, one speed change for a second chat, and three fixes. The option is a read hint for the n-gram
(PLE) table when the table is in host memory. The change for a second chat is the late copy of the shared header point.
The fixes are the start-time memory message, a reply with no tokens in the vision check, and the fixed-state size on the
start line. The two draft switches and the smaller arena of v0.32.0 are in this release too.

### Speed

- `CRUCIBLE_PLE_FETCH=1` (opt-in, default 0; needs `CRUCIBLE_PLE_HOST=1`) adds a read hint to the host gather of the n-gram
  table. The bytes are the same. In the runs by context length (Nov 3, 2026, same tool and prompts, public files) the rate
  against v0.32.0 changes by +1.5% to +2.6% for prefill and by +1.7% to +2.9% for decode on the 3.05 pack, at every length
  from 32K to 250K. On the 4.05 pack prefill changes by -1.8% to +1.9% and decode by -3.2% to +0.4%; prefill reads 1.2%
  higher at 32K and 1.8% lower at 192K, and decode reads equal at 32K and 3.2% lower at 250K. The two runs are on the same
  day but not in one session, so we do not claim a gain or a loss on that pack.
- The late copy of the shared header point lets a second cold chat use the header point of its first block, also when
  the server runs several slots. It acts only with `CRUCIBLE_EXACT_RESUME=1`. A night run of Nov 1 to 2, 2026 measured a
  test build of Nov 1, 2026 with this change against the train 32 engine, both with `CRUCIBLE_EXACT_RESUME=1`. All of
  its checks passed: the token gates on both packs, the exact-extend load, short and long groups on both packs, and 4
  pairs of the burst and sequential times to first token, pooled at 0.927 of the old build (7.3% lower). These are
  numbers of that test build, not of the build of this release. The runs by context length did not set
  `CRUCIBLE_EXACT_RESUME`, so the tables and the charts do not include this change.
- The v0.33.0 column of the [README](README.md) tables and the charts is the release ladder and the runs by context
  length, measured Nov 3, 2026 on the build of this release, balanced power profile (GPU clock 2,289 to 2,295 MHz for
  the ladder), with the public files and no tuned draft head file.
  The v0.32.0 column of the same tables is the ladder run of Nov 3, 2026. It has no 4-session value on the 4.05 pack
  (one session of each run stopped early) and one run on the 3.05 pack. The runs use `CRUCIBLE_DRAFT_HEAD=q2`,
  `CRUCIBLE_DRAFT_CHAIN=hc8,lead` and `CRUCIBLE_PF_LAUNCH_ROWS=16384+128`, as v0.32.0 did.
- The engine of this release adds the mem-error text, the fixed-state field and the HF vision loader after the chart
  binary was built. These changes act at start or at load and change no prefill or decode arithmetic.

### Exactness and checks

- The output tokens of a request do not change with the PLE read hint: it reads the same table bytes. The late copy of
  the header point keeps the exact state (296 rows restored in both packs of the night run) and the token gates and
  exact-extend groups passed on both packs.
- The engine of this release is the build that our own server ran in daily use on Nov 4, 2026, plus the start-time
  and load changes named under Speed. We publish after that day.
- `make test-cpu` passes on the export tree and `make test-fast` on the private tree. The export scan with gitleaks reports no finding.

### Fixes

- The start-time memory refusal names the free host and device memory, the size of the state that does not fit, and a
  setting that the same check accepts. The example is a 1-stream 256K request in Docker (8.5 GiB free, 9 GiB reserve):
  the message now says that all of the state is already allocated, and that a lower `--streams` or `--ctx` cannot help.
  The accept or refuse decision does not change.
- The vision token check fails only a reply with no tokens. A reply that ends with a natural stop after a few tokens no
  longer fails the check.
- The compact-state start line prints the fixed part of one slot's state as `fixed_state_bytes=<n>`, after
  `logical_state_bytes`. The field carries sizes only, never request content. A dashboard can draw the KV band and the
  recurrent band separately.
- A `Dockerfile` builds the engine for gfx1151 from a pinned release tag and installs the HTTP server; the image
  holds no model. `docs/DOCKER.md` lists the tested versions and the run line.
- `tests/draft_tally_test.cpp` compiles with g++ 13 and glibc 2.39 (Ubuntu 24.04). The compiler read the comma in the
  compared type as a second macro argument. A second pair of parentheses keeps the argument whole, and the test checks the
  same thing. This is a test-only fix, with no change to the engine. The public CPU tests job failed on this line
  (thanks @m-max).
- The cache rule in `docs/ARCHITECTURE.md` names its exception: a turn that resumes from the
  answer of an earlier turn (session replay) keeps the rows that the decode path wrote for that answer, so a later
  restore of that session is not bitwise equal to a cold prefill. A hit on a snapshot or a block entry
  still gives the same bits as a miss. Docs only (thanks @m-max).

### Measurements

The speed numbers above are on one machine (Strix Halo, 128 GB) with the balanced power profile and the same tool as
v0.32.0. The tables for both packs, with the run counts and the GPU clock, are in `docs/RESULTS.md`.

## v0.32.0 (2026-11-03)

Release v0.32.0 is the engine of train 32. It follows v0.31.0.
It has three speed changes and one loader fix. The speed changes are a low-bit draft head for the MTP drafts, a draft chain that overlaps the first draft pass with the server work, and a smaller prefill stage arena that lets a prefill launch of 16,384 rows fit on the 4.05 pack. The loader fix is for a checkpoint with a plain HF vision tower (see Fixes).
The two draft changes are opt-in switches (default off). The arena change is on by default.

### Speed

- Decode is faster. `CRUCIBLE_DRAFT_HEAD=q2` makes the MTP draft head read a 2-bit copy of its rows for coarse logits. The 8 best blocks of 128 tokens are
  then recomputed from the EXL3 tiles. The draft changes only; the verify pass checks every draft, so the output does not change.
  `CRUCIBLE_DRAFT_CHAIN=hc8,lead` changes the draft calls: `hc8` reads the hyper-connection weights of the MTP layer as int8, and `lead` starts the first
  MTP pass of the next round while the server handles the tokens of the step. `CRUCIBLE_DRAFT_HEAD` and `CRUCIBLE_DRAFT_CHAIN` are unset by default.
- The prefill stage arena is smaller. Row buffers of the attention and MoE views share bytes where their lifetimes do not overlap. At the launch of
  16,384 rows (`CRUCIBLE_PF_LAUNCH_ROWS`, the default, `16384+128`) the arena is 2,174 MiB instead of 4,656 MiB (2,482 MiB less). The arithmetic does not change.
  The launch of 16,384 rows now fits on the 4.05 pack with the long-context layout.
- Gate numbers (4.05 pack, ABAB, 4 pairs, order 2 AB and 2 BA, order-corrected; the v0.32.0 engine with `CRUCIBLE_DRAFT_HEAD=q2`, `CRUCIBLE_DRAFT_CHAIN=hc8,lead`
  and `CRUCIBLE_PF_LAUNCH_ROWS=16384+128` against the v0.31.0 engine with its own launch rows of 8192+128; both engines with the same tuned draft head file, which is not part of the release; the cold prefill gain includes the launch-row change):
  cold 32K prefill +4.21% (se 0.14%, faster in 4 of 4 pairs), decode of 4 sessions at 32K +4.45%, decode of 1 stream with code text +2.07%, serial decode +0.55% (se 0.07%, a small change),
  energy per arm -3.2%. D1 (mixed text, 6 pairs): +4.09%, equal token IDs in 30 of 30 requests of every pair.
- Measured Nov 3, 2026 on the build of this release, balanced power profile, in the runs by context length (32K to 250K), same tool and prompts as v0.31.0, public files, no tuned draft
  head file: cold prefill against v0.31.0 changes by +1.0% to +3.4% on the 4.05 pack (v0.31.0 measured Nov 1, v0.32.0 on Nov 3) and by -1.9% to -1.0% on the 3.05 pack (v0.31.0 measured Nov 2, v0.32.0 on Nov 3);
  decode changes by +2.8% to +4.7% on the 4.05 pack and by +0.9% to +2.6% on the 3.05 pack. The v0.32.0 runs set `CRUCIBLE_DRAFT_HEAD=q2`, `CRUCIBLE_DRAFT_CHAIN=hc8,lead` and launch rows 16,384, and also
  `CRUCIBLE_PLE_HOST=1` and `CRUCIBLE_COMPACT_STATE=1`. The v0.31.0 runs set neither draft switch, and on the 4.05 pack they used 8,192 launch rows.
  On the 3.05 pack both builds used 16,384 launch rows; prefill reads 1.0% to 1.9% lower at every length with a higher GPU clock. The runs are on different nights (Nov 2 and Nov 3) and there is no same-night
  comparison, so we do not know yet if this is a loss. With the defaults (no `CRUCIBLE_DRAFT_HEAD` and no `CRUCIBLE_DRAFT_CHAIN`) we expect decode as in v0.31.0; we did not measure the defaults.
  The vision loader fix of this release came after these runs; it changes only the vision load path, not prefill or decode.
  The tables and the run counts are in `docs/RESULTS.md` (section Decode by context length).
- The release ladder row of v0.32.0 is not in the tables: the ladder run of Nov 3 did not complete, and the row comes with v0.33.0.

### Exactness and checks

- The output tokens of a request do not change with speculation, the draft head, the draft chain or the arena: the verify pass checks every draft, and the arena change does not change the arithmetic.
- Token gates on both packs (4.05 and 3.05) with the three switches on: the output tokens equal the v0.31.0 engine on the fixed gate set (CHAIN PASS).
- The exact-extend load gave the same token IDs on both packs: 6 of 6 cases each.
- A memory soak of 3 runs passed (lowest free memory 13.1 GiB, 124 turns and 123 cache hits per run).
- `tests/stage_plan_test.cpp` reproduces the old arena sizes and checks that the live buffers of each pass stay disjoint. The draft head and the draft chain have GPU and host tests
  (`tests/draft_quant_test.hip`, `tests/draft_chain_switch_test.cpp`).

### Abliterated mode on both packs (opt-in)

- `docs/ABLITERATED.md` now covers the 4.05 pack. `tools/ablation_pack.py repackage` makes the 4.05 copy of the public andrevp sidecar from the downloaded file
  and refuses to write it unless the copy has the sha256 that the engine pins. Crucible hosts no sidecar file. The mode is off by default.
- The sidecar was tuned on the 3.05 pack. On the 4.05 pack we measured (26 to 29 Oct 2026): 0 of 25 harmful prompts refused (stock 25 of 25); KL from the 16-bit model
  0.0248 against 0.0206 (896 positions); decode 2.7% slower; cold prefill 1.3% slower at 32K. 2 of 2,337 turns had a tool call that came back as text
  (0 of 10,721 on stock builds). The page lists the details and the responsibility note.

### Fixes

- History messages may carry `reasoning` next to or instead of `reasoning_content` (LiteLLM and Hermes send both). An empty or null value counts as absent. Only two different non-empty texts, or a value that is not a string, give a 400. Request errors now name the field and the message index (`messages[N]`) (thanks @dblaber).
- A checkpoint whose vision tower is in plain HF format (BF16, no EXL3 tower) now runs vision when `vision_k6.safetensors` from the stock 4.05 pack is in the model folder. Without that file, `--vision` stops at start with a message that names the file and the `hf download` command. The engine runs the stock EXL3 tower from `vision_k6.safetensors` and does not use the HF tower of the checkpoint; images are exact only when the HF tower holds the stock weights. The loader also accepts BF16 for the patch embedding and the vision biases and converts them to F16 at load (thanks @OleksKon).
- `repetition_penalty` at `1` or null is now accepted and has no effect, like `presence_penalty` and `frequency_penalty`
  at `0`. It was refused as an unknown field. Any other value returns HTTP 400.

### Measurements

Every speed number above except the last two bullets of Speed is on the 4.05 pack, measured with an A/B script on one machine (Strix Halo, 128 GB), alternating the old and
new build in balanced order, with the balanced power profile. The tables for both packs, with the run counts and the GPU clock,
are in `docs/RESULTS.md`.

## v0.31.0 (2026-11-02)

Release v0.31.0 is the engine of train 31. It follows v0.30.0.
It has one engine change: the sparse attention step of prefill (the QSA union flash) uses a new kernel. The output is exact.
The decode path does not change. The server, the cache and the options do not change.

### Speed

- Prefill is faster (measured at 32K and 130K on the 4.05 pack, see below). The new kernel `QsaUnionStageKernel` replaces
  `QsaUnionHalfQKernel` in the QSA union flash (12 attention layers). It does the same arithmetic in the same order, so
  the bytes are equal. It splits the work between two groups of waves: loader waves bring K and V into LDS with buffer
  loads, one step ahead, and the other waves run the QK chain, the softmax and the P V step. It is a build-time constant
  (`kQsaUnionStage` in `engine/layers/kernels/prefill_qsa_union.h`), not a `CRUCIBLE_*` switch. Set it to `false` and
  rebuild to get the old kernel.
  - First gate (an engine build of Oct 30 with the new kernel against the same engine without it; 4.05 pack; ABAB, 4 pairs,
    order-corrected): cold 32K prefill +2.62% and cold 130K prefill +3.86%.
  - Against the v0.30.0 engine (4.05 pack; ABAB, 4 pairs, order 2 AB and 2 BA, order-corrected): cold 32K prefill
    +2.91% (medians of the arms: 1,729 and 1,766 tok/s), faster in 4 of 4 pairs; the tool rated this as within noise, because the old arm spread 3.4%.
- Measured Nov 1 to 2, 2026 on the build of this release, balanced power profile, in the runs by context length (32K to 250K), same tool and prompts as v0.30.0:
  cold prefill against v0.30.0 changes by +1.6% to +4.5% on the 4.05 pack (both measured Nov 1) and by +2.6% to +4.7% on the 3.05 pack (v0.30.0 on Nov 1,
  v0.31.0 on Nov 2, a repeat; public files, no tuned draft head file); decode changes by -0.2% to +2.1% and -0.3% to +0.4%. The tables and the reason for the
  repeat are in `docs/RESULTS.md` (section Decode by context length).

### Exactness and checks

- Token gates on both packs (4.05 and 3.05): the output tokens are equal to the old kernel on the fixed gate set.
- The bench `bench/qsa_attn.hip` compares the new kernel with a verbatim copy of the old one (`bench/kernels/qsa_attn_trunk.h`)
  on 9 shapes, and 3 deliberate errors in the new kernel must show a difference. All checks passed.
- The exact-extend load (a restore of a long state against a cold prompt) gave the same token IDs on both packs: 6 of 6 cases.
- Decode: D1 (6 pairs, 4.05 pack, against the v0.30.0 engine, both engines with the same tuned draft head file, which is not part of the release) is within noise (corrected ratio 0.9987), and the token IDs are equal in all
  30 of 30 requests of every pair. A memory soak of 3 runs passed.

### Measurements

Every speed number above except the last bullet of Speed is on the 4.05 pack, measured with an A/B script on one machine (Strix Halo, 128 GB), alternating the old and
new build in balanced order, with the balanced power profile. The tables for both packs, with the run counts and the GPU clock,
are in `docs/RESULTS.md`.

## v0.30.0 (2026-10-31)

Release v0.30.0 is the engine of train 30. It follows v0.26.1.
The v0.26.1 engine is the v0.26.0 engine (v0.26.1 changed docs and one tool). So this list covers all engine changes since v0.26.0.
It makes prefill and decode faster, adds a gateway request id to the server log, and stops at a bad draft head file.
Every change below keeps the output exact, except where the text says that it is opt-in.

### Speed

Against v0.26.0 on the same machine with the same tool and prompts (v0.26.0 measured Oct 28 to 29, v0.30.0 on Oct 31; balanced power profile,
prefill launches of 8,192 rows): cold prefill is 5% to 6% faster on the 4.05 pack and 8% to 11% faster on the 3.05 pack, from 32K to 250K
(8% to 9% against v0.26.0 measured the same night). Decode is 6% to 8% faster on the 4.05 pack from 64K to 250K (same reply text, same tokens
per verify pass). On the 3.05 pack, decode is about equal (-2% to +1% against v0.26.0 measured the same night). The 3.05
decode values come from builds that used a tuned draft head file that is not part of the release. In the ladder, decode with 4 sessions on the
4.05 pack reads 4% lower (102.4 against 106.9 tok/s); we have not found a cause. The tables are in docs/RESULTS.md in the repository.

- Long prompts prefill in launches of 16,384 rows (was 8,192). The MoE layers have tile kernels for launches of 8,192
  rows or more. Method: one lease on Oct 25, 4.05 pack, same prompts, engine build of that date with the tile kernels, three arms in
  ABC order (not ABAB). Cold 32K prefill is 3.1% faster (1,746.7 tok/s) and cold 130K is 3.0% faster, with equal token IDs.
  The launch needs more device memory: the stage arena is 4,656 MiB instead of 2,346 MiB (3.3 GB more in total).
  To keep the old size, set `CRUCIBLE_PF_LAUNCH_ROWS=8192+128`.
- The dense prefill GEMMs have a new half-precision kernel. The kernel is 4.4% to 4.8% faster per call, and its output
  bytes are equal to the old kernel. Method: ABAB, 4 pairs, 4.05 pack. Cold 32K prefill is 1.3% faster
  (1,709.95 to 1,732.60 tok/s) and cold 130K is 1.2% faster, in 4 of 4 pairs. Energy per prompt is 1.2% lower.
- Decode with several sessions is faster. A new MoE kernel (`CRUCIBLE_MOE_B128`, default now 12) serves the verify
  passes of 2 to 16 rows with a capped grid. Method: ABAB, 6 pairs, 4.05 pack, no draft head file, 4 sessions at 32K, measured on an earlier build
  (train 28, switch value 3; the default is now 12, which we did not measure the same way). Decode total was 8.3% faster, in 6 of 6 pairs. Serial decode and cold prefill do not change (within 0.1%).
  The start check stops the server with a clear message if the grid is under one block.
  To go back to the old kernel, set `CRUCIBLE_MOE_B128=0`. A bad value stops the start with exit code 2.
- The server now starts the weight lookahead for batched passes. Method: ABAB, 6 pairs, 4.05 pack. Serial decode
  through the server is 2.9% faster, in 6 of 6 pairs.
- Serial decode on the 4.05 pack is 1.7% faster with a new exact kernel for the hyper-connection weights
  (ABAB, 5 pairs, equal token IDs).
- Other exact prefill changes: a deeper load pipeline in the MoE gate and up kernel, row tiles for MoE prefill from
  1,024 rows, a fused GDN scan, and a carried GDN scale. The tail rows of a prompt join the next launch.
  Each one is bitwise equal to the old path. We do not claim a separate gain for them.

### Conversation cache

- A cache restore of a long state uses far less host memory. The host copy of a spill to the disk tier fell from
  3.1 to 5.2 GB to about 250 MB, written in pieces of 8 MiB. The state is restored exactly: a restore of 18,285 tokens
  gave the same token IDs as a cold prompt.
- A fix makes the write ledger of the disk tier exact: the day count no longer adds a file twice when two block writers
  charge at the same time.
- `CRUCIBLE_EXACT_RESUME` (opt-in, default 0) makes the resume of a longer prompt equal to a cold prompt. It costs at
  most 20 extra prefill rows, one small launch and one save per prompt. See `docs/SESSION_REPLAY.md`.
  On the default path, a resume point 1 to 8 rows below the end of a bulk launch is now refused, so more resumes
  equal a cold prompt.
- `CRUCIBLE_LAZY_SHADOW` (default off) is a check for testing: it keeps the old eager copy and compares each piece.

### Server

- New option `--client-request-id-header NAME` (also `CRUCIBLE_CLIENT_REQUEST_ID_HEADER`). The default header is
  `X-Client-Request-Id`. A valid value (1 to 64 letters, digits and `_ . : -`) goes into every `crucible_prefill`,
  `crucible_decode`, `crucible_turn`, `crucible_error` and `crucible_cancel` line as `client_request_id`. A gateway can
  pair a request with its lines when requests overlap. The value never goes into a reply.
- If `CRUCIBLE_MTP_SIDECAR` names a file that is missing, or one that replaces fewer than 17 tensors, the engine now
  stops with exit code 2 and the line `mtp sidecar error`. Before, the engine ran on, and the exact verify step hid the fault: only the draft acceptance fell.
- The engine writes an `active config:` line at start. It lists the `CRUCIBLE_*` switches that the environment sets,
  with their values. A value with a path shows as `(path)`, and the draft head file shows as its size and a hash prefix.

### Build

- `make split-facts` and `SPLIT_CHECK=1` (default) keep the parallel build exact: a part that sees other compile facts
  than the whole module stops the build. Run `make -j 8 split-facts` when the build says `argument facts drift`.
  `SPLIT_CHECK=0` skips the check. See `docs/BUILD.md`.
- `CRUCIBLE_GDN_VERIFY` (default off) is a GDN verify kernel that is bitwise equal. `CRUCIBLE_HC_WMMA=2` (default off)
  reads int8 hyper-connection weights. It is a quality change, so it stays off.

### Measurements

Every number above is a speed change on the 4.05 pack, measured with an A/B script on one machine (Strix Halo, 128 GB),
alternating the old and new build in balanced order. The tables for both packs, with the run counts and the GPU clock,
are in `docs/RESULTS.md`. The speed table of the README compares v0.30.0 with v0.26.0 and v0.22.0.

## v0.26.1 (2026-10-25)

- README: new section "16-bit n-gram table (opt-in, experimental)" with one command, `python tools/ngram16/fetch.py`, that downloads only the 33 source files (104.3 GB), builds the BF16 table and checks its size and sha256. Opt-in with `CRUCIBLE_PLE_HOST=1 CRUCIBLE_PLE_BF16=<file>`.
- `docs/OPENAI_CLIENTS.md`: request fields and refusals for OpenAI clients (pull request 1).

## v0.26.0 (2026-10-24)

Release v0.26.0 is the engine of train 26. It follows v0.22.4.
It adds new packs, a faster build, a conversation cache that explains its misses, and a way to update.

### New packs

- The loader reads the `h6_ng8` EXL3 packs from turboderp, with self-calibrated weights. The sizes are 2.18, 2.68,
  3.18, 3.68 and 4.18 bits per weight.
- Each of these packs loads and generates text.
- Weight widths go from 2 to 8 bits. Routed experts can also have half-bit widths (for example 3.5).
- The `h6_ng8` packs hold an 8-bit n-gram table. `CRUCIBLE_PLE_BF16` (off by default) reads the n-gram table at
  16 bits from a file instead. It needs `CRUCIBLE_PLE_HOST=1`.
- The packs 3.05 and 4.05 stay the qualified packs. The README tables describe them.

### Speed

- Several parts of decode and prefill are faster. Speculative decoding is still exact: the target model checks every draft.
- A draft chain now stops at a stop token and at the token limit.
- Long prompts prefill in launches of 8,192 rows. A short last part joins the last launch.

The tables compare v0.22.0 (A) with v0.26.0 (B). A is the engine of train 22. The releases v0.22.1 to v0.22.4 use the
same engine. Each cell is the median of 5 pairs. The v0.22.0 cell for 4 sessions on the 4.05 pack is the median of 4 pairs, because one pair gave no value. The ratio is B divided by A. For the rows in tok/s, a ratio above 1
means that v0.26.0 is faster. For the rows in seconds, a ratio below 1 means that it is faster.

Setup: Strix Halo with 128 GB. Each pack ran 5 pairs of A and B, in the order AB, BA, AB, BA, AB. A pair that failed a
rule (low GPU clock, high disk wait, or a missing value) was run again once. The 4.05 pack had 7 tries (14 runs) and the 3.05 pack had 5 tries (10 runs). The median GPU clock
was about 2,330 MHz in both builds. The server ran 4 sessions of 262,144 tokens, with MTP on and image input on. Only
the serial row has MTP off. The cold rows use prompts that no cache has seen.

**4.05 bpw pack**

| Row | A: v0.22.0 | B: v0.26.0 | B/A |
|---|---:|---:|---:|
| Serial decode, short context (tok/s) | 30.9 | 32.0 | 1.036 |
| Decode at 32K, 1 session, code (tok/s) | 66.1 | 64.9 | 0.982 |
| Decode at 32K, 1 session, prose (tok/s) | 49.0 | 49.3 | 1.005 |
| Decode at 32K, 4 sessions, total (tok/s) | 110.8 | 105.2 | 0.949 |
| Cold prefill 32K, time to first token (s) | 19.74 | 19.04 | 0.965 |
| Cold prefill 130K, time to first token (s) | 84.15 | 81.94 | 0.974 |
| Re-send of the same 32K prompt, time to first token (s) | 1.872 | 0.021 | 0.011 |

**3.05 bpw pack**

| Row | A: v0.22.0 | B: v0.26.0 | B/A |
|---|---:|---:|---:|
| Serial decode, short context (tok/s) | 35.4 | 36.5 | 1.029 |
| Decode at 32K, 1 session, code (tok/s) | 75.3 | 73.4 | 0.974 |
| Decode at 32K, 1 session, prose (tok/s) | 59.5 | 59.2 | 0.995 |
| Decode at 32K, 4 sessions, total (tok/s) | 129.6 | 129.2 | 0.997 |
| Cold prefill 32K, time to first token (s) | 18.75 | 18.78 | 1.001 |
| Cold prefill 130K, time to first token (s) | 80.39 | 80.77 | 1.005 |
| Re-send of the same 32K prompt, time to first token (s) | 1.728 | 0.021 | 0.012 |

What the tables show:

- A prompt that you send again gets its first token in about 0.02 s instead of 1.7 to 1.9 s.
- Cold prefill is 2.6% to 3.5% faster on the 4.05 pack. On the 3.05 pack it is equal within 0.5%.
- Serial decode is 3% to 4% faster.
- Decode with 4 sessions on the 4.05 pack is 5% slower: 110.8 to 105.2 tok/s (0.949). One of the 5 runs of v0.26.0
  was low (86.4 tok/s). In the 4.05 test, 3 of the 14 runs gave no value for this row: 2 of v0.22.0 and 1 of v0.26.0.
  This is a known fault of the measurement and is not new in this release. The 3.05 test had no failed run.
- The 3.05 pack shows no loss with 4 sessions (0.997).
- Decode at 32K with 1 session varies more between runs in v0.26.0 on the code prompt. The medians are within 3%.

### Conversation cache

- A prompt that you send again restores from the cache in more cases. This includes a prompt that the cache holds
  in full.
- A cache miss has a stated cause. The `crucible_turn` line of the server log now has `miss_reason`. On a miss it also
  has `closest_shared_tokens`, `closest_age_s` and `closest_outcome`.
- The slot, host and disk tiers count their own lookups and hits (`cache_slot_*`, `cache_host_*` and `cache_disk_*`
  on each turn line). Each host lookup writes one `cache_lookup tier=host` line.
- The newest state of a conversation stays in the cache.
- A slot can copy the state of another slot on the device when the host cache has a miss.
- The disk tier (`--state-dir`) stores equal blocks once. It reads a state back during a run. A state that the host
  cache refuses goes straight to disk.
- The disk files have a new format. At start, the server removes the state files of v0.22 from the state folder.
  The cache starts empty.

### Server

- `GET /health` answers `{"status":"ok"}` when the engine runs. Use it as a readiness check, for example in llama-swap.
  This is already in v0.22.1.
- The server accepts any `Host` header by default. Set `CRUCIBLE_ALLOWED_HOSTS` or `--allowed-host` for a strict list.
  This is already in v0.22.1.
- `GET /v1/crucible/version` gives the version and the git commit. `GET /v1/models` has `crucible_version`.
- The server flags `--temperature`, `--top-k`, `--top-p`, `--min-p` and `--default-max-tokens N|max` set defaults for
  requests that omit those fields. A field in the request always wins. Without a flag, nothing changes.
  A request for log-probabilities stays at temperature 0. This is already in v0.22.2.
- During a speculative turn, the server writes a `crucible_decode` line with draft counters. The line holds counts
  only. It never holds token IDs or text.
- A completion with `max_tokens` 1, temperature 0 and no stream accepts `top_logprobs` (0 to 20).
- With `--serial`, the engine stops a thinking turn at the point where the server closes the think block (the
  thinking budget, or the end of the answer room), not at `max_tokens`. Before, the engine decoded on to `max_tokens`
  after the close, and the answer waited behind it. In our GPU test of 2026-10-23, the wait after the forced close fell
  from 12.0 s to 0.3 s, and the engine made no extra tokens.

### Build

- The engine device code compiles in parallel parts (64 by default). An edit rebuilds only the parts that it changes.
  `make SPLIT=0` uses plain `hipcc`. `make isa-check` shows that the ISA is equal.
- The number of parallel compiles follows `make -j`. The default is 1.

### Updates

- The file `VERSION` and the file `CHANGELOG.md` give the release number and the notes.
- `build/engine/crucible --version` and the server start line show the version.
- The server asks GitHub once a day if a newer release exists, and prints a notice. It changes nothing. Turn the
  check off with `--no-update-check` or `CRUCIBLE_NO_UPDATE_CHECK=1`.
- `make update` moves your clone to the newest release, builds it, and runs a self-test. If a step fails, it restores
  the previous release. If a kill or a power loss stops an update, the next `make update` or `make rollback` restores the
  previous release first. `make rollback` undoes the last `make update` (once, with no local changes). It then holds the
  release that you left: `make update`, and the timer, skip it until a newer release exists.
- Automatic updates are off. "Updates" in the README shows how to turn on the timer.

### How to update from v0.22

1. Stop the server.
2. Run `git fetch --tags` and `git checkout v0.26.0` in your clone, then `make`.
3. Start the server. For later releases, run `make update`.

The cache starts empty after the update. Your pack files and checkpoints do not change.

Needs your action, if you come from v0.22.3 or earlier: a thinking request with `max_tokens` below 1280 and no
thinking budget now gets no thinking. `--min-thinking-tokens 0` restores the old behavior. If you use v0.22.4, you
have this change already.

## v0.22.4 (2026-10-23)

A server release for the train 22 engine. It improves thinking requests with a small `max_tokens`. It adds a flag for
the listen address. The engine did not change.

- New flag `--min-thinking-tokens N` (default 256, range 0 to 262144). With thinking on, the server keeps room for the
  answer. Before, a small `max_tokens` left thinking 1 token. Now, when thinking would get fewer than N tokens, the
  server serves the request with thinking off, and the answer gets all of `max_tokens`. The usage block then shows
  `completion_tokens_details.reasoning_skipped_by: "min_thinking_tokens"`.
- New flag `--host ADDRESS` (default `127.0.0.1`) sets the listen address, IPv4 or IPv6. For an address that is not
  loopback, the server writes a warning at start, because it has no authentication.
- `/tokenize` and `/apply-template` return the prompt that chat sends, also when the server skips thinking.

Needs your action:

- With the default settings, a thinking request with `max_tokens` below 1280 now gets no thinking. This includes a
  request with no `max_tokens` (the default is 128). To keep thinking, send `max_tokens` 1280 or more, or a thinking
  budget (`max_thinking_tokens` in the request, or `--max-thinking-tokens` on the server).
  `--min-thinking-tokens 0` restores the old behavior.
- If you serve other machines with `--host`, also turn on the strict Host check (`--allowed-host`) and use a firewall.

Requests with a larger `max_tokens` or with thinking off give the same tokens as in v0.22.3.

## v0.22.3 (2026-10-23)

A server release for the train 22 engine. The engine did not change.

- The server accepts `presence_penalty` and `frequency_penalty` when the value is `0` or null. Some clients, such as
  SillyTavern, send both in every request. The fields have no effect, because the engine has no penalty sampler.
  Any other value gets HTTP 400.

## v0.22.2 (2026-10-23)

A server release for the train 22 engine. It adds server-level request defaults. The engine did not change.

- New flags `--temperature`, `--top-k`, `--top-p` and `--min-p` set the value for each request that does not send the
  field. A field in the request always wins.
- New flag `--default-max-tokens N|max` sets the output tokens for a request that sends no `max_tokens` and no
  `max_completion_tokens`. With `max`, the request gets all the context room after its prompt.
- The server checks each value at start. A bad value stops the start with exit code 2.
- Without the new flags, the server works as before: greedy decoding and 128 output tokens when a request sends no
  count.
- The server README names the range of `--top-k` (0 to 248320), says that `"max_tokens": null` gets HTTP 400 with
  `--default-max-tokens max`, and says that `--short-streams` can make a `max` request wait longer.

## v0.22.1 (2026-10-23)

A fix for the Host check of the server.

- The server accepts any `Host` header by default, so a proxy or a container name works. `CRUCIBLE_ALLOWED_HOSTS` and
  `--allowed-host` turn on a strict list. A request with two `Host` headers gets HTTP 400 when the list is on.
- `GET /health` answers `{"status":"ok"}` when the engine runs. It gives HTTP 503 when the engine has stopped.
- The engine never inherits `CRUCIBLE_ALLOWED_HOSTS`.
- The server README has an example for llama-swap.

## v0.22.0 (2026-10-21)

The first numbered release. It is the engine of train 22.

- The engine runs EXL3 checkpoints of Qwen3.8-Flash-Next on one AMD Strix Halo APU (gfx1151).
- Speculative decoding is exact: the target model checks every draft.
- A local OpenAI-compatible server gives chat, streaming, tools and up to eight sessions on one engine.
- The conversation cache keeps session states in host memory. An optional disk tier keeps them across restarts.
- Image input is opt-in (`--vision`).
- Two packs are supported: `3.05bpw_h5_ng5` and `4.05bpw_h6_ng6`.
- The README states the measured speed and the distance from the 16-bit model.
