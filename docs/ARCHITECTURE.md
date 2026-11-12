# Architecture

This page gives an overview of the Crucible source tree.
Each module has a charter: `engine/*/README.md`, `engine/*/kernels/README.md`, `server/README.md`
and `tools/README.md`. When a charter and this page do not agree, the charter is correct.
The [forward specification](FLASHNEXT_FORWARD.md) gives the equations and the shapes.
The [EXL3 specification](EXL3_FORMAT.md) gives the storage and rounding rules.

Comments and charters use the word *train* for one numbered integration of the development history.
For example, "deleted in train 20" means that the 20th integration deleted the code.
Commit hashes in the comments and in `docs/BENCHMARKS.md` name that history. They are not in this repository.

## Overview

Crucible is an exact inference engine.
It runs one model family: Qwen3.8-Flash-Next in EXL3 weight packs, on AMD Strix Halo (gfx1151).
It has two parts:

- The engine: one C++20 and HIP program, `build/engine/crucible`.
- The server: a Python HTTP frontend in `server/`. It speaks the OpenAI Chat Completions API on `127.0.0.1`.

The request path, from HTTP to the kernels and back:

1. A client sends a chat request to `server/` on `127.0.0.1`.
   A reverse proxy in front of the server is optional. The server itself has no authentication.
2. `server/api.py` checks the request. `template.py` renders the chat template.
   `tokenization.py`, `tool_requests.py` and `vision.py` make the prompt IDs.
3. A backend puts the request in a queue: `BatchedEngineBackend` (the default, `--streams N`)
   or `EngineBackend` (`--serial`).
4. The backend writes JSON lines to the engine process `crucible serve CKPT`.
5. `main` (`engine/cli/gen_cli.h`) starts `BatchedServe` or `Serve` (`engine/serve/`).
6. The serve loop calls the runtime: prefill, decode, MTP drafts, verification and sampling.
7. The runtime calls the layer operators and the EXL3 linear operation. They start the HIP kernels.
8. The verified token IDs go back as JSON lines. Only verification commits a token. A draft never does.
9. The server decodes the IDs, parses the tool calls, and sends the answer to the client (with SSE when the
   request streams).

The server never runs a tool call. It only parses and checks it.
The engine is one translation unit: `engine/engine.hip` includes all other headers.
`make` builds it to `build/engine/crucible` (see the [build guide](BUILD.md)).

## Layer diagram

An include goes down only. A module can include its own rank or a lower rank.

```text
rank  module                 holds
----  ---------------------  ------------------------------------------
  8   bench/kernels          bench-only kernel headers (engine never includes)
  7   engine/engine.hip      class Engine, the one translation unit
  6   cli                    gen, gen --jobs, batch mode, pfbench, main
  5   serve                  JSON-line serve (serial and batched), caches
  4   runtime                Engine parts, decode, drafts, sampling, caches
  3   model/flashnext        config, weights, tensor names, state layout
  2   layers + layers/kernels  GDN, QSA, HC, MoE route, PLE operators
  1   exl3 + exl3/kernels    the format: linear op, plans, decode kernels
  0   gfx1151                device helpers, WMMA, sync, pinned memory

leaves (any module may include them; they include only the standard library):
  engine/model/flashnext/geometry.h   engine/switches.h   engine/runtime/sha256.h

kernels (layers/kernels, exl3/kernels): include only kernel headers,
gfx1151 and the leaves. Never host engine code.
```

The Python side is outside the ranks:

```text
client -> server/ (api, template, tokenization, tool_*, backend, batch_backend)
       -> JSON lines -> engine binary (serve) -> the ranked modules above
tools/ (gates, layout rules, measurement tools)
```

## Code ownership

`tools/layout_map.tsv` gives each engine and kernel file one module and one path.
`tools/layout.py` holds the ranks and the include rules.
`tests/test_layering.py` checks the ownership, the include direction and this table.
Each module has one row in the table.

| Directory | Module | Rank | Charter | Responsibility |
|---|---|---:|---|---|
| `engine/gfx1151/` | gfx1151 | 0 | `engine/gfx1151/README.md` | Device helpers, WMMA, sync, pinned memory |
| `engine/exl3/` | exl3 | 1 | `engine/exl3/README.md` | EXL3 linear op, plans, load support |
| `engine/exl3/kernels/` | exl3/kernels | 1 | `engine/exl3/kernels/README.md` | EXL3 GEMV and GEMM device code |
| `engine/layers/` | layers | 2 | `engine/layers/README.md` | GDN, QSA, HC, MoE route, PLE operators |
| `engine/layers/kernels/` | layers/kernels | 2 | `engine/layers/kernels/README.md` | Prefill GDN, QSA, HC, MoE, vision device code |
| `engine/model/flashnext/` | model/flashnext | 3 | `engine/model/flashnext/README.md` | Config, weights, tensor names, state layout |
| `engine/runtime/` | runtime | 4 | `engine/runtime/README.md` | `Engine` parts, decode, sampling, drafts, caches |
| `engine/serve/` | serve | 5 | `engine/serve/README.md` | Serve protocols and batched sessions |
| `engine/cli/` | cli | 6 | `engine/cli/README.md` | `gen` and bench modes |
| `engine/` | engine | 7 | `engine/README.md` | `engine.hip`, the one translation unit |
| `bench/kernels/` | bench | 8 | `bench/README.md` | Bench-only kernel headers; engine code never includes them |

The include check has a baseline of known violations (`BASELINE` in `tests/test_layering.py`).
The baseline can only become smaller. A new violation fails the test.
A fixed violation fails the test until you remove its entry.

`engine.hip` is one translation unit, so a file can use a name of a higher module without an `#include`.
The symbol check (`tools/layout_symbols.py`, `SYMBOL_BASELINE` in `tests/test_layering.py`) reads the
top-level names that each file defines and uses. It fails when a module uses a name that only a higher module
defines. To fix a new hit, move the definition down.

## Modules

### gfx1151 (rank 0)

Wave32 device helpers and host memory helpers: block sums, WMMA, barriers, LDS handoff and pinned memory.
This module has no model, format or layer knowledge. A helper keeps its arithmetic order, because the
callers need it for bitwise output. gfx1151 has only `s_barrier`.

### exl3 and exl3/kernels (rank 1)

The EXL3 format. `exl3.h` holds the linear operation `Lin`: the input transform, the multi-matrix matvec
and the output transform. The kernel headers hold the trellis extraction, the codebook decode, the ordered
GEMV and the WMMA prefill GEMM bodies.

- The codebook decode is exact and stays inside the GEMV and GEMM inner loops.
- The input scale comes before the 128-element Hadamard transform. The output scale comes after the
  output transform.
- Each output element keeps one contraction order for all row counts.

### layers and layers/kernels (rank 2)

The operators of the forward pass that are not EXL3 linear operations: norms, Gated DeltaNet (GDN),
the sparse attention (QSA), the hyper-connections (HC), the MoE route and the per-layer embedding (PLE).

- Row r of a multi-row call gives the bits of the one-row call for row r.
- The fp32 GDN state is part of the exactness contract.
- The forward loops are in `engine.hip` and `runtime/`, not here.

### model/flashnext (rank 3)

The model: the named geometry (`geometry.h`), the weights and tensor names (`model.h`), the state layout
and the prefill row schedule. A geometry value is a named constant. There are no dynamic shapes on the hot
path. The loader checks the metadata before it uses a tensor.

### runtime (rank 4)

The parts of class `Engine` and their rules: decode, verification, sampling, drafts, prefill scheduling,
batch contracts and state caches.

- The output of a request depends only on the request. Batches, the cache state, the load and the
  speculation depth do not change it.
- A restored state is bitwise equal to the state that a new prefill makes.

### serve (rank 5)

The JSON-line protocol between the Python server and the engine process.
`BatchedServe` (`batched_serve.h`) is the default. `Serve` (`serve.h`) is the serial stack.
All request checks run before a state reset or device work. Stdout carries protocol messages only.

### cli (rank 6)

The command line modes: `gen`, `gen --jobs`, the greedy multi-stream mode and `pfbench`.
The CLI parses and checks every option before the model allocates device memory.
A CLI mode adds no forward path. It calls the runtime.

### engine (rank 7)

`engine/engine.hip` holds class `Engine` and includes every module.

### bench (rank 8)

Bench-only kernel variants in `bench/kernels/`. Engine code never includes them.

### server (Python)

The local HTTP API, the chat template, tokenization and the tool-call checks.
The server binds to `127.0.0.1` by default (`--host` selects another address and prints a warning). It passes only the allowed switches to the engine
(`ENGINE_KEEP_SWITCHES` in `server/backend.py`). See the [server guide](../server/README.md).

## Serving

Both stacks use the same HTTP frontend, the same template and the same JSON-line protocol.

| Stack | Select | Python | Engine | State |
|---|---|---|---|---|
| Batched (default) | no flag, or `--streams N` (1 to 8) | `BatchedEngineBackend`, `batch_scheduler.py` | `BatchedServe` | Up to N sessions in one engine |
| Serial | `--serial` | `EngineBackend` | `Serve` | One request at a time |

In the batched stack:

- One worker selects the prompt chunks and the packed decode steps.
- The scheduler gives the first token first priority: the shortest remaining prefill goes first, with aging.
- Packed verification checks the drafts of several sessions in one pass (up to 16 rows).
- At one stream, the output is equal to the serial output.

## Caches

- The conversation cache (`batch_conversation_cache.h`, `--conversation-cache-mib N`) keeps prompt
  snapshots in host memory. A later turn that repeats an earlier prompt starts from the snapshot.
- The disk tier (`batch_state_disk.h`, `--state-dir`) keeps the states that the memory cache removes.
- Session replay and slot resume use the decode-time state of a finished turn.
  See [session replay](SESSION_REPLAY.md).

A cache hit on a snapshot or a block entry gives the same bits as a miss. Session replay is the accepted
exception, which the [session replay](SESSION_REPLAY.md) guide describes. A turn that resumes from the answer of an
earlier turn keeps the rows that the decode path wrote for that answer, so the state of that turn is not a cold state.
A later restore of that state, from a slot checkpoint or a session tail, is not bitwise equal to a cold prefill of the
same tokens. A request that repeats a whole prompt at a saved end point runs no prompt row: it gives the bits of the run
that saved the checkpoint, not the bits of a cold prefill. `CRUCIBLE_EXACT` and `CRUCIBLE_EXACT_RESUME` do not change
this. A request with `cache_control: "bypass"` always runs a cold prefill.

## Speculative decoding

The draft source is the MTP head of the checkpoint. An n-gram lookup can add drafts.

- The target model verifies every draft. A rejected suffix never becomes committed state.
- Greedy IDs do not change with the speculation depth.
- A sampled request uses its own random stream (a keyed sampling law). A seeded request gives the same IDs
  at each speculation depth and in a packed batch pass.

The acceptance rate changes the speed. It does not change the output.

## Exactness rules

The default build keeps these rules:

- The prompt chunk schedule is a fixed function of the request. The load and other sessions never change it.
- Packed decode rows give the same bits as one row.
  Prefill rows (WMMA) and decode rows (VALU) of different requests never share a pass.
- A cache hit on a snapshot or a block entry gives the same bits as a miss. The block cache never uses generated
  decode rows as prompt rows. Session replay is the exception that the cache section names.

Numerics mode: some changes give a speed gain but are not bitwise equal to the reference path.
They are on by default. Set `CRUCIBLE_EXACT=1` to turn off all of them. The output is then bitwise equal
to the named baseline. A change of mode needs no rebuild. The engine reads the numerics switches only
through `engine/layers/numerics.h`.
A lossy change needs the approval of a maintainer. It must also pass the KL, top-1 and long-context checks
of the gate tool.

## Gates

`tools/gates.py` is the gate tool. Run `python3 tools/gates.py --help` for the options.

- Gate 1 compares the prompt state with a named baseline binary: all 49 residual files, bitwise.
  When bitwise equality cannot apply, it compares with an fp64 oracle (`ref/oracle.py`) and the
  tables in `docs/oracle/`.
- Gate 2 compares the residual files of one-row and eight-row prompt chunks, byte for byte.
- Gate 3 compares the complete greedy serial (`--spec 0`) and speculative (`--spec 2`) outputs.

The CPU tests check parsers, state rules, sampling, templates, tool calls, codecs and the measurement
tools. They do not replace the GPU gates.

## Switches

`engine/switches.h` is the registry of the `CRUCIBLE_*` switches. It is the only reader of the environment.

- A name that is not in the registry stops the build.
- Each row gives the name, the kind, the default, the module and a note.
- `tests/test_switch_registry.py` and `tests/test_switch_lists.py` compare the registry with the engine
  reads and with the lists in `tools/gates.py`, `tools/batch_gate_report.py` and `server/backend.py`.
- `SWITCH_CEILING` in `tests/test_switch_ratchet.py` is the number of switch names. It can only go down.
  A new switch needs the removal of an old one.

| Kind | Meaning |
|---|---|
| `kNumerics` | selects a path that is not bitwise; off when `CRUCIBLE_EXACT=1` |
| `kExactSpeed` | a bitwise speed option |
| `kLayout` | a bitwise memory or storage layout |
| `kDiagnostic` | prints, counts, captures or dumps; the output does not change |
| `kBuild` | a compile-time macro; the environment does not set it |
| `kProfile` | changes the model on purpose (an opt-in profile input) |
