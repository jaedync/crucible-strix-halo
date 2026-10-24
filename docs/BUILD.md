# Build and test

Use GNU Make as the build entry point.
Run these commands from the repository root.

## Requirements

The tested GPU uses gfx1151, Linux, and ROCm 7.2.4.
The engine requires a HIP compiler with C++20 support and POSIX threads.
Other GPU targets and ROCm versions are not qualified.

CPU tests require Linux, GNU Make, a C++20 compiler, and Python 3.12 or newer.
They do not require ROCm, model weights, or a network after dependency installation.

Create a Python environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

## Engine and probes

Build the engine:

```sh
make
```

The output is `build/engine/crucible`.
The device code compiles in parallel parts, and an edit rebuilds only the parts it touches.
The number of parallel compiles follows `make -j` (default 1). On the test machine, build with `make -j 1` only.
The ISA is the same as with plain `hipcc`. Run `make SPLIT=0` for plain `hipcc`.
After a kernel change, run `make isa-check`.
Before you add a kernel, read the comment above the `isa-check` target in the Makefile.
The same binary provides `gen`, `serve`, and `pfbench` commands.
Experimental batching uses options on `gen`, not a separate `batch` command.
The HTTP frontend runs as `python -m server`. It needs no separate compilation.

Build the standard C++ and HIP executables, without running them:

```sh
make binaries
```

This includes calibration probes, experimental probes, the resident suite, and device test programs.
Some probes require checkpoints or fixtures when run.
Their build targets are not a claim that each experiment passes on current hardware.

The three `isa_hc_*` probes require an archived HC reference.
The generic target excludes them. Their explicit file targets remain available.
Build them by hand with `ISA_HC_REFERENCE` set to the quoted path of a pinned `engine/layers/prefill_hc.h` (see `bench/isa_hc_ws.h`).

Use a specific target for a shorter build:

```sh
make build/bench/membw
make build/bench/suite/resident
make gpu-tests
```

Use `make -j2` only when memory permits parallel HIP compilation.
A full engine compile can take several minutes.

Override toolchain settings as needed:

```sh
make ROCM_PATH=/opt/rocm HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1151
make cpu-tests CXX=g++
```

`CPPFLAGS`, `CXXFLAGS`, `HIPFLAGS`, `LDFLAGS`, and `LDLIBS` accept standard build flags.
Generated dependency files track included headers.
CPU assertion tests retain assertions, even when `CXXFLAGS` contains `-DNDEBUG`.
Changing compiler flags alone does not invalidate existing binaries.
Use `make -B` when changing flags or compilers.

## CPU tests

Run the normal offline suite:

```sh
make -j2 test-cpu
```

`make test` is an alias for this target.
It runs C++ state, parser, sampling, and layout tests.
It also runs Python server, tooling, suite, and NumPy reference tests.
Tests that require a checkpoint, GPU binary, or upstream fixtures report skips when those inputs are absent.
They are not gate passes.

Run the same suite with sanitizers:

```sh
make -B -j2 test-cpu SANITIZERS='-fsanitize=address,undefined -fno-omit-frame-pointer'
```

Optional synthetic codec tests require the CPU build of PyTorch and safetensors:

```sh
python -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'safetensors>=0.4,<1'
make test-reference
```

These tests do not run the full model oracle.
For optional real tokenizer tests, set `CRUCIBLE_TEST_CKPT` to a complete checkpoint directory.

## Manual GPU gates

GPU gates stay outside GitHub Actions.
Read the gate procedure before running them.
The gate tool owns queue admission and oracle locking.
Do not wrap the whole gate tool in a second GPU lock.

```sh
make test-gpu GATE_ARGS="$CKPT --engine build/engine/crucible \
  --baseline-engine build/engine/main-baseline --baseline-name MAIN_COMMIT \
  --ids prompt.ids --output results/new-gates"
```

Use a distinct, named baseline binary. Do not compare the candidate with itself.
Pass `--prefill-rows` when the changed path includes GEMM prefill.
Keep the complete `report.json` and its fresh artifacts.

The CPU workflow cannot establish GPU correctness, throughput, or a clean HIP build.
