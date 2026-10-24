# Public build entry point. GPU execution is always explicit.
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
ROCM_PATH ?= /opt/rocm
HIPCC ?= $(ROCM_PATH)/bin/hipcc
GPU_ARCH ?= gfx1151
PYTHON ?= python3
CXXFLAGS ?= -O2 -g
HIPFLAGS ?= -O3
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -pthread
SANITIZERS ?=

# These probes require an archived reference header, not the current engine.
PINNED_HC_SOURCES := bench/isa_hc_regs.hip bench/isa_hc_pitch.hip bench/isa_hc_bounds.hip
PINNED_HC_BINARIES := $(patsubst %.hip,build/%,$(PINNED_HC_SOURCES))
# bench/split2048.hip includes split2048_prompt.h, which tools/split2048_probe.py generates.
# The probe refuses an existing --output dir, so make cannot run it. Build that binary with the probe's --build path.
HIP_SOURCES := engine/engine.hip $(filter-out $(PINNED_HC_SOURCES) bench/split2048.hip,$(wildcard bench/*.hip bench/suite/*.hip tests/*.hip))
HIP_BINARIES := $(patsubst %.hip,build/%,$(filter-out engine/engine.hip,$(HIP_SOURCES))) build/engine/crucible
# The Python runner generates the driver header and supplies its CPU engine stubs.
CPU_SOURCES := $(filter-out tests/gen_cli_cpu_test.cpp tests/gen_jobs_cpu_test.cpp,$(wildcard tests/*.cpp bench/*.cpp bench/*.cc))
CPU_BINARIES := $(addprefix build/,$(basename $(CPU_SOURCES)))
CPU_TESTS := $(filter-out build/tests/draft_vocab_test build/tests/gen_cli_cpu_test build/tests/gen_jobs_cpu_test,$(patsubst %.cpp,build/%,$(wildcard tests/*_test.cpp)))
CPU_TESTS += build/bench/static_layout build/bench/test_migration_wait
# Header dependencies come from the -MMD files (DEPFILES, included at the end), per program. Do not list all
# headers as prerequisites: one header edit would rebuild every program (34 min to 3 h).
DEPFILES := $(addsuffix .d,$(HIP_BINARIES) $(PINNED_HC_BINARIES) $(CPU_BINARIES))

.PHONY: update rollback isa-check split-groups all engine binaries benchmarks cpu-tests gpu-tests test test-fast test-server test-cpu test-reference test-gpu help
all: engine
engine: build/engine/crucible
binaries: $(HIP_BINARIES) $(CPU_BINARIES)
benchmarks: $(filter build/bench/%,$(HIP_BINARIES) $(CPU_BINARIES))
cpu-tests: $(filter build/tests/%,$(CPU_BINARIES)) $(CPU_TESTS)
gpu-tests: $(filter build/tests/%,$(HIP_BINARIES))

# The bench provenance hash uses OpenSSL; production binaries do not link it.
build/tests/batch_spec_provenance_test build/bench/batch_spec_probe: LDLIBS += -lcrypto
# The probe stops at #error without its source digest. Use the digest that tools/build_batch_gates.py records for
# the same source set, so the probe and the gates report one digest. A caller can pin another digest in the environment.
# The shell runs only when a recipe needs the digest.
SPEC_SOURCE_DIGEST = $(strip $(or $(CRUCIBLE_BATCH_SPEC_SOURCE_SHA),$(shell $(PYTHON) -I -c 'import sys;sys.path.insert(0,"tools");import build_batch_gates as g;print(g.source_digest(g.source_hashes(g.ROOT)))')))
# An empty or malformed digest stops the build here. Otherwise the probe compiles with a bad provenance value.
SPEC_SOURCE_DIGEST_CHECKED = $(if $(shell printf '%s\n' '$(SPEC_SOURCE_DIGEST)' | grep -Ex '[0-9a-f]{64}'),$(SPEC_SOURCE_DIGEST),$(error CRUCIBLE_BATCH_SPEC_SOURCE_SHA must be 64 lowercase hex characters, got '$(SPEC_SOURCE_DIGEST)'))
build/bench/batch_spec_probe: CPPFLAGS += -DCRUCIBLE_BATCH_SPEC_SOURCE_SHA='"$(SPEC_SOURCE_DIGEST_CHECKED)"'

# The engine's device code (642 kernels) compiles in one clang process under plain hipcc. tools/devsplit.py runs
# the same driver steps but compiles the kernels in SPLIT_PARTS parts in parallel and caches the parts that an
# edit did not touch, with the same ISA. Exit code 2 means it could not plan the
# build, and the plain hipcc call runs. `make SPLIT=0` always runs hipcc.
SPLIT ?= 1
SPLIT_PARTS ?= 64
# The parts follow make's -j (default 1), so `make -j 1` runs one compiler at a time. Parallel compiles on the
# machine that runs a timing measurement can lower its GPU clock: build with `make -j 1` there, and with a higher -j
# on any other machine.
SPLIT_JOBS ?= $(or $(patsubst -j%,%,$(filter -j%,$(MAKEFLAGS))),1)
SPLIT_GROUPS ?= tools/devsplit_groups.tsv
SPLIT_WORK ?= build/engine/split
ENGINE_FLAGS = $(CPPFLAGS) $(HIPFLAGS) -std=c++20 --offload-arch=$(GPU_ARCH) --rocm-path=$(ROCM_PATH) -I$(ROCM_PATH)/include -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS)
ENGINE_HIPCC = $(HIPCC) $(ENGINE_FLAGS)
build/engine/crucible: engine/engine.hip Makefile tools/devsplit.py tools/devsplit_ir.py tools/devsplit_groups.tsv
	@mkdir -p $(@D)
ifeq ($(SPLIT),1)
	@$(PYTHON) -I tools/devsplit.py --out $@ --work $(SPLIT_WORK) --parts $(SPLIT_PARTS) -j $(SPLIT_JOBS) --groups $(SPLIT_GROUPS) --hipcc $(HIPCC) -- $(ENGINE_FLAGS); \
	rc=$$?; if [ $$rc -eq 2 ]; then echo "devsplit could not plan the build, using hipcc"; $(ENGINE_HIPCC) -o $@; else exit $$rc; fi
else
	$(ENGINE_HIPCC) -o $@
endif

# After a kernel change: `make isa-check` builds the engine with plain hipcc (about 20 min) and compares the ISA of
# every kernel with the split build. If a kernel differs, run
# `make split-groups CHANGED=<file>`, which writes the kernels that need one part to tools/devsplit_groups.tsv.
build/engine/crucible.mono: engine/engine.hip Makefile
	@mkdir -p $(@D)
	$(ENGINE_HIPCC) -o $@
isa-check: build/engine/crucible build/engine/crucible.mono
	$(PYTHON) -I tools/isa_diff.py same build/engine/crucible.mono build/engine/crucible
split-groups: engine/engine.hip
	@test -n "$(CHANGED)" || { echo 'Set CHANGED=<file with the names of the changed kernels>.' >&2; exit 2; }
	$(PYTHON) -I tools/devsplit.py --derive $(CHANGED) --groups $(SPLIT_GROUPS) --out build/engine/crucible --work build/engine/split-derive --hipcc $(HIPCC) -- $(ENGINE_FLAGS)

build/%: %.hip Makefile
	@mkdir -p $(@D)
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) -std=c++20 --offload-arch=$(GPU_ARCH) --rocm-path=$(ROCM_PATH) -I$(ROCM_PATH)/include -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS) -o $@

build/%: %.cpp Makefile
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SANITIZERS) -std=c++20 -UNDEBUG -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS) -o $@

build/%: %.cc Makefile
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SANITIZERS) -std=c++20 -UNDEBUG -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS) -o $@

# make update: move this checkout to the newest release, rebuild and self-test (CKPT=<model folder> adds the GPU
# test). make rollback: undo the last update. tools/update.sh has the rules. MAKEFLAGS is cleared so that the
# build inside the script does not borrow this make's job server.
update:
	@env -u MAKEFLAGS -u MFLAGS bash tools/update.sh
rollback:
	@env -u MAKEFLAGS -u MFLAGS bash tools/update.sh --rollback

test: test-cpu

# The edit loop: the C++ unit tests (incremental build, run in parallel) and the tests/ Python suite on 8 workers.
# server/tests (make test-server) is the slow part; run it when server/ changes. make test-cpu runs everything.
TEST_JOBS ?= 8
test-fast:
	$(MAKE) -j$(TEST_JOBS) cpu-tests
	tools/run_cpu_tests.sh $(TEST_JOBS) $(CPU_TESTS)
	build/tests/draft_vocab_test tests/data/draft/blocks65536.txt
	CXX="$(CXX)" $(PYTHON) -m pytest -q -p no:cacheprovider -n $(TEST_JOBS) tests

test-server:
	$(PYTHON) -m pytest -q -p no:cacheprovider -n $(TEST_JOBS) server/tests

test-cpu: cpu-tests
	@set -eu; for test in $(CPU_TESTS); do echo "TEST $$test"; "$$test"; done
	build/tests/draft_vocab_test tests/data/draft/blocks65536.txt
	CXX="$(CXX)" $(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) -m pytest -q -n $(TEST_JOBS) server/tests
	$(PYTHON) -m pytest -q -p no:cacheprovider tools/tests/test_isa_diff.py
	# Own run: the ngram16 tools import their sibling modules (common, codec) by plain name.
	$(PYTHON) -m pytest -q -p no:cacheprovider tools/ngram16
	$(PYTHON) -m unittest discover -s bench/suite -p 'test_*.py'
	$(PYTHON) -m pytest -q ref/test_exl3_ref.py ref/test_prefill.py ref/test_vision_ref.py

# PyTorch CPU is an optional dependency. No fp64 model oracle runs here.
test-reference:
	$(PYTHON) -m pytest -q ref/test_quality_formats.py ref/test_vision_hf.py

# The gate tool owns GPU scheduling, fresh artifacts, and the CPU oracle lock.
test-gpu: engine
	@test -n "$(GATE_ARGS)" || { echo 'Set GATE_ARGS; see python3 tools/gates.py --help.' >&2; exit 2; }
	$(PYTHON) tools/gates.py $(GATE_ARGS)

help:
	@printf '%s\n' 'make: build the engine' 'make update: move to the newest release (CKPT=<model folder> adds the GPU self-test); make rollback: undo it' 'make binaries: build standard C++ and HIP binaries' 'make test-fast: C++ unit tests and tests/ on 8 workers (the edit loop)' 'make test-server: server/tests on 8 workers' 'make test-cpu: run all offline CPU tests' 'make test-reference: run optional PyTorch CPU codec tests' 'make test-gpu GATE_ARGS="...": run shared manual gates' 'See docs/BUILD.md for dependencies and options.'

-include $(DEPFILES) build/engine/crucible.mono.d
