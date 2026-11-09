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

.PHONY: update rollback isa-check split-groups split-facts all engine binaries benchmarks cpu-tests gpu-tests test test-fast release-engine test-server test-service test-cpu test-reference test-gpu help
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
# Argument facts of the one-module build: each part gets a stub kernel that passes
# them, so IPSCCP finds the same facts in a part as in one module. SPLIT_CHECK=1 (default) also runs IPSCCP on the
# whole module next to the part compiles and stops the build when a part sees other facts (`make split-facts`
# fixes that). SPLIT_CHECK=0 skips the check (8 to 14 s on the test machine with -j 8) in an edit loop.
SPLIT_FACTS ?= tools/devsplit_facts.tsv
SPLIT_CHECK ?= 1
SPLIT_WORK ?= build/engine/split
ENGINE_FLAGS = $(CPPFLAGS) $(HIPFLAGS) -std=c++20 --offload-arch=$(GPU_ARCH) --rocm-path=$(ROCM_PATH) -I$(ROCM_PATH)/include -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS)
# Every hipcc recipe runs through tools/hipcc_checked.sh: hipcc exits 0 when its clang child is killed (out of memory),
# so the helper removes the old output first and fails when the compiler leaves none.
HIPCC_CHECKED = bash tools/hipcc_checked.sh $@
ENGINE_HIPCC = $(HIPCC_CHECKED) $(HIPCC) $(ENGINE_FLAGS)
# The engine recipe also writes build/engine/crucible.build-info (commit, clean tree, hash of the binary); release-engine reads it.
ENGINE_STAMP = $(PYTHON) -I tools/release_engine.py stamp
# The end of the stamp also records the compiler, its version, the compile settings and ROCM_PATH; it reads the hipcc environment
# variables (HIPCC_COMPILE_FLAGS_APPEND, HIP_CLANG_PATH and the like) itself. A release refuses non-default values and an unknown compiler version.
shq = $(subst ','\'',$(1))
ENGINE_STAMP_END = $(ENGINE_STAMP) end --binary $@ --hipcc='$(call shq,$(HIPCC))' --gpu-arch='$(call shq,$(GPU_ARCH))' --hipflags='$(call shq,$(HIPFLAGS))' --cppflags='$(call shq,$(CPPFLAGS))' --ldflags='$(call shq,$(LDFLAGS))' --ldlibs='$(call shq,$(LDLIBS))' --rocm-path='$(call shq,$(ROCM_PATH))'
build/engine/crucible: engine/engine.hip Makefile tools/devsplit.py tools/devsplit_ir.py tools/devsplit_groups.tsv tools/devsplit_facts.tsv
	@mkdir -p $(@D)
	@rm -f $@
	@$(ENGINE_STAMP) begin --binary $@
ifeq ($(SPLIT),1)
	@$(PYTHON) -I tools/devsplit.py --out $@ --work $(SPLIT_WORK) --parts $(SPLIT_PARTS) -j $(SPLIT_JOBS) --groups $(SPLIT_GROUPS) --facts $(SPLIT_FACTS) $(if $(filter 1,$(SPLIT_CHECK)),--check) --hipcc $(HIPCC) -- $(ENGINE_FLAGS); \
	rc=$$?; if [ $$rc -eq 2 ]; then echo "devsplit could not plan the build, using hipcc"; $(ENGINE_HIPCC) -o $@; else exit $$rc; fi
else
	$(ENGINE_HIPCC) -o $@
endif
	@$(ENGINE_STAMP_END)

# After a kernel change: `make isa-check` builds the engine with plain hipcc (about 20 min) and compares the ISA of
# every kernel with the split build. If a kernel differs, run
# `make split-groups CHANGED=<file>`, which writes the kernels that need one part to tools/devsplit_groups.tsv.
# `make split-facts` writes tools/devsplit_facts.tsv (the test machine: about 40 s with -j 8, 110 s with -j 1): run it when the
# build stops with "argument facts drift", and commit the file.
build/engine/crucible.mono: engine/engine.hip Makefile
	@mkdir -p $(@D)
	$(ENGINE_HIPCC) -o $@
isa-check: build/engine/crucible build/engine/crucible.mono
	$(PYTHON) -I tools/isa_diff.py same build/engine/crucible.mono build/engine/crucible
split-groups: engine/engine.hip
	@test -n "$(CHANGED)" || { echo 'Set CHANGED=<file with the names of the changed kernels>.' >&2; exit 2; }
	$(PYTHON) -I tools/devsplit.py --derive $(CHANGED) --groups $(SPLIT_GROUPS) --out build/engine/crucible --work build/engine/split-derive --hipcc $(HIPCC) -- $(ENGINE_FLAGS)
split-facts: engine/engine.hip
	$(PYTHON) -I tools/devsplit.py --derive-facts --facts $(SPLIT_FACTS) -j $(SPLIT_JOBS) --out build/engine/crucible --work build/engine/split-derive --hipcc $(HIPCC) -- $(ENGINE_FLAGS)

build/%: %.hip Makefile
	@mkdir -p $(@D)
	$(HIPCC_CHECKED) $(HIPCC) $(CPPFLAGS) $(HIPFLAGS) -std=c++20 --offload-arch=$(GPU_ARCH) --rocm-path=$(ROCM_PATH) -I$(ROCM_PATH)/include -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS) -o $@

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

# make release-engine: build the engine for the release tag in VERSION and write the three release files to dist/.
# Run it on a commit that carries the tag vVERSION, with a clean tree. RELEASE_LIBDIRS names folders that hold the
# libraries ROCm needs when ROCM_PATH does not (for example fmt). The release refuses an engine whose build-info (written by
# the engine recipe) shows another commit, a dirty tree or another hash: delete build/engine/crucible and run it again. Upload the files of dist/ to the release of the tag.
RELEASE_TAG = v$(shell cat VERSION)
release-engine: engine
	@test -z "$$(git status --porcelain --untracked-files=no)" || { echo 'release-engine: the tree has local changes. Commit them first.' >&2; exit 1; }
	@test "$$(git rev-parse '$(RELEASE_TAG)^{commit}' 2>/dev/null)" = "$$(git rev-parse HEAD)" || { echo 'release-engine: HEAD is not the commit of tag $(RELEASE_TAG). Tag the release commit first.' >&2; exit 1; }
	$(PYTHON) tools/release_engine.py release --binary build/engine/crucible --rocm-path $(ROCM_PATH) $(foreach d,$(RELEASE_LIBDIRS),--libdir $(d)) --tag $(RELEASE_TAG) --commit "$$(git rev-parse HEAD)" --out dist/release-$(RELEASE_TAG)

test: test-cpu

# The edit loop: the C++ unit tests (incremental build, run in parallel) and the tests/ Python suite on 8 workers.
# server/tests (make test-server) is the slow part; run it when server/ changes. make test-cpu runs everything.
TEST_JOBS ?= 8
# The server tests that are cheap (under a second) and check a documented line: part of test-fast. The rest is make test-server.
SERVER_FAST_TESTS = server/tests/test_docker_hosts.py
test-fast:
	$(MAKE) -j$(TEST_JOBS) cpu-tests
	tools/run_cpu_tests.sh $(TEST_JOBS) $(CPU_TESTS)
	build/tests/draft_vocab_test tests/data/draft/blocks65536.txt
	CXX="$(CXX)" $(PYTHON) -m pytest -q -p no:cacheprovider -n $(TEST_JOBS) tests $(ABAB_TESTS) $(SERVER_FAST_TESTS)

test-server:
	$(PYTHON) -m pytest -q -p no:cacheprovider -n $(TEST_JOBS) server/tests

# The service tests need a systemd user manager (systemd-run --user), so the bwrap workers skip them: run this target on a
# normal user session. tests/service holds the fault tests F1 to F12 (CRUCIBLE_SERVICE_TESTS=1 turns them on). The last line is the
# CPU dry run of the night driver (design s5 C): every step N1 to N11 against the fake engine, with scaled times. It writes
# ~/.cache/crucible-ft/work/report.json and removes its own unit at the end.
test-service:
	$(if $(wildcard tests/service),CRUCIBLE_SERVICE_TESTS=1 $(PYTHON) -m pytest -q -p no:cacheprovider tests/service)
	$(PYTHON) tools/service_faulttest.py --fake

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
	@printf '%s\n' 'make: build the engine' 'make update: move to the newest release (CKPT=<model folder> adds the GPU self-test); make rollback: undo it' 'make binaries: build standard C++ and HIP binaries' 'make release-engine: build the engine for the release tag and write the release files to dist/' 'make test-fast: C++ unit tests and tests/ on 8 workers (the edit loop)' 'make test-server: server/tests on 8 workers' 'make test-service: systemd fault tests and the dry run of the night driver (needs a user session)' 'make test-cpu: run all offline CPU tests' 'make test-reference: run optional PyTorch CPU codec tests' 'make test-gpu GATE_ARGS="...": run shared manual gates' 'See docs/BUILD.md for dependencies and options.'

-include $(DEPFILES) build/engine/crucible.mono.d
