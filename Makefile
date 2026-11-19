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

.PHONY: FORCE install-hooks update rollback isa-check kernel-regs split-groups split-facts all engine binaries benchmarks cpu-tests gpu-tests test test-fast release-engine test-server test-service test-cpu test-reference test-gpu help
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
STATE_EPOCH ?= 1
# Structured output: xgrammar 0.2.9 (third_party/xgrammar, Apache-2.0, pinned tag v0.2.9)
# and the one file that includes it (engine/runtime/constraint_xgr.cc), as one host static library. Built with the host
# compiler, -O2, no -Werror and no LTO (the library's own CMake flags), so the hipcc link takes it as plain objects.
# Only the engine and the constraint tests link it; the other programs never reference it.
XGR_DIR := third_party/xgrammar
XGR_SOURCES := $(sort $(wildcard $(XGR_DIR)/cpp/*.cc $(XGR_DIR)/cpp/support/*.cc $(XGR_DIR)/cpp/converter_ext/*.cc))
# A literal -I path: tests/test_quoted_includes.py reads the repo include dirs from this file.
XGR_INCLUDES := -Ithird_party/xgrammar/include -isystem $(XGR_DIR)/3rdparty/picojson -isystem $(XGR_DIR)/3rdparty/dlpack/include
XGR_DEFINES := -DXGRAMMAR_ENABLE_CPPTRACE=0 -DXGRAMMAR_LOG_CUSTOMIZE=1
CONSTRAINT_LIB := build/third_party/libcrucible_constraint.a
# The settings of the library: the compiler that CXX runs, its version, XGR_DEFINES and XGR_INCLUDES. tools/release_engine.py
# lib-settings writes this file again only when its text changes. Every library object depends on it, so a build with other
# settings compiles the library again and does not link the objects of an older build. The engine build-info names the same values.
CONSTRAINT_SETTINGS := $(CONSTRAINT_LIB).settings
# The grammar compile helper (engine/runtime/constraint_helper.h): a host program from
# the library objects that the library then holds as bytes (constraint_helper_image.S), so the engine and the tests
# start it from memory and need no install step.
CONSTRAINT_CORE := $(patsubst %.cc,build/%.o,$(XGR_SOURCES)) build/engine/runtime/constraint_xgr.o
CONSTRAINT_HELPER := build/engine/crucible-grammar-helper
CONSTRAINT_OBJECTS := $(CONSTRAINT_CORE) build/engine/runtime/constraint_helper_image.o build/engine/runtime/constraint_helper_image_data.o
DEPFILES_CONSTRAINT := $(addsuffix .d,$(CONSTRAINT_OBJECTS))
# The tree keeps each upstream file as it is (the leak guard pins it). A local patch is a file under
# third_party/patches/ that tools/vendor_patch.py applies to a copy under build/patched/ (exact context, no fuzz). The
# copy compiles with the include dir of its upstream folder and the upstream path in __FILE__, so its object is the object
# of the patched file in place.
XGR_PATCHED := $(XGR_DIR)/cpp/json_schema_converter.cc
XGR_PATCH_json_schema_converter := third_party/patches/xgrammar/0001-bounded-string-control-characters.patch
$(patsubst %.cc,build/%.o,$(filter-out $(XGR_PATCHED),$(XGR_SOURCES))): build/%.o: %.cc Makefile
	@mkdir -p $(@D)
	$(CXX) -std=c++17 -O2 -fPIC $(XGR_DEFINES) $(XGR_INCLUDES) -MMD -MP -MF $@.d -c $< -o $@
build/patched/$(XGR_DIR)/cpp/json_schema_converter.cc: $(XGR_DIR)/cpp/json_schema_converter.cc $(XGR_PATCH_json_schema_converter) tools/vendor_patch.py
	python3 -I tools/vendor_patch.py $< $(XGR_PATCH_json_schema_converter) $@ --path $<
build/$(XGR_DIR)/cpp/json_schema_converter.o: build/patched/$(XGR_DIR)/cpp/json_schema_converter.cc Makefile
	@mkdir -p $(@D)
	$(CXX) -std=c++17 -O2 -fPIC $(XGR_DEFINES) -iquote $(XGR_DIR)/cpp -fmacro-prefix-map=build/patched/= $(XGR_INCLUDES) -MMD -MP -MF $@.d -c $< -o $@
# -isystem: the codec reads the mask cache of a compiled grammar (third_party/xgrammar/cpp/compiled_grammar_impl.h).
build/engine/runtime/constraint_xgr.o: engine/runtime/constraint_xgr.cc Makefile
	@mkdir -p $(@D)
	$(CXX) -std=c++20 -O2 -g -fPIC $(XGR_DEFINES) -isystem $(XGR_DIR)/cpp $(XGR_INCLUDES) -MMD -MP -MF $@.d -c $< -o $@
$(CONSTRAINT_HELPER): engine/runtime/constraint_helper_main.cc $(CONSTRAINT_CORE) Makefile
	@mkdir -p $(@D)
	$(CXX) -std=c++20 -O2 -s -MMD -MP -MF $@.d $< $(CONSTRAINT_CORE) -pthread -o $@
build/engine/runtime/constraint_helper_image.o: engine/runtime/constraint_helper_image.cc Makefile
	@mkdir -p $(@D)
	$(CXX) -std=c++20 -O2 -fPIC -MMD -MP -MF $@.d -c $< -o $@
build/engine/runtime/constraint_helper_image_data.o: engine/runtime/constraint_helper_image.S $(CONSTRAINT_HELPER) Makefile
	@mkdir -p $(@D)
	$(CXX) -c -DGRAMMAR_HELPER_PATH='"$(CONSTRAINT_HELPER)"' $< -o $@
$(CONSTRAINT_LIB): $(CONSTRAINT_OBJECTS)
	@mkdir -p $(@D)
	rm -f $@ && ar rcs $@ $^
FORCE:
$(CONSTRAINT_SETTINGS): FORCE
	@$(PYTHON) -I tools/release_engine.py lib-settings --out $@ --cxx='$(call shq,$(CXX))' --xgr-defines='$(call shq,$(XGR_DEFINES))' --xgr-includes='$(call shq,$(XGR_INCLUDES))'
$(CONSTRAINT_OBJECTS) $(CONSTRAINT_HELPER): $(CONSTRAINT_SETTINGS)
CONSTRAINT_TESTS := build/tests/constraint_test build/tests/constraint_property_test build/tests/constraint_session_test build/tests/constraint_guard_test build/tests/constraint_hostile_test build/tests/constraint_fuzz_runner build/tests/constraint_text_check
# The guard test compares the helper's grammars with an in-process compile, so it includes the library headers.
build/tests/constraint_guard_test build/tests/constraint_fuzz_runner: CPPFLAGS += $(XGR_DEFINES) $(XGR_INCLUDES)
$(CONSTRAINT_TESTS): $(CONSTRAINT_LIB)
# A host-linker option, not the archive path or a plain -l: hipcc compiles every input file after its -x hip as
# device source, and it unbundles a -l archive for the device link (tools/devsplit.py does not run that step).
CONSTRAINT_LDLIBS := -Wl,-L$(dir $(CONSTRAINT_LIB)),-lcrucible_constraint
$(CONSTRAINT_TESTS): LDLIBS += $(CONSTRAINT_LDLIBS)
# The stub trees of the build-recipe tests (tests/test_hipcc_series.py, test_hipcc_checked.py, test_sec_server.py) hold
# no constraint source, so they build no library. A real tree always holds it: engine.hip does not link without it.
ifneq ($(wildcard engine/runtime/constraint_xgr.cc),)
build/engine/crucible build/engine/crucible.mono: $(CONSTRAINT_LIB)
build/engine/crucible build/engine/crucible.mono: LDLIBS += $(CONSTRAINT_LDLIBS)
endif

ENGINE_FLAGS = $(CPPFLAGS) $(HIPFLAGS) -std=c++20 --offload-arch=$(GPU_ARCH) --rocm-path=$(ROCM_PATH) -I$(ROCM_PATH)/include -MMD -MP -MF $@.d $< $(LDFLAGS) $(LDLIBS)
# Every hipcc recipe runs through tools/hipcc_checked.sh: hipcc exits 0 when its clang child is killed (out of memory),
# so the helper removes the old output first and fails when the compiler leaves none.
HIPCC_CHECKED = bash tools/hipcc_checked.sh $@
ENGINE_HIPCC = $(HIPCC_CHECKED) $(HIPCC) $(ENGINE_FLAGS)
# The engine recipe also writes build/engine/crucible.build-info (commit, clean tree, hash of the binary); release-engine reads it.
ENGINE_STAMP = $(PYTHON) -I tools/release_engine.py stamp
# The end of the stamp also records the compiler, its version, the compile settings, ROCM_PATH and the settings of the constraint
# library (CXX, its path and version, XGR_DEFINES, XGR_INCLUDES: the values of CONSTRAINT_SETTINGS); it reads the hipcc environment
# variables (HIPCC_COMPILE_FLAGS_APPEND, HIP_CLANG_PATH and the like) itself. A release refuses non-default values and an unknown compiler version.
shq = $(subst ','\'',$(1))
ENGINE_STAMP_END = $(ENGINE_STAMP) end --binary $@ --hipcc='$(call shq,$(HIPCC))' --gpu-arch='$(call shq,$(GPU_ARCH))' --hipflags='$(call shq,$(HIPFLAGS))' --cppflags='$(call shq,$(CPPFLAGS))' --ldflags='$(call shq,$(LDFLAGS))' --ldlibs='$(call shq,$(LDLIBS))' --rocm-path='$(call shq,$(ROCM_PATH))' \
	--cxx='$(call shq,$(CXX))' --xgr-defines='$(call shq,$(XGR_DEFINES))' --xgr-includes='$(call shq,$(XGR_INCLUDES))'
# ALLOW_ROCM10=1 has no effect. v0.37.2 stopped a ROCm 10.x build without it (the v0.37.0 engine spilled on clang 24); the
# source fix cb97f418 removed that spill, and the v0.37.2 CHANGELOG still names the word, so make accepts it.
ALLOW_ROCM10 ?= 0
# With clang 24 (ROCm 10.x) the engine recipe builds plain hipcc (SPLIT=0): the gated 10.1 binary is that build, and the
# clang 24 split build gives other ISA in 92 functions. The split tables below stay for that gate.
HIPCC_SERIES = $(PYTHON) -I tools/hipcc_series.py
# ROCm 10.1 (AMD clang 24, the new offload driver): devsplit runs the prelink cc1 and the LTO link of each part, and
# reads two tables next to SPLIT_FACTS that carry the compiler's name (tools/devsplit_facts.clang-24.tsv for the
# prelink, .clang-24.lto.tsv for the LTO pipeline). `make split-facts` with that compiler writes both (about 15 min
# on the test machine: one whole-module prelink). SPLIT_CHECK checks the prelink facts only; the LTO facts are checked by
# `make isa-check`.
build/engine/crucible: engine/engine.hip Makefile tools/devsplit.py tools/devsplit_ir.py tools/devsplit_groups.tsv tools/devsplit_facts.tsv $(wildcard tools/devsplit_facts.*.tsv) tools/hipcc_series.py
	@mkdir -p $(@D)
	@rm -f $@
	@$(ENGINE_STAMP) begin --binary $@
ifeq ($(SPLIT),1)
	@if $(HIPCC_SERIES) clang24 --hipcc='$(call shq,$(HIPCC))'; then echo "hipcc is clang 24 (ROCm 10.x): the gated build is plain hipcc, using it (SPLIT=0)"; \
	$(ENGINE_HIPCC) -o $@; else \
	$(PYTHON) -I tools/devsplit.py --out $@ --work $(SPLIT_WORK) --parts $(SPLIT_PARTS) -j $(SPLIT_JOBS) --groups $(SPLIT_GROUPS) --facts $(SPLIT_FACTS) $(if $(filter 1,$(SPLIT_CHECK)),--check) --hipcc $(HIPCC) -- $(ENGINE_FLAGS); \
	rc=$$?; if [ $$rc -eq 2 ]; then echo "devsplit could not plan the build, using hipcc"; $(ENGINE_HIPCC) -o $@; else exit $$rc; fi; fi
else
	$(ENGINE_HIPCC) -o $@
endif
# Stamp the state epoch into the binary (tools/state_epoch.py, engine/runtime/state_epoch.h): the SSD conversation cache
# keeps its states over a new binary that has the same epoch. A failed stamp removes the binary, so an unstamped one
# does not stay behind. STATE_EPOCH=0 skips the stamp (the binary then uses its own sha256 as the epoch: the old rule).
ifneq ($(STATE_EPOCH),0)
	@ROCM_PATH='$(call shq,$(ROCM_PATH))' HIPCC='$(call shq,$(HIPCC))' $(PYTHON) -I tools/state_epoch.py stamp $@ || { rm -f $@; exit 1; }
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
# The public CI's shellcheck (ubuntu-24.04 apt 0.9.0) flags notes that newer versions do not. When the pinned release binary
# is installed (koalaman v0.9.0 linux.x86_64, tarball sha256 700324c6...), tests/test_install_sh.py must run it, not skip.
SHELLCHECK_CI ?= $(wildcard $(HOME)/.local/share/shellcheck-0.9.0/shellcheck)
test-fast:
	$(MAKE) -j$(TEST_JOBS) cpu-tests
	tools/run_cpu_tests.sh $(TEST_JOBS) $(CPU_TESTS)
	build/tests/draft_vocab_test tests/data/draft/blocks65536.txt
	CRUCIBLE_SHELLCHECK_CI="$(SHELLCHECK_CI)" CXX="$(CXX)" $(PYTHON) -m pytest -q -rs -p no:cacheprovider -n $(TEST_JOBS) tests $(SERVER_FAST_TESTS)


test-server:
	$(PYTHON) -m pytest -q -p no:cacheprovider -n $(TEST_JOBS) server/tests

# The service tests need a systemd user manager (systemd-run --user), so a sandboxed build worker skips them: run this target on a
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
