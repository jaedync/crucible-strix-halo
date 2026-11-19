# third_party: charter

Code that another project owns, copied into this repository at a fixed version. Engine code never includes it
directly: one wrapper per library is the only file that includes its headers. The Makefile builds each library into
its own archive under `build/third_party/`. The engine and the tests link that archive with a host-linker option,
not with hipcc's `-l` (hipcc unbundles a `-l` archive for the device link, and `tools/devsplit.py` does not run that
step).

## xgrammar

| Item | Value |
|---|---|
| Upstream | https://github.com/mlc-ai/xgrammar |
| Version | v0.2.9, commit `8f9a1279` |
| License | Apache-2.0 (`xgrammar/LICENSE`, `xgrammar/NOTICE`); see the top-level `NOTICE` |
| Use | Structured output: JSON schema and structural tag grammars, token masks |
| Wrapper | `engine/runtime/constraint_xgr.cc` (the only includer); interface `engine/runtime/constraint_vocab.h` |
| Archive | `build/third_party/libcrucible_constraint.a` (C++17, `-O2 -fPIC`, no cpptrace, custom logging) |

Copied: `cpp/` without `cpp/tvm_ffi/` and the CMake files, `include/`, `3rdparty/picojson/picojson.h`
(BSD-2-Clause, license in the header) and `3rdparty/dlpack/include/dlpack/dlpack.h` with `3rdparty/dlpack/LICENSE`
(Apache-2.0). Nothing else of the upstream tree is here (no Python package, no tests, no CUDA kernels).

Every file under `third_party/xgrammar/` is the upstream file of commit `8f9a1279` byte for byte. Local patches are
separate files under `third_party/patches/xgrammar/`; the build applies each one to a copy under `build/patched/`
with `tools/vendor_patch.py` (exact context, no offset, no fuzz: a patch that does not fit stops the build). The copy
compiles with `-iquote` of its upstream folder and `-fmacro-prefix-map=build/patched/=`, so its object is the object of
the patched file in place (`cmp`: identical).

1. `0001-bounded-string-control-characters.patch` on `cpp/json_schema_converter.cc`: a length-bounded string
   (`maxLength` or `minLength`) used a character class that allowed raw control characters (U+0000 to U+001F), so
   `json.loads` refused the output. The bounded class now excludes them, as the unbounded class does. Test:
   `tests/constraint_test.cpp` (a raw TAB is refused).

Logging: the wrapper defines `XGRAMMAR_LOG_CUSTOMIZE=1` and supplies the log hooks: warnings are dropped (no schema
text reaches a log) and a fatal check becomes `std::runtime_error`, which the wrapper turns into a fixed error code.

Leak guard: the vendored class passes a file here only when its sha256 and its git blob id are
the ones that the pinned upstream tree names at its path. A patched file would not pass, so no file here is patched.

To update: copy the same file set from the new tag, write the leak guard pin again (from a git clone of the tag and of the
dlpack submodule commit; no code of the download runs), make the patches fit the new
files, run `make test-fast`, and record the new version here and in `engine/runtime/constraint_xgr.cc`
(`ConstraintLibraryVersion`).
