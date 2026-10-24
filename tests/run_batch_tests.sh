#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
compiler=${CXX:-g++}
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
for test in batch_contract_test batch_options_test batch_transfer_test batch_snapshot_test batch_conversation_cache_test \
            packed_state_test batch_session_test end_resume_test batch_admission_test batched_serve_protocol_test \
            batched_serve_settings_test wide_projection_layout_test wide_dense_validate_test batch_spec_contract_test \
            state_disk_test batch_state_disk_test; do
  "$compiler" -std=c++20 -O1 -g -fsanitize=address,undefined "tests/$test.cpp" -pthread -o "$build/$test"
  "$build/$test"
done
"$compiler" -std=c++20 -O1 -g -fsanitize=address,undefined tests/batch_spec_provenance_test.cpp -lcrypto -o "$build/batch_spec_provenance_test"
"$build/batch_spec_provenance_test"
"$compiler" -std=c++20 -O3 -ffast-math tests/wide_dense_validate_test.cpp -o "$build/wide_dense_fast"
"$build/wide_dense_fast"
python3 -m unittest tests.test_interleave_check tests.test_batch_union tests.test_batch_gate_report tests.test_build_batch_gates tests.test_batched_spec_model tests.test_batch_spec_compare tests.test_batch_spec_pair
