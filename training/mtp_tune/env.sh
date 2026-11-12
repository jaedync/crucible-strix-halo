# Runtime environment for torch on the test machine: the wheel's bundled HSA runtime crashes at
# queue creation on gfx1151 (core dump 2026-10-06), so preload the host runtime.
export LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1
export TRITON_CACHE_DIR=$HOME/crucible-work/refenv/triton_cache
export PYTORCH_ROCM_ARCH=gfx1151
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
unset HSA_OVERRIDE_GFX_VERSION
# Leak guard: diagnostic switches that write tensors or routing records to files must not reach a dump or a run on
# user data (the engine reads them from the environment).
unset CRUCIBLE_QSA_CAPTURE_AT CRUCIBLE_QSA_CAPTURE_DIR CRUCIBLE_QSA_CAPTURE_INDEX_ONLY CRUCIBLE_ROUTE_DUMP
PY=${PY:-$HOME/crucible-work/refenv/.venv/bin/python}
CK305=${CK305:-$HOME/models/exl3/Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5}
DATA=${DATA:-$HOME/crucible-work/mtp-data}
