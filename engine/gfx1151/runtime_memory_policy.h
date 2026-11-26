#pragma once
#include <cstdlib>

namespace crucible {
inline bool SetRuntimeMemoryDefaults() {
  // Migrating an SVM range stops every KFD queue, including decode queues.
  if (setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 0) != 0 ||
      setenv("HSA_USE_SVM", "0", 0) != 0) return false;
  // CLR measures this threshold in MiB. Stage pageable uploads, rather than
  // pin movable pages. Keep explicit user settings for controlled probes.
  return setenv("GPU_PINNED_MIN_XFER_SIZE", "65536", 0) == 0;
}

// One KFD allocation for each hipMalloc, rounded to pages, in place of ROCr's fragment allocator, which puts each
// hipMalloc of 2 MiB or less in its own 2 MiB block. A model of many small weights needs it: on the test machine (2026-11-19)
// the default commits 1.769 B for each B requested, the switch 1.002. ROCr reads the
// variable once, at the first hsa_init, so call this before the first HIP call of the process. An explicit value
// stays (controlled probes). The caller selects the model: the Qwen default path does not call it.
inline bool DisableFragmentAllocator() { return setenv("HSA_DISABLE_FRAGMENT_ALLOCATOR", "1", 0) == 0; }
}  // namespace crucible
