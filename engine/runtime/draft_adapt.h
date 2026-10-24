// Retain selected EXL3 blocks and optionally add blocks observed in the exact prefix.
#pragma once

#include <vector>

#include "../exl3/front_order.h"

namespace crucible {

class BlockDraftHead {
 public:
  BlockDraftHead(const Lin& source, const std::vector<int>& blocks, bool adaptive)
      : source_(source), head_(source), adaptive_(adaptive), seen_(source.O / 128, false) {
    head_.front_s = 0;
    const int capacity = adaptive ? source.O : (int)blocks.size() * 128;
    HIP_OK(hipMalloc(&head_.tiles, (size_t)capacity * source.I * source.K / 8));
    HIP_OK(hipMalloc(&front_.tiles, (size_t)capacity * source.I * source.K / 8));
    HIP_OK(hipMalloc(&head_.svh, (size_t)capacity * sizeof(__half)));
    HIP_OK(hipMalloc(&mapping_, (size_t)capacity / 128 * sizeof(int)));
    head_.O = 0;
    for (int block : blocks) Append(block);
    UploadMapping();
  }

  ~BlockDraftHead() {
    HIP_OK(hipFree(head_.tiles));
    HIP_OK(hipFree(front_.tiles));
    HIP_OK(hipFree(head_.svh));
    HIP_OK(hipFree(mapping_));
  }

  bool Observe(const int* tokens, int count) {
    if (!adaptive_) return false;
    const int before = head_.O;
    for (int i = 0; i < count; ++i) {
      if (tokens[i] < 0 || tokens[i] >= source_.O) std::abort();
      const int block = tokens[i] / 128;
      if (!seen_[block]) Append(block);
    }
    if (head_.O == before) return false;
    UploadMapping();
    return true;
  }

  const Lin& Weights() const { return head_; }
  // The same head in front tile order for split 1, so that the decode ring
  // reads it (verify-rows: LinPolicyKernel read it at 198 GB/s). Every column
  // keeps its tile order, so the logits stay bitwise equal.
  const Lin& Front() const { return front_; }
  int* Mapping() const { return mapping_; }

 private:
  void Append(int block) {
    const size_t words = (size_t)128 * source_.I * source_.K / 32;
    const size_t index = blocks_.size();
    // The draft head keeps engine order; a front-order source is gathered.
    if (source_.front_s)
      FrontColumnsKernel<<<256, 256>>>(source_.tiles, head_.tiles + index * words, source_.I / 16,
                                       source_.O / 16, 8 * source_.K, source_.front_s, block);
    else
      HIP_OK(hipMemcpyAsync(head_.tiles + index * words, source_.tiles + block * words,
                            words * sizeof(uint32_t), hipMemcpyDeviceToDevice, 0));
    HIP_OK(hipMemcpyAsync(head_.svh + index * 128, source_.svh + block * 128,
                          128 * sizeof(__half), hipMemcpyDeviceToDevice, 0));
    blocks_.push_back(block);
    seen_[block] = true;
    head_.O += 128;
  }

  void UploadMapping() {
    HIP_OK(hipMemcpy(mapping_, blocks_.data(), blocks_.size() * sizeof(int), hipMemcpyHostToDevice));
    // A new block changes the chunk stride of every item, so the whole front copy is rebuilt.
    uint32_t* const tiles = front_.tiles;
    front_ = head_;
    front_.tiles = tiles;
    front_.front_s = 1;
    EngineToFrontKernel<<<256, 256>>>(head_.tiles, front_.tiles, head_.I / 16, head_.O / 16, 8 * head_.K);
    HIP_OK(hipGetLastError());
  }

  Lin source_, head_, front_;
  bool adaptive_;
  std::vector<int> blocks_;
  std::vector<bool> seen_;
  int* mapping_;
};

}  // namespace crucible
