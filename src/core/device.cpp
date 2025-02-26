#include "device.h"

#include "utils.h"

#define ALIGN_SIZE (32)

namespace inferllm {
void* Device::aligned_alloc(size_t size) {
  void* ptr = nullptr;
  auto err = posix_memalign(&ptr, ALIGN_SIZE, size);
  INFER_ASSERT(!err, "failed to malloc.");
  return ptr;
}

void Device::aligned_free(void* ptr) { free(ptr); }
}  // namespace inferllm