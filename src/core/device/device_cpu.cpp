#include "device_cpu.h"

#include <cstddef>
#include <cstdlib>

#include "utils.h"

namespace inferllm {
void* CPUDevice::allocate(size_t len) {
  auto it = this->free_memory_.lower_bound(len);
  void* ptr = nullptr;
  if (it != free_memory_.end() && it->second.size() > 0) {
    // find the memory block larger than len from free memory
    ptr = it->second.back();
    it->second.pop_back();
    if (it->second.size() < 1) {
      free_memory_.erase(it);
    }
  } else {
    // allocate new memory
    ptr = aligned_alloc(len);
    alloc_memory_[ptr] = len;
  }
  return ptr;
}

void* CPUDevice::allocate_host(size_t len) { return aligned_alloc(len); }

void CPUDevice::free_host(void* ptr) { aligned_free(ptr); }

void CPUDevice::free_device(void* ptr) {
  INFER_ASSERT(alloc_memory_.count(ptr) == 1,
               "memory is not allocated by the DeviceCPU.");
  auto len = alloc_memory_[ptr];
  free_memory_[len].push_back(ptr);
}

CPUDevice::~CPUDevice() {
  for (auto it : free_memory_) {
    for (auto ptr : it.second) {
      INFER_ASSERT(alloc_memory_.count(ptr) == 1,
                   "memory is not allocated by the DeviceCPU.");
      aligned_free(ptr);
    }
  }
}
}  // namespace inferllm