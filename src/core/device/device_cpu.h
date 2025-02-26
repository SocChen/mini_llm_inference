#pragma once

#include <cstring>

#include "core/thread_pool.h"
#include "device.h"

namespace inferllm {
class CPUDevice : public Device {
 public:
  CPUDevice();
  CPUDevice(const CPUDevice&) = delete;
  CPUDevice& operator=(const CPUDevice&) = delete;
  CPUDevice(CPUDevice&&) noexcept = delete;
  CPUDevice& operator=(CPUDevice&&) noexcept = delete;

  ~CPUDevice() override;

  void* allocate(size_t len) override;
  void* allocate_host(size_t len) override;

  void free_device(void* ptr) override;
  void free_host(void* ptr) override;

  void deactive() override;

  void host2device_copy(void* device, const void* host, size_t size,
                        bool async) override {
    memcpy(device, host, size);
  }

  void device2host_copy(void* host, const void* device, size_t size,
                        bool async) override {
    memcpy(host, device, size);
  }

  void device2device_copy(void* dst, const void* src, size_t size,
                          bool async) override {
    memcpy(dst, src, size);
  }
  void sync() override {}

 private:
  std::unique_ptr<ThreadPool> thead_pool_;
};
}  // namespace inferllm