#pragma once

#include <map>
#include <memory>
#include <vector>
namespace inferllm {

class Device {
 public:
  Device() = default;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;

  virtual ~Device() = default;

  virtual void* allocate(size_t len) = 0;

  virtual void* allocate_host(size_t len) = 0;

  virtual void free_device(void* ptr) = 0;

  virtual void free_host(void* ptr) = 0;

  //   Kernel* kernel() { return m_kernel.get(); }

  //   KernelType type() { return m_kernel->m_kernel_type; };

  virtual void* aligned_alloc(size_t size);

  virtual void aligned_free(void* ptr);

  virtual void deactive() {}

  virtual void host2device_copy(void* device, const void* host, size_t size,
                                bool async) = 0;

  virtual void device2host_copy(void* host, const void* device, size_t size,
                                bool async) = 0;

  virtual void device2device_copy(void* dst, const void* src, size_t size,
                                  bool async) = 0;

  virtual void sync() = 0;

 protected:
  // std::unique_ptr<Kernel> m_kernel;
  std::map<void*, size_t> alloc_memory_;
  std::map<size_t, std::vector<void*>> free_memory_;
};
}  // namespace inferllm