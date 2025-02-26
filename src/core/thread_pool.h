#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "kern/kernel_define.h"
namespace inferllm {
struct Worker {
 public:
  explicit Worker(std::function<void()>&& run) : thread{std::move(run)} {}
  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;
  Worker(Worker&&) noexcept = delete;
  Worker& operator=(Worker&&) noexcept = delete;

  ~Worker() { thread.join(); }
  //! Worker thread
  std::thread thread;
  //! Indicate whether the Worker thread need run
  std::atomic<bool> work_flag{false};
};

class ThreadPool {
 public:
  //! The number of iterations < main thread yeild resource>
  static constexpr int MAIN_THREAD_ACTIVE_WAIT = 10000;
  //! The number of iterations < worker thread yeild resource>
  static constexpr int WORKER_ACTIVE_WAIT = 2000;
  //! The number of iterations <pause>
  static constexpr int ACTIVE_WAIT_PAUSE_LIMIT = 16;

 public:
  explicit ThreadPool(uint32_t threads_number);
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) noexcept = delete;
  ThreadPool& operator=(ThreadPool&&) noexcept = delete;

  void add_task(const MultiThreadingTask& task, uint32_t nr_task);

  inline void sync();
  inline void active();
  void deactive();
  uint32_t threads_number() const { return threads_number_; }
  ~ThreadPool();

 private:
  uint32_t threads_number_ = 1;
  uint32_t task_number_ = 0;
  uint32_t task_per_thread_ = 0;
  std::atomic_bool stop_{false};
  std::atomic_bool active_{false};

  MultiThreadingTask task_;
  std::vector<std::shared_ptr<Worker>> workers_;
  std::condition_variable cv_;
  std::mutex mutex_;
};
}  // namespace inferllm