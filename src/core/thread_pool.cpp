#include "thread_pool.h"

#include <sys/types.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "kern/kernel_define.h"
#include "utils.h"

namespace inferllm {
ThreadPool::ThreadPool(uint32_t threads_number)
    : threads_number_(threads_number), stop_(false), active_(false) {
  if (threads_number < 1) {
    threads_number_ = 1;
  }
  if (threads_number > 1) {
    auto system_thread_count = std::thread::hardware_concurrency();
    if (threads_number > system_thread_count) {
      INFER_LOG(
          "The number of threads is bigger than number of "
          "physical cpu cores, got: %d core_number: %d",
          system_thread_count, this->threads_number());
    }
    for (uint32_t i = 0; i < threads_number_ - 1; i++) {
      workers_.push_back(std::make_shared<Worker>([this, i]() {
        while (!stop_) {
          while (active_) {
            //! if the thread should work
            if (workers_[i]->work_flag.load(std::memory_order_acquire)) {
              task_(TaskId{i * task_per_thread_,
                           std::min((i + 1) * task_per_thread_, task_number_),
                           i});
              //! Flag worker is finished
              workers_[i]->work_flag.store(false, std::memory_order_release);
            }
            //! Wait next task coming
            for (int it = 0; it < WORKER_ACTIVE_WAIT; it++) {
              if (workers_[i]->work_flag.load(std::memory_order_acquire)) {
                break;
              }
              if (it < ACTIVE_WAIT_PAUSE_LIMIT || (it & 1)) {
                INFER_PAUSE(16);  // Spin lock's CPU-level yield
              } else {
                // Spin lock's OS-level yield
                std::this_thread::yield();
              }
            }
          }
          {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!stop_ && !active_) {
              cv_.wait(lock, [this] { return stop_ || active_; });
            }
          }
        }
      }));
    }
  }
}

void ThreadPool::add_task(const MultiThreadingTask& task,
                          uint32_t task_number) {
  if (threads_number_ == 1 || task_number == 1) {
    task({0, task_number, threads_number_ - 1});
    return;
  } else {
    active();
    INFER_ASSERT(active_, "thread pool is not actived.");
    task_number_ = task_number;

    task_per_thread_ = (task_number_ + threads_number_ - 1) / threads_number_;
    task_ = std::move(task);
    for (uint32_t i = 0; i < threads_number_ - 1; i++) {
      workers_[i]->work_flag.store(true, std::memory_order_release);
    }
    uint32_t start = (threads_number_ - 1) * task_per_thread_;
    task_({start, task_number, threads_number_ - 1});
    sync();
  }
}

inline void ThreadPool::sync() {
  bool no_finished = true;
  uint32_t no_finished_id = 0;
  while (no_finished) {
    no_finished = false;
    for (uint32_t i = no_finished_id; i < threads_number_ - 1; ++i) {
      if (workers_[i]->work_flag.load(std::memory_order_acquire)) {
        no_finished = true;
        no_finished_id = i;
        break;
      }
    }
    if (no_finished) {
      for (int it = 0; it < MAIN_THREAD_ACTIVE_WAIT; it++) {
        if (!workers_[no_finished_id]->work_flag.load(
                std::memory_order_acquire)) {
          break;
        }
        if ((it < ACTIVE_WAIT_PAUSE_LIMIT || (it & 1))) {
          INFER_PAUSE(16);
        } else {
          std::this_thread::yield();
        }
      }
    }
  }
}

inline void ThreadPool::active() {
  if (!active_) {
    std::lock_guard<std::mutex> lock(mutex_);
    active_ = true;
    cv_.notify_all();
  }
}

void ThreadPool::deactive() {
  std::lock_guard<std::mutex> lock(mutex_);
  active_ = false;
}

ThreadPool::~ThreadPool() {
  std::lock_guard<std::mutex> lock(mutex_);
  stop_ = true;
  active_ = false;
  cv_.notify_all();
}
}  // namespace inferllm