#pragma once

#include <cstdint>
#include <functional>
namespace inferllm {

struct TaskId {
  uint32_t start;
  uint32_t end;
  uint32_t thread_id;
};

using MultiThreadingTask = std::function<void(TaskId)>;
using TaskSet = std::vector<std::pair<MultiThreadingTask, uint32_t>>;

}  // namespace inferllm