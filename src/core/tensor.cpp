#include "tensor.h"

namespace inferllm {
Tensor::~Tensor() {
  if (is_own() && data_ != nullptr) {
    device_->free_device(data_.get());
    data_ = nullptr;
  }
  state_ = TensorState::OutSide;
}

void Tensor::set_shape(std::vector<size_t> shape) {
  dims_ = shape.size();
  shape_ = shape;
  stride_.resize(dims_);
  stride_[dims_ - 1] = 1;
  for (auto i = 1; i < dims_; ++i) {
    stride_[dims_ - i - 1] = stride_[dims_ - i] * shape_[dims_ - i];
  }
  length_ = stride_[0] * shape_[0];
}
}  // namespace inferllm