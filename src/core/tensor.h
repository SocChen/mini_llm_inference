#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/device.h"
#include "utils.h"
namespace inferllm {
enum class DType {
  Float32 = 0,
  Float16 = 1,
  Float8 = 2,
  Int32 = 3,
  Int16 = 4,
  Int8 = 5,
  Uint8 = 6,
  Int4 = 7,
  Uint4 = 8,
  Int2 = 9,
};

/**
 * @brief get the size of a dtype in bytes
 *
 * @param dtype
 * @return float
 */
float dtype_in_byte(DType dtype);

/**
 * @brief the data arrangement of a dtype
 *
 * @param dtype
 * @return uint32_t
 */
uint32_t dtype_block_size(DType dtype);

/**
 * @brief the state of a tensor
 *
 */
enum class TensorState {
  Own = 0,
  OutSide = 1,
};

class OpBase;

/**
 * @brief  Tensor is a data structure that can be used to store and manipulate
 *
 */
/// the tensor memory is from three ways:
/// 1. the tensor is own the memory, allocate by itself
/// 2. the tensor memory is shared from outside, such as the input tensor,output
/// tensor
/// 3. the tensor memory is map from file, such as the weight tensor

class Tensor {
 public:
  Tensor(std::shared_ptr<Device> device, std::string name)
      : device_(device), name_(name), state_(TensorState::OutSide) {}

  Tensor(std::vector<size_t> shape, DType dtype, std::shared_ptr<Device> device)
      : dtype_(dtype), device_(device), state_(TensorState::OutSide) {
    set_shape(shape_);
  };
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;
  Tensor(Tensor&& other) = delete;
  Tensor& operator=(const Tensor&& other) = delete;

  ~Tensor();

  std::shared_ptr<Device> device() { return device_; }

  std::shared_ptr<OpBase> owner_op() { return owner_op_; }

  void set_owner_op(std::shared_ptr<OpBase> owner_op) { owner_op_ = owner_op; }

  std::string name() { return name_; }

  void set_name(const std::string name) { name_ = name; }

  bool is_own() const { return state_ == TensorState::Own; }

  uint32_t dims() { return dims_; }

  size_t length() { return length_; }

  void set_dtype(DType dtype) { dtype_ = dtype; }

  DType dtype() const { return dtype_; }

  std::vector<size_t> shape() const { return shape_; };

  void set_shape(std::vector<size_t> shape, DType dtype) {
    set_shape(shape);
    set_dtype(dtype);
  };

  void set_shape(std::vector<size_t> shape);

  std::vector<size_t> stride() const { return stride_; }

  std::shared_ptr<void> data() {
    INFER_ASSERT(is_own(),
                 "Tensor is OutSide the device, can't get the memory.");
    return data_;
  }

  const std::shared_ptr<void> data() const {
    INFER_ASSERT(is_own(),
                 "Tensor is OutSide the device, can't get the memory.");
    return data_;
  }

  template <typename T>
  std::shared_ptr<T> data() {
    INFER_ASSERT(is_own(),
                 "Tensor is OutSide the device, can't get the memory.");
    return data_;
  }

 private:
  ///
  std::shared_ptr<Device> device_ = nullptr;
  ///
  std::shared_ptr<OpBase> owner_op_ = nullptr;

  std::string name_;
  TensorState state_;
  uint32_t dims_ = 0;
  size_t length_ = 0;
  DType dtype_ = DType::Float32;
  std::vector<size_t> shape_;
  std::vector<size_t> stride_;
  std::shared_ptr<void> data_ = nullptr;
};

}  // namespace inferllm