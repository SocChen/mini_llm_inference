#pragma once

#include <cstdint>
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

/**
 * @brief  Tensor is a data structure that can be used to store and manipulate
 *
 */
/// the tensor memory is from three ways:
/// 1. the tensor is own the memory, allocate by itself
/// 2. the tensor memory is shared from outside, such as the input tensor,output
/// tensor
/// 3. the tensor memory is map from file, such as the weight tensor

// class Tensor {
//  public:
//   Tensor(Device* device, )
// }

}  // namespace inferllm