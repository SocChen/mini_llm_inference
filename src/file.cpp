#include "file.h"

#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "utils.h"
namespace inferllm {
InputFile::InputFile(const std::string& path, bool enable_mmap)
    : enable_mmap_(enable_mmap) {
  file_ = std::shared_ptr<FILE>(std::fopen(path.c_str(), "rb"), ::fclose);

  INFER_ASSERT(file_, "Failed to open model file.");
  fd_ = fileno(file_.get());
  fseek(file_.get(), 0, SEEK_END);
  size_ = ftell(file_.get());
  rewind();
  if (enable_mmap_) {
    mmap_addr_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    INFER_ASSERT(mmap_addr_ != MAP_FAILED, "mmap failed.");
    madvise(mmap_addr_, size_, MADV_WILLNEED);
  }
}

InputFile::~InputFile() {
  if (enable_mmap_) {
    munmap(mmap_addr_, size_);
  }
}

void InputFile::skip(int64_t bytes) {
  auto err = fseek(file_.get(), bytes, SEEK_CUR);
  INFER_ASSERT(!err, "skip file error");
}

void InputFile::seek(int64_t offset, FilePos pos) {
  auto err = fseek(file_.get(), offset, static_cast<int>(pos));
  INFER_ASSERT(!err, "skip file error");
}

void InputFile::read_raw(void* dst, size_t size) {
  if (size == 0) {
    return;
  }
  auto nr = fread(dst, 1, size, file_.get());
  INFER_ASSERT(nr == size, "read file error");
}

void InputFile::read_data(void* dst, size_t size, int64_t offset) {
  fseek(file_.get(), offset, SEEK_SET);
  read_raw(dst, size);
}

void* InputFile::get_mmap_data(size_t len, size_t offset) {
  INFER_ASSERT(offset < size_, "offset error when get mmap data.");
  return static_cast<void*>(static_cast<int8_t*>(mmap_addr_) + offset);
}
std::uint32_t InputFile::read_u32() {
  std::uint32_t ret = 0;
  read_raw(&ret, sizeof(ret));
  return ret;
}

std::string InputFile::read_string(std::uint32_t len) {
  std::vector<char> chars(len);
  read_raw(chars.data(), len);
  return std::string(chars.data(), len);
}
}  // namespace inferllm