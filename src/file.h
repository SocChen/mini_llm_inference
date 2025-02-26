#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

namespace inferllm {

enum class FilePos {
  Begin = SEEK_SET,
  Current = SEEK_CUR,
  End = SEEK_END,
};

class InputFile {
 public:
  explicit InputFile(const std::string& path, bool enable_mmap = false);
  ~InputFile();

  bool enable_mmap() { return enable_mmap_; }

  bool eof() { return tell() == size_; }

  void rewind() { std::rewind(file_.get()); }

  void skip(int64_t bytes);

  void seek(int64_t offset, FilePos pos = FilePos::Begin);

  void read_raw(void* dst, size_t size);

  void read_data(void* dst, size_t size, int64_t offset);

  size_t tell() { return std::ftell(file_.get()); }

  void* get_mmap_data(size_t len, size_t offset);

  std::uint32_t read_u32();

  std::string read_string(std::uint32_t len);

 private:
  //   FILE* file_ = nullptr;
  std::shared_ptr<FILE> file_ = nullptr;
  size_t size_ = 0;
  int fd_ = -1;
  bool enable_mmap_ = false;
  void* mmap_addr_ = nullptr;
};
}  // namespace inferllm