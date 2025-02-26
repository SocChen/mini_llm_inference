#include "utils.h"

#include <cstdarg>
#include <vector>

namespace inferllm {
std::string format(const char* fmt, ...) {
  va_list ap, ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int size = vsnprintf(NULL, 0, fmt, ap);
  std::vector<char> buf(size + 1);
  int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
  va_end(ap2);
  va_end(ap);
  return std::string(buf.data(), size);
}
}  // namespace inferllm