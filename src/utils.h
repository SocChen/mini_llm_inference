#pragma once

#include <string>
namespace inferllm {
std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
}  // namespace inferllm

//! branch prediction hint: likely to take
#define infer_likely(v) __builtin_expect(static_cast<bool>(v), 1)

//! branch prediction hint: unlikely to take
#define infer_unlikely(v) __builtin_expect(static_cast<bool>(v), 0)

#define INFER_LOG(format, ...) fprintf(stderr, format, ##__VA_ARGS__)
#define INFER_ERROR(format, ...) fprintf(stderr, format, ##__VA_ARGS__)

#define INFER_ASSERT(expr, message)                                 \
  do {                                                              \
    if (infer_unlikely(!(expr))) {                                  \
      INFER_ERROR(                                                  \
          "Assert \' %s \' failed at file : %s \n"                  \
          "line %d : %s,\nextra "                                   \
          "message: %s",                                            \
          #expr, __FILE__, __LINE__, __PRETTY_FUNCTION__, message); \
      abort();                                                      \
    }                                                               \
  } while (0)

#define INFER_PAUSE(v)                                \
  do {                                                \
    for (int __delay = (v); __delay > 0; --__delay) { \
      _mm_pause();                                    \
    }                                                 \
  } while (0)