#include <string>

#include "logger.h"

[[noreturn]]
void assertion_failed(const char* msg, const char* func, const char* file, long line);

#define UNIMPLEMENTED(msg) assertion_failed(msg, __func__, __FILE__, __LINE__)

#define UNIMPLEMENTED_IF(cond, msg)                                                                                    \
    if (cond)                                                                                                          \
    UNIMPLEMENTED(msg)
