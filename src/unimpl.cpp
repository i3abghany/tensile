#include "unimpl.h"

[[noreturn]]
void assertion_failed(const char* msg, const char* func, const char* file, long line)
{
    Tensile::Log::get_ostream_logger()->log("Assertion failed: `" + std::string(msg) + "` in " + std::string(func)
                                            + " at " + std::string(file) + ":" + std::to_string(line));
    std::abort();
}