#include "logger.h"

namespace Tensile::Log {

std::shared_ptr<Logger<std::ostream>> get_default_logger() { return Logger<std::ostream>::get_ostream_logger(); }

};