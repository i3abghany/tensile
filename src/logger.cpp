#include "tensile/logger.h"

namespace Tensile::Log {

static std::shared_ptr<LoggerBase<std::ostream>> logger;
static Severity severity = Severity::INFO;

std::shared_ptr<LoggerBase<std::ostream>> get_ostream_logger(Severity s)
{
    if (!logger) {
        auto logger_base = std::make_shared<LoggerBase<std::ostream>>(std::cout);
        auto timestamp_logger = std::make_shared<TimestampLogger<std::ostream>>(logger_base);
        logger = std::make_shared<SeverityLogger<std::ostream>>(timestamp_logger);
    }
    if (severity != s) {
        auto* severity_logger = dynamic_cast<SeverityLogger<std::ostream>*>(logger.get());
        severity_logger->set_severity(s);
        severity = s;
    }
    return logger;
}

}