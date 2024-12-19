#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

namespace Tensile::Log {

template <typename S> class LoggerDecorator;
template <typename S> class TimestampLogger;
template <typename S> class SeverityLogger;

enum Severity { INFO, WARNING, ERROR };

template <typename S> class LoggerBase {
public:
    explicit LoggerBase(S& stream)
        : stream_(stream)
    {
    }

    explicit LoggerBase(std::shared_ptr<LoggerBase<S>> logger)
        : stream_(logger->stream_)
    {
    }

    virtual void log(const std::string& message) { stream_ << message << std::endl; }

    virtual ~LoggerBase() = default;

protected:
    S& stream_;
};

template <typename S> class LoggerDecorator : public LoggerBase<S> {
public:
    explicit LoggerDecorator(std::shared_ptr<LoggerBase<S>> logger)
        : LoggerBase<S>(logger)
        , logger(logger)
    {
    }

    void log(const std::string& message) override { logger->log(message); }

protected:
    std::shared_ptr<LoggerBase<S>> logger;
};

template <typename S> class TimestampLogger : public LoggerDecorator<S> {
public:
    explicit TimestampLogger(std::shared_ptr<LoggerBase<S>> logger)
        : LoggerDecorator<S>(std::move(logger))
    {
    }

    void log(const std::string& message) override
    {
        auto msg = "[" + get_time_stamp() + "] " + message;
        LoggerDecorator<S>::logger->log(msg);
    }

    [[nodiscard]] std::string get_time_stamp() const
    {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }
};

template <typename S> class SeverityLogger : public LoggerDecorator<S> {
public:
    explicit SeverityLogger(std::shared_ptr<LoggerBase<S>> logger)
        : LoggerDecorator<S>(std::move(logger))
        , severity_(INFO)
    {
    }

    void log(const std::string& message) override
    {
        auto msg = "[" + get_severity_str() + "] " + message;
        LoggerDecorator<S>::logger->log(msg);
    }

    void set_severity(Severity severity) { severity_ = severity; }

private:
    [[nodiscard]] std::string get_severity_str() const
    {
        switch (severity_) {
        case INFO:
            return "INFO";
        case WARNING:
            return "WARNING";
        case ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
        }
    }

    Severity severity_;
};

std::shared_ptr<LoggerBase<std::ostream>> get_ostream_logger(Tensile::Log::Severity);

#define LOG_INFO(message)                                                                                              \
    do {                                                                                                               \
        auto logger = Tensile::Log::get_ostream_logger(Tensile::Log::Severity::INFO);                                  \
        logger->log(message);                                                                                          \
    } while (0)

#define LOG_WARNING(message)                                                                                           \
    do {                                                                                                               \
        auto logger = Tensile::Log::get_ostream_logger(Tensile::Log::Severity::WARNING);                               \
        logger->log(message);                                                                                          \
    } while (0)

#define LOG_ERROR(message)                                                                                             \
    do {                                                                                                               \
        auto logger = Tensile::Log::get_ostream_logger(Tensile::Log::Severity::ERROR);                                 \
        logger->log(message);                                                                                          \
    } while (0)
}