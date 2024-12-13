#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>

template <typename StreamType> class LoggerDecorator;

template <typename StreamType> class TimestampLogger;

template <typename StreamType> class SeverityLogger;

template <typename StreamType> class Logger {
public:
    explicit Logger(StreamType& stream)
        : stream_(stream)
    {
    }
    explicit Logger(std::shared_ptr<Logger<StreamType>> wrapped_logger)
        : stream_(wrapped_logger->stream_)
    {
    }
    virtual ~Logger() = default;

    virtual void log(const std::string& message) { stream_ << message << std::endl; }

    static auto get_ostream_logger()
    {
        static std::shared_ptr<SeverityLogger<std::ostream>> logger = nullptr;

        if (logger != nullptr)
            return logger;

        auto created_logger = std::make_shared<Logger<std::ostream>>(std::cout);
        auto timestame_logger = std::make_shared<TimestampLogger<std::ostream>>(created_logger);
        auto severity_logger = std::make_shared<SeverityLogger<std::ostream>>(timestame_logger);

        return logger = severity_logger;
    }

protected:
    StreamType& stream_;
};

template <typename StreamType> class LoggerDecorator : public Logger<StreamType> {
public:
    explicit LoggerDecorator(std::shared_ptr<Logger<StreamType>> wrapped_logger)
        : Logger<StreamType>(wrapped_logger)
        , wrapped_logger_(wrapped_logger)
    {
    }

    void log(const std::string& message) override { wrapped_logger_->log(message); }

protected:
    std::shared_ptr<Logger<StreamType>> wrapped_logger_;
};

template <typename StreamType> class TimestampLogger : public LoggerDecorator<StreamType> {
public:
    explicit TimestampLogger(std::shared_ptr<Logger<StreamType>> wrapped_logger)
        : LoggerDecorator<StreamType>(wrapped_logger)
    {
    }

    void log(const std::string& message) override
    {
        std::string timestamp = get_current_time();
        this->wrapped_logger_->log("[" + timestamp + "] " + message);
    }

private:
    std::string get_current_time()
    {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }
};

template <typename StreamType> class SeverityLogger : public LoggerDecorator<StreamType> {
public:
    enum Severity { INFO, WARNING, ERROR };

    explicit SeverityLogger(std::shared_ptr<Logger<StreamType>> wrapped_logger)
        : LoggerDecorator<StreamType>(wrapped_logger)
    {
    }

    void log(const std::string& message, Severity severity)
    {
        std::string severityStr = severity_to_string(severity);
        this->wrapped_logger_->log("[" + severityStr + "] " + message);
    }

private:
    std::string severity_to_string(Severity severity) const
    {
        switch (severity) {
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
};