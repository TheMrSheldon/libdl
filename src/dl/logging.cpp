#include <dl/logging.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>

#include <format>

using LoggerPtr = dl::logging::LoggerPtr;

const char* dl::logging::getVersionStr() noexcept {
	static auto version = std::format("spdlog v.{}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);
	return version.c_str();
}

void dl::logging::setVerbosity(dl::logging::Verbosity verbosity) noexcept {
	auto level = spdlog::level::off - static_cast<int>(verbosity);
	spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
}

LoggerPtr dl::logging::getLogger(std::string name) {
	auto logger = spdlog::get(name);
	if (logger == nullptr) {
		logger = spdlog::stdout_color_mt(name);
	}
	return logger;
}