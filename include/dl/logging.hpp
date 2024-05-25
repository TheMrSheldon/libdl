#pragma once

#include <memory>

#include <spdlog/spdlog.h>

namespace dl::log {
	using LoggerPtr = std::shared_ptr<spdlog::logger>;

	enum class Verbosity : int { Off, Critical, Error, Warning, Info, Debug, Trace };

	const char* getVersionStr() noexcept;

	/**
	 * @brief Globally set the verbosity for all loggers to the one specified.
	 * 
	 * @param verbosity The verbosity to use for logging.
	 */
	void setVerbosity(Verbosity verbosity) noexcept;

	/**
	 * @brief Creates a new named logger instance with the given name or fetches an existing one associated with
	 * the name.
	 * 
	 * @param name The name of the the logger to be returned.
	 * @return A logger with the specified name.
	 */
	LoggerPtr getLogger(std::string name);
} // namespace dl::log

///////////////////////////
// Custom loggable types //
///////////////////////////
#include <spdlog/fmt/bundled/format.h>

template <typename T>
struct fmt::formatter<std::optional<T>> {
	constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.end(); }

	template <typename FormatContext>
	auto format(const std::optional<T>& input, FormatContext& ctx) -> decltype(ctx.out()) {
		if (input.has_value())
			return fmt::format_to(ctx.out(), "{}", input.value());
		return fmt::format_to(ctx.out(), "<EMPTY>");
	}
};

/** \todo remove when formatting ranges is supported **/
template <typename T>
struct fmt::formatter<std::vector<T>> {
	constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) { return ctx.end(); }

	template <typename FormatContext>
	auto format(const std::vector<T>& input, FormatContext& ctx) -> decltype(ctx.out()) {
		return fmt::format_to(ctx.out(), "{{{}}}", fmt::join(input, ", "));
	}
};