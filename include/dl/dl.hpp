#pragma once

namespace dl {
	/**
	 * @brief The current version of libdl.
	 */
	constexpr const auto version = "0.0.1";

	/**
	 * @brief The git commit hash of the current build.
	 */
	constexpr const auto commit_hash = GIT_COMMIT_HASH;
} // namespace dl
