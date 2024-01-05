#pragma once

#include <assert.h>
#include <functional>

namespace dl::utils {
	/**
	 * @brief 
	 * @details
	 * ```{cpp}
	 * {ScopeGuard _([] { std::cout << "Scope Exited" << std::endl; });
	 *     std::cout << "Inside Scope" << std::endl;
	 * }
	 * ```
	 */
	class ScopeGuard final {
	private:
		std::function<void()> onScopeExit;

		ScopeGuard(const ScopeGuard&) = delete;
		ScopeGuard& operator=(const ScopeGuard&) = delete;

	public:
		explicit ScopeGuard(auto&& onScopeExit) try : onScopeExit(std::forward<decltype(onScopeExit)>(onScopeExit)) {
		} catch (...) {
			onScopeExit();
			throw;
		}

		explicit ScopeGuard(ScopeGuard&& other) : onScopeExit(std::move(other.onScopeExit)) {
			assert(other.onScopeExit == nullptr);
		}

		~ScopeGuard() {
			if (onScopeExit != nullptr)
				onScopeExit();
		}
	};
} // namespace dl::utils