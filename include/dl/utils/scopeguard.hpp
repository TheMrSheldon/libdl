#pragma once

#include <assert.h>
#include <functional>

namespace dl::utils {
	template <typename T>
	class ScopeGuard final {
	private:
		std::function<void()> onScopeExit;

		ScopeGuard(const ScopeGuard&) = delete;
		ScopeGuard& operator=(const ScopeGuard&) = delete;

	public:
		explicit ScopeGuard(T && onScopeExit) try : onScopeExit(std::forward<T>(onScopeExit)) {
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