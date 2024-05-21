#pragma once

#include <functional>

namespace dl::utils {
	/**
     * @brief Composes a sequence of functions into a new anonymouse function by invoking the functions sequentially.
     * @details The code `composed(f, g, h)(...)` is equivalent to `f(g(h(...)))`.
     * 
     * @tparam F 
     * @tparam Fs 
     * @param arg 
     * @param args 
     * @return auto 
     */
	template <class F, class... Fs>
	auto composed(F&& arg, Fs&&... args) {
		return [fun = std::forward<F>(arg), ... functions = std::forward<Fs>(args)]<class X>(X&& x) mutable {
			if constexpr (sizeof...(Fs)) {
				return composed(std::forward<Fs>(functions)...)(std::invoke(std::forward<F>(fun), std::forward<X>(x)));
			} else {
				return std::invoke(std::forward<F>(fun), std::forward<X>(x));
			}
		};
	}
} // namespace dl::utils