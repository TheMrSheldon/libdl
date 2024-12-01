#ifndef DL_UTILS_OVERLOADED_HPP
#define DL_UTILS_OVERLOADED_HPP

namespace dl::utils {
	/** Taken from https://en.cppreference.com/w/cpp/utility/variant/visit **/
	template <class... Ts>
	struct overloaded : Ts... {
		using Ts::operator()...;
	};
} // namespace dl::utils

#endif