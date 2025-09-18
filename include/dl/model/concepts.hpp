#ifndef DL_MODEL_CONCEPTS_HPP
#define DL_MODEL_CONCEPTS_HPP

#include <functional>
#include <utility>

#include <type_traits>

namespace dl {

	template <typename F1, typename F2>
	concept Chainable = requires(F1 f1) { std::invoke(f1, std::declval<typename std::invoke_result_t<F2>>()); };

	//static_assert(std::is_same<typename std::invoke_result_t<int&(float)>, int&>::value);

	static_assert(Chainable<int (*)(int), int (*)()>);
	static_assert(Chainable<int (*)(int), int& (*)()>);
	static_assert(Chainable<int (*)(const int&), int (*)()>);
	static_assert(Chainable<int (*)(int), double (*)()>);
	static_assert(Chainable<decltype([](int) { return 0; }), decltype([]() { return 0.0; })>);
	//static_assert(Chainable<int (*)(int), double (*)(int)>);
	static_assert(Chainable<decltype([](auto) { return 0; }), decltype([]() { return 0.0; })>);
	//static_assert(Chainable<decltype([](int) { return 0; }), decltype([](int) { return 0.0; })>);
	//static_assert(Chainable<int(int), int&(float)>);

} // namespace dl

#endif