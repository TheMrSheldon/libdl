#pragma once

#include "../utils/generic_iterator.hpp"

#include <tuple>
#include <vector>

namespace dl {
	template <typename>
	class Dataloader;

	template <typename R, typename... Args>
	class Dataloader<R(Args...)> {
	public:
		using Instance = std::tuple<R, Args...>;
		using Batch = std::vector<Instance>;
	private:
	public:
		virtual dl::utils::GenericIterator<std::tuple<R, Args...>> begin() = 0;
		virtual dl::utils::GenericIterator<std::tuple<R, Args...>> end() = 0;
	};
} // namespace dl