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
		virtual ~Dataloader() = default;

		virtual dl::utils::GenericIterator<Instance> begin() = 0;
		virtual dl::utils::GenericIterator<Instance> end() = 0;
	};
} // namespace dl