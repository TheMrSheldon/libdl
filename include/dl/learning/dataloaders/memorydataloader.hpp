#ifndef DL_LEARNING_DATALOADERS_MEMORYDATALOADER_HPP
#define DL_LEARNING_DATALOADERS_MEMORYDATALOADER_HPP

#include "../../utils/generic_iterator.hpp"
#include "../dataloader.hpp"

#include <vector>

namespace dl {

	template <typename>
	class MemoryDataloader;

	template <typename R, typename... Args>
	class MemoryDataloader<R(Args...)> : public Dataloader<R(Args...)> {
	private:
		using Instance = typename std::tuple<R, std::remove_reference_t<Args>...>;
		using Iterator = typename dl::utils::GenericIterator<Instance>;
		std::vector<Instance> data;

	public:
		explicit MemoryDataloader(std::vector<Instance> data) : data(data) {}
		virtual ~MemoryDataloader() = default;

		Iterator begin() override { return Iterator(data.begin()); }
		Iterator end() override { return Iterator(data.end()); }
	};

} // namespace dl

#endif