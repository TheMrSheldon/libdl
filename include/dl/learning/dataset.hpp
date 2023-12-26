#pragma once

#include "dataloader.hpp"

namespace dl {
	template <typename>
	class Dataset;

	template <typename R, typename... Args>
	class Dataset<R(Args...)> {
	public:
        virtual std::unique_ptr<Dataloader<R(Args...)>> trainingData() = 0;
        virtual std::unique_ptr<Dataloader<R(Args...)>> validationData() = 0;
        virtual std::unique_ptr<Dataloader<R(Args...)>> testData() = 0;
	};
}; // namespace dl