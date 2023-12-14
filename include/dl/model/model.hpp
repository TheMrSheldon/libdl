#pragma once

#include "../tensor/tensor.hpp"

namespace dl {
	class Device;

	class Model {
	private:
	protected:
		void registerSubmodel(const Model& model);
		void registerWeights(const TensorPtr& tensor);

	public:
		void to(const Device& device) noexcept;

		template <typename R, typename... Args>
		R operator()(Args&&... args);
	};
} // namespace dl