#pragma once

#include "../tensor/tensor.hpp"
#include "./model.hpp"

namespace dl {
	class Linear final : public Model {
	private:
		TensorPtr weights;
		TensorPtr bias;

	public:
		Linear(unsigned inFeatures, unsigned outFeatures, Device& device, bool bias = true) noexcept;
		Linear(unsigned inFeatures, unsigned outFeatures, bool bias = true) noexcept;

		TensorPtr forward(TensorPtr input) noexcept;
	};
} // namespace dl