#pragma once

#include "../tensor/tensor.hpp"
#include "./model.hpp"

namespace dl {
	class Linear final : public Model<TensorPtr(TensorPtr)> {
	private:
		TensorPtr weights;
		TensorPtr bias;

	public:
		Linear(unsigned inFeatures, unsigned outFeatures, const Device& device, bool bias = true) noexcept;
		Linear(unsigned inFeatures, unsigned outFeatures, bool bias = true) noexcept;

	protected:
		virtual TensorPtr forward(TensorPtr input) noexcept override;
		virtual TensorPtr forward(TensorPtr input) const noexcept override;
	};
} // namespace dl