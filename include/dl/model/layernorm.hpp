#pragma once

#include "../device.hpp"
#include "../tensor/tensor.hpp"
#include "./model.hpp"

namespace dl {
	/**
     * @brief Implements layer normalization as proposed by \cite layernorm.
     * 
     */
	class LayerNorm final : public Model<Tensor(Tensor&&)>, Model<Tensor(Tensor&)> {
	private:
		Tensor beta;
		Tensor gamma;

	public:
		LayerNorm(Shape normShape, const Device& device = Device::getDefault()) noexcept;

	public:
		virtual Tensor forward(Tensor& input) noexcept override;
		virtual Tensor forward(Tensor&& input) noexcept override;
	};
} // namespace dl