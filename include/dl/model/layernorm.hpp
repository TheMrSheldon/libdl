#pragma once

#include "../device.hpp"
#include "../tensor/tensorptr.hpp"
#include "./model.hpp"

namespace dl {
	/**
     * @brief Implements layer normalization as proposed by \cite layernorm.
     * 
     */
	class LayerNorm final : public Model<TensorPtr(TensorPtr)> {
	private:
		TensorPtr beta;
		TensorPtr gamma;

	public:
		LayerNorm(Shape normShape, const Device& device = Device::getDefault()) noexcept;
		virtual ~LayerNorm() = default;

	public:
		virtual TensorPtr forward(TensorPtr input) noexcept override;
	};
} // namespace dl