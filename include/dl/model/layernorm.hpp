#pragma once

#include "../device.hpp"
#include "../tensor/tensorptr.hpp"
#include "./model.hpp"

namespace dl {
	/**
     * @brief Implements layer normalization as proposed by \cite layernorm.
     * 
     */
	class LayerNorm final : public Model {
	private:
		TensorPtr beta;
		TensorPtr gamma;

	public:
		LayerNorm(Shape normShape, const Device& device = Device::getDefault()) noexcept;
		virtual ~LayerNorm() = default;

	public:
		TensorPtr operator()(TensorPtr input) noexcept;
	};
} // namespace dl