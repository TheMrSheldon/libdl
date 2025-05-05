#pragma once

#include "../tensor/tensorptr.hpp"
#include "./model.hpp"

namespace dl {
	/**
	 * @brief Applies a learnable linear transformation with optional bias.
	 * @details Given a vector \f(x\f) computes \f(xW^\top + b\f).
	 * 
	 */
	class Linear final : public Model {
	private:
		TensorPtr _weights;
		TensorPtr _bias;

	public:
		Linear(size_t inFeatures, size_t outFeatures, const Device& device, bool bias = true) noexcept;
		Linear(size_t inFeatures, size_t outFeatures, bool bias = true) noexcept;

	public:
		TensorPtr operator()(TensorPtr input) noexcept;

		inline TensorPtr& weights() noexcept { return _weights; }
		inline const TensorPtr& weights() const noexcept { return _weights; }
		inline TensorPtr& bias() noexcept { return _bias; }
		inline const TensorPtr& bias() const noexcept { return _bias; }
	};
} // namespace dl