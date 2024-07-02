#pragma once

#include "../tensor/tensorptr.hpp"
#include "./model.hpp"

namespace dl {
	/**
	 * @brief Applies a learnable linear transformation with optional bias.
	 * @details Given a vector \f(x\f) computes \f(xW^\top + b\f).
	 * 
	 */
	class Linear final : public Model<TensorPtr(TensorPtr)> {
	private:
		TensorPtr _weights;
		TensorPtr _bias;

	public:
		Linear(size_t inFeatures, size_t outFeatures, const Device& device, bool bias = true) noexcept;
		Linear(size_t inFeatures, size_t outFeatures, bool bias = true) noexcept;

	public:
		virtual TensorPtr forward(TensorPtr input) noexcept override;

		inline TensorPtr& weights() noexcept { return _weights; }
		inline const TensorPtr& weights() const noexcept { return _weights; }
		inline TensorPtr& bias() noexcept { return _bias; }
		inline const TensorPtr& bias() const noexcept { return _bias; }

		dl::TensorPtr operator()(TensorPtr input) noexcept { return forward(input); }
	};
} // namespace dl