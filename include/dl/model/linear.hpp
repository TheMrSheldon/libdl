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
		TensorPtr weights;
		TensorPtr bias;

	public:
		Linear(size_t inFeatures, size_t outFeatures, const Device& device, bool bias = true) noexcept;
		Linear(size_t inFeatures, size_t outFeatures, bool bias = true) noexcept;

	public:
		virtual TensorPtr forward(TensorPtr input) noexcept override;
		/** \todo For later: these const member functions make sense to indicate that we know at compile time that the
		 * instance is not modified (e.g. since it is not part of the computation graph for auto differentiation. **/
	};
} // namespace dl