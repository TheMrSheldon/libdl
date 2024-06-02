#pragma once

#include "../tensor/tensor.hpp"
#include "./model.hpp"

namespace dl {
	/**
	 * @brief Applies a learnable linear transformation with optional bias.
	 * @details Given a vector \f(x\f) computes \f(xW^\top + b\f).
	 * 
	 */
	class Linear final : public Model<Tensor(Tensor&&)>, Model<Tensor(Tensor&)>, Model<Tensor(const Tensor&)> {
	private:
		Tensor weights;
		Tensor bias;

	public:
		Linear(size_t inFeatures, size_t outFeatures, const Device& device, bool bias = true) noexcept;
		Linear(size_t inFeatures, size_t outFeatures, bool bias = true) noexcept;

	public:
		virtual Tensor forward(Tensor& input) noexcept override;
		virtual Tensor forward(Tensor&& input) noexcept override;
		virtual Tensor forward(const Tensor& input) noexcept override;
		/** \todo For later: these const member functions make sense to indicate that we know at compile time that the
		 * instance is not modified (e.g. since it is not part of the computation graph for auto differentiation. **/
		//virtual Tensor forward(Tensor&& input) const noexcept override;
	};
} // namespace dl