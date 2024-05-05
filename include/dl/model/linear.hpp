#pragma once

#include "../tensor/tensor.hpp"
#include "./model.hpp"

namespace dl {
	class Linear final : public Model<TensorPtr(TensorPtr&)> {
	private:
		TensorPtr weights;
		TensorPtr bias;

	public:
		Linear(unsigned inFeatures, unsigned outFeatures, const Device& device, bool bias = true) noexcept;
		Linear(unsigned inFeatures, unsigned outFeatures, bool bias = true) noexcept;

	protected:
		virtual TensorPtr forward(TensorPtr& input) noexcept override;
		/** \todo For later: these const member functions make sense to indicate that we know at compile time that the
		 * instance is not modified (e.g. since it is not part of the computation graph for auto differentiation. **/
		//virtual TensorPtr forward(TensorPtr& input) const noexcept override;
	};
} // namespace dl