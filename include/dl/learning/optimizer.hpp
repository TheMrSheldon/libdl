#pragma once

#include "../tensor/tensor.hpp"

namespace dl {

	class Optimizer {
	private:
	protected:
		Optimizer(/*std::vector<Parameter> parameters*/) = default;

	public:
		virtual void step(TensorPtr tensor) = 0;
	};

} // namespace dl