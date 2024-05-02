#pragma once

#include "../tensor/tensor.hpp"

namespace dl {

	/**
	 * @brief Defines an optimization strategy for a given set of Parameters.
	 * @details In its simplest form
	 * 
	 */
	class Optimizer {
	private:
	protected:
		Optimizer(/*std::vector<Parameter> parameters*/) = default;

	public:
		virtual ~Optimizer() = default;

		virtual void step(TensorPtr tensor) = 0;
	};

} // namespace dl