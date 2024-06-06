#pragma once

namespace dl {
	class Tensor;

	/**
	 * @brief Defines an optimization strategy for a given set of Parameters.
	 * @details In its simplest form
	 * 
	 */
	class Optimizer {
	private:
	protected:
		Optimizer() = default;

	public:
		virtual ~Optimizer() = default;

		virtual void step(const Tensor& tensor) = 0;
	};

} // namespace dl