/**
 * @file evaluators.hpp
 * @brief Contains some general purpose evaluators.
 */

#ifndef DL_LEARNING_EVALUATORS_HPP
#define DL_LEARNING_EVALUATORS_HPP

#include "../device.hpp"
#include "../tensor/tensorptr.hpp"

namespace dl {

	class MeanError {
	private:
		dl::TensorPtr sum;
		size_t num;

	public:
		MeanError() noexcept : sum(dl::constant(0)), num(0) {}

		MeanError& operator+=(dl::TensorPtr loss) noexcept {
			sum = sum + dl::sum(loss);
			num += dl::numEntries(loss);
			return *this;
		}

		dl::TensorPtr aggregated() const { return sum / (float)num; }
	};

} // namespace dl

#endif
