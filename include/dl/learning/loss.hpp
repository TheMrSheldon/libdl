/**
 * @file loss.hpp
 * @brief Contains some general purpose loss objectives.
 */

#ifndef DL_LEARNING_LOSS_HPP
#define DL_LEARNING_LOSS_HPP

#include "../tensor/math.hpp"

namespace dl::loss {
	TensorPtr mse(TensorPtr x, TensorPtr y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }
} // namespace dl::loss

#endif