/**
 * @file loss.hpp
 * @brief Contains some general purpose loss objectives.
 */

#ifndef DL_LEARNING_LOSS_HPP
#define DL_LEARNING_LOSS_HPP

#include "../tensor/math.hpp"

namespace dl::loss {
	Tensor mse(Tensor&& x, const Tensor& y) noexcept { return dl::mean(dl::pow(std::move(x) - y, 2.0f)); }
} // namespace dl::loss

#endif