#pragma once

#include "../tensor/math.hpp"

namespace dl::loss {
	// Tensor mse(Tensor& x, Tensor& y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }
	Tensor mse(Tensor&& x, Tensor& y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }
	// Tensor mse(Tensor& x, Tensor&& y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }
	// Tensor mse(Tensor&& x, Tensor&& y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }
} // namespace dl::loss