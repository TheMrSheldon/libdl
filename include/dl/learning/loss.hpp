#pragma once

#include "../tensor/math.hpp"

namespace dl::loss {
	// TensorPtr mse(TensorPtr& x, TensorPtr& y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }
	TensorPtr mse(TensorPtr&& x, TensorPtr& y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }
	// TensorPtr mse(TensorPtr& x, TensorPtr&& y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }
	// TensorPtr mse(TensorPtr&& x, TensorPtr&& y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }
} // namespace dl::loss