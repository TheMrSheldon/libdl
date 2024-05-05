/**
 * @file math.hpp
 * @brief Implements auto-diff enabled wrapers around their concrete tensor implementations.
 */

#pragma once

#include "tensorptr.hpp"
#include <iostream>

namespace dl {
	TensorPtr pow(TensorPtr& base, float exponent) noexcept;
	TensorPtr pow(TensorPtr&& base, float exponent) noexcept;
	TensorPtr mean(TensorPtr& x) noexcept;
	TensorPtr mean(TensorPtr&& x) noexcept;
	TensorPtr sum(TensorPtr& x) noexcept;

	TensorPtr min(TensorPtr& x, float bound) noexcept;
	TensorPtr max(TensorPtr& x, float bound) noexcept;

	TensorPtr relu(TensorPtr& x) noexcept;

	TensorPtr operator+(TensorPtr& left, TensorPtr& right) noexcept;
	TensorPtr operator-(TensorPtr& left, TensorPtr& right) noexcept;
	TensorPtr operator*(TensorPtr& left, TensorPtr& right) noexcept;
	TensorPtr operator/(TensorPtr& left, TensorPtr& right) noexcept;
	TensorPtr operator/(TensorPtr& left, TensorPtr right) noexcept;

	TensorPtr matmul(TensorPtr& left, TensorPtr& right) noexcept;

	std::ostream& operator<<(std::ostream&, const TensorPtr& tensor) noexcept;
	bool operator==(const TensorPtr& left, const TensorPtr& right) noexcept;
} // namespace dl