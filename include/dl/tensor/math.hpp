/**
 * @file math.hpp
 * @brief Implements auto-diff enabled wrapers around their concrete tensor implementations.
 */

#pragma once

#include "tensor.hpp"
#include <iostream>

namespace dl {
	Tensor pow(Tensor& base, float exponent) noexcept;
	Tensor pow(Tensor&& base, float exponent) noexcept;
	Tensor mean(Tensor& x) noexcept;
	Tensor mean(Tensor&& x) noexcept;
	Tensor sum(Tensor& x) noexcept;

	Tensor min(Tensor& x, float bound) noexcept;
	Tensor max(Tensor& x, float bound) noexcept;

	Tensor relu(Tensor& x) noexcept;

	Tensor operator+(Tensor& left, Tensor& right) noexcept;
	Tensor operator-(Tensor& left, Tensor& right) noexcept;
	Tensor operator*(Tensor& left, Tensor& right) noexcept;
	Tensor operator/(Tensor& left, Tensor& right) noexcept;
	Tensor operator/(Tensor& left, Tensor right) noexcept;

	Tensor matmul(Tensor& left, Tensor& right) noexcept;

	std::ostream& operator<<(std::ostream&, const Tensor& tensor) noexcept;
	bool operator==(const Tensor& left, const Tensor& right) noexcept;
} // namespace dl