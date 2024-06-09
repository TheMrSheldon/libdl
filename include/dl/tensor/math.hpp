/**
 * @file math.hpp
 * @brief Implements auto-diff enabled wrappers around their concrete tensor implementations.
 */

#pragma once

#include "shape.hpp"
#include "tensorptr.hpp"

#include <iostream>
#include <vector>

namespace dl {
	/**
	 * @brief Computes the \p exponent -th power of each element in \p base and returns the resulting tensor.
	 * @details
	 * @param base the basis for which to compute the \p exponent -th power.
	 * @param exponent the exponent.
	 * @return the \p exponent -th power of each component of \p base .
	 */
	[[nodiscard]] TensorPtr pow(TensorPtr base, float exponent) noexcept;

	[[nodiscard]] TensorPtr exp(TensorPtr base) noexcept;

	[[nodiscard]] TensorPtr sqrt(TensorPtr x) noexcept;

	/**
	 * @brief Computes the reciprocal square root for each element in \p x.
	 * 
	 * @param x 
	 * @return 
	 */
	[[nodiscard]] TensorPtr rsqrt(TensorPtr x) noexcept;

	[[nodiscard]] TensorPtr mean(TensorPtr x) noexcept;
	[[nodiscard]] TensorPtr mean(TensorPtr x, int dim, bool keepdim = false) noexcept;

	[[nodiscard]] TensorPtr sum(TensorPtr x) noexcept;
	[[nodiscard]] TensorPtr sum(TensorPtr x, int dim, bool keepdim = false) noexcept;

	[[nodiscard]] TensorPtr min(TensorPtr x) noexcept;
	[[nodiscard]] TensorPtr min(TensorPtr x, int dim, bool keepdim = false) noexcept;

	[[nodiscard]] TensorPtr max(TensorPtr x) noexcept;
	[[nodiscard]] TensorPtr max(TensorPtr x, int dim, bool keepdim = false) noexcept;
	[[nodiscard]] TensorPtr max(TensorPtr x, TensorPtr y) noexcept;

	/**
	 * @brief Wrapper around std::size_t to discern between var(TensorPtr, DOF) and var(TensorPtr, size_t).
	 */
	struct DOF {
		size_t dof;
	};

	[[nodiscard]] TensorPtr var(TensorPtr x, DOF dof = DOF{1}) noexcept;
	[[nodiscard]] TensorPtr var(TensorPtr x, int dim, DOF dof = DOF{1}) noexcept;

	[[nodiscard]] TensorPtr erf(TensorPtr x) noexcept;

	[[nodiscard]] TensorPtr relu(TensorPtr x) noexcept;

	// Tensor-Scalar Operations
	[[nodiscard]] TensorPtr operator+(TensorPtr left, float right) noexcept;
	[[nodiscard]] TensorPtr operator-(TensorPtr left, float right) noexcept;
	[[nodiscard]] TensorPtr operator*(TensorPtr left, float right) noexcept;
	[[nodiscard]] TensorPtr operator/(TensorPtr left, float right) noexcept;
	[[nodiscard]] TensorPtr operator+(float left, TensorPtr right) noexcept;
	[[nodiscard]] TensorPtr operator-(float left, TensorPtr right) noexcept;
	[[nodiscard]] TensorPtr operator*(float left, TensorPtr right) noexcept;
	[[nodiscard]] TensorPtr operator/(float left, TensorPtr right) noexcept;

	// Tensor-Tensor Operations
	[[nodiscard]] TensorPtr operator+(TensorPtr left, TensorPtr right) noexcept;
	[[nodiscard]] TensorPtr operator-(TensorPtr left, TensorPtr right) noexcept;
	[[nodiscard]] TensorPtr operator*(TensorPtr left, TensorPtr right) noexcept;
	[[nodiscard]] TensorPtr operator/(TensorPtr left, TensorPtr right) noexcept;

	[[nodiscard]] TensorPtr fma(const TensorPtr& factor1, const TensorPtr& factor2, const TensorPtr& summand) noexcept;

	[[nodiscard]] TensorPtr matmul(TensorPtr left, TensorPtr right) noexcept;

	/**
	 * @brief Transposes the given tensor at the specified coordinates.
	 * @details Transposes the given tensor by permuting the dimensions. For example, if the input was a matrix, then
	 * \f(x \in \mathbb{R}^n\times m\f) the matrix transposition could be computed via `dl::transpose(x, {0, 1});`.
	 * 
	 * @param x the tensor to be tranposed.
	 * @param permutation the permutation to apply to the dimensions.
	 * @return a new tensor with the permutation \p permutation applied to the dimensions.
	 */
	[[nodiscard]] TensorPtr transpose(TensorPtr x, std::vector<int>&& permutation) noexcept;

	[[nodiscard]] TensorPtr softmax(TensorPtr x) noexcept;
	/**
	 * @brief Computes the softmax function of the input vector.
	 * @details Let \f(x \in \mathbb R^n\f) be a vector, the softmax is defined as
	 * \f[
	 * 	\text{softmax}\colon \mathbb R^n \to \mathbb R^n
	 * 	~\text{via}~ x\mapsto \left(\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}\right)_{1\leq i\leq n}.
	 * \f]
	 * 
	 * @param x the input tensor.
	 * @return TensorPtr the softmax of \p x .
	 */
	[[nodiscard]] TensorPtr softmax(TensorPtr x) noexcept;
	[[nodiscard]] TensorPtr softmax(TensorPtr x, int dim) noexcept;

	std::ostream& operator<<(std::ostream&, const TensorPtr& tensor) noexcept;
	[[nodiscard]] bool operator==(const TensorPtr& left, const TensorPtr& right) noexcept;
	[[nodiscard]] bool
	allclose(const TensorPtr& left, const TensorPtr& right, float rtol = 1e-5, float atol = 1e-8) noexcept;

	/**
	 * @brief Returns the number of entries in the tensor.
	 * @details This is equivalent to the product of the shape.
	 * 
	 * @param tensor The tensor to count the number of entries for.
	 * @return size_t The number of entries in \p tensor.
	 */
	[[nodiscard]] size_t numEntries(const TensorPtr& tensor) noexcept;

	TensorPtr reshape(TensorPtr tensor, SShape shape) noexcept;
} // namespace dl