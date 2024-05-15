/**
 * @file math.hpp
 * @brief Implements auto-diff enabled wrappers around their concrete tensor implementations.
 */

#pragma once

#include "tensor.hpp"

#include <iostream>
#include <vector>

namespace dl {
	/**
	 * @brief \copybrief dl::pow(const dl::Tensor&,float)
	 * @details This function is part of automatic differentiation. If \p base requires a gradient
	 * (dl::TensorImpl::requiresGrad() is \c true ), a reference to it is stored in the compute graph for the backwards
	 * pass. As such, **do not** delete the base tensor before the backwards pass finished. If you can't don't want to
	 * keep the tensor, move it into the compute graph via dl::pow(dl::Tensor&&,float).
	 * 
	 * ```{cpp}
	 * dl::Tensor out;
	 * {
	 *     dl::Tensor base = {1, 2, 3};
	 *     base->setRequiresGrad(true);
	 *     // The following will result in a segmentation fault since backward() is called when base was deleted
	 *     // out = dl::pow(base, 2);
	 *     // Solution: Move the tensor into the computation graph
	 *     out = dl::pow(std::move(base), 2);
	 * }
	 * out.backward();
	 * ```
	 * 
	 * @param base the basis for which to compute the \p exponent -th power.
	 * @param exponent the exponent.
	 * @return Tensor the \p exponent -th power of each component of \p base .
	 * @see dl::pow(dl::Tensor&&,float)
	 */
	[[nodiscard]] Tensor pow(Tensor& base, float exponent) noexcept;
	/**
	 * @brief \copybrief dl::pow(const dl::Tensor&,float)
	 * 
	 * @param base the basis for which to compute the \p exponent -th power.
	 * @param exponent the exponent.
	 * @return Tensor the \p exponent -th power of each component of \p base .
	 * @see dl::pow(dl::Tensor&,float)
	 */
	[[nodiscard]] Tensor pow(Tensor&& base, float exponent) noexcept;
	/**
	 * @brief Computes the \p exponent -th power of each element in \p base and returns the resulting tensor.
	 * @details This overload does **not** compute a gradient (note that \p base is constant). Use
	 * dl::pow(dl::Tensor&,float) or dl::pow(dl::Tensor&&,float) to enable gradient calculation.
	 * 
	 * @param base the basis for which to compute the \p exponent -th power.
	 * @param exponent the exponent.
	 * @return Tensor the \p exponent -th power of each component of \p base .
	 * @see dl::pow(dl::Tensor&,float)
	 */
	[[nodiscard]] Tensor pow(const Tensor& base, float exponent) noexcept;

	[[nodiscard]] Tensor exp(Tensor& base) noexcept;
	[[nodiscard]] Tensor exp(Tensor&& base) noexcept;
	[[nodiscard]] Tensor exp(const Tensor& base) noexcept;

	[[nodiscard]] Tensor sqrt(const Tensor& x) noexcept;

	/**
	 * @brief Computes the reciprocal square root for each element in \p x.
	 * 
	 * @param x 
	 * @return 
	 */
	[[nodiscard]] Tensor rsqrt(const Tensor& x) noexcept;

	[[nodiscard]] Tensor mean(Tensor& x) noexcept;
	[[nodiscard]] Tensor mean(Tensor&& x) noexcept;
	[[nodiscard]] Tensor mean(const Tensor& x) noexcept;

	[[nodiscard]] Tensor mean(Tensor& x, size_t dim) noexcept;
	[[nodiscard]] Tensor mean(Tensor&& x, size_t dim) noexcept;
	[[nodiscard]] Tensor mean(const Tensor& x, size_t dim) noexcept;

	[[nodiscard]] Tensor sum(Tensor& x) noexcept;
	[[nodiscard]] Tensor sum(Tensor&& x) noexcept;
	[[nodiscard]] Tensor sum(const Tensor& x) noexcept;

	[[nodiscard]] Tensor sum(Tensor& x, size_t dim) noexcept;
	[[nodiscard]] Tensor sum(Tensor&& x, size_t dim) noexcept;
	[[nodiscard]] Tensor sum(const Tensor& x, size_t dim) noexcept;

	[[nodiscard]] Tensor min(Tensor& x) noexcept;
	[[nodiscard]] Tensor min(Tensor&& x) noexcept;
	[[nodiscard]] Tensor min(const Tensor& x) noexcept;

	[[nodiscard]] Tensor min(Tensor& x, size_t dim) noexcept;
	[[nodiscard]] Tensor min(Tensor&& x, size_t dim) noexcept;
	[[nodiscard]] Tensor min(const Tensor& x, size_t dim) noexcept;

	[[nodiscard]] Tensor max(Tensor& x) noexcept;
	[[nodiscard]] Tensor max(Tensor&& x) noexcept;
	[[nodiscard]] Tensor max(const Tensor& x) noexcept;

	[[nodiscard]] Tensor max(const Tensor& x, size_t dim) noexcept;

	[[nodiscard]] Tensor max(const Tensor& x, const Tensor& y) noexcept;

	/**
	 * @brief Wrapper around std::size_t to discern between var(const Tensor&, DOF) and var(const Tensor&, size_t).
	 */
	struct DOF {
		size_t dof;
	};

	[[nodiscard]] Tensor var(const Tensor& x, DOF dof = DOF{1}) noexcept;

	[[nodiscard]] Tensor var(const Tensor& x, size_t dim, DOF dof = DOF{1}) noexcept;

	[[nodiscard]] Tensor relu(Tensor& x) noexcept;
	[[nodiscard]] Tensor relu(Tensor&& x) noexcept;
	[[nodiscard]] Tensor relu(const Tensor& x) noexcept;

	[[nodiscard]] Tensor operator+(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator+(Tensor&& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator+(Tensor&& left, Tensor&& right) noexcept;
	[[nodiscard]] Tensor operator-(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator-(Tensor& left, Tensor&& right) noexcept;
	[[nodiscard]] Tensor operator-(Tensor&& left, Tensor&& right) noexcept;
	[[nodiscard]] Tensor operator-(const Tensor& left, const Tensor& right) noexcept;
	[[nodiscard]] Tensor operator*(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator*(Tensor&& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator*(Tensor&& left, Tensor&& right) noexcept;
	[[nodiscard]] Tensor operator/(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator/(Tensor& left, Tensor&& right) noexcept;
	[[nodiscard]] Tensor operator/(const Tensor& left, const Tensor& right) noexcept;

	[[nodiscard]] Tensor fma(const Tensor& factor1, const Tensor& factor2, const Tensor& summand) noexcept;

	[[nodiscard]] Tensor matmul(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor matmul(Tensor&& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor matmul(Tensor& left, Tensor&& right) noexcept;
	[[nodiscard]] Tensor matmul(Tensor&& left, Tensor&& right) noexcept;
	[[nodiscard]] Tensor matmul(const Tensor& left, const Tensor& right) noexcept;

	/**
	 * @brief Transposes the given tensor at the specified coordinates.
	 * @details Transposes the given tensor by permuting the dimensions. For example, if the input was a matrix, then
	 * \f(x \in \mathbb{R}^n\times m\f) the matrix transposition could be computed via `dl::transpose(x, {0, 1});`.
	 * 
	 * @param x the tensor to be tranposed.
	 * @param permutation the permutation to apply to the dimensions.
	 * @return a new tensor with the permutation \p permutation applied to the dimensions.
	 */
	[[nodiscard]] Tensor transpose(Tensor&& x, std::vector<int>&& permutation) noexcept;

	/**
	 * @brief Computes the softmax function of the input vector.
	 * @details Let \f(x \in \mathbb R^n\f) be a vector, the softmax is defined as
	 * \f[
	 * 	\text{softmax}\colon \mathbb R^n \to \mathbb R^n
	 * 	~\text{via}~ x\mapsto \left(\frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}\right)_{1\leq i\leq n}.
	 * \f]
	 * 
	 * @param x the input tensor.
	 * @return Tensor the softmax of \p x .
	 */
	[[nodiscard]] Tensor softmax(Tensor&& x) noexcept;
	[[nodiscard]] Tensor softmax(const Tensor& x) noexcept;
	[[nodiscard]] Tensor softmax(const Tensor& x, size_t dim) noexcept;

	std::ostream& operator<<(std::ostream&, const Tensor& tensor) noexcept;
	[[nodiscard]] bool operator==(const Tensor& left, const Tensor& right) noexcept;
	[[nodiscard]] bool allclose(const Tensor& left, const Tensor& right, float rtol = 1e-5, float atol = 1e-8) noexcept;

	/**
	 * @brief Returns the number of entries in the tensor.
	 * @details This is equivalent to the product of the shape.
	 * 
	 * @param tensor The tensor to count the number of entries for.
	 * @return size_t The number of entries in \p tensor.
	 */
	[[nodiscard]] size_t numEntries(const Tensor& tensor) noexcept;
} // namespace dl