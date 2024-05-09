/**
 * @file math.hpp
 * @brief Implements auto-diff enabled wrappers around their concrete tensor implementations.
 */

#pragma once

#include "tensor.hpp"
#include <iostream>

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

	[[nodiscard]] Tensor mean(Tensor& x) noexcept;
	[[nodiscard]] Tensor mean(Tensor&& x) noexcept;

	[[nodiscard]] Tensor sum(Tensor& x) noexcept;

	[[nodiscard]] Tensor min(Tensor& x, float bound) noexcept;
	[[nodiscard]] Tensor max(Tensor& x, float bound) noexcept;

	[[nodiscard]] Tensor relu(Tensor& x) noexcept;

	[[nodiscard]] Tensor operator+(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator-(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator*(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator/(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor operator/(Tensor& left, Tensor right) noexcept;

	[[nodiscard]] Tensor matmul(Tensor& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor matmul(Tensor&& left, Tensor& right) noexcept;
	[[nodiscard]] Tensor matmul(Tensor& left, Tensor&& right) noexcept;
	[[nodiscard]] Tensor matmul(Tensor&& left, Tensor&& right) noexcept;

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

	std::ostream& operator<<(std::ostream&, const Tensor& tensor) noexcept;
	[[nodiscard]] bool operator==(const Tensor& left, const Tensor& right) noexcept;
} // namespace dl