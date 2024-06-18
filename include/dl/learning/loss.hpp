/**
 * @file loss.hpp
 * @brief Contains some general purpose loss objectives.
 */

#ifndef DL_LEARNING_LOSS_HPP
#define DL_LEARNING_LOSS_HPP

#include "../tensor/math.hpp"

namespace dl::loss {
	/**
	 * @brief Mean Square Error
	 * @details Implements the mean square error loss function. Let
	 * \f(\vec{x}=\begin{pmatrix}x_1 & \dots & x_n\end{pmatrix}\f) be the model's outputs and
	 * \f(\vec{y}=\begin{pmatrix}y_1 & \dots & y_n\end{pmatrix}\f) the targets, the margin square error is defined as
	 * \f[\text{MSE}(\vec{x}, \vec{y}) := \frac{1}{n}\sum_{i=1}^n (x_i - y_i)^2.\f]
	 * 
	 * @param x the model outputs
	 * @param y the targets (desired outputs)
	 * @return the mean square error
	 */
	TensorPtr mse(TensorPtr x, TensorPtr y) noexcept { return dl::mean(dl::pow(x - y, 2.0f)); }

	/**
	 * @brief Binary Cross Entropy Loss
	 * @details Implements the binary cross entropy loss function. Let
	 * \f(\vec{x}=\begin{pmatrix}x_1 & \dots & x_n\end{pmatrix}\f) be the model's outputs and
	 * \f(\vec{y}=\begin{pmatrix}y_1 & \dots & y_n\end{pmatrix}\f) the targets, the binary cross entropy loss is defined
	 * as
	 * \f[\text{BCE}(\vec{x}, \vec{y}) := -\frac{1}{n}\sum_{i=1}^n (y_i\log x_i + (1-y_i)\log(1-x_i)).\f]
	 * 
	 * @param x the model outputs
	 * @param y the targets (desired outputs)
	 * @return the binary cross entropy 
	 */
	TensorPtr bce(TensorPtr x, TensorPtr y) noexcept { return -dl::mean(y * dl::log(x) + (1 - y) * dl::log(1 - x)); }
} // namespace dl::loss

#endif