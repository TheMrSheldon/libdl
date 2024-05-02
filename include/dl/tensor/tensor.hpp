#pragma once

#include "shape.hpp"
#include "tensorptr.hpp"

#include <functional>
#include <iostream>

namespace dl {
	class Device;

	class Tensor {
		using GradFn = std::function<void(TensorPtr)>;

	private:
		bool _requiresGrad;
		Device const& _device;

	public:
		GradFn gradfn = nullptr;
		TensorPtr grad = nullptr;
		/** \todo add retain graph option **/

	protected:
		Tensor(Device const& device, bool requiresGrad) noexcept;

	public:
		/**
		 * @brief Creates a copy of this tensor on the requested device and returns a pointer to it. If the new device
		 * is the same as the device the tensor is already on, the tensor will be copied.
		 * 
		 * @param other The device to copy the tensor onto.
		 * @return The newly created tensor on the specified device.
		 * @see Tensor::device()
		 */
		TensorPtr to(Device const& other) const noexcept;
		/**
		 * @brief Returns the device this tensor is stored on.
		 * 
		 * @return The device this tensor is stored on.
		 * @see Tensor::to(const Device& other)
		 */
		Device const& device() const noexcept;

		/**
		 * @brief Set this tensors requirements for a gradient.
		 * 
		 * @param requiresGrad wether the tensor requires a gradient.
		 * @see Tensor::requiresGrad()
		 */
		void setRequiresGrad(bool requiresGrad) noexcept;
		/**
		 * @brief Returns true iff this tensor requires a gradient, i.e., needs to be updated during backpropagation.
		 * 
		 * @return True iff this tensor requires a gradient, i.e., needs to be updated during backpropagation.
		 * @see Tensor::setRequiresGrad()
		 */
		bool requiresGrad() const noexcept;

		void backward(bool enableAutodiff = false) noexcept;

		const TensorPtr gradient() const noexcept { return grad; }
		void discardGradient() noexcept {
			gradfn = nullptr;
			grad = nullptr;
		}

		virtual std::ostream& writeToStream(std::ostream& stream) const noexcept = 0;
		virtual TensorPtr add(const TensorPtr& other) const noexcept = 0;
		virtual TensorPtr sub(const TensorPtr& other) const noexcept = 0;
		virtual TensorPtr mul(const TensorPtr& other) const noexcept = 0;
		virtual TensorPtr div(const TensorPtr& other) const noexcept = 0;

		virtual TensorPtr matmul(const TensorPtr& other) const noexcept = 0;

		virtual TensorPtr pow(float exponent) const noexcept = 0;
		virtual TensorPtr mean() const noexcept = 0;

		virtual void mul_inplace(const TensorPtr& other) noexcept = 0;

		virtual TensorPtr clone() const noexcept = 0;

		virtual Shape shape() const noexcept = 0;
		virtual size_t shape(size_t dim) const noexcept = 0;

		virtual TensorPtr flatten() const noexcept = 0;
	};
} // namespace dl