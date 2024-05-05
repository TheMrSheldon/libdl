#pragma once

#include "shape.hpp"
#include "tensor.hpp"

#include <functional>
#include <iostream>

namespace dl {
	class Device;

	class TensorImpl {
		using GradFn = std::function<void(Tensor&)>;

	private:
		bool _requiresGrad;
		Device const& _device;

	public:
		GradFn gradfn = nullptr;
		Tensor grad = nullptr;
		/** \todo add retain graph option **/

	protected:
		TensorImpl(Device const& device, bool requiresGrad) noexcept;

	public:
		/**
		 * @brief Creates a copy of this tensor on the requested device and returns a pointer to it. If the new device
		 * is the same as the device the tensor is already on, the tensor will be copied.
		 * 
		 * @param other The device to copy the tensor onto.
		 * @return The newly created tensor on the specified device.
		 * @see TensorImpl::device()
		 */
		Tensor to(Device const& other) const noexcept;
		/**
		 * @brief Returns the device this tensor is stored on.
		 * 
		 * @return The device this tensor is stored on.
		 * @see TensorImpl::to(const Device& other)
		 */
		Device const& device() const noexcept;

		/**
		 * @brief Set this tensors requirements for a gradient.
		 * 
		 * @param requiresGrad wether the tensor requires a gradient.
		 * @see TensorImpl::requiresGrad()
		 */
		void setRequiresGrad(bool requiresGrad) noexcept;
		/**
		 * @brief Returns true iff this tensor requires a gradient, i.e., needs to be updated during backpropagation.
		 * 
		 * @return True iff this tensor requires a gradient, i.e., needs to be updated during backpropagation.
		 * @see TensorImpl::setRequiresGrad()
		 */
		bool requiresGrad() const noexcept;

		void backward(bool enableAutodiff = false) noexcept;

		const Tensor gradient() const noexcept { return grad; }
		void discardGradient() noexcept {
			gradfn = nullptr;
			grad = nullptr;
		}

		virtual std::ostream& writeToStream(std::ostream& stream) const noexcept = 0;
		virtual Tensor add(const Tensor& other) const noexcept = 0;
		virtual Tensor sub(const Tensor& other) const noexcept = 0;
		virtual Tensor mul(const Tensor& other) const noexcept = 0;
		virtual Tensor div(const Tensor& other) const noexcept = 0;

		virtual Tensor matmul(const Tensor& other) const noexcept = 0;

		virtual Tensor pow(float exponent) const noexcept = 0;
		virtual Tensor mean() const noexcept = 0;

		virtual void mul_inplace(const Tensor& other) noexcept = 0;

		virtual Tensor clone() const noexcept = 0;

		virtual Shape shape() const noexcept = 0;
		virtual size_t shape(size_t dim) const noexcept = 0;

		virtual Tensor flatten() const noexcept = 0;
	};
} // namespace dl