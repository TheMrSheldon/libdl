#pragma once

#include "math.hpp"
#include "shape.hpp"

#include <any>
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
		virtual bool operator==(const Tensor& other) const noexcept = 0;
		virtual bool allclose(const Tensor& other, float rtol = 1e-5, float atol = 1e-8) const noexcept = 0;

		virtual Tensor add(const Tensor& other) const noexcept = 0;
		virtual Tensor sub(const Tensor& other) const noexcept = 0;
		virtual Tensor mul(const Tensor& other) const noexcept = 0;
		virtual Tensor div(const Tensor& other) const noexcept = 0;

		/**
		 * @brief Performs "fused multiply and add".
		 * @details Multiplies this tensor by \p factor and adds \p summand to the result. This specialized function
		 * exists since some devices (e.g., CUDA and some SIMD instruction sets) provide such a function.
		 * 
		 * @param factor the factor to multiply with this tensor.
		 * @param summand the summand to add to the product of this with \p factor.
		 * @return the result.
		 */
		virtual Tensor fma(const Tensor& factor, const Tensor& summand) const noexcept = 0;
		virtual Tensor matmul(const Tensor& other) const noexcept = 0;
		virtual Tensor transpose(std::vector<size_t>&& permutation) const noexcept = 0;

		// Powers:
		virtual Tensor pow(float exponent) const noexcept = 0;
		virtual Tensor exp() const noexcept = 0;
		virtual Tensor sqrt() const noexcept = 0;
		virtual Tensor rsqrt() const noexcept = 0;

		// Statistical functions:
		virtual Tensor mean() const noexcept = 0;
		virtual Tensor mean(size_t dim) const noexcept = 0;
		virtual Tensor sum() const noexcept = 0;
		virtual Tensor sum(size_t dim) const noexcept = 0;
		virtual Tensor min() const noexcept = 0;
		virtual Tensor min(size_t dim) const noexcept = 0;
		/**
		 * @brief Computes the element-wise minimum between this tensor and the other.
		 * 
		 * @param other 
		 * @return Tensor 
		 */
		virtual Tensor min(const Tensor& other) const noexcept = 0;
		virtual Tensor max() const noexcept = 0;
		virtual Tensor max(size_t dim) const noexcept = 0;
		virtual Tensor max(const Tensor& other) const noexcept = 0;
		virtual Tensor var(DOF dof) const noexcept = 0;
		virtual Tensor var(size_t dim, DOF dof) const noexcept = 0;

		virtual void mul_inplace(const Tensor& other) noexcept = 0;
		/**
		 * @brief Reshapes the tensor to fit the specified size.
		 * @details At most one of the entries in the shape may be -1 and will then be inferred by the remaining shape.
		 * The element count may not change through reshaping.
		 * 
		 * @param shape The new shape for the tensor.
		 */
		virtual void reshape(std::vector<int> shape) noexcept = 0;

		virtual Tensor clone() const noexcept = 0;

		virtual Shape shape() const noexcept = 0;
		virtual size_t shape(size_t dim) const noexcept = 0;
		size_t numDim() const noexcept { return shape().size(); }

		virtual Tensor flatten() const noexcept = 0;
	};
} // namespace dl