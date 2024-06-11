#pragma once

#include "math.hpp"
#include "shape.hpp"

#include <any>
#include <functional>
#include <iostream>

namespace dl {
	class Device;

	class TensorImpl {
		using GradFn = std::function<void(TensorPtr&)>;

	private:
		bool _requiresGrad;
		Device const& _device;

	public:
		GradFn gradfn = nullptr;
		TensorPtr grad = nullptr;

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
		TensorPtr to(Device const& other) const noexcept;
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

		const TensorPtr& gradient() const noexcept { return grad; }
		void discardGradient() noexcept {
			gradfn = nullptr;
			grad = nullptr;
		}

		virtual std::ostream& writeToStream(std::ostream& stream) const noexcept = 0;
		virtual bool operator==(const TensorPtr& other) const noexcept = 0;
		virtual bool allclose(const TensorPtr& other, float rtol = 1e-5, float atol = 1e-8) const noexcept = 0;

		virtual TensorPtr add(const TensorPtr& other) const noexcept = 0;
		virtual TensorPtr sub(const TensorPtr& other) const noexcept = 0;
		virtual TensorPtr mul(const TensorPtr& other) const noexcept = 0;
		virtual TensorPtr div(const TensorPtr& other) const noexcept = 0;

		/**
		 * @brief Performs "fused multiply and add".
		 * @details Multiplies this tensor by \p factor and adds \p summand to the result. This specialized function
		 * exists since some devices (e.g., CUDA and some SIMD instruction sets) provide such a function.
		 * 
		 * @param factor the factor to multiply with this tensor.
		 * @param summand the summand to add to the product of this with \p factor.
		 * @return the result.
		 */
		virtual TensorPtr fma(const TensorPtr& factor, const TensorPtr& summand) const noexcept = 0;
		virtual TensorPtr matmul(const TensorPtr& other) const noexcept = 0;
		virtual TensorPtr transpose(std::vector<size_t>&& permutation) const noexcept = 0;

		// Powers:
		virtual TensorPtr pow(float exponent) const noexcept = 0;
		virtual TensorPtr exp() const noexcept = 0;
		virtual TensorPtr sqrt() const noexcept = 0;
		virtual TensorPtr rsqrt() const noexcept = 0;

		// Statistical functions:
		virtual TensorPtr mean() const noexcept = 0;
		virtual TensorPtr mean(int dim, bool keepdim) const noexcept = 0;
		virtual TensorPtr sum() const noexcept = 0;
		virtual TensorPtr sum(int dim, bool keepdim) const noexcept = 0;
		virtual TensorPtr min() const noexcept = 0;
		virtual TensorPtr min(int dim, bool keepdim) const noexcept = 0;
		/**
		 * @brief Computes the element-wise minimum between this tensor and the other.
		 * 
		 * @param other 
		 * @return TensorPtr 
		 */
		virtual TensorPtr min(const TensorPtr& other) const noexcept = 0;
		virtual TensorPtr max() const noexcept = 0;
		virtual TensorPtr max(int dim, bool keepdim) const noexcept = 0;
		virtual TensorPtr max(const TensorPtr& other) const noexcept = 0;
		virtual TensorPtr var(DOF dof) const noexcept = 0;
		virtual TensorPtr var(int dim, DOF dof) const noexcept = 0;

		virtual TensorPtr erf() const noexcept = 0;

		virtual void mul_inplace(const TensorPtr& other) noexcept = 0;
		/**
		 * @brief Reshapes the tensor to fit the specified size.
		 * @details At most one of the entries in the shape may be -1 and will then be inferred by the remaining shape.
		 * The element count may not change through reshaping.
		 * 
		 * @param shape The new shape for the tensor.
		 */
		virtual void reshape(SShape shape) noexcept = 0;

		virtual TensorPtr clone() const noexcept = 0;

		virtual Shape shape() const noexcept = 0;
		virtual size_t shape(int dim) const noexcept = 0;
		size_t numDim() const noexcept { return shape().size(); }

		virtual TensorPtr flatten() const noexcept = 0;

		/**
		 * @brief Writes this tensor's data into the byte array.
		 * @details 
		 * 
		 * @param buffer The byte to write to. If it is a nullpointer, no data will be written but the number of bytes
		 * that the tensor uses will still be returned.
		 * @param buflen The maximum number of bytes to write.
		 * @return The number of bytes that were written. If the buffer was too small, 0 is returned.
		 */
		virtual size_t toBytes(char* buffer, size_t buflen) const noexcept = 0;
	};
} // namespace dl