#pragma once

#include <vector>
#include <memory>

namespace dl {
	class Device;

	using Shape = std::vector<unsigned>;
	using TensorPtr = std::shared_ptr<Tensor>;

	class Tensor {
	private:
		bool _requiresGrad;
		const Device& _device;
	protected:
		Tensor(const Device& device, bool requiresGrad) noexcept;
	public:
		/**
		 * @brief Creates a copy of this tensor on the requested device and returns a pointer to it. If the new device
		 * is the same as the device the tensor is already on, the tensor will be copied.
		 * 
		 * @param other The device to copy the tensor onto.
		 * @return The newly created tensor on the specified device.
		 * @see Tensor::device()
		 */
		TensorPtr to(const Device& other) const noexcept;
		/**
		 * @brief Returns the device this tensor is stored on.
		 * 
		 * @return The device this tensor is stored on.
		 * @see Tensor::to(const Device& other)
		 */
		const Device& device() const noexcept;

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
	};

	TensorPtr operator+(TensorPtr left, TensorPtr right);
	TensorPtr operator-(TensorPtr left, TensorPtr right);
	TensorPtr operator*(TensorPtr left, TensorPtr right);
	TensorPtr operator/(TensorPtr left, TensorPtr right);

	TensorPtr emptyTensor(Shape size, Device& device);
	TensorPtr zeroTensor(Shape size, Device& device);

} // namespace dl
