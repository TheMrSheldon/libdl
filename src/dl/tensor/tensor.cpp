#include <dl/tensor/tensor.hpp>

#include <dl/device.hpp>

using Device = dl::Device;
using Shape = dl::Shape;
using Tensor = dl::Tensor;
using TensorPtr = dl::TensorPtr;

Tensor::Tensor(const Device& device, bool requiresGrad) noexcept : _device(device), _requiresGrad(requiresGrad) {}

const Device& Tensor::device() const noexcept { return _device; }

void Tensor::setRequiresGrad(bool requiresGrad) noexcept { _requiresGrad = requiresGrad; }
bool Tensor::requiresGrad() const noexcept { return _requiresGrad; }

void Tensor::backward(bool enableAutodiff) noexcept {
	/** \todo implement autodiff **/
	grad = dl::constant(1.0f, device());
	gradfn(grad);
}