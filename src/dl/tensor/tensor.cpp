#include <dl/tensor/tensor.hpp>

using Device = dl::Device;
using Shape = dl::Shape;
using Tensor = dl::Tensor;
using TensorPtr = dl::TensorPtr;

Tensor::Tensor(const Device& device, bool requiresGrad) noexcept : _device(device), _requiresGrad(requiresGrad) {}

void Tensor::setRequiresGrad(bool requiresGrad) noexcept { _requiresGrad = requiresGrad; }
bool Tensor::requiresGrad() const noexcept { return requiresGrad; }

const Device& Tensor::device() const noexcept { return _device; }


TensorPtr dl::operator+(TensorPtr left, TensorPtr right) {
	/** \todo implement **/
	throw std::exception("Not yet implemented");
}
TensorPtr dl::operator-(TensorPtr left, TensorPtr right) {
	/** \todo implement **/
	throw std::exception("Not yet implemented");
}
TensorPtr dl::operator*(TensorPtr left, TensorPtr right) {
	/** \todo implement **/
	throw std::exception("Not yet implemented");
}
TensorPtr dl::operator/(TensorPtr left, TensorPtr right) {
	/** \todo implement **/
	throw std::exception("Not yet implemented");
}

TensorPtr dl::emptyTensor(Shape size, Device& device) {
	/** \todo implement **/
	throw std::exception("Not yet implemented");
}
TensorPtr dl::zeroTensor(Shape size, Device& device) {
	/** \todo implement **/
	throw std::exception("Not yet implemented");
}