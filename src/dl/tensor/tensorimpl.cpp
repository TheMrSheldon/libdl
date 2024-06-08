#include <dl/tensor/tensorimpl.hpp>

#include <dl/device.hpp>

using dl::Device;
using dl::Shape;
using dl::TensorImpl;

TensorImpl::TensorImpl(const Device& device, bool requiresGrad) noexcept
		: _device(device), _requiresGrad(requiresGrad) {}

const Device& TensorImpl::device() const noexcept { return _device; }

void TensorImpl::setRequiresGrad(bool requiresGrad) noexcept { _requiresGrad = requiresGrad; }
bool TensorImpl::requiresGrad() const noexcept { return _requiresGrad; }

void TensorImpl::backward(bool enableAutodiff) noexcept {
	grad = dl::constant(1.0f, device());
	grad->setRequiresGrad(enableAutodiff);
	assert(gradfn != nullptr);
	gradfn(grad);
}