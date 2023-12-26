#include <dl/tensor/tensor.hpp>

#include <dl/device.hpp>

using Device = dl::Device;
using Shape = dl::Shape;
using Tensor = dl::Tensor;
using TensorPtr = dl::TensorPtr;

Tensor::Tensor(const Device& device, bool requiresGrad) noexcept : _device(device), _requiresGrad(requiresGrad) {}

void Tensor::setRequiresGrad(bool requiresGrad) noexcept { _requiresGrad = requiresGrad; }
bool Tensor::requiresGrad() const noexcept { return _requiresGrad; }

const Device& Tensor::device() const noexcept { return _device; }

/*
TensorPtr dl::empty(Shape size, const Device& device) { return device.empty(size); }
TensorPtr dl::zero(Shape size, const Device& device) { return device.zero(size); }
TensorPtr dl::ones(Shape size, const Device& device) { return device.ones(size); }

TensorPtr constant(int value, Device const& device);
TensorPtr constant(float value, Device const& device);
TensorPtr constant(double value, Device const& device) { return device.constant(); }*/