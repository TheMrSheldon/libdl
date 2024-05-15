#include <dl/device.hpp>
#include <dl/tensor/tensor.hpp>
#include <dl/tensor/tensorimpl.hpp>

using dl::Tensor;

Tensor::Tensor(const Tensor& other) : Tensor(std::move(other->clone())) {}
Tensor::Tensor(int value) : Tensor(std::move(dl::constant(value))) {}
Tensor::Tensor(float value) : Tensor(std::move(dl::constant(value))) {}
Tensor::Tensor(double value) : Tensor(std::move(dl::constant(value))) {}
// Tensor::Tensor(InitializerTensor<int> value) : Tensor(std::move(dl::constant(std::move(value)))) {}
Tensor::Tensor(InitializerTensor<float> value) : Tensor(std::move(dl::constant(std::move(value)))) {}

// Tensor::Tensor(InitializerTensor<double> value) : Tensor(std::move(dl::constant(std::move(value)))) {}

Tensor& Tensor::operator=(const Tensor& other) {
	*this = std::move(other->clone());
	return *this;
}