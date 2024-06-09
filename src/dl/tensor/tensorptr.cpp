#include <dl/device.hpp>
#include <dl/tensor/tensorimpl.hpp>
#include <dl/tensor/tensorptr.hpp>

using dl::TensorPtr;

TensorPtr::TensorPtr(const TensorPtr& other) : TensorPtr(std::move(other->clone())) {}
TensorPtr::TensorPtr(int value) : TensorPtr(std::move(dl::constant(value))) {}
TensorPtr::TensorPtr(float value) : TensorPtr(std::move(dl::constant(value))) {}
TensorPtr::TensorPtr(double value) : TensorPtr(std::move(dl::constant(value))) {}
// TensorPtr::TensorPtr(InitializerTensor<int> value) : TensorPtr(std::move(dl::constant(std::move(value)))) {}
TensorPtr::TensorPtr(InitializerTensor<float> value) : TensorPtr(std::move(dl::constant(std::move(value)))) {}

// TensorPtr::TensorPtr(InitializerTensor<double> value) : TensorPtr(std::move(dl::constant(std::move(value)))) {}

TensorPtr& TensorPtr::operator=(const TensorPtr& other) {
	*this = std::move(other->clone());
	return *this;
}