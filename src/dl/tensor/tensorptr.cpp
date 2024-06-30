#include <dl/device.hpp>
#include <dl/tensor/tensorimpl.hpp>
#include <dl/tensor/tensorptr.hpp>

using dl::Device;
using dl::TensorPtr;

TensorPtr::TensorPtr(const TensorPtr& other) : data(other.data) {}
TensorPtr::TensorPtr(int value) : TensorPtr(std::move(dl::constant(value))) {}
TensorPtr::TensorPtr(float value) : TensorPtr(std::move(dl::constant(value))) {}
TensorPtr::TensorPtr(double value) : TensorPtr(std::move(dl::constant(value))) {}
// TensorPtr::TensorPtr(InitializerTensor<int> value) : TensorPtr(std::move(dl::constant(std::move(value)))) {}
TensorPtr::TensorPtr(InitializerTensor<float> value) : TensorPtr(std::move(dl::constant(std::move(value)))) {}

// TensorPtr::TensorPtr(InitializerTensor<double> value) : TensorPtr(std::move(dl::constant(std::move(value)))) {}

TensorPtr& TensorPtr::operator=(const TensorPtr& other) {
	data = other.data;
	return *this;
}
TensorPtr& TensorPtr::operator=(TensorPtr&& other) {
	this->data = std::move(other.data);
	return *this;
}
TensorPtr& TensorPtr::operator=(std::nullptr_t p) {
	data = nullptr;
	return *this;
}
TensorPtr& TensorPtr::operator=(int value) {
	auto& device = (data == nullptr) ? Device::getDefault() : data->device();
	return *this = dl::constant(value, data->device());
}
TensorPtr& TensorPtr::operator=(float value) {
	auto& device = (data == nullptr) ? Device::getDefault() : data->device();
	return *this = dl::constant(value, data->device());
}
TensorPtr& TensorPtr::operator=(double value) {
	auto& device = (data == nullptr) ? Device::getDefault() : data->device();
	return *this = dl::constant(value, data->device());
}
/*TensorPtr& TensorPtr::operator=(InitializerTensor<int>&& value) {
	auto& device = (data == nullptr) ? Device::getDefault() : data->device();
	return *this = dl::constant(std::move(value), data->device());
}*/
TensorPtr& TensorPtr::operator=(InitializerTensor<float>&& value) {
	auto& device = (data == nullptr) ? Device::getDefault() : data->device();
	return *this = dl::constant(std::move(value), data->device());
}
/*TensorPtr& TensorPtr::operator=(InitializerTensor<double>&& value) {
	auto& device = (data == nullptr) ? Device::getDefault() : data->device();
	return *this = dl::constant(std::move(value), data->device());
}*/
