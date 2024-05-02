#include <dl/device.hpp>
#include <dl/tensor/tensorptr.hpp>

/** \todo move this to tensorptr.hpp **/
dl::TensorPtr::TensorPtr(int value) : dl::TensorPtr(std::move(dl::constant(value))) {}
dl::TensorPtr::TensorPtr(float value) : dl::TensorPtr(std::move(dl::constant(value))) {}
dl::TensorPtr::TensorPtr(double value) : dl::TensorPtr(std::move(dl::constant(value))) {}
// dl::TensorPtr::TensorPtr(std::initializer_list<int> value) : dl::TensorPtr(std::move(dl::constant(std::move(value)))) {}
dl::TensorPtr::TensorPtr(std::initializer_list<float> value)
		: dl::TensorPtr(std::move(dl::constant(std::move(value)))) {}

// dl::TensorPtr::TensorPtr(std::initializer_list<double> value)
//      : dl::TensorPtr(std::move(dl::constant(std::move(value)))) {}