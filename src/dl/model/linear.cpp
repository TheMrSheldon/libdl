#include <dl/device.hpp>
#include <dl/model/linear.hpp>

#include <dl/tensor/math.hpp>

using dl::Linear;
using dl::TensorPtr;

Linear::Linear(size_t inFeatures, size_t outFeatures, const Device& device, bool bias) noexcept
		: _weights(dl::empty({outFeatures, inFeatures}, device)),
		  _bias(bias ? dl::empty({outFeatures}, device) : dl::zeros({outFeatures}, device)) {
	registerParameter("weight", _weights);
	if (bias)
		registerParameter("bias", _bias);
}

Linear::Linear(size_t inFeatures, size_t outFeatures, bool bias) noexcept
		: Linear(inFeatures, outFeatures, Device::getDefault(), bias) {}

TensorPtr Linear::operator()(TensorPtr input) noexcept {
	return dl::matmul(input, dl::transpose(_weights, {-1, -2})) + _bias;
}