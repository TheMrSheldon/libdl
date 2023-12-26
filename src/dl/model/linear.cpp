#include <dl/device.hpp>
#include <dl/model/linear.hpp>

#include <dl/tensor/math.hpp>

using Linear = dl::Linear;
using TensorPtr = dl::TensorPtr;

Linear::Linear(unsigned inFeatures, unsigned outFeatures, const Device& device, bool bias) noexcept
		: weights(dl::empty({inFeatures, outFeatures}, device)),
		  bias(bias ? dl::empty({outFeatures}, device) : dl::zero({outFeatures}, device)) {
	registerParameter(weights);
	if (bias)
		registerParameter(this->bias);
}

Linear::Linear(unsigned inFeatures, unsigned outFeatures, bool bias) noexcept
		: Linear(inFeatures, outFeatures, Device::getDefault(), bias) {}

TensorPtr Linear::forward(TensorPtr input) noexcept { return dl::matmul(input, weights) + bias; }

TensorPtr Linear::forward(TensorPtr input) const noexcept { return dl::matmul(input, weights) + bias; }