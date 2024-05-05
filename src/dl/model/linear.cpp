#include <dl/device.hpp>
#include <dl/model/linear.hpp>

#include <dl/tensor/math.hpp>

using dl::Linear;
using dl::Tensor;

Linear::Linear(unsigned inFeatures, unsigned outFeatures, const Device& device, bool bias) noexcept
		: weights(dl::empty({inFeatures, outFeatures}, device)),
		  bias(bias ? dl::empty({outFeatures}, device) : dl::zero({outFeatures}, device)) {
	registerParameter(weights);
	if (bias)
		registerParameter(this->bias);
}

Linear::Linear(unsigned inFeatures, unsigned outFeatures, bool bias) noexcept
		: Linear(inFeatures, outFeatures, Device::getDefault(), bias) {}

Tensor Linear::forward(Tensor& input) noexcept { return dl::matmul(input, weights) + bias; }

// Tensor Linear::forward(Tensor& input) const noexcept { return dl::matmul(input, weights) + bias; }