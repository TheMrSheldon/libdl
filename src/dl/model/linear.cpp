#include <dl/device.hpp>
#include <dl/model/linear.hpp>

#include <dl/tensor/math.hpp>

using dl::Linear;
using dl::Tensor;

Linear::Linear(size_t inFeatures, size_t outFeatures, const Device& device, bool bias) noexcept
		: weights(dl::empty({outFeatures, inFeatures}, device)),
		  bias(bias ? dl::empty({outFeatures}, device) : dl::zeros({outFeatures}, device)) {
	registerParameter("weight", weights);
	if (bias)
		registerParameter("bias", this->bias);
}

Linear::Linear(size_t inFeatures, size_t outFeatures, bool bias) noexcept
		: Linear(inFeatures, outFeatures, Device::getDefault(), bias) {}

Tensor Linear::forward(Tensor& input) noexcept {
	return dl::matmul(input, dl::transpose(dl::Tensor(weights), {0, 1})) + bias;
}
Tensor Linear::forward(Tensor&& input) noexcept {
	return dl::matmul(std::move(input), dl::transpose(dl::Tensor(weights), {0, 1})) + bias;
}

// Tensor Linear::forward(Tensor&& input) const noexcept { return dl::matmul(std::move(input), weights) + bias; }