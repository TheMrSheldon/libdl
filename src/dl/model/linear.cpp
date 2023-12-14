#include <dl/device.hpp>
#include <dl/model/linear.hpp>

using Linear = dl::Linear;
using TensorPtr = dl::TensorPtr;

Linear::Linear(unsigned inFeatures, unsigned outFeatures, Device& device, bool bias = true) noexcept
		: weights(dl::emptyTensor({inFeatures, outFeatures}, device)),
		  bias(bias ? dl::emptyTensor({outFeatures}, device) : dl::zeroTensor({outFeatures}, device)) {
	registerWeights(weights);
	registerWeights(this->bias);
}

Linear::Linear(unsigned inFeatures, unsigned outFeatures, bool bias = true) noexcept
		: Linear(inFeatures, outFeatures, Device::getDefault(), bias) {}

TensorPtr Linear::forward(TensorPtr input) noexcept { return input * weights + bias; }