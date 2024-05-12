#include <dl/model/layernorm.hpp>

#include <dl/tensor/math.hpp>

using dl::LayerNorm;
using dl::Tensor;

LayerNorm::LayerNorm(Shape normShape, const Device& device) noexcept
		: beta(dl::empty(normShape, device)), gamma(dl::empty(normShape, device)) {
	registerParameter("beta", beta);
	registerParameter("gamma", gamma);
}

Tensor LayerNorm::forward(Tensor& input) noexcept {
	return ((input - dl::mean(input)) * dl::rsqrt(dl::var(input) + dl::constant(1e-5))) * gamma + beta;
}
Tensor LayerNorm::forward(Tensor&& input) noexcept {
	/** \todo currently, this will result in memory errors for the backward pass since input will be deleted. **/
	return ((input - dl::mean(input)) * dl::rsqrt(dl::var(input) + dl::constant(1e-5))) * gamma + beta;
}