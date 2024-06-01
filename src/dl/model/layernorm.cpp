#include <dl/model/layernorm.hpp>

#include <dl/tensor/math.hpp>

using dl::LayerNorm;
using dl::Tensor;

LayerNorm::LayerNorm(Shape normShape, const Device& device) noexcept
		: beta(dl::zeros(normShape, device)), gamma(dl::ones(normShape, device)) {
	registerParameter("beta", beta);
	registerParameter("gamma", gamma);
}

Tensor LayerNorm::forward(Tensor& input) noexcept {
	auto numerator = input - dl::reshape(dl::mean(input, 1), {-1, 1});
	auto denominator = dl::reshape(dl::rsqrt(dl::var(input, 1, dl::DOF{0}) + dl::constant(1e-5)), {-1, 1});
	return (std::move(numerator) * std::move(denominator)) * gamma + beta;
}
Tensor LayerNorm::forward(Tensor&& input) noexcept {
	/** \todo currently, this will result in memory errors for the backward pass since input will be deleted. **/
	auto numerator = input - dl::reshape(dl::mean(input, 1), {-1, 1});
	auto denominator = dl::reshape(dl::rsqrt(dl::var(input, 1, dl::DOF{0}) + dl::constant(1e-5)), {-1, 1});
	return (std::move(numerator) * std::move(denominator)) * gamma + beta;
}