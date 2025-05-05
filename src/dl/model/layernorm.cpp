#include <dl/model/layernorm.hpp>

#include <dl/tensor/math.hpp>

using dl::LayerNorm;
using dl::TensorPtr;

LayerNorm::LayerNorm(Shape normShape, const Device& device) noexcept
		: beta(dl::zeros(normShape, device)), gamma(dl::ones(normShape, device)) {
	registerParameter("beta", beta);
	registerParameter("gamma", gamma);
}

TensorPtr LayerNorm::operator()(TensorPtr input) noexcept {
	auto numerator = input - dl::reshape(dl::mean(input, 1), {-1, 1});
	auto denominator = dl::reshape(dl::rsqrt(dl::var(input, 1, dl::DOF{0}) + 1e-5f), {-1, 1});
	return (numerator * denominator) * gamma + beta;
}
