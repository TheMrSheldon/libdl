#include <dl/model/transformer/transformer.hpp>

#include <dl/tensor/math.hpp>

#include <cmath>

using dl::Linear;
using dl::Tensor;
using dl::Transformer;

Transformer::Transformer(size_t dimModel, size_t numHeads, size_t dimKeys, size_t dimValues) noexcept
		: dimModel(dimModel), dimKeys(dimKeys), dimValues(dimValues), weightOut(numHeads * dimValues, dimModel),
		  heads() {
	for (size_t i = 0; i < numHeads; ++i)
		heads.emplace_back(dimModel, dimKeys, dimValues);
}

Tensor Transformer::multiHeadAttention(Tensor&& query, Tensor&& key, Tensor&& value) noexcept {
	std::vector<Tensor> attn;
	for (auto&& head : heads)
		attn.emplace_back(head(query, key, value));
	return weightOut(dl::concat(std::move(attn)));
}

Transformer::AttnHead::AttnHead(size_t dimModel, size_t dimKeys, size_t dimValues) noexcept
		: weightQuery(dimModel, dimKeys), weightKey(dimModel, dimKeys), weightValue(dimModel, dimValues),
		  dimKeysInvSqrt(std::sqrt(dimKeys)) {}

Tensor Transformer::AttnHead::forward(Tensor query, Tensor key, Tensor value) {
	return scaledDotProductAttention(
			weightQuery(std::move(query)), weightKey(std::move(key)), weightValue(std::move(value))
	);
}

Tensor Transformer::AttnHead::scaledDotProductAttention(Tensor&& query, Tensor&& key, Tensor&& value) noexcept {
	return dl::matmul(dl::softmax(dl::matmul(query, dl::transpose(key)) * dimKeysInvSqrt), value);
}