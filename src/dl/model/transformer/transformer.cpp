#include <dl/model/transformer/transformer.hpp>

#include <dl/tensor/math.hpp>

#include <cmath>
#include <format>

using dl::Linear;
using dl::Tensor;
using dl::Transformer;
using dl::TransformerEncoder;

Transformer::Transformer(TransformerConf conf) noexcept
		: conf(conf), weightOut(conf.numAttnHeads * conf.dimensions.value, conf.dimensions.model), encoders() {
	for (size_t i = 0; i < conf.numEncoders; ++i) {
		const auto& encoder = encoders.emplace_back(std::make_unique<TransformerEncoder>(conf));
		registerSubmodel(std::format("encoder.layer.{}", i), *encoder);
	}
}

Tensor Transformer::forward(Tensor& input) {
	/** \todo: implement */
	return nullptr;
}

TransformerEncoder::TransformerEncoder(TransformerConf conf) noexcept
		: conf(conf), weightQuery(conf.dimensions.model, conf.numAttnHeads * conf.dimensions.key),
		  weightKey(conf.dimensions.model, conf.numAttnHeads * conf.dimensions.key),
		  weightValue(conf.dimensions.model, conf.numAttnHeads * conf.dimensions.value),
		  weightOut(conf.numAttnHeads * conf.dimensions.value, conf.dimensions.model), mhaNorm({conf.dimensions.model}),
		  weightIntermed({conf.dimensions.model, conf.dimensions.inner}),
		  weightIntermedOut({conf.dimensions.inner, conf.dimensions.model}), ffnNorm({conf.dimensions.model}),
		  dimKeysInvSqrt(std::sqrt(conf.dimensions.key)) {
	registerSubmodel("attention.self.key", weightKey);
	registerSubmodel("attention.self.query", weightQuery);
	registerSubmodel("attention.self.value", weightValue);
	registerSubmodel("attention.output.dense", weightOut);
	registerSubmodel("attention.output.LayerNorm", mhaNorm);
	registerSubmodel("intermediate.dense", weightIntermed);
	registerSubmodel("output.dense", weightIntermedOut);
	registerSubmodel("output.LayerNorm", ffnNorm);
}

Tensor TransformerEncoder::multiHeadAttention(Tensor&& query, Tensor&& key, Tensor&& value) noexcept {
	auto attn = dl::transpose(scaledDotProductAttention(std::move(query), std::move(key), std::move(value)), {0, 1});
	attn->reshape({-1, (int)(conf.numAttnHeads * conf.dimensions.value)});
	return weightOut.forward(std::move(attn));
}

Tensor TransformerEncoder::scaledDotProductAttention(Tensor&& query, Tensor&& key, Tensor&& value) noexcept {
	// Reshape key and query and compute (QK^T) batched
	query->reshape({-1, (int)(conf.numAttnHeads), (int)(conf.dimensions.key)});
	key->reshape({-1, (int)(conf.numAttnHeads), (int)(conf.dimensions.key)});
	auto facq = dl::transpose(std::move(query), {0, 1});
	auto fack = dl::transpose(std::move(key), {0, 1});
	// (12, 10, 64) @ (12, 64, 10) -> (12, 10, 10)
	auto prod = dl::matmul(std::move(facq), dl::transpose(std::move(fack), {-1, -2}));
	std::cout << facq->numDim() << ':' << facq->shape(0) << ',' << facq->shape(1) << ',' << facq->shape(2) << std::endl;
	std::cout << fack->numDim() << ':' << fack->shape(0) << ',' << fack->shape(1) << ',' << fack->shape(2) << std::endl;
	std::cout << prod->numDim() << ':' << prod->shape(0) << ',' << prod->shape(1) << ',' << prod->shape(2) << ','
			  << prod->shape(3) << std::endl;
	// Compute softmax(prod / sqrt(d_k)) * W^V + b^V
	return weightValue.forward(dl::softmax(std::move(prod) * dl::constant(dimKeysInvSqrt), -1));
}

Tensor TransformerEncoder::forward(Tensor& input) {
	auto mha = multiHeadAttention(weightQuery.forward(input), weightKey.forward(input), weightValue.forward(input));
	auto tmp = mhaNorm.forward(std::move(mha) + input);
	auto ffn = weightIntermedOut.forward(dl::relu(weightIntermed.forward(std::move(tmp))));
	return ffnNorm.forward(std::move(ffn));
}