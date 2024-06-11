#include <dl/model/transformer/transformer.hpp>

#include <dl/tensor/math.hpp>

#include <cmath>
#include <format>

using dl::Linear;
using dl::TensorPtr;
using dl::Transformer;
using dl::TransformerEncoder;

static dl::TensorPtr BertGELU(const dl::TensorPtr& input) {
	return input * 0.5f * (1.0f + dl::erf(input / std::sqrt(2.0f)));
}

Transformer::Transformer(TransformerConf conf) noexcept
		: conf(conf), weightOut(conf.numAttnHeads * conf.dimensions.value, conf.dimensions.model), encoders() {
	for (size_t i = 0; i < conf.numEncoders; ++i) {
		const auto& encoder = encoders.emplace_back(std::make_unique<TransformerEncoder>(conf));
		registerSubmodel(std::format("encoder.layer.{}", i), *encoder);
	}
}

TensorPtr Transformer::forward(TensorPtr input) {
	TensorPtr tmp = input;
	for (size_t i = 0; i < encoders.size(); ++i)
		tmp = encoders[i]->forward(tmp);
	return tmp;
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

TensorPtr TransformerEncoder::multiHeadAttention(TensorPtr query, TensorPtr key, TensorPtr value) noexcept {
	auto attn = scaledDotProductAttention(query, key, value);
	return weightOut.forward(attn);
}

TensorPtr TransformerEncoder::scaledDotProductAttention(TensorPtr query, TensorPtr key, TensorPtr value) noexcept {
	/** \todo use std::move to reduce copying temporaries **/
	// Reshape key and query and compute (QK^T) batched
	query->reshape({-1, (int)(conf.numAttnHeads), (int)(conf.dimensions.key)});
	key->reshape({-1, (int)(conf.numAttnHeads), (int)(conf.dimensions.key)});
	value->reshape({-1, (int)conf.numAttnHeads, (int)conf.dimensions.value});
	auto facq = dl::transpose(query, {0, 1});
	auto fack = dl::transpose(key, {0, 1});
	auto facv = dl::transpose(value, {0, 1});
	// (12, 10, dmodel) @ (12, dmodel, 10) -> (12, 10, 10)
	auto prod = dl::matmul(facq, dl::transpose(fack, {-1, -2}));
	// Compute smax = softmax(prod / sqrt(d_k))
	auto smax = dl::softmax(prod * dimKeysInvSqrt, -1);
	// compute smax * V
	auto tmp = dl::matmul(smax, facv);
	return dl::reshape(dl::transpose(tmp, {0, 1}), {-1, (int)conf.dimensions.model});
}

TensorPtr TransformerEncoder::forward(TensorPtr input) {
	auto mha = multiHeadAttention(weightQuery.forward(input), weightKey.forward(input), weightValue.forward(input));
	auto attention = mhaNorm.forward(mha + input);
	auto intermed = BertGELU(weightIntermed.forward(attention));
	return ffnNorm.forward(weightIntermedOut.forward(intermed) + attention);
}