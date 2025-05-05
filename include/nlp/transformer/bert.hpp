#pragma once

#include <dl/model/embedding.hpp>
#include <dl/model/model.hpp>
#include <dl/model/transformer/transformer.hpp>
#include <dl/tensor/tensorptr.hpp>

#include <dl/utils/composed.hpp>

namespace nlp {

	struct BERTConfig {
		size_t vocabSize;
		size_t maxPositionEmbeddings;
		size_t typeVocabSize;
	};

	class BERTEmbeddings : public dl::Model {
	private:
		dl::Embedding wordEmbeddings;
		dl::Embedding positionalEmbeddings;
		dl::Embedding tokenTypeEmbeddings;
		dl::LayerNorm layerNorm;

	public:
		BERTEmbeddings(const BERTConfig& bertConf, const dl::TransformerConf& config)
				: wordEmbeddings(bertConf.vocabSize, config.dimensions.model),
				  positionalEmbeddings(bertConf.maxPositionEmbeddings, config.dimensions.model),
				  tokenTypeEmbeddings(bertConf.typeVocabSize, config.dimensions.model),
				  layerNorm({config.dimensions.model}) {
			registerSubmodel("word_embeddings", wordEmbeddings);
			registerSubmodel("position_embeddings", positionalEmbeddings);
			registerSubmodel("token_type_embeddings", tokenTypeEmbeddings);
			registerSubmodel("LayerNorm", layerNorm);
		}

		dl::TensorPtr operator()(const dl::TensorPtr& inputIds, const dl::TensorPtr& inputTokenTypes) {
			auto posEmbeds = positionalEmbeddings(dl::arange(0, inputIds->shape(-1)));
			auto inputEmbeds = wordEmbeddings(inputIds);
			//auto typeEmbeds = tokenTypeEmbeddings(inputTokenTypes);
			return layerNorm(inputEmbeds /*+ typeEmbeds*/ + posEmbeds);
		}
	};

	class BERTPooling : public dl::Model {
	private:
		dl::Linear dense;

	public:
		explicit BERTPooling(const dl::TransformerConf& conf) noexcept
				: dense(conf.dimensions.model, conf.dimensions.model) {
			registerSubmodel("dense", dense);
		}

		dl::TensorPtr operator()(const dl::TensorPtr& input) { return nullptr; }
	};

	/**
     * @brief \cite bert
     * 
     */
	class BERT : public dl::Model {
	private:
		static constexpr dl::TransformerConf transformerConf{
				.dimensions = {.model = 768, .key = 64, .value = 64, .inner = 3072},
				.numEncoders = 12,
				.numAttnHeads = 12
		};
		BERTEmbeddings embeddings;
		dl::Transformer encoder;
		BERTPooling pooling;

	public:
		explicit BERT(const BERTConfig& config)
				: embeddings(config, transformerConf), encoder(transformerConf), pooling(transformerConf) {
			registerSubmodel("bert.embeddings", embeddings);
			registerSubmodel("bert", encoder);
			registerSubmodel("bert.pooler", pooling);
		}

		dl::TensorPtr operator()(const dl::TensorPtr& input) {
			/** \todo support tokentypes **/
			return pooling(encoder(embeddings(input, nullptr)));
		}
	};

	class BERTMLMPrediction : public dl::Model {
	private:
		BERT bert;

	public:
		explicit BERTMLMPrediction(const BERTConfig& config) : bert(config) {}

		dl::TensorPtr operator()(const dl::TensorPtr& input) {
			/** \todo implement **/
			throw std::runtime_error("Not implemmented");
		}
	};
} // namespace nlp