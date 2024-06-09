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

	class BERTEmbeddings : public dl::Model<dl::TensorPtr(const dl::TensorPtr&, const dl::TensorPtr&)> {
	private:
		dl::Embedding wordEmbeddings;
		dl::Embedding positionalEmbeddings;
		dl::Embedding tokenTypeEmbeddings;
		dl::LayerNorm layerNorm;

	public:
		BERTEmbeddings(BERTConfig bertConf, dl::TransformerConf config)
				: wordEmbeddings(bertConf.vocabSize, config.dimensions.model),
				  positionalEmbeddings(bertConf.maxPositionEmbeddings, config.dimensions.model),
				  tokenTypeEmbeddings(bertConf.typeVocabSize, config.dimensions.model),
				  layerNorm({config.dimensions.model}) {
			registerSubmodel("word_embeddings", wordEmbeddings);
			registerSubmodel("position_embeddings", positionalEmbeddings);
			registerSubmodel("token_type_embeddings", tokenTypeEmbeddings);
			registerSubmodel("LayerNorm", layerNorm);
		}

		virtual dl::TensorPtr forward(const dl::TensorPtr& inputIds, const dl::TensorPtr& inputTokenTypes) {
			/** Note that the embeddings returned are references to tensors instead of tensors since they return the
			 * references to the learned embeddings. **/
			// auto& inputEmbeds = wordEmbeddings.forward(std::forward<decltype(inputIds)>(inputIds));
			// auto& typeEmbeds = tokenTypeEmbeddings.forward(std::forward<decltype(inputTokenTypes)>(inputTokenTypes));
			// return layerNorm.forward(inputEmbeds + typeEmbeds + posEmbeds);
			/** \todo implement **/
			throw std::runtime_error("Not yet implemented");
		}
	};

	class BERTPooling : public dl::Model<dl::TensorPtr(const dl::TensorPtr&)> {
	private:
		dl::Linear dense;

	public:
		BERTPooling(dl::TransformerConf conf) noexcept : dense(conf.dimensions.model, conf.dimensions.model) {
			registerSubmodel("dense", dense);
		}

		virtual dl::TensorPtr forward(const dl::TensorPtr& input) { return nullptr; }
	};

	/**
     * @brief \cite bert
     * 
     */
	class BERT : public dl::Model<dl::TensorPtr(const dl::TensorPtr&)> {
	public:
		static constexpr dl::TransformerConf transformerConf{
				.dimensions = {.model = 768, .key = 64, .value = 64, .inner = 3072},
				.numEncoders = 12,
				.numAttnHeads = 12
		};
		BERTEmbeddings embeddings;
		dl::Transformer encoder;
		BERTPooling pooling;

	public:
		BERT(BERTConfig config)
				: embeddings(config, transformerConf), encoder(transformerConf), pooling(transformerConf) {
			registerSubmodel("bert.embeddings", embeddings);
			registerSubmodel("bert", encoder);
			registerSubmodel("bert.pooler", pooling);
		}

		virtual dl::TensorPtr forward(const dl::TensorPtr& input) override {
			/** \todo implement **/
			throw std::runtime_error("Not yet implemented");
			// return pooling.forward(encoder.forward(embeddings.forward(input)));
		}
	};
} // namespace nlp