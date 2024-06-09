#pragma once

#include "../layernorm.hpp"
#include "../linear.hpp"
#include "../model.hpp"

#include <cmath>
#include <vector>

namespace dl {

	/**
     * @brief Computes positional encodings as described by @cite transformer .
     * 
     * @param pos 
     * @param i 
     * @param dimModel 
     * @return the positional encoding at the specified position, dimension and model dimension.
     */
	constexpr double calcPosEncoding(size_t pos, size_t i, size_t dimModel) {
		return (i % 2 == 0) ? std::sin(pos / std::pow(10000, i / dimModel))
							: std::sin(pos / std::pow(10000, (i - 1) / dimModel));
	}

	struct TransformerConf {
		struct {
			size_t model; /**< \f(d_\text{model}}}\f) from \cite transformer . **/
			size_t key;	  /**< \f(d_k}}\f) from \cite transformer . **/
			size_t value; /**< \f(d_v}}\f) from \cite transformer . **/
			size_t inner; /**< \f(d_\text{ff}}}\f) from \cite transformer . **/
		} dimensions;
		size_t numEncoders;
		size_t numAttnHeads;
	};

	class TransformerEncoder final : public Model<TensorPtr(TensorPtr)> {
	public:
		TransformerConf conf;
		// Multi-Head Attention
		dl::Linear weightQuery;
		dl::Linear weightKey;
		dl::Linear weightValue;
		dl::Linear weightOut;
		dl::Linear weightIntermed;
		dl::LayerNorm mhaNorm;
		// FFN
		dl::Linear weightIntermedOut;
		dl::LayerNorm ffnNorm;

		TransformerEncoder(TransformerEncoder& other) = delete;
		TransformerEncoder(TransformerEncoder&& other) = delete;

	public:
		TransformerEncoder(TransformerConf conf) noexcept;
		virtual TensorPtr forward(TensorPtr input) override;

		/**
		 * @brief The precomputed inverse square root of dimKeys.
		 * @details This is the precomputed normalization factor \f$\sqrt{d_k}^{-1}\f$ used in the scaled dot-product
		 * attention.
		 */
		const float dimKeysInvSqrt;

		/**
		 * @brief Implements the scaled dot-product attention.
		 * @details Scaled dot-product attention is Eq. (1) in the transformer paper \cite transformer :
		 * \f[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.\f]
		 * 
		 * @param value 
		 * @param key 
		 * @param query 
		 * @return the scaled dot-product attention.
		 */
		TensorPtr scaledDotProductAttention(TensorPtr query, TensorPtr key, TensorPtr value) noexcept;

		/**
		 * @brief Implements the transformer's multi-head attention.
		 * @details Multi-head attention is chapter 3.2.2 in the transformer paper \cite transformer. Let
		 * \f(W_i^Q, W_i^K\in \mathbb{R}^{d_\text{model} \times d_k}, W_i^V \in \mathbb{R}^{d_\text{model} \times d_v}\f)
		 * denote the query, key and value matrix of the i-th head respectively for \f(1\leq i \leq h\f), where \f(h\f)
		 * is the total number of heads. Further, let \f(W^O\f) denote the output linearity. Multi-head attention is
		 * defined as the concatinated attention of each attention head:
		 * \f[
		 * 	\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\cdot W^O \text{ with }
		 * 	\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V),
		 * \f]
		 * where "Attention" denotes scaledDotProductAttention (TransformerEncoder::scaledDotProductAttention).
		 * 
		 * \note In practice, the linearities are usually accompanied with biases
		 * \f(b^O, b_i^Q, b_i^K, b_i^V \in \mathbb{R}^{d_\text{model}}\f), such that MHA is more accurately described as
		 * \f[\text{MHA}(Q, K, V) = \text{Concat}(
		 * 		\text{Attention}(QW_1^Q+b_1^Q, KW_1^K+b_1^K, VW_1^V+b_1^V), \dots)\cdot W^O + b^O.\f]
		 * 
		 * @param query 
		 * @param key 
		 * @param value 
		 * @return TensorPtr 
		 * @see For a more detailed description, please read the \ref technicalTransformer page.
		 */
		TensorPtr multiHeadAttention(TensorPtr query, TensorPtr key, TensorPtr value) noexcept;
	};

	/**
     * @brief @cite transformer
     * @details
     */
	class Transformer final : public Model<TensorPtr(TensorPtr)> {
	public:
		const TransformerConf conf;
		std::vector<std::unique_ptr<TransformerEncoder>> encoders;
		dl::Linear weightOut;

	public:
		Transformer(TransformerConf conf) noexcept;

		virtual TensorPtr forward(TensorPtr input) override;
	};
}; // namespace dl