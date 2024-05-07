#pragma once

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

	/**
     * @brief @cite transformer
     * @details
     */
	class Transformer final : public Model<Tensor(Tensor&)> {
	private:
		class AttnHead final : public Model<Tensor(Tensor, Tensor, Tensor)> {
		private:
			dl::Linear weightQuery;
			dl::Linear weightKey;
			dl::Linear weightValue;
			/**
             * @brief The precomputed inverse square root of dimKeys.
             * @details This is the precomputed normalization factor \f$\sqrt{d_k}^{-1}\f$ used in the scaled dot-product
             * attention.
             */
			const double dimKeysInvSqrt;

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
			Tensor scaledDotProductAttention(Tensor&& query, Tensor&& key, Tensor&& value) noexcept;

		public:
			AttnHead(size_t dimModel, size_t dimKeys, size_t dimValues) noexcept;
			virtual Tensor forward(Tensor, Tensor, Tensor) override;
		};

		const size_t dimModel;
		const size_t dimKeys;
		const size_t dimValues;
		std::vector<AttnHead> heads;
		dl::Linear weightOut;
		Tensor multiHeadAttention(Tensor&& query, Tensor&& key, Tensor&& value) noexcept;

	public:
		Transformer(size_t dimModel = 512, size_t numHeads = 8, size_t dimKeys = 64, size_t dimValues = 64) noexcept;
	};
}; // namespace dl