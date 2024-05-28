#pragma once

#include "../device.hpp"
#include "../tensor/tensor.hpp"
#include "model.hpp"

namespace dl {
	class Embedding : public dl::Model<dl::Tensor(dl::Tensor&)> {
	private:
		dl::Tensor weight;

	public:
		Embedding(size_t numEmbeddings, size_t embeddingDim) : weight(dl::empty({numEmbeddings, embeddingDim})) {
			registerParameter("weight", weight);
		}

		virtual dl::Tensor forward(dl::Tensor& input) { return nullptr; }
	};
} // namespace dl