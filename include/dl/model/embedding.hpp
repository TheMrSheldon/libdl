#pragma once

#include "../device.hpp"
#include "../tensor/tensorptr.hpp"
#include "model.hpp"

namespace dl {
	class Embedding : public dl::Model {
	private:
		dl::TensorPtr weight;

	public:
		Embedding(size_t numEmbeddings, size_t embeddingDim) : weight(dl::empty({numEmbeddings, embeddingDim})) {
			registerParameter("weight", weight);
		}
		virtual ~Embedding() = default;

		dl::TensorPtr operator()(const dl::TensorPtr input) { return weight->get({input}); }
	};
} // namespace dl