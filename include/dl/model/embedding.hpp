#pragma once

#include "../device.hpp"
#include "../tensor/tensorptr.hpp"
#include "model.hpp"

namespace dl {
	class Embedding : public dl::Model<dl::TensorPtr(const dl::TensorPtr)> {
	private:
		dl::TensorPtr weight;

	public:
		Embedding(size_t numEmbeddings, size_t embeddingDim) : weight(dl::empty({numEmbeddings, embeddingDim})) {
			registerParameter("weight", weight);
		}
		virtual ~Embedding() = default;

		virtual dl::TensorPtr forward(const dl::TensorPtr input) {
			/** \todo implement **/
			throw std::runtime_error("Not yet implemented");
		}
	};
} // namespace dl