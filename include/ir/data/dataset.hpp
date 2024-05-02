#pragma once

#include <dl/learning/dataset.hpp>

#include <variant>

namespace ir {
	struct Document {};
	struct Query {};

	template <typename>
	class IRDataset;

	template <>
	class IRDataset<float(Query, Document)> : public dl::Dataset<float(Query, Document)> {
	public:
		virtual ~IRDataset() = default;
	};

	template <>
	class IRDataset<void(Query, Document, Document)> : public dl::Dataset<void(Query, Document, Document)> {
	public:
		virtual ~IRDataset() = default;
	};

	using PointwiseDataset = IRDataset<float(Query, Document)>;
	using PairwiseDataset = IRDataset<void(Query, Document, Document)>;

	using AnyDataset = std::variant<PointwiseDataset, PairwiseDataset>;
} // namespace ir