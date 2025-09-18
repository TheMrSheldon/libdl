#ifndef DL_TENSOR_INDEXSPEC_HPP
#define DL_TENSOR_INDEXSPEC_HPP

#include <variant>
#include <vector>

namespace dl {
	class TensorPtr;

	using Index = long;
	using UIndex = size_t;

	namespace idx {
		struct All {};
		struct NewDim {};
		struct Range {
			Index from;
			Index to;
		};

		constexpr All all;
		constexpr NewDim newdim;
	}; // namespace idx

	using IdxSlice = std::variant<Index, idx::All, idx::NewDim, idx::Range, TensorPtr>;
	using IndexSpec = std::vector<IdxSlice>;
} // namespace dl

#endif