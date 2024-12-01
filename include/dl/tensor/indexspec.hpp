#ifndef DL_TENSOR_INDEXSPEC_HPP
#define DL_TENSOR_INDEXSPEC_HPP

#include <variant>
#include <vector>

namespace dl {
	class TensorPtr;

	namespace idx {
		struct All {};
		struct NewDim {};
		struct Range {
			signed from;
			signed to;
		};

		constexpr All all;
		constexpr NewDim newdim;
	}; // namespace idx

	using IdxSlice = std::variant<signed, idx::All, idx::NewDim, idx::Range, TensorPtr>;
	using IndexSpec = std::vector<IdxSlice>;
} // namespace dl

#endif