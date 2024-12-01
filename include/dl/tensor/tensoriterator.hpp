#ifndef DL_TENSOR_TENSORITERATOR_HPP
#define DL_TENSOR_TENSORITERATOR_HPP

#include "tensorptr.hpp"

#include <iterator>

namespace dl {

	struct TensorSentinel final {};

	struct TensorIter final {
	private:
		TensorPtr ptr;
		signed idx;

	public:
		using iterator_category = std::input_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = TensorPtr;
		using pointer = TensorPtr*;
		using reference = TensorPtr&;

		TensorIter();
		explicit TensorIter(TensorSentinel);
		explicit TensorIter(TensorPtr ptr);
		TensorIter(const TensorIter& other);
		TensorIter(TensorIter&& other) noexcept;

		TensorIter& operator=(const TensorIter& other) noexcept;
		TensorIter& operator=(TensorIter&& other) noexcept;

		TensorPtr operator*() const;

		TensorIter& operator++() noexcept;
		TensorIter operator++(int) noexcept;

		bool operator==(TensorSentinel s) const noexcept;
	};

	static_assert(std::input_or_output_iterator<TensorIter>);
	static_assert(std::sentinel_for<TensorSentinel, TensorIter>);

	// So that TensorPtr models std::ranges::input_range
	static_assert(std::input_iterator<TensorIter>);

} // namespace dl

#endif