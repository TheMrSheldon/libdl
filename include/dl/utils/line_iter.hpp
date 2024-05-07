#pragma once

#include <istream>
#include <iterator>
#include <string>

namespace dl::utils {
	struct LineIterator final {
		using iterator_category = std::input_iterator_tag;
		using value_type = std::string;
		using difference_type = std::ptrdiff_t;
		using reference = const value_type&;
		using pointer = const value_type*;

		LineIterator() : input_(nullptr) {}
		LineIterator(std::istream& input) : input_(&input) { ++*this; }

		reference operator*() const { return s_; }
		pointer operator->() const { return &**this; }

		LineIterator& operator++() {
			if (!std::getline(*input_, s_))
				input_ = nullptr;
			return *this;
		}

		LineIterator operator++(int) {
			auto copy(*this);
			++*this;
			return copy;
		}

		friend bool operator==(const LineIterator& x, const LineIterator& y) { return x.input_ == y.input_; }

		friend bool operator!=(const LineIterator& x, const LineIterator& y) { return !(x == y); }

	private:
		std::istream* input_;
		std::string s_;
	};
} // namespace dl::utils