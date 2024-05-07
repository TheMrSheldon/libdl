#pragma once

#include "../../utils/generic_iterator.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace dl {
	/**
	 * @brief Implements WordPiece tokenizaton as proposed in \cite wordpiece using the optimized algorithm by
	 * \cite fast_wordpiece .
	 * @details 
	 */
	class WordPieceTokenizer final {
	public:
		using StrIter = utils::GenericIterator<std::string>;

	private:
		std::vector<std::string> wordPieces;

		WordPieceTokenizer(std::vector<std::string>&& wordPieces) noexcept;

	public:
		[[nodiscard]] std::vector<size_t> tokenize(const std::string& text) const noexcept;

		[[nodiscard]] static WordPieceTokenizer fromWordPieces(StrIter begin, StrIter end) noexcept;
		[[nodiscard]] static WordPieceTokenizer fromStream(std::istream& stream) noexcept;
	};
}; // namespace dl