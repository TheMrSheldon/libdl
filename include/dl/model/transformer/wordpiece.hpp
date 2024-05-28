#pragma once

#include "../../utils/generic_iterator.hpp"

#include <experimental/propagate_const>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Forward declaration for pImpl
namespace tsl {
	namespace ah {
		template <class CharT>
		struct str_hash;
	}
	template <class CharT, class T, class Hash, class KeySizeT>
	class htrie_map;
}; // namespace tsl

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
		std::experimental::propagate_const<
				std::unique_ptr<tsl::htrie_map<char, size_t, tsl::ah::str_hash<char>, std::uint16_t>>>
				trie;

		WordPieceTokenizer(StrIter begin, StrIter end) noexcept;

	public:
		~WordPieceTokenizer();

		[[nodiscard]] std::vector<size_t> tokenize(const std::string& text) const noexcept;

		[[nodiscard]] static WordPieceTokenizer fromWordPieces(StrIter begin, StrIter end) noexcept;
		[[nodiscard]] static WordPieceTokenizer fromStream(std::istream& stream) noexcept;
	};
}; // namespace dl