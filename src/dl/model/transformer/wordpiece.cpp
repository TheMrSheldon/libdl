#include <dl/model/transformer/wordpiece.hpp>

#include <dl/utils/line_iter.hpp>

using dl::WordPieceTokenizer;

WordPieceTokenizer::WordPieceTokenizer(std::vector<std::string>&& pieces) noexcept : wordPieces(pieces) {}

std::vector<size_t> WordPieceTokenizer::tokenize(const std::string& text) const noexcept {}

WordPieceTokenizer WordPieceTokenizer::fromWordPieces(StrIter begin, StrIter end) noexcept {
	std::vector<std::string> pieces;
	for (auto it = begin; it != end; ++it) {
		pieces.emplace_back(std::move(*it));
	}
	return WordPieceTokenizer(std::move(pieces));
}
WordPieceTokenizer WordPieceTokenizer::fromStream(std::istream& stream) noexcept {
	StrIter begin(std::move(dl::utils::LineIterator(stream)));
	StrIter end(std::move(dl::utils::LineIterator()));
	return WordPieceTokenizer::fromWordPieces(begin, end);
}