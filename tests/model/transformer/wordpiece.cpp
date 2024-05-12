#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dl/model/transformer/wordpiece.hpp>

using Catch::Matchers::RangeEquals;

TEST_CASE("WordPiece", "[Tokenization]") {
	{
		// Taken from figure 1 of \cite fast_wordpiece
		std::vector<std::string> wordpieces{"a", "abcdx", "##b", "##c", "##cdy", "##dz"};
		auto tokenizer = dl::WordPieceTokenizer::fromWordPieces(
				dl::WordPieceTokenizer::StrIter{std::begin(wordpieces)},
				dl::WordPieceTokenizer::StrIter{std::end(wordpieces)}
		);
		// The IDs of: a, ##b, ##c, ##dz
		CHECK_THAT(tokenizer.tokenize("abcdz"), RangeEquals(std::vector{0, 2, 3, 5}));

		// The following edge cases are from the "corner cases" section of \cite fast_wordpiece
		// 1) ##bc -> [##b, ##c]
		CHECK_THAT(tokenizer.tokenize("##bc"), RangeEquals(std::vector{2, 3}));
		// 2) # -> [#]
	}
}