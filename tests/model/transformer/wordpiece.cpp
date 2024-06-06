#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dl/model/transformer/wordpiece.hpp>

using Catch::Matchers::RangeEquals;

constexpr auto SimpleTokenizerConf = R"({
	"normalizer": null,
	"model": {
		"unk_token": "[UNK]",
		"continuing_subword_prefix": "##",
		"vocab": {
			"a": 0,
			"abcdx": 1,
			"##b": 2,
			"##c": 3,
			"##cdy": 4,
			"##dz": 5
		}
	}
})";

TEST_CASE("WordPiece", "[Tokenization]") {
	{
		// Taken from figure 1 of \cite fast_wordpiece
		auto stream = std::istringstream(std::string(SimpleTokenizerConf));
		auto tokenizer = dl::WordPieceTokenizer::fromConf(stream);
		// The IDs of: a, ##b, ##c, ##dz
		CHECK_THAT(tokenizer.tokenize("abcdz"), RangeEquals(std::vector{0, 2, 3, 5}));

		// The following edge cases are from the "corner cases" section of \cite fast_wordpiece
		// 1) ##bc -> [##b, ##c]
		CHECK_THAT(tokenizer.tokenize("##bc"), RangeEquals(std::vector{2, 3}));
		// 2) # -> [#]
	}
}