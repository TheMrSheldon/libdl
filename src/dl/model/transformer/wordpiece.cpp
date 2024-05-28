#include <dl/model/transformer/wordpiece.hpp>

#include <dl/utils/line_iter.hpp>

#include <nlohmann/json.hpp>
#include <tsl/htrie_map.h>

#include <regex>

using json = nlohmann::json;

using dl::WordPieceTokenizer;

WordPieceTokenizer::WordPieceTokenizer(StrIter begin, StrIter end) noexcept
		: trie(std::make_unique<tsl::htrie_map<char, size_t>>()) {
	size_t idx = 0;
	for (auto it = begin; it != end; ++it) {
		trie->insert(*it, idx);
		++idx;
	}
}
WordPieceTokenizer::~WordPieceTokenizer() { trie = nullptr; }

std::vector<size_t> WordPieceTokenizer::tokenize(const std::string& text) const noexcept {
	/** \todo use a generator in the future **/
	/** \todo this implementation does not yet use the algorithm by \cite wordpiece **/
	std::vector<size_t> ids;
	std::regex wordPattern("\\S+");
	for (std::sregex_iterator it = std::sregex_iterator(text.begin(), text.end(), wordPattern);
		 it != std::sregex_iterator(); ++it) {
		auto word = it->str();
		/** \todo this is just a crutch and does not work for unicode **/
		std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c) { return std::tolower(c); });
		do {
			auto pIt = trie->longest_prefix(word);
			assert(pIt != std::end(*trie)); /** \todo handle more gracefully **/
			ids.push_back(*pIt);
			word = "##" + word.substr(pIt.key().size());
		} while (word.size() > 2);
	}
	return ids;
}

WordPieceTokenizer WordPieceTokenizer::fromWordPieces(StrIter begin, StrIter end) noexcept {
	return WordPieceTokenizer(begin, end);
}
WordPieceTokenizer WordPieceTokenizer::fromStream(std::istream& stream) noexcept {
	/*StrIter begin(std::move(dl::utils::LineIterator(stream)));
	StrIter end(std::move(dl::utils::LineIterator()));
	return WordPieceTokenizer::fromWordPieces(begin, end);*/
	auto conf = json::parse(stream);
	std::cout << conf["model"]["vocab"].dump() << std::endl;
	exit(0);
}