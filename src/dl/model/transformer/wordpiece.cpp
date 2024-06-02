#include <dl/model/transformer/wordpiece.hpp>

#include <dl/utils/line_iter.hpp>

#include <nlohmann/json.hpp>
#include <tsl/htrie_map.h>

#include <regex>

using json = nlohmann::json;

using dl::WordPieceTokenizer;

WordPieceTokenizer::WordPieceTokenizer(Conf conf, PieceIter begin, PieceIter end) noexcept
		: logger(dl::log::getLogger("WordPiece")), contSubwordPrefix(conf.contSubwordPrefix),
		  trie(std::make_unique<tsl::htrie_map<char, size_t>>()) {
	for (auto it = begin; it != end; ++it) {
		trie->insert(std::get<0>(*it), std::get<1>(*it));
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
			word = contSubwordPrefix + word.substr(pIt.key().size());
		} while (word.size() > 2);
	}
	return ids;
}

WordPieceTokenizer WordPieceTokenizer::fromConf(std::istream& stream) noexcept {
	auto confJson = json::parse(stream);
	std::vector<std::tuple<std::string, std::size_t>> wordpieces;
	for (auto wordpiece : confJson["model"]["vocab"].items())
		wordpieces.emplace_back((std::string)wordpiece.key(), (std::size_t)wordpiece.value());
	Conf conf{.contSubwordPrefix = (std::string)confJson["model"]["continuing_subword_prefix"]};
	/** \todo set and use the UNK token **/
	std::cout << (std::string)confJson["model"]["unk_token"] << std::endl;
	/** \todo handle and use the normalizer configuration **/
	std::cout << confJson["normalizer"].dump() << std::endl;
	/** \todo handle and use the pre_tokenizer configuration **/
	/** \todo handle and use the post_processor configuration **/
	return WordPieceTokenizer(conf, PieceIter{wordpieces.begin()}, PieceIter{wordpieces.end()});
}