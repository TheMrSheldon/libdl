#include <dl/device.hpp>
#include <dl/io/weightsfile.hpp>
#include <dl/model/transformer/wordpiece.hpp>
#include <dl/utils/urlstream.hpp>
#include <nlp/transformer/bert.hpp>

#include <iostream>

#include <nlohmann/json.hpp>

#include <dl/logging.hpp>

using json = nlohmann::json;

namespace hf {
	struct Conf {
		std::string baseUrl;
	};
	static Conf defaultConf{.baseUrl = "https://huggingface.co"};
} // namespace hf

nlp::BERTConfig loadConfigFromHuggingFace(std::string repoURL, const hf::Conf& conf = hf::defaultConf) {
	dl::utils::URLStream confStream{
			std::format("{}/{}/resolve/main/config.json?download=true", conf.baseUrl, repoURL).c_str()
	};
	auto config = json::parse(confStream);
	return nlp::BERTConfig{
			.vocabSize = config["vocab_size"],
			.maxPositionEmbeddings = config["max_position_embeddings"],
			.typeVocabSize = config["type_vocab_size"]
	};
}

nlp::BERT loadModelFromHuggingFace(std::string repoURL, const hf::Conf& conf = hf::defaultConf) {
	auto modelConf = loadConfigFromHuggingFace("google-bert/bert-base-uncased");
	nlp::BERT bert(modelConf);
	dl::utils::URLStream in{
			std::format("{}/{}/resolve/main/model.safetensors?download=true", conf.baseUrl, repoURL).c_str()
	};
	auto success = dl::io::safetensorsFormat.loadModelFromStream(bert, in);
	std::cout << (success ? "Success!" : "Failed :(") << std::endl;
	return bert;
}

dl::WordPieceTokenizer loadTokenizerFromHuggingFace(std::string repoURL, const hf::Conf& conf = hf::defaultConf) {
	/*dl::utils::URLStream vocabStream{
			std::format("{}/{}/resolve/main/vocab.txt?download=true", conf.baseUrl, repoURL).c_str()
	};*/
	dl::utils::URLStream vocabStream{
			std::format("{}/{}/resolve/main/tokenizer.json?download=true", conf.baseUrl, repoURL).c_str()
	};
	return dl::WordPieceTokenizer::fromStream(vocabStream);
}

int main(void) {
	/*dl::InitializerTensor<float> tmp = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {0, 1, 2}}};
	auto tensorA = dl::constant(std::move(tmp));
	auto tensorB = dl::transpose(dl::Tensor(tensorA), {-1, -2});

	for (auto tmp : tensorA->shape())
		std::cout << tmp << ',';
	std::cout << std::endl;
	for (auto tmp : tensorB->shape())
		std::cout << tmp << ',';
	std::cout << std::endl;

	auto tensorC = dl::matmul(tensorA, tensorB);
	for (auto tmp : tensorC->shape())
		std::cout << tmp << ',';
	std::cout << std::endl;

	std::cout << tensorC << std::endl;*/

	auto logger = dl::log::getLogger("main");

	auto tokenizer = loadTokenizerFromHuggingFace("google-bert/bert-base-uncased");
	auto text = "Hugging Face ABCDEF";
	auto tokens = tokenizer.tokenize(text);
	for (auto token : tokens)
		std::cout << token << ", " << std::endl;
	std::cout << std::endl;

	auto bert = loadModelFromHuggingFace("google-bert/bert-base-uncased");

	// auto input = dl::ones({10, 768});
	// auto output = bert.forward(input);
	// std::cout << output->shape(0) << ',' << output->shape(1) << std::endl;

	/*dl::utils::URLStream in{"https://huggingface.co/google-bert/bert-base-uncased/raw/main/vocab.txt"};
	auto tokenizer = dl::WordPieceTokenizer::fromStream(in);
	tokenizer.tokenize("HuggingFace");*/
}
