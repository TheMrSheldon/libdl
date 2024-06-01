#include <dl/device.hpp>
#include <dl/io/weightsfile.hpp>
#include <dl/logging.hpp>
#include <dl/model/transformer/wordpiece.hpp>
#include <dl/utils/urlstream.hpp>
#include <nlp/transformer/bert.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>

using json = nlohmann::json;

namespace hf {
	struct Conf {
		std::string baseUrl;
	};
	static Conf defaultConf{.baseUrl = "https://huggingface.co"};
} // namespace hf

void download(std::string url, std::filesystem::path file) {
	dl::utils::URLStream input(url.c_str());
	std::ofstream output(file, std::ios::binary);
	output << input.rdbuf();
}

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
	dl::log::setVerbosity(dl::log::Verbosity::Debug);
	auto logger = dl::log::getLogger("main");

	/*auto tokenizer = loadTokenizerFromHuggingFace("google-bert/bert-base-uncased");
	auto text = "Hugging Face ABCDEF";
	auto tokens = tokenizer.tokenize(text);
	for (auto token : tokens)
		std::cout << token << ", " << std::endl;
	std::cout << std::endl;*/

	/*download(
			std::format(
					"{}/{}/resolve/main/model.safetensors?download=true", hf::defaultConf.baseUrl,
					"google-bert/bert-base-uncased"
			)
					.c_str(),
			"bertmodel.safetensors"
	);*/

	//auto bert = loadModelFromHuggingFace("google-bert/bert-base-uncased");
	auto modelConf = loadConfigFromHuggingFace("google-bert/bert-base-uncased");
	nlp::BERT bert(modelConf);
	std::ifstream stream("bertmodel.safetensors", std::ios::binary);
	auto success = dl::io::safetensorsFormat.loadModelFromStream(bert, stream);
	std::cout << (success ? "Success!" : "Failed :(") << std::endl;

	auto input = dl::ones({10, 768});
	auto& encoder = *bert.encoder.encoders[0];
	auto output = encoder.forward(input);
	std::cout << output->numDim() << ':' << output->shape(0) << ',' << output->shape(1) << ',' << output->shape(2)
			  << std::endl;
	std::cout << output << std::endl;

	/*dl::utils::URLStream in{"https://huggingface.co/google-bert/bert-base-uncased/raw/main/vocab.txt"};
	auto tokenizer = dl::WordPieceTokenizer::fromStream(in);
	tokenizer.tokenize("HuggingFace");*/
}
