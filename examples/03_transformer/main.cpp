#include <dl/device.hpp>
#include <dl/model/transformer/transformer.hpp>
#include <dl/model/transformer/wordpiece.hpp>
#include <dl/utils/urlstream.hpp>

#include <iostream>

int main(void) {
	dl::TransformerEncoder transformer(dl::TransformerConf{
			.dimensions = {.model = 768, .key = 64, .value = 64}, .numEncoders = 12, .numAttnHeads = 12
	});
	std::cout << transformer.numParameters() << std::endl;
	for (auto&& [key, tensor] : transformer.parameters()) {
		std::cout << key << std::endl;
	}
	auto input = dl::ones({10, 768});
	auto output = transformer.forward(input);
	std::cout << output->shape(0) << ',' << output->shape(1) << std::endl;
	/*dl::utils::URLStream in{"https://huggingface.co/google-bert/bert-base-uncased/raw/main/vocab.txt"};
	auto tokenizer = dl::WordPieceTokenizer::fromStream(in);
	tokenizer.tokenize("HuggingFace");*/
}
