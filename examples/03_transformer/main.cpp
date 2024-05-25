#include <dl/device.hpp>
#include <dl/io/weightsfile.hpp>
#include <dl/model/transformer/transformer.hpp>
#include <dl/model/transformer/wordpiece.hpp>
#include <dl/utils/urlstream.hpp>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>

#include <iostream>

namespace dl {
	class BERT : public dl::Model<dl::Tensor(dl::Tensor&)> {
	private:
		static constexpr dl::TransformerConf transformerConf{
				.dimensions = {.model = 768, .key = 64, .value = 64, .inner = 3072},
				.numEncoders = 12,
				.numAttnHeads = 12
		};
		dl::Transformer encoder;

	public:
		BERT() : encoder(transformerConf) {
			/** \todo define and register the embedding layer **/
			registerSubmodel("bert", encoder);
		}

		virtual Tensor forward(Tensor& input) override {
			/** \todo implement **/
			return nullptr;
		}
	};
} // namespace dl

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

	dl::BERT bert;
	// auto input = dl::ones({10, 768});
	// auto output = transformer.forward(input);
	// std::cout << output->shape(0) << ',' << output->shape(1) << std::endl;

	dl::utils::URLStream in{
			"https://huggingface.co/google-bert/bert-base-uncased/resolve/main/model.safetensors?download=true"
	};
	auto success = dl::io::safetensorsFormat.loadModelFromStream(bert, in);
	std::cout << (success ? "Success!" : "Failed :(") << std::endl;

	/*dl::utils::URLStream in{"https://huggingface.co/google-bert/bert-base-uncased/raw/main/vocab.txt"};
	auto tokenizer = dl::WordPieceTokenizer::fromStream(in);
	tokenizer.tokenize("HuggingFace");*/
}
