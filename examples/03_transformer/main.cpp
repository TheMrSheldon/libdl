#include <dl/model/transformer/wordpiece.hpp>
#include <dl/utils/urlstream.hpp>

#include <iostream>

int main(void) {
	dl::utils::URLStream in{"https://huggingface.co/google-bert/bert-base-uncased/raw/main/vocab.txt"};
	auto tokenizer = dl::WordPieceTokenizer::fromStream(in);
	tokenizer.tokenize("HuggingFace");
}
