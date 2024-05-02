#include <iostream>

#include <dl/utils/urlstream.hpp>
#include <ir/data/datasets.hpp>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>

#include <fstream>
#include <stdio.h>
#include <string>

int main(void) {
	dl::utils::URLStream stream{
			"https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz"
	};
	boost::iostreams::filtering_istream in;
	in.push(boost::iostreams::gzip_decompressor());
	in.push((std::istream&)stream);
	for (std::string line; std::getline(in, line);) {
		std::cout << line << std::endl;
	}
}

/*int main(int argc, char* argv[]) {
	dl::utils::URLStream istream("https://www.google.com");
	for (std::string line; std::getline(istream, line);)
		std::cout << line << std::endl;

	auto dataset = ir::datasets::load<ir::PointwiseDataset>("msmarco-passage/train/judged");
	auto dataset2 = ir::datasets::load<ir::PairwiseDataset>("msmarco-passage/train/judged");
	return 0;
}*/