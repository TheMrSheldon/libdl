#include <iostream>

#include <dl/utils/urlstream.hpp>
#include <ir/data/datasets.hpp>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>

#include <fstream>
#include <stdio.h>
#include <string>

#include <arrow/csv/api.h>

int main(void) {
	/*dl::utils::URLStream stream{
			"https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz"
	};
	boost::iostreams::filtering_istream in;
	in.push(boost::iostreams::gzip_decompressor());
	in.push((std::istream&)stream);
	for (std::string line; std::getline(in, line);) {
		std::cout << line << std::endl;
	}*/
	/*dl::utils::URLStream stream{
			"https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz"
	};
	boost::iostreams::filtering_istream in;
	in.push(boost::iostreams::gzip_decompressor());
	in.push((std::istream&)stream);

	arrow::io::IOContext io_context = arrow::io::default_io_context();
	std::shared_ptr<arrow::io::InputStream> input = in;

	auto read_options = arrow::csv::ReadOptions::Defaults();
	auto parse_options = arrow::csv::ParseOptions::Defaults();
	auto convert_options = arrow::csv::ConvertOptions::Defaults();

	// Instantiate StreamingReader from input stream and options
	auto maybe_reader =
			arrow::csv::StreamingReader::Make(io_context, input, read_options, parse_options, convert_options);
	if (!maybe_reader.ok()) {
		// Handle StreamingReader instantiation error...
	}
	std::shared_ptr<arrow::csv::StreamingReader> reader = *maybe_reader;

	// Set aside a RecordBatch pointer for re-use while streaming
	std::shared_ptr<arrow::RecordBatch> batch;

	while (true) {
		// Attempt to read the first RecordBatch
		arrow::Status status = reader->ReadNext(&batch);

		if (!status.ok()) {
			// Handle read error
		}

		if (batch == NULL) {
			// Handle end of file
			break;
		}

		// Do something with the batch
	}*/
}

/*int main(int argc, char* argv[]) {
	dl::utils::URLStream istream("https://www.google.com");
	for (std::string line; std::getline(istream, line);)
		std::cout << line << std::endl;

	auto dataset = ir::datasets::load<ir::PointwiseDataset>("msmarco-passage/train/judged");
	auto dataset2 = ir::datasets::load<ir::PairwiseDataset>("msmarco-passage/train/judged");
	return 0;
}*/