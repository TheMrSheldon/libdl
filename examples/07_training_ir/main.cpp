#include <dl/learning/loss.hpp>
#include <dl/learning/optimizers/adam.hpp>
#include <dl/learning/trainer.hpp>
#include <dl/model/model.hpp>
#include <dl/utils/urlstream.hpp>
#include <ir/data/datasets.hpp>

#include <arrow/buffer.h>
#include <arrow/csv/api.h>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>

#include <iostream>

/*
class MonoBERT : public dl::ModelBase {
private:
public:
};

class TrecEvaluator final {
public:
	using GradeTriple = std::tuple<ir::Query, ir::Document, float>;

private:
	std::map<size_t, std::map<size_t, float>> grades;

public:
	TrecEvaluator(auto& dataset, std::vector<std::string> metrics) {}

	TrecEvaluator& operator+=(GradeTriple result) {
		grades[std::get<0>(result).id][std::get<1>(result).id] = std::get<2>(result);
		return *this;
	}

	std::map<std::string, float> aggregated();
};

dl::TensorPtr pairwiseTrainer(auto& model, ir::Query query, ir::Document pos, ir::Document neg) {
	return model(query, neg) - model(query, pos);
}

TrecEvaluator::GradeTriple trecEvalAdapter(auto& model, ir::Query query, ir::Document doc) {
	return std::make_tuple(query, doc, model(query, doc));
}*/

class StdIStream : public arrow::io::InputStream {
private:
	std::istream& stream;
	int64_t pos;

public:
	StdIStream(std::istream& stream) : stream(stream), pos(0) { set_mode(arrow::io::FileMode::READ); }

	arrow::Status Close() override { return arrow::Status::OK(); }
	bool closed() const override { return false; }

	arrow::Result<int64_t> Tell() const override { return pos; }

	arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
		stream.read(reinterpret_cast<char*>(out), nbytes);
		nbytes = std::cin.gcount();
		pos += nbytes;
		return nbytes;
	}

	arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
		ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(nbytes));
		ARROW_ASSIGN_OR_RAISE(int64_t bytes_read, Read(nbytes, buffer->mutable_data()));
		ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read, false));
		buffer->ZeroPadding();
		// R build with openSUSE155 requires an explicit shared_ptr construction
		return std::shared_ptr<arrow::Buffer>(std::move(buffer));
	}
};

int main(int argc, char* argv[]) {
	dl::utils::URLStream collection("https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.full.tsv.gz"
	);
	boost::iostreams::filtering_istreambuf in;
	in.push(boost::iostreams::gzip_decompressor());
	in.push(collection);
	std::istream decompressed(&in);

	arrow::io::IOContext io_context = arrow::io::default_io_context();
	std::shared_ptr<arrow::io::InputStream> input = std::make_shared<StdIStream>(decompressed);

	auto read_options = arrow::csv::ReadOptions::Defaults();
	auto parse_options = arrow::csv::ParseOptions::Defaults();
	//parse_options.delimiter = '\t';
	auto convert_options = arrow::csv::ConvertOptions::Defaults();

	// Instantiate StreamingReader from input stream and options
	auto maybe_reader =
			arrow::csv::StreamingReader::Make(io_context, input, read_options, parse_options, convert_options);
	if (!maybe_reader.ok()) {
		// Handle StreamingReader instantiation error...
		throw std::runtime_error("reader not okay");
	}
	std::shared_ptr<arrow::csv::StreamingReader>& reader = *maybe_reader;

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
		for (auto&& c : batch->columns())
			std::cout << c << std::endl;
	}

	int i = 0;
	for (std::string line; std::getline(decompressed, line) && i < 5; ++i) {
		std::cout << line << std::endl;
	}

	return 0;

	/*MonoBERT model;
	auto dataset = ir::datasets::load<ir::PointwiseDataset>("msmarco-passage/train/judged");
	// auto dataset2 = ir::datasets::load<ir::PairwiseDataset>("msmarco-passage/train/judged");

	auto conf = dl::TrainerConfBuilder<MonoBERT>()
						.setDataset<ir::PointwiseDataset>(std::move(dataset))
						.setOptimizer<dl::optim::Adam>()
						.addObserver(dl::observers::limitEpochs(10))
						.addObserver(dl::observers::earlyStopping(3))
						.addObserver(dl::observers::consoleUI())
						.build();
	auto trainer = dl::Trainer(std::move(conf));
	trainer.fit(model, pairwiseTrainer<MonoBERT>);
	auto results = trainer.test(model, TrecEvaluator(dataset, {"MRR@10", "MAP", "nDCG@10"}), trecEvalAdapter<MonoBERT>);
	for (auto&& [metric, score] : results)
		std::cout << metric << ": " << score << std::endl;*/

	return 0;
}