#include <bit>
#include <filesystem>
#include <format>
#include <iostream>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <dl/learning/adapters.hpp>
#include <dl/learning/evaluators.hpp>
#include <dl/learning/loss.hpp>
#include <dl/learning/optimizers/gradientdescent.hpp>
#include <dl/learning/trainer.hpp>
#include <dl/logging.hpp>
#include <dl/model/linear.hpp>
#include <dl/model/model.hpp>
#include <dl/utils/urlstream.hpp>

/**
 * @brief Converts the given uint32 from network byte order (big endian) to the host byte order.
 * 
 * @param netlong 
 * @return 
 */
static constexpr uint32_t ntohl(uint32_t netlong) {
	if constexpr (std::endian::native == std::endian::little)
		return std::byteswap(netlong);
	return netlong;
}

struct MNISTImageFile {
	struct {
		uint32_t magic;
		uint32_t numImages;
		uint32_t numRows;
		uint32_t numCols;
	} header;
	std::vector<uint8_t> data;

	/**
	 * @brief Converts the image data into a NxRxC tensor
	 * @details
	 * \todo Return a uint8 tensor when other tensor types than float are implemented
	 * 
	 * @return dl::TensorPtr 
	 */
	dl::TensorPtr asTensor() const noexcept {
		return dl::reshape(
				dl::constant(dl::InitializerTensor<float>(data)), {header.numImages, header.numRows, header.numCols}
		);
	}

	static MNISTImageFile FromStream(std::istream& stream) noexcept {
		MNISTImageFile file;
		stream.read(reinterpret_cast<char*>(&file.header), sizeof(file.header));
		file.header.magic = ntohl(file.header.magic);
		file.header.numImages = ntohl(file.header.numImages);
		file.header.numRows = ntohl(file.header.numRows);
		file.header.numCols = ntohl(file.header.numCols);
		assert(file.header.magic == 0x00000803);
		file.data = std::vector<uint8_t>(file.header.numImages * file.header.numRows * file.header.numCols);
		stream.read(reinterpret_cast<char*>(file.data.data()), file.data.size());
		assert(!stream.fail());
		return file;
	}
};

struct MNISTLabelFile {
	struct {
		uint32_t magic;
		uint32_t numLabels;
	} header;
	std::vector<uint8_t> data;

	/**
	 * @brief Converts the label data into a N tensor
	 * @details
	 * \todo Return a uint8 tensor when other tensor types than float are implemented
	 * 
	 * @return dl::TensorPtr 
	 */
	dl::TensorPtr asTensor() const noexcept { return dl::constant(dl::InitializerTensor<float>(data)); }

	static MNISTLabelFile FromStream(std::istream& stream) noexcept {
		MNISTLabelFile file;
		stream.read(reinterpret_cast<char*>(&file.header), sizeof(file.header));
		file.header.magic = ntohl(file.header.magic);
		file.header.numLabels = ntohl(file.header.numLabels);
		assert(file.header.magic == 0x00000801);
		file.data = std::vector<uint8_t>(file.header.numLabels);
		stream.read(reinterpret_cast<char*>(file.data.data()), file.data.size());
		assert(!stream.fail());
		return file;
	}
};

static_assert(sizeof(MNISTImageFile::header) == 16);

class MNIST : public dl::Dataset<dl::TensorPtr(dl::TensorPtr)> {
private:
	dl::logging::LoggerPtr logger;
	std::string mirror = "https://ossci-datasets.s3.amazonaws.com/mnist/";
	dl::TensorPtr trainImages = nullptr;
	dl::TensorPtr trainLabels = nullptr;

	void download() {
		/** \todo Confirm hashes **/
		logger->info("Downloading MNIST");
		{ // Train Images
			dl::utils::URLStream download{std::format("{}train-images-idx3-ubyte.gz", mirror).c_str()};
			boost::iostreams::filtering_istream decompressed{boost::iostreams::gzip_decompressor()};
			decompressed.push(download);
			trainImages = MNISTImageFile::FromStream(decompressed).asTensor();
			logger->info(
					"Loaded {} Trainig Images of size {}x{}", trainImages->shape(0), trainImages->shape(1),
					trainImages->shape(2)
			);
		}
		{ // Train Labels
			dl::utils::URLStream download{std::format("{}train-labels-idx1-ubyte.gz", mirror).c_str()};
			boost::iostreams::filtering_istream decompressed{boost::iostreams::gzip_decompressor()};
			decompressed.push(download);
			trainLabels = MNISTLabelFile::FromStream(decompressed).asTensor();
			logger->info("Loaded {} Trainig Labels", trainLabels->shape(0));
		}
		logger->info("Download Done");
	}

public:
	MNIST() : logger(dl::logging::getLogger("MNIST")) { download(); }

	virtual std::unique_ptr<dl::Dataloader<dl::TensorPtr(dl::TensorPtr)>> trainingData() override {
		throw std::runtime_error("Not yet implemented");
	}
	virtual std::unique_ptr<dl::Dataloader<dl::TensorPtr(dl::TensorPtr)>> validationData() override {
		throw std::runtime_error("Not yet implemented");
	}
	virtual std::unique_ptr<dl::Dataloader<dl::TensorPtr(dl::TensorPtr)>> testData() override {
		throw std::runtime_error("Not yet implemented");
	}
};

int main(int argc, char* argv[]) {
	dl::logging::setVerbosity(dl::logging::Verbosity::Debug);
	auto logger = dl::logging::getLogger("main");

	dl::Linear model(28 * 28, 1);
	auto conf = dl::TrainerConfBuilder<decltype(model)>()
						.setDataset<MNIST>()
						.setOptimizer<dl::optim::GradientDescent>(model.parameters())
						.addObserver(dl::observers::limitEpochs(10))
						.build();
	auto trainer = dl::Trainer(std::move(conf));
	trainer.fit(model, dl::lossAdapter(dl::loss::bce));
	trainer.test(model, dl::MeanError(), dl::lossAdapter(dl::loss::bce));
	return 0;
}