#include <bit>
#include <filesystem>
#include <format>
#include <iostream>

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <dl/learning/adapters.hpp>
#include <dl/learning/dataloaders/memorydataloader.hpp>
#include <dl/learning/evaluators.hpp>
#include <dl/learning/loss.hpp>
#include <dl/learning/optimizers/gradientdescent.hpp>
#include <dl/learning/trainer.hpp>
#include <dl/logging.hpp>
#include <dl/model/linear.hpp>
#include <dl/model/model.hpp>
#include <dl/utils/urlstream.hpp>

static_assert(std::ranges::viewable_range<dl::TensorPtr>);
static_assert(std::ranges::input_range<dl::TensorPtr>);

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
	using MemDL = dl::MemoryDataloader<dl::TensorPtr(dl::TensorPtr)>;

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
		std::vector<MemDL::Instance> data;
		for (const auto& entry : std::views::zip(trainLabels, trainImages))
			data.emplace_back(entry);
		return std::make_unique<MemDL>(data);
		//throw std::runtime_error("Not yet implemented");
	}
	virtual std::unique_ptr<dl::Dataloader<dl::TensorPtr(dl::TensorPtr)>> validationData() override {
		throw std::runtime_error("Not yet implemented");
	}
	virtual std::unique_ptr<dl::Dataloader<dl::TensorPtr(dl::TensorPtr)>> testData() override {
		throw std::runtime_error("Not yet implemented");
	}
};

/*namespace dl {
	template <typename>
	struct _Apply;
	template <typename R, typename... Args>
	struct _Apply<R(Args...)> : public Model {
		std::function<R(Args...)> fun;

		explicit _Apply(std::function<R(Args...)> fun) noexcept : fun(fun) {}

		R operator()(Args... args) { return fun(std::forward<Args>(args)...); }
	};

	template <typename... Args>
	auto apply(std::invocable<Args...> auto fun) {
		//auto apply(std::function<Sign> fun) -> _Apply<Sign> {
		using R = decltype(fun(std::declval<Args>()...));
		return _Apply<R(Args...)>{fun};
	}
} // namespace dl*/

namespace dl {
	template <typename T>
	struct _Apply : public Model {
		T val;

		explicit _Apply(T&& val) noexcept : val(val) {}

		template <typename... Args>
		auto operator()(Args&&... args) {
			return val(std::forward<Args>(args)...);
		}
	};

	auto apply(auto&& fun, auto&&... args) {
		auto bound = std::bind(fun, std::forward<decltype(args)>(args)...);
		return _Apply<decltype(bound)>{std::move(bound)};
	}
} // namespace dl

int main(int argc, char* argv[]) {
	dl::logging::setVerbosity(dl::logging::Verbosity::Debug);
	auto logger = dl::logging::getLogger("main");

	// dl::Linear model(28 * 28, 1);
	//auto reshape = dl::apply<dl::TensorPtr>(std::bind(dl::reshape, std::placeholders::_1, dl::SShape{-1, 28 * 28}));
	auto reshape = dl::apply(dl::reshape, std::placeholders::_1, dl::SShape{-1, 28 * 28});
	auto model = reshape | dl::Linear{28 * 28, 1};
	auto conf = dl::TrainerConfBuilder<decltype(model)>()
						.setDataset<MNIST>()
						.setOptimizer<dl::optim::GradientDescent>(model.parameters())
						.addObserver(dl::observers::limitEpochs(10))
						.addObserver(dl::observers::consoleUI())
						.build();
	auto trainer = dl::Trainer(std::move(conf));
	trainer.fit(model, dl::lossAdapter(dl::loss::bce));
	// trainer.test(model, dl::MeanError(), dl::lossAdapter(dl::loss::bce));
	return 0;
}