#include <dl/io/weightsfile.hpp>

#include <dl/logging.hpp>

#include <nlohmann/json.hpp>

#include <fstream>
#include <set>
#include <vector>

using json = nlohmann::json;

class SafetensorsFormat final : public dl::io::WeightsFileFormat {
private:
	dl::log::LoggerPtr logger;

public:
	SafetensorsFormat() noexcept : logger(dl::log::getLogger("Safetensors")) {}
	virtual bool canOpen(std::istream& stream) override {
		/** No magic bytes or anything at the beginning of the stream that could give a hint. **/
		return true;
	}
	virtual bool loadModelFromStream(dl::ModelBase& model, std::istream& stream) override {
		std::set<std::string> paramNames;
		for (auto&& [key, _] : model.parameters())
			paramNames.insert(key);

		// Read the header
		uint64_t headerSize;
		stream.read((char*)&headerSize, sizeof(headerSize));
		std::vector<char> headerJSON(headerSize);
		stream.read(headerJSON.data(), headerSize);
		auto json = json::parse(headerJSON.begin(), headerJSON.end());
		// Iterate all tensors defined in the header and check that they match the model. Also track how much data at
		// the end of the file makes up tensor data.
		size_t largestOffset = 0;
		for (auto& [tensorName, metadata] : json.items()) {
			if (tensorName != "__metadata__") {
				auto shape = metadata["shape"];
				assert(shape.is_array());
				auto it = model.parameters().find(tensorName);
				if (it != model.parameters().end()) {
					dl::Tensor& param = it->second;
					bool shapesMatch = std::equal(shape.begin(), shape.end(), param->shape().begin());
					if (!shapesMatch) {
						logger->error(
								"Tensor shapes did not match: {} vs {} for {}", param->shape(), shape.dump(), tensorName
						);
						return false;
					}
					largestOffset = std::max(largestOffset, (size_t)metadata["data_offsets"][1]);
				} else {
					logger->warn("Tensor appears in file but not in the model: {}", tensorName);
				}
			}
		}
		logger->debug("Tensor values take {} bytes", largestOffset);
		// Preload

		return true;
	}

	virtual bool loadModelFromFile(dl::ModelBase& model, std::filesystem::path& path, bool mmap) override {
		/** \todo optimized implementation **/
		std::ifstream stream(path);
		return loadModelFromStream(model, stream);
	}
};

static SafetensorsFormat _safetensorsFormat;
dl::io::WeightsFileFormat& dl::io::safetensorsFormat = _safetensorsFormat;