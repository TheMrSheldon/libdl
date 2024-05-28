#include <dl/io/weightsfile.hpp>

#include <dl/device.hpp>
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
			} else {
				logger->info("Metadata: {}", metadata.dump());
			}
		}
		logger->debug("Tensor values take {} bytes", largestOffset);
		// Preload
		std::vector<char> tensorData(largestOffset, 0);
		stream.read(tensorData.data(), tensorData.size());
		// Init the tensors
		for (auto& [tensorName, metadata] : json.items()) {
			if (tensorName != "__metadata__") {
				auto shape = metadata["shape"];
				size_t from = metadata["data_offsets"][0];
				size_t to = metadata["data_offsets"][1];
				auto it = model.parameters().find(tensorName);
				if (it != model.parameters().end()) {
					dl::Tensor& param = it->second;
					param = dl::fromBytes<float>(tensorData.data() + from, to - from, shape, param->device());
				}
			}
		}
		return true;
	}

	virtual bool loadModelFromFile(dl::ModelBase& model, std::filesystem::path& path, bool mmap) override {
		/** \todo optimized implementation **/
		std::ifstream stream(path);
		return loadModelFromStream(model, stream);
	}

	virtual void writeModelToStream(dl::ModelBase& model, std::ostream& stream) override {
		json header;
		header["__metadata__"] = json{};
		size_t offset = 0;
		size_t largestTensorBytes = 0;
		for (auto&& [name, tensor] : model.parameters()) {
			/** \todo support different element types **/
			size_t numBytes = dl::numEntries(tensor) * sizeof(float);
			header[name]["dtype"] = "F32";
			header[name]["shape"] = tensor.get()->shape();
			header[name]["data_offsets"] = {offset, offset + numBytes};
			offset += numBytes;
			largestTensorBytes = std::max(largestTensorBytes, numBytes);
		}

		// Write header to stream
		std::string headerBytes = header.dump();
		uint64_t tmp = headerBytes.size();
		stream.write(reinterpret_cast<char*>(&tmp), sizeof(tmp));
		stream.write(headerBytes.c_str(), headerBytes.size());
		// Write tensordata
		std::vector<char> buffer(largestTensorBytes, 0);
		for (auto&& [name, tensor] : model.parameters()) {
			auto numBytes = tensor.get()->toBytes(buffer.data(), buffer.size());
			stream.write(buffer.data(), numBytes);
		}
	}
};

static SafetensorsFormat _safetensorsFormat;
dl::io::WeightsFileFormat& dl::io::safetensorsFormat = _safetensorsFormat;