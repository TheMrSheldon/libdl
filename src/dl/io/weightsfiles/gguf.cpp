#include <dl/io/weightsfile.hpp>

#include <dl/logging.hpp>

#include <fstream>

using dl::io::WeightsFileFormat;

static constexpr uint32_t MagicWord = 'G' | 'G' << 8 | 'U' << 16 | 'F' << 24;

struct GGUFHeader {
	union {
		char magic[4];
		uint32_t magicWord;
	};
	uint32_t version;
	uint64_t numTensors;
	uint64_t numMetadataKV;
};

struct GGUFTensor {};

class GGUFFormat final : public WeightsFileFormat {
private:
	dl::logging::LoggerPtr logger;

public:
	GGUFFormat() noexcept : logger(dl::logging::getLogger("GGUF")) {}

	virtual bool loadModelFromStream(dl::ModelBase& model, std::istream& stream) override {
		GGUFHeader header;
		stream.read(reinterpret_cast<char*>(&header), sizeof(header));
		// Could have an endianness problem here on some machines... but as long as no one says anything
		if (header.magicWord != MagicWord) {
			logger->error("Invalid sequence at begin of file");
			logger->debug(
					"Magic String was {}{}{}{}", header.magic[0], header.magic[1], header.magic[2], header.magic[3]
			);
			return false;
		} else if (header.version != 3) {
			logger->error("Unsupported GGUF file version: Got {} but I only support {}", header.version, 3);
		}
		/** \todo support metadata **/
		/** \todo implement **/
		return false;
	}
	virtual bool loadModelFromFile(dl::ModelBase& model, std::filesystem::path& path, bool mmap) override {
		/** \todo optimized implementation **/
		std::ifstream stream(path);
		return loadModelFromStream(model, stream);
	}
	virtual void writeModelToStream(dl::ModelBase& model, std::ostream& stream) override {
		/** \todo support metadata **/
		GGUFHeader header{
				.magic = {'G', 'G', 'U', 'F'}, .version = 3, .numTensors = model.numParameters(), .numMetadataKV = 0
		};
		stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
		/** \todo implement **/
	}
};

static GGUFFormat _ggufFormat;
dl::io::WeightsFileFormat& dl::io::ggufFormat = _ggufFormat;