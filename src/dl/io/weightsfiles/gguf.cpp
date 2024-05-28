#include <dl/io/weightsfile.hpp>

using dl::io::WeightsFileFormat;

struct GGUFHeader {
	char magic[4];
	uint32_t version;
	uint64_t numTensors;
	uint64_t numMetadataKV;
};

struct GGUFTensor {};

class GGUFFormat final : public WeightsFileFormat {
public:
	virtual bool loadModelFromStream(dl::ModelBase& model, std::istream& stream) override {
		/** \todo implement **/
		return false;
	}
	virtual bool loadModelFromFile(dl::ModelBase& model, std::filesystem::path& path, bool mmap) override {
		/** \todo implement **/
		return false;
	}
	virtual void writeModelToStream(dl::ModelBase& model, std::ostream& stream) override { /** \todo implement **/ }
};

static GGUFFormat _ggufFormat;
dl::io::WeightsFileFormat& dl::io::ggufFormat = _ggufFormat;