#include <dl/io/weightsfile.hpp>

using dl::io::GGUFFormat;

struct GGUFHeader {
	char magic[4];
	uint32_t version;
	uint64_t numTensors;
	uint64_t numMetadataKV;
};

struct GGUFTensor {};

bool GGUFFormat::canOpen(std::istream& stream) {
	/** \todo reset stream after reading and work with streams shorter than 4 byte **/
	char magic[4];
	stream.read(magic, sizeof(magic));
	return magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F';
}
bool GGUFFormat::loadModelFromStream(dl::ModelBase& model, std::istream& stream) {}
bool GGUFFormat::loadModelFromFile(dl::ModelBase& model, std::filesystem::path& path, bool mmap) {}