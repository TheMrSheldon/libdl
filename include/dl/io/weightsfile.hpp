#pragma once

#include "../model/model.hpp"

#include <filesystem>
#include <istream>
#include <string>

namespace dl::io {

	class WeightsFileFormat {
	public:
		virtual ~WeightsFileFormat() = default;

		/**
		 * @brief Populates the model's weights and metainformation from the provided stream.
		 * @details This implementation does not assume the stream to be seekable. That is, it does not "jump around".
		 * Because of this, it may be more memory intensive than
		 * WeightsFileFormat::loadModelFromFile(dl::ModelBase&,std::filesystem::path&,bool), which can make use of that
		 * it knows that it reads from a file and can jump around and memory map it. Instead, this implementation
		 * provides more flexibility since the stream could for example be a dl::utils::URLStream.
		 * 
		 * @param model The model to load the information into.
		 * @param stream 
		 * @return true iff the model was successfully loaded from the stream. 
		 */
		virtual bool loadModelFromStream(dl::ModelBase& model, std::istream& stream) = 0;

		/**
		 * @brief Populates the model's weights and metainformation from the given file.
		 * 
		 * @param model The model to load the information into.
		 * @param path 
		 * @param mmap
		 * @return true iff the model was successfully loaded from the file.
		 */
		virtual bool loadModelFromFile(dl::ModelBase& model, std::filesystem::path& path, bool mmap = true) = 0;

		virtual void writeModelToStream(dl::ModelBase& model, std::ostream& stream) = 0;
	};

	extern dl::io::WeightsFileFormat& safetensorsFormat;
	extern dl::io::WeightsFileFormat& ggufFormat;

} // namespace dl::io