#ifndef DL_UTILS_URLSTREAM_HPP
#define DL_UTILS_URLSTREAM_HPP

#include <experimental/propagate_const>
#include <istream>
#include <memory>

namespace dl::utils {
	/**
	 * @brief Opens an input stream which serves the information pointed at by the provided URL.
	 * @details
	 */
	class URLStream : virtual public std::basic_istream<char> {
	private:
		struct Data;
		std::experimental::propagate_const<std::unique_ptr<Data>> pImpl;

		URLStream() = delete;
		URLStream(const URLStream& other) = delete;
		// Deleted move constructor because we would have to move semaphores and we capture "this" in a lambda
		// expression within the constructor.
		URLStream(URLStream&& other) = delete;

	public:
		/**
		 * @brief Instantiates a new URLStream that serves the data at the given URL.
		 * 
		 * @param url The URL to download data from.
		 */
		explicit URLStream(const char* url) noexcept;
		virtual ~URLStream();
	};
} // namespace dl::utils

#endif