#ifndef DL_UTILS_URLSTREAM_HPP
#define DL_UTILS_URLSTREAM_HPP

#include <experimental/propagate_const>
#include <istream>
#include <memory>
#include <string>
#include <vector>

namespace dl::utils {
	/**
	 * @brief Opens an input stream which serves the information pointed at by the provided URL.
	 * @details
	 * \note The implementation is currently unstable
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

		/**
		 * @brief Called internally by the implementation for every chunk of data.
		 * @details Writes the received data chunk to URLStreamBase::buffer, signals that new data is available such
		 * that a waiting (or a future call) to URLStreamBase::underflow() returns the fetched data. Afterwards it waits
		 * until this data was processed and returns.
		 * 
		 * @param data The datachunk that was read.
		 * @param userdata A pointer to some userdata (ignored).
		 * @return true iff we want to continue reading data.
		 */
		bool onDataCallback(std::string data, intptr_t userdata);

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