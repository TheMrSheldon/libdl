#ifndef DL_UTILS_URLSTREAM_HPP
#define DL_UTILS_URLSTREAM_HPP

#include <experimental/propagate_const>
#include <istream>
#include <memory>
#include <semaphore>
#include <streambuf>
#include <string>
#include <vector>

// Forward declaration for pImpl
// Not for information hiding but the user should not have to add cpr to their dependencies.
namespace cpr {
	template <typename T, bool isCancellable>
	class AsyncWrapper;
	class Response;
} // namespace cpr

namespace dl::utils {
	/**
	 * @brief Opens an input stream which serves the information pointed at by the provided URL.
	 * @see URLStream
	 */
	class URLStreamBase : public std::streambuf {
	private:
		std::experimental::propagate_const<std::unique_ptr<cpr::AsyncWrapper<void, false>>> pImpl;
		std::vector<char> buffer;
		std::binary_semaphore dataIncommingSem;
		std::binary_semaphore dataHandledSem;
		bool ended = false;

		URLStreamBase() = delete;
		URLStreamBase(const URLStreamBase& other) = delete;
		// Deleted move constructor because we would have to move semaphores and we capture "this" in a lambda
		// expression within the constructor.
		URLStreamBase(URLStreamBase&& other) = delete;

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
		explicit URLStreamBase(const char* url) noexcept;
		virtual ~URLStreamBase() override;
		int_type underflow();
	};

	/**
	 * @brief Opens an input stream which serves the information pointed at by the provided URL.
	 */
	class URLStream : virtual URLStreamBase, public std::istream {
	public:
		/**
		 * @brief Instantiates a new URLStream that serves the data at the given URL.
		 * 
		 * @param url The URL to download data from.
		 */
		URLStream(const char* url) : URLStreamBase(url), std::istream(this) {}
	};
} // namespace dl::utils

#endif