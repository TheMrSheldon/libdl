#ifndef DL_UTILS_PIPE_HPP
#define DL_UTILS_PIPE_HPP

#include <assert.h>
#include <barrier>
#include <streambuf>
#include <vector>

namespace dl::utils {
	template <class CharT, class Traits = std::char_traits<CharT>, typename Allocator = std::allocator<CharT>>
	class basic_pipebuf : public std::basic_streambuf<CharT, Traits> {
	private:
		using int_type = typename Traits::int_type;
		bool readerClosed = false;
		bool writerClosed = false;
		std::size_t numLastWritten = 0;
		using pipebuf_type = basic_pipebuf<CharT, Traits, Allocator>;
		std::barrier<> barrier{2};
		std::vector<CharT, Allocator> readbuf;
		std::vector<CharT, Allocator> writebuf;

	public:
		basic_pipebuf(const basic_pipebuf& other) = default;
		basic_pipebuf(basic_pipebuf&& other) = delete;
		basic_pipebuf(size_t bufsize = 1024) : readbuf(bufsize), writebuf(bufsize) {
			// Read Pointers are all at the end of the buffer
			this->setg(readbuf.data(), readbuf.data() + readbuf.size(), readbuf.data() + readbuf.size());
			// Set write buffer to whole buffer
			this->setp(writebuf.data(), writebuf.data() + writebuf.size());
		}
		~basic_pipebuf() {
			if (basic_pipebuf::is_open())
				basic_pipebuf::overflow(Traits::eof());
		}

		bool is_open() const noexcept { return !writerClosed; }

		basic_pipebuf<CharT, Traits, Allocator>* close() noexcept {
			if (!is_open())
				return nullptr;
			overflow(Traits::eof()); // Will close the writer
			return this;
		}

		int_type underflow() override {
			if (readerClosed)
				return Traits::eof();
			barrier.arrive_and_wait();
			// Swap buffers
			std::swap(readbuf, writebuf);
			writerClosed = readerClosed = readerClosed || writerClosed;
			numLastWritten = this->pptr() - this->pbase();
			assert(numLastWritten <= readbuf.size());
			barrier.arrive_and_wait();
			if (readerClosed && numLastWritten == 0) // Reader was freshly closed and no new data was given
				return Traits::eof();
			// Reset Pointer
			this->setg(readbuf.data(), readbuf.data(), readbuf.data() + numLastWritten);
			return Traits::to_int_type(*this->gptr());
		}

		int_type overflow(int_type c = Traits::eof()) override {
			if (c == Traits::eof())
				writerClosed = true;
			barrier.arrive_and_wait();
			// Reader will switch buffers and update *Closed flags
			barrier.arrive_and_wait();
			if (writerClosed)
				return (c == Traits::eof()) ? 1 : Traits::eof();
			// Reset Pointers
			this->setp(writebuf.data(), writebuf.data() + writebuf.size());
			*this->pptr() = c;
			this->pbump(1);
			return c;
		}
	};

	using pipebuf = basic_pipebuf<char>;
	using wpipebuf = basic_pipebuf<wchar_t>;

	template <class CharT, class Traits = std::char_traits<CharT>, typename Allocator = std::allocator<CharT>>
	class basic_pipestream : public std::basic_iostream<CharT, Traits> {
	private:
		basic_pipebuf<CharT, Traits, Allocator> buf;

	public:
		basic_pipestream() noexcept : std::basic_iostream<CharT, Traits>(nullptr) {
			std::basic_iostream<CharT, Traits>::rdbuf(&buf);
		}
		basic_pipestream(const basic_pipestream<CharT, Traits, Allocator>& other) = delete;
		/*basic_pipestream(basic_pipestream<CharT, Traits, Allocator>&& other)
				: std::basic_iostream<CharT, Traits>(nullptr), buf(std::move(other.buf)) {
			std::basic_iostream<CharT, Traits>::rdbuf(&buf);
		}*/

		bool is_open() const noexcept { return buf.is_open(); }
		void close() {
			if (buf.close() == nullptr)
				this->setstate(std::ios_base::failbit);
		}
	};

	using pipestream = basic_pipestream<char>;
	using wpipestream = basic_pipestream<wchar_t>;
} // namespace dl::utils

#endif