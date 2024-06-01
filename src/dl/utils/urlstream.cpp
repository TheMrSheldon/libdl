#include <dl/utils/urlstream.hpp>

#include <boost/process/pipe.hpp>
#include <cpr/cpr.h>

#include <iostream>

using namespace std::placeholders;
using dl::utils::URLStream;

struct URLStream::Data {
	cpr::AsyncWrapper<void, false> wrapper;
	boost::process::pstream pipe;
};

URLStream::URLStream(const char* url) noexcept : pImpl{nullptr} {
	// Init pImpl last since it directly starts the download
	auto onDoneCallback = [this](cpr::Response) {
		pImpl->pipe.close();
		pImpl->pipe.pipe().close();
	};
	pImpl = std::make_unique<URLStream::Data>(URLStream::Data{
			.wrapper = cpr::GetCallback(
					onDoneCallback, cpr::Url{url},
					cpr::WriteCallback(std::bind(&URLStream::onDataCallback, this, _1, _2))
			),
			.pipe = {}
	});
	this->rdbuf(pImpl->pipe.rdbuf());
}
URLStream::~URLStream() {}
bool URLStream::onDataCallback(std::string data, intptr_t userdata) {
	// Update the buffer with the received data
	pImpl->pipe.pipe().write(data.data(), data.size());
	return !pImpl->pipe.bad();
}