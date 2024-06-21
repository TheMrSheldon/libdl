#include <dl/utils/urlstream.hpp>

#include <dl/utils/pipe.hpp>

#include <cpr/cpr.h>

#include <iostream>

using namespace std::placeholders;
using dl::utils::pipestream;
using dl::utils::URLStream;

struct URLStream::Data {
	cpr::AsyncWrapper<void, false> wrapper;
	pipestream pipe;
};

URLStream::URLStream(const char* url) noexcept : pImpl{nullptr} {
	// Init pImpl last since it directly starts the download
	auto onDoneCallback = [this](cpr::Response) { pImpl->pipe.close(); };
	auto onDataCallback = [this](std::string data, intptr_t userdata) {
		pImpl->pipe.write(data.data(), data.size());
		return !pImpl->pipe.bad();
	};
	pImpl = std::make_unique<URLStream::Data>(
			cpr::GetCallback(onDoneCallback, cpr::Url{url}, cpr::WriteCallback(onDataCallback))
	);
	this->rdbuf(pImpl->pipe.rdbuf());
}
URLStream::~URLStream() {}
