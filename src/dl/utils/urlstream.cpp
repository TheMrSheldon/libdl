#include <dl/utils/urlstream.hpp>

#include <cpr/cpr.h>
#include <iostream>

using namespace std::placeholders;
using dl::utils::URLStreamBase;

URLStreamBase::URLStreamBase(const char* url) noexcept
		: pImpl{nullptr}, buffer{}, dataIncommingSem{0}, dataHandledSem{0} {
	buffer.reserve(1024);
	setg(0, 0, 0);
	// Init pImpl last since it directly starts the download
	auto onDoneCallback = [this](cpr::Response) {
		ended = true;
		// An empty buffer notifies the underflow() function to return eof()
		buffer.clear();
		dataIncommingSem.release();
	};
	pImpl = std::make_unique<cpr::AsyncWrapper<void, false>>(cpr::GetCallback(
			onDoneCallback, cpr::Url{url}, cpr::WriteCallback(std::bind(&URLStreamBase::onDataCallback, this, _1, _2))
	));
}
URLStreamBase::~URLStreamBase() {
	if (pImpl != nullptr) {
		sync();
	}
}
bool URLStreamBase::onDataCallback(std::string data, intptr_t userdata) {
	// Update the buffer with the received data
	buffer.clear();
	std::copy(data.begin(), data.end(), std::back_inserter(buffer));
	// Signal that new data was read
	dataIncommingSem.release();
	// Wait for the data to be handled
	dataHandledSem.acquire();
	return true;
}
URLStreamBase::int_type URLStreamBase::underflow() {
	if (ended)
		return traits_type::eof();
	// Wait for data
	dataIncommingSem.acquire();
	// Read/handle the information
	URLStreamBase::int_type ret;
	if (pImpl == nullptr || buffer.empty()) {
		ret = traits_type::eof();
	} else {
		setg(buffer.data(), buffer.data(), buffer.data() + buffer.size());
		ret = traits_type::to_int_type(*gptr());
	}
	// Signal CPR's callback that we have handled the information and can read more
	dataHandledSem.release();
	return ret;
}