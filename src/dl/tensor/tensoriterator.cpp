#include <dl/tensor/tensoriterator.hpp>

#include <dl/tensor/tensorimpl.hpp>

using dl::TensorIter;
using dl::TensorPtr;
using dl::TensorSentinel;

TensorIter::TensorIter() : ptr(nullptr), idx(0) {}
TensorIter::TensorIter(TensorSentinel) : TensorIter() {}
TensorIter::TensorIter(TensorPtr ptr) : ptr(ptr), idx(0) {}
TensorIter::TensorIter(const TensorIter& other) : ptr(other.ptr), idx(other.idx) {}
TensorIter::TensorIter(TensorIter&& other) noexcept : ptr(std::move(other.ptr)), idx(std::move(other.idx)) {}

TensorIter& TensorIter::operator=(const TensorIter& other) noexcept {
	this->ptr = other.ptr;
	this->idx = other.idx;
	return *this;
}
TensorIter& TensorIter::operator=(TensorIter&& other) noexcept {
	this->ptr = std::move(other.ptr);
	this->idx = std::move(other.idx);
	return *this;
}

TensorPtr TensorIter::operator*() const { return ptr->get({idx}); }

TensorIter& TensorIter::operator++() noexcept {
	++idx;
	return *this;
}
TensorIter TensorIter::operator++(int) noexcept {
	auto copy = *this;
	(*this)++;
	return copy;
}

bool TensorIter::operator==(TensorSentinel s) const noexcept { return ptr == nullptr || idx >= ptr->shape(0); }