#pragma once

#include <concepts>
#include <memory>

namespace dl {
	class Tensor;

	class TensorPtr : public std::shared_ptr<Tensor> {
	public:
		TensorPtr(std::shared_ptr<Tensor>&& other) : std::shared_ptr<Tensor>(std::move(other)){};
		TensorPtr(const std::shared_ptr<Tensor>& other) : std::shared_ptr<Tensor>(other){};
		TensorPtr(TensorPtr&& other) : std::shared_ptr<Tensor>(std::move(other)){};
		TensorPtr(const TensorPtr& other) : std::shared_ptr<Tensor>(other){};

		TensorPtr(std::nullptr_t p) : std::shared_ptr<Tensor>(p) {}
		TensorPtr(int value);
		TensorPtr(float value);
		TensorPtr(double value);
		TensorPtr(std::initializer_list<int> value);
		TensorPtr(std::initializer_list<float> value);
		TensorPtr(std::initializer_list<double> value);

		/*using std::shared_ptr<Tensor>::operator bool;
        using std::shared_ptr<Tensor>::operator*;
        using std::shared_ptr<Tensor>::operator->;
        using std::shared_ptr<Tensor>::operator=;*/

		TensorPtr& operator=(const TensorPtr& other) {
			this->std::shared_ptr<Tensor>::operator=(other);
			return *this;
		}
		TensorPtr& operator=(TensorPtr&& other) {
			this->std::shared_ptr<Tensor>::operator=(std::move(other));
			return *this;
		}

		bool operator==(const std::nullptr_t& other) { return get() == nullptr; }
	};
} // namespace dl

// These are OK since we only add template specializations: https://en.cppreference.com/w/cpp/language/extending_std
/*
template <>
auto std::begin(dl::TensorPtr& ptr) {
	return ptr->begin();
}
template <>
auto std::end(dl::TensorPtr& ptr) {
	return ptr->end();
}*/