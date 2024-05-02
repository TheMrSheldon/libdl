#pragma once

#include <experimental/propagate_const>
#include <memory>

namespace dl {
	class Tensor;

	/**
	 * @brief The TensorPtr is a reference counted pointer to a tensor. It can generally be thought of like an
	 * std::shared_ptr<Tensor>.
	 * @details The TensorPtr serves multiple purposes: (1) implementation hiding and (2) memory management. (1) is
	 * important since the concrete tensor implementation could be, e.g., a sparse tensor, a dense tensor or even
	 * tensors on different devices (i.e., using different drivers like cuda or xtensor). The implementation, generally,
	 * is agnostic to these differences and TensorPtr hides them. (2) is necessary since, in some cases, holding
	 * multiple references to the same tensor can improve memory consumption (e.g., no need to copy the tensor for
	 * gradient calculation when storing it for automatic differentation in the computation graph).
	 * 
	 * \todo Design decision: maybe it would make sense to make the TensorPtr a unique_ptr and only consider (1)
	 * implementation hiding. Automatic differentiation could keep temporary results without copying and inputs have to
	 * be cloned (at least right now) anyway. Without the reference counted pointer behavior, users could maybe better
	 * optimize the data usage in their code and choose to move or copy their data appropriately. It could also be more
	 * natural to C++ developers that
	 * ```cpp
	 * TensorPtr a = ...;
	 * TensorPtr b = a;
	 * ```
	 * copies the data instead of referencing it (though the "Ptr" suffix to this class should make that clear anyway).
	 */
	class TensorPtr final {
	private:
		std::shared_ptr<std::experimental::propagate_const<Tensor>> data;

	public:
		TensorPtr(TensorPtr&& other) : data(std::move(other.data)){};
		TensorPtr(const TensorPtr& other) : data(other.data){};
		TensorPtr(std::nullptr_t p) : data(p) {}
		TensorPtr(int value);
		TensorPtr(float value);
		TensorPtr(double value);
		TensorPtr(std::initializer_list<int> value);
		TensorPtr(std::initializer_list<float> value);
		TensorPtr(std::initializer_list<double> value);

		TensorPtr& operator=(const TensorPtr& other) {
			this->data = other.data;
			return *this;
		}
		TensorPtr& operator=(TensorPtr&& other) {
			this->data = std::move(other.data);
			return *this;
		}

		bool operator==(const std::nullptr_t& other) { return data == other; }
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