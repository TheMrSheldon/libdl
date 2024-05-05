#pragma once

#include <experimental/propagate_const>
#include <memory>

namespace dl {
	class Tensor;

	/**
	 * @brief The TensorPtr is a managed pointer to a tensor. It can generally be thought of like an
	 * std::unique_ptr<Tensor>.
	 * @details The TensorPtr's main purpose is implementation hiding, which is especially importan since the concrete
	 * tensor implementation could be, e.g., a sparse tensor, a dense tensor or even tensors on different devices (i.e.,
	 * using different drivers like cuda or xtensor). The implementation, generally, is agnostic to these differences
	 * and TensorPtr hides them.
	 */
	class TensorPtr final {
	private:
		std::experimental::propagate_const<std::unique_ptr<Tensor>> data;

		explicit TensorPtr(std::experimental::propagate_const<std::unique_ptr<Tensor>>&& data)
				: data(std::move(data)) {}

	public:
		TensorPtr(TensorPtr&& other) : data(std::move(other.data)){};
		TensorPtr(const TensorPtr& other);
		TensorPtr(std::nullptr_t p) : data(p) {}
		TensorPtr(int value);
		TensorPtr(float value);
		TensorPtr(double value);
		TensorPtr(std::initializer_list<int> value);
		TensorPtr(std::initializer_list<float> value);
		TensorPtr(std::initializer_list<double> value);

		Tensor* operator->() noexcept { return data.get(); }
		const Tensor* operator->() const noexcept { return data.get(); }

		Tensor& operator*() noexcept { return *data; }
		const Tensor& operator*() const noexcept { return *data; }

		TensorPtr& operator=(const TensorPtr& other);
		TensorPtr& operator=(TensorPtr&& other) {
			this->data = std::move(other.data);
			return *this;
		}

		bool operator==(const std::nullptr_t& other) const noexcept { return data == other; }
		operator bool() const noexcept { return (bool)data; }

		template <typename T, typename... Args>
		static TensorPtr create(Args&&... args) noexcept {
			return TensorPtr(std::make_unique<T>(std::forward<Args>(args)...));
		}
	};

	using TensorRef = std::reference_wrapper<TensorPtr>;
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