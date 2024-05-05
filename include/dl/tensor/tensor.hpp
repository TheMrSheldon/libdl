#pragma once

#include <experimental/propagate_const>
#include <memory>

namespace dl {
	class TensorImpl;

	/**
	 * @brief The Tensor is a managed pointer to a tensor. It can generally be thought of like an
	 * std::unique_ptr<TensorImpl>.
	 * @details The Tensor's main purpose is implementation hiding, which is especially importan since the concrete
	 * tensor implementation could be, e.g., a sparse tensor, a dense tensor or even tensors on different devices (i.e.,
	 * using different drivers like cuda or xtensor). The implementation, generally, is agnostic to these differences
	 * and Tensor hides them.
	 */
	class Tensor final {
	private:
		std::experimental::propagate_const<std::unique_ptr<TensorImpl>> data;

		explicit Tensor(std::experimental::propagate_const<std::unique_ptr<TensorImpl>>&& data)
				: data(std::move(data)) {}

	public:
		Tensor(Tensor&& other) : data(std::move(other.data)){};
		Tensor(const Tensor& other);
		Tensor(std::nullptr_t p) : data(p) {}
		Tensor(int value);
		Tensor(float value);
		Tensor(double value);
		Tensor(std::initializer_list<int> value);
		Tensor(std::initializer_list<float> value);
		Tensor(std::initializer_list<double> value);

		TensorImpl* operator->() noexcept { return data.get(); }
		const TensorImpl* operator->() const noexcept { return data.get(); }

		TensorImpl& operator*() noexcept { return *data; }
		const TensorImpl& operator*() const noexcept { return *data; }

		Tensor& operator=(const Tensor& other);
		Tensor& operator=(Tensor&& other) {
			this->data = std::move(other.data);
			return *this;
		}

		bool operator==(const std::nullptr_t& other) const noexcept { return data == other; }
		operator bool() const noexcept { return (bool)data; }

		template <typename T, typename... Args>
		static Tensor create(Args&&... args) noexcept {
			return Tensor(std::make_unique<T>(std::forward<Args>(args)...));
		}
	};

	using TensorRef = std::reference_wrapper<Tensor>;
} // namespace dl

// These are OK since we only add template specializations: https://en.cppreference.com/w/cpp/language/extending_std
/*
template <>
auto std::begin(dl::Tensor& ptr) {
	return ptr->begin();
}
template <>
auto std::end(dl::Tensor& ptr) {
	return ptr->end();
}*/