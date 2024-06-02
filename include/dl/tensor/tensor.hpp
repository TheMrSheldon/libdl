#pragma once

#include <concepts>
#include <experimental/propagate_const>
#include <memory>
#include <numeric>
#include <vector>

namespace dl {
	class TensorImpl;

	template <typename T>
	struct InitializerTensor {
	private:
		InitializerTensor() = delete;

	public:
		std::vector<T> data;
		std::vector<size_t> shape;

		InitializerTensor(InitializerTensor<T>&& other) noexcept : data(std::move(data)), shape(std::move(shape)) {}
		InitializerTensor(std::initializer_list<T>&& value) noexcept : data(value), shape({value.size()}) {}
		InitializerTensor(std::initializer_list<InitializerTensor>&& value) noexcept : data(), shape() {
			/** \todo Check if all values have the same size**/
			shape = {value.size()};
			data.reserve(std::accumulate(value.begin(), value.end(), 0, [](size_t acc, auto& v) {
				return acc + v.data.size();
			}));
			for (auto&& v : value) {
				data.insert(data.end(), v.data.begin(), v.data.end());
			}
			shape.insert(shape.end(), std::begin(value.begin()->shape), std::end(value.begin()->shape));
		}
		InitializerTensor(std::ranges::range auto range) noexcept
				: data(std::begin(range), std::end(range)), shape({data.size()}) {}
	};

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
		Tensor(InitializerTensor<int> value);
		Tensor(InitializerTensor<float> value);
		Tensor(InitializerTensor<double> value);

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
