#pragma once

#include <concepts>
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
	class TensorPtr final {
	private:
		std::shared_ptr<TensorImpl> data;

		explicit TensorPtr(std::shared_ptr<TensorImpl>&& data) : data(std::move(data)) {}

	public:
		TensorPtr(TensorPtr&& other) : data(std::move(other.data)){};
		TensorPtr(const TensorPtr& other);
		TensorPtr(std::nullptr_t p) : data(p) {}
		TensorPtr(int value);
		TensorPtr(float value);
		TensorPtr(double value);
		TensorPtr(InitializerTensor<int> value);
		TensorPtr(InitializerTensor<float> value);
		TensorPtr(InitializerTensor<double> value);

		TensorImpl* operator->() noexcept { return data.get(); }
		const TensorImpl* operator->() const noexcept { return data.get(); }

		TensorImpl& operator*() noexcept { return *data; }
		const TensorImpl& operator*() const noexcept { return *data; }

		TensorPtr& operator=(TensorPtr&& other);
		TensorPtr& operator=(const TensorPtr& other);
		TensorPtr& operator=(std::nullptr_t p);
		TensorPtr& operator=(int value);
		TensorPtr& operator=(float value);
		TensorPtr& operator=(double value);
		TensorPtr& operator=(InitializerTensor<int>&& value);
		TensorPtr& operator=(InitializerTensor<float>&& value);
		TensorPtr& operator=(InitializerTensor<double>&& value);

		bool operator==(const std::nullptr_t& other) const noexcept { return data == other; }
		operator bool() const noexcept { return (bool)data; }

		template <typename T, typename... Args>
		static TensorPtr create(Args&&... args) noexcept {
			return TensorPtr(std::make_unique<T>(std::forward<Args>(args)...));
		}
	};

	using TensorRef = std::reference_wrapper<TensorPtr>;
} // namespace dl
