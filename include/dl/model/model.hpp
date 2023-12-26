#pragma once

#include "../tensor/tensor.hpp"

namespace dl {
	class Device;

	template <typename>
	class Model {};

	template <typename R, typename... Args>
	class Model<R(Args...)> {
	public:
		using signature=R(Args...);
	private:
	protected:
		void registerSubmodel(const Model& model);
		void registerWeights(const TensorPtr& tensor);

		virtual R forward(Args... args) = 0;
		virtual R forward(Args... args) const = 0;

	public:
		void to(const Device& device) noexcept;

		R operator()(Args... args) { return this->forward(std::forward<Args>(args)...); }
		R operator()(Args... args) const { return this->forward(std::forward<Args>(args)...); }
	};
} // namespace dl