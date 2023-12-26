#pragma once

#include "../tensor/tensorptr.hpp"
#include "../tensor/tensor.hpp"

#include <ranges>
#include <vector>

namespace dl {
	class Device;

	template <typename>
	class Model {};

	template <typename R, typename... Args>
	class Model<R(Args...)> {
	public:
		using signature=R(Args...);
	private:
		std::vector<TensorPtr> _parameters;
	protected:
		template<typename T>
		void registerSubmodel(const Model<T>& model) {
			registerParameters(model.parameters());
		}
		void registerParameter(const TensorPtr& tensor) {
			tensor->setRequiresGrad(true);
			_parameters.push_back(tensor);
		}

		void registerParameters(const std::ranges::range auto& tensors) {
			for (auto&& params : tensors)
				params->setRequiresGrad(true);
			_parameters.insert(std::end(_parameters), std::begin(tensors), std::begin(tensors));
		}

		virtual R forward(Args... args) = 0;
		virtual R forward(Args... args) const = 0;

	public:
		void to(const Device& device) noexcept;
		const std::vector<TensorPtr>& parameters() const noexcept { return _parameters; }
		

		R operator()(Args... args) { return this->forward(std::forward<Args>(args)...); }
		R operator()(Args... args) const { return this->forward(std::forward<Args>(args)...); }
	};
} // namespace dl