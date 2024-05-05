#pragma once

#include "../tensor/tensor.hpp"
#include "../tensor/tensorptr.hpp"

#include <ranges>
#include <vector>

namespace dl {
	class Device;

	template <typename>
	class Model {};

	template <typename R, typename... Args>
	class Model<R(Args...)> {
	public:
		using signature = R(Args...);

	private:
		std::vector<dl::TensorRef> _parameters;

	protected:
		template <typename T>
		void registerSubmodel(Model<T>& model) {
			registerParameters(model.parameters());
		}
		void registerParameter(TensorPtr& tensor) {
			tensor->setRequiresGrad(true);
			_parameters.push_back(tensor);
		}

		void registerParameters(std::ranges::range auto& tensors) {
			for (TensorPtr& params : tensors)
				params->setRequiresGrad(true);
			_parameters.insert(std::end(_parameters), std::begin(tensors), std::begin(tensors));
		}

		virtual R forward(Args... args) = 0;
		/** \todo For later: these const member functions make sense to indicate that we know at compile time that the
		 * instance is not modified (e.g. since it is not part of the computation graph for auto differentiation. **/
		// virtual R forward(Args... args) const = 0;

	public:
		void to(const Device& device) noexcept;
		std::vector<dl::TensorRef>& parameters() noexcept { return _parameters; }
		const std::vector<dl::TensorRef>& parameters() const noexcept { return _parameters; }

		R operator()(Args... args) { return this->forward(std::forward<Args>(args)...); }
		/** \todo For later: these const member functions make sense to indicate that we know at compile time that the
		 * instance is not modified (e.g. since it is not part of the computation graph for auto differentiation. **/
		// R operator()(Args... args) const { return this->forward(std::forward<Args>(args)...); }
	};
} // namespace dl