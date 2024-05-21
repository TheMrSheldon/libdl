#pragma once

#include "../tensor/tensor.hpp"
#include "../tensor/tensorimpl.hpp"

#include <format>
#include <map>
#include <ranges>

namespace dl {
	class Device;

	class ModelBase {
	private:
		std::map<std::string, dl::TensorRef> _parameters;

	protected:
		void registerParameter(std::string name, Tensor& tensor);
		void registerParameters(std::string prefix, std::ranges::range auto& tensors) {
			for (auto&& [key, value] : tensors)
				_parameters.insert({std::format("{}.{}", prefix, key), value});
		}

	public:
		virtual ~ModelBase() = default;
		size_t numParameters() const noexcept;
		size_t numTrainableParams() const noexcept;
		std::map<std::string, dl::TensorRef>& parameters() noexcept { return _parameters; }
		const std::map<std::string, dl::TensorRef>& parameters() const noexcept { return _parameters; }
	};

	template <typename>
	class Model {};

	template <typename R, typename... Args>
	class Model<R(Args...)> : public virtual ModelBase {
	public:
		using signature = R(Args...);

	protected:
		void registerSubmodel(std::string prefix, const ModelBase& model) {
			registerParameters(prefix, model.parameters());
		}

		virtual R forward(Args... args) = 0;
		/** \todo For later: these const member functions make sense to indicate that we know at compile time that the
		 * instance is not modified (e.g. since it is not part of the computation graph for auto differentiation. **/
		// virtual R forward(Args... args) const = 0;

	public:
		virtual ~Model() = default;
		void to(const Device& device) noexcept;

		R operator()(Args&&... args) { return this->forward(std::forward<Args>(args)...); }
		/** \todo For later: these const member functions make sense to indicate that we know at compile time that the
		 * instance is not modified (e.g. since it is not part of the computation graph for auto differentiation. **/
		// R operator()(Args&&... args) const { return this->forward(std::forward<Args>(args)...); }
	};
} // namespace dl