#ifndef DL_MODEL_MODEL_HPP
#define DL_MODEL_MODEL_HPP

#include "../tensor/tensorimpl.hpp"
#include "../tensor/tensorptr.hpp"

#include <format>
#include <map>
#include <ranges>

namespace dl {
	class Device;

	class ModelBase {
	private:
		std::map<std::string, dl::TensorRef> _parameters;

	protected:
		void registerParameter(const std::string_view& name, TensorPtr& tensor);
		void registerParameters(const std::string_view& prefix, std::ranges::range auto& tensors) {
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

	template <typename M1, typename M2>
	class SeqModel;

	class Model : public virtual ModelBase {
	protected:
		void registerSubmodel(const std::string_view& prefix, const ModelBase& model) {
			registerParameters(prefix, model.parameters());
		}

	public:
		virtual ~Model() = default;
		void to(const Device& device) noexcept;

#ifndef __cpp_explicit_this_parameter
#error "I need deducing this :("
#endif
		// Concatinating two models (i.e., executing the right after the left)
		/**
		 * @brief Concatenates two models (i.e., executing the right after the left (this))
		 * @details Note that the parameters **must** be passed by value to ensure that the inferred type is not downcasted.
		 * 
		 * @param self 
		 * @param right 
		 * @return A model that invokes the self (this) model first and then right on the outputs of self.
		 */
		auto operator|(this auto self, auto right) -> SeqModel<decltype(self), decltype(right)>;
	};
} // namespace dl

#include "seqmodel.hpp"

#endif