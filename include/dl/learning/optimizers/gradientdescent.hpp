#ifndef DL_LEARNING_OPTIMIZERS_GRADIENTDESCENT_HPP
#define DL_LEARNING_OPTIMIZERS_GRADIENTDESCENT_HPP

#include "../../tensor/tensorptr.hpp"
#include "../optimizer.hpp"

#include <map>
#include <string>

namespace dl::optim {
	class GradientDescent : public dl::Optimizer {
	private:
		std::map<std::string, dl::TensorRef> parameters;
		const float learnrate;

	public:
		explicit GradientDescent(std::map<std::string, dl::TensorRef>& parameters, float learnrate = 0.001f)
				: dl::Optimizer(), parameters(parameters), learnrate(learnrate) {}

		virtual void step(dl::TensorPtr& loss) override {
			loss->backward();
			for (auto&& [_, tensor] : parameters) {
				auto& gradient = tensor.get()->gradient();
				assert(gradient != nullptr);
				tensor.get() = tensor.get()->add(gradient->mul(dl::constant(-learnrate, gradient->device())));
			}
		}
	};
} // namespace dl::optim

#endif
