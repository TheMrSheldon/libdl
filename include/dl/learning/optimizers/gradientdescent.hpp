#ifndef DL_LEARNING_OPTIMIZERS_GRADIENTDESCENT_HPP
#define DL_LEARNING_OPTIMIZERS_GRADIENTDESCENT_HPP

#include "../../tensor/tensor.hpp"
#include "../optimizer.hpp"

#include <map>
#include <string>

namespace dl::optim {
	class GradientDescent : public dl::Optimizer {
	private:
		const std::map<std::string, dl::TensorRef> parameters;
		const float learnrate;

	public:
		explicit GradientDescent(const std::map<std::string, dl::TensorRef>& parameters, float learnrate = 0.001f)
				: dl::Optimizer(), parameters(parameters), learnrate(learnrate) {}

		virtual void step(const dl::Tensor& loss) override {
			/** \todo reintroduce, currently this gives a segmentation fault **/
			//for (dl::Tensor& tensor : parameters)
			//	tensor->mul_inplace(tensor->grad);
		}
	};
} // namespace dl::optim

#endif
