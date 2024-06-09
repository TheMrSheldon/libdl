#ifndef DL_LEARNING_OPTIMIZERS_ADAM_HPP
#define DL_LEARNING_OPTIMIZERS_ADAM_HPP

#include "../optimizer.hpp"

namespace dl::optim {
	/**
     * @brief Implements the Adam optimization algorithm \cite adam .
     */
	class Adam : public dl::Optimizer {
	private:
		const float lr;
		const float beta1;
		const float beta2;
		const float eps;

	public:
		Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
				: dl::Optimizer(), lr(lr), beta1(beta1), beta2(beta2), eps(eps) {}

		virtual void step(dl::TensorPtr& loss) override { throw std::runtime_error("Not yet implemented"); }
	};
} // namespace dl::optim

#endif
