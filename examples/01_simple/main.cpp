#include <iostream>

//#include <dl/learning/loss.hpp>
//#include <dl/learning/trainer.hpp>
//#include <dl/model/linear.hpp>
//#include <dl/model/model.hpp>
#include <dl/tensor/tensor.hpp>
#include <dl/tensor/math.hpp>

/*class MyModel : public dl::Model<dl::TensorPtr(dl::TensorPtr)> {
private:
	dl::Linear linear;

public:
	MyModel() : linear(1, 1, true) {}

	dl::TensorPtr forward(dl::TensorPtr input) { return linear(input); }
	dl::TensorPtr forward(dl::TensorPtr input) const { return linear(input); }
};

// https://arxiv.org/abs/1412.6980
class Adam : public dl::Optimizer {
private:
	float lr;
	float beta1;
	float beta2;
	float eps;

public:
	Adam(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
			: Optimizer(), lr(lr), beta1(beta1), beta2(beta2), eps(eps) {}

	virtual void step(dl::TensorPtr loss) override {
		auto m0 = 0;
		auto v0 = 0;
		int t = 0;
	}
};*/

int main(int argc, char* argv[]) {
	std::cout << "Hello world" << std::endl;

	auto tensor = dl::ones({2, 2});
	std::cout << tensor << std::endl;

	/*MyModel model;

	// using Trainer = dl::InferTrainer<MyModel>;
	using Trainer = dl::InferTrainer<MyModel>;
	static_assert(!std::is_same_v<Trainer, void>);
	static_assert(std::is_same_v<Trainer, dl::Trainer<dl::TensorPtr(dl::TensorPtr)>>);
	Trainer::Settings settings = {
		.loss = dl::loss::mse,
		.optimizer = std::make_unique<Adam>()
	};
	Trainer trainer(std::move(settings));
	trainer.fit(model);
	trainer.test(model);*/
	return 0;
}