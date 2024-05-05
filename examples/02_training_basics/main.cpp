#include <iostream>

#include <dl/device.hpp>
#include <dl/learning/dataset.hpp>
#include <dl/learning/loss.hpp>
#include <dl/learning/trainer.hpp>
#include <dl/model/linear.hpp>
#include <dl/model/model.hpp>
#include <dl/tensor/math.hpp>
#include <dl/tensor/tensor.hpp>

#include <cmath>

class MyModel : public dl::Model<dl::TensorPtr(dl::TensorPtr&)> {
private:
	dl::Linear linear;

public:
	MyModel() : linear(1, 1, false) { registerSubmodel(linear); }

	//Maybe this solves the duplication https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0847r7.html
	dl::TensorPtr forward(dl::TensorPtr& input) { return linear(input); }
	// dl::TensorPtr forward(dl::TensorPtr& input) const { return linear(input); }
};

class MemoryDataloader : public dl::Dataloader<dl::TensorPtr(dl::TensorPtr&)> {
private:
	using Iterator = dl::utils::GenericIterator<Instance>;
	std::vector<Instance> data;

public:
	explicit MemoryDataloader(std::vector<Instance> data) : data(data) {}
	virtual ~MemoryDataloader() = default;

	Iterator begin() override { return Iterator(data.begin()); }
	Iterator end() override { return Iterator(data.end()); }
};

class MyDataset : public dl::Dataset<dl::TensorPtr(dl::TensorPtr&)> {
private:
	using _DataLoader = dl::Dataloader<dl::TensorPtr(dl::TensorPtr&)>;

public:
	MyDataset() {}
	virtual ~MyDataset() = default;

	std::unique_ptr<_DataLoader> trainingData() override {
		std::vector<_DataLoader::Instance> data = {{1.0f, 2.0f}, {3.0f, 4.0f}, {-2.0f, -1.0f}};
		return {std::make_unique<MemoryDataloader>(data)};
	}
	std::unique_ptr<_DataLoader> validationData() override {
		std::vector<_DataLoader::Instance> data = {{2.0f, 3.0f}, {100.0f, 101.0f}};
		return {std::make_unique<MemoryDataloader>(data)};
	}
	std::unique_ptr<_DataLoader> testData() override {
		std::vector<_DataLoader::Instance> data = {{-20.0f, -19.0f}, {202.0f, 203.0f}};
		return {std::make_unique<MemoryDataloader>(data)};
	}
};

class GradientDescent : public dl::Optimizer {
private:
	const std::vector<dl::TensorRef> parameters;
	const float learnrate;

public:
	explicit GradientDescent(const std::vector<dl::TensorRef>& parameters, float learnrate = 0.001)
			: dl::Optimizer(), parameters(parameters), learnrate(learnrate) {}

	virtual void step(const dl::TensorPtr& loss) override {
		for (dl::TensorPtr& tensor : parameters)
			tensor->mul_inplace(tensor->grad);
	}
};

// https://arxiv.org/abs/1412.6980
class Adam : public dl::Optimizer {
private:
	const float lr;
	const float beta1;
	const float beta2;
	const float eps;

public:
	Adam(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
			: dl::Optimizer(), lr(lr), beta1(beta1), beta2(beta2), eps(eps) {}

	virtual void step(const dl::TensorPtr& loss) override {
		auto m0 = 0;
		auto v0 = 0;
		int t = 0;
	}
};

int main(int argc, char* argv[]) {
	MyModel model;

	using Trainer = dl::InferTrainer<MyModel>;
	Trainer trainer(Trainer::Settings{
			.createDataset = [] { return std::make_unique<MyDataset>(); },
			.loss = dl::loss::mse,
			.optimizer = std::make_unique<GradientDescent>(model.parameters()),
			.limitEpochs = 10,
	});
	trainer.fit(model);
	trainer.test(model);
	return 0;
}