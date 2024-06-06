#include <dl/device.hpp>
#include <dl/learning/adapters.hpp>
#include <dl/learning/dataset.hpp>
#include <dl/learning/loss.hpp>
#include <dl/learning/optimizers/gradientdescent.hpp>
#include <dl/learning/trainer.hpp>
#include <dl/model/linear.hpp>
#include <dl/model/model.hpp>
#include <dl/tensor/math.hpp>
#include <dl/tensor/tensorimpl.hpp>

#include <cmath>
#include <iostream>
#include <thread>

class MyModel : public dl::Model<dl::Tensor(dl::Tensor&)> {
private:
	dl::Linear linear;

public:
	MyModel() : linear(1, 1, true) { registerSubmodel("linear", linear); }

	//Maybe this solves the duplication https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p0847r7.html
	dl::Tensor forward(dl::Tensor& input) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Simulate doing work
		return linear.forward(std::move(input));
	}
	// dl::Tensor forward(dl::Tensor& input) const { return linear(input); }

	dl::Tensor operator()(dl::Tensor& input) { return forward(input); }
};

class MemoryDataloader : public dl::Dataloader<dl::Tensor(dl::Tensor&&)> {
private:
	using Iterator = dl::utils::GenericIterator<Instance>;
	std::vector<Instance> data;

public:
	explicit MemoryDataloader(std::vector<Instance> data) : data(data) {}
	virtual ~MemoryDataloader() = default;

	Iterator begin() override { return Iterator(data.begin()); }
	Iterator end() override { return Iterator(data.end()); }
};

class MyDataset : public dl::Dataset<dl::Tensor(dl::Tensor&&)> {
private:
	using _DataLoader = dl::Dataloader<dl::Tensor(dl::Tensor&&)>;

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

int main(int argc, char* argv[]) {
	MyModel model;

	auto conf = dl::TrainerConfBuilder<MyModel>()
						.setDataset<MyDataset>()
						.setOptimizer<dl::optim::GradientDescent>(model.parameters())
						.addObserver(dl::observers::limitEpochs(10))
						.addObserver(dl::observers::earlyStopping(3))
						.addObserver(dl::observers::consoleUI())
						.build();
	auto trainer = dl::Trainer(std::move(conf));
	trainer.fit(model, dl::lossAdapter(dl::loss::mse));
	// trainer.test(model);
	return 0;
}