#include <dl/device.hpp>
#include <dl/learning/adapters.hpp>
#include <dl/learning/dataloaders/memorydataloader.hpp>
#include <dl/learning/dataset.hpp>
#include <dl/learning/evaluators.hpp>
#include <dl/learning/loss.hpp>
#include <dl/learning/optimizers/gradientdescent.hpp>
#include <dl/learning/trainer.hpp>
#include <dl/model/linear.hpp>

#include <iostream>

class MyDataset : public dl::Dataset<dl::TensorPtr(dl::TensorPtr)> {
private:
	using _DataLoader = dl::Dataloader<dl::TensorPtr(dl::TensorPtr)>;
	using MemDL = dl::MemoryDataloader<dl::TensorPtr(dl::TensorPtr)>;

public:
	MyDataset() {}
	virtual ~MyDataset() override = default;

	std::unique_ptr<_DataLoader> trainingData() override {
		// y = 5x+3
		std::vector<_DataLoader::Instance> data = {{13.0f, 2.0f}, {23.0f, 4.0f}, {-22.0f, -5.0f}};
		return {std::make_unique<MemDL>(data)};
	}
	std::unique_ptr<_DataLoader> validationData() override {
		// y = 5x+3
		std::vector<_DataLoader::Instance> data = {{18.0f, 3.0f}, {508.0f, 101.0f}};
		return {std::make_unique<MemDL>(data)};
	}
	std::unique_ptr<_DataLoader> testData() override {
		// y = 5x+3
		std::vector<_DataLoader::Instance> data = {{-92.0f, -19.0f}, {1018.0f, 203.0f}};
		return {std::make_unique<MemDL>(data)};
	}
};

int main(int argc, char* argv[]) {
	dl::Linear model(1, 1, true);
	model.bias()->setRequiresGrad(true);
	model.weights()->setRequiresGrad(true);

	auto conf = dl::TrainerConfBuilder<dl::Linear>()
						.setDataset<MyDataset>()
						.setOptimizer<dl::optim::GradientDescent>(model.parameters())
						.addObserver(dl::observers::limitEpochs(10000))
						.addObserver(dl::observers::earlyStopping(3))
						// .addObserver(dl::observers::consoleUI())
						.build();
	auto trainer = dl::Trainer(std::move(conf));
	auto before = trainer.test(model, dl::MeanError(), dl::lossAdapter(dl::loss::mse));
	trainer.fit(model, dl::lossAdapter(dl::loss::mse));
	auto after = trainer.test(model, dl::MeanError(), dl::lossAdapter(dl::loss::mse));

	std::cout << "Test Loss Before Training: " << before << std::endl;
	std::cout << "Test Loss After Training:  " << after << std::endl;
	std::cout << "Learnt function: " << model.weights() << "x + " << model.bias() << std::endl;
	return 0;
}