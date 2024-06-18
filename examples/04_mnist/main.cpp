#include <filesystem>
#include <iostream>

#include <dl/learning/adapters.hpp>
#include <dl/learning/evaluators.hpp>
#include <dl/learning/loss.hpp>
#include <dl/learning/optimizers/gradientdescent.hpp>
#include <dl/learning/trainer.hpp>
#include <dl/model/linear.hpp>
#include <dl/model/model.hpp>

class MNIST : public dl::Dataset<dl::TensorPtr(dl::TensorPtr)> {
private:
	std::filesystem::path directory;

	void download();

public:
	MNIST(std::filesystem::path downloaddir) : directory(downloaddir) { download(); }

	virtual std::unique_ptr<dl::Dataloader<dl::TensorPtr(dl::TensorPtr)>> trainingData() override {
		throw std::runtime_error("Not yet implemented");
	}
	virtual std::unique_ptr<dl::Dataloader<dl::TensorPtr(dl::TensorPtr)>> validationData() override {
		throw std::runtime_error("Not yet implemented");
	}
	virtual std::unique_ptr<dl::Dataloader<dl::TensorPtr(dl::TensorPtr)>> testData() override {
		throw std::runtime_error("Not yet implemented");
	}
};

int main(int argc, char* argv[]) {
	dl::Linear model(28 * 28, 1);
	auto conf = dl::TrainerConfBuilder<decltype(model)>()
						.setDataset<MNIST>(std::filesystem::current_path() / "tmp" / "mnist")
						.setOptimizer<dl::optim::GradientDescent>()
						.addObserver(dl::observers::limitEpochs(10))
						.build();
	auto trainer = dl::Trainer(std::move(conf));
	// TODO: implement binary cross entropy loss
	trainer.fit(model, dl::lossAdapter(dl::loss::mse));
	trainer.test(model, dl::MeanError(), dl::lossAdapter(dl::loss::mse));
	return 0;
}