#include <dl/learning/trainer.hpp>

#include <dl/model/model.hpp>

#include <assert.h>

using dl::ModelBase;
using dl::TrainerObserver;
using dl::TrainerSubject;
using dl::TrainStage;

class LimitEpochs final : public TrainerObserver {
private:
	size_t maxEpochs;
	TrainerSubject* trainer;

public:
	LimitEpochs(size_t maxEpochs) noexcept : maxEpochs(maxEpochs) {}

	virtual void setSubject(TrainerSubject& trainer) override { this->trainer = &trainer; }
	virtual void onTrainingBegun(const ModelBase& model) override {}
	virtual void onTrainingEnded(const ModelBase& model) override {}
	virtual void enterTrainingStage(TrainStage stage) override {}
	virtual void exitTrainingStage() override {}
	virtual void progressChanged(size_t epoch, size_t total, size_t step) override {
		if (epoch >= maxEpochs) {
			assert(trainer != nullptr); // If we don't observe anyone, why should this be called?
			trainer->stop();
		}
	}
};

std::unique_ptr<TrainerObserver> dl::observers::limitEpochs(size_t maxEpochs) noexcept {
	return std::make_unique<LimitEpochs>(maxEpochs);
}