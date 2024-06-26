#include <dl/learning/trainer.hpp>

#include <dl/model/model.hpp>

/** \todo implement me **/

using dl::ModelBase;
using dl::TrainerObserver;
using dl::TrainerSubject;
using dl::TrainStage;

class EarlyStopping final : public TrainerObserver {
public:
	enum class Mode { Min, Max };

public:
	EarlyStopping(size_t patience, Mode mode = Mode::Min) {}

	virtual void setSubject(TrainerSubject& trainer) override {}
	virtual void onTrainingBegun(const ModelBase& model) override {}
	virtual void onTrainingEnded(const ModelBase& model) override {}
	virtual void enterTrainingStage(TrainStage stage) override {}
	virtual void exitTrainingStage() override {}
	virtual void progressChanged(size_t epoch, size_t total, size_t step) override {}
};

std::unique_ptr<TrainerObserver> dl::observers::earlyStopping(size_t patience) noexcept {
	return std::make_unique<EarlyStopping>(patience, EarlyStopping::Mode::Min);
}