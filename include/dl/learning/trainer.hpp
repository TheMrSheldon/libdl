#pragma once

#include "../model/model.hpp"
#include "dataloader.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace dl {
	class ModelBase;
	template <typename>
	class Model;
	class Device;
	template <typename>
	class Dataset;

	/**
	 * @brief A utilizty class for fitting and evaluating models.
	 */
	template <typename>
	class Trainer;

	enum class TrainStage { Fitting, Evaluation, Validation };

	class TrainerObserver {
	private:
	public:
		virtual ~TrainerObserver() = default;
		virtual void onBeginTraining(const ModelBase& model) = 0;
		virtual void enterTrainingStage(TrainStage stage) = 0;
		virtual void exitTrainingStage() = 0;
		virtual void progressChanged(size_t epoch, size_t total, size_t step) = 0;
	};

	template <typename R, typename... Args>
	class Trainer<R(Args...)> final {
	public:
		using T = R(Args...);
		using TDataset = Dataset<T>;
		using TDataloader = Dataloader<T>;
		using DatasetSetupFn = std::function<std::unique_ptr<TDataset>(void)>;
		using LossFn = std::function<Tensor(R&&, R&)>;

		struct Settings {
			bool enableCheckpointing = false;
			DatasetSetupFn createDataset;
			LossFn loss;
			std::unique_ptr<Optimizer> optimizer;
			unsigned limitEpochs = 0;
			std::vector<std::unique_ptr<TrainerObserver>> observers;
		};

	private:
		Settings settings;

		Trainer(const Trainer& other) = delete;
		Trainer(Trainer&& other) = delete;
		Trainer& operator=(const Trainer& other) = delete;
		Trainer& operator=(Trainer&& other) = delete;

		/*inline static float validation_step(TDataloader& dataloader) const noexcept {
			unsigned count = 0;
			float sum = 0;
			for (auto&& [out, in] : *dataloader) {
				auto loss = settings.loss(model(in), out);
				sum += (float) loss;
				count++;
			}
			return sum / count;
		}*/

	public:
		explicit Trainer(Settings&& settings) : settings(std::move(settings)) {
			assert(this->settings.createDataset != nullptr);
			assert(this->settings.loss != nullptr);
		}

		/**
		 * @brief Trains the model.
		 * @details Calling fit will instantiate the dataset as it was configured in Trainer::Settings and then trains
		 * the model until any of the termination criteria specified in Trainer::Settings are satisfied.
		 * 
		 * @param model The model to train
		 */
		void fit(Model<T>& model) noexcept {
			auto dataset = settings.createDataset();
			assert(dataset != nullptr);
			auto dataloader = dataset->trainingData();
			assert(dataloader != nullptr);
			auto validationData = dataset->validationData();
			assert(validationData != nullptr);

			std::for_each(settings.observers.begin(), settings.observers.end(), [&model](auto& o) {
				o->onBeginTraining(model);
			});

			// model.fit();
			const auto trainsetSize = std::distance(std::begin(*dataloader), std::end(*dataloader));
			for (unsigned epoch = 0; settings.limitEpochs == 0 || epoch < settings.limitEpochs; ++epoch) {
				size_t progress = 0;
				for (auto&& [out, in] : *dataloader) {
					auto loss = settings.loss(model(std::move(in)), out);
					settings.optimizer->step(loss);
					std::for_each(
							settings.observers.begin(), settings.observers.end(),
							[epoch, trainsetSize, progress](auto& o) {
								o->progressChanged(epoch, trainsetSize, progress);
							}
					);
					progress += 1; /** \todo with batching increment by batch size**/
				}
				/** \todo validation loss if configured **/
				// auto valLoss = validation_step(validationData);
			}
		}

		void validate(const Model<T>& model) const noexcept {
			auto dataset = settings.createDataset();
			assert(dataset != nullptr);
			auto dataloader = dataset->validationData();
			assert(dataloader != nullptr);
			// model.inference();
			for (auto&& [out, in] : *dataloader) {
				auto loss = settings.loss(model(std::move(in)), out);
			}
		}

		void test(/*const */ Model<T>& model) const noexcept {
			auto dataset = settings.createDataset();
			assert(dataset != nullptr);
			auto dataloader = dataset->testData();
			assert(dataloader != nullptr);
			// model.inference();
			for (auto&& [out, in] : *dataloader) {
				auto loss = settings.loss(model(std::move(in)), out);
			}
		}
	};

	/// \todo refine to check that T is actually a Model

	namespace detail {
		/**
		 * @brief Infers the model signature from the provided model type.
		 * 
		 * @tparam T the model type for which to infer the model signature.
		 */
		template <typename T>
		struct _ModelSignature {
			using type = T::signature;
		};
	} // namespace detail

	/**
	 * @brief Infers the model signature from the provided model type.
	 * 
	 * @tparam T the model type for which to infer the model signature.
	 */
	template <typename T>
	using ModelSignature = typename detail::_ModelSignature<T>::type;

	/**
	 * @brief Represents the trainer type required to train the specified model type.
	 * 
	 * @tparam T the type of the model that should be trained.
	 */
	template <typename T>
	using InferTrainer = Trainer<ModelSignature<T>>;

	namespace observers {
		std::unique_ptr<TrainerObserver> earlyStopping(size_t patience) noexcept;
		std::unique_ptr<TrainerObserver> consoleUI() noexcept;
	} // namespace observers
} // namespace dl