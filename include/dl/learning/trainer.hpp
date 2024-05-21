#pragma once

#include "../model/model.hpp"
#include "dataloader.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"

#include <functional>
#include <memory>
#include <variant>
#include <vector>

namespace dl {
	class ModelBase;
	template <typename>
	class Model;
	class Device;
	template <typename>
	class Dataset;

	enum class TrainStage { Fitting, Evaluation, Validation };

	class TrainerObserver {
	private:
	public:
		virtual ~TrainerObserver() = default;
		virtual void onTrainingBegun(const ModelBase& model) = 0;
		virtual void onTrainingEnded(const ModelBase& model) = 0;
		virtual void enterTrainingStage(TrainStage stage) = 0;
		virtual void exitTrainingStage() = 0;
		virtual void progressChanged(size_t epoch, size_t total, size_t step) = 0;
	};

	template <typename Model, typename Dataset, typename Optimizer>
	struct TrainerConf {
	public:
		std::vector<std::unique_ptr<TrainerObserver>> observers;
		std::unique_ptr<Dataset> dataset;
		std::unique_ptr<Optimizer> optimizer;
	};

	template <typename Model, typename Dataset, typename Optimizer>
	class Trainer final {
	private:
		TrainerConf<Model, Dataset, Optimizer> conf;
		bool stopped;

		Trainer(const Trainer&) = delete;
		Trainer(Trainer&& other) = delete;
		Trainer& operator=(const Trainer&) = delete;
		Trainer& operator=(Trainer&&) = delete;

		template <typename... Args>
		void notify(void (TrainerObserver::*fn)(Args...), Args&&... args) {
			for (auto&& observer : conf.observers)
				std::invoke(fn, observer, std::forward<Args>(args)...);
		}

	public:
		Trainer(TrainerConf<Model, Dataset, Optimizer>&& conf) : conf(std::move(conf)), stopped(false) {}

		/**
		 * @brief Stops any currently running training, validation and test processes.
		 * @details They may not immediately stop or be interrupted but run until the next step finished.
		 */
		void stop() { stopped = true; }

		void fit(Model model, auto adapter) {
			stopped = false;
			auto dataset = conf.dataset;
			assert(dataset != nullptr);
			auto dataloader = dataset->trainingData();
			assert(dataloader != nullptr);
			notify(&TrainerObserver::onTrainingBegun, model);
			const auto trainsetSize = std::distance(std::begin(*dataloader), std::end(*dataloader));
			for (size_t epoch = 0; !stopped; ++epoch) {
				size_t progress = 0;
				for (auto&& [out, in] : *dataloader) {
					if (stopped)
						break;
					auto loss = adapter(model, std::forward<decltype(in)>(in), std::forward<decltype(out)>(out));
					conf.optimizer->step(loss);
					/** \todo log the loss **/
					notify(&TrainerObserver::progressChanged, epoch, trainsetSize, progress);
					progress += 1; /** \todo with batching increment by batch size **/
				}
			}
			notify(&TrainerObserver::onTrainingEnded, model);
		}
		void validate(Model model, auto evaluator, auto adapter) {}
		auto test(Model model, auto evaluator, auto adapter) {
			auto dataset = conf.dataset;
			assert(dataset != nullptr);
			auto dataloader = dataset->testData();
			assert(dataloader != nullptr);
			for (auto&& [out, in] : *dataloader) {
				evaluator += adapter(model, std::forward<decltype(in)>(in), std::forward<decltype(out)>(out));
			}
			return evaluator.aggregated();
		}
	};

#if 0
	/**
	 * @brief A utilizty class for fitting and evaluating models.
	 */
	template <typename>
	class Trainer;

	template <typename R, typename... Args>
	class Trainer<R(Args...)> final {
	public:
		using T = R(Args...);
		using TDataset = Dataset<T>;
		using TDataloader = Dataloader<T>;
		using DatasetSetupFn = std::function<std::unique_ptr<TDataset>(void)>;
		using OptimizerSetupFn = std::function<std::unique_ptr<TDataset>()>;
		using LossFn = std::function<Tensor(R&&, R&)>;

		struct Settings {
			bool enableCheckpointing = false;
			std::variant<DatasetSetupFn, std::unique_ptr<TDataset>> dataset;
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

		template <typename... Args>
		void notify(void (TrainerObserver::*)(Args...) fn, Args&&... args) {
			for (auto&& observer : settings.observers)
				observer.*fn(std::forward<Args>(args)...);
		}

	protected:
		/** \todo infer correct types **/
		using BatchedOutput = void*;
		using BatchedInput = void*;
		virtual void trainStep(BatchedInput in, BatchedOutput out);

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
			notify(TrainerObserver::onBeginTraining, model);

			// model.fit();
			const auto trainsetSize = std::distance(std::begin(*dataloader), std::end(*dataloader));
			for (unsigned epoch = 0; settings.limitEpochs == 0 || epoch < settings.limitEpochs; ++epoch) {
				size_t progress = 0;
				for (auto&& [out, in] : *dataloader) {
					trainStep(std::forward<decltype(in)>(in), std::forward<decltype(out)>(out));
					auto loss = settings.loss(model(std::forward<decltype(in)>(in)), out);
					settings.optimizer->step(loss);
					notify(TrainerObserver::progressChanged, epoch, trainsetSize, progress);
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
#endif

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
	//template <typename T>
	//using InferTrainer = Trainer<ModelSignature<T>>;

	namespace observers {
		std::unique_ptr<TrainerObserver> earlyStopping(size_t patience) noexcept;
		std::unique_ptr<TrainerObserver> consoleUI() noexcept;
	} // namespace observers
} // namespace dl