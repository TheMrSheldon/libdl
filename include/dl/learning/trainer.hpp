#pragma once

#include "../model/model.hpp"
#include "dataloader.hpp"
#include "dataset.hpp"
#include "optimizer.hpp"

#include <functional>
#include <memory>

namespace dl {
	template <typename>
	class Model;
	class Device;
	template <typename>
	class Dataset;

	template<typename>
	class Trainer;

	template <typename R, typename... Args>
	class Trainer<R(Args...)> final {
	public:
		using T = R(Args...);
		using DatasetSetupFn = std::function<std::unique_ptr<Dataset<T>>(void)>;
		using LossFn = std::function<TensorPtr(R, R)>;

		struct Settings {
			bool enableCheckpointing = false;
			DatasetSetupFn createDataset;
			LossFn loss;
			std::unique_ptr<Optimizer> optimizer;
			unsigned limitEpochs = 0;
		};
	private:
		Settings settings;

		Trainer(const Trainer& other) = delete;
		Trainer(Trainer&& other) = delete;
		Trainer& operator=(const Trainer& other) = delete;
		Trainer& operator=(Trainer&& other) = delete;

	public:
		explicit Trainer(Settings&& settings) : settings(std::move(settings)) {
			assert(this->settings.createDataset != nullptr);
			assert(this->settings.loss != nullptr);
		}

		void fit(Model<T>& model) noexcept {
			auto dataset = settings.createDataset();
			assert(dataset != nullptr);
			auto dataloader = dataset->trainingData();
			assert(dataloader != nullptr);
			// model.fit();
			for (unsigned epoch = 0; settings.limitEpochs == 0 || epoch < settings.limitEpochs; ++epoch) {
				for (auto&& [out, in] : *dataloader) {
					auto loss = settings.loss(model(in), out);
					settings.optimizer->step(loss);
				}
				/** \todo: validation loss if configured **/
			}
			
		}

		void validate(const Model<T>& model) const noexcept {
			auto dataset = settings.createDataset();
			assert(dataset != nullptr);
			auto dataloader = dataset->testData();
			assert(dataloader != nullptr);
			// model.inference();
			for (auto&& [out, in] : *dataloader) {
				auto loss = settings.loss(model(in), out);
			}
		}

		void test(const Model<T>& model) const noexcept {
			auto dataset = settings.createDataset();
			assert(dataset != nullptr);
			auto dataloader = dataset->testData();
			assert(dataloader != nullptr);
			// model.inference();
			for (auto&& [out, in] : *dataloader) {
				auto loss = settings.loss(model(in), out);
			}
		}
	};

	// TODO: refine to check that T is actually a Model
	template<typename T>
	struct _ModelSignature {
		using type = T::signature;
	};

	template<typename T>
	using ModelSignature = typename _ModelSignature<T>::type;

	template<typename T>
	using InferTrainer = Trainer<ModelSignature<T>>;
} // namespace dl