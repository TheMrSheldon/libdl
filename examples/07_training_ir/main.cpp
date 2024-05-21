#include <iostream>

#include <dl/learning/loss.hpp>
#include <dl/learning/trainer.hpp>
#include <dl/model/model.hpp>
#include <dl/utils/urlstream.hpp>
#include <ir/data/datasets.hpp>

class MonoBERT : public dl::ModelBase {
private:
public:
};

class TrecEvaluator final {
public:
	using GradeTriple = std::tuple<ir::Query, ir::Document, float>;

private:
	std::map<size_t, std::map<size_t, float>> grades;

public:
	TrecEvaluator(auto& dataset, std::vector<std::string> metrics) {}

	TrecEvaluator& operator+=(GradeTriple result) {
		grades[std::get<0>(result).id][std::get<1>(result).id] = std::get<2>(result);
		return *this;
	}

	std::map<std::string, float> aggregated();
};

dl::Tensor pairwiseTrainer(auto& model, ir::Query query, ir::Document pos, ir::Document neg) {
	return model(query, neg) - model(query, pos);
}

TrecEvaluator::GradeTriple trecEvalAdapter(auto& model, ir::Query query, ir::Document doc) {
	return std::make_tuple(query, doc, model(query, doc));
}

namespace dl {
	class AdamW;
	class LimitEpochs;
	class EarlyStopping;
	class ConsoleUI;

	template <typename Model = void*, typename Dataset = void*, typename Optimizer = void*>
	class TrainerConfBuilder {
	public:
		template <typename DSet, typename... Args>
		TrainerConfBuilder<Model, DSet, Optimizer> setDataset(Args&&... args);

		template <typename Opt, typename... Args>
		TrainerConfBuilder<Model, Dataset, Opt> setOptimizer(Args&&... args);

		template <typename O, typename... Args>
		TrainerConfBuilder& addObserver(Args&&... args);

		TrainerConf<Model, Dataset, Optimizer> build();
	};
} // namespace dl

int main(int argc, char* argv[]) {
	MonoBERT model;
	auto dataset = ir::datasets::load<ir::PointwiseDataset>("msmarco-passage/train/judged");
	// auto dataset2 = ir::datasets::load<ir::PairwiseDataset>("msmarco-passage/train/judged");

	auto conf = dl::TrainerConfBuilder<MonoBERT>()
						.setDataset<ir::PointwiseDataset>(std::move(dataset))
						.setOptimizer<dl::AdamW>()
						.addObserver<dl::LimitEpochs>(10)
						.addObserver<dl::EarlyStopping>(3)
						.addObserver<dl::ConsoleUI>()
						.build();
	auto trainer = dl::Trainer(std::move(conf));
	trainer.fit(model, pairwiseTrainer<MonoBERT>);
	auto results = trainer.test(model, TrecEvaluator(dataset, {"MRR@10", "MAP", "nDCG@10"}), trecEvalAdapter<MonoBERT>);
	for (auto&& [metric, score] : results)
		std::cout << metric << ": " << score << std::endl;

	return 0;
}