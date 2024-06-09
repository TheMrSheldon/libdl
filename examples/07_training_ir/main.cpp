#include <dl/learning/loss.hpp>
#include <dl/learning/optimizers/adam.hpp>
#include <dl/learning/trainer.hpp>
#include <dl/model/model.hpp>
#include <dl/utils/urlstream.hpp>
#include <ir/data/datasets.hpp>

#include <iostream>

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

dl::TensorPtr pairwiseTrainer(auto& model, ir::Query query, ir::Document pos, ir::Document neg) {
	return model(query, neg) - model(query, pos);
}

TrecEvaluator::GradeTriple trecEvalAdapter(auto& model, ir::Query query, ir::Document doc) {
	return std::make_tuple(query, doc, model(query, doc));
}

int main(int argc, char* argv[]) {
	MonoBERT model;
	auto dataset = ir::datasets::load<ir::PointwiseDataset>("msmarco-passage/train/judged");
	// auto dataset2 = ir::datasets::load<ir::PairwiseDataset>("msmarco-passage/train/judged");

	auto conf = dl::TrainerConfBuilder<MonoBERT>()
						.setDataset<ir::PointwiseDataset>(std::move(dataset))
						.setOptimizer<dl::optim::Adam>()
						.addObserver(dl::observers::limitEpochs(10))
						.addObserver(dl::observers::earlyStopping(3))
						.addObserver(dl::observers::consoleUI())
						.build();
	auto trainer = dl::Trainer(std::move(conf));
	trainer.fit(model, pairwiseTrainer<MonoBERT>);
	auto results = trainer.test(model, TrecEvaluator(dataset, {"MRR@10", "MAP", "nDCG@10"}), trecEvalAdapter<MonoBERT>);
	for (auto&& [metric, score] : results)
		std::cout << metric << ": " << score << std::endl;

	return 0;
}