\page libIRTraining Training and Evaluating a Retrieval Model
\tableofcontents

The goal of this tutorial is to train and evaluate a ranking model (in our case "monoBERT" as described by \cite monobert) on the MS MARCO Passage \cite msmarco dataset. But keep in mind that everything discussed here can, of course, also be adapted to other models, other datasets, different training regimes (in-batch negatives, distillation, ...).

We start of by defining our ranking model.

```cpp
class MonoBERT : public dl::ModelBase {
private:
public:
    dl::Tensor operator()(ir::Query, ir::Document);
};
```

\note A ranking model maps a query and a document to a relevance label: \f(\mathcal{Q}\times \mathcal{D} \to \mathbb{R}\f). But, flexibility is king and if you want a different signature, have a look at adapters.

Ranking models are not different from any other ranking model. Thus, you can use any of the other facilities (dataset, optimizer, loss objectives, trainers, ...) to also train and evaluate your ranking model. But, for your convenience, and to reduce the amount of boilerplate such that you can quickly get started working on what is interesting, novel, and important, we provide some general implementations for you as well. We will discuss these in the following.

# Datasets
libir provides two dataset types:
1. `ir::PointwiseDataset`, where each instance is a tuple \f((q, d, r)\f) consisting of a query \f(q \in \mathcal{Q}\f), document \f(d \in \mathcal{D}\f), and relevance label \f(r \in \mathbb{R}\f). Such a dataset can be used for training the ranking model "pointwise", i.e., as a **relevance classifier**. But also admits pairwise training using some negative sampling strategy (e.g., in-batch negatives).
2. `ir::PairwiseDataset`, where each instance is a triple \f((q, d^+, d^-)\f) consisting of a query \f(q\in\mathcal{Q}\f), a positive (more relevant), and a negative (less relevant) document \f(d^+, d^-\in\mathcal{D}\f).

It may look like there is a mismatch here since we previously defined ranking models to solely be mappings \f(\mathcal{Q}\times \mathcal{D} \to \mathbb{R}\f) and the `ir::PairwiseDataset` has the signature \f(\mathcal{Q}\times \mathcal{D} \times \mathcal{D}\f) but herein lies the magic of separation of concerns: Choose the signature that makes sense for your
ranking model and use a trainer adapter to specify how it should be trained.

```cpp
auto dataset = ir::datasets::load<ir::PairwiseDataset>("msmarco-passage");
assert(dataset != nullptr);
```

# Training
```cpp
MonoBERT model;

auto conf = dl::TrainerConfBuilder<MonoBERT>()
        .setDataset<ir::PairwiseDataset>(std::move(dataset))
        .setOptimizer<dl::AdamW>()
        .addObserver<dl::LimitEpochs>(10)
        .addObserver<dl::EarlyStopping>(3)
        .addObserver<dl::ConsoleUI>()
        .build();
auto trainer = dl::Trainer(std::move(conf));
trainer.fit(model, pairwiseTrainer<MonoBERT>);
```

# Evaluation
```cpp
auto results = trainer.test(model, TrecEvaluator(dataset, {"MRR@10", "MAP", "nDCG@10"}), trecEvalAdapter<MonoBERT>);
for (auto&& [metric, score] : results)
    std::cout << metric << ": " << score << std::endl;
```
Output
```
MRR@10: XX.XX
MAP: XX.XX
nDCG@10: XX.XX
```