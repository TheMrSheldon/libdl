#include <ir/data/datasets.hpp>

#include <dl/logging.hpp>

#include <iostream>

using namespace ir;

static auto logger = dl::logging::getLogger("datasets");

template <>
std::unique_ptr<PointwiseDataset> ir::datasets::load<>(std::string identifier) noexcept {
	logger->info("Loading Pointwise Dataset {}", identifier);
	return nullptr;
}

template <>
std::unique_ptr<PairwiseDataset> ir::datasets::load<>(std::string identifier) noexcept {
	logger->info("Loading Pairwise Dataset {}", identifier);
	return nullptr;
}