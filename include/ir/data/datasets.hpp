#pragma once

#include "dataset.hpp"

#include <memory>

namespace ir::datasets {
	template <typename T>
	std::unique_ptr<T> load(std::string identifier) noexcept;

	template <>
	std::unique_ptr<ir::PointwiseDataset> load<>(std::string identifier) noexcept;

	template <>
	std::unique_ptr<ir::PairwiseDataset> load<>(std::string identifier) noexcept;
} // namespace ir::datasets