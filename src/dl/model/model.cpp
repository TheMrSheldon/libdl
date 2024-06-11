#include <dl/model/model.hpp>

#include <dl/tensor/math.hpp>

#include <numeric>
#include <ranges>

using dl::ModelBase;
using dl::TensorPtr;

void ModelBase::registerParameter(std::string name, TensorPtr& tensor) {
	tensor->setRequiresGrad(true);
	_parameters.insert({name, tensor});
}

size_t ModelBase::numParameters() const noexcept {
	using citer = std::pair<std::string, dl::TensorPtr>;
	return std::accumulate(_parameters.cbegin(), _parameters.cend(), 0, [](size_t acc, const citer& pair) {
		return acc + dl::numEntries(pair.second);
	});
}
size_t ModelBase::numTrainableParams() const noexcept {
	using citer = std::pair<std::string, dl::TensorPtr>;
	auto filtered = _parameters | std::views::transform([](const citer& pair) { return pair.second; }) |
					std::views::filter([](const dl::TensorPtr& tensor) { return tensor->requiresGrad(); }) |
					std::views::transform(dl::numEntries);
	return std::accumulate(std::ranges::begin(filtered), std::ranges::end(filtered), 0);
}