#include <dl/model/model.hpp>

#include <dl/tensor/math.hpp>

#include <numeric>
#include <ranges>

using dl::ModelBase;
using dl::Tensor;

void ModelBase::registerParameter(std::string name, Tensor& tensor) {
	tensor->setRequiresGrad(true);
	_parameters.insert({name, tensor});
}

size_t ModelBase::numParameters() const noexcept {
	using citer = std::pair<std::string, dl::TensorRef>;
	return std::accumulate(_parameters.cbegin(), _parameters.cend(), 0, [](size_t acc, const citer& pair) {
		return acc + dl::numEntries(pair.second.get());
	});
}
size_t ModelBase::numTrainableParams() const noexcept {
	using citer = std::pair<std::string, dl::TensorRef>;
	auto filtered = _parameters | std::views::transform([](const citer& pair) { return pair.second; }) |
					std::views::filter([](const dl::Tensor& tensor) { return tensor->requiresGrad(); }) |
					std::views::transform(dl::numEntries);
	return std::accumulate(std::ranges::begin(filtered), std::ranges::end(filtered), 0);
}