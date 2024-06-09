#include <dl/device.hpp>

#include <dl/tensor/tensorimpl.hpp>

namespace dl {
	TensorPtr zeros_like(const TensorPtr& tensor, Device const& device) { return device.zeros(tensor->shape()); }
	TensorPtr ones_like(const TensorPtr& tensor, Device const& device) { return device.ones(tensor->shape()); }
	TensorPtr rand_like(const TensorPtr& tensor, Device const& device) { return device.rand(tensor->shape()); }

	TensorPtr clone(const TensorPtr& tensor) { return tensor->clone(); }
} // namespace dl