#include <dl/device.hpp>

#include <dl/tensor/tensor.hpp>

namespace dl {
	TensorPtr ones_like(const TensorPtr tensor, Device const& device) { return device.ones(tensor->shape()); }

	TensorPtr clone(const TensorPtr tensor) { return tensor->clone(); }
} // namespace dl