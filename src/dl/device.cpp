#include <dl/device.hpp>

#include <dl/tensor/tensorimpl.hpp>

namespace dl {
	Tensor zeros_like(const Tensor& tensor, Device const& device) { return device.zeros(tensor->shape()); }
	Tensor ones_like(const Tensor& tensor, Device const& device) { return device.ones(tensor->shape()); }
	Tensor rand_like(const Tensor& tensor, Device const& device) { return device.rand(tensor->shape()); }

	Tensor clone(const Tensor& tensor) { return tensor->clone(); }
} // namespace dl