#pragma once

#include <memory>
#include <vector>

namespace dl {
    class Tensor;
    using TensorPtr = std::shared_ptr<dl::Tensor>;
	using Shape = std::vector<std::size_t>;
}