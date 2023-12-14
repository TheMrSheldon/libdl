#pragma once

#include "../device.hpp"

namespace dl::cpu {

	class CPUDevice final : dl::Device {
	private:

	public:
		virtual TensorPtr emptyTensor(Shape shape) noexcept override;
	};

	class CPUDenseTensor final : dl::Tensor {
	private:

	public:
		
	};

}