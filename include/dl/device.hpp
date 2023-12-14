#pragma once

#include "./tensor/tensor.hpp"

namespace dl {
	class Device {
	private:
		Device(const Device& other) = delete;
		Device(Device&& other) = delete;
		Device& operator=(const Device& other) = delete;
		Device& operator=(Device&& other) = delete;

	public:
		virtual TensorPtr emptyTensor(Shape shape) noexcept;

		template <typename T>
		void setDefaultFloatTensorType();
		template <>
		virtual void setDefaultFloatTensorType<float>();
		template <>
		virtual void setDefaultFloatTensorType<double>();

		static void setDefault(Device& device) noexcept;
		static Device& getDefault() noexcept;
	};
} // namespace dl