#pragma once

#include "tensor/tensorptr.hpp"
#include "utils/scopeguard.hpp"

namespace dl {
	class Device {
	private:
		thread_local static Device const* defaultDevice;

		Device(const Device& other) = delete;
		Device(Device&& other) = delete;
		Device& operator=(const Device& other) = delete;
		Device& operator=(Device&& other) = delete;

	protected:
		Device() = default;

	public:
		static Device const& cpu;

		virtual TensorPtr empty(Shape shape, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr zero(Shape shape, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr ones(Shape shape, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr constant(int value, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr constant(float value, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr constant(double value, bool requiresGrad = false) const noexcept = 0;

		template <typename T>
		void setDefaultFloatTensorType();

		static const Device& getDefault() noexcept { return *defaultDevice; }

		static auto changeDefaultDevice(Device& device) {
			auto oldDefault = defaultDevice;
			defaultDevice = &device;
			return dl::utils::ScopeGuard([oldDefault] { Device::defaultDevice = oldDefault; });
		}
	};

	template <>
	void Device::setDefaultFloatTensorType<float>();
	template <>
	void Device::setDefaultFloatTensorType<double>();

	inline TensorPtr empty(const Shape& size, Device const& device = Device::getDefault()) {
		return device.empty(size);
	}
	inline TensorPtr zero(const Shape& size, Device const& device = Device::getDefault()) { return device.zero(size); }
	inline TensorPtr ones(const Shape& size, Device const& device = Device::getDefault()) { return device.ones(size); }
	inline TensorPtr constant(int value, Device const& device = Device::getDefault()) { return device.constant(value); }
	inline TensorPtr constant(float value, Device const& device = Device::getDefault()) {
		return device.constant(value);
	}
	inline TensorPtr constant(double value, Device const& device = Device::getDefault()) {
		return device.constant(value);
	}
} // namespace dl
