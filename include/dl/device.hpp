#pragma once

#include "tensor/shape.hpp"
#include "tensor/tensorimpl.hpp"
#include "utils/scopeguard.hpp"

namespace dl {
	class Device {
	private:
		/**
		 * @brief The default device for this thread.
		 * 
		 * @see Device::getDefault()
		 * @see Device::changeDefaultDevice(Device&)
		 */
		thread_local static Device const* defaultDevice;

		Device(const Device& other) = delete;
		Device(Device&& other) = delete;
		Device& operator=(const Device& other) = delete;
		Device& operator=(Device&& other) = delete;

	protected:
		Device() = default;

	public:
		virtual ~Device() = default;

		/**
		 * @brief The CPU-device implementation.
		 */
		static Device const& cpu;

		/**
		 * @brief Creates a new tensor of the specified shape without initializing the memory.
		 * 
		 * @param shape The shape of the tensor.
		 * @param requiresGrad True if a gradient should be calculate for the newly constructed tensor.
		 * @return The newly created tensor.
		 */
		virtual Tensor empty(Shape shape, bool requiresGrad = false) const noexcept = 0;
		/**
		 * @brief Creates a new tensor of the specified shape. All entries are initialized to zero.
		 * 
		 * @param shape The shape of the tensor.
		 * @param requiresGrad True if a gradient should be calculate for the newly constructed tensor.
		 * @return The newly created tensor.
		 */
		virtual Tensor zero(Shape shape, bool requiresGrad = false) const noexcept = 0;
		/**
		 * @brief Creates a new tensor of the specified shape. All entries are initialized to one.
		 * 
		 * @param shape The shape of the tensor.
		 * @param requiresGrad True if a gradient should be calculate for the newly constructed tensor.
		 * @return The newly created tensor.
		 */
		virtual Tensor ones(Shape shape, bool requiresGrad = false) const noexcept = 0;
		virtual Tensor constant(int value, bool requiresGrad = false) const noexcept = 0;
		virtual Tensor constant(float value, bool requiresGrad = false) const noexcept = 0;
		virtual Tensor constant(double value, bool requiresGrad = false) const noexcept = 0;
		virtual Tensor constant(std::initializer_list<float> value, bool requiresGrad = false) const noexcept = 0;

		template <typename T>
		void setDefaultFloatTensorType();

		/**
		 * @brief Returns the default device for this thread.
		 * @details The default device is thread local. That means that modifying the default device from another thread
		 * will not effect other threads.
		 * 
		 * @return The current default device.
		 * @see Device::changeDefaultDevice(Device&)
		 */
		static const Device& getDefault() noexcept { return *defaultDevice; }

		/**
		 * @brief Temporarily modifies the default device.
		 * @details
		 * Example call:
		 * ```{cpp}
		 * {Device::changeDefaultDevice(mydevice);
		 *     // mydevice is default.
		 * }
		 * // mydevice is not default anymore.
		 * ```
		 * 
		 * @param device The device that should be default for the current scope.
		 * @return A scopeguard for the default device.
		 * @see Device::getDefault()
		 */
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

	inline Tensor empty(const Shape& size, Device const& device = Device::getDefault()) { return device.empty(size); }
	inline Tensor zero(const Shape& size, Device const& device = Device::getDefault()) { return device.zero(size); }
	inline Tensor ones(const Shape& size, Device const& device = Device::getDefault()) { return device.ones(size); }
	Tensor ones_like(const Tensor& tensor, Device const& device = Device::getDefault());
	inline Tensor constant(int value, Device const& device = Device::getDefault()) { return device.constant(value); }
	inline Tensor constant(float value, Device const& device = Device::getDefault()) { return device.constant(value); }
	inline Tensor constant(double value, Device const& device = Device::getDefault()) { return device.constant(value); }
	inline Tensor constant(std::initializer_list<float> value, Device const& device = Device::getDefault()) {
		return device.constant(std::move(value));
	}

	Tensor clone(const Tensor& tensor);
} // namespace dl
