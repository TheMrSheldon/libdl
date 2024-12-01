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
		virtual TensorPtr empty(Shape shape, bool requiresGrad = false) const noexcept = 0;
		/**
		 * @brief Creates a new tensor of the specified shape. All entries are initialized to zero.
		 * 
		 * @param shape The shape of the tensor.
		 * @param requiresGrad True if a gradient should be calculate for the newly constructed tensor.
		 * @return The newly created tensor.
		 */
		virtual TensorPtr zeros(Shape shape, bool requiresGrad = false) const noexcept = 0;
		/**
		 * @brief Creates a new tensor of the specified shape. All entries are initialized to one.
		 * 
		 * @param shape The shape of the tensor.
		 * @param requiresGrad True if a gradient should be calculate for the newly constructed tensor.
		 * @return The newly created tensor.
		 */
		virtual TensorPtr ones(Shape shape, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr rand(Shape shape, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr constant(int value, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr constant(float value, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr constant(double value, bool requiresGrad = false) const noexcept = 0;
		virtual TensorPtr constant(InitializerTensor<float>&& value, bool requiresGrad = false) const noexcept = 0;

		virtual TensorPtr arange(int32_t start, int32_t stop, int32_t step) const noexcept = 0;
		virtual TensorPtr linspace(int32_t start, int32_t stop, int32_t numsamples) const noexcept = 0;
		virtual TensorPtr logspace(int32_t start, int32_t stop, int32_t numsamples) const noexcept = 0;

		virtual TensorPtr fromBytesFP32(const char* buffer, size_t bufsize, Shape shape) const noexcept = 0;

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

	inline TensorPtr empty(const Shape& size, Device const& device = Device::getDefault()) {
		return device.empty(size);
	}
	inline TensorPtr zeros(const Shape& size, Device const& device = Device::getDefault()) {
		return device.zeros(size);
	}
	inline TensorPtr ones(const Shape& size, Device const& device = Device::getDefault()) { return device.ones(size); }
	inline TensorPtr rand(const Shape& size, Device const& device = Device::getDefault()) { return device.rand(size); }
	TensorPtr zeros_like(const TensorPtr& tensor, Device const& device = Device::getDefault());
	TensorPtr ones_like(const TensorPtr& tensor, Device const& device = Device::getDefault());
	TensorPtr rand_like(const TensorPtr& tensor, Device const& device = Device::getDefault());
	inline TensorPtr constant(int value, Device const& device = Device::getDefault()) { return device.constant(value); }
	inline TensorPtr constant(float value, Device const& device = Device::getDefault()) {
		return device.constant(value);
	}
	inline TensorPtr constant(double value, Device const& device = Device::getDefault()) {
		return device.constant(value);
	}
	inline TensorPtr constant(InitializerTensor<float>&& value, Device const& device = Device::getDefault()) {
		return device.constant(std::move(value));
	}

	inline TensorPtr
	arange(int32_t start, int32_t stop, int32_t step = 1, Device const& device = Device::getDefault()) {
		return device.arange(start, stop, step);
	}
	inline TensorPtr
	linspace(int32_t start, int32_t stop, int32_t numsamples, Device const& device = Device::getDefault()) {
		return device.linspace(start, stop, numsamples);
	}
	inline TensorPtr
	logspace(int32_t start, int32_t stop, int32_t numsamples, Device const& device = Device::getDefault()) {
		return device.logspace(start, stop, numsamples);
	}

	template <typename T>
	TensorPtr
	fromBytes(const char* buffer, size_t buflen, const Shape& shape, Device const& device = Device::getDefault());
	template <>
	inline TensorPtr fromBytes<float>(const char* buffer, size_t buflen, const Shape& shape, Device const& device) {
		return device.fromBytesFP32(buffer, buflen, shape);
	}

	TensorPtr clone(const TensorPtr& tensor);
} // namespace dl
