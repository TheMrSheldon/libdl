#include <dl/device.hpp>
#include <dl/tensor/tensor.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xio.hpp>

namespace dl {
	class CPUDenseFloatTensor final : public Tensor {
	private:
		xt::xarray<float> data;

	public:
		CPUDenseFloatTensor(const CPUDenseFloatTensor& other)
				: CPUDenseFloatTensor(other.data, other.device(), other.requiresGrad()) {}
		CPUDenseFloatTensor(CPUDenseFloatTensor&& other)
				: CPUDenseFloatTensor(other.data, other.device(), other.requiresGrad()) {}
		explicit CPUDenseFloatTensor(xt::xarray<float> data, const Device& device, bool requiresGrad)
				: Tensor(device, requiresGrad), data(data) {}
		explicit CPUDenseFloatTensor(xt::xarray<float>&& data, const Device& device, bool requiresGrad)
				: Tensor(device, requiresGrad), data(data) {}

		virtual std::ostream& writeToStream(std::ostream& stream) const noexcept override { return stream << data; }
	};

	class CPUDevice final : public Device {
	private:
	public:
		CPUDevice() = default;

		virtual TensorPtr empty(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::empty<float>(shape);
			return std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual TensorPtr zero(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::zeros<float>(shape);
			return std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}
		
		virtual TensorPtr ones(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::ones<float>(shape);
			return std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}
	};

} // namespace dl

// Register the device
static dl::CPUDevice cpuDevice;
dl::Device const& dl::Device::cpu = cpuDevice;
thread_local dl::Device const* dl::Device::defaultDevice = &cpuDevice;