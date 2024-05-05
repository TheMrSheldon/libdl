#include <dl/device.hpp>
#include <dl/tensor/tensorimpl.hpp>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xio.hpp>

namespace dl {
	class CPUDenseFloatTensor final : public TensorImpl {
	private:
		xt::xarray<float> data;

	public:
		CPUDenseFloatTensor(const CPUDenseFloatTensor& other)
				: CPUDenseFloatTensor(other.data, other.device(), other.requiresGrad()) {}
		CPUDenseFloatTensor(CPUDenseFloatTensor&& other)
				: CPUDenseFloatTensor(other.data, other.device(), other.requiresGrad()) {}
		explicit CPUDenseFloatTensor(const xt::xarray<float>& data, const Device& device, bool requiresGrad)
				: TensorImpl(device, requiresGrad), data(data) {}
		explicit CPUDenseFloatTensor(xt::xarray<float>&& data, const Device& device, bool requiresGrad)
				: TensorImpl(device, requiresGrad), data(data) {}

		virtual std::ostream& writeToStream(std::ostream& stream) const noexcept override { return stream << data; }

		virtual Tensor add(const Tensor& other) const noexcept override {
			return createResult(data + downcast(other).data, requiresGrad() || other->requiresGrad());
		}
		virtual Tensor sub(const Tensor& other) const noexcept override {
			return createResult(data - downcast(other).data, requiresGrad() || other->requiresGrad());
		}
		virtual Tensor mul(const Tensor& other) const noexcept override {
			return createResult(data * downcast(other).data, requiresGrad() || other->requiresGrad());
		}
		virtual Tensor div(const Tensor& other) const noexcept override {
			return createResult(data / downcast(other).data, requiresGrad() || other->requiresGrad());
		}

		virtual Tensor matmul(const Tensor& other) const noexcept override {
			return createResult(std::move(xt::linalg::dot(data, downcast(other).data)), requiresGrad());
		}

		virtual Tensor pow(float exponent) const noexcept override {
			return createResult(std::move(xt::pow(data, exponent)), requiresGrad());
		}

		virtual Tensor mean() const noexcept override {
			return createResult(std::move(xt::mean(data)), requiresGrad());
		}

		virtual void mul_inplace(const Tensor& other) noexcept override { data *= downcast(other).data; }

		virtual Tensor clone() const noexcept override { return Tensor::create<CPUDenseFloatTensor>(*this); }

		virtual size_t shape(size_t dim) const noexcept { return data.shape(dim); }

		virtual Shape shape() const noexcept { return Shape(std::begin(data.shape()), std::end(data.shape())); }

		virtual Tensor flatten() const noexcept { return createResult(std::move(xt::flatten(data)), requiresGrad()); }

		inline Tensor createResult(xt::xarray<float> data, bool requireGrad) const noexcept {
			return Tensor::create<CPUDenseFloatTensor>(std::move(data), device(), requireGrad);
		}
		inline static const CPUDenseFloatTensor& downcast(const Tensor& other) noexcept {
			return static_cast<const CPUDenseFloatTensor&>(*other);
		}
	};

	class CPUDevice final : public Device {
	private:
	public:
		CPUDevice() = default;

		virtual Tensor empty(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::empty<float>(shape);
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual Tensor zero(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::zeros<float>(shape);
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual Tensor ones(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::ones<float>(shape);
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual Tensor constant(int value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value; /** \todo: allow tensors of different datatypes **/
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}
		virtual Tensor constant(float value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value; /** \todo: allow tensors of different datatypes **/
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}
		virtual Tensor constant(double value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value; /** \todo: allow tensors of different datatypes **/
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}
		virtual Tensor constant(std::initializer_list<float> value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value;
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}
	};

} // namespace dl

// Register the device
static dl::CPUDevice cpuDevice;
dl::Device const& dl::Device::cpu = cpuDevice;
thread_local dl::Device const* dl::Device::defaultDevice = &cpuDevice;