#include <dl/device.hpp>
#include <dl/tensor/tensor.hpp>

#include <xtensor-blas/xlinalg.hpp>
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
		explicit CPUDenseFloatTensor(const xt::xarray<float>& data, const Device& device, bool requiresGrad)
				: Tensor(device, requiresGrad), data(data) {}
		explicit CPUDenseFloatTensor(xt::xarray<float>&& data, const Device& device, bool requiresGrad)
				: Tensor(device, requiresGrad), data(data) {}

		virtual std::ostream& writeToStream(std::ostream& stream) const noexcept override { return stream << data; }

		virtual TensorPtr add(const TensorPtr& other) const noexcept override {
			return createResult(data + downcast(other).data, requiresGrad() || other->requiresGrad());
		}
		virtual TensorPtr sub(const TensorPtr& other) const noexcept override {
			return createResult(data - downcast(other).data, requiresGrad() || other->requiresGrad());
		}
		virtual TensorPtr mul(const TensorPtr& other) const noexcept override {
			return createResult(data * downcast(other).data, requiresGrad() || other->requiresGrad());
		}
		virtual TensorPtr div(const TensorPtr& other) const noexcept override {
			return createResult(data / downcast(other).data, requiresGrad() || other->requiresGrad());
		}

		virtual TensorPtr matmul(const TensorPtr& other) const noexcept override {
			return createResult(std::move(xt::linalg::dot(data, downcast(other).data)), requiresGrad());
		}

		virtual TensorPtr pow(float exponent) const noexcept override {
			return createResult(std::move(xt::pow(data, exponent)), requiresGrad());
		}

		virtual TensorPtr mean() const noexcept override {
			return createResult(std::move(xt::mean(data)), requiresGrad());
		}

		virtual void mul_inplace(const TensorPtr& other) noexcept override { data *= downcast(other).data; }

		virtual TensorPtr clone() const noexcept override { return {std::make_shared<CPUDenseFloatTensor>(*this)}; }

		virtual size_t shape(size_t dim) const noexcept { return data.shape(dim); }

		virtual Shape shape() const noexcept { return Shape(std::begin(data.shape()), std::end(data.shape())); }

		virtual TensorPtr flatten() const noexcept {
			return createResult(std::move(xt::flatten(data)), requiresGrad());
		}

		inline TensorPtr createResult(xt::xarray<float> data, bool requireGrad) const noexcept {
			return {std::make_shared<CPUDenseFloatTensor>(std::move(data), device(), requireGrad)};
		}
		inline static const CPUDenseFloatTensor& downcast(const TensorPtr& other) noexcept {
			return static_cast<const CPUDenseFloatTensor&>(*other);
		}
	};

	class CPUDevice final : public Device {
	private:
	public:
		CPUDevice() = default;

		virtual TensorPtr empty(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::empty<float>(shape);
			return {std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad)};
		}

		virtual TensorPtr zero(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::zeros<float>(shape);
			return {std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad)};
		}

		virtual TensorPtr ones(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::ones<float>(shape);
			return {std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad)};
		}

		virtual TensorPtr constant(int value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value; /** \todo: allow tensors of different datatypes **/
			return {std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad)};
		}
		virtual TensorPtr constant(float value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value; /** \todo: allow tensors of different datatypes **/
			return {std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad)};
		}
		virtual TensorPtr constant(double value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value; /** \todo: allow tensors of different datatypes **/
			return {std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad)};
		}
		virtual TensorPtr constant(std::initializer_list<float> value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value;
			return {std::make_shared<CPUDenseFloatTensor>(expr, *this, requiresGrad)};
		}
	};

} // namespace dl

// Register the device
static dl::CPUDevice cpuDevice;
dl::Device const& dl::Device::cpu = cpuDevice;
thread_local dl::Device const* dl::Device::defaultDevice = &cpuDevice;