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

		/**
		 * @brief Performs "fused multiply and add".
		 * @details Multiplies this tensor by \p factor and adds \p summand to the result. This specialized function
		 * exists since some devices (e.g., CUDA and some SIMD instruction sets) provide such a function.
		 * 
		 * @param factor the factor to multiply with this tensor.
		 * @param summand the summand to add to the product of this with \p factor.
		 * @return the result.
		 */
		virtual Tensor fma(const Tensor& factor, const Tensor& summand) const noexcept override {
			return createResult(
					xt::fma(data, downcast(factor).data, downcast(summand).data),
					requiresGrad() || factor->requiresGrad() || summand->requiresGrad()
			);
		}
		virtual Tensor matmul(const Tensor& other) const noexcept override {
			return createResult(xt::linalg::dot(data, downcast(other).data), requiresGrad());
		}

		virtual Tensor transpose(std::vector<size_t>&& perm) const noexcept {
			// libdl expects the permutation as a cycle (e.g., {0, 1, 3}) but xtensor can get multiple cycles for the
			// permutation. We convert this here such that the example cycle would be the permutation:
			// {1, 3, 2, 0}
			std::vector<size_t> p(numDim(), 0);
			for (size_t i = 0; i < p.size(); ++i)
				p[i] = i;
			auto prev = perm.back();
			for (auto& i : perm) {
				p[prev] = i;
				prev = i;
			}
			return createResult(xt::transpose(data, p), requiresGrad());
		}

		virtual Tensor pow(float exponent) const noexcept override {
			return createResult(xt::pow(data, exponent), requiresGrad());
		}

		virtual Tensor exp() const noexcept override { return createResult(xt::exp(data), requiresGrad()); }
		virtual Tensor sqrt() const noexcept override { return createResult(xt::sqrt(data), requiresGrad()); }
		virtual Tensor rsqrt() const noexcept override { return createResult(1 / xt::sqrt(data), requiresGrad()); }

		virtual Tensor mean() const noexcept override { return createResult(xt::mean(data), requiresGrad()); }
		virtual Tensor sum() const noexcept override { return createResult(xt::sum(data), requiresGrad()); }
		virtual Tensor max() const noexcept override { return createResult(xt::amax(data), requiresGrad()); }
		virtual Tensor min() const noexcept override { return createResult(xt::amin(data), requiresGrad()); }
		virtual Tensor max(const Tensor& other) const noexcept {
			return createResult(xt::maximum(data, downcast(other).data), requiresGrad());
		}
		virtual Tensor min(const Tensor& other) const noexcept {
			return createResult(xt::minimum(data, downcast(other).data), requiresGrad());
		}
		virtual Tensor var() const noexcept override { return createResult(xt::variance(data), requiresGrad()); }

		virtual void mul_inplace(const Tensor& other) noexcept override { data *= downcast(other).data; }
		virtual void reshape(std::vector<int> shape) noexcept override { data.reshape(std::move(shape)); }

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