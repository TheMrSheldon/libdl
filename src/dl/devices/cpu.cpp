#include <dl/device.hpp>
#include <dl/tensor/tensorimpl.hpp>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include <numeric>

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
		virtual bool operator==(const Tensor& other) const noexcept override { return data == downcast(other).data; }
		virtual bool allclose(const Tensor& other, float rtol = 1e-5, float atol = 1e-8) const noexcept override {
			return xt::allclose(data, downcast(other).data, rtol, atol);
		}

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
		virtual Tensor mean(size_t dim) const noexcept override {
			return createResult(xt::mean(data, {dim}), requiresGrad());
		}
		virtual Tensor sum() const noexcept override { return createResult(xt::sum(data), requiresGrad()); }
		virtual Tensor sum(size_t dim) const noexcept override {
			return createResult(xt::sum(data, {dim}), requiresGrad());
		}
		virtual Tensor min() const noexcept override { return createResult(xt::amin(data), requiresGrad()); }
		virtual Tensor min(size_t dim) const noexcept override {
			return createResult(xt::amin(data, {dim}), requiresGrad());
		}
		virtual Tensor max() const noexcept override { return createResult(xt::amax(data), requiresGrad()); }
		virtual Tensor max(size_t dim) const noexcept override {
			return createResult(xt::amax(data, {dim}), requiresGrad());
		}
		virtual Tensor max(const Tensor& other) const noexcept {
			return createResult(xt::maximum(data, downcast(other).data), requiresGrad());
		}
		virtual Tensor min(const Tensor& other) const noexcept {
			return createResult(xt::minimum(data, downcast(other).data), requiresGrad());
		}
		virtual Tensor var(DOF dof) const noexcept override {
			//auto result = xt::detail::mean_noaxis<void>(
			//		xt::square(data - xt::mean(data)), dof.dof, xt::evaluation_strategy::immediate
			//);
			//return createResult(std::move(result), requiresGrad());
			return createResult(xt::variance(data, dof.dof), requiresGrad());
		}
		virtual Tensor var(size_t dim, DOF dof) const noexcept override {
			return createResult(
					xt::detail::mean<void>(
							xt::square(data - xt::reshape_view(xt::mean(data, {dim}), {-1, 1})), {dim}, dof.dof,
							xt::evaluation_strategy::lazy
					),
					requiresGrad()
			);
			//return createResult(xt::variance(data, {dim}, dof.dof), requiresGrad());
		}

		virtual void mul_inplace(const Tensor& other) noexcept override { data *= downcast(other).data; }
		virtual void reshape(SShape shape) noexcept override { data.reshape(std::move(shape)); }

		virtual Tensor clone() const noexcept override { return Tensor::create<CPUDenseFloatTensor>(*this); }

		virtual size_t shape(size_t dim) const noexcept { return data.shape(dim); }

		virtual Shape shape() const noexcept { return Shape(std::begin(data.shape()), std::end(data.shape())); }

		virtual Tensor flatten() const noexcept { return createResult(std::move(xt::flatten(data)), requiresGrad()); }

		virtual size_t toBytes(char* buffer, size_t buflen) const noexcept override {
			size_t numentries =
					std::accumulate(data.shape().cbegin(), data.shape().cend(), 1, std::multiplies<size_t>{});
			size_t totalBytes = numentries * sizeof(float);
			if (buffer == nullptr) // No buffer provided
				return totalBytes;
			if (buflen < totalBytes) // Buffer is not big enough
				return 0;
			std::memcpy(buffer, data.data(), totalBytes);
			return totalBytes;
		}

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

		virtual Tensor zeros(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::zeros<float>(shape);
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual Tensor ones(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::ones<float>(shape);
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual Tensor rand(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::random::rand<float>(shape, 0, 1);
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
		virtual Tensor constant(InitializerTensor<float>&& value, bool requiresGrad) const noexcept override {
			void* data = std::malloc(value.data.size() * sizeof(float));
			data = std::memcpy(data, value.data.data(), value.data.size() * sizeof(float));
			xt::xarray<float> expr = xt::adapt((float*)data, value.data.size(), xt::acquire_ownership(), value.shape);
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual Tensor fromBytesFP32(const char* buffer, size_t bufsize, Shape shape) const noexcept override {
			auto numentries = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<size_t>{});
			assert(bufsize == sizeof(float) * numentries);
			void* data = std::malloc(bufsize);
			float* fptr = reinterpret_cast<float*>(data);
			data = std::memcpy(data, buffer, bufsize);
			xt::xarray<float> expr = xt::adapt((float*)data, numentries, xt::acquire_ownership(), shape);
			return Tensor::create<CPUDenseFloatTensor>(expr, *this, false);
		}
	};

} // namespace dl

// Register the device
static dl::CPUDevice cpuDevice;
dl::Device const& dl::Device::cpu = cpuDevice;
thread_local dl::Device const* dl::Device::defaultDevice = &cpuDevice;