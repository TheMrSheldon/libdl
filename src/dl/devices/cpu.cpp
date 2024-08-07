#include <dl/device.hpp>
#include <dl/tensor/tensorimpl.hpp>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include <numeric>

namespace tmp {
	// https://github.com/xtensor-stack/xtensor/issues/1375
	/** \todo reimplement myself **/
	template <class _Tp>
	xt::xarray<_Tp> matmul(xt::xarray<_Tp> a, xt::xarray<_Tp> b) {
		using Arr = xt::xarray<_Tp>;

		if (a.dimension() == 1 && b.dimension() == 1) {
			return xt::linalg::outer(a, b);
		} else if (a.dimension() <= 2 && b.dimension() <= 2) {
			return xt::linalg::dot(a, b);
		} else {
			if (a.dimension() == b.dimension()) {
				assert(a.shape()[0] == b.shape()[0]);
				size_t layers = a.shape()[0];

				Arr tmp;
				{
					Arr a0 = xt::view(a, 0);
					Arr b0 = xt::view(b, 0);
					tmp = matmul(std::move(a0), std::move(b0));
				}

				auto out_shape = tmp.shape();
				out_shape.insert(out_shape.begin(), layers);

				auto result = Arr::from_shape(out_shape);
				xt::view(result, 0) = tmp;

				for (size_t i = 1; i < layers; i++) {
					Arr ai = xt::view(a, i);
					Arr bi = xt::view(b, i);
					xt::view(result, i) = matmul(std::move(ai), std::move(bi));
				}

				return result;
			} else if (a.dimension() > b.dimension()) {
				size_t layers = a.shape()[0];

				Arr tmp;
				{
					Arr a0 = xt::view(a, 0);
					tmp = matmul(std::move(a0), b);
				}

				auto out_shape = tmp.shape();
				out_shape.insert(out_shape.begin(), layers);

				auto result = Arr::from_shape(out_shape);
				xt::view(result, 0) = std::move(tmp);

				for (size_t i = 1; i < layers; i++) {
					Arr ai = xt::view(a, i);
					xt::view(result, i) = matmul(std::move(ai), b);
				}

				return result;
			} else {
				assert(a.dimension() < b.dimension());
				size_t layers = b.shape().back();

				Arr tmp;
				{
					Arr b0 = xt::strided_view(b, {xt::ellipsis(), 0});
					tmp = matmul(a, std::move(b0));
				}

				auto out_shape = tmp.shape();
				out_shape.insert(out_shape.end(), layers);

				auto result = Arr::from_shape(out_shape);
				xt::strided_view(result, {xt::ellipsis(), 0}) = std::move(tmp);

				for (size_t i = 1; i < layers; i++) {
					Arr bi = xt::strided_view(b, {xt::ellipsis(), i});
					xt::strided_view(result, {xt::ellipsis(), i}) = matmul(a, std::move(bi));
				}

				return result;
			}
		}
	}
} // namespace tmp

static dl::Shape removeDim(dl::Shape shape, int dim) {
	// shape.erase(std::next(shape.begin(), dim));
	shape[(dim >= 0) ? dim : (shape.size() + dim)] = 1;
	return shape;
}

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
		virtual bool operator==(const TensorPtr& other) const noexcept override { return data == downcast(other).data; }
		virtual bool allclose(const TensorPtr& other, float rtol = 1e-5, float atol = 1e-8) const noexcept override {
			return xt::allclose(data, downcast(other).data, rtol, atol);
		}

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

		/**
		 * @brief Performs "fused multiply and add".
		 * @details Multiplies this tensor by \p factor and adds \p summand to the result. This specialized function
		 * exists since some devices (e.g., CUDA and some SIMD instruction sets) provide such a function.
		 * 
		 * @param factor the factor to multiply with this tensor.
		 * @param summand the summand to add to the product of this with \p factor.
		 * @return the result.
		 */
		virtual TensorPtr fma(const TensorPtr& factor, const TensorPtr& summand) const noexcept override {
			return createResult(
					xt::fma(data, downcast(factor).data, downcast(summand).data),
					requiresGrad() || factor->requiresGrad() || summand->requiresGrad()
			);
		}
		virtual TensorPtr matmul(const TensorPtr& other) const noexcept override {
			// return createResult(xt::linalg::dot(data, downcast(other).data), requiresGrad());
			return createResult(tmp::matmul(data, downcast(other).data), requiresGrad() || other->requiresGrad());
		}

		virtual TensorPtr transpose(std::vector<size_t>&& perm) const noexcept {
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

		virtual TensorPtr pow(float exponent) const noexcept override {
			return createResult(xt::pow(data, exponent), requiresGrad());
		}

		virtual TensorPtr exp() const noexcept override { return createResult(xt::exp(data), requiresGrad()); }
		virtual TensorPtr log() const noexcept override { return createResult(xt::log(data), requiresGrad()); }
		virtual TensorPtr sqrt() const noexcept override { return createResult(xt::sqrt(data), requiresGrad()); }
		virtual TensorPtr rsqrt() const noexcept override { return createResult(1 / xt::sqrt(data), requiresGrad()); }

		virtual TensorPtr mean() const noexcept override { return createResult(xt::mean(data), requiresGrad()); }
		virtual TensorPtr mean(int dim, bool keepdim) const noexcept override {
			/** \todo implement keepdim **/
			return createResult(xt::mean(data, {dim}), requiresGrad());
		}
		virtual TensorPtr sum() const noexcept override { return createResult(xt::sum(data), requiresGrad()); }
		virtual TensorPtr sum(int dim, bool keepdim) const noexcept override {
			xt::xarray<float> result = xt::sum(data, {dim});
			if (keepdim) {
				auto shape = removeDim(this->shape(), dim);
				result = xt::reshape_view(result, shape);
			}
			return createResult(result, requiresGrad());
		}
		virtual TensorPtr min() const noexcept override { return createResult(xt::amin(data), requiresGrad()); }
		virtual TensorPtr min(int dim, bool keepdim) const noexcept override {
			/** \todo implement keepdim **/
			return createResult(xt::amin(data, {dim}), requiresGrad());
		}
		virtual TensorPtr max() const noexcept override { return createResult(xt::amax(data), requiresGrad()); }
		virtual TensorPtr max(int dim, bool keepdim) const noexcept override {
			xt::xarray<float> result = xt::amax(data, {dim});
			if (keepdim) {
				auto shape = removeDim(this->shape(), dim);
				result = xt::reshape_view(result, shape);
			}
			return createResult(result, requiresGrad());
		}
		virtual TensorPtr max(const TensorPtr& other) const noexcept {
			return createResult(xt::maximum(data, downcast(other).data), requiresGrad());
		}
		virtual TensorPtr min(const TensorPtr& other) const noexcept {
			return createResult(xt::minimum(data, downcast(other).data), requiresGrad());
		}
		virtual TensorPtr var(DOF dof) const noexcept override {
			//auto result = xt::detail::mean_noaxis<void>(
			//		xt::square(data - xt::mean(data)), dof.dof, xt::evaluation_strategy::immediate
			//);
			//return createResult(std::move(result), requiresGrad());
			return createResult(xt::variance(data, dof.dof), requiresGrad());
		}
		virtual TensorPtr var(int dim, DOF dof) const noexcept override {
			return createResult(
					xt::detail::mean<void>(
							xt::square(data - xt::reshape_view(xt::mean(data, {dim}), {-1, 1})), {dim}, dof.dof,
							xt::evaluation_strategy::lazy
					),
					requiresGrad()
			);
			//return createResult(xt::variance(data, {dim}, dof.dof), requiresGrad());
		}

		virtual TensorPtr erf() const noexcept override { return createResult(xt::erf(data), requiresGrad()); }

		virtual void mul_inplace(const TensorPtr& other) noexcept override { data *= downcast(other).data; }
		virtual void reshape(SShape shape) noexcept override { data.reshape(std::move(shape)); }

		virtual TensorPtr clone() const noexcept override { return TensorPtr::create<CPUDenseFloatTensor>(*this); }

		virtual size_t shape(int dim) const noexcept { return data.shape(dim); }

		virtual Shape shape() const noexcept { return Shape(std::begin(data.shape()), std::end(data.shape())); }

		virtual TensorPtr flatten() const noexcept {
			return createResult(std::move(xt::flatten(data)), requiresGrad());
		}

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

		inline TensorPtr createResult(xt::xarray<float> data, bool requireGrad) const noexcept {
			return TensorPtr::create<CPUDenseFloatTensor>(std::move(data), device(), requireGrad);
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
			return TensorPtr::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual TensorPtr zeros(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::zeros<float>(shape);
			return TensorPtr::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual TensorPtr ones(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::ones<float>(shape);
			return TensorPtr::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual TensorPtr rand(Shape shape, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = xt::random::rand<float>(shape, 0, 1);
			return TensorPtr::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual TensorPtr constant(int value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value; /** \todo: allow tensors of different datatypes **/
			return TensorPtr::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}
		virtual TensorPtr constant(float value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value; /** \todo: allow tensors of different datatypes **/
			return TensorPtr::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}
		virtual TensorPtr constant(double value, bool requiresGrad) const noexcept override {
			xt::xarray<float> expr = value; /** \todo: allow tensors of different datatypes **/
			return TensorPtr::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}
		virtual TensorPtr constant(InitializerTensor<float>&& value, bool requiresGrad) const noexcept override {
			void* data = std::malloc(value.data.size() * sizeof(float));
			data = std::memcpy(data, value.data.data(), value.data.size() * sizeof(float));
			xt::xarray<float> expr = xt::adapt((float*)data, value.data.size(), xt::acquire_ownership(), value.shape);
			return TensorPtr::create<CPUDenseFloatTensor>(expr, *this, requiresGrad);
		}

		virtual TensorPtr fromBytesFP32(const char* buffer, size_t bufsize, Shape shape) const noexcept override {
			auto numentries = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<size_t>{});
			assert(bufsize == sizeof(float) * numentries);
			void* data = std::malloc(bufsize);
			float* fptr = reinterpret_cast<float*>(data);
			data = std::memcpy(data, buffer, bufsize);
			xt::xarray<float> expr = xt::adapt((float*)data, numentries, xt::acquire_ownership(), shape);
			return TensorPtr::create<CPUDenseFloatTensor>(expr, *this, false);
		}
	};

} // namespace dl

// Register the device
static dl::CPUDevice cpuDevice;
dl::Device const& dl::Device::cpu = cpuDevice;
thread_local dl::Device const* dl::Device::defaultDevice = &cpuDevice;