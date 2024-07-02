#include <dl/tensor/math.hpp>

#include <dl/device.hpp>
#include <dl/tensor/tensorimpl.hpp>

#include <functional>
#include <numeric>
#include <ranges>

using namespace dl;

TensorPtr dl::pow(TensorPtr base, float exponent) noexcept {
	auto result = base->pow(exponent);
	if (base->requiresGrad()) {
		result->gradfn = [base = std::move(base), exponent](TensorPtr& ptr) mutable {
			if (base->grad == nullptr)
				base->grad = ptr * exponent * dl::pow(base, exponent - 1);
			else
				base->grad = base->grad + (ptr * exponent * dl::pow(base, exponent - 1));
			if (base->gradfn)
				base->gradfn(base->grad);
			else
				assert(base->requiresGrad());
		};
	}
	return result;
}

TensorPtr dl::exp(TensorPtr base) noexcept {
	/** \todo implement autodiff **/
	return base->exp();
}

TensorPtr dl::log(TensorPtr base) noexcept {
	/** \todo implement autodiff **/
	return base->log();
}

TensorPtr dl::sqrt(TensorPtr x) noexcept {
	/** \todo implement autodiff **/
	return x->sqrt();
}

TensorPtr dl::rsqrt(TensorPtr x) noexcept {
	/** \todo implement autodiff **/
	return x->rsqrt();
}

TensorPtr dl::mean(TensorPtr x) noexcept {
	if (x->numDim() == 0) // Mean of a scalar is the scalar itself
		return x;
	auto tensor = x->mean();
	if (tensor->requiresGrad()) {
		auto size = (float)x->shape(0);
		tensor->gradfn = [x = std::move(x), size](TensorPtr& ptr) mutable {
			if (x->grad == nullptr)
				x->grad = (ptr * dl::ones_like(x) / size);
			else
				x->grad = (x->grad + (ptr * dl::ones_like(x) / size));
			if (x->gradfn)
				x->gradfn(x->grad);
			else
				assert(x->requiresGrad());
		};
	}
	return tensor;
}

TensorPtr dl::mean(TensorPtr x, int dim, bool keepdim) noexcept {
	/** \todo implement autodiff **/
	return x->mean(dim, keepdim);
}

TensorPtr dl::sum(TensorPtr x) noexcept {
	/** \todo implement autodiff **/
	return x->sum();
}

TensorPtr dl::sum(TensorPtr x, int dim, bool keepdim) noexcept {
	/** \todo implement autodiff **/
	return x->sum(dim, keepdim);
}

TensorPtr dl::min(TensorPtr x) noexcept { return x->min(); }
TensorPtr dl::min(TensorPtr x, int dim, bool keepdim) noexcept { return x->min(dim, keepdim); }

TensorPtr dl::max(TensorPtr x) noexcept { return x->max(); }
TensorPtr dl::max(TensorPtr x, int dim, bool keepdim) noexcept { return x->max(dim, keepdim); }
TensorPtr dl::max(TensorPtr x, TensorPtr y) noexcept { return x->max(y); }

TensorPtr dl::var(TensorPtr x, DOF dof) noexcept { return x->var(dof); }

TensorPtr dl::var(TensorPtr x, int dim, DOF dof) noexcept { return x->var(dim, dof); }

TensorPtr dl::erf(TensorPtr x) noexcept {
	/** \todo implement gradient **/
	return x->erf();
}

TensorPtr dl::relu(TensorPtr x) noexcept { return dl::max(x, dl::zeros_like(x)); }

TensorPtr dl::softmax(TensorPtr x) noexcept {
	const auto power = dl::exp(x - dl::max(x));
	return power / dl::sum(power);
}
TensorPtr dl::softmax(TensorPtr x, int dim) noexcept {
	const auto power = dl::exp(x - dl::max(x, dim, true));
	return power / dl::sum(power, dim, true);
}

dl::TensorPtr dl::operator+(dl::TensorPtr left, float right) noexcept {
	return left + dl::constant(right, left->device());
}
dl::TensorPtr dl::operator-(dl::TensorPtr left, float right) noexcept {
	return left - dl::constant(right, left->device());
}
dl::TensorPtr dl::operator*(dl::TensorPtr left, float right) noexcept {
	return left * dl::constant(right, left->device());
}
dl::TensorPtr dl::operator/(dl::TensorPtr left, float right) noexcept {
	return left / dl::constant(right, left->device());
}
dl::TensorPtr dl::operator+(float left, dl::TensorPtr right) noexcept {
	return dl::constant(left, right->device()) + right;
}
dl::TensorPtr dl::operator-(float left, dl::TensorPtr right) noexcept {
	return dl::constant(left, right->device()) - right;
}
dl::TensorPtr dl::operator*(float left, dl::TensorPtr right) noexcept {
	return dl::constant(left, right->device()) * right;
}
dl::TensorPtr dl::operator/(float left, dl::TensorPtr right) noexcept {
	return dl::constant(left, right->device()) / right;
}

TensorPtr dl::operator+(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	auto tensor = left->add(right);
	if (tensor->requiresGrad()) {
		tensor->gradfn = [left = std::move(left), right = std::move(right)](TensorPtr& ptr) mutable {
			// Left Gradient
			if (left->grad == nullptr)
				left->grad = dl::ones_like(left) * ptr;
			else
				left->grad = (left->grad + ptr);
			if (left->gradfn)
				left->gradfn(left->grad);
			// Right Gradient
			if (right->grad == nullptr)
				right->grad = dl::ones_like(left) * ptr;
			else
				right->grad = (right->grad + ptr);
			if (right->gradfn)
				right->gradfn(right->grad);
		};
	}
	return tensor;
}
TensorPtr dl::operator-(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	auto tensor = left->sub(right);
	if (tensor->requiresGrad()) {
		tensor->gradfn = [left = std::move(left), right = std::move(right)](TensorPtr& ptr) mutable {
			// Left Gradient
			if (left->requiresGrad()) {
				left->grad = (left->grad == nullptr) ? ptr : (left->grad + ptr);
				if (left->gradfn)
					left->gradfn(left->grad);
			}
			// Right Gradient
			if (right->requiresGrad()) {
				right->grad = (right->grad == nullptr) ? (-1.0f * ptr) : (right->grad - ptr);
				if (right->gradfn)
					right->gradfn(right->grad);
			}
		};
	}
	return tensor;
}
TensorPtr dl::operator*(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	auto tensor = left->mul(right);
	if (tensor->requiresGrad()) {
		tensor->gradfn = [left = std::move(left), right = std::move(right)](TensorPtr& ptr) mutable {
			if (left->requiresGrad()) {
				auto lgrad = right->clone();
				lgrad->discardGradient();
				left->grad = (left->grad == nullptr) ? (lgrad * ptr) : (left->grad + (lgrad * ptr));
				if (left->gradfn)
					left->gradfn(left->grad);
			}
			if (right->requiresGrad()) {
				auto rgrad = left->clone();
				rgrad->discardGradient();
				right->grad = (right->grad == nullptr) ? (rgrad * ptr) : (right->grad + (rgrad * ptr));
				if (right->gradfn)
					right->gradfn(right->grad);
			}
		};
	}
	return tensor;
}
TensorPtr dl::operator/(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	return left->div(right);
}

TensorPtr dl::fma(const TensorPtr& factor1, const TensorPtr& factor2, const TensorPtr& summand) noexcept {
	return factor1->fma(factor2, summand);
}

TensorPtr dl::matmul(TensorPtr left, TensorPtr right) noexcept {
	/** \todo add support for autograd **/
	auto tensor = left->matmul(right);
	if (tensor->requiresGrad()) {
		tensor->gradfn = [left = std::move(left), right = std::move(right)](TensorPtr& ptr) mutable {
			//throw std::runtime_error("Not yet implemented");
			if (left->requiresGrad()) {
				auto lgrad = (right->numDim() >= 2) ? dl::transpose(right, {-1, -2}) : dl::clone(right);
				lgrad->discardGradient();
				left->grad = (left->grad == nullptr) ? dl::matmul(ptr, lgrad) : (left->grad + dl::matmul(ptr, lgrad));
				if (left->gradfn)
					left->gradfn(left->grad);
			}
			if (right->requiresGrad()) {
				auto rgrad = (left->numDim() >= 2) ? dl::transpose(left, {-1, -2}) : dl::clone(left);
				rgrad->discardGradient();
				right->grad = (right->grad == nullptr)
									  ? dl::transpose(dl::matmul(ptr, rgrad), {-1, -2})
									  : (right->grad + dl::transpose(dl::matmul(ptr, rgrad), {-1, -2}));
				if (right->gradfn)
					right->gradfn(right->grad);
			}
		};
	}
	return tensor;
}

TensorPtr dl::transpose(TensorPtr x, std::vector<int>&& perm) noexcept {
	auto p = perm | std::views::transform([&x](int d) { return (d < 0) ? (x->numDim() + d) : d; });
	/** \todo use ranges::to() when this is available **/
	std::vector<size_t> vec;
	for (auto tmp : p)
		vec.push_back(tmp);
	auto tensor = x->transpose(std::move(vec));
	if (tensor->requiresGrad()) {
		tensor->gradfn = [x = std::move(x), perm](TensorPtr& ptr) mutable {
			/** \todo calculate correct inversion of perm **/
			x->grad = (x->grad == nullptr) ? dl::transpose(ptr, std::move(perm))
										   : (x->grad + dl::transpose(ptr, std::move(perm)));
			if (x->gradfn)
				x->gradfn(x->grad);
		};
	}
	return tensor;
}

std::ostream& dl::operator<<(std::ostream& stream, const TensorPtr& tensor) noexcept {
	if (tensor == nullptr)
		return stream << "null";
	return tensor->writeToStream(stream);
}

bool dl::operator==(const dl::TensorPtr& left, const dl::TensorPtr& right) noexcept {
	if (left == nullptr || right == nullptr)
		return left == nullptr && right == nullptr;
	return *left == right;
}
bool dl::allclose(const TensorPtr& left, const TensorPtr& right, float rtol, float atol) noexcept {
	if ((left == nullptr) || (right == nullptr))
		return (left == nullptr) && (right == nullptr);
	return left->allclose(right, rtol, atol);
}

size_t dl::numEntries(const dl::TensorPtr& tensor) noexcept {
	const auto shape = tensor->shape();
	return std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<size_t>{});
}

dl::TensorPtr dl::reshape(dl::TensorPtr tensor, dl::SShape shape) noexcept {
	/** \todo add autodiff support **/
	tensor->reshape(shape);
	return tensor;
}