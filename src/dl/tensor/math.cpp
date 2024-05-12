#include <dl/tensor/math.hpp>

#include <dl/device.hpp>
#include <dl/tensor/tensorimpl.hpp>

#include <functional>
#include <numeric>
#include <ranges>

using namespace dl;

Tensor dl::pow(Tensor& base, float exponent) noexcept {
	auto result = dl::pow((const Tensor&)base, exponent);
	if (base->requiresGrad()) {
		result->gradfn = [base, exponent](Tensor& ptr) mutable {
			if (base->grad == nullptr)
				base->grad = std::move(ptr * exponent);
			else
				base->grad = std::move(base->grad + (ptr * exponent));
			if (base->gradfn)
				base->gradfn(base->grad);
			else
				assert(base->requiresGrad());
		};
	}
	return result;
}

Tensor dl::pow(Tensor&& base, float exponent) noexcept {
	auto result = dl::pow((const Tensor&)base, exponent);
	if (base->requiresGrad()) {
		result->gradfn = [base = std::move(base), exponent](Tensor& ptr) mutable {
			if (base->grad == nullptr)
				base->grad = std::move(ptr * exponent);
			else
				base->grad = std::move(base->grad + (ptr * exponent));
			if (base->gradfn)
				base->gradfn(base->grad);
			else
				assert(base->requiresGrad());
		};
	}
	return result;
}

Tensor dl::pow(const Tensor& base, float exponent) noexcept { return base->pow(exponent); }

Tensor dl::exp(Tensor&& base) noexcept {
	/** \todo add support for autodiff **/
	return base->exp();
}
Tensor dl::exp(const Tensor& base) noexcept { return base->exp(); }

Tensor dl::sqrt(const Tensor& x) noexcept { return x->sqrt(); }

Tensor dl::rsqrt(const Tensor& x) noexcept { return x->rsqrt(); }

Tensor dl::mean(Tensor& x) noexcept {
	/** \todo add support for autograd **/
	auto tensor = x->mean();
	if (tensor->requiresGrad()) {
		auto size = (float)x->shape(0);
		tensor->gradfn = [x, size](Tensor& ptr) mutable {
			if (x->grad == nullptr)
				x->grad = std::move(ptr * dl::ones_like(x) / size);
			else
				x->grad = std::move(x->grad + (ptr * dl::ones_like(x) / size));
			if (x->gradfn)
				x->gradfn(x->grad);
			else
				assert(x->requiresGrad());
		};
	}
	return tensor;
}

Tensor dl::mean(Tensor&& x) noexcept {
	auto tensor = x->mean();
	if (tensor->requiresGrad()) {
		auto size = (float)x->shape(0);
		tensor->gradfn = [copy = std::move(x), size](Tensor& ptr) mutable {
			if (copy->grad == nullptr)
				copy->grad = std::move(ptr * dl::ones_like(copy) / size);
			else
				copy->grad = std::move(copy->grad + (ptr * dl::ones_like(copy) / size));
			if (copy->gradfn)
				copy->gradfn(copy->grad);
			else
				assert(copy->requiresGrad());
		};
	}
	return tensor;
}

Tensor dl::sum(const Tensor& x) noexcept { return x->sum(); }

Tensor dl::min(const Tensor& x) noexcept { return x->min(); }

Tensor dl::max(const Tensor& x) noexcept { return x->max(); }

Tensor dl::max(const Tensor& x, const Tensor& y) noexcept { return x->max(y); }

Tensor dl::var(const Tensor& x) noexcept { return x->var(); }

Tensor dl::relu(Tensor&& x) noexcept { return dl::max(x, 0); }

Tensor dl::softmax(Tensor&& x) noexcept {
	/** \todo implement gradient **/
	return dl::softmax(x);
}
Tensor dl::softmax(const Tensor& x) noexcept {
	const auto power = dl::exp(dl::max(x));
	return power / dl::sum(power);
}

Tensor dl::operator+(Tensor&& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	if (left->requiresGrad())
		left->gradfn = [left](Tensor& ptr) { dl::constant(1.0, left->device()); };
	if (right->requiresGrad())
		right->gradfn = [right](Tensor& ptr) { dl::constant(1.0, right->device()); };
	return left->add(right);
}
Tensor dl::operator+(Tensor&& left, Tensor&& right) noexcept {
	/** \todo add support for autodiff **/
	return left->add(right);
}
Tensor dl::operator-(Tensor& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	left->gradfn = [left](Tensor& ptr) { dl::constant(1.0, left->device()); };
	right->gradfn = [right](Tensor& ptr) { dl::constant(-1.0, right->device()); };
	return left->sub(right);
}
Tensor dl::operator-(Tensor& left, Tensor&& right) noexcept {
	/** \todo add support for autodiff **/
	return left->sub(right);
}
Tensor dl::operator-(Tensor&& left, Tensor&& right) noexcept {
	/** \todo add support for autodiff **/
	return left->sub(right);
}
Tensor dl::operator*(Tensor& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	auto tensor = left->mul(right);
	if (tensor->requiresGrad()) {
		tensor->gradfn = [&left, &right](Tensor& ptr) {
			auto lgrad = right;
			lgrad->discardGradient();
			auto rgrad = left;
			rgrad->discardGradient();

			left->grad = lgrad * ptr;
			right->grad = rgrad * ptr;
		};
	}
	return tensor;
}
Tensor dl::operator*(Tensor&& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	return left->mul(right);
}
Tensor dl::operator*(Tensor&& left, Tensor&& right) noexcept {
	/** \todo add support for autograd **/
	return left->mul(right);
}
Tensor dl::operator/(Tensor& left, Tensor right) noexcept {
	/** \todo add support for autograd **/
	return left->div(right);
}
Tensor dl::operator/(Tensor& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	return left->div(right);
}

Tensor dl::fma(const Tensor& factor1, const Tensor& factor2, const Tensor& summand) noexcept {
	return factor1->fma(factor2, summand);
}

Tensor dl::matmul(Tensor& left, Tensor& right) noexcept {
	std::cout << "matmul" << std::endl;
	std::cout << '(' << left->shape(0) << ',' << left->shape(1) << ')' << std::endl;
	std::cout << '(' << right->shape(0) << ',' << right->shape(1) << ')' << std::endl;
	/** \todo add support for autograd **/
	auto tmp = left->matmul(right);
	std::cout << '(' << tmp->shape(0) << ',' << tmp->shape(1) << ')' << std::endl;
	return tmp;
}
Tensor dl::matmul(Tensor&& left, Tensor& right) noexcept {
	/** \todo add support for autograd **/
	return left->matmul(right);
}
Tensor dl::matmul(Tensor&& left, Tensor&& right) noexcept {
	/** \todo add support for autograd **/
	return left->matmul(right);
}
Tensor dl::matmul(const Tensor& left, const Tensor& right) noexcept { return left->matmul(right); }

Tensor dl::transpose(Tensor&& x, std::vector<int>&& perm) noexcept {
	auto p = perm | std::views::transform([&x](int d) { return (d < 0) ? (x->numDim() + d) : d; });
	/** \todo use ranges::to() when this is available **/
	std::vector<size_t> vec;
	for (auto tmp : p)
		vec.push_back(tmp);
	return x->transpose(std::move(vec));
}

std::ostream& dl::operator<<(std::ostream& stream, const Tensor& tensor) noexcept {
	if (tensor == nullptr)
		return stream << "null";
	return tensor->writeToStream(stream);
}

bool dl::operator==(const dl::Tensor& left, const dl::Tensor& right) noexcept {
	if (left->shape() != right->shape())
		return false;
	throw std::runtime_error("Not Implemented");
	/*auto flatleft = left->flatten();
	auto flatright = right->flatten();
	for (auto it1 = std::begin(flatleft), it2 = std::begin(flatright);
		 it1 != std::end(flatleft) && it2 != std::end(flatright); ++it1, ++it2)
		if ((double)*it1 != (double)*it2)
			return false;
	return true;*/
}

size_t dl::numEntries(const dl::Tensor& tensor) noexcept {
	const auto shape = tensor->shape();
	return std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<size_t>{});
}